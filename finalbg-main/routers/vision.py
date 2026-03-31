import json
import re
import base64
import cv2
import numpy as np
import requests
import uuid

from collections import Counter
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

import prompts

DEFAULT_VISION_ANALYZE_PROMPT = """
Analyze the garment in the image and return strict JSON only with keys:
name, category, sub_category, occasions, pattern.
- category must be one of: Tops, Bottoms, Outerwear, Footwear, Dresses, Accessories, Bags, Jewelry, Makeup, Skincare.
- occasions must be an array of short lowercase strings.
- pattern should be a short lowercase string (e.g. plain, striped, checked, floral).
Do not include markdown fences or extra text.
"""
from services.embedding_service import encode_metadata
from services.image_validation import validate_image_bytes
from services.qdrant_service import qdrant_service
from middleware.auth_middleware import ensure_user_scope, get_current_user
from services.rate_limiter import limiter
from services.task_claim_check import store_image_claim_check
try:
    from worker import vision_analyze_task
except Exception:
    vision_analyze_task = None

router = APIRouter()


class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    userId: str = "demo_user"


def _hex_to_color_name(hex_color: str) -> str:
    try:
        color = hex_color.lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except Exception:
        return "Multicolor"

    if max(r, g, b) < 40:
        return "Black"
    if min(r, g, b) > 220:
        return "White"
    if abs(r - g) < 14 and abs(g - b) < 14:
        return "Gray"
    if r > 180 and g < 110 and b < 110:
        return "Red"
    if r > 170 and g > 120 and b < 90:
        return "Orange"
    if r > 170 and g > 170 and b < 90:
        return "Yellow"
    if g > 150 and r < 130 and b < 130:
        return "Green"
    if b > 150 and r < 130 and g < 150:
        return "Blue"
    if r > 150 and b > 150 and g < 130:
        return "Purple"
    if r > 140 and g > 100 and b > 70:
        return "Brown"
    return "Multicolor"


def _infer_pattern(cv_image) -> str:
    try:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        return "checked" if edge_density > 0.09 else "plain"
    except Exception:
        return "plain"


def _extract_foreground_mask(decoded_img) -> np.ndarray | None:
    try:
        if decoded_img is None:
            return None

        if decoded_img.ndim == 3 and decoded_img.shape[2] == 4:
            # BGRA image from background-removal step.
            alpha = decoded_img[:, :, 3]
            mask = alpha > 18
        else:
            bgr = decoded_img if decoded_img.ndim == 3 else None
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            maxc = rgb.max(axis=2)
            minc = rgb.min(axis=2)
            near_white = (
                (rgb[:, :, 0] >= 236)
                & (rgb[:, :, 1] >= 236)
                & (rgb[:, :, 2] >= 236)
                & ((maxc - minc) <= 20)
            )
            saturated = hsv[:, :, 1] > 18
            mask = (~near_white) | saturated

        # Remove tiny noise.
        mask_u8 = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = mask_u8 > 0

        if final_mask.sum() < 200:
            return None
        return final_mask
    except Exception:
        return None


def _infer_garment_hint(decoded_img) -> tuple[str, str]:
    """
    Returns a coarse garment hint from silhouette:
    - Tops/Shirt
    - Bottoms/Trousers
    """
    mask = _extract_foreground_mask(decoded_img)
    if mask is None:
        return ("Tops", "Shirt")

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return ("Tops", "Shirt")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w = max(1, x_max - x_min + 1)
    box_h = max(1, y_max - y_min + 1)
    aspect = float(box_h) / float(box_w)

    crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    row_fill = crop.mean(axis=1)
    seg = max(2, int(0.2 * len(row_fill)))
    top_fill = float(row_fill[:seg].mean())
    mid_fill = float(row_fill[len(row_fill) // 2 - seg // 2: len(row_fill) // 2 + seg // 2].mean())
    bottom_fill = float(row_fill[-seg:].mean())

    h = crop.shape[0]
    upper_y = min(h - 1, max(0, int(h * 0.15)))
    mid_y = min(h - 1, max(0, int(h * 0.50)))
    low_y = min(h - 1, max(0, int(h * 0.85)))
    upper_width = int(np.count_nonzero(crop[upper_y]))
    mid_width = int(np.count_nonzero(crop[mid_y]))
    low_width = int(np.count_nonzero(crop[low_y]))

    lower = crop[int(h * 0.55):, :]
    cc_count = 0
    if lower.size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((lower.astype(np.uint8)) * 255)
        large_components = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= max(120, int(lower.size * 0.01)):
                large_components.append(area)
        cc_count = len(large_components)

    likely_bottom = (
        aspect > 1.15
        and (
            cc_count >= 2
            or bottom_fill < (mid_fill * 0.86)
            or (low_width > 0 and mid_width > 0 and (low_width / float(mid_width)) < 0.78)
            or (upper_width > 0 and low_width > 0 and (low_width / float(upper_width)) < 0.74)
        )
    )
    if likely_bottom:
        return ("Bottoms", "Trousers")

    return ("Tops", "Shirt")


def _build_smart_fallback(color_hex: str, hint_category: str = "Tops", hint_sub_category: str = "Shirt"):
    color_name = _hex_to_color_name(color_hex)
    category = hint_category
    sub_category = hint_sub_category
    occasions = ["casual", "office"]
    name = f"{color_name} {sub_category}"
    return {
        "name": name,
        "category": category,
        "sub_category": sub_category,
        "occasions": occasions,
        "pattern": "plain",
    }


# -------------------------
# COLOR EXTRACTION
# -------------------------
def get_dominant_color(cv_image, k=3):
    try:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        crop_h, crop_w = int(h * 0.25), int(w * 0.25)
        center_image = image[crop_h:h - crop_h, crop_w:w - crop_w]

        center_image = cv2.resize(center_image, (100, 100), interpolation=cv2.INTER_AREA)

        hsv_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV)
        pixels_rgb = center_image.reshape((-1, 3))
        pixels_hsv = hsv_image.reshape((-1, 3))

        mask = (pixels_hsv[:, 1] > 20) & (pixels_hsv[:, 2] > 70) & (pixels_hsv[:, 2] < 245)
        filtered_rgb = pixels_rgb[mask]

        if len(filtered_rgb) < 100:
            mask = (pixels_hsv[:, 2] > 30) & (pixels_hsv[:, 2] < 250)
            filtered_rgb = pixels_rgb[mask]
            if len(filtered_rgb) == 0:
                filtered_rgb = pixels_rgb

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(filtered_rgb)

        counts = Counter(kmeans.labels_)
        dominant_cluster_index = counts.most_common(1)[0][0]
        dominant_rgb = [int(x) for x in kmeans.cluster_centers_[dominant_cluster_index]]

        return "#{:02x}{:02x}{:02x}".format(*dominant_rgb).upper()

    except Exception as e:
        print(f"Color Math Error: {e}")
        return "#000000"


# -------------------------
# MAIN ENDPOINT
# -------------------------
@router.post("/analyze-image")
@limiter.limit("10/minute")
def analyze_image(request: Request, payload: ImageAnalyzeRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.userId)
        payload.userId = user.get("user_id", payload.userId)

    # -------------------------
    # STEP 1: Decode + Color
    # -------------------------
    base64_data = payload.image_base64
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    try:
        img_data = base64.b64decode(base64_data, validate=True)
        if len(img_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="image payload too large (max 5MB)")
        validate_image_bytes(img_data, allowed_formats=("PNG", "JPEG"), field_name="image_base64")

        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        if decoded.ndim == 3 and decoded.shape[2] == 4:
            cv_image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
        elif decoded.ndim == 3 and decoded.shape[2] == 3:
            cv_image = decoded
        else:
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        extracted_color_hex = get_dominant_color(cv_image)
        hint_category, hint_sub_category = _infer_garment_hint(decoded)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {str(e)}")

    # -------------------------
    # STEP 2: LLM ANALYSIS
    # -------------------------
    llm_payload = {
        "model": "llama3.2-vision",
        "prompt": getattr(prompts, "VISION_ANALYZE_PROMPT", DEFAULT_VISION_ANALYZE_PROMPT),
        "images": [payload.image_base64],
        "stream": False,
        "format": "json",
    }

    llm_fallback = False
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=llm_payload,
            timeout=35,
        )
        response.raise_for_status()

        raw_response = response.json().get("response", "{}")
        clean_response = re.sub(r"```json|```", "", raw_response).strip()
        final_data = json.loads(clean_response)

    except Exception as e:
        print(f"Image Analyze Error: {str(e)}")
        llm_fallback = True
        final_data = _build_smart_fallback(
            extracted_color_hex,
            hint_category=hint_category,
            hint_sub_category=hint_sub_category,
        )

    # Normalize weak LLM answers to practical defaults.
    name = str(final_data.get("name", "")).strip()
    if not name or name.lower() in {"new garment", "garment", "new item", "item"}:
        final_data["name"] = _build_smart_fallback(
            extracted_color_hex,
            hint_category=hint_category,
            hint_sub_category=hint_sub_category,
        )["name"]

    category = str(final_data.get("category", "")).strip()
    if not category:
        final_data["category"] = hint_category
    elif category.lower() == "tops" and hint_category == "Bottoms" and llm_fallback:
        # If LLM failed and heuristic strongly suggests bottoms, trust heuristic.
        final_data["category"] = hint_category

    sub_category = str(final_data.get("sub_category", "")).strip()
    if not sub_category or sub_category.lower() in {"unknown", "other", "na", "n/a"}:
        final_data["sub_category"] = hint_sub_category
    elif sub_category.lower() == "shirt" and hint_sub_category == "Trousers" and llm_fallback:
        final_data["sub_category"] = hint_sub_category

    occasions = final_data.get("occasions", [])
    if not isinstance(occasions, list) or not occasions:
        final_data["occasions"] = ["casual", "office"]

    if not final_data.get("pattern"):
        final_data["pattern"] = _infer_pattern(cv_image)

    # -------------------------
    # STEP 3: FORCE TRUE COLOR
    # -------------------------
    final_data["color_code"] = extracted_color_hex
    final_data["userId"] = payload.userId

    # -------------------------
    # STEP 4: EMBEDDING + STORAGE
    # -------------------------
    qdrant_saved = False
    try:
        vector = encode_metadata(final_data)
        item_id = str(uuid.uuid4())
        qdrant_service.upsert_item(item_id=item_id, vector=vector, payload=final_data)
        qdrant_saved = True
        print("Stored in Qdrant:", item_id)
    except Exception as e:
        print("Qdrant store error:", str(e))

    # -------------------------
    # RESPONSE
    # -------------------------
    return {
        "success": True,
        "data": final_data,
        "meta": {
            "qdrant_saved": qdrant_saved,
            "llm_fallback": llm_fallback,
        },
    }


@router.post("/analyze-image/async", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("10/minute")
def analyze_image_async(request: Request, payload: ImageAnalyzeRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.userId)
        payload.userId = user.get("user_id", payload.userId)
    if vision_analyze_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        image_url = store_image_claim_check(
            payload.image_base64,
            user_id=payload.userId,
            task_type="vision_analyze",
        )
        task = vision_analyze_task.delay(image_url, payload.userId)
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "task_type": "vision_analyze_task",
            "image_claim_check_url": image_url,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue vision analysis: {exc}")

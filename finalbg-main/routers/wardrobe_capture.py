import base64
import io
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

from middleware.auth_middleware import ensure_user_scope, get_current_user
from services.appwrite_proxy import AppwriteProxy
from services.embedding_service import encode_metadata
from services.image_validation import validate_image_bytes
from services.qdrant_service import qdrant_service
from services.rate_limiter import limiter
from services.r2_storage import R2Storage, R2StorageError
from services.task_claim_check import store_image_claim_check
try:
    from worker import capture_analyze_task, capture_save_selected_task, process_upload_task
except Exception:
    capture_analyze_task = None
    capture_save_selected_task = None
    process_upload_task = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])

_detector = None
_classifier = None


def clear_model_cache() -> None:
    global _detector, _classifier
    _detector = None
    _classifier = None

DETECTION_LABELS = [
    "shirt",
    "t-shirt",
    "blouse",
    "top",
    "jacket",
    "blazer",
    "dress",
    "skirt",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "shoes",
    "sneakers",
    "heels",
    "boots",
    "sandals",
    "watch",
    "bag",
    "handbag",
    "jewelry",
    "necklace",
    "earrings",
    "accessory",
]

CLASSIFICATION_LABELS = [
    "formal shirt",
    "casual shirt",
    "t-shirt",
    "blouse",
    "crop top",
    "jacket",
    "blazer",
    "dress",
    "skirt",
    "jeans",
    "trousers",
    "shorts",
    "sneakers",
    "formal shoes",
    "heels",
    "boots",
    "sandals",
    "watch",
    "handbag",
    "jewelry",
]


class CaptureAnalyzeRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


class DetectedItem(BaseModel):
    item_id: str
    name: str
    category: str
    sub_category: str
    color_code: str
    pattern: str = "plain"
    occasions: List[str] = ["casual"]
    confidence: float = 0.0
    reasoning: str
    bbox: Dict[str, int]
    raw_crop_base64: str
    segmented_png_base64: str


class SaveSelectedRequest(BaseModel):
    user_id: str
    selected_item_ids: List[str]
    detected_items: List[DetectedItem]


class ProcessUploadRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


def _load_detector():
    global _detector
    if _detector is not None:
        return _detector
    if pipeline is None:
        return None
    try:
        _detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")
    except Exception:
        _detector = None
    return _detector


def _load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    if pipeline is None:
        return None
    try:
        _classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
    except Exception:
        _classifier = None
    return _classifier


def _decode_image_base64(value: str) -> Image.Image:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 5MB)")
    validate_image_bytes(data, allowed_formats=("PNG", "JPEG"), field_name="image_base64")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}")
    return image


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _dominant_hex(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = arr.reshape((-1, 3))
    if len(pixels) == 0:
        return "#000000"
    mean = np.mean(pixels, axis=0).astype(int)
    rgb = (int(mean[2]), int(mean[1]), int(mean[0]))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _segment_png_base64(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        return base64.b64encode(cv2.imencode(".png", rgba)[1].tobytes()).decode("utf-8")

    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (2, 2, max(1, w - 4), max(1, h - 4))
    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    except Exception:
        alpha = np.full((h, w), 255, dtype=np.uint8)

    rgba = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode segmented PNG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _label_to_category(label: str) -> Tuple[str, str]:
    l = (label or "").lower()
    if any(k in l for k in ["shirt", "blouse", "top", "jacket", "blazer"]):
        return "top", label
    if any(k in l for k in ["pants", "trousers", "jeans", "shorts", "skirt"]):
        return "bottom", label
    if any(k in l for k in ["dress"]):
        return "dress", label
    if any(k in l for k in ["shoe", "sneaker", "heel", "boot", "sandal"]):
        return "shoes", label
    if any(k in l for k in ["watch", "bag", "jewelry", "necklace", "earrings", "accessory"]):
        return "accessory", label
    return "top", label


def _nms_boxes(items: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    kept: List[Dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for item in sorted_items:
        box = item.get("bbox_tuple")
        if box is None:
            continue
        if any(iou(box, k.get("bbox_tuple")) > iou_threshold for k in kept):
            continue
        kept.append(item)
    return kept


def _detect_multi_items(image: Image.Image) -> List[Dict[str, Any]]:
    detector = _load_detector()
    width, height = image.size

    if detector is None:
        # Fallback: one full-image item when detector unavailable.
        return [
            {
                "label": "apparel",
                "score": 0.35,
                "bbox_tuple": (0, 0, width, height),
            }
        ]

    try:
        raw = detector(image, candidate_labels=DETECTION_LABELS, threshold=0.10)
    except Exception:
        raw = []

    candidates = []
    for row in raw or []:
        box = row.get("box", {}) if isinstance(row, dict) else {}
        xmin = max(0, int(box.get("xmin", 0)))
        ymin = max(0, int(box.get("ymin", 0)))
        xmax = min(width, int(box.get("xmax", width)))
        ymax = min(height, int(box.get("ymax", height)))
        if xmax - xmin < 24 or ymax - ymin < 24:
            continue
        candidates.append(
            {
                "label": str(row.get("label", "apparel")),
                "score": float(row.get("score", 0.0)),
                "bbox_tuple": (xmin, ymin, xmax, ymax),
            }
        )

    deduped = _nms_boxes(candidates, iou_threshold=0.45)
    if not deduped:
        deduped = [
            {
                "label": "apparel",
                "score": 0.35,
                "bbox_tuple": (0, 0, width, height),
            }
        ]
    return deduped[:12]


def _classify_crop(crop: Image.Image, detected_label: str) -> Tuple[str, str, float]:
    clf = _load_classifier()
    if clf is None:
        category, sub = _label_to_category(detected_label)
        return category, sub.lower(), 0.4

    try:
        ranked = clf(crop, candidate_labels=CLASSIFICATION_LABELS, hypothesis_template="a photo of {}")
    except Exception:
        ranked = []

    if not ranked:
        category, sub = _label_to_category(detected_label)
        return category, sub.lower(), 0.4

    best = ranked[0]
    label = str(best.get("label", detected_label)).lower()
    score = float(best.get("score", 0.4))
    category, sub = _label_to_category(label)
    return category, sub.lower(), score


def _draw_overlay(image: Image.Image, items: List[Dict[str, Any]]) -> str:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(items):
        bbox = item.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=3)
        label = f"{idx + 1}. {item.get('name', item.get('sub_category', 'item'))}"
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill="cyan")
    return _image_to_base64(canvas, fmt="JPEG")


@router.post("/analyze")
@limiter.limit("8/minute")
def analyze_capture(request: Request, payload: CaptureAnalyzeRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.user_id)
    image = _decode_image_base64(payload.image_base64)
    detections = _detect_multi_items(image)

    items: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_tuple"]
        crop = image.crop((x1, y1, x2, y2))
        category, sub_category, cls_score = _classify_crop(crop, det.get("label", "apparel"))
        color_code = _dominant_hex(crop)
        segmented_png_base64 = _segment_png_base64(crop)
        raw_crop_base64 = _image_to_base64(crop, fmt="JPEG")

        score = max(float(det.get("score", 0.0)), float(cls_score))
        item_id = str(uuid.uuid4())
        reasoning = (
            f"Detected '{sub_category}' with confidence {round(score, 3)} "
            f"using object region + visual classification."
        )

        items.append(
            {
                "item_id": item_id,
                "name": sub_category.replace("_", " ").title(),
                "category": category,
                "sub_category": sub_category,
                "color_code": color_code,
                "pattern": "plain",
                "occasions": ["casual"],
                "confidence": round(score, 4),
                "reasoning": reasoning,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "raw_crop_base64": raw_crop_base64,
                "segmented_png_base64": segmented_png_base64,
            }
        )

    overlay_base64 = _draw_overlay(image, items)

    return {
        "success": True,
        "board": "wardrobe_capture",
        "type": "multi_detect",
        "message": f"Detected {len(items)} items. Select and save the ones you want.",
        "overlay_preview_base64": overlay_base64,
        "items": items,
    }


@router.post("/analyze/async", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("8/minute")
def analyze_capture_async(request: Request, payload: CaptureAnalyzeRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.user_id)
    if capture_analyze_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        image_url = store_image_claim_check(
            payload.image_base64,
            user_id=payload.user_id,
            task_type="capture_analyze",
        )
        task = capture_analyze_task.delay(payload.user_id, image_url)
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "task_type": "capture_analyze_task",
            "image_claim_check_url": image_url,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue capture analyze: {exc}")


def _decode_simple_base64(value: str) -> bytes:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid item base64: {exc}")


@router.post("/save-selected")
@limiter.limit("20/minute")
def save_selected(request: Request, payload: SaveSelectedRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.user_id)
    selected = set(payload.selected_item_ids or [])
    if not selected:
        raise HTTPException(status_code=400, detail="selected_item_ids cannot be empty")

    storage = R2Storage()
    proxy = AppwriteProxy()
    saved: List[Dict[str, Any]] = []

    for item in payload.detected_items:
        if item.item_id not in selected:
            continue

        raw_bytes = _decode_simple_base64(item.raw_crop_base64)
        seg_bytes = _decode_simple_base64(item.segmented_png_base64)
        file_id = str(uuid.uuid4())

        try:
            upload = storage.upload_wardrobe_images(
                file_id=file_id,
                raw_image_bytes=raw_bytes,
                masked_image_bytes=seg_bytes,
            )
        except R2StorageError as exc:
            raise HTTPException(status_code=500, detail=f"R2 upload failed: {exc}")

        metadata = {
            "userId": payload.user_id,
            "name": item.name,
            "category": item.category,
            "sub_category": item.sub_category,
            "color_code": item.color_code,
            "pattern": item.pattern,
            "occasions": item.occasions,
            "status": "active",
            "image_url": upload["raw_image_url"],
            "masked_url": upload["masked_image_url"],
            "image_id": upload["raw_file_name"],
            "masked_id": upload["masked_file_name"],
            "worn": 0,
            "liked": False,
            "notes": item.reasoning,
        }

        try:
            doc = proxy.create_document("outfits", metadata)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Appwrite save failed: {exc}")

        vector = encode_metadata(
            {
                "category": item.category,
                "sub_category": item.sub_category,
                "color_code": item.color_code,
                "pattern": item.pattern,
                "occasions": item.occasions,
            }
        )
        qdrant_service.upsert_item(
            item_id=file_id,
            vector=vector,
            payload={
                "userId": payload.user_id,
                "category": item.category,
                "sub_category": item.sub_category,
                "color_code": item.color_code,
                "image_url": upload["masked_image_url"],
            },
        )

        saved.append(
            {
                "item_id": item.item_id,
                "outfit_doc_id": doc.get("$id"),
                "image_url": upload["masked_image_url"],
                "raw_image_url": upload["raw_image_url"],
                "name": item.name,
                "category": item.category,
                "sub_category": item.sub_category,
            }
        )

    return {
        "success": True,
        "message": f"Saved {len(saved)} selected items to wardrobe.",
        "saved_items": saved,
    }


@router.post("/save-selected/async", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("20/minute")
def save_selected_async(request: Request, payload: SaveSelectedRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.user_id)
    if capture_save_selected_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        detected = []
        for item in payload.detected_items:
            if hasattr(item, "model_dump"):
                detected.append(item.model_dump())
            else:
                detected.append(item.dict())
        payload = {
            "user_id": payload.user_id,
            "selected_item_ids": payload.selected_item_ids,
            "detected_items": detected,
        }
        task = capture_save_selected_task.delay(payload)
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "task_type": "capture_save_selected_task",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue save-selected: {exc}")


@router.post("/process-upload/async", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("6/minute")
def process_upload_async(request: Request, payload: ProcessUploadRequest, user=Depends(get_current_user)):
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, payload.user_id)
    if process_upload_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        image_url = store_image_claim_check(
            payload.image_base64,
            user_id=payload.user_id,
            task_type="process_upload",
        )
        task = process_upload_task.delay(payload.user_id, image_url)
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "task_type": "process_upload",
            "image_claim_check_url": image_url,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue process-upload: {exc}")

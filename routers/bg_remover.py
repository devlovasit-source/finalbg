import os
import io
import base64
import torch
import threading
import numpy as np
from collections import deque
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from fastapi import status

try:
    from worker import bg_remove_task
except Exception:
    bg_remove_task = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

print("BG_REMOVER LOADED")

router = APIRouter()


class BGRemoveRequest(BaseModel):
    image_base64: str

    @validator("image_base64")
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid image data")
        return v


model = None
model_last_error = None
model_lock = threading.Lock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_session = None
onnx_lock = threading.Lock()
onnx_last_error = None
onnx_runtime_disabled = False
USE_TORCH_MODEL = os.getenv("BG_USE_TORCH_MODEL", "0") == "1"
BG_DISABLE_ONNX = os.getenv("BG_DISABLE_ONNX", "1") == "1"

print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "RMBG_2_0")
ONNX_DIR = os.path.join(MODEL_PATH, "onnx")
ONNX_MODEL_CANDIDATES = [
    "model_q4f16.onnx",
    "model_q4.onnx",
    "model_int8.onnx",
    "model_uint8.onnx",
    "model_quantized.onnx",
    "model_fp16.onnx",
    "model.onnx",
]
ONNX_INPUT_SIZES = [1024]

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    global model, model_last_error

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        print("Loading model from:", MODEL_PATH)

        if not os.path.exists(MODEL_PATH):
            model_last_error = f"Model folder not found: {MODEL_PATH}"
            print("Model load failed:", model_last_error)
            model = None
            return model

        attempts = [
            {"use_safetensors": True},
            {"use_safetensors": False},
        ]
        errors = []

        for attempt in attempts:
            try:
                model_local = AutoModelForImageSegmentation.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    dtype=torch.float32,
                    **attempt,
                )

                if any(p.is_meta for p in model_local.parameters()):
                    raise RuntimeError("Model contains meta tensors after load")

                model_local = model_local.to(device=device, dtype=torch.float32)
                model_local.eval()
                model = model_local
                model_last_error = None
                print("Model ready.")
                return model
            except Exception as exc:
                errors.append(f"{attempt}: {exc}")

        model_last_error = " | ".join(errors)
        print("Model load failed:", model_last_error)
        model = None

    return model


def load_onnx_session():
    global onnx_session, onnx_last_error, onnx_runtime_disabled
    if BG_DISABLE_ONNX:
        onnx_last_error = "ONNX disabled by BG_DISABLE_ONNX=1"
        return None
    if onnx_session is not None:
        return onnx_session
    if onnx_runtime_disabled:
        onnx_last_error = "onnx runtime disabled after repeated OOM/runtime failures"
        return None
    if ort is None:
        onnx_last_error = "onnxruntime is not installed"
        return None

    with onnx_lock:
        if onnx_session is not None:
            return onnx_session
        if not os.path.exists(ONNX_DIR):
            onnx_last_error = f"ONNX directory not found: {ONNX_DIR}"
            return None

        session_options = ort.SessionOptions()
        # Keep graph rewrites off to avoid fusion-related init failures on some builds.
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = False
        session_options.log_severity_level = 3

        errors = []
        for name in ONNX_MODEL_CANDIDATES:
            candidate_path = os.path.join(ONNX_DIR, name)
            if not os.path.exists(candidate_path):
                continue
            try:
                onnx_session = ort.InferenceSession(
                    candidate_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                onnx_last_error = None
                print(f"ONNX fallback ready: {candidate_path}")
                return onnx_session
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if not errors:
            onnx_last_error = (
                "No supported ONNX model file found in "
                f"{ONNX_DIR}. Tried: {', '.join(ONNX_MODEL_CANDIDATES)}"
            )
        else:
            onnx_last_error = " | ".join(errors)
        print("ONNX load failed:", onnx_last_error)
        return None


def _original_png_response(image_data: bytes, reason: str):
    try:
        original = Image.open(io.BytesIO(image_data)).convert("RGBA")
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "success": True,
            "image_base64": result_base64,
            "bg_removed": False,
            "fallback_reason": reason,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Processing failed")


def _white_bg_cutout_mask(orig_image: Image.Image):
    arr = np.asarray(orig_image.convert("RGB"))
    h, w, _ = arr.shape

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    white = (
        (arr[:, :, 0] >= 232)
        & (arr[:, :, 1] >= 232)
        & (arr[:, :, 2] >= 232)
        & ((maxc - minc) <= 24)
    )

    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    for x in range(w):
        if white[0, x]:
            visited[0, x] = 1
            q.append((0, x))
        if white[h - 1, x]:
            visited[h - 1, x] = 1
            q.append((h - 1, x))
    for y in range(h):
        if white[y, 0]:
            visited[y, 0] = 1
            q.append((y, 0))
        if white[y, w - 1]:
            visited[y, w - 1] = 1
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if y + 1 < h and not visited[y + 1, x] and white[y + 1, x]:
            visited[y + 1, x] = 1
            q.append((y + 1, x))
        if y - 1 >= 0 and not visited[y - 1, x] and white[y - 1, x]:
            visited[y - 1, x] = 1
            q.append((y - 1, x))
        if x + 1 < w and not visited[y, x + 1] and white[y, x + 1]:
            visited[y, x + 1] = 1
            q.append((y, x + 1))
        if x - 1 >= 0 and not visited[y, x - 1] and white[y, x - 1]:
            visited[y, x - 1] = 1
            q.append((y, x - 1))

    return 255 - (visited.astype(np.uint8) * 255)


def _best_effort_png_response(image_data: bytes, reason: str):
    try:
        orig = Image.open(io.BytesIO(image_data)).convert("RGB")
        fg_mask = _white_bg_cutout_mask(orig)

        fg_pixels = int((fg_mask > 0).sum())
        total = fg_mask.size
        if fg_pixels < int(total * 0.02) or fg_pixels > int(total * 0.98):
            return _original_png_response(image_data, reason)

        alpha = Image.fromarray(fg_mask)
        out = orig.copy()
        out.putalpha(alpha)

        buffer = io.BytesIO()
        out.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "success": True,
            "image_base64": result_base64,
            "bg_removed": True,
            "fallback_reason": f"Used white-background heuristic cutout: {reason}",
        }
    except Exception:
        return _original_png_response(image_data, reason)


def _is_likely_white_background(orig_image: Image.Image):
    arr = np.asarray(orig_image.convert("RGB"))
    h, w, _ = arr.shape
    if h < 8 or w < 8:
        return False

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    white = (
        (arr[:, :, 0] >= 232)
        & (arr[:, :, 1] >= 232)
        & (arr[:, :, 2] >= 232)
        & ((maxc - minc) <= 24)
    )

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[h - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, w - 1] = True
    border_white_ratio = float((white & border_mask).sum()) / float(border_mask.sum())
    return border_white_ratio >= 0.55


@router.post("/remove-bg")
async def remove_background(request: BGRemoveRequest):
    global onnx_runtime_disabled
    model_instance = load_model() if USE_TORCH_MODEL else None
    onnx_instance = None

    if model_instance is None:
        onnx_instance = load_onnx_session()

    try:
        try:
            base64_data = request.image_base64.split(",")[-1]
            image_data = base64.b64decode(base64_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large")

        if model_instance is None and onnx_instance is None:
            detail = "Model unavailable"
            if model_last_error:
                detail = f"Model unavailable: {model_last_error}"
            if onnx_last_error:
                detail = f"{detail} | ONNX fallback failed: {onnx_last_error}"
            print("BG fallback to original:", detail)
            return _best_effort_png_response(image_data, detail)

        orig_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        w, h = orig_image.size

        # Fast path for catalog/demo style images: avoid heavy ONNX when white BG is obvious.
        if model_instance is None and _is_likely_white_background(orig_image):
            return _best_effort_png_response(image_data, "fast white-background heuristic")

        if model_instance is not None:
            input_tensor = transform_image(orig_image).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model_instance(input_tensor)[-1].sigmoid().cpu()
            mask = preds[0].squeeze().numpy()
        else:
            input_name = onnx_instance.get_inputs()[0].name
            input_shape = onnx_instance.get_inputs()[0].shape

            # RMBG ONNX decoders are often exported with a hard 1024 spatial assumption.
            # Running at smaller resolutions triggers Split_64 shape errors.
            fixed_side = 1024
            if len(input_shape) >= 4:
                maybe_h = input_shape[-2]
                maybe_w = input_shape[-1]
                if (
                    isinstance(maybe_h, int)
                    and isinstance(maybe_w, int)
                    and maybe_h == maybe_w
                    and maybe_h > 0
                ):
                    fixed_side = maybe_h
            candidate_sizes = [fixed_side]
            onnx_errors = []
            mask = None

            for side in candidate_sizes:
                try:
                    img_onnx = orig_image.resize((side, side), Image.BILINEAR)
                    arr = np.asarray(img_onnx).astype(np.float32) / 255.0
                    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                        [0.229, 0.224, 0.225], dtype=np.float32
                    )
                    arr = np.transpose(arr, (2, 0, 1))[None, ...]
                    pred = onnx_instance.run(None, {input_name: arr})[0]
                    if pred.ndim == 4:
                        pred = pred[0, 0]
                    elif pred.ndim == 3:
                        pred = pred[0]
                    mask = 1.0 / (1.0 + np.exp(-pred))
                    break
                except Exception as exc:
                    onnx_errors.append(f"{side}: {exc}")

            if mask is None:
                reason = "ONNX inference failed: " + " | ".join(onnx_errors)
                # Demo stability: disable ONNX for the current process after any runtime failure.
                # Repeated retries are expensive on low-memory CPU boxes.
                onnx_runtime_disabled = True
                if any(
                    ("bad allocation" in err.lower())
                    or ("allocate" in err.lower())
                    or ("out of memory" in err.lower())
                    for err in onnx_errors
                ):
                    reason += " | ONNX disabled for this process due to memory failure"
                else:
                    reason += " | ONNX disabled for this process due to runtime failure"
                print("BG fallback to original:", reason)
                return _best_effort_png_response(image_data, reason)

        mask = (mask > 0.5).astype("uint8") * 255
        mask_pil = Image.fromarray(mask).resize((w, h), Image.LANCZOS)

        final_image = orig_image.copy()
        final_image.putalpha(mask_pil)

        buffer = io.BytesIO()
        final_image.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "image_base64": result_base64,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("BG error:", e)
        try:
            base64_data = request.image_base64.split(",")[-1]
            image_data = base64.b64decode(base64_data)
            return _best_effort_png_response(image_data, f"Unhandled BG error: {e}")
        except Exception:
            raise HTTPException(status_code=500, detail="Processing failed")


@router.post("/remove-bg/async", status_code=status.HTTP_202_ACCEPTED)
async def remove_background_async(request: BGRemoveRequest):
    if bg_remove_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        task = bg_remove_task.delay(request.image_base64)
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "task_type": "bg_remove_task",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue bg removal: {exc}")

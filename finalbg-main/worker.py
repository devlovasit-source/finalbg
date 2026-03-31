import sys
import os
import asyncio
import base64
import logging

import requests

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from services.settings import configure_logging

try:
    from sentry_sdk.integrations.redis import RedisIntegration
except Exception:
    RedisIntegration = None


# =========================
# SENTRY SETUP
# =========================
def _has_redis_client() -> bool:
    try:
        import redis  # noqa
        return True
    except Exception:
        return False


_sentry_integrations = [CeleryIntegration()]
if RedisIntegration is not None and _has_redis_client():
    _sentry_integrations.append(RedisIntegration())

_sentry_dsn = os.getenv("SENTRY_DSN")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=1.0,
        integrations=_sentry_integrations,
    )


# =========================
# CELERY INIT
# =========================
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ahvi_tasks",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

MAX_CLAIM_IMAGE_BYTES = 5 * 1024 * 1024
configure_logging()
logger = logging.getLogger("ahvi.worker")


def _download_image_claim_to_base64(image_url: str) -> str:
    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        raise ValueError(f"Failed to download claim-check image: {exc}") from exc

    image_bytes = response.content or b""
    if not image_bytes:
        raise ValueError("Claim-check image is empty")
    if len(image_bytes) > MAX_CLAIM_IMAGE_BYTES:
        raise ValueError("Claim-check image exceeds max allowed size")

    return base64.b64encode(image_bytes).decode("utf-8")


def _resolve_image_payload(image_ref: str) -> str:
    raw = str(image_ref or "").strip()
    if not raw:
        raise ValueError("image_ref is empty")

    if raw.startswith("redis://"):
        redis_key = raw.replace("redis://", "", 1)
        try:
            import redis

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            redis_client = redis.Redis.from_url(redis_url)
            value = redis_client.get(redis_key)
            if value is None:
                raise ValueError("Claim-check redis key expired or missing")
            redis_client.delete(redis_key)
            decoded = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            if not decoded.strip():
                raise ValueError("Claim-check redis value is empty")
            return decoded
        except Exception as exc:
            raise ValueError(f"Failed to load claim-check redis payload: {exc}") from exc

    if raw.startswith("http://") or raw.startswith("https://"):
        return _download_image_claim_to_base64(raw)
    return raw


def _raw_handler(func):
    return getattr(func, "__wrapped__", func)


# =========================
# AUDIO TASK
# =========================
@celery_app.task(name="generate_audio")
def run_heavy_audio_task(text_to_clone, lang):
    from services import audio_service

    try:
        audio_base64 = audio_service.generate_cloned_audio(text_to_clone, lang)
        return {"status": "success", "audio_base64": audio_base64}
    except Exception as e:
        logger.exception("AUDIO TASK ERROR")
        return {"status": "error", "message": str(e)}


# =========================
# IMAGE TASKS
# =========================
@celery_app.task(name="bg_remove_task")
def bg_remove_task(image_ref: str):
    from routers.bg_remover import BGRemoveRequest, remove_background

    try:
        image_base64 = _resolve_image_payload(image_ref)
        req = BGRemoveRequest(image_base64=image_base64)
        result = asyncio.run(_raw_handler(remove_background)(None, req, {}))
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("BG TASK ERROR")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="vision_analyze_task")
def vision_analyze_task(image_ref: str, user_id: str = "demo_user"):
    from routers.vision import ImageAnalyzeRequest, analyze_image

    try:
        image_base64 = _resolve_image_payload(image_ref)
        req = ImageAnalyzeRequest(image_base64=image_base64, userId=user_id)
        result = _raw_handler(analyze_image)(None, req, {})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("VISION TASK ERROR")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="capture_analyze_task")
def capture_analyze_task(user_id: str, image_ref: str):
    from routers.wardrobe_capture import CaptureAnalyzeRequest, analyze_capture

    try:
        image_base64 = _resolve_image_payload(image_ref)
        req = CaptureAnalyzeRequest(user_id=user_id, image_base64=image_base64)
        result = _raw_handler(analyze_capture)(None, req, {})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("CAPTURE ANALYZE TASK ERROR")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="capture_save_selected_task")
def capture_save_selected_task(payload: dict):
    from routers.wardrobe_capture import DetectedItem, SaveSelectedRequest, save_selected

    try:
        user_id = str(payload.get("user_id", ""))
        selected_item_ids = payload.get("selected_item_ids", []) or []
        detected_items_raw = payload.get("detected_items", []) or []
        detected_items = [DetectedItem(**item) for item in detected_items_raw]

        req = SaveSelectedRequest(
            user_id=user_id,
            selected_item_ids=selected_item_ids,
            detected_items=detected_items,
        )
        result = _raw_handler(save_selected)(None, req, {})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("CAPTURE SAVE TASK ERROR")
        return {"status": "error", "message": str(e)}


# =========================
# COMBINED ASYNC UPLOAD PIPELINE
# =========================
@celery_app.task(name="process_upload")
def process_upload_task(user_id: str, image_ref: str):
    """
    1) Analyze capture
    2) Auto-select all detected items
    3) Save selected items
    """
    try:
        analyzed = capture_analyze_task(user_id=user_id, image_ref=image_ref)
        if analyzed.get("status") != "success":
            return analyzed

        analysis_result = analyzed.get("result", {})
        items = analysis_result.get("items", []) or []
        selected_ids = [i.get("item_id") for i in items if i.get("item_id")]

        saved = capture_save_selected_task(
            payload={
                "user_id": user_id,
                "selected_item_ids": selected_ids,
                "detected_items": items,
            }
        )

        return {
            "status": "success",
            "analysis": analysis_result,
            "save": saved,
        }
    except Exception as e:
        logger.exception("PROCESS UPLOAD TASK ERROR")
        return {"status": "error", "message": str(e)}

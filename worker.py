import sys
import os
import asyncio

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration

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
        print("AUDIO TASK ERROR:", str(e))
        return {"status": "error", "message": str(e)}


# =========================
# IMAGE TASKS
# =========================
@celery_app.task(name="bg_remove_task")
def bg_remove_task(image_base64: str):
    from routers.bg_remover import BGRemoveRequest, remove_background

    try:
        req = BGRemoveRequest(image_base64=image_base64)
        result = asyncio.run(remove_background(req))
        return {"status": "success", "result": result}
    except Exception as e:
        print("BG TASK ERROR:", str(e))
        return {"status": "error", "message": str(e)}


@celery_app.task(name="vision_analyze_task")
def vision_analyze_task(image_base64: str, user_id: str = "demo_user"):
    from routers.vision import ImageAnalyzeRequest, analyze_image

    try:
        req = ImageAnalyzeRequest(image_base64=image_base64, userId=user_id)
        result = analyze_image(req)
        return {"status": "success", "result": result}
    except Exception as e:
        print("VISION TASK ERROR:", str(e))
        return {"status": "error", "message": str(e)}


@celery_app.task(name="capture_analyze_task")
def capture_analyze_task(user_id: str, image_base64: str):
    from routers.wardrobe_capture import CaptureAnalyzeRequest, analyze_capture

    try:
        req = CaptureAnalyzeRequest(user_id=user_id, image_base64=image_base64)
        result = analyze_capture(req)
        return {"status": "success", "result": result}
    except Exception as e:
        print("CAPTURE ANALYZE TASK ERROR:", str(e))
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
        result = save_selected(req)
        return {"status": "success", "result": result}
    except Exception as e:
        print("CAPTURE SAVE TASK ERROR:", str(e))
        return {"status": "error", "message": str(e)}


# =========================
# COMBINED ASYNC UPLOAD PIPELINE
# =========================
@celery_app.task(name="process_upload")
def process_upload_task(user_id: str, image_base64: str):
    """
    1) Analyze capture
    2) Auto-select all detected items
    3) Save selected items
    """
    try:
        analyzed = capture_analyze_task(user_id=user_id, image_base64=image_base64)
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
        print("PROCESS UPLOAD TASK ERROR:", str(e))
        return {"status": "error", "message": str(e)}

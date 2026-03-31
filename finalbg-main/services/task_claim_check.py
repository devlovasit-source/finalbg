import base64
import os
import re
import uuid
from typing import Tuple

from fastapi import HTTPException

from services.image_validation import validate_image_bytes
from services.r2_storage import R2Storage, R2StorageError


MAX_CLAIM_IMAGE_BYTES = 5 * 1024 * 1024


def decode_image_base64_payload(image_base64: str, *, max_bytes: int = MAX_CLAIM_IMAGE_BYTES) -> Tuple[bytes, str]:
    raw = (image_base64 or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    extension = "png"
    if raw.startswith("data:image/"):
        match = re.match(r"^data:image/([a-zA-Z0-9]+);base64,", raw)
        if match:
            extension = match.group(1).lower()
        raw = raw.split(",", 1)[1] if "," in raw else raw
    elif "," in raw:
        raw = raw.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {exc}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Decoded image payload is empty")
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image too large (max {max_bytes // (1024 * 1024)}MB)")
    detected_format = validate_image_bytes(
        image_bytes,
        allowed_formats=("PNG", "JPEG"),
        field_name="image_base64",
    )
    extension = "jpg" if detected_format == "JPEG" else "png"
    return image_bytes, extension


def store_image_claim_check(
    image_base64: str,
    *,
    user_id: str,
    task_type: str,
) -> str:
    image_bytes, extension = decode_image_base64_payload(image_base64)
    storage = R2Storage()
    safe_task_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_type or "task").strip("_") or "task"
    safe_user_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id or "anonymous").strip("_") or "anonymous"
    claim_id = uuid.uuid4().hex

    try:
        uploaded = storage.upload_task_claim_image(
            user_id=safe_user_id,
            task_type=safe_task_type,
            image_bytes=image_bytes,
            extension=extension,
            claim_id=claim_id,
        )
    except R2StorageError as exc:
        try:
            import redis

            key = f"task_claim:{safe_task_type}:{claim_id}"
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            ttl_seconds = int(os.getenv("TASK_CLAIM_TTL_SECONDS", "300"))
            redis_client = redis.Redis.from_url(redis_url)
            redis_client.setex(key, ttl_seconds, image_base64)
            return f"redis://{key}"
        except Exception as redis_exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store task image claim-check in R2 ({exc}) and Redis ({redis_exc})",
            )

    image_url = uploaded.get("image_url", "").strip()
    if not image_url:
        raise HTTPException(status_code=500, detail="Claim-check upload succeeded but URL is missing")
    return image_url

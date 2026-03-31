import base64
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.r2_storage import R2Storage, R2StorageError
from services.qdrant_service import qdrant_service
from services.image_validation import validate_image_bytes

router = APIRouter(tags=["utilities"])


class AnthropicMessage(BaseModel):
    role: str
    content: Any


class AnthropicRequest(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 380
    system: Optional[str] = None
    messages: List[AnthropicMessage]


class AvatarUploadRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=10)


class WardrobeUploadRequest(BaseModel):
    file_id: str
    raw_image_base64: str = Field(..., min_length=10)
    masked_image_base64: str = Field(..., min_length=10)


def _decode_base64_image(value: str, *, max_bytes: int, field_name: str) -> bytes:
    text = value or ""
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid base64: {exc}")
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{field_name} too large (max {max_bytes // (1024 * 1024)}MB)")
    validate_image_bytes(data, allowed_formats=("PNG", "JPEG"), field_name=field_name)
    return data


@router.post("/api/anthropic")
def anthropic_messages(request: AnthropicRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")

    payload: Dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [m.model_dump() for m in request.messages],
    }
    if request.system:
        payload["system"] = request.system

    try:
        res = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic request failed: {exc}")

    if res.status_code >= 400:
        raise HTTPException(status_code=res.status_code, detail=res.text)
    return res.json()


@router.get("/api/weather")
def weather(latitude: float, longitude: float):
    # Open-Meteo is keyless and suitable for quick weather snapshots.
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current=temperature_2m,weather_code,wind_speed_10m"
    )
    try:
        res = requests.get(url, timeout=20)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather request failed: {exc}")

    if res.status_code >= 400:
        raise HTTPException(status_code=res.status_code, detail=res.text)
    return res.json()


@router.post("/api/uploads/avatar")
def upload_avatar(request: AvatarUploadRequest):
    try:
        image_bytes = _decode_base64_image(
            request.image_base64,
            max_bytes=5 * 1024 * 1024,
            field_name="image_base64",
        )
        storage = R2Storage()
        avatar_url = storage.upload_avatar(user_id=request.user_id, image_bytes=image_bytes)
        return {"avatar_url": avatar_url}
    except R2StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid avatar payload: {exc}")


@router.post("/api/uploads/wardrobe")
def upload_wardrobe_images(request: WardrobeUploadRequest):
    try:
        raw_bytes = _decode_base64_image(
            request.raw_image_base64,
            max_bytes=5 * 1024 * 1024,
            field_name="raw_image_base64",
        )
        masked_bytes = _decode_base64_image(
            request.masked_image_base64,
            max_bytes=5 * 1024 * 1024,
            field_name="masked_image_base64",
        )
        storage = R2Storage()
        result = storage.upload_wardrobe_images(
            file_id=request.file_id,
            raw_image_bytes=raw_bytes,
            masked_image_bytes=masked_bytes,
        )
        return result
    except R2StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid wardrobe upload payload: {exc}")


@router.get("/api/qdrant/status")
def qdrant_status():
    return qdrant_service.status()


@router.get("/api/architecture/status")
def architecture_status():
    return {
        "experience_layer": {
            "frontend": "external_flutter_app",
            "boards": {"style_boards": True, "life_boards": True},
        },
        "core_intelligence_layer": {
            "context_engine": True,
            "style_graph_engine": True,
            "style_dna_engine": True,
            "agent_system": True,
            "execution_engine": True,
        },
        "api_ai_processing": {
            "fastapi_backend": True,
            "llm_service": True,
            "vision_pipeline": True,
            "async_processing": True,
        },
        "data_storage": {
            "appwrite": True,
            "qdrant": qdrant_service.status().get("enabled", False),
            "cloudflare_r2": True,
        },
        "response_flow": "boards_updated -> ui_rendered -> user_feedback",
    }

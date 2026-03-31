import base64
import re
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from middleware.auth_middleware import ensure_user_scope, get_current_user
from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.image_validation import validate_image_bytes
from services.r2_storage import R2Storage, R2StorageError

router = APIRouter(prefix="/api/boards", tags=["boards"])
proxy = AppwriteProxy()


class SaveBoardRequest(BaseModel):
    user_id: str
    title: str
    occasion: str = "Occasion"
    description: str = ""
    image_url: str = ""
    image_base64: str = ""
    board_ids: Optional[str] = None
    payload: Dict[str, Any] = {}


class SaveLifeBoardRequest(BaseModel):
    user_id: str
    title: str
    board_type: str = "daily_wear"
    description: str = ""
    payload: Dict[str, Any] = {}


def _clean_occasion(raw: str) -> str:
    v = (raw or "").strip().lower()
    mapping = {
        "party looks": "Party",
        "party": "Party",
        "office fit": "Office",
        "office": "Office",
        "vacation": "Vacation",
        "occasion": "Occasion",
    }
    return mapping.get(v, (raw or "Occasion").strip().title())


def _decode_image_base64(value: str) -> tuple[bytes, str]:
    text = (value or "").strip()
    if not text:
        return b"", "png"

    extension = "png"
    if text.startswith("data:image/"):
        match = re.match(r"^data:image/([a-zA-Z0-9]+);base64,", text)
        if match:
            extension = match.group(1).lower()
        text = text.split(",", 1)[1] if "," in text else text

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="image_base64 is empty")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image_base64 too large (max 5MB)")
    detected = validate_image_bytes(data, allowed_formats=("PNG", "JPEG"), field_name="image_base64")
    extension = "jpg" if detected == "JPEG" else "png"

    return data, extension


@router.get("")
def list_boards(user_id: str, occasion: Optional[str] = None, limit: int = 100, user=Depends(get_current_user)):
    ensure_user_scope(user, user_id)
    try:
        docs = proxy.list_documents(
            "saved_boards",
            user_id=user_id,
            occasion=_clean_occasion(occasion) if occasion else None,
            limit=limit,
        )
        return {"documents": docs}
    except AppwriteProxyError as exc:
        print(f"[boards.list] user_id={user_id} occasion={occasion} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/save")
def save_board(request: SaveBoardRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, request.user_id)
    try:
        final_image_url = request.image_url.strip()
        if request.image_base64.strip():
            image_bytes, extension = _decode_image_base64(request.image_base64)
            storage = R2Storage()
            uploaded = storage.upload_style_board_image(
                user_id=request.user_id,
                image_bytes=image_bytes,
                extension=extension,
            )
            final_image_url = uploaded.get("image_url", final_image_url)

        item_ids: list[str] = []
        if request.board_ids:
            item_ids = [x.strip() for x in request.board_ids.split(",") if x.strip()]
        elif isinstance(request.payload, dict):
            raw_item_ids = request.payload.get("itemIds") or request.payload.get("boardIds") or []
            if isinstance(raw_item_ids, list):
                item_ids = [str(x).strip() for x in raw_item_ids if str(x).strip()]

        doc = {
            "userId": request.user_id,
            "occasion": _clean_occasion(request.occasion),
            "imageUrl": final_image_url,
            "itemIds": item_ids,
        }
        created = proxy.create_document("saved_boards", doc)
        return {"document": created}
    except R2StorageError as exc:
        raise HTTPException(status_code=500, detail=f"R2 upload failed: {exc}")
    except AppwriteProxyError as exc:
        print(f"[boards.save] user_id={request.user_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/life")
def list_life_boards(user_id: str, limit: int = 100, user=Depends(get_current_user)):
    ensure_user_scope(user, user_id)
    try:
        docs = proxy.list_documents("life_boards", user_id=user_id, limit=limit)
        return {"documents": docs}
    except AppwriteProxyError as exc:
        print(f"[boards.life.list] user_id={user_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/life/save")
def save_life_board(request: SaveLifeBoardRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, request.user_id)
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        doc = {
            "userId": request.user_id,
            "title": request.title.strip() or "Life Board",
            "boardType": request.board_type.strip() or "daily_wear",
            "description": request.description.strip(),
            "payload": request.payload or {},
            "createdAt": now_iso,
            "updatedAt": now_iso,
        }
        created = proxy.create_document("life_boards", doc)
        return {"document": created}
    except AppwriteProxyError as exc:
        print(f"[boards.life.save] user_id={request.user_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{document_id}")
def delete_board(document_id: str, user=Depends(get_current_user)):
    try:
        existing = proxy.get_document("saved_boards", document_id)
        ensure_user_scope(user, str(existing.get("userId", "")))
        proxy.delete_document("saved_boards", document_id)
        return {"ok": True}
    except AppwriteProxyError as exc:
        print(f"[boards.delete] document_id={document_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))

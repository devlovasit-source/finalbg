from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError

router = APIRouter(prefix="/api/data", tags=["data"])
proxy = AppwriteProxy()


class CreateRequest(BaseModel):
    resource: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    document_id: Optional[str] = None


class UpdateRequest(BaseModel):
    resource: str
    data: Dict[str, Any]


class DeleteRequest(BaseModel):
    resource: str
    document_id: str


def _http_error_from_proxy(exc: AppwriteProxyError) -> HTTPException:
    msg = str(exc)
    if "connection failed" in msg.lower():
        return HTTPException(status_code=503, detail=msg)
    if "(404)" in msg:
        return HTTPException(status_code=404, detail=msg)
    if "(401)" in msg or "(403)" in msg:
        return HTTPException(status_code=502, detail=msg)
    return HTTPException(status_code=400, detail=msg)


@router.get("/{resource}")
def list_documents(resource: str, user_id: Optional[str] = None, occasion: Optional[str] = None, limit: int = 100):
    try:
        docs = proxy.list_documents(resource, user_id=user_id, occasion=occasion, limit=limit)
        return {"documents": docs}
    except AppwriteProxyError as exc:
        print(f"[data.list_documents] resource={resource} user_id={user_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.get("/{resource}/{document_id}")
def get_document(resource: str, document_id: str):
    try:
        doc = proxy.get_document(resource, document_id)
        return {"document": doc}
    except AppwriteProxyError as exc:
        print(f"[data.get_document] resource={resource} document_id={document_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.post("")
def create_document(request: CreateRequest):
    try:
        payload = dict(request.data)
        user_field = proxy.user_field_map.get(request.resource)
        if request.user_id and user_field and user_field not in payload:
            payload[user_field] = request.user_id

        doc = proxy.create_document(
            request.resource,
            payload,
            document_id=request.document_id or "unique()",
        )
        return {"document": doc}
    except AppwriteProxyError as exc:
        print(f"[data.create_document] resource={request.resource} error={exc}")
        raise _http_error_from_proxy(exc)


@router.patch("/{document_id}")
def update_document(document_id: str, request: UpdateRequest):
    try:
        doc = proxy.update_document(request.resource, document_id, request.data)
        return {"document": doc}
    except AppwriteProxyError as exc:
        print(f"[data.update_document] resource={request.resource} document_id={document_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.delete("")
def delete_document(request: DeleteRequest):
    try:
        proxy.delete_document(request.resource, request.document_id)
        return {"ok": True}
    except AppwriteProxyError as exc:
        print(f"[data.delete_document] resource={request.resource} document_id={request.document_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.put("/users/{user_id}")
def upsert_user_profile(user_id: str, body: Dict[str, Any]):
    try:
        try:
            doc = proxy.update_document("users", user_id, body)
        except AppwriteProxyError:
            doc = proxy.create_document("users", body, document_id=user_id)
        return {"document": doc}
    except AppwriteProxyError as exc:
        print(f"[data.upsert_user_profile] user_id={user_id} error={exc}")
        raise _http_error_from_proxy(exc)

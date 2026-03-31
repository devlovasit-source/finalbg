#!/usr/bin/env python3
import os
from typing import Dict, List

from dotenv import load_dotenv


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        val = os.getenv(key, "").strip()
        if val:
            return val
    return default


def init_qdrant() -> None:
    try:
        from services.qdrant_service import QdrantService

        svc = QdrantService()
        svc.init()
        status = svc.status()
        print(f"[qdrant] initialized={status.get('initialized')} collections={status.get('collections', [])}")
    except Exception as exc:
        print(f"[qdrant] init failed: {exc}")


def _appwrite_client():
    from appwrite.client import Client

    endpoint = _env("APPWRITE_ENDPOINT", "EXPO_PUBLIC_APPWRITE_ENDPOINT", default="https://cloud.appwrite.io/v1")
    project = _env("APPWRITE_PROJECT_ID", "EXPO_PUBLIC_APPWRITE_PROJECT_ID")
    api_key = _env("APPWRITE_API_KEY", "EXPO_PUBLIC_APPWRITE_API_KEY", "APPWRITE_KEY")

    if not endpoint or not project or not api_key:
        raise RuntimeError("Missing APPWRITE endpoint/project/key env for DB init")

    client = Client()
    client.set_endpoint(endpoint)
    client.set_project(project)
    client.set_key(api_key)
    return client


def _ensure_collection(databases, database_id: str, collection_id: str, name: str) -> None:
    existing = databases.list_collections(database_id)
    docs = existing.get("collections", []) if isinstance(existing, dict) else []
    existing_ids = {str(c.get("$id", "")) for c in docs if isinstance(c, dict)}
    if collection_id in existing_ids:
        print(f"[appwrite] collection exists: {collection_id}")
        return

    try:
        databases.create_collection(
            database_id=database_id,
            collection_id=collection_id,
            name=name,
            permissions=[],
            document_security=False,
            enabled=True,
        )
    except TypeError:
        # Compatibility with older SDK signatures.
        databases.create_collection(database_id, collection_id, name, [], False, True)

    print(f"[appwrite] created collection: {collection_id}")


def _safe_create_string_attribute(databases, database_id: str, collection_id: str, key: str, size: int = 255) -> None:
    try:
        databases.create_string_attribute(
            database_id=database_id,
            collection_id=collection_id,
            key=key,
            size=size,
            required=False,
            xdefault="",
            array=False,
        )
        print(f"[appwrite] attribute created: {collection_id}.{key}")
    except Exception:
        # attribute may already exist or backend policy may lock schema changes
        pass


def init_appwrite() -> None:
    try:
        from appwrite.services.databases import Databases

        client = _appwrite_client()
        databases = Databases(client)

        database_id = _env("APPWRITE_DATABASE_ID", "EXPO_PUBLIC_APPWRITE_DATABASE_ID")
        if not database_id:
            raise RuntimeError("Missing APPWRITE_DATABASE_ID")

        collection_map: Dict[str, str] = {
            "outfits": _env("APPWRITE_COLLECTION_OUTFITS", "EXPO_PUBLIC_APPWRITE_COLLECTION_OUTFITS", default="outfits"),
            "saved_boards": _env("APPWRITE_COLLECTION_SAVED_BOARDS", "EXPO_PUBLIC_APPWRITE_COLLECTION_SAVED_BOARDS", default="saved_boards"),
            "users": _env("APPWRITE_COLLECTION_USERS", "EXPO_PUBLIC_APPWRITE_COLLECTION_USERS", default="users"),
            "life_boards": _env("APPWRITE_COLLECTION_LIFE_BOARDS", "EXPO_PUBLIC_APPWRITE_COLLECTION_LIFE_BOARDS", default="life_boards"),
        }

        for name, cid in collection_map.items():
            _ensure_collection(databases, database_id, cid, name.replace("_", " ").title())

        # Best-effort core fields used by backend.
        for cid in collection_map.values():
            _safe_create_string_attribute(databases, database_id, cid, "userId", 64)

        _safe_create_string_attribute(databases, database_id, collection_map["outfits"], "category", 64)
        _safe_create_string_attribute(databases, database_id, collection_map["outfits"], "sub_category", 64)
        _safe_create_string_attribute(databases, database_id, collection_map["outfits"], "image_url", 2048)
        _safe_create_string_attribute(databases, database_id, collection_map["saved_boards"], "imageUrl", 2048)

        print("[appwrite] initialization complete")
    except Exception as exc:
        print(f"[appwrite] init failed: {exc}")


def main() -> None:
    load_dotenv()
    print("Starting DB/bootstrap initialization...")
    init_qdrant()
    init_appwrite()
    print("Initialization finished.")


if __name__ == "__main__":
    main()

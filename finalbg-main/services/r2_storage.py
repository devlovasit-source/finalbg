import os
from io import BytesIO
from typing import Dict
import uuid
import re

try:
    from minio import Minio
except Exception:
    Minio = None


class R2StorageError(Exception):
    pass


def _env(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback)


def _load_local_env() -> None:
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    candidate_paths = [
        os.path.join(cwd, ".env"),
        os.path.join(parent, ".env"),
        os.path.join(parent, "backend", ".env"),
        os.path.join(parent, "backend-master", ".env"),
        os.path.join(parent, "ahvi", ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and (key not in os.environ or not str(os.environ.get(key, "")).strip()):
                        os.environ[key] = value
        except Exception:
            continue


class R2Storage:
    def __init__(self) -> None:
        _load_local_env()
        self.s3_url = _env("R2_S3_API_URL") or _env("EXPO_PUBLIC_R2_S3_API_URL")
        self.access_key = _env("R2_ACCESS_KEY_ID") or _env("EXPO_PUBLIC_R2_ACCESS_KEY_ID")
        self.secret_key = _env("R2_SECRET_ACCESS_KEY") or _env("EXPO_PUBLIC_R2_SECRET_ACCESS_KEY")
        self.raw_bucket = _env("R2_BUCKET_RAW_IMAGES") or _env("EXPO_PUBLIC_R2_BUCKET_RAW_IMAGES")
        self.raw_public_url = _env("R2_URL_RAW_IMAGES") or _env("EXPO_PUBLIC_R2_URL_RAW_IMAGES")
        self.wardrobe_bucket = _env("R2_BUCKET_WARDROBE") or _env("EXPO_PUBLIC_R2_BUCKET_WARDROBE")
        self.wardrobe_public_url = _env("R2_URL_WARDROBE") or _env("EXPO_PUBLIC_R2_URL_WARDROBE")
        self.style_boards_bucket = _env("R2_BUCKET_STYLE_BOARDS") or _env("EXPO_PUBLIC_R2_BUCKET_STYLE_BOARDS")
        self.style_boards_public_url = _env("R2_URL_STYLE_BOARDS") or _env("EXPO_PUBLIC_R2_URL_STYLE_BOARDS")

    def _client(self):
        if Minio is None:
            raise R2StorageError("minio package is not installed on backend.")

        if not self.s3_url or not self.access_key or not self.secret_key:
            raise R2StorageError("Missing R2 backend configuration.")

        endpoint = self.s3_url.replace("https://", "").replace("http://", "")
        return Minio(
            endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region="auto",
        )

    def upload_avatar(self, *, user_id: str, image_bytes: bytes) -> str:
        if not self.raw_bucket or not self.raw_public_url:
            raise R2StorageError("Missing raw bucket/public URL configuration.")

        file_name = f"avatar_{user_id}_{int.from_bytes(os.urandom(4), 'big')}.png"
        client = self._client()
        client.put_object(
            self.raw_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/png",
        )
        return f"{self.raw_public_url}/{file_name}"

    def upload_wardrobe_images(
        self,
        *,
        file_id: str,
        raw_image_bytes: bytes,
        masked_image_bytes: bytes,
    ) -> Dict[str, str]:
        if not self.raw_bucket or not self.raw_public_url:
            raise R2StorageError("Missing raw bucket/public URL configuration.")
        if not self.wardrobe_bucket or not self.wardrobe_public_url:
            raise R2StorageError("Missing wardrobe bucket/public URL configuration.")

        raw_file_name = f"raw_{file_id}.png"
        masked_file_name = f"wardrobe_{file_id}.png"
        client = self._client()

        client.put_object(
            self.raw_bucket,
            raw_file_name,
            BytesIO(raw_image_bytes),
            length=len(raw_image_bytes),
            content_type="image/png",
        )
        client.put_object(
            self.wardrobe_bucket,
            masked_file_name,
            BytesIO(masked_image_bytes),
            length=len(masked_image_bytes),
            content_type="image/png",
        )

        return {
            "raw_file_name": raw_file_name,
            "masked_file_name": masked_file_name,
            "raw_image_url": f"{self.raw_public_url}/{raw_file_name}",
            "masked_image_url": f"{self.wardrobe_public_url}/{masked_file_name}",
        }

    def upload_style_board_image(
        self,
        *,
        user_id: str,
        image_bytes: bytes,
        extension: str = "png",
    ) -> Dict[str, str]:
        # Prefer dedicated style-boards bucket if configured, otherwise fall back to raw bucket.
        target_bucket = self.style_boards_bucket or self.raw_bucket
        target_public_url = self.style_boards_public_url or self.raw_public_url
        if not target_bucket or not target_public_url:
            raise R2StorageError("Missing style board bucket/public URL configuration.")

        ext = (extension or "png").lower().strip(".")
        if ext not in ("png", "jpg", "jpeg", "webp"):
            ext = "png"

        file_name = f"style_board_{user_id}_{uuid.uuid4().hex}.{ext}"
        content_type = "image/png"
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext == "webp":
            content_type = "image/webp"

        client = self._client()
        client.put_object(
            target_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=content_type,
        )

        return {
            "file_name": file_name,
            "image_url": f"{target_public_url}/{file_name}",
            "content_type": content_type,
        }

    def upload_task_claim_image(
        self,
        *,
        user_id: str,
        task_type: str,
        image_bytes: bytes,
        extension: str = "png",
        claim_id: str | None = None,
    ) -> Dict[str, str]:
        target_bucket = self.raw_bucket or self.style_boards_bucket
        target_public_url = self.raw_public_url or self.style_boards_public_url
        if not target_bucket or not target_public_url:
            raise R2StorageError("Missing task claim bucket/public URL configuration.")

        ext = (extension or "png").lower().strip(".")
        if ext not in ("png", "jpg", "jpeg", "webp"):
            ext = "png"

        content_type = "image/png"
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext == "webp":
            content_type = "image/webp"

        safe_user = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id or "anonymous").strip("_") or "anonymous"
        safe_task = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_type or "task").strip("_") or "task"
        file_name = f"claims/{safe_task}_{safe_user}_{(claim_id or uuid.uuid4().hex)}.{ext}"

        client = self._client()
        client.put_object(
            target_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=content_type,
        )

        return {
            "file_name": file_name,
            "image_url": f"{target_public_url}/{file_name}",
            "content_type": content_type,
        }

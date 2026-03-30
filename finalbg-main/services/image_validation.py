from __future__ import annotations

from io import BytesIO
from typing import Iterable

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError


def validate_image_bytes(
    image_bytes: bytes,
    *,
    allowed_formats: Iterable[str] = ("PNG", "JPEG"),
    field_name: str = "image",
) -> str:
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")

    allowed = {str(x).upper() for x in allowed_formats}
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(BytesIO(image_bytes)) as img2:
            fmt = str(img2.format or "").upper()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} failed validation: {exc}")

    if fmt not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"{field_name} must be one of: {', '.join(sorted(allowed))}",
        )
    return fmt

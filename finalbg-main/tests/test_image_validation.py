from io import BytesIO

from PIL import Image

from services.image_validation import validate_image_bytes


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_validate_image_bytes_accepts_png():
    png = _make_png_bytes()
    fmt = validate_image_bytes(png, allowed_formats=("PNG", "JPEG"), field_name="image")
    assert fmt == "PNG"

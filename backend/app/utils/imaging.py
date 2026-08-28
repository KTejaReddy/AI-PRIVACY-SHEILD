"""Image helpers: safe loading, orientation normalization, encoding, validation."""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import settings

logger = logging.getLogger(__name__)

# Magic-byte signatures -> format. We never trust file extensions alone.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "webp"),  # RIFF....WEBP
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)


@dataclass
class ImageRecord:
    """A validated, orientation-normalized image loaded into memory."""

    pil: Image.Image  # RGB, orientation-corrected
    array: np.ndarray  # HxWx3 uint8 RGB
    width: int
    height: int
    source_format: str  # detected from magic bytes, e.g. "jpeg"
    source_size_bytes: int
    had_exif: bool
    had_gps: bool
    had_xmp: bool

    def numpy(self) -> np.ndarray:
        return self.array

    def to_pil(self) -> Image.Image:
        return self.pil.copy()


def sniff_format(raw: bytes) -> str | None:
    """Detect the image format from magic bytes (no extension trust)."""
    if len(raw) < 12:
        return None
    for magic, fmt in _MAGIC:
        if raw.startswith(magic):
            if fmt == "webp":
                # confirm "WEBP" appears in the RIFF header
                return "webp" if raw[8:12] == b"WEBP" else None
            return fmt
    return None


def exif_has_gps(exif) -> bool:
    try:
        return bool(exif and exif.get_ifd(0x8825))
    except Exception:  # noqa: BLE001 - malformed EXIF must not crash validation
        return False


def load_and_normalize(raw: bytes, source_size_bytes: int) -> ImageRecord:
    """Validate raw bytes, load, transpose EXIF orientation, and convert to RGB.

    Raises ``ValueError`` with a user-safe message for anything invalid.
    """
    fmt = sniff_format(raw)
    if fmt is None:
        raise ValueError(
            "Unsupported or unrecognized image format. Supported: JPEG, PNG, WebP, BMP, GIF, TIFF."
        )
    if len(raw) > settings.MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Image exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("The uploaded file could not be decoded as an image.") from exc

    exif = pil.getexif()
    had_exif = bool(exif)
    had_gps = exif_has_gps(exif)
    had_xmp = "xmp" in (pil.info or {})

    # Normalize orientation so the working image matches what the user sees.
    pil = ImageOps.exif_transpose(pil)

    if pil.mode not in ("RGB", "RGBA", "L", "P"):
        # TIFF with unusual modes and CMYK JPEGs -> convert through RGB path
        pil = pil.convert("RGB")
    if pil.mode != "RGB":
        rgba = pil.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        pil = background

    if max(pil.size) > settings.MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image is too large: max dimension is {settings.MAX_IMAGE_DIMENSION} pixels."
        )

    arr = np.asarray(pil, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Image could not be converted to RGB.")
    return ImageRecord(
        pil=pil,
        array=arr,
        width=arr.shape[1],
        height=arr.shape[0],
        source_format=fmt,
        source_size_bytes=source_size_bytes,
        had_exif=had_exif,
        had_gps=had_gps,
        had_xmp=had_xmp,
    )


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to clean PNG bytes (no metadata)."""
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def array_to_bytes(arr: np.ndarray, fmt: str = "png", quality: int = 95) -> bytes:
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=quality, exif=b"")
    else:
        img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def to_data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def normalize_array_dtype(arr: np.ndarray) -> np.ndarray:
    """Clip and cast a float array to uint8 (safe for encoders)."""
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)

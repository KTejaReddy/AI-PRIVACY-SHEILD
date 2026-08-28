"""Sensitive-content analysis and metadata sanitization."""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest
from PIL import Image

from app.vision.sensitive import analyze_sensitive, _classify_text
from app.metadata.sanitizer import sanitize_and_encode

sys.path.insert(0, str(io))


def _qr_image_bytes(text: str) -> bytes:
    qrcode = pytest.importorskip("qrcode")
    img = qrcode.make(text)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_qr_code_detected_and_flagged_sensitive():
    data = _qr_image_bytes("https://example.com/personal?id=12345")
    arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    result = analyze_sensitive(arr, face_boxes=[])
    assert len(result["qr_codes"]) == 1
    region = result["qr_codes"][0]
    assert region["sensitive"] is True
    assert region["kind"] == "qr"
    assert "example.com" in (region["content"] or "")


def test_no_qr_in_plain_image(gradient_png_bytes):
    arr = np.asarray(Image.open(io.BytesIO(gradient_png_bytes)).convert("RGB"))
    result = analyze_sensitive(arr, face_boxes=[])
    assert result["qr_codes"] == []


def test_pii_classification():
    sensitive, cats = _classify_text("Contact me at alice@example.com or 555-123-4567")
    assert sensitive is True
    assert "email" in cats
    assert "phone" in cats

    sensitive, cats = _classify_text("Call me on 5551234567 tomorrow")
    assert sensitive is True
    assert "phone" in cats

    sensitive, _ = _classify_text("The weather is nice today")
    assert sensitive is False


def test_metadata_sanitized():
    # Build a JPEG with EXIF, then confirm the sanitizer drops it.
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 2] = 90
    base = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x010F] = "Test Camera"  # Make
    exif[0x8825] = {1: "N", 2: (45, 1), 3: "W", 4: (12, 1)}  # GPS IFD
    base.save(buf, format="JPEG", exif=exif)

    loaded = Image.open(io.BytesIO(buf.getvalue()))
    assert bool(loaded.getexif())

    data, report = sanitize_and_encode(
        arr, source_had_exif=True, source_had_gps=True, source_had_xmp=False
    )
    out = Image.open(io.BytesIO(data))
    exif_out = out.getexif()
    assert not bool(exif_out)
    assert "EXIF" in " ".join(report.removed)
    assert "GPS" in " ".join(report.removed)

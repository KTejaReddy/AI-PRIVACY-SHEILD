"""Metadata sanitization.

The protected image is re-encoded from raw pixel data, so EXIF / XMP / IPTC
metadata from the original is never carried over. PNG (the default output) has
no metadata container at all. JPEG output is written with an empty EXIF block.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from ..utils.imaging import array_to_bytes

logger = logging.getLogger(__name__)


@dataclass
class MetadataReport:
    source_had_exif: bool
    source_had_gps: bool
    source_had_xmp: bool
    output_format: str
    removed: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "source_had_exif": self.source_had_exif,
            "source_had_gps": self.source_had_gps,
            "source_had_xmp": self.source_had_xmp,
            "removed": self.removed,
            "output_format": self.output_format,
            "note": self.note,
        }


def sanitize_and_encode(
    protected_rgb: np.ndarray,
    source_had_exif: bool,
    source_had_gps: bool,
    source_had_xmp: bool,
) -> tuple[bytes, MetadataReport]:
    """Encode the protected image with no metadata. Returns (bytes, report)."""
    fmt = settings.OUTPUT_FORMAT
    quality = settings.JPEG_QUALITY
    data = array_to_bytes(protected_rgb, fmt=fmt, quality=quality)

    removed: list[str] = []
    if source_had_exif:
        removed.append("EXIF (camera, lens, settings)")
    if source_had_gps:
        removed.append("GPS location")
    if source_had_xmp:
        removed.append("XMP/IPTC")
    if fmt == "png":
        removed.append("metadata container (PNG has none)")
    else:
        removed.append("EXIF on re-encode")

    note = (
        "Output is re-encoded from pixel data only; no source metadata is preserved."
        if fmt == "png"
        else "Output re-encoded as high-quality JPEG with no EXIF."
    )
    report = MetadataReport(
        source_had_exif=source_had_exif,
        source_had_gps=source_had_gps,
        source_had_xmp=source_had_xmp,
        output_format=fmt,
        removed=removed,
        note=note,
    )
    return data, report

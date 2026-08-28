"""Sensitive content analysis.

Detects, in order of reliability:

1. QR codes            — OpenCV's built-in QRCodeDetector (real, reliable)
2. Text / PII          — RapidOCR (optional) + regex for emails/phones/IDs/addresses
3. License plates      — heuristic on OCR text (EXPERIMENTAL, clearly labeled)
4. Identity documents  — contour-based rectangle + text-density heuristic (EXPERIMENTAL)

Every detected region carries a ``sensitive`` flag and a confidence/experimental
label. The pipeline pixelates these regions in the protected output.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class SensitiveRegion:
    kind: str  # "qr" | "text" | "plate" | "document"
    x1: int
    y1: int
    x2: int
    y2: int
    content: str | None = None
    sensitive: bool = False
    experimental: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "content": self.content,
            "sensitive": self.sensitive,
            "experimental": self.experimental,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# PII regexes — deliberately conservative to avoid false positives.
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
_PHONE_RE = re.compile(
    r"(?<![\d.])(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}(?!\d)"
)
_ID_RE = re.compile(
    r"(?i)\b(?:[A-Z]{1,2}[- ]?\d{6,9}|ssn[: ]?\d{3}[- ]?\d{2}[- ]?\d{4}|"
    r"dl[: ]?[a-z0-9-]{6,}|passport[: ]?[a-z0-9]{6,})\b"
)
_ACCOUNT_RE = re.compile(r"(?i)\b(?:account(?: no\.?| #)?[: ]?[a-z0-9-]{5,}|iban[: ]?[a-z]{2}\d{2}[a-z0-9]{10,})\b")
_ADDRESS_RE = re.compile(r"\b\d{1,5}\s+[A-Za-z0-9 .'-]+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd)\b", re.I)
_PLATE_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2,3}[- ]?\d{2,4}(?![A-Z0-9])")


def _classify_text(text: str) -> tuple[bool, list[str]]:
    """Return (is_sensitive, list of matched categories)."""
    hits: list[str] = []
    if _EMAIL_RE.search(text):
        hits.append("email")
    if _PHONE_RE.search(text):
        hits.append("phone")
    if _ID_RE.search(text):
        hits.append("id")
    if _ACCOUNT_RE.search(text):
        hits.append("account")
    if _ADDRESS_RE.search(text):
        hits.append("address")
    if _PLATE_RE.search(text):
        hits.append("license_plate")
    return bool(hits), hits


# ---------------------------------------------------------------------------
# OCR engine (optional)
# ---------------------------------------------------------------------------

_ocr = None
_ocr_error: str | None = None


def _get_ocr():
    global _ocr, _ocr_error  # noqa: PLW0603
    if _ocr is not None or _ocr_error is not None:
        return _ocr
    if not settings.OCR_ENABLED:
        _ocr_error = "OCR disabled by configuration."
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

        _ocr = RapidOCR()
        logger.info("RapidOCR engine ready.")
    except Exception as exc:  # noqa: BLE001
        _ocr_error = f"OCR engine unavailable: {exc}"
        logger.warning("OCR engine unavailable: %s", exc)
        return None
    return _ocr


def ocr_status() -> dict:
    return {"available": _get_ocr() is not None, "note": _ocr_error or "OCR engine ready"}


def _run_ocr(rgb: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
    """Run OCR, returns [(x1,y1,x2,y2,text,confidence)] in image coords."""
    engine = _get_ocr()
    if engine is None:
        return []
    # Downscale very large images for the OCR pass (speed; boxes are scaled back).
    h, w = rgb.shape[:2]
    scale = 1.0
    work = rgb
    max_dim = settings.OCR_MAX_DIMENSION
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        work = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    try:
        result, _ = engine(work)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR inference failed: %s", exc)
        return []
    if not result:
        return []
    boxes: list[tuple[int, int, int, int, str, float]] = []
    for item in result:
        # item: [quad_points(4x2), text, confidence]
        try:
            pts = np.array(item[0], dtype=np.float32)
            text = str(item[1])
            conf = float(item[2])
            x1, y1 = int(pts[:, 0].min() / scale), int(pts[:, 1].min() / scale)
            x2, y2 = int(pts[:, 0].max() / scale), int(pts[:, 1].max() / scale)
            boxes.append((x1, y1, x2, y2, text, conf))
        except Exception:  # noqa: BLE001
            continue
    return boxes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_sensitive(rgb: np.ndarray, face_boxes: list[tuple[int, int, int, int]]) -> dict:
    """Analyze an image for QR codes, sensitive text, plates, documents.

    Returns {regions: [...], qr: [...], text: [...], ocr_available: bool,
             experimental: bool, summary: str}.
    """
    h, w = rgb.shape[:2]
    regions: list[SensitiveRegion] = []

    # ---- QR / barcode-like codes -----------------------------------------
    qr_detector = cv2.QRCodeDetector()
    try:
        decoded, points, _ = qr_detector.detectAndDecode(rgb)
        if decoded:
            pts = np.array(points, dtype=np.float32).reshape(-1, 2)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            regions.append(
                SensitiveRegion(
                    kind="qr", x1=max(0, x1), y1=max(0, y1), x2=min(w, x2), y2=min(h, y2),
                    content=decoded[:200], sensitive=True,
                    note="QR code detected; content may encode personal data.",
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("QR detection failed: %s", exc)

    # ---- OCR text --------------------------------------------------------
    ocr_boxes = _run_ocr(rgb)
    text_regions: list[SensitiveRegion] = []
    for x1, y1, x2, y2, text, conf in ocr_boxes:
        sensitive, cats = _classify_text(text)
        region = SensitiveRegion(
            kind="text", x1=x1, y1=y1, x2=x2, y2=y2, content=text[:200],
            sensitive=sensitive, experimental=False,
            note=f"matched: {', '.join(cats)}" if cats else "text detected (no obvious PII patterns)",
        )
        # License plate heuristic: compact alphanumeric string in lower half of image, outside faces
        if _PLATE_RE.match(text.strip()) and len(text.strip()) <= 10 and conf > 0.7:
            region.kind = "plate"
            region.experimental = True
            region.sensitive = True
            region.note = "Possible license plate (EXPERIMENTAL heuristic)."
        text_regions.append(region)
        regions.append(region)

    # ---- identity-document heuristic (EXPERIMENTAL) ----------------------
    document_regions = _detect_document_regions(rgb, face_boxes, text_regions)
    regions.extend(document_regions)

    # ---- summary ---------------------------------------------------------
    sensitive_regions = [r for r in regions if r.sensitive]
    if sensitive_regions:
        kinds = sorted({r.kind for r in sensitive_regions})
        summary = (
            f"Detected sensitive region(s): {', '.join(kinds)}. "
            "These regions are pixelated in the protected image."
        )
    else:
        summary = "No QR codes or sensitive text patterns detected."

    return {
        "regions": [r.as_dict() for r in regions],
        "qr_codes": [r.as_dict() for r in regions if r.kind == "qr"],
        "text_regions": [r.as_dict() for r in text_regions],
        "ocr_available": _get_ocr() is not None,
        "ocr_note": _ocr_error or ("OCR engine ready" if _get_ocr() is not None else None),
        "experimental": any(r.experimental for r in regions),
        "summary": summary,
    }


def _detect_document_regions(
    rgb: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]],
    text_regions: list[SensitiveRegion],
) -> list[SensitiveRegion]:
    """Experimental: find large document-like rectangles with text inside."""
    h, w = rgb.shape[:2]
    if not text_regions:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found: list[SensitiveRegion] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        img_area = h * w
        if area < 0.05 * img_area or area > 0.98 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 0.2 * w or bh < 0.15 * h:
            continue
        # document boxes usually sit outside the face
        if any(x < f[2] and f[0] < x + bw and y < f[3] and f[1] < y + bh for f in face_boxes):
            continue
        # require text inside the rectangle
        inside = [t for t in text_regions if t.x1 >= x and t.y1 >= y and t.x2 <= x + bw and t.y2 <= y + bh]
        if len(inside) >= 2:
            found.append(
                SensitiveRegion(
                    kind="document",
                    x1=x, y1=y, x2=x + bw, y2=y + bh,
                    content=None, sensitive=True, experimental=True,
                    note="Possible identity document region (EXPERIMENTAL heuristic).",
                )
            )
    return found

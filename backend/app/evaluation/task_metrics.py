"""Task-specific, region-aware edit-success metrics.

A single raw CLIP cosine is not evidence that an edit "happened". Every
benchmark task therefore gets a *pixel-level, deterministic* measurement of
whether the requested change actually occurred, computed in the image region
the edit should affect:

    t01 shirt color   -> redness change in the shirt region
    t02 background    -> structural change in the background mask
    t03 add a hat     -> structural change in the hair/top-of-head region
    t04 lighting      -> warm-light shift (mean R - B) across the photo
    t05 pencil sketch -> saturation drop + edge-density increase
    t06 hairstyle     -> structural change in the hair region

Each metric returns ``(raw, success)`` where ``success`` is normalized to
[0, 1] (0 = the edit did not measurably happen, 1 = a strong change). The
scaling constants are documented per metric and were calibrated against the
local editors' actual output magnitudes.

CLIP remains only an *auxiliary* semantic check (see editing_benchmark.py);
it never is the primary evidence of edit success.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# region helpers
# ---------------------------------------------------------------------------


def _clip_box(box: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def union_box(boxes: Iterable, h: int, w: int) -> Optional[tuple[int, int, int, int]]:
    """Union of zero or more (x1, y1, x2, y2) boxes."""
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x1 = min(int(b[0]) for b in boxes)
    y1 = min(int(b[1]) for b in boxes)
    x2 = max(int(b[2]) for b in boxes)
    y2 = max(int(b[3]) for b in boxes)
    return _clip_box((x1, y1, x2, y2), h, w)


def shirt_region(h: int, w: int, faces=None, persons=None) -> tuple[int, int, int, int]:
    """The torso: person box rows below the face; falls back to a mid band."""
    person = union_box(persons or [], h, w)
    face = union_box(faces or [], h, w)
    if person is not None:
        px1, py1, px2, py2 = person
        if face is not None:
            fx1, fy1, fx2, fy2 = face
            top = min(py2 - 1, max(py1, fy2))
        else:
            top = py1 + (py2 - py1) // 3
        bottom = py2
        # the torso is roughly face-height deep, centered on the person
        bottom = min(py2, top + int(1.1 * (face[3] - face[1])) if face is not None else top + (py2 - py1) // 2)
        return _clip_box((px1, top, px2, max(top + 1, bottom)), h, w)
    if face is not None:
        fx1, fy1, fx2, fy2 = face
        fh = fy2 - fy1
        return _clip_box((fx1, fy2, fx2, fy2 + int(1.4 * fh)), h, w)
    # fallback: middle band
    return _clip_box((int(w * 0.2), int(h * 0.55), int(w * 0.8), int(h * 0.85)), h, w)


def hair_region(h: int, w: int, faces=None, persons=None) -> tuple[int, int, int, int]:
    """Band of the head above the top of the face (hats / hairstyles live here)."""
    face = union_box(faces or [], h, w)
    person = union_box(persons or [], h, w)
    if face is not None:
        fx1, fy1, fx2, fy2 = face
        fh = fy2 - fy1
        top = max(0, fy1 - int(0.9 * fh))
        width_pad = int(0.35 * (fx2 - fx1))
        return _clip_box((fx1 - width_pad, top, fx2 + width_pad, fy1 + int(0.25 * fh)), h, w)
    if person is not None:
        px1, py1, px2, py2 = person
        ph = py2 - py1
        return _clip_box((px1, max(0, py1 - int(0.15 * ph)), px2, py1 + int(0.2 * ph)), h, w)
    return _clip_box((int(w * 0.3), 0, int(w * 0.7), int(h * 0.25)), h, w)


def background_mask(h: int, w: int, faces=None, persons=None) -> np.ndarray:
    """Soft background mask: everything outside the person/face boxes."""
    mask = np.ones((h, w), dtype=np.float32)
    for box in (union_box(persons or [], h, w), union_box(faces or [], h, w)):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 0.0
    # feather the boundary so region edges do not dominate the metric
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6)
    return mask


def person_mask(h: int, w: int, faces=None, persons=None) -> np.ndarray:
    box = union_box(persons or [], h, w) or union_box(faces or [], h, w)
    mask = np.zeros((h, w), dtype=np.float32)
    if box is None:
        return mask
    x1, y1, x2, y2 = _clip_box(box, h, w)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def irregular_mask(h: int, w: int, faces=None, persons=None) -> np.ndarray:
    """Ellipse over the person region (an attacker choosing a non-rectangle)."""
    box = union_box(persons or [], h, w) or union_box(faces or [], h, w)
    mask = np.zeros((h, w), dtype=np.float32)
    if box is None:
        return mask
    x1, y1, x2, y2 = _clip_box(box, h, w)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2 * 1.15, (y2 - y1) / 2 * 1.15
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2 <= 1.0
    mask[inside] = 1.0
    return mask


MASK_KINDS = {
    "shirt": lambda h, w, faces, persons: _box_to_mask(h, w, shirt_region(h, w, faces, persons)),
    "hair": lambda h, w, faces, persons: _box_to_mask(h, w, hair_region(h, w, faces, persons)),
    "background": background_mask,
    "person": person_mask,
    "irregular": irregular_mask,
    "face": lambda h, w, faces, persons: _box_to_mask(h, w, union_box(faces or [], h, w)),
}


def _box_to_mask(h: int, w: int, box) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    if box is None:
        return mask
    x1, y1, x2, y2 = _clip_box(box, h, w)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def make_mask(kind: str, h: int, w: int, faces=None, persons=None) -> np.ndarray:
    fn = MASK_KINDS.get(kind)
    if fn is None:
        return np.ones((h, w), dtype=np.float32)
    return fn(h, w, faces, persons)


# ---------------------------------------------------------------------------
# low-level pixel measurements
# ---------------------------------------------------------------------------


def _region_stats(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    arr = img.astype(np.float32) / 255.0
    if mask is not None and mask.sum() > 0:
        arr = arr * mask[..., None]
        return arr
    return arr


def redness(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean red-dominance R - (G+B)/2 in [0,1] units, over the mask region."""
    arr = img.astype(np.float32) / 255.0
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() == 0:
            return 0.0
        r, g, b = arr[sel, 0], arr[sel, 1], arr[sel, 2]
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(np.mean(r - (g + b) / 2.0))


def warmth(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean (R - B) in [0,1] units — warm light raises it, cool light lowers it."""
    arr = img.astype(np.float32) / 255.0
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() == 0:
            return 0.0
        return float(np.mean(arr[sel, 0] - arr[sel, 2]))
    return float(np.mean(arr[..., 0] - arr[..., 2]))


def mean_saturation(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() == 0:
            return 0.0
        return float(hsv[sel].mean())
    return float(hsv.mean())


def edge_density(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() == 0:
            return 0.0
        return float(edges[sel].mean())
    return float(edges.mean())


def region_l2_change(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean per-pixel RGB L2 distance between a and b (0..sqrt(3)), over mask."""
    a = _resize_like(a, b)
    diff = (a.astype(np.float32) - b.astype(np.float32)) / 255.0
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() == 0:
            return 0.0
        return float(dist[sel].mean())
    return float(dist.mean())


def _resize_like(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[:2] == b.shape[:2]:
        return a
    from PIL import Image  # noqa: PLC0415

    return np.asarray(Image.fromarray(a.astype(np.uint8)).convert("RGB").resize((b.shape[1], b.shape[0]), Image.LANCZOS))


# ---------------------------------------------------------------------------
# task metrics
# ---------------------------------------------------------------------------

# Scaling constants (calibrated on real editor output):
#   redness: a full red-shirt swap moves mean redness by ~0.2-0.4
#   background L2: a beach replacement moves the background by ~0.2-0.5
#   head L2: a hat/hair edit moves the region by ~0.15-0.35
#   warmth: sunset lighting moves R-B by ~0.1-0.25
#   sketch: saturation drops ~0.2-0.5 and edge density rises ~0.1-0.3
SCALES = {
    "shirt_color": 0.25,
    "background": 0.30,
    "hat": 0.30,
    "lighting": 0.20,
    "sketch": 0.30,
    "hairstyle": 0.30,
}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def metric_shirt_color(img_in, img_out, faces, persons, mask=None) -> tuple[float, float]:
    region = mask if mask is not None else _box_to_mask(img_in.shape[0], img_in.shape[1], shirt_region(img_in.shape[0], img_in.shape[1], faces, persons))
    raw = redness(img_out, region) - redness(img_in, region)
    return raw, _clip01(raw / SCALES["shirt_color"])


def metric_background(img_in, img_out, faces, persons, mask=None) -> tuple[float, float]:
    region = mask if mask is not None else background_mask(img_in.shape[0], img_in.shape[1], faces, persons)
    raw = region_l2_change(img_in, img_out, region)
    return raw, _clip01(raw / SCALES["background"])


def metric_hat(img_in, img_out, faces, persons, mask=None) -> tuple[float, float]:
    region = mask if mask is not None else _box_to_mask(img_in.shape[0], img_in.shape[1], hair_region(img_in.shape[0], img_in.shape[1], faces, persons))
    raw = region_l2_change(img_in, img_out, region)
    return raw, _clip01(raw / SCALES["hat"])


def metric_lighting(img_in, img_out, faces, persons, mask=None) -> tuple[float, float]:
    raw = warmth(img_out, mask) - warmth(img_in, mask)
    return raw, _clip01(raw / SCALES["lighting"])


def metric_sketch(img_in, img_out, faces, persons, mask=None) -> tuple[float, float]:
    region = mask if mask is not None else person_mask(img_in.shape[0], img_in.shape[1], faces, persons)
    sat_drop = mean_saturation(img_in, region) - mean_saturation(img_out, region)
    edge_inc = edge_density(img_out, region) - edge_density(img_in, region)
    raw = 0.5 * sat_drop + 0.5 * edge_inc
    return raw, _clip01(raw / SCALES["sketch"])


def metric_hairstyle(img_in, img_out, faces, persons, mask=None) -> tuple[float, float]:
    region = mask if mask is not None else _box_to_mask(img_in.shape[0], img_in.shape[1], hair_region(img_in.shape[0], img_in.shape[1], faces, persons))
    raw = region_l2_change(img_in, img_out, region)
    return raw, _clip01(raw / SCALES["hairstyle"])


TASK_METRICS = {
    "shirt_color": metric_shirt_color,
    "background": metric_background,
    "hat": metric_hat,
    "lighting": metric_lighting,
    "sketch": metric_sketch,
    "hairstyle": metric_hairstyle,
}


def measure(task_id: str, img_in: np.ndarray, img_out: np.ndarray, faces, persons, mask: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Run the task-specific metric. Returns (raw, success) with success in [0,1]."""
    fn = TASK_METRICS.get(task_id)
    if fn is None:
        # unknown task -> structural change over the whole image (fallback)
        raw = region_l2_change(img_in, img_out)
        return raw, _clip01(raw / 0.3)
    return fn(img_in, img_out, faces, persons, mask=mask)

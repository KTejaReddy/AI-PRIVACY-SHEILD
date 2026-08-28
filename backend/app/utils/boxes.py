"""Shared bounding-box / crop helpers.

These were originally private helpers of the legacy face-protector stage;
the perception and robustness evaluation modules still need them, so they
live here as small, dependency-light utilities.
"""
from __future__ import annotations

import numpy as np


def expand_box(box: tuple[int, int, int, int], margin: float, h: int, w: int) -> tuple[int, int, int, int]:
    """Expand a box to a square crop centered on the box, clamped to the image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * margin
    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    return (
        max(0, nx1),
        max(0, ny1),
        min(w, nx1 + int(round(side))),
        min(h, ny1 + int(round(side))),
    )


def numpy_crops(img: np.ndarray, boxes: list[tuple[int, int, int, int]], size: int) -> list[np.ndarray]:
    """Square-resized RGB crops for every box."""
    import cv2  # noqa: PLC0415

    crops: list[np.ndarray] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((size, size, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        crops.append(crop)
    return crops


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2

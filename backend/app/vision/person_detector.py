"""Person detection using OpenCV's built-in HOG + linear SVM person detector.

The detector ships with OpenCV (``cv2.HOGDescriptor_getDefaultPeopleDetector``),
so no model download is required and nothing leaves the machine. Confidence is
the raw SVM decision weight (positive => "person-like"); it is a real measured
value used both for the perception report and as an adversarial objective
(suppression) in the black-box refinement phase.

Limitations (documented honestly): HOG is a full-body detector — upper-body
portraits, occluded people, or non-upright poses are often missed. That is a
property of the detector, not a bug; the system reports what it actually finds.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class PersonBox:
    """A detected person bounding box (image pixel coordinates)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float  # raw HOG/SVM decision weight

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(float(self.confidence), 4),
        }


class PersonDetector:
    """Thread-safe wrapper around the OpenCV HOG person detector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hog: cv2.HOGDescriptor | None = None
        self._error: str | None = None

    def _load(self) -> cv2.HOGDescriptor:
        with self._lock:
            if self._hog is not None:
                return self._hog
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self._hog = hog
                logger.info("Person detector loaded (OpenCV HOG + SVM).")
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                logger.warning("Person detector unavailable: %s", exc)
                raise
            return self._hog

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:  # noqa: BLE001
            return False

    @property
    def error_message(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    def detect(self, rgb_image: np.ndarray, confidence: float | None = None) -> list[PersonBox]:
        """Detect persons in an RGB uint8 image.

        Large images are downscaled for the HOG pass (its window is 64x128 and
        detection cost scales with resolution); boxes are scaled back to the
        original coordinate space.
        """
        hog = self._load()
        conf = confidence if confidence is not None else settings.PERSON_CONFIDENCE
        h, w = rgb_image.shape[:2]

        scale = 1.0
        work = rgb_image
        max_dim = settings.PERSON_DETECT_MAX_DIM
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            work = cv2.resize(
                rgb_image, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # The HOG window is 64x128 (+ padding); running it on anything smaller
        # crashes natively in OpenCV. There is nothing to detect in a sub-window
        # image, so report honestly instead of crashing.
        wh, ww = work.shape[:2]
        if wh < 128 or ww < 64:
            return []

        try:
            rects, weights = hog.detectMultiScale(
                work, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HOG person detection failed: %s", exc)
            return []

        persons: list[PersonBox] = []
        for (x, y, bw, bh), weight in zip(rects, weights):
            weight = float(weight)
            if weight < conf:
                continue
            x1 = int(round(x / scale))
            y1 = int(round(y / scale))
            x2 = int(round((x + bw) / scale))
            y2 = int(round((y + bh) / scale))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            persons.append(PersonBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=weight))

        if len(persons) > 1:
            rects_np = np.array([[p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1] for p in persons], dtype=np.float32)
            scores = np.array([p.confidence for p in persons], dtype=np.float32)
            keep = cv2.dnn.NMSBoxes(rects_np.tolist(), scores.tolist(), 0.0, 0.45)
            if keep is not None and len(keep) > 0:
                keep = keep.flatten().tolist()
                persons = [persons[i] for i in keep]

        return persons


_detector: PersonDetector | None = None
_detector_lock = threading.Lock()


def get_person_detector() -> PersonDetector:
    global _detector  # noqa: PLW0603
    with _detector_lock:
        if _detector is None:
            _detector = PersonDetector()
        return _detector

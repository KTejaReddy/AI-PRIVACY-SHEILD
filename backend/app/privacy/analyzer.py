"""Privacy analyzer: face detection + sensitive content analysis.

Also provides the sensitive-region treatment step: detected QR codes, sensitive
text, and (experimental) plates/documents are pixelated in the protected image.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import settings
from ..utils.imaging import ImageRecord
from ..vision.face_detector import DetectedFace, get_face_detector
from ..vision.mtcnn_face_detector import get_mtcnn_face_detector
from ..vision.neural_person_detector import get_neural_person_detector
from ..vision.person_detector import PersonBox, get_person_detector
from ..vision.sensitive import analyze_sensitive

logger = logging.getLogger(__name__)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1)


def merge_persons(hog_persons: list[PersonBox], neural_persons: list[PersonBox]) -> list[PersonBox]:
    """Union of HOG and neural detections, deduped by IoU (keep higher score)."""
    merged: list[PersonBox] = list(hog_persons)
    for np_ in neural_persons:
        dup = any(_iou(np_.box, p.box) > 0.5 for p in merged)
        if not dup:
            merged.append(np_)
    return merged


@dataclass
class AnalysisResult:
    faces: list[DetectedFace] = field(default_factory=list)
    persons: list[PersonBox] = field(default_factory=list)  # union (mask + suppression)
    persons_hog: list[PersonBox] = field(default_factory=list)
    persons_neural: list[PersonBox] = field(default_factory=list)
    neural_available: bool = False
    sensitive: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def person_count_hog(self) -> int:
        return len(self.persons_hog)

    @property
    def person_count_neural(self) -> int:
        return len(self.persons_neural)


def _pixelate(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int, block: int) -> None:
    """Pixelate a region in place (block-average)."""
    h, w = arr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = arr[y1:y2, x1:x2]
    small_h = max(1, region.shape[0] // block)
    small_w = max(1, region.shape[1] // block)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    arr[y1:y2, x1:x2] = up


class PrivacyAnalyzer:
    def __init__(self) -> None:
        self._detector = get_face_detector()
        self._mtcnn_detector = get_mtcnn_face_detector()
        self._person_detector = get_person_detector()
        self._neural_person_detector = get_neural_person_detector()

    # ------------------------------------------------------------------
    def analyze(self, record: ImageRecord) -> AnalysisResult:
        """Run face + person detection, sensitive-content analysis, metadata audit."""
        rgb = record.array

        # 1) faces
        faces: list[DetectedFace] = []
        try:
            faces = self._detector.detect(rgb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Face detection unavailable: %s", exc)

        # 2) persons — HOG (lightweight) + neural Faster R-CNN (modern), unioned
        persons_hog: list[PersonBox] = []
        try:
            persons_hog = self._person_detector.detect(rgb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Person detection unavailable: %s", exc)

        neural_available = False
        persons_neural: list[PersonBox] = []
        try:
            if self._neural_person_detector.available:
                neural_available = True
                persons_neural = self._neural_person_detector.detect(rgb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neural person detection unavailable: %s", exc)
        persons = merge_persons(persons_hog, persons_neural)

        # 3) sensitive content (QR, text/PII, experimental plates/documents)
        face_boxes = [f.box for f in faces]
        sensitive = analyze_sensitive(rgb, face_boxes)

        # 4) metadata audit
        metadata = {
            "source_had_exif": record.had_exif,
            "source_had_gps": record.had_gps,
            "source_had_xmp": record.had_xmp,
            "source_format": record.source_format,
            "source_size_bytes": record.source_size_bytes,
        }
        return AnalysisResult(
            faces=faces,
            persons=persons,
            persons_hog=persons_hog,
            persons_neural=persons_neural,
            neural_available=neural_available,
            sensitive=sensitive,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def treat_sensitive_regions(self, protected_rgb: np.ndarray, regions: list[dict]) -> int:
        """Pixelate sensitive regions in place. Returns number of regions treated."""
        treated = 0
        for region in regions:
            if not region.get("sensitive"):
                continue
            try:
                _pixelate(
                    protected_rgb,
                    int(region["x1"]),
                    int(region["y1"]),
                    int(region["x2"]),
                    int(region["y2"]),
                    block=max(8, settings.SENSITIVE_REGION_BLUR),
                )
                treated += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to treat region %s: %s", region.get("kind"), exc)
        return treated

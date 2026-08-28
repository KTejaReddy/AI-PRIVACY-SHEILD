"""Face detection using OpenCV's DNN SSD (res10) detector.

Model files are downloaded automatically on first use (public OpenCV
3rd-party release). If the download fails the detector reports itself
unavailable and the pipeline degrades gracefully.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

_PROTO = "deploy.prototxt"
_CAFFE = "res10_300x300_ssd_iter_140000.caffemodel"


@dataclass
class DetectedFace:
    """A detected face in image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

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


class FaceDetector:
    """Thread-safe wrapper around the OpenCV DNN face detector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._net: cv2.dnn.Net | None = None
        self._load_error: str | None = None

    # ------------------------------------------------------------------
    def _ensure_model_files(self) -> tuple[str, str]:
        proto_path = settings.FACE_DETECTOR_DIR / _PROTO
        caffe_path = settings.FACE_DETECTOR_DIR / _CAFFE
        if not proto_path.exists() or not caffe_path.exists():
            from ...scripts.download_models import ensure_face_detector  # type: ignore[import-not-found]

            ensure_face_detector(settings.FACE_DETECTOR_DIR)
        if not proto_path.exists() or not caffe_path.exists():
            raise RuntimeError(
                "Face detection model files are missing. Run: "
                "python scripts/download_models.py (see models/README.md)."
            )
        return str(proto_path), str(caffe_path)

    def _load(self) -> cv2.dnn.Net:
        with self._lock:
            if self._net is not None:
                return self._net
            try:
                proto, caffe = self._ensure_model_files()
                self._net = cv2.dnn.readNetFromCaffe(proto, caffe)
                logger.info("Face detector loaded (OpenCV DNN SSD).")
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.warning("Face detector unavailable: %s", exc)
                raise
            return self._net

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:  # noqa: BLE001
            return False

    @property
    def error_message(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------------
    def detect(self, rgb_image: np.ndarray, confidence: float | None = None) -> list[DetectedFace]:
        """Detect faces in an RGB uint8 image. Returns normalized boxes."""
        net = self._load()
        conf = confidence if confidence is not None else settings.FACE_CONFIDENCE
        h, w = rgb_image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            rgb_image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False
        )
        net.setInput(blob)
        detections = net.forward()

        faces: list[DetectedFace] = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < conf:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            faces.append(DetectedFace(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score))

        # Non-maximum suppression on the raw detections.
        if len(faces) > 1:
            rects = np.array([[f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1] for f in faces], dtype=np.float32)
            scores = np.array([f.confidence for f in faces], dtype=np.float32)
            keep = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), conf, 0.3)
            if keep is not None and len(keep) > 0:
                keep = keep.flatten().tolist()
                faces = [faces[i] for i in keep]

        return faces


_detector: FaceDetector | None = None
_detector_lock = threading.Lock()


def get_face_detector() -> FaceDetector:
    global _detector  # noqa: PLW0603
    with _detector_lock:
        if _detector is None:
            _detector = FaceDetector()
        return _detector

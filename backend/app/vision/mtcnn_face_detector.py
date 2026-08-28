"""MTCNN face detection (facenet-pytorch) — a second, independent detector family.

The OpenCV DNN SSD is the primary detector; MTCNN (P-Net/R-Net/O-Net cascade)
is a structurally different neural detector used for **evaluation** (the
perception test and benchmark) so that "one detector failing" is never
mistaken for "all detectors failing". Weights auto-download from the
``facenet-pytorch`` GitHub release to the torch cache on first use.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import settings
from .face_detector import DetectedFace

logger = logging.getLogger(__name__)


class MTCNNFaceDetector:
    def __init__(self, device: str | None = None) -> None:
        from ..config import resolve_device  # noqa: PLC0415

        self.device = resolve_device(device)
        self._lock = threading.Lock()
        self._mtcnn = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    def _load(self):
        with self._lock:
            if self._mtcnn is not None:
                return self._mtcnn
            try:
                from facenet_pytorch import MTCNN  # noqa: PLC0415

                self._mtcnn = MTCNN(
                    keep_all=True,
                    thresholds=[0.6, 0.7, 0.7],
                    device=self.device,
                )
                logger.info("MTCNN face detector loaded (%s).", self.device)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                logger.warning("MTCNN face detector unavailable: %s", exc)
                raise
            return self._mtcnn

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
    def pnet_objectness(self, x):
        """Differentiable P-Net face-objectness map for a [1,3,H,W] tensor in [0,1].

        Returns the raw face-probability map (``probs[:, 1]``) so the optimizer
        can push it down with real gradients. This is the **differentiable
        surrogate** for face detection; the full MTCNN cascade (with NMS) and
        the OpenCV SSD remain evaluation targets.
        """
        mtcnn = self._load()
        x_norm = x * 2.0 - 1.0  # match MTCNN's internal normalization
        _, probs = mtcnn.pnet(x_norm)
        return probs[:, 1]  # [1, H', W'] face probability

    def rnet_objectness(self, x):
        """Differentiable R-Net face-score for a [1,3,H,W] tensor in [0,1].

        R-Net is the cascade's second stage: its 0.7-threshold decision removes
        proposals before O-Net. Pushing its score below the threshold deletes
        the face entirely; it also responds more strongly to perturbation than
        O-Net (measured ~5x larger gradients). Returns ``probs[:, 1]``.
        """
        mtcnn = self._load()
        x_norm = x * 2.0 - 1.0  # match MTCNN's internal normalization
        _, probs = mtcnn.rnet(x_norm)
        return probs[:, 1]

    def onet_objectness(self, x):
        """Differentiable O-Net face-score for a [1,3,H,W] tensor in [0,1].

        O-Net is the final cascade stage and its face probability **is** the
        confidence the user sees from ``detect()``. Exposing it lets the
        optimizer push the final cascade confidence down directly (the P-Net
        map alone is an indirect lever). Returns ``probs[:, 1]``.
        """
        mtcnn = self._load()
        x_norm = x * 2.0 - 1.0  # match MTCNN's internal normalization
        _, probs, _ = mtcnn.onet(x_norm)
        return probs[:, 1]

    # ------------------------------------------------------------------
    def detect(self, rgb_image: np.ndarray, confidence: float | None = None) -> list[DetectedFace]:
        """Detect faces with MTCNN. Confidence = reported face probability."""
        mtcnn = self._load()
        conf = confidence if confidence is not None else settings.FACE_CONFIDENCE
        h, w = rgb_image.shape[:2]

        try:
            boxes, probs = mtcnn.detect(rgb_image, landmarks=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTCNN detection failed: %s", exc)
            return []
        if boxes is None or probs is None:
            return []

        faces: list[DetectedFace] = []
        for box, prob in zip(boxes, probs):
            prob = float(prob)
            if prob < conf:
                continue
            x1 = int(round(float(box[0])))
            y1 = int(round(float(box[1])))
            x2 = int(round(float(box[2])))
            y2 = int(round(float(box[3])))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            faces.append(DetectedFace(x1=x1, y1=y1, x2=x2, y2=y2, confidence=prob))
        return faces


_detector: MTCNNFaceDetector | None = None
_detector_lock = threading.Lock()


def get_mtcnn_face_detector() -> MTCNNFaceDetector:
    global _detector  # noqa: PLW0603
    with _detector_lock:
        if _detector is None:
            _detector = MTCNNFaceDetector()
        return _detector

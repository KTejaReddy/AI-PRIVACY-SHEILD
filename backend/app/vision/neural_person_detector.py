"""Neural person detection using torchvision Faster R-CNN (COCO).

This is the modern neural counterpart to the lightweight HOG detector. It runs
entirely locally — weights auto-download from the PyTorch model zoo on first
use (``FasterRCNN_ResNet50_FPN_Weights.DEFAULT``, ~160 MB, BSD-3 style
torchvision weights). It reports the actual COCO ``person`` class score.

Hardware: requires the torch backend (CPU works but is slow; CUDA is used when
available). If the weights cannot be downloaded or the model fails to load,
``available`` is False and callers must degrade gracefully (the HOG detector
and the mask construction already handle absence).
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import settings
from ..vision.person_detector import PersonBox

logger = logging.getLogger(__name__)


class NeuralPersonDetector:
    def __init__(self, device: str | None = None) -> None:
        from ..config import resolve_device  # noqa: PLC0415

        self.device = resolve_device(device)
        self._lock = threading.Lock()
        self._model = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch  # noqa: PLC0415
                from torchvision.models.detection import (  # noqa: PLC0415
                    FasterRCNN_ResNet50_FPN_Weights,
                    fasterrcnn_resnet50_fpn,
                )

                model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                self._model = model.eval().to(self.device)
                logger.info("Neural person detector loaded (Faster R-CNN ResNet50 FPN, %s).", self.device)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                logger.warning("Neural person detector unavailable: %s", exc)
                raise
            return self._model

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
        """Detect COCO 'person' boxes in an RGB uint8 image.

        Large images are downscaled for speed; boxes are scaled back to the
        original coordinate space. Returns ``PersonBox`` objects with the
        real class score as ``confidence``.
        """
        model = self._load()
        conf = confidence if confidence is not None else settings.NEURAL_PERSON_CONFIDENCE
        h, w = rgb_image.shape[:2]

        scale = 1.0
        work = rgb_image
        max_dim = settings.NEURAL_PERSON_MAX_DIM
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            import cv2  # noqa: PLC0415

            work = cv2.resize(
                rgb_image, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        import torch  # noqa: PLC0415

        tensor = torch.from_numpy(work.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = model(tensor)[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        persons: list[PersonBox] = []
        for box, score, label in zip(boxes, scores, labels):
            if label != 1:  # COCO class 1 == person
                continue
            score = float(score)
            if score < conf:
                continue
            x1 = int(round(box[0] / scale))
            y1 = int(round(box[1] / scale))
            x2 = int(round(box[2] / scale))
            y2 = int(round(box[3] / scale))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            persons.append(PersonBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score))

        return persons


_detector: NeuralPersonDetector | None = None
_detector_lock = threading.Lock()


def get_neural_person_detector() -> NeuralPersonDetector:
    global _detector  # noqa: PLW0603
    with _detector_lock:
        if _detector is None:
            _detector = NeuralPersonDetector()
        return _detector

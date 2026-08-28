"""AI perception test engine.

Measures *real* before/after values for the tested AI systems:

  * face detector confidence  — OpenCV DNN SSD **and** MTCNN (two independent
    detector families; one failing never implies all failing)
  * person detector weight     — HOG **and** neural Faster R-CNN (COCO person)
  * per-model embedding similarity (cosine) and L2 distance for face crops
    (FaceNet/ArcFace) and whole-image global features (MobileNetV3/ResNet50)

Every number is measured on the actual images; nothing is fabricated. When a
detector or model is unavailable, the affected rows are reported as "not
tested". This powers the "AI Perception Test" panel in the UI.

Metric definitions (kept strict):
  * detector "confidence"  = the detector's own reported score/probability
  * person "weight"        = HOG/SVM decision weight (real, can be > 1 or < 0)
  * embedding "similarity" = cosine similarity in [0, 1]
  * embedding "distance"   = L2 distance on L2-normalized vectors, [0, 2]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..utils.boxes import boxes_overlap, expand_box, numpy_crops
from ..config import settings
from ..models.face_models import FaceModelRegistry
from ..vision.face_detector import get_face_detector
from ..vision.mtcnn_face_detector import get_mtcnn_face_detector
from ..vision.neural_person_detector import get_neural_person_detector
from ..vision.person_detector import get_person_detector

logger = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    faces: dict = field(default_factory=dict)  # OpenCV SSD
    faces_mtcnn: dict = field(default_factory=dict)  # MTCNN
    persons: dict = field(default_factory=dict)  # HOG
    persons_neural: dict = field(default_factory=dict)  # Faster R-CNN
    embeddings: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "faces": self.faces,
            "faces_mtcnn": self.faces_mtcnn,
            "persons": self.persons,
            "persons_neural": self.persons_neural,
            "embeddings": self.embeddings,
            "note": self.note,
        }


def _change_pct(before: float, after: float) -> float | None:
    if before <= 1e-9:
        return None
    return round((after - before) / before * 100.0, 1)


def _box_stats(name: str, detected: int, before: float, after: float) -> dict:
    return {
        "name": name,
        "tested": True,
        "detected": detected,
        "before": round(before, 4),
        "after": round(after, 4),
        "change_pct": _change_pct(before, after),
    }


class PerceptionEngine:
    def __init__(self, registry: FaceModelRegistry, device: str) -> None:
        self.registry = registry
        self.device = device
        self._face_detector = get_face_detector()
        self._mtcnn_detector = get_mtcnn_face_detector()
        self._person_detector = get_person_detector()
        self._neural_person_detector = get_neural_person_detector()

    # ------------------------------------------------------------------
    def evaluate(
        self,
        original_rgb: np.ndarray,
        protected_rgb: np.ndarray,
        faces,
        persons=None,
        neural_persons=None,
    ) -> PerceptionResult:
        h, w = original_rgb.shape[:2]
        face_boxes = [f.box for f in faces]
        person_boxes = [p.box for p in persons or []]
        neural_boxes = [p.box for p in neural_persons or []]
        notes: list[str] = []

        # ---- face detectors (SSD + MTCNN, before/after) -------------------
        faces_block = self._measure_max_conf(
            "OpenCV SSD", self._face_detector, original_rgb, protected_rgb, face_boxes
        )
        if not faces_block["tested"]:
            reason = faces_block.get("note") or "detector unavailable"
            notes.append(f"Face confidence (SSD): not tested ({reason}).")
        faces_mtcnn = self._measure_max_conf(
            "MTCNN", self._mtcnn_detector, original_rgb, protected_rgb, face_boxes
        )
        if not faces_mtcnn["tested"]:
            reason = faces_mtcnn.get("note") or "detector unavailable"
            notes.append(f"Face confidence (MTCNN): not tested ({reason}).")

        # ---- person detectors (HOG + neural, before/after) ----------------
        persons_block = self._measure_max_weight(
            "HOG", self._person_detector, original_rgb, protected_rgb, person_boxes
        )
        if not persons_block["tested"]:
            notes.append("Person confidence (HOG): not tested (detector unavailable).")
        elif persons_block["detected"] == 0:
            notes.append(
                "Person (HOG): none detected in the original (full-body detector often "
                "misses portraits)."
            )
        persons_neural = self._measure_max_weight(
            "Faster R-CNN", self._neural_person_detector, original_rgb, protected_rgb, neural_boxes
        )
        if not persons_neural["tested"]:
            notes.append("Person confidence (neural): not tested (detector unavailable).")
        elif persons_neural["detected"] == 0:
            notes.append(
                "Person (neural): none detected in the original by Faster R-CNN."
            )

        # ---- per-model embedding similarity --------------------------------
        embeddings = self._measure_embeddings(original_rgb, protected_rgb, face_boxes, h, w)
        if not embeddings:
            notes.append("Embedding similarity: not tested (no verification models loaded).")

        note = " ".join(notes) if notes else (
            "Measured on the protected image against the tested detectors and models. "
            "Lower confidence / similarity means the tested AI is less reliable on the protected image."
        )
        return PerceptionResult(
            faces=faces_block,
            faces_mtcnn=faces_mtcnn,
            persons=persons_block,
            persons_neural=persons_neural,
            embeddings=embeddings,
            note=note,
        )

    # ------------------------------------------------------------------
    def _measure_max_conf(self, name, detector, orig, prot, boxes) -> dict:
        empty = {
            "name": name,
            "tested": False,
            "detected": len(boxes),
            "before": None,
            "after": None,
            "change_pct": None,
        }
        if not boxes:
            return empty | {"detected": 0, "note": "no face detected in the original"}
        try:
            if not getattr(detector, "available", True):
                return empty
            before = self._max_overlap(detector, orig, boxes)
            after = self._max_overlap(detector, prot, boxes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s measurement failed: %s", name, exc)
            return empty
        if before is None:
            return empty
        return _box_stats(name, len(boxes), float(np.mean(before)), float(np.mean(after)))

    def _max_overlap(self, detector, img, boxes) -> list[float] | None:
        dets = detector.detect(img, confidence=0.05)
        if not dets:
            return [0.0] * len(boxes)
        out: list[float] = []
        for box in boxes:
            best = max((float(d.confidence) for d in dets if boxes_overlap(d.box, box)), default=0.0)
            out.append(best)
        return out

    def _measure_max_weight(self, name, detector, orig, prot, boxes) -> dict:
        empty = {
            "name": name,
            "tested": False,
            "detected": len(boxes),
            "before": None,
            "after": None,
            "change_pct": None,
        }
        try:
            if not getattr(detector, "available", True):
                return empty
            before = self._max_weight(detector, orig, boxes)
            after = self._max_weight(detector, prot, boxes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s measurement failed: %s", name, exc)
            return empty
        if before is None:
            return empty
        return _box_stats(name, len(boxes), before, after)

    def _max_weight(self, detector, img, boxes) -> float | None:
        try:
            dets = detector.detect(img, confidence=-1.0)
        except Exception:  # noqa: BLE001
            return None
        if not boxes:
            return max((float(d.confidence) for d in dets), default=0.0)
        best = 0.0
        for d in dets:
            if any(boxes_overlap(d.box, b) for b in boxes):
                best = max(best, float(d.confidence))
        return best

    # ------------------------------------------------------------------
    def _measure_embeddings(
        self,
        orig: np.ndarray,
        prot: np.ndarray,
        face_boxes: list[tuple[int, int, int, int]],
        h: int,
        w: int,
    ) -> dict:
        models = self.registry.verification_models
        if not models:
            return {}
        face_crops = [expand_box(f, settings.FACE_CROP_MARGIN, h, w) for f in face_boxes]

        def _boxes_for(model) -> list[tuple[int, int, int, int]]:
            if getattr(model, "mode", "face") == "global":
                return [(0, 0, w, h)]
            return face_crops

        result: dict[str, dict] = {}
        for model in models:
            boxes = _boxes_for(model)
            if not boxes:
                continue
            try:
                orig_embs = model.embed_crops(numpy_crops(orig, boxes, model.input_size), self.device)
                prot_embs = model.embed_crops(numpy_crops(prot, boxes, model.input_size), self.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding similarity failed for %s: %s", model.info.id, exc)
                continue
            sims = [
                float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                for a, b in zip(orig_embs, prot_embs)
            ]
            dists = [float(np.linalg.norm(a - b)) for a, b in zip(orig_embs, prot_embs)]
            mean_sim = float(np.mean(sims))
            result[model.info.id] = {
                "display_name": model.info.display_name,
                "kind": "face" if getattr(model, "mode", "face") == "face" else "vision",
                "before": 1.0,  # self-similarity baseline of the original
                "after": round(mean_sim, 4),
                "mean_distance": round(float(np.mean(dists)), 4),
                "change_pct": _change_pct(1.0, mean_sim),
                "tested": True,
            }
        return result

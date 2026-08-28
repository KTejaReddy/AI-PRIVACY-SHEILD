"""Verification & transformation-robustness testing.

The protected image is evaluated against **all** configured verification models
(including any non-differentiable ones). For every real transformation
(JPEG compression, resize, crop, brightness, contrast, re-encoding) the
transformed protected image is embedded and compared to the original identity's
embedding. Verdicts come from the actual measured L2 distances — nothing is
hard-coded.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..utils.boxes import expand_box
from ..config import settings
from ..models.face_models import FaceModelRegistry

logger = logging.getLogger(__name__)


def _jpeg_bytes(arr: np.ndarray, quality: int) -> np.ndarray:
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _png_reencode(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img2 = Image.fromarray(np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8), mode="RGB")
    buf2 = io.BytesIO()
    img2.save(buf2, format="JPEG", quality=85)
    return np.asarray(Image.open(buf2).convert("RGB"), dtype=np.uint8)


def _apply_transform(name: str, arr: np.ndarray) -> np.ndarray:
    """Apply one of the configured real transformations."""
    h, w = arr.shape[:2]
    if name == "jpeg_compression":
        return _jpeg_bytes(arr, 70)
    if name == "resize":
        small = cv2.resize(arr, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if name == "crop":
        # center crop to 90% then back to full size (face stays in frame)
        cw, ch = int(w * 0.9), int(h * 0.9)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        cropped = arr[y0 : y0 + ch, x0 : x0 + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    if name == "brightness":
        return np.clip(arr.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    if name == "contrast":
        mean = arr.mean(axis=(0, 1), keepdims=True)
        return np.clip((arr.astype(np.float32) - mean) * 1.15 + mean, 0, 255).astype(np.uint8)
    if name == "reencode":
        return _png_reencode(arr)
    raise ValueError(f"Unknown transform: {name}")


@dataclass
class RobustnessResult:
    overall: str
    transforms: dict
    base_distances: dict[str, float]
    model_summary: dict
    thresholds: dict
    note: str

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "transforms": self.transforms,
            "base_distances": self.base_distances,
            "model_summary": self.model_summary,
            "thresholds": self.thresholds,
            "note": self.note,
        }


class RobustnessTester:
    def __init__(self, registry: FaceModelRegistry, device: str) -> None:
        self.registry = registry
        self.device = device

    # ------------------------------------------------------------------
    def evaluate(
        self,
        original_rgb: np.ndarray,
        protected_rgb: np.ndarray,
        faces,
    ) -> RobustnessResult | None:
        models = self.registry.verification_models
        if not models or not faces:
            return None

        h, w = original_rgb.shape[:2]
        face_crops = [expand_box(f.box, settings.FACE_CROP_MARGIN, h, w) for f in faces]

        def _boxes_for(model) -> list[tuple[int, int, int, int]]:
            if getattr(model, "mode", "face") == "global":
                return [(0, 0, w, h)]
            return face_crops

        def _embeddings(img: np.ndarray, model) -> list[np.ndarray]:
            crops = []
            for box in _boxes_for(model):
                x1, y1, x2, y2 = box
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((model.input_size, model.input_size, 3), dtype=np.uint8)
                crop = cv2.resize(crop, (model.input_size, model.input_size), interpolation=cv2.INTER_LINEAR)
                crops.append(crop)
            return model.embed_crops(crops, self.device)

        orig_embs: dict[str, list[np.ndarray]] = {}
        prot_embs: dict[str, list[np.ndarray]] = {}
        for model in models:
            orig_embs[model.info.id] = _embeddings(original_rgb, model)
            prot_embs[model.info.id] = _embeddings(protected_rgb, model)

        def _mean_dist(orig: list[np.ndarray], prot: list[np.ndarray]) -> float:
            if not orig:
                return 0.0
            return float(np.mean([np.linalg.norm(a - b) for a, b in zip(orig, prot)]))

        # ---- base disruption (no transform) ------------------------------
        base = {
            model.info.id: round(_mean_dist(orig_embs[model.info.id], prot_embs[model.info.id]), 4)
            for model in models
        }

        # ---- transformed protected images ---------------------------------
        transforms: dict[str, dict] = {}
        for tname in settings.ROBUSTNESS_TRANSFORMS:
            try:
                variant = _apply_transform(tname, protected_rgb)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transform %s failed: %s", tname, exc)
                transforms[tname] = {"verdict": "FAIL", "per_model": {}, "mean": 0.0, "error": str(exc)}
                continue
            per_model: dict[str, float] = {}
            for model in models:
                dist = _mean_dist(orig_embs[model.info.id], _embeddings(variant, model))
                per_model[model.info.id] = round(dist, 4)
            mean = float(np.mean(list(per_model.values())))
            verdict = self._verdict(per_model, mean)
            transforms[tname] = {"verdict": verdict, "per_model": per_model, "mean": round(mean, 4)}

        # ---- per-model summary --------------------------------------------
        model_summary: dict[str, dict] = {}
        for model in models:
            dists = [transforms[t]["per_model"].get(model.info.id, 0.0) for t in settings.ROBUSTNESS_TRANSFORMS]
            mean = float(np.mean(dists)) if dists else 0.0
            model_summary[model.info.id] = {
                "mean_distance": round(mean, 4),
                "base_distance": base.get(model.info.id, 0.0),
                "verdict": self._verdict({model.info.id: mean}, mean),
            }

        # ---- overall -------------------------------------------------------
        verdicts = [t["verdict"] for t in transforms.values()]
        if all(v == "PASS" for v in verdicts):
            overall = "PASS"
        elif any(v == "FAIL" for v in verdicts):
            overall = "FAIL"
        else:
            overall = "PARTIAL"

        thresholds = {
            "pass": settings.DISRUPT_PASS,
            "partial": settings.DISRUPT_PARTIAL,
            "unit": "L2 distance in normalized embedding space",
        }
        note = (
            "Verdicts are based on measured embedding distances. PASS means every tested model "
            "kept a distance above the disruption threshold under every tested transformation."
        )
        return RobustnessResult(
            overall=overall,
            transforms=transforms,
            base_distances=base,
            model_summary=model_summary,
            thresholds=thresholds,
            note=note,
        )

    @staticmethod
    def _verdict(per_model: dict[str, float], mean: float) -> str:
        if not per_model:
            return "FAIL"
        worst = min(per_model.values())
        if worst >= settings.DISRUPT_PASS:
            return "PASS"
        if worst >= settings.DISRUPT_PARTIAL:
            return "PARTIAL"
        return "FAIL"


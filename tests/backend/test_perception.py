"""AI perception test engine + person detector tests."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.evaluation.perception import PerceptionEngine
from app.models.face_models import get_registry
from app.vision.face_detector import get_face_detector
from app.vision.person_detector import PersonDetector


@pytest.fixture(scope="module")
def registry():
    reg = get_registry()
    reg.load_all()
    return reg


def _load(arr_bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(arr_bytes)).convert("RGB"))


def test_person_detector_loads_and_returns_boxes():
    det = PersonDetector()
    assert det.available
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    persons = det.detect(img)
    # never crashes; returns a list of valid boxes
    assert isinstance(persons, list)
    for p in persons:
        d = p.as_dict()
        assert d["x2"] > d["x1"] and d["y2"] > d["y1"]
        assert np.isfinite(d["confidence"])


def test_perception_face_confidence_measured(face_jpg_bytes, registry, fast_settings):
    arr = _load(face_jpg_bytes)
    faces = get_face_detector().detect(arr)
    if not faces:
        pytest.skip("Face fixture did not yield a detection.")

    engine = PerceptionEngine(registry, registry.device)
    # protected == original: confidence should be unchanged
    result = engine.evaluate(arr, arr, faces)
    assert result.faces["tested"] is True
    assert result.faces["detected"] == len(faces)
    assert result.faces["before"] is not None
    assert result.faces["after"] == pytest.approx(result.faces["before"], abs=0.001)
    assert np.isfinite(result.faces["before"])
    # embeddings measured (self-similarity = 1.0 when nothing changed)
    assert result.embeddings
    for emb in result.embeddings.values():
        assert emb["before"] == 1.0
        assert emb["after"] == pytest.approx(1.0, abs=0.02)
        assert emb["tested"] is True


def test_perception_embedding_similarity_drops_with_perturbation(face_jpg_bytes, registry, fast_settings):
    arr = _load(face_jpg_bytes)
    faces = get_face_detector().detect(arr)
    if not faces:
        pytest.skip("Face fixture did not yield a detection.")

    class FakeModel:
        info = type("Info", (), {"id": "fake_emb", "display_name": "Fake Embedder"})
        input_size = 160

        def embed_crops(self, crops, device):
            # sign-binarized 8x8 image hash: any pixel crossing mid-gray
            # flips a component, so even a mild brightness shift rotates the
            # embedding substantially
            import cv2  # noqa: PLC0415

            out = []
            for crop in crops:
                small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
                vec = np.where(small.astype(np.float32) > 127.5, 1.0, -1.0).flatten()
                out.append(vec / (np.linalg.norm(vec) + 1e-12))
            return out

    class FakeReg:
        verification_models = [FakeModel()]
        device = "cpu"

    engine = PerceptionEngine(FakeReg(), "cpu")
    protected = np.clip(arr.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    result = engine.evaluate(arr, protected, faces)
    emb = result.embeddings["fake_emb"]
    assert emb["before"] == 1.0
    assert emb["after"] < emb["before"]
    assert emb["change_pct"] is not None and emb["change_pct"] < 0
    assert emb["mean_distance"] > 0


def test_perception_no_face_reports_honestly(gradient_png_bytes, registry):
    arr = _load(gradient_png_bytes)
    faces = get_face_detector().detect(arr)
    assert len(faces) == 0

    engine = PerceptionEngine(registry, registry.device)
    result = engine.evaluate(arr, arr, faces)
    assert result.faces["tested"] is False
    assert result.faces["detected"] == 0
    # no face boxes -> no *face-embedding* measurements, but the global-mode
    # vision encoders still measure whole-image feature similarity
    assert result.embeddings
    for emb in result.embeddings.values():
        assert emb["kind"] == "vision"

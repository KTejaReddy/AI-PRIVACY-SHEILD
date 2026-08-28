"""End-to-end processing pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from app.cleanup.manager import get_store
from app.config import settings
from app.processing.pipeline import run_pipeline_locked

from fastapi.testclient import TestClient
from app.main import app


def _fake_editing_protection(image_rgb: np.ndarray, device: str, **kwargs):
    """Deterministic fake of the unified multi-family protection stage.

    Applies a tiny perturbation and reports applied=True so the pipeline
    wiring (protection block, stage events, quality, cleanup) is exercised
    without running the SD1.5 surrogate.
    """
    from app.editing.protector import EditingProtectionResult  # noqa: PLC0415

    perturbed = np.clip(image_rgb.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    return EditingProtectionResult(
        applied=True,
        iterations=4,
        epsilon=0.01,
        protected=perturbed,
        families=["diffusion_editing", "identity_reference", "vision_encoder"],
        note="Fake unified protection applied (test).",
    )


def _run(store, raw, extension, monkeypatch):
    sid = store.create()
    store.save_upload(sid, raw, extension=extension)
    events = []
    # Route the protection stage through the fake so the heavy SD1.5
    # surrogate is never loaded in the unit suite.
    monkeypatch.setattr(
        "app.processing.pipeline.apply_editing_protection", _fake_editing_protection
    )
    monkeypatch.setattr(settings, "EDITING_ENABLED", True)
    monkeypatch.setattr(settings, "EDITING_SURROGATE_ENABLED", True)

    def emit(ev):
        events.append(ev)

    result = run_pipeline_locked(sid, emit)
    return sid, events, result


def test_pipeline_face_image(face_jpg_bytes, fast_settings, monkeypatch, cleanup_sessions_between_tests):
    store = get_store()
    sid, events, result = _run(store, face_jpg_bytes, ".jpg", monkeypatch)

    assert result["face_count"] == 1
    assert result["protection_applied"] is True
    assert result["protection"]["applied"] is True
    assert result["protection"]["iterations"] >= 1
    assert result["quality"]["ssim"] > 0.7
    assert result["robustness"] is not None
    assert result["robustness"]["overall"] in ("PASS", "PARTIAL", "FAIL")
    assert result["metadata"]["output_format"] == "png"
    # new-spec fields: persons, perception, VLM status, provenance, timing
    assert "person_count" in result and result["person_count"] >= 0
    assert "persons" in result and isinstance(result["persons"], list)
    assert result["perception"] is not None
    assert result["perception"]["faces"]["tested"] is True
    assert result["perception"]["faces"]["detected"] == 1
    assert result["perception"]["embeddings"]
    assert result["vlm"]["enabled"] is False
    assert result["provenance"]["applied"] is True  # c2pa-python installed
    assert result["processing_time_ms"] > 0
    assert any(e["type"] == "faces" and "person_count" in e for e in events)
    assert result["protected_data_url"].startswith("data:image/png;base64,")
    assert result["original_data_url"].startswith("data:image/png;base64,")
    assert result["hardware"]["device"] in ("cuda", "cpu")

    # the pipeline emits real stage events including the result
    types = [e["type"] for e in events]
    assert "stage" in types and "stage_done" in types and "result" in types and "done" in types
    assert any(e["type"] == "faces" and e["count"] == 1 for e in events)
    # unified protection: a single "protect" stage, no stacked face stage
    assert any(e.get("stage") == "protect" for e in events if e["type"] == "stage_done")

    # temp files are gone
    assert not store.exists(sid)


def test_pipeline_no_face_image(gradient_png_bytes, fast_settings, monkeypatch, cleanup_sessions_between_tests):
    store = get_store()
    sid, events, result = _run(store, gradient_png_bytes, ".png", monkeypatch)

    assert result["face_count"] == 0
    assert result["protection_applied"] is True  # image-wide protection still applies
    assert result["protection"]["applied"] is True
    assert "No face detected" in result["faces_message"]
    assert result["robustness"] is None
    assert result["perception"] is not None
    assert result["perception"]["faces"]["tested"] is False
    assert result["vlm"]["enabled"] is False
    # pipeline still produces a sanitized output
    assert result["protected_data_url"].startswith("data:image/png;base64,")
    assert not store.exists(sid)


def test_api_sse_stream_processes_no_face_image(gradient_png_bytes, fast_settings, cleanup_sessions_between_tests):
    """Full HTTP path: upload -> SSE stream -> result, no optimization needed."""
    client = TestClient(app)
    upload = client.post("/api/upload", files={"file": ("g.png", gradient_png_bytes, "image/png")})
    assert upload.status_code == 200
    sid = upload.json()["session_id"]

    stream = client.get(f"/api/process/{sid}")
    assert stream.status_code == 200
    body = stream.text
    assert '"type": "result"' in body
    assert '"face_count": 0' in body
    assert '"type": "done"' in body

    # backend cleaned up its own temp dir
    assert not get_store().exists(sid)

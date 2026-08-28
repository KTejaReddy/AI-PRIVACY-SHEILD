"""C2PA provenance layer + profile config wiring."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.metadata.provenance import (
    ProvenanceResult,
    add_c2pa_manifest,
    c2pa_available,
)


def _png_bytes(size: int = 24) -> bytes:
    arr = np.full((size, size, 3), 140, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_c2pa_available_after_install():
    assert c2pa_available() is True


def test_manifest_embeds_and_reads_back(monkeypatch, tmp_path):
    if not c2pa_available():
        pytest.skip("c2pa-python not installed")
    monkeypatch.setattr(settings, "C2PA_ENABLED", True)
    monkeypatch.setattr(settings, "C2PA_KEY_DIR", tmp_path)
    png = _png_bytes()
    res = add_c2pa_manifest(png, width=24, height=24, output_format="png")
    assert isinstance(res, ProvenanceResult)
    assert res.applied is True
    assert res.signed_bytes.startswith(b"\x89PNG")
    assert len(res.signed_bytes) > len(png)  # manifest embedded

    import c2pa  # noqa: PLC0415

    reader = c2pa.Reader("image/png", io.BytesIO(res.signed_bytes))
    doc = json.loads(reader.json())
    assert "manifests" in doc
    manifest = next(iter(doc["manifests"].values()))
    assert manifest["claim_generator_info"][0]["name"] == "AI Privacy Shield"
    assert manifest["signature_info"]  # cryptographically signed


def test_manifest_disabled_returns_input_untouched(monkeypatch):
    monkeypatch.setattr(settings, "C2PA_ENABLED", False)
    png = _png_bytes()
    res = add_c2pa_manifest(png, width=24, height=24, output_format="png")
    assert res.enabled is False
    assert res.applied is False
    assert res.signed_bytes == png
    assert "disabled" in res.note.lower()


def test_manifest_failure_never_drops_image(monkeypatch):
    """Even a signing failure must return the original bytes (honest status)."""
    if not c2pa_available():
        pytest.skip("c2pa-python not installed")
    monkeypatch.setattr(settings, "C2PA_ENABLED", True)

    import app.metadata.provenance as prov  # noqa: PLC0415

    # No keypair available -> the manifest cannot be signed, but the image
    # must come back untouched with an honest status.
    monkeypatch.setattr(prov, "c2pa_available", lambda: True)
    monkeypatch.setattr(prov, "_load_external_key", lambda: None)
    monkeypatch.setattr(prov, "_self_signed_keypair", lambda: None)
    png = _png_bytes()
    res = add_c2pa_manifest(png, width=24, height=24, output_format="png")
    assert res.applied is False
    assert res.signed_bytes == png
    assert "could not be created" in res.note


def test_profile_flags_wired_from_yaml():
    # conftest forces the research profile; the research.yaml flags must be
    # reflected in Settings (benchmark on, red-team rounds configured).
    assert settings.PROFILE == "research"
    assert settings.EDITING_BENCHMARK_ENABLED is True
    assert settings.RED_TEAM_ROUNDS >= 0

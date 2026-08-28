"""Shared fixtures for the backend test suite."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

PROJECT_ROOT = BACKEND_DIR.parent
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"

# The test suite exercises the research side of the app (editing benchmark
# included), so force the research profile and benchmark on before app.config
# is imported (settings are read at import time).
os.environ.setdefault("AIPS_PROFILE", "research")
os.environ["AIPS_EDITING_BENCHMARK_ENABLED"] = "1"

from app.config import settings  # noqa: E402


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_png_bytes() -> bytes:
    """A tiny but valid 4x4 PNG."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., 0] = 200
    arr[..., 1] = 120
    arr[..., 2] = 40
    return _png_bytes(arr)


@pytest.fixture
def gradient_png_bytes() -> bytes:
    """A 96x64 gradient image — contains no faces."""
    x = np.linspace(0, 255, 96, dtype=np.uint8)
    arr = np.stack([x, 255 - x, np.full(96, 128, dtype=np.uint8)], axis=-1)
    arr = np.repeat(arr[np.newaxis, :, :], 64, axis=0).reshape(64, 96, 3)
    return _png_bytes(arr)


@pytest.fixture
def text_file_bytes() -> bytes:
    """A text file disguised as a PNG (extension lies; magic bytes don't)."""
    return b"this is definitely not a png image, just plain text" * 20


@pytest.fixture
def face_jpg_bytes() -> bytes:
    path = FIXTURES / "einstein.jpg"
    if not path.exists():
        pytest.skip("Face fixture missing (tests/fixtures/einstein.jpg).")
    return path.read_bytes()


@pytest.fixture
def fast_settings(monkeypatch):
    """Keep the ML-heavy tests fast."""
    monkeypatch.setattr(settings, "OPT_ITERATIONS_GPU", 8)
    monkeypatch.setattr(settings, "OPT_ITERATIONS_CPU", 8)
    monkeypatch.setattr(settings, "OPT_ROBUSTNESS_INTERVAL", 2)
    monkeypatch.setattr(settings, "REFINE_MAX_ITERS_GPU", 8)
    monkeypatch.setattr(settings, "REFINE_MAX_ITERS_CPU", 6)
    monkeypatch.setattr(settings, "REFINE_DIRECTIONS_GPU", 4)
    monkeypatch.setattr(settings, "REFINE_DIRECTIONS_CPU", 3)
    monkeypatch.setattr(settings, "DET_DIRECTIONS_GPU", 3)
    monkeypatch.setattr(settings, "DET_DIRECTIONS_CPU", 2)
    monkeypatch.setattr(settings, "DET_BLOB_DIRECTIONS_GPU", 3)
    monkeypatch.setattr(settings, "DET_BLOB_DIRECTIONS_CPU", 2)
    monkeypatch.setattr(settings, "DET_GRAD_INTERVAL_GPU", 2)
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    # The AI-editing stages run real diffusion models (minutes per image);
    # they are covered by their own tests and the benchmark script, so the
    # pipeline tests skip them for speed.
    monkeypatch.setattr(settings, "EDITING_ENABLED", False)
    monkeypatch.setattr(settings, "EDITING_BENCHMARK_ENABLED", False)
    monkeypatch.setattr(settings, "EDITING_ROBUSTNESS_ENABLED", False)
    return settings


def _torch_checkpoint_dir() -> Path:
    import os  # noqa: PLC0415

    cache = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    return cache / "checkpoints"


@pytest.fixture
def face_models_available() -> bool:
    """True when the FaceNet weights are present (downloaded once)."""
    try:
        checkpoints = _torch_checkpoint_dir()
        names = [p.name for p in checkpoints.iterdir()]
    except OSError:
        return False
    return any("vggface2" in n for n in names) and any("casia" in n for n in names)


@pytest.fixture
def cleanup_sessions_between_tests():
    """Remove leftover session dirs created by the tests."""
    from app.cleanup.manager import get_store  # noqa: PLC0415

    store = get_store()
    yield
    store.cleanup_stale(max_age_seconds=0)

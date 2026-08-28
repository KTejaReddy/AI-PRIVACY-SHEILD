"""Image-quality metrics behave correctly on known inputs."""
from __future__ import annotations

import numpy as np

from app.quality.metrics import _PSNR_CAP_DB, compute_quality, psnr, ssim


def test_psnr_identical_images_is_capped_finite():
    """Identical images: infinite PSNR is reported as a finite cap (JSON-safe)."""
    arr = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert psnr(arr, arr) == _PSNR_CAP_DB
    assert np.isfinite(psnr(arr, arr))


def test_psnr_known_value():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 10, dtype=np.uint8)
    expected = 10 * np.log10(255.0**2 / 100.0)
    assert abs(psnr(a, b) - expected) < 1e-6


def test_ssim_identical_images_is_one():
    arr = np.random.default_rng(0).integers(0, 256, (32, 32, 3)).astype(np.uint8)
    assert abs(ssim(arr, arr) - 1.0) < 1e-6


def test_ssim_drops_with_corruption():
    """Structurally destroying half the pixels must tank SSIM."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (48, 48, 3)).astype(np.uint8)
    b = a.copy()
    mask = rng.random((48, 48, 3)) < 0.5
    b[mask] = rng.integers(0, 256, int(mask.sum())).astype(np.uint8)
    assert ssim(a, b) < 0.5


def test_compute_quality_reports_perturbation_norms():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.zeros((16, 16, 3), dtype=np.uint8)
    b[0, 0] = 10
    q = compute_quality(a, b)
    assert q.perturbation_linf == 10.0
    assert q.mse > 0
    assert q.psnr_db > 0

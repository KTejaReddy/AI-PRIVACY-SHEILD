"""Image quality metrics — every number reported to the user is computed here.

- PSNR  (peak signal-to-noise ratio, dB)
- SSIM  (structural similarity, 0..1) implemented in pure numpy with a Gaussian window
- MSE / MAE
- perturbation L2 and L-inf norms
- LPIPS (deep perceptual similarity) when the optional ``lpips`` package is installed
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """2D Gaussian window normalized to sum 1."""
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def _correlate2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D correlation (same size output) via FFT, pure numpy."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode="reflect")
    kp = np.zeros_like(xp)
    kp[:kh, :kw] = kernel
    fft_x = np.fft.rfft2(xp)
    fft_k = np.fft.rfft2(np.fft.ifftshift(kp))
    y = np.fft.irfft2(fft_x * fft_k, s=xp.shape)
    return y[ph : ph + x.shape[0], pw : pw + x.shape[1]]


_PSNR_CAP_DB = 99.0  # reported when the images are (effectively) identical


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 RGB images.

    Mathematically identical images have infinite PSNR; since ``Infinity`` is
    not valid JSON, we report a finite cap (``_PSNR_CAP_DB``) instead. The
    metric is computed from the actual pixels either way.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse < _EPS:
        return _PSNR_CAP_DB
    return min(float(10.0 * np.log10(255.0**2 / mse)), _PSNR_CAP_DB)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM in [0, 1] on the luminance channel (classic formulation)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("SSIM inputs must have the same shape")
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        b = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]

    window = _gaussian_window()

    mu1 = _correlate2d(a, window)
    mu2 = _correlate2d(b, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _correlate2d(a * a, window) - mu1_sq
    sigma2_sq = _correlate2d(b * b, window) - mu2_sq
    sigma12 = _correlate2d(a * b, window) - mu1_mu2

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + _EPS
    )
    return float(np.mean(ssim_map))


_LPIPS_MODEL = None


def _get_lpips(device: str = "cpu"):
    global _LPIPS_MODEL  # noqa: PLW0603
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips  # type: ignore[import-not-found]

        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        return _LPIPS_MODEL
    except Exception as exc:  # noqa: BLE001
        logger.info("LPIPS not available: %s", exc)
        return None


def lpips_distance(a: np.ndarray, b: np.ndarray, device: str = "cpu") -> float | None:
    """LPIPS distance between two uint8 RGB images, or None if unavailable."""
    model = _get_lpips(device)
    if model is None:
        return None
    try:
        import torch  # noqa: PLC0415

        ta = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        tb = torch.from_numpy(b.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        ta = ta.to(device)
        tb = tb.to(device)
        with torch.no_grad():
            return float(model(ta, tb).item())
    except Exception as exc:  # noqa: BLE001
        logger.info("LPIPS computation failed: %s", exc)
        return None


@dataclass
class QualityMetrics:
    psnr_db: float
    ssim: float
    mse: float
    mae: float
    perturbation_l2: float
    perturbation_linf: float
    lpips: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "psnr_db": self.psnr_db,
            "ssim": self.ssim,
            "mse": self.mse,
            "mae": self.mae,
            "perturbation_l2": self.perturbation_l2,
            "perturbation_linf": self.perturbation_linf,
            "lpips": self.lpips,
            **self.extras,
        }


def compute_quality(original: np.ndarray, protected: np.ndarray, device: str = "cpu") -> QualityMetrics:
    """Compute all quality metrics between two uint8 RGB arrays."""
    orig_f = original.astype(np.float64)
    prot_f = protected.astype(np.float64)
    diff = prot_f - orig_f

    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))

    try:
        ssim_val = ssim(original, protected)
    except Exception as exc:  # noqa: BLE001 - never let a metric crash the pipeline
        logger.warning("SSIM computation failed: %s", exc)
        ssim_val = float("nan")

    return QualityMetrics(
        psnr_db=psnr(original, protected),
        ssim=ssim_val,
        mse=mse,
        mae=mae,
        perturbation_l2=float(np.sqrt(np.sum(diff**2))),
        perturbation_linf=float(np.max(np.abs(diff))),
        lpips=lpips_distance(original, protected, device=device),
    )

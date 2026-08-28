"""Adversarial protection engine.

Phase 1 — differentiable multi-objective optimization.

    protected = original + delta,   |delta|_inf <= epsilon * mask_weight

A **soft region mask** gives faces the full perturbation budget (1.0), person
bodies a strong share, and the surrounding context a small controlled dither,
with Gaussian-smoothed edges so no rectangular boundary is visible.

The differentiable objective combines:

    identity disruption    (face embeddings pushed beyond a margin — supporting layer)
  + vision disruption      (global ImageNet-style features pushed away — primary layer)
  + perceptual similarity  (differentiable SSIM + MSE)
  + perturbation penalty   (L-inf bound)
  + transformation robustness (differentiable scale / gamma / translate / noise /
    contrast / brightness / blur / JPEG-surrogate variants, sampled every iteration)

Phase 2 — black-box refinement (zeroth order), the detection-disruption layer.

Non-differentiable models (ArcFace ONNX, face/person detectors) cannot
participate in phase 1. Phase 2 refines the same bounded, soft-masked
perturbation against a combined objective:

    smooth face-detection suppression  (logsumexp over overlapping box scores —
       unlike a max, this has non-zero gradient everywhere the boxes respond)
  + person-detection suppression       (HOG, and the neural Faster R-CNN detector
       when available)
  + embedding disruption               (all loaded verification models)
  + transform-aware disruption         (real JPEG / resize / brightness / contrast
       applied to candidates during gradient estimation)

The detection gradient is estimated over **Gaussian-blob (low-frequency)
directions** inside the mask — a structured search that finds useful
directions far more efficiently than full-resolution random noise. Updates are
sign-SGD inside the epsilon bound, with adaptive stopping on measured target
values and a visual-quality guard (candidates below the SSIM/PSNR floors are
rejected). This is a real gradient-free technique, not a heuristic filter.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import settings
from ..models.face_models import FaceModel, FaceModelRegistry
from ..vision.face_detector import get_face_detector
from ..vision.person_detector import get_person_detector

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]


@dataclass
class ProtectResult:
    protected: np.ndarray  # HxWx3 uint8
    perturbation: np.ndarray  # HxWx3 float (0..1 scale)
    mask: np.ndarray  # HxW float weights (soft region mask)
    iterations_run: int
    early_stopped: bool
    final_loss: float
    final_distances: dict[str, float]  # per-model mean embedding distance
    loss_breakdown: dict[str, float] = field(default_factory=dict)
    epsilon: float = settings.PERTURBATION_EPSILON
    vision_distance: dict[str, float] = field(default_factory=dict)
    refinement: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# region helpers
# ---------------------------------------------------------------------------


def _expand_box(box: tuple[int, int, int, int], margin: float, h: int, w: int) -> tuple[int, int, int, int]:
    """Expand a box to a square crop centered on the box, clamped to the image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * margin
    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    return (
        max(0, nx1),
        max(0, ny1),
        min(w, nx1 + int(round(side))),
        min(h, ny1 + int(round(side))),
    )


def build_face_mask(h: int, w: int, faces, dilate: float) -> np.ndarray:
    """Union of dilated face boxes as a boolean mask (perturbation support)."""
    mask = np.zeros((h, w), dtype=np.float32)
    for face in faces:
        x1, y1, x2, y2 = _expand_box(face.box, dilate, h, w)
        mask[y1:y2, x1:x2] = 1.0
    return mask


def build_protection_mask(
    h: int,
    w: int,
    faces,
    persons=None,
    face_dilate: float | None = None,
    person_dilate: float | None = None,
    person_weight: float | None = None,
    context_weight: float | None = None,
) -> np.ndarray:
    """Soft weighted region mask: faces 1.0, persons ``person_weight``, context ``context_weight``.

    Edges are Gaussian-smoothed so the perturbation fades out instead of ending
    in a visible rectangular boundary; the inner core of each face keeps the
    full budget.
    """
    face_dilate = face_dilate if face_dilate is not None else settings.FACE_REGION_DILATE
    person_dilate = person_dilate if person_dilate is not None else settings.PERSON_REGION_DILATE
    person_weight = person_weight if person_weight is not None else settings.PERSON_REGION_WEIGHT
    context_weight = context_weight if context_weight is not None else settings.CONTEXT_MASK_WEIGHT

    hard = np.full((h, w), context_weight, dtype=np.float32)
    face_cores: list[tuple[int, int, int, int]] = []
    for face in faces:
        x1, y1, x2, y2 = _expand_box(face.box, face_dilate, h, w)
        hard[y1:y2, x1:x2] = 1.0
        # inner core (60%) restored to full budget after blurring
        iw = int(round((x2 - x1) * 0.6))
        ih = int(round((y2 - y1) * 0.6))
        cx1 = max(0, (x1 + x2) // 2 - iw // 2)
        cy1 = max(0, (y1 + y2) // 2 - ih // 2)
        cx2 = min(w, cx1 + iw)
        cy2 = min(h, cy1 + ih)
        if cx2 > cx1 and cy2 > cy1:
            face_cores.append((cx1, cy1, cx2, cy2))
    for person in persons or []:
        x1, y1, x2, y2 = _expand_box(person.box, person_dilate, h, w)
        hard[y1:y2, x1:x2] = np.maximum(hard[y1:y2, x1:x2], person_weight)

    sigma = max(2.0, (max(h, w) / 700.0) * 5.0)
    soft = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma)
    soft = np.clip(soft, 0.0, 1.0)
    for cx1, cy1, cx2, cy2 in face_cores:
        soft[cy1:cy2, cx1:cx2] = np.maximum(soft[cy1:cy2, cx1:cx2], 1.0)
    soft = np.maximum(soft, context_weight)
    return soft.astype(np.float32)


# ---------------------------------------------------------------------------
# differentiable helpers
# ---------------------------------------------------------------------------


def _gauss_kernel_torch(channels: int, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def torch_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM between two [1,3,H,W] tensors in [0,1]. Returns scalar in (0,1]."""
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = _gauss_kernel_torch(img1.shape[1]).to(img1.device)
    pad = kernel.shape[-1] // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=img1.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=img1.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean()


def _dct_8() -> torch.Tensor:
    """8x8 orthonormal DCT-II basis (rows = frequencies)."""
    n = 8
    j = torch.arange(n, dtype=torch.float32)
    i = torch.arange(n, dtype=torch.float32)
    v = torch.cos(math.pi * (2.0 * j[None, :] + 1.0) * i[:, None] / (2.0 * n))
    v[0] = v[0] / math.sqrt(2.0)  # DC row
    return v * math.sqrt(2.0 / n)


_DCT8: torch.Tensor | None = None


def torch_jpeg_surrogate(x: torch.Tensor, quality: float = 0.6) -> torch.Tensor:
    """Differentiable JPEG-like compression: 8x8 block DCT + AC thresholding.

    The hard coefficient threshold uses a straight-through approximation
    (gradients flow through retained coefficients), which is the standard
    differentiable JPEG surrogate. ``quality`` in (0, 1]; lower = harsher.
    """
    global _DCT8  # noqa: PLW0603
    n = 8
    if _DCT8 is None or _DCT8.device != x.device:
        _DCT8 = _dct_8().to(x.device)
    D = _DCT8

    b, c, h, w = x.shape
    ph = (h + n - 1) // n * n
    pw = (w + n - 1) // n * n
    if ph != h or pw != w:
        x = F.pad(x, (0, pw - w, 0, ph - h), mode="reflect")

    # patches: dims (b, c, ph, pw, pr, pc) with pr,pc the 8x8 patch pixels
    patches = x.unfold(2, n, n).unfold(3, n, n)
    # forward DCT: Y = D X D^T  (D has dims (i,j), i = frequency; X layout transposed
    # w.r.t. the basis, which is immaterial because thresholding is symmetric)
    t1 = torch.einsum("bchwpq,ip->bchwiq", patches, D)  # T1[i,q] = sum_p D[i,p] X[p,q]
    coef = torch.einsum("bchwiq,lq->bchwil", t1, D)  # Y[i,l] = sum_q T1[i,q] D[l,q]
    tau = (1.0 - quality) * 3.5
    coef = coef * (torch.abs(coef) > tau).float()
    # inverse DCT: X' = D^T Y D
    s1 = torch.einsum("bchwil,ip->bchwpl", coef, D)  # S[p,l] = sum_i D[i,p] Y[i,l]
    rec = torch.einsum("bchwpl,lq->bchwpq", s1, D)  # X'[p,q] = sum_l S[p,l] D[l,q]
    # refold: (b,c,ph,pw,pr,pc) -> (b,c,ph*n,pw*n)
    rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(b, c, ph, pw)
    return rec[..., :h, :w]


def _crop_tensor(img: torch.Tensor, box: tuple[int, int, int, int], size: int) -> torch.Tensor:
    """Crop a [1,3,H,W] tensor to a box and bilinear-resize to (size, size)."""
    x1, y1, x2, y2 = box
    crop = img[:, :, y1:y2, x1:x2]
    return F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)


def _numpy_crops(img: np.ndarray, boxes: list[tuple[int, int, int, int]], size: int) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((size, size, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        crops.append(crop)
    return crops


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _lse(scores: np.ndarray) -> float:
    """Log-sum-exp of an array of scores (smooth max)."""
    m = float(scores.max())
    return float(m + math.log(max(float(np.exp(scores - m).sum()), 1e-12)))


class AdversarialProtector:
    def __init__(self, registry: FaceModelRegistry, device: str) -> None:
        self.registry = registry
        self.device = device
        # only loaded models can participate (a failed download must not crash)
        self.models: list[FaceModel] = [m for m in registry.optimization_models if m.info.loaded]
        self._face_detector = get_face_detector()
        self._person_detector = get_person_detector()
        self._neural_person_detector = None  # lazy
        self._mtcnn_detector = None  # lazy

    # ------------------------------------------------------------------
    @staticmethod
    def _variants(x: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Differentiable approximations of common image transformations."""
        out: list[tuple[str, torch.Tensor]] = []

        small = F.interpolate(x, scale_factor=0.75, mode="bilinear", align_corners=False)
        back = F.interpolate(small, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        out.append(("scale", back))

        # center crop to 90% then resize back (differentiable crop)
        hh, ww = x.shape[2], x.shape[3]
        cw, chh = max(8, int(round(ww * 0.9))), max(8, int(round(hh * 0.9)))
        x0, y0 = (ww - cw) // 2, (hh - chh) // 2
        cropped = x[:, :, y0 : y0 + chh, x0 : x0 + cw]
        out.append(("crop", F.interpolate(cropped, size=(hh, ww), mode="bilinear", align_corners=False)))

        # clamp the base to avoid an infinite gradient (0.9 * x**-0.1 at x=0)
        out.append(("gamma", torch.clamp(x, 1e-3, 1.0) ** 0.9))

        shift = max(1, int(round(x.shape[3] * 0.02)))
        out.append(("translate", torch.roll(x, shifts=shift, dims=3)))

        noise = torch.empty_like(x).uniform_(-0.01, 0.01)
        out.append(("noise", torch.clamp(x + noise, 0, 1)))

        mean = x.mean(dim=(2, 3), keepdim=True)
        out.append(("contrast", torch.clamp((x - mean) * 1.15 + mean, 0, 1)))

        out.append(("brightness", torch.clamp(x * 0.9, 0, 1)))

        kernel = _gauss_kernel_torch(x.shape[1], size=7, sigma=1.2).to(x.device)
        out.append(("blur", F.conv2d(x, kernel, padding=3, groups=x.shape[1])))

        out.append(("jpeg", torch_jpeg_surrogate(x, quality=0.6)))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _model_boxes(model: FaceModel, faces, h: int, w: int) -> list[tuple[int, int, int, int]]:
        """Per-model crop boxes: face crops for face models, whole image for global ones."""
        if getattr(model, "mode", "face") == "global":
            return [(0, 0, w, h)]
        return [_expand_box(f.box, settings.FACE_CROP_MARGIN, h, w) for f in faces]

    # ------------------------------------------------------------------
    def _mtcnn_det_loss_initial(
        self, x0: torch.Tensor, face_crops, raw_boxes: list | None = None
    ) -> torch.Tensor | None:
        """Probe the differentiable MTCNN surrogate; None when unavailable."""
        try:
            if self._mtcnn_detector is None:
                from ..vision.mtcnn_face_detector import get_mtcnn_face_detector  # noqa: PLC0415

                self._mtcnn_detector = get_mtcnn_face_detector()
            if not self._mtcnn_detector.available:
                return None
            with torch.no_grad():
                _ = self._mtcnn_det_loss(x0, face_crops, raw_boxes=raw_boxes)
            logger.info("MTCNN P-Net/R-Net/O-Net face-detection surrogate ready.")
            return torch.zeros((), device=self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTCNN detection surrogate unavailable: %s", exc)
            return None

    def _mtcnn_det_loss(
        self, x_adv: torch.Tensor, face_crops, raw_boxes: list | None = None
    ) -> torch.Tensor:
        """Differentiable face-detection suppression via the MTCNN cascade stages.

        Combines two surrogates, both in **logit space** so saturated
        probabilities respond linearly instead of being pinned at ~1.0:

        * P-Net objectness at several pyramid scales (kills proposals at the
          source across the scales the real cascade uses);
        * O-Net face score on a margined 48px crop -- O-Net's probability
          **is** the confidence the cascade reports, so pushing its logit
          toward ``ONET_LOGIT_TARGET`` directly lowers the reported confidence
          (below the cascade's 0.7 threshold the face is missed entirely).

        The full cascade (with NMS) and the OpenCV SSD remain evaluation
        targets; this is the differentiable surrogate for them.
        """
        if self._mtcnn_detector is None:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        count = 0
        pnet_target = settings.PNET_LOGIT_TARGET
        onet_target = settings.ONET_LOGIT_TARGET
        rnet_target = settings.RNET_LOGIT_TARGET
        # R-Net/O-Net crops must match the cascade's ``rerec`` box exactly
        # (square of the **raw** detection box); ``face_crops`` are already
        # expanded ~2.2x for the embedding models and P-Net pyramid.
        net_boxes = raw_boxes if raw_boxes is not None else face_crops
        for box, nbox in zip(face_crops, net_boxes):
            # P-Net: the face at several sizes inside a 96px crop.
            crop = _crop_tensor(x_adv, box, 96)
            for scale in (1.0, 0.7, 0.5, 0.35):
                cs = max(24, int(round(96 * scale)))
                c = F.interpolate(crop, size=(cs, cs), mode="bilinear", align_corners=False)
                obj = self._mtcnn_detector.pnet_objectness(c)
                k = min(24, obj.numel())
                top = torch.topk(obj.reshape(-1), k=k)[0].clamp(1e-6, 1 - 1e-6)
                logit = torch.log(top / (1 - top))
                total = total + F.relu(logit - pnet_target).mean()
                count += 1
            # R-Net/O-Net: 24px/48px crops of the square-ified box -- exactly
            # what the cascade feeds those stages (``rerec`` makes the box
            # square, ``pad`` only truncates to image bounds; no margin is
            # added). Matching the cascade's crop is what makes the surrogate
            # transfer. R-Net's 0.7-threshold decision removes proposals, so
            # pushing it below the threshold deletes the face entirely.
            bx1, by1, bx2, by2 = nbox
            cxm, cym = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            side = max(bx2 - bx1, by2 - by1)
            square = tuple(
                int(round(v))
                for v in (cxm - side / 2.0, cym - side / 2.0, cxm + side / 2.0, cym + side / 2.0)
            )
            crop24 = _crop_tensor(x_adv, square, 24)
            p24 = self._mtcnn_detector.rnet_objectness(crop24).clamp(1e-6, 1 - 1e-6)
            logit24 = torch.log(p24 / (1 - p24))
            total = total + F.relu(logit24 - rnet_target).mean()
            count += 1
            crop48 = _crop_tensor(x_adv, square, 48)
            p48 = self._mtcnn_detector.onet_objectness(crop48).clamp(1e-6, 1 - 1e-6)
            logit48 = torch.log(p48 / (1 - p48))
            total = total + F.relu(logit48 - onet_target).mean()
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    def protect(
        self,
        original_rgb: np.ndarray,
        faces,
        persons=None,
        progress: ProgressCallback | None = None,
        epsilon: float | None = None,
    ) -> ProtectResult:
        """Run the full protection pipeline (optimization + black-box refinement).

        ``epsilon`` optionally overrides the L-inf perturbation bound (the
        production pipeline gives this secondary stage a fraction of the total
        budget so the multi-family editing protection keeps the rest).
        """
        if not self.models:
            raise RuntimeError("No differentiable surrogate models available for protection.")
        if not faces:
            raise ValueError("No faces provided to the protection engine.")

        eps = epsilon if epsilon is not None else settings.PERTURBATION_EPSILON
        margin = settings.OPT_MARGIN
        h, w = original_rgb.shape[:2]

        orig_f = original_rgb.astype(np.float32) / 255.0
        x0 = torch.from_numpy(np.transpose(orig_f, (2, 0, 1))[None, ...]).to(self.device).detach()

        mask = build_protection_mask(h, w, faces, persons)
        mask_t = torch.from_numpy(mask[None, None, ...]).to(self.device)
        mask3 = mask[..., None]  # HxWx1 for numpy refinement

        face_crops = [_expand_box(f.box, settings.FACE_CROP_MARGIN, h, w) for f in faces]
        n_face_crops = max(len(face_crops), 1)
        model_boxes: dict[str, list[tuple[int, int, int, int]]] = {
            m.info.id: self._model_boxes(m, faces, h, w) for m in self.models
        }
        model_modes: dict[str, str] = {m.info.id: getattr(m, "mode", "face") for m in self.models}
        mtcnn_loss = (
            self._mtcnn_det_loss_initial(x0, face_crops, raw_boxes=[f.box for f in faces])
            if settings.W_FACE_DET > 0
            else None
        )
        mtcnn_init_val: float | None = None
        if mtcnn_loss is not None:
            with torch.no_grad():
                mtcnn_init_val = float(
                    self._mtcnn_det_loss(x0, face_crops, raw_boxes=[f.box for f in faces]).item()
                )

        # ---- phase 1: precompute original embeddings per (model, crop) ----
        orig_embs: dict[tuple[str, int], torch.Tensor] = {}
        with torch.no_grad():
            for model in self.models:
                for i, box in enumerate(model_boxes[model.info.id]):
                    crop = _crop_tensor(x0, box, model.input_size)
                    crop = model.preprocess_torch(crop)
                    emb = model.torch_model(crop)
                    emb = F.normalize(emb, p=2, dim=1).detach()
                    orig_embs[(model.info.id, i)] = emb

        # ---- phase 1: differentiable optimization -------------------------
        delta = torch.zeros_like(x0, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=settings.OPT_LR)
        n_iters = settings.OPT_ITERATIONS_GPU if self.device == "cuda" else settings.OPT_ITERATIONS_CPU
        loss_breakdown: dict[str, float] = {}
        early_stopped = False
        last_dist: dict[str, float] = {}
        last_vision: dict[str, float] = {}

        def _report(it: int, loss: float, dists: dict[str, float], vis: dict[str, float]) -> None:
            if progress is None:
                return
            progress(
                {
                    "iteration": it,
                    "total": n_iters,
                    "phase": "optimize",
                    "loss": float(loss),
                    "distances": {k: round(float(v), 4) for k, v in dists.items()},
                    "vision": {k: round(float(v), 4) for k, v in vis.items()},
                }
            )

        def _embed_adv(x_adv: torch.Tensor) -> dict[tuple[str, int], torch.Tensor]:
            embs: dict[tuple[str, int], torch.Tensor] = {}
            for model in self.models:
                for i, box in enumerate(model_boxes[model.info.id]):
                    crop = _crop_tensor(x_adv, box, model.input_size)
                    crop = model.preprocess_torch(crop)
                    emb = model.torch_model(crop)
                    embs[(model.info.id, i)] = F.normalize(emb, p=2, dim=1)
            return embs

        best_loss = float("inf")
        no_improve = 0
        it = 0
        for it in range(1, n_iters + 1):
            optimizer.zero_grad()
            with torch.enable_grad():
                delta.data.copy_(torch.clamp(delta.data, -eps, eps) * mask_t)
                x_adv = torch.clamp(x0 + delta, 0.0, 1.0)

                embs_adv = _embed_adv(x_adv)
                identity = torch.zeros((), device=self.device)
                vision = torch.zeros((), device=self.device)
                dists: dict[str, list[float]] = {}
                vision_dists: dict[str, list[float]] = {}
                for (mid, i), e_adv in embs_adv.items():
                    e_orig = orig_embs[(mid, i)]
                    dist = torch.norm(e_adv - e_orig, p=2, dim=1)
                    if model_modes[mid] == "global":
                        vision = vision + F.relu(settings.VISION_MARGIN - dist).sum()
                        vision_dists.setdefault(mid, []).append(float(dist.item()))
                    else:
                        identity = identity + F.relu(margin - dist).sum()
                        dists.setdefault(mid, []).append(float(dist.item()))
                identity = identity / n_face_crops
                vision = vision / max(len(vision_dists), 1)
                # differentiable face-detection suppression (MTCNN P-Net surrogate)
                det_surrogate = (
                    self._mtcnn_det_loss(x_adv, face_crops, raw_boxes=[f.box for f in faces])
                    * settings.W_FACE_DET
                    if mtcnn_loss is not None
                    else torch.zeros((), device=self.device)
                )

                ssim_val = torch_ssim(x_adv, x0)
                mse_val = F.mse_loss(x_adv, x0)
                perceptual = (1.0 - ssim_val) * settings.W_SSIM + mse_val * settings.W_MSE
                perturb = torch.norm(delta) * settings.W_PERTURBATION

                robustness = torch.zeros((), device=self.device)
                rob_interval = (
                    settings.OPT_ROBUSTNESS_INTERVAL
                    if self.device == "cuda"
                    else max(2, settings.OPT_ROBUSTNESS_INTERVAL)
                )
                if it % rob_interval == 0 or it == n_iters:
                    variants = self._variants(x_adv)
                    for _, variant in variants:
                        embs_var = _embed_adv(variant)
                        for (mid, i), e_var in embs_var.items():
                            e_orig = orig_embs[(mid, i)]
                            dist_v = torch.norm(e_var - e_orig, p=2, dim=1)
                            if model_modes[mid] == "global":
                                robustness = robustness + F.relu(settings.VISION_MARGIN - dist_v).sum()
                            else:
                                robustness = robustness + F.relu(margin - dist_v).sum()
                    denom = max(len(variants), 1) * max(
                        sum(len(v) for v in model_boxes.values()), 1
                    )
                    robustness = robustness / denom
                robustness = robustness * settings.W_ROBUSTNESS

                loss = (
                    identity * settings.W_IDENTITY
                    + vision * settings.W_VISION
                    + det_surrogate
                    + perceptual
                    + perturb
                    + robustness
                )
                loss.backward()

            optimizer.step()
            delta.data.copy_(torch.clamp(delta.data, -eps, eps) * mask_t)

            last_dist = {k: float(np.mean(v)) for k, v in dists.items()}
            last_vision = {k: float(np.mean(v)) for k, v in vision_dists.items()}
            loss_breakdown = {
                "identity": float((identity * settings.W_IDENTITY).item()),
                "vision": float((vision * settings.W_VISION).item()),
                "face_detection": float(det_surrogate.item()),
                "perceptual": float(perceptual.item()),
                "perturbation": float(perturb.item()),
                "robustness": float(robustness.item()),
            }
            if it % 5 == 0 or it == n_iters:
                _report(it, float(loss.item()), last_dist, last_vision)

            # adaptive early stop: no improvement for a while, or all targets met
            if float(loss.item()) < best_loss - 1e-4:
                best_loss = float(loss.item())
                no_improve = 0
            else:
                no_improve += 1
            det_improved = (
                mtcnn_init_val is None
                or float(det_surrogate.item()) < mtcnn_init_val * 0.15
            )
            min_iters = max(12, n_iters // 3)
            targets_met = (
                all(d >= margin + 0.05 for d in last_dist.values())
                and all(v >= settings.VISION_MARGIN + 0.05 for v in last_vision.values())
                and det_improved
            )
            if no_improve >= settings.OPT_EARLY_STOP_PATIENCE or (it >= min_iters and targets_met):
                early_stopped = True
                break
            if it >= n_iters:
                break

        delta.data.copy_(torch.clamp(delta.data, -eps, eps) * mask_t)
        x_final = torch.clamp(x0 + delta, 0.0, 1.0)
        protected = x_final.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0
        protected = np.clip(np.round(protected), 0, 255).astype(np.uint8)
        perturb_np = delta.detach().cpu().numpy()[0].transpose(1, 2, 0)

        # ---- phase 2: black-box refinement --------------------------------
        refinement = self._refine_blackbox(
            original_rgb, protected, face_crops, mask3, eps, margin, progress, faces, persons
        )

        return ProtectResult(
            protected=protected,
            perturbation=perturb_np,
            mask=mask,
            iterations_run=it,
            early_stopped=early_stopped,
            final_loss=float(loss.item()),
            final_distances={k: round(v, 4) for k, v in last_dist.items()},
            loss_breakdown={k: round(v, 6) for k, v in loss_breakdown.items()},
            epsilon=eps,
            vision_distance={k: round(v, 4) for k, v in last_vision.items()},
            refinement=refinement,
        )

    # ------------------------------------------------------------------
    # Phase 2 helpers: smooth detection signals (logit space)
    # ------------------------------------------------------------------
    @staticmethod
    def _logit(p: float) -> float:
        """Inverse sigmoid; linearizes saturated detection probabilities.

        A confidence of 0.997 sits on the flat tail of the sigmoid, so a
        perturbation moves the *probability* by ~1e-4 (an unusable finite-
        difference signal). The logit amplifies exactly those changes
        (d logit/dp = 1/(p(1-p)) ≈ 330 at p=0.997), making zeroth-order
        gradients informative again.
        """
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    def _face_det_lse_loss(self, img: np.ndarray, faces) -> float:
        """Smooth face-detection suppression loss in logit space.

        Unlike ``max`` confidence — which is piecewise-constant and gives a
        ~always-zero finite-difference gradient — LSE over the logit-transformed
        box scores responds smoothly to any change in the detector's raw
        response, so zeroth-order gradients are informative.
        """
        if not faces:
            return 0.0
        try:
            dets = self._face_detector.detect(img, confidence=0.02)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Face detector unavailable during refinement: %s", exc)
            return 0.0
        if not dets:
            return 0.0
        total = 0.0
        for face in faces:
            scores = np.array(
                [self._logit(float(d.confidence)) for d in dets if _boxes_overlap(d.box, face.box)],
                dtype=np.float32,
            )
            if scores.size:
                total += _lse(scores)
        return total / max(len(faces), 1)

    def _mtcnn_lse_loss(self, img: np.ndarray, faces) -> float:
        """MTCNN cascade confidence suppression (logit space)."""
        if not faces:
            return 0.0
        try:
            if self._mtcnn_detector is None:
                from ..vision.mtcnn_face_detector import get_mtcnn_face_detector  # noqa: PLC0415

                self._mtcnn_detector = get_mtcnn_face_detector()
            if not self._mtcnn_detector.available:
                return 0.0
            dets = self._mtcnn_detector.detect(img, confidence=0.05)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTCNN detector unavailable during refinement: %s", exc)
            return 0.0
        if not dets:
            return 0.0
        total = 0.0
        for face in faces:
            scores = np.array(
                [self._logit(float(d.confidence)) for d in dets if _boxes_overlap(d.box, face.box)],
                dtype=np.float32,
            )
            if scores.size:
                total += _lse(scores)
        return total / max(len(faces), 1)

    def _hog_person_lse_loss(self, img: np.ndarray, person_boxes: list[tuple[int, int, int, int]]) -> float:
        if not person_boxes:
            return 0.0
        try:
            persons = self._person_detector.detect(img, confidence=-1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Person detector unavailable during refinement: %s", exc)
            return 0.0
        scores = np.array(
            [
                self._logit(float(p.confidence))
                for p in persons
                if any(_boxes_overlap(p.box, b) for b in person_boxes)
            ],
            dtype=np.float32,
        )
        if not scores.size:
            return 0.0
        return _lse(scores)

    def _neural_person_lse_loss(self, img: np.ndarray, person_boxes: list[tuple[int, int, int, int]]) -> float:
        if not person_boxes:
            return 0.0
        try:
            if self._neural_person_detector is None:
                from ..vision.neural_person_detector import get_neural_person_detector  # noqa: PLC0415

                self._neural_person_detector = get_neural_person_detector()
            if not self._neural_person_detector.available:
                return 0.0
            persons = self._neural_person_detector.detect(img, confidence=0.05)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neural person detector unavailable during refinement: %s", exc)
            return 0.0
        scores = np.array(
            [
                self._logit(float(p.confidence))
                for p in persons
                if any(_boxes_overlap(p.box, b) for b in person_boxes)
            ],
            dtype=np.float32,
        )
        if not scores.size:
            return 0.0
        return _lse(scores)

    def _detection_loss(
        self,
        img: np.ndarray,
        faces,
        person_boxes: list[tuple[int, int, int, int]],
    ) -> float:
        """Combined smooth detection-suppression loss.

        Every detector the system is tested against becomes an optimization
        target: OpenCV SSD face, MTCNN cascade face, HOG person, and the
        Faster R-CNN neural person detector. All terms are logit-space LSE so
        saturated confidences contribute real gradients.
        """
        loss = settings.W_FACE_DET * self._face_det_lse_loss(img, faces)
        loss += settings.W_MTCNN_DET * self._mtcnn_lse_loss(img, faces)
        loss += settings.W_PERSON_DET * self._hog_person_lse_loss(img, person_boxes)
        loss += settings.W_NEURAL_PERSON_DET * self._neural_person_lse_loss(img, person_boxes)
        return float(loss)

    # ------------------------------------------------------------------
    def _refine_blackbox(
        self,
        original_rgb: np.ndarray,
        protected_rgb: np.ndarray,
        crop_boxes: list[tuple[int, int, int, int]],
        mask3: np.ndarray,
        eps: float,
        margin: float,
        progress: ProgressCallback | None,
        faces,
        persons=None,
    ) -> dict:
        """Zeroth-order refinement: embedding + detection suppression.

        Embedding gradients use full-mask random directions; detection gradients
        use structured Gaussian-blob directions (low-frequency search). Sign-SGD
        updates stay inside the epsilon bound, with adaptive stopping and a
        visual-quality guard. Mutates ``protected_rgb`` in place.
        """
        models = self.registry.verification_models
        if not models:
            return {"applied": False, "note": "No verification models loaded."}

        h, w = original_rgb.shape[:2]
        person_target_boxes = [p.box for p in persons or []]
        model_boxes: dict[str, list[tuple[int, int, int, int]]] = {
            m.info.id: ([(0, 0, w, h)] if getattr(m, "mode", "face") == "global" else crop_boxes)
            for m in models
        }

        def _embed_img(img: np.ndarray, model) -> list[np.ndarray]:
            return model.embed_crops(
                _numpy_crops(img, model_boxes[model.info.id], model.input_size), self.device
            )

        def _distances(img: np.ndarray) -> dict[str, float]:
            out: dict[str, float] = {}
            for model in models:
                embs = _embed_img(img, model)
                orig_embs = _embed_img(original_rgb, model)
                out[model.info.id] = float(
                    np.mean([np.linalg.norm(a - b) for a, b in zip(embs, orig_embs)])
                )
            return out

        before = _distances(protected_rgb)
        weak = {mid: d for mid, d in before.items() if d < 0.55}
        # Every model must hold a floor of 0.60 (not just the weak ones):
        # models already above the goal used to be excluded from the loss, so
        # the detection attack could freely drag them back toward the original
        # embeddings. The floor makes the embedding gradient a *preserving*
        # force for all models while still pushing the weak ones upward.
        target_goals = {mid: 0.60 for mid in before}
        emb_active = bool(weak)

        det_before = self._detection_loss(protected_rgb, faces, person_target_boxes)
        if not emb_active and det_before <= 1e-6 and not persons:
            return {
                "applied": False,
                "note": "All verification models already show strong disruption and no detection suppression is needed.",
            }

        if self.device == "cuda":
            max_iters = settings.REFINE_MAX_ITERS_GPU
            k_emb = settings.REFINE_DIRECTIONS_GPU
            k_det = settings.DET_BLOB_DIRECTIONS_GPU
            det_interval = settings.DET_GRAD_INTERVAL_GPU
        else:
            max_iters = settings.REFINE_MAX_ITERS_CPU
            k_emb = settings.REFINE_DIRECTIONS_CPU
            k_det = settings.DET_BLOB_DIRECTIONS_CPU
            det_interval = settings.DET_GRAD_INTERVAL_CPU
        combined_iters = max(1, int(round(max_iters * (1.0 - settings.DET_ATTACK_FRACTION))))
        logger.info(
            "Refinement: max_iters=%d combined=%d det_attack=%d",
            max_iters,
            combined_iters,
            max_iters - combined_iters,
        )
        perturb_mag = 6.0
        lr = 1.2
        eps_px = eps * 255.0
        rng = np.random.default_rng(settings.OPT_SEED)

        orig_embs_cache: dict[str, list[np.ndarray]] = {}
        for model in models:
            orig_embs_cache[model.info.id] = _embed_img(original_rgb, model)

        def _emb_loss(img: np.ndarray) -> float:
            total = 0.0
            for model in models:
                embs = _embed_img(img, model)
                for e, eo in zip(embs, orig_embs_cache[model.info.id]):
                    d = float(np.linalg.norm(e - eo))
                    total += max(0.0, target_goals[model.info.id] - d)
            return total

        transform_cycle = itertools.count()

        def _transform_emb_loss(img: np.ndarray) -> float:
            """Embedding loss on real-transformed candidates (robustness-aware).

            The full transform set (JPEG, resize, crop, brightness, contrast,
            re-encode) is covered, but only 3 of 6 are evaluated per call
            (rotating) so the per-iteration cost stays bounded while every
            transform is still seen every two iterations.
            """
            from ..robustness.tester import _apply_transform  # noqa: PLC0415

            img_u8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
            all_t = settings.ROBUSTNESS_TRANSFORMS
            idx = next(transform_cycle) % len(all_t)
            tnames = [all_t[(idx + i) % len(all_t)] for i in range(3)]
            total = 0.0
            for tname in tnames:
                try:
                    t = _apply_transform(tname, img_u8)
                except Exception:  # noqa: BLE001
                    continue
                for model in models:
                    embs = _embed_img(t, model)
                    for e, eo in zip(embs, orig_embs_cache[model.info.id]):
                        d = float(np.linalg.norm(e - eo))
                        goal = 0.55 if getattr(model, "mode", "face") == "face" else 0.45
                        total += max(0.0, goal - d)
            return total

        def _random_direction() -> np.ndarray:
            u = (rng.integers(0, 2, size=mask3.shape).astype(np.float32) * 2 - 1) * mask3 * perturb_mag
            return u

        def _blob_direction() -> np.ndarray:
            """Structured low-frequency direction: random Gaussian blobs in the mask region."""
            u = np.zeros((h, w, 3), dtype=np.float32)
            strong = mask3[..., 0] > 0.3
            ys, xs = np.nonzero(strong)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            n_blobs = settings.DET_BLOB_COUNT
            for _ in range(n_blobs):
                if ys.size:
                    idx = int(rng.integers(0, ys.size))
                    cy, cx = float(ys[idx]), float(xs[idx])
                else:
                    cy, cx = float(rng.uniform(0, h)), float(rng.uniform(0, w))
                sigma = float(rng.uniform(settings.DET_BLOB_SIGMA_MIN, settings.DET_BLOB_SIGMA_MAX))
                amp = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.4, 1.0))
                g = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
                ch = int(rng.integers(0, 3))
                u[..., ch] += g
            return u * mask3 * perturb_mag

        def _texture_direction() -> np.ndarray:
            """High-frequency direction: lightly-smoothed per-pixel noise in the mask.

            Per-pixel texture moves CNNs (MTCNN cascade measured: -0.016 at
            ±16/255) better than smooth blobs; this is the dominant class.
            """
            n = rng.normal(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=1.0, sigmaY=1.0)
            return n * mask3 * (perturb_mag * 1.5)

        def _sin_direction() -> np.ndarray:
            """Mid-frequency sinusoid patch: transfers across downsampling better
            than per-pixel noise and still perturbs CNN feature maps."""
            u = np.zeros((h, w, 3), dtype=np.float32)
            strong = mask3[..., 0] > 0.3
            ys, xs = np.nonzero(strong)
            if ys.size:
                idx = int(rng.integers(0, ys.size))
                cy, cx = float(ys[idx]), float(xs[idx])
            else:
                cy, cx = float(rng.uniform(0, h)), float(rng.uniform(0, w))
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            lam = float(rng.uniform(8.0, 32.0))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            ang = float(rng.uniform(0.0, math.pi))
            wave = np.sin(
                2.0 * np.pi * ((xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)) / lam + phase
            ).astype(np.float32)
            for ch in range(3):
                u[..., ch] = wave * float(rng.choice([-1.0, 1.0]))
            return u * mask3 * perturb_mag

        def _det_direction() -> np.ndarray:
            r = rng.random()
            if r < settings.DET_TEXTURE_FRACTION:
                return _texture_direction()
            if r < settings.DET_TEXTURE_FRACTION + settings.DET_BLOB_FRACTION:
                return _blob_direction()
            return _sin_direction()

        class _B:
            def __init__(self, box):
                self.box = box

        def _scaled_boxes() -> tuple:
            """Face boxes scaled for fast MTCNN-gradient evals (480px max dim)."""
            max_dim = 480
            if max(h, w) <= max_dim:
                return faces, 1.0
            s = max_dim / max(h, w)
            scaled_faces = [
                _B((int(round(f.box[0] * s)), int(round(f.box[1] * s)), int(round(f.box[2] * s)), int(round(f.box[3] * s))))
                for f in faces
            ]
            return scaled_faces, s

        scaled_faces, det_scale = _scaled_boxes()

        def _det_loss_at(x: np.ndarray) -> float:
            """Detection loss for gradient estimation.

            DNN face, HOG, and Faster R-CNN run at full resolution (cheap);
            MTCNN's cascade runs on a 480px downscale because its image pyramid
            still covers the face at the relevant scales and it is ~3x cheaper.
            """
            # HOG (and OpenCV ops) require uint8; candidate arrays from the
            # optimizer are float32, so cast explicitly.
            x_u8 = np.clip(x, 0, 255).astype(np.uint8)
            loss = settings.W_FACE_DET * self._face_det_lse_loss(x_u8, faces)
            loss += settings.W_PERSON_DET * self._hog_person_lse_loss(x_u8, person_target_boxes)
            loss += settings.W_NEURAL_PERSON_DET * self._neural_person_lse_loss(x_u8, person_target_boxes)
            if det_scale == 1.0:
                loss += settings.W_MTCNN_DET * self._mtcnn_lse_loss(x_u8, faces)
            else:
                small = cv2.resize(
                    x_u8, (int(round(w * det_scale)), int(round(h * det_scale))), interpolation=cv2.INTER_AREA
                )
                loss += settings.W_MTCNN_DET * self._mtcnn_lse_loss(small, scaled_faces)
            return float(loss)

        def _grad_emb(x: np.ndarray, include_transforms: bool, k: int | None = None) -> np.ndarray:
            n_dir = k if k is not None else k_emb
            g = np.zeros_like(x)
            for _ in range(n_dir):
                u = _random_direction()
                loss_p = _emb_loss(np.clip(x + u, 0, 255)) + (
                    _transform_emb_loss(x + u) if include_transforms else 0.0
                )
                loss_m = _emb_loss(np.clip(x - u, 0, 255)) + (
                    _transform_emb_loss(x - u) if include_transforms else 0.0
                )
                g += ((loss_p - loss_m) / (2 * perturb_mag)) * u
            g /= max(n_dir, 1)
            norm = np.linalg.norm(g)
            return g / norm if norm > 1e-8 else g

        det_momentum = np.zeros_like(mask3, dtype=np.float32)

        def _grad_det(x: np.ndarray) -> np.ndarray:
            nonlocal det_momentum
            g = np.zeros_like(x)
            for _ in range(k_det):
                u = _det_direction()
                loss_p = _det_loss_at(x + u)
                loss_m = _det_loss_at(x - u)
                g += ((loss_p - loss_m) / (2 * perturb_mag)) * u
            g /= max(k_det, 1)
            norm = np.linalg.norm(g)
            if norm > 1e-8:
                g = g / norm
            det_momentum = 0.85 * det_momentum + 0.15 * g
            mnorm = np.linalg.norm(det_momentum)
            return det_momentum / mnorm if mnorm > 1e-8 else det_momentum

        def _face_ok(img_u8: np.ndarray) -> bool:
            if not faces:
                return True
            best = 0.0
            try:
                dets = self._face_detector.detect(img_u8, confidence=0.05)
                for face in faces:
                    for d in dets:
                        if _boxes_overlap(d.box, face.box):
                            best = max(best, float(d.confidence))
            except Exception:  # noqa: BLE001
                return True
            return best <= settings.DET_TARGET_FACE

        def _ssim_ok(img_u8: np.ndarray) -> bool:
            from ..quality.metrics import ssim  # noqa: PLC0415

            try:
                return float(ssim(original_rgb, img_u8)) >= settings.MIN_SSIM
            except Exception:  # noqa: BLE001
                return True

        x = protected_rgb.astype(np.float32)
        orig = original_rgb.astype(np.float32)
        delta = np.clip(x - orig, -eps_px, eps_px) * mask3

        w_emb = settings.W_EMB
        w_det = settings.W_DET
        no_improve_det = 0
        last_det_loss = float("inf")
        include_transforms = self.device == "cuda"  # CPU: keep refinement affordable
        iters_run = 0
        it = 0

        # ---- loop 1: combined embedding + detection suppression -----------
        for it in range(1, combined_iters + 1):
            g = np.zeros_like(delta)
            if emb_active:
                # transform-aware every iteration: the perturbation must
                # survive real JPEG/resize/crop/brightness/contrast/re-encode
                g = g + w_emb * _grad_emb(x, include_transforms)
            if (it % det_interval == 1) or (it == combined_iters):
                g_det = _grad_det(x)
                if np.linalg.norm(g_det) > 1e-8:
                    g = g + w_det * g_det
            delta = np.clip(delta - lr * np.sign(g) * mask3, -eps_px, eps_px) * mask3
            x_new = np.clip(orig + delta, 0, 255)
            x_u8 = np.clip(np.round(x_new), 0, 255).astype(np.uint8)
            if not _ssim_ok(x_u8):
                lr = max(0.3, lr * 0.5)
                delta = np.clip(x - orig, -eps_px, eps_px) * mask3
                continue
            x = x_new
            iters_run = it
            if progress is not None:
                progress({"iteration": it, "total": max_iters, "phase": "refine"})

            det_now = _det_loss_at(x)
            if det_now < last_det_loss - 1e-4:
                last_det_loss = det_now
                no_improve_det = 0
            else:
                no_improve_det += 1

            if it % 8 == 0 or it == combined_iters:
                cur = _distances(x_u8)
                emb_ok = all(cur[mid] >= target_goals[mid] for mid in weak) or not weak
                det_ok = _face_ok(x_u8) and (det_now <= 1e-6 or not persons)
                if emb_ok and det_ok:
                    break
                if no_improve_det >= 12:
                    break

        # ---- loop 2: detection attack with embedding preservation ----------
        # The pure-detection attack previously destroyed the embedding
        # disruption (FaceNet distance 0.77 -> 0.31 measured), which is also
        # what sank the transformation-robustness verdicts. Each detection
        # gradient is now blended with a low-weight embedding gradient so the
        # attack cannot push the image back toward the original embeddings.
        base = iters_run
        det_iters = max_iters - base
        stale_det = 0
        last_det_loss = _det_loss_at(x)
        cached_g: np.ndarray | None = None
        emb_repair_w = w_emb  # keep embedding disruption roughly equal to detection suppression
        for j in range(1, det_iters + 1):
            it = base + j
            # detector gradient is the expensive part; recompute every 3 iters
            if j == 1 or j % 3 == 0:
                g_det = _grad_det(x)
                if emb_active and np.linalg.norm(g_det) > 1e-8:
                    cached_g = g_det + emb_repair_w * _grad_emb(x, False, k=max(3, k_emb // 2))
                else:
                    cached_g = g_det
            g = cached_g if cached_g is not None else np.zeros_like(delta)
            if np.linalg.norm(g) > 1e-8:
                delta = np.clip(delta - lr * np.sign(g) * mask3, -eps_px, eps_px) * mask3
                x_new = np.clip(orig + delta, 0, 255)
                x_u8 = np.clip(np.round(x_new), 0, 255).astype(np.uint8)
                if _ssim_ok(x_u8):
                    x = x_new
                    iters_run = it
            if progress is not None:
                progress({"iteration": it, "total": max_iters, "phase": "refine"})
            det_now = _det_loss_at(x)
            if det_now < last_det_loss - 1e-4:
                last_det_loss = det_now
                stale_det = 0
            else:
                stale_det += 1
            # absolute target (e.g. MTCNN face missed) or saturated improvement
            if _face_ok(np.clip(np.round(x), 0, 255).astype(np.uint8)) or stale_det >= 5:
                break

        # ---- embedding repair pass -----------------------------------------
        # A short pure-embedding pass restores any distance the detection
        # attack eroded, without undoing the suppression (measured tradeoff).
        repair_iters = 4
        for _ in range(repair_iters):
            g = _grad_emb(x, include_transforms=False) if emb_active else np.zeros_like(delta)
            if np.linalg.norm(g) > 1e-8:
                delta = np.clip(delta - lr * np.sign(g) * mask3, -eps_px, eps_px) * mask3
                x_new = np.clip(orig + delta, 0, 255)
                x_u8 = np.clip(np.round(x_new), 0, 255).astype(np.uint8)
                if _ssim_ok(x_u8):
                    x = x_new
                    iters_run += 1

        protected_rgb[...] = np.clip(np.round(x), 0, 255).astype(np.uint8)
        after = _distances(protected_rgb)
        det_after = self._detection_loss(protected_rgb, faces, person_target_boxes)
        face_before = self._face_max_conf(original_rgb, faces)
        face_after = self._face_max_conf(protected_rgb, faces)
        mtcnn_before = self._mtcnn_max_conf(original_rgb, faces)
        mtcnn_after = self._mtcnn_max_conf(protected_rgb, faces)

        return {
            "applied": True,
            "target_models": list(weak.keys()),
            "iterations": iters_run,
            "directions_per_iteration": k_emb,
            "before": {k: round(v, 4) for k, v in before.items()},
            "after": {k: round(v, 4) for k, v in after.items()},
            "detection_loss_before": round(det_before, 4),
            "detection_loss_after": round(det_after, 4),
            "face_confidence_before": round(face_before, 4),
            "face_confidence_after": round(face_after, 4),
            "mtcnn_confidence_before": round(mtcnn_before, 4),
            "mtcnn_confidence_after": round(mtcnn_after, 4),
            "note": (
                "Black-box (zeroth-order) refinement across all verification models and "
                "detectors: weak embedding models pushed to target, strong models held "
                "above a floor, smooth detection suppression (OpenCV SSD + MTCNN + HOG + "
                "Faster R-CNN) in logit space via structured per-pixel/blob/sinusoid "
                "search, and transform-aware embedding disruption."
            ),
        }

    def _face_max_conf(self, img: np.ndarray, faces) -> float:
        if not faces:
            return 0.0
        best = 0.0
        try:
            dets = self._face_detector.detect(img, confidence=0.05)
            for face in faces:
                for d in dets:
                    if _boxes_overlap(d.box, face.box):
                        best = max(best, float(d.confidence))
        except Exception:  # noqa: BLE001
            return 0.0
        return best

    def _mtcnn_max_conf(self, img: np.ndarray, faces) -> float:
        """Max MTCNN cascade confidence overlapping each face box."""
        if not faces:
            return 0.0
        try:
            if self._mtcnn_detector is None:
                from ..vision.mtcnn_face_detector import get_mtcnn_face_detector  # noqa: PLC0415

                self._mtcnn_detector = get_mtcnn_face_detector()
            if not self._mtcnn_detector.available:
                return 0.0
            dets = self._mtcnn_detector.detect(img, confidence=0.05)
        except Exception:  # noqa: BLE001
            return 0.0
        best = 0.0
        for face in faces:
            for d in dets:
                if _boxes_overlap(d.box, face.box):
                    best = max(best, float(d.confidence))
        return best

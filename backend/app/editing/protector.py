"""Multi-family editing-protection pass.

An adversarial perturbation is optimized in pixel space against SEVERAL AI
attack families at once (spec §3, §6-7) — one perturbation, not one per model:

  * diffusion editing / instruction editing / img2img / inpainting (A-D):
    maximize the denoising reconstruction error of the differentiable SD1.5
    anti-diffusion surrogate (PhotoGuard-style end-to-end attack);
  * identity-reference generation / face-swap (E-F): push the protected
    face's embedding away from the original identity (FaceNet surrogate);
  * general vision encoders (I): disrupt the global image representation
    (MobileNetV3 surrogate).

Each term is a real, measurable model objective; weights are centrally
configurable (``settings.EDITING_*_WEIGHT``) and can be overridden per call
by the adaptive red-team loop. The result keeps a hard SSIM quality floor —
if the perturbation would visibly damage the image the stage is reverted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..config import settings
from .manager import get_editing_manager
from .surrogate import AntiDiffusionSurrogate

logger = logging.getLogger(__name__)


@dataclass
class EditingProtectionResult:
    applied: bool
    iterations: int = 0
    epsilon: float = 0.0
    resolution: int = 0
    timestep: int = 0
    denoising_loss_before: float = 0.0
    denoising_loss_after: float = 0.0
    loss_increase_pct: float = 0.0
    verified_loss_after: float = 0.0
    verified_increase_pct: float = 0.0
    reverted: bool = False
    surrogate_model: str = ""
    note: str = ""
    protected: np.ndarray | None = None
    # multi-family terms (identity-reference / face-swap / vision encoders)
    identity_similarity_before: float | None = None
    identity_similarity_after: float | None = None
    vision_similarity_before: float | None = None
    vision_similarity_after: float | None = None
    # face-swap protection: ArcFace w600k (the swap-encoder family) + the
    # in-place zeroth-order refinement report
    arcface_similarity_before: float | None = None
    arcface_similarity_after: float | None = None
    identity_refinement: dict = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    weights: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "applied": self.applied,
            "iterations": self.iterations,
            "epsilon": round(self.epsilon, 4),
            "resolution": self.resolution,
            "timestep": self.timestep,
            "denoising_loss_before": round(self.denoising_loss_before, 4),
            "denoising_loss_after": round(self.denoising_loss_after, 4),
            "loss_increase_pct": round(self.loss_increase_pct, 1),
            "verified_loss_after": round(self.verified_loss_after, 4),
            "verified_increase_pct": round(self.verified_increase_pct, 1),
            "identity_similarity_before": (
                round(self.identity_similarity_before, 4)
                if self.identity_similarity_before is not None
                else None
            ),
            "identity_similarity_after": (
                round(self.identity_similarity_after, 4)
                if self.identity_similarity_after is not None
                else None
            ),
            "vision_similarity_before": (
                round(self.vision_similarity_before, 4)
                if self.vision_similarity_before is not None
                else None
            ),
            "vision_similarity_after": (
                round(self.vision_similarity_after, 4)
                if self.vision_similarity_after is not None
                else None
            ),
            "arcface_similarity_before": (
                round(self.arcface_similarity_before, 4)
                if self.arcface_similarity_before is not None
                else None
            ),
            "arcface_similarity_after": (
                round(self.arcface_similarity_after, 4)
                if self.arcface_similarity_after is not None
                else None
            ),
            "identity_refinement": self.identity_refinement,
            "reverted": self.reverted,
            "surrogate_model": self.surrogate_model,
            "note": self.note,
            "families": self.families,
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "objective": (
                "single multi-family perturbation: diffusion editing (A-D) + "
                "identity-reference/face-swap (E-F) + vision encoders (I)"
            ),
        }


def _gaussian_kernel(ksize: int = 9, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = g[:, None] * g[None, :]
    return (k / k.sum()).view(1, 1, ksize, ksize)


def _rgb_to_tensor(rgb: np.ndarray, device: str, resolution: int | None = None) -> torch.Tensor:
    arr = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    if resolution is not None and arr.shape[2:] != (resolution, resolution):
        arr = TF.resize(arr, (resolution, resolution), antialias=True)
    return arr.half().to(device)


def torch_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM for the quality floor."""
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = _gaussian_kernel(11, 1.5).half().to(img1.device).repeat(3, 1, 1, 1)
    pad = 5
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=3) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean()


def _variants(x: torch.Tensor) -> list[torch.Tensor]:
    """Differentiable approximations of common image transformations.

    The perturbation must survive JPEG / resize / crop / brightness / contrast
    / re-encoding, so the optimizer maximizes the denoising error on several
    cheap differentiable proxies of those transforms in addition to the raw
    image. The real (non-differentiable) transforms are validated separately
    by the editing benchmark's robustness pass.
    """
    h, w = x.shape[2], x.shape[3]
    out = [x]
    small = F.interpolate(x, scale_factor=0.8, mode="bilinear", align_corners=False)
    out.append(F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False))
    out.append(torch.clamp(x * 0.94, 0.0, 1.0))
    out.append(torch.clamp((x - 0.5) * 1.08 + 0.5, 0.0, 1.0))
    ch, cw = int(h * 0.9), int(w * 0.9)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = x[:, :, y0 : y0 + ch, x0 : x0 + cw]
    out.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False))
    block = F.avg_pool2d(x, kernel_size=8, stride=8)
    up = F.interpolate(block, size=(h, w), mode="bilinear", align_corners=False)
    out.append(torch.clamp(x + 0.03 * (up - x), 0.0, 1.0))
    noise = torch.empty_like(x).uniform_(-0.02, 0.02)
    out.append(torch.clamp(x + noise, 0.0, 1.0))
    return out


def _crop_tensor(img: torch.Tensor, box, size: int) -> torch.Tensor | None:
    """Crop ``img`` [1,3,H,W] to ``box`` (xyxy, float ok) and resize square."""
    x0, y0, x1, y1 = box
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))
    x0 = max(0, min(x0, img.shape[3] - 1))
    y0 = max(0, min(y0, img.shape[2] - 1))
    x1 = max(x0 + 1, min(x1, img.shape[3]))
    y1 = max(y0 + 1, min(y1, img.shape[2]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = img[:, :, y0:y1, x0:x1]
    return TF.resize(crop, (size, size), antialias=True)


def _embed_face(model, crop: torch.Tensor, device: str) -> torch.Tensor:
    """Differentiable face-embedding of a [1,3,S,S] [0,1] crop.

    ``model`` is a torch network (not the registry wrapper); its input space
    is applied by the caller's preprocessing (FaceNet VGGFace2 / CASIA share
    the same 160px input space here).
    """
    from ..models.face_models import facenet_preprocess_torch  # noqa: PLC0415

    inp = facenet_preprocess_torch(crop.float()).to(device)
    emb = model(inp)
    return F.normalize(emb, dim=1)


class _MultiFamilyTerms:
    """Identity-reference + vision-encoder loss terms for the PGD loop.

    Loaded lazily; any term that cannot run (model unavailable / OOM) is
    skipped gracefully — the diffusion term always keeps the loop alive.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.identity_models: list = []  # list of torch networks
        self.vision_model = None

    def _identity_models(self):
        """All differentiable identity encoders available for the PGD.

        TWO FaceNet variants (VGGFace2 + CASIA) are pushed simultaneously so
        the embedding objective does not overfit a single encoder — the old
        single-encoder term barely moved other identity models. This is the
        multi-encoder identity-space concept of ID-Eraser/Phantom applied to
        the encoders we can differentiate.
        """
        if self.identity_models:
            return self.identity_models
        try:
            from ..models.face_models import get_registry  # noqa: PLC0415

            reg = get_registry(self.device)
            for mid in ("facenet_vggface2", "facenet_casia"):
                model = reg._models.get(mid)
                if model is not None and model.info.loaded and model.torch_model is not None:
                    model.torch_model.eval().to(self.device)
                    self.identity_models.append(model.torch_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Identity terms unavailable: %s", exc)
        return self.identity_models

    def _vision(self):
        if self.vision_model is not None:
            return self.vision_model
        try:
            from ..models.face_models import get_registry  # noqa: PLC0415

            reg = get_registry(self.device)
            model = reg._models.get("mobilenet_v3_large")
            if model is not None and model.info.loaded and model.torch_model is not None:
                model.torch_model.eval().to(self.device)
                self.vision_model = model.torch_model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision term unavailable: %s", exc)
        return self.vision_model


def _identity_similarity(
    encoders: list, img: torch.Tensor, boxes, refs: list[list], device: str
) -> float | None:
    """Mean cosine similarity over encoders and face boxes vs reference.

    ``refs`` is a list aligned with ``encoders``; each entry is a list of
    reference embeddings aligned with ``boxes``.
    """
    if not encoders or not boxes or not refs:
        return None
    sims = []
    with torch.no_grad():
        for enc, enc_refs in zip(encoders, refs):
            for box, ref in zip(boxes, enc_refs):
                crop = _crop_tensor(img, box, 160)
                if crop is None:
                    continue
                emb = _embed_face(enc, crop, device)
                sims.append(float((emb * ref).sum().item()))
    if not sims:
        return None
    return float(np.mean(sims))


def _identity_region_mask(
    boxes: list, resolution: int, emphasis: float = 0.3
) -> torch.Tensor:
    """Soft elliptical emphasis over the face region (Phantom spatial
    constraint, simplified): perturbation stays strongest on identity-relevant
    facial regions while the surroundings keep a floor fraction.

    Returns a [1,1,H,W] float mask in [floor, 1.0] with ``floor = 1 - emphasis``.
    """
    floor = max(0.4, 1.0 - emphasis)
    mask = torch.full((1, 1, resolution, resolution), floor, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(resolution, dtype=torch.float32),
        torch.arange(resolution, dtype=torch.float32),
        indexing="ij",
    )
    for box in boxes:
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(1.0, (x1 - x0) * 0.75)
        ry = max(1.0, (y1 - y0) * 0.9)
        d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        ellipse = torch.clamp(1.0 - d2, 0.0, 1.0)
        mask = torch.maximum(mask, floor + (1.0 - floor) * ellipse)
    return mask


def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] [0,1] tensor -> uint8 RGB numpy array."""
    return (
        t.float().squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
    ).astype(np.uint8)


def _get_arcface(device: str):
    """Registry ArcFace w600k_mbf wrapper (ONNX, black-box) or None."""
    try:
        from ..models.face_models import get_registry  # noqa: PLC0415

        reg = get_registry(device)
        model = reg._models.get("arcface_mbf")
        if model is not None and model.info.loaded and model.onnx_session is not None:
            return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("ArcFace unavailable: %s", exc)
    return None


def _arcface_similarity(
    image_rgb: np.ndarray, original_rgb: np.ndarray, face_boxes: list, device: str
) -> float | None:
    """Mean cosine similarity of ArcFace embeddings vs the original identity.

    Measured on square crops (2.2x margin, 112px) — the exact embedding family
    real swap pipelines (SimSwap IIM / INSwapper) use for identity transfer.
    Lower = the protected face is less useful as the swap identity source.
    """
    arcface = _get_arcface(device)
    if arcface is None or not face_boxes:
        return None
    try:
        from ..utils.boxes import expand_box, numpy_crops  # noqa: PLC0415

        h, w = image_rgb.shape[:2]
        boxes = [expand_box(b, 2.2, h, w) for b in face_boxes]
        embs = arcface.embed_crops(numpy_crops(image_rgb, boxes, arcface.input_size), "cpu")
        orig_embs = arcface.embed_crops(numpy_crops(original_rgb, boxes, arcface.input_size), "cpu")
        sims = [
            float(np.dot(e, eo) / (np.linalg.norm(e) * np.linalg.norm(eo) + 1e-12))
            for e, eo in zip(embs, orig_embs)
        ]
        return float(np.mean(sims)) if sims else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ArcFace similarity failed: %s", exc)
        return None


def _refine_identity_blackbox(
    original_rgb: np.ndarray,
    protected_rgb: np.ndarray,
    face_boxes: list,
    eps: float,
    device: str,
    progress=None,
) -> tuple[np.ndarray | None, dict]:
    """In-place zeroth-order ArcFace identity refinement of the SAME δ.

    ArcFace w600k_mbf (ONNX) is the identity-encoder family actual swap
    pipelines use; it has no gradient path through PyTorch, so the refinement
    estimates the identity-distance gradient by central finite differences over
    random directions (sign-SGD), keeping every update inside the same epsilon
    bound as the PGD. This continues optimizing the identical perturbation — it
    is NOT a second stacked perturbation.

    Returns ``(refined, report)``; ``refined`` is None when nothing could be
    done (models missing / already strong / quality guard tripped).
    """
    arcface = _get_arcface(device)
    if arcface is None or not face_boxes:
        return None, {"applied": False, "note": "ArcFace refinement unavailable (model not loaded)."}
    try:
        from ..quality.metrics import ssim as _ssim_np  # noqa: PLC0415
        from ..robustness.tester import _apply_transform  # noqa: PLC0415
        from ..utils.boxes import expand_box, numpy_crops  # noqa: PLC0415

        h, w = original_rgb.shape[:2]
        crop_boxes = [expand_box(b, 2.0, h, w) for b in face_boxes]
        size = arcface.input_size

        # reference distances are measured against the ORIGINAL crops' identity
        orig_embs = arcface.embed_crops(numpy_crops(original_rgb, crop_boxes, size), "cpu")

        def _dist(img: np.ndarray) -> float:
            embs = arcface.embed_crops(numpy_crops(img, crop_boxes, size), "cpu")
            return float(
                np.mean(
                    [np.linalg.norm(e - eo) for e, eo in zip(embs, orig_embs)]
                )
            )

        before = _dist(protected_rgb)
        goal = settings.EDITING_IDENTITY_REFINE_GOAL  # ArcFace L2 target
        if before >= goal:
            return None, {
                "applied": False,
                "note": f"ArcFace identity already disrupted (L2 {before:.3f} ≥ goal {goal}).",
            }

        if device == "cuda":
            max_iters = settings.EDITING_IDENTITY_REFINE_ITERS_GPU
            k_dirs = settings.EDITING_IDENTITY_REFINE_DIRECTIONS_GPU
        else:
            max_iters = settings.EDITING_IDENTITY_REFINE_ITERS_CPU
            k_dirs = settings.EDITING_IDENTITY_REFINE_DIRECTIONS_CPU

        eps_px = eps * 255.0
        perturb_mag = 6.0
        lr = 1.4
        rng = np.random.default_rng(settings.OPT_SEED)

        # soft elliptical emphasis mask at full resolution (Phantom spatial)
        emphasis = settings.EDITING_IDENTITY_REGION_EMPHASIS
        floor = max(0.4, 1.0 - emphasis)
        mask = np.full((h, w), floor, dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for b in face_boxes:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            rx = max(1.0, (b[2] - b[0]) * 0.9)
            ry = max(1.0, (b[3] - b[1]) * 1.0)
            d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            mask = np.maximum(mask, floor + (1.0 - floor) * np.clip(1.0 - d2, 0.0, 1.0))

        cur = protected_rgb.astype(np.float32)
        orig_f = original_rgb.astype(np.float32)
        best_dist = before
        best_img = cur.copy()
        stalled = 0

        transform_cycle = iter(["jpeg_compression", "resize", "brightness"])

        for it in range(max_iters):
            if progress is not None:
                try:
                    progress({"iteration": it + 1, "total": max_iters, "phase": "refine", "loss": before})
                except Exception:  # noqa: BLE001
                    pass

            grad = np.zeros_like(cur)
            mask3 = mask[..., None]  # (h,w,1)
            for _ in range(k_dirs):
                u = (
                    (rng.integers(0, 2, size=(h, w, 1)).astype(np.float32) * 2 - 1)
                    * mask3
                    * perturb_mag
                )
                u = np.repeat(u, 3, axis=2)
                pos = np.clip(cur + u, 0.0, 255.0)
                neg = np.clip(cur - u, 0.0, 255.0)
                dpos = _dist(pos.astype(np.uint8))
                dneg = _dist(neg.astype(np.uint8))
                grad += (dpos - dneg) * u / (np.linalg.norm(u) + 1e-8)

            cand = cur + lr * np.sign(grad)
            delta = np.clip(cand - orig_f, -eps_px, eps_px) * mask3
            cand = np.clip(orig_f + delta, 0.0, 255.0)

            # quality guard: never visibly damage the image
            if _ssim_np(original_rgb, np.clip(cand, 0, 255).astype(np.uint8)) < settings.EDITING_MIN_SSIM:
                lr *= 0.5
                continue

            cur = cand
            d = _dist(np.clip(cur, 0, 255).astype(np.uint8))

            # transform-aware identity check (interference layer): the
            # disruption must also survive re-encoding of the protected image
            if it % 3 == 0:
                try:
                    tname = next(transform_cycle)
                    t = _apply_transform(tname, np.clip(cur, 0, 255).astype(np.uint8))
                    d = min(d, _dist(t))
                except Exception:  # noqa: BLE001
                    pass

            if d > best_dist:
                best_dist = d
                best_img = cur.copy()
                stalled = 0
                if d >= goal:
                    break
            else:
                stalled += 1
                if stalled >= max(3, max_iters // 4):
                    break

        after = _dist(np.clip(best_img, 0, 255).astype(np.uint8))
        if after <= before + 1e-6:
            return None, {
                "applied": False,
                "note": f"ArcFace refinement made no progress (L2 {before:.3f} → {after:.3f}).",
            }
        return (
            np.clip(best_img, 0, 255).astype(np.uint8),
            {
                "applied": True,
                "iters": max_iters,
                "l2_before": round(float(before), 4),
                "l2_after": round(float(after), 4),
                "note": (
                    f"In-place ArcFace identity refinement: L2 distance {before:.3f} -> "
                    f"{after:.3f} within the same epsilon bound."
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ArcFace refinement failed: %s", exc)
        return None, {"applied": False, "note": f"ArcFace refinement failed: {type(exc).__name__}"}


def apply_editing_protection(
    image_rgb: np.ndarray,
    device: str,
    surrogate: AntiDiffusionSurrogate | None = None,
    progress=None,
    face_boxes: list | None = None,
    weights: dict | None = None,
) -> EditingProtectionResult:
    """Optimize ONE perturbation against multiple attack families.

    ``face_boxes`` are xyxy boxes in the FULL-RES image (from the face
    detector) and drive the identity-reference / face-swap term. ``weights``
    optionally overrides the per-family term weights (used by the adaptive
    red-team loop); keys: ``diffusion``, ``identity``, ``vision``.
    """
    if not settings.EDITING_ENABLED or not settings.EDITING_SURROGATE_ENABLED:
        return EditingProtectionResult(applied=False, note="AI-editing protection disabled by configuration.")

    resolution = settings.EDITING_SURROGATE_RESOLUTION
    timestep = settings.EDITING_SURROGATE_TIMESTEP
    iters = settings.EDITING_SURROGATE_ITERS_GPU if device == "cuda" else settings.EDITING_SURROGATE_ITERS_CPU
    lr = settings.EDITING_SURROGATE_LR

    eps = settings.PERTURBATION_EPSILON * settings.EDITING_SURROGATE_EPSILON_FRACTION
    min_ssim = settings.EDITING_MIN_SSIM

    w_diff = float((weights or {}).get("diffusion", 1.0))
    w_id = float((weights or {}).get("identity", settings.EDITING_IDENTITY_WEIGHT))
    w_vis = float((weights or {}).get("vision", settings.EDITING_VISION_WEIGHT))

    manager = get_editing_manager(device)
    own_surrogate = surrogate is None
    if own_surrogate:
        surrogate = manager.get_surrogate(resolution=resolution, timestep=timestep)

    try:
        if not surrogate.loaded:
            surrogate.load()
    except Exception as exc:
        logger.warning("Anti-diffusion surrogate unavailable: %s", exc)
        return EditingProtectionResult(
            applied=False,
            note=f"Anti-diffusion surrogate unavailable on this hardware ({type(exc).__name__}).",
        )

    result = EditingProtectionResult(
        applied=True,
        epsilon=eps,
        resolution=resolution,
        timestep=timestep,
        surrogate_model=surrogate.model_id,
        families=["diffusion_editing", "instruction_editing", "inpainting", "image_to_image"],
        weights={"diffusion": w_diff, "identity": w_id, "vision": w_vis},
    )

    try:
        # ---- multi-family terms (lazy; each may be unavailable) -----------
        terms = _MultiFamilyTerms(device)
        encoders = terms._identity_models() if settings.EDITING_IDENTITY_ENABLED and face_boxes else []
        vision_model = terms._vision() if settings.EDITING_VISION_ENABLED else None
        if encoders:
            result.families += ["identity_reference", "face_swap"]
        if vision_model is not None:
            result.families += ["vision_encoder"]

        x0 = _rgb_to_tensor(image_rgb, device)
        x0_r = _rgb_to_tensor(image_rgb, device, resolution)
        kernel = _gaussian_kernel().half().to(device).repeat(3, 1, 1, 1)

        h, w = image_rgb.shape[:2]
        sx, sy = resolution / w, resolution / h
        scaled_boxes = [((b[0] * sx), (b[1] * sy), (b[2] * sx), (b[3] * sy)) for b in (face_boxes or [])]

        # Phantom-style spatial constraint: a soft elliptical emphasis keeps
        # the perturbation concentrated on identity-relevant facial regions.
        identity_mask = _identity_region_mask(
            scaled_boxes, resolution, settings.EDITING_IDENTITY_REGION_EMPHASIS
        ).half().to(device)

        # reference embeddings per encoder (original identity / representation)
        refs: list[list] = []
        if encoders:
            with torch.no_grad():
                for enc in encoders:
                    enc_refs = []
                    for box in scaled_boxes:
                        crop = _crop_tensor(x0_r, box, 160)
                        if crop is not None:
                            enc_refs.append(_embed_face(enc, crop, device).squeeze(0))
                    refs.append(enc_refs)
        vision_ref = None
        if vision_model is not None:
            with torch.no_grad():
                from ..models.face_models import imagenet_preprocess_torch  # noqa: PLC0415

                x224 = F.interpolate(x0_r, (224, 224), mode="bilinear", antialias=True).float()
                emb = vision_model(imagenet_preprocess_torch(x224).to(device))
                vision_ref = F.normalize(emb, dim=1).squeeze(0)

        with torch.no_grad():
            base_loss = float(surrogate.denoising_loss(surrogate.encode(x0_r.detach())))
        result.denoising_loss_before = base_loss

        # before values at full resolution (honest, matches final output)
        with torch.no_grad():
            if encoders and refs:
                result.identity_similarity_before = _identity_similarity(
                    encoders, x0, face_boxes, refs, device
                )
            if vision_model is not None and vision_ref is not None:
                from ..models.face_models import imagenet_preprocess_torch  # noqa: PLC0415

                x224 = F.interpolate(x0, (224, 224), mode="bilinear", antialias=True).float()
                emb = vision_model(imagenet_preprocess_torch(x224).to(device))
                result.vision_similarity_before = float(
                    (F.normalize(emb, dim=1).squeeze(0) * vision_ref).sum().item()
                )

        # ArcFace (w600k) similarity before — the encoder family real swap
        # pipelines use; measured at full resolution via the ONNX session.
        result.arcface_similarity_before = _arcface_similarity(
            image_rgb, image_rgb, face_boxes, device
        )

        x_adv = x0_r.clone().requires_grad_(True)
        best_total = -1e18
        best = (base_loss, None)
        best_iter = 0
        stalled = 0
        # pure identity steps: the image-wide diffusion gradient (sign-SGD)
        # otherwise swamps the face-region identity gradient, leaving the
        # embedding nearly untouched. These iterations apply ONLY the identity
        # objective to the face region.
        identity_iters = max(1, int(round(iters * settings.EDITING_IDENTITY_ATTACK_FRACTION)))

        for it in range(iters):
            if progress is not None:
                try:
                    progress({"iteration": it + 1, "total": iters, "phase": "editing", "loss": base_loss})
                except Exception:  # noqa: BLE001
                    pass

            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            identity_step = it < identity_iters and bool(encoders) and bool(refs)
            if not identity_step:
                for var_x in _variants(x_adv):
                    total_loss = total_loss - w_diff * surrogate.denoising_loss(
                        surrogate.encode(var_x)
                    ).float()

            if encoders and refs:
                try:
                    id_scale = w_id * (2.0 if identity_step else 1.0)
                    for enc, enc_refs in zip(encoders, refs):
                        for box, ref in zip(scaled_boxes, enc_refs):
                            crop = _crop_tensor(x_adv, box, 160)
                            if crop is None:
                                continue
                            emb = _embed_face(enc, crop, device)
                            total_loss = total_loss - id_scale * (emb * ref).sum()
                            if settings.EDITING_IDENTITY_TRANSFORMS:
                                # ID-Eraser interference-layer: the identity
                                # disruption must survive re-encoding, so it is
                                # also scored on transformed crops.
                                for var_crop in _variants(crop)[1:4]:
                                    vemb = _embed_face(enc, var_crop, device)
                                    total_loss = total_loss - id_scale * 0.25 * (vemb * ref).sum()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Identity loss term failed at iter %d: %s", it, exc)
                    encoders = []

            if vision_model is not None and vision_ref is not None:
                try:
                    from ..models.face_models import imagenet_preprocess_torch  # noqa: PLC0415

                    x224 = F.interpolate(x_adv, (224, 224), mode="bilinear", antialias=True).float()
                    emb = vision_model(imagenet_preprocess_torch(x224).to(device))
                    total_loss = total_loss - w_vis * (emb * vision_ref).sum()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Vision loss term failed at iter %d: %s", it, exc)
                    vision_model = None

            grad = torch.autograd.grad(total_loss, x_adv)[0]

            with torch.no_grad():
                cand = (x_adv - lr * torch.sign(grad)).clamp(0.0, 1.0)
                delta = (cand - x0_r).clamp(-eps, eps)
                delta = delta - F.conv2d(delta, kernel, padding=4, groups=3)
                # spatial emphasis: full budget on identity regions, a floor
                # elsewhere (Phantom spatial constraint, kept mild for quality)
                delta = delta * identity_mask
                x_adv = (x0_r + delta).clamp(0.0, 1.0)
                if torch_ssim(x_adv, x0_r) < min_ssim:
                    x_adv = (x_adv + lr * torch.sign(grad) * 0.5).clamp(0.0, 1.0)
                cur_denoise = float(surrogate.denoising_loss(surrogate.encode(x_adv.detach())))

                identity_score = 0.0
                if encoders and refs:
                    try:
                        sims = _identity_similarity(encoders, x_adv, scaled_boxes, refs, device)
                        if sims is not None:
                            identity_score = 1.0 - sims
                    except Exception:  # noqa: BLE001
                        pass
                vision_score = 0.0
                if vision_model is not None and vision_ref is not None:
                    try:
                        from ..models.face_models import imagenet_preprocess_torch  # noqa: PLC0415

                        x224 = F.interpolate(x_adv, (224, 224), mode="bilinear", antialias=True).float()
                        emb = vision_model(imagenet_preprocess_torch(x224).to(device))
                        sim = float((F.normalize(emb, dim=1).squeeze(0) * vision_ref).sum().item())
                        vision_score = 1.0 - sim
                    except Exception:  # noqa: BLE001
                        pass
                # During pure identity steps, select candidates by identity
                # disruption ONLY — otherwise the (much larger) denoising score
                # rejects every identity gain, which is exactly why the old
                # single-objective selection never moved the embedding.
                if identity_step:
                    cur_total = w_id * identity_score
                else:
                    cur_total = cur_denoise + w_id * identity_score + w_vis * vision_score

            x_adv = x_adv.detach().requires_grad_(True)
            if cur_total > best_total:
                best_total = cur_total
                best = (cur_denoise, x_adv.detach().clone())
                best_iter = it + 1
                stalled = 0
            else:
                stalled += 1
                if stalled >= max(5, iters // 3):
                    break

        result.iterations = best_iter if best_iter else iters
        best_x = best[1] if best[1] is not None else x_adv.detach()
        result.denoising_loss_after = best[0]
        if best[0] > base_loss > 1e-9:
            result.loss_increase_pct = (best[0] - base_loss) / base_loss * 100.0

        delta_full = TF.resize((best_x - x0_r).float(), image_rgb.shape[:2], antialias=True).half()
        protected = (x0 + delta_full).clamp(0.0, 1.0)

        # ---- in-place ArcFace refinement (same δ, no stacking) -----------
        # ID-Eraser identity-space concept applied to the ACTUAL encoder family
        # swap pipelines use: continue optimizing the SAME bounded perturbation
        # with zeroth-order ArcFace gradients (ONNX, black-box).
        refine_report = {"applied": False}
        if settings.EDITING_IDENTITY_REFINE_ENABLED:
            prot_np = (
                protected.float().squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            refined_np, refine_report = _refine_identity_blackbox(
                image_rgb, prot_np, face_boxes, eps, device, progress=progress
            )
            result.identity_refinement = refine_report
            if refined_np is not None:
                protected = _rgb_to_tensor(refined_np, device)

        with torch.no_grad():
            prot_r = TF.resize(protected, (resolution, resolution), antialias=True)
            verified = float(surrogate.denoising_loss(surrogate.encode(prot_r)))
            if encoders and refs:
                result.identity_similarity_after = _identity_similarity(
                    encoders, protected, face_boxes, refs, device
                )
            result.arcface_similarity_after = _arcface_similarity(
                _tensor_to_rgb(protected), image_rgb, face_boxes, device
            )
            if vision_model is not None and vision_ref is not None:
                from ..models.face_models import imagenet_preprocess_torch  # noqa: PLC0415

                x224 = F.interpolate(protected, (224, 224), mode="bilinear", antialias=True).float()
                emb = vision_model(imagenet_preprocess_torch(x224).to(device))
                result.vision_similarity_after = float(
                    (F.normalize(emb, dim=1).squeeze(0) * vision_ref).sum().item()
                )
        result.verified_loss_after = verified
        if verified > base_loss > 1e-9:
            result.verified_increase_pct = (verified - base_loss) / base_loss * 100.0

        from ..quality.metrics import ssim as _ssim_np  # noqa: PLC0415

        arr = (protected.float().squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        ssim_val = float(_ssim_np(image_rgb, arr))

        if ssim_val < min_ssim:
            result.reverted = True
            result.applied = False
            result.note = (
                f"Editing-protection pass reverted: SSIM {ssim_val:.3f} fell below the "
                f"{min_ssim:.2f} quality floor."
            )
            return result

        result.protected = arr
        fam_names = {
            "diffusion_editing": "diffusion editing",
            "instruction_editing": "instruction editing",
            "inpainting": "inpainting",
            "image_to_image": "image-to-image",
            "identity_reference": "identity-reference",
            "face_swap": "face-swap",
            "vision_encoder": "vision encoders",
        }
        fam_txt = ", ".join(fam_names.get(f, f) for f in result.families)
        parts = [
            f"single multi-family perturbation applied (eps={eps:.4f}, {result.iterations} iters)",
            f"denoising error {base_loss:.3f} -> {best[0]:.3f}",
            f"SSIM {ssim_val:.3f}",
        ]
        if result.identity_similarity_before is not None and result.identity_similarity_after is not None:
            parts.append(
                f"identity similarity {result.identity_similarity_before:.3f} -> "
                f"{result.identity_similarity_after:.3f}"
            )
        if result.arcface_similarity_before is not None and result.arcface_similarity_after is not None:
            parts.append(
                f"ArcFace w600k similarity {result.arcface_similarity_before:.3f} -> "
                f"{result.arcface_similarity_after:.3f} (swap-encoder family)"
            )
        if result.identity_refinement.get("applied"):
            parts.append(
                f"in-place ArcFace refinement: L2 {result.identity_refinement.get('l2_before', 0):.3f} -> "
                f"{result.identity_refinement.get('l2_after', 0):.3f}"
            )
        if result.vision_similarity_before is not None and result.vision_similarity_after is not None:
            parts.append(
                f"vision similarity {result.vision_similarity_before:.3f} -> "
                f"{result.vision_similarity_after:.3f}"
            )
        result.note = f"Families targeted: {fam_txt}. " + "; ".join(parts) + "."
        return result
    except Exception as exc:
        logger.exception("Editing-protection pass failed")
        result.applied = False
        result.note = f"Editing-protection pass failed: {type(exc).__name__}"
        return result

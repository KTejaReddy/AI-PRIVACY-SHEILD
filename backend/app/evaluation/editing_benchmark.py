"""AI-editing benchmark: original vs protected through the same local editors.

For every task the *exact same* edit is run on the original photo and on the
protected photo — same editing model, same instruction, same seed, same
resolution, same inference steps, same guidance. Only the input image changes.

Editors (all evaluation-only; the SD1.5 anti-diffusion surrogate is the only
optimization model and is never used here):

  * InstructPix2Pix       — instruction-guided editing (held out from optimization)
  * SD1.5 inpainting      — masked inpainting with real region masks derived from
                            the detected face/person boxes (never a mock mask)
  * SD1.5 image-to-image  — text-guided regeneration / style transfer

Scoring — primary evidence is a **task-specific pixel metric** computed in the
region the edit should affect (see evaluation/task_metrics.py). CLIP cosine
alignment is only an auxiliary semantic check:

    success = W_TASK * task_metric(0..1)
            + W_CLIP * clip( (cos(edit, target) - cos(input, target)) / scale, 0, 1 )

Change reporting is honest: we report the absolute change and a relative change
*only when it is meaningful* (the original score is non-trivial). The old
practice of reporting a percentage that could exceed 100% when the metric
crossed zero is gone — the raw CLIP deltas remain visible so nothing is hidden.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..config import settings
from ..editing.adapter import EditRequest
from ..editing.manager import get_editing_manager
from ..editing.tasks import EditingTask, tasks_for
from . import task_metrics

logger = logging.getLogger(__name__)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _relative_change(orig: float, prot: float) -> Optional[float]:
    """Relative change % when meaningful, else None.

    ``None`` means the relative change is not reported: the original success
    was too close to zero for a percentage to be meaningful.
    """
    if orig >= 0.02:
        return (orig - prot) / orig * 100.0
    return None


@dataclass
class SeedStats:
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0

    @classmethod
    def of(cls, values: list[float]) -> "SeedStats":
        if not values:
            return cls()
        return cls(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values)) if len(values) > 1 else 0.0,
            min=float(np.min(values)),
            max=float(np.max(values)),
        )

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
        }


@dataclass
class TaskResult:
    task_id: str
    name: str
    editor_type: str
    instruction: str
    target: str
    mask_kind: str | None
    # primary evidence: task-specific pixel metric (raw)
    task_metric_original: float
    task_metric_protected: float
    # auxiliary evidence: CLIP cosine alignment deltas (raw, signed)
    clip_delta_original: float
    clip_delta_protected: float
    # composite success in [0, 1] (documented formula)
    success_original: float
    success_protected: float
    # honest change reporting
    absolute_change: float
    relative_change_pct: Optional[float]
    # secondary metrics
    semantic_preservation_original: float
    semantic_preservation_protected: float
    edit_magnitude_original: float
    edit_magnitude_protected: float
    protected_region_change_original: float
    protected_region_change_protected: float
    # aggregation over seeds / prompt variants
    samples: int = 1
    stats_original: SeedStats = field(default_factory=SeedStats)
    stats_protected: SeedStats = field(default_factory=SeedStats)
    stats_absolute: SeedStats = field(default_factory=SeedStats)

    def as_dict(self) -> dict:
        rel = None if self.relative_change_pct is None else round(self.relative_change_pct, 1)
        return {
            "id": self.task_id,
            "name": self.name,
            "editor_type": self.editor_type,
            "instruction": self.instruction,
            "target": self.target,
            "mask_kind": self.mask_kind,
            "task_metric_original": round(self.task_metric_original, 4),
            "task_metric_protected": round(self.task_metric_protected, 4),
            "clip_delta_original": round(self.clip_delta_original, 4),
            "clip_delta_protected": round(self.clip_delta_protected, 4),
            "success_original": round(self.success_original, 4),
            "success_protected": round(self.success_protected, 4),
            "absolute_change": round(self.absolute_change, 4),
            "relative_change_pct": rel,
            "semantic_preservation_original": round(self.semantic_preservation_original, 4),
            "semantic_preservation_protected": round(self.semantic_preservation_protected, 4),
            "edit_magnitude_original": round(self.edit_magnitude_original, 4),
            "edit_magnitude_protected": round(self.edit_magnitude_protected, 4),
            "protected_region_change_original": round(self.protected_region_change_original, 4),
            "protected_region_change_protected": round(self.protected_region_change_protected, 4),
            "samples": self.samples,
            "stats": {
                "success_original": self.stats_original.as_dict(),
                "success_protected": self.stats_protected.as_dict(),
                "absolute_change": self.stats_absolute.as_dict(),
            },
        }


@dataclass
class EditingBenchmarkResult:
    available: bool
    tasks: list[TaskResult] = field(default_factory=list)
    robustness: list[dict] = field(default_factory=list)
    mean_original: float = 0.0
    mean_protected: float = 0.0
    mean_absolute_change: float = 0.0
    mean_relative_change_pct: Optional[float] = None
    tasks_reduced: int = 0
    tasks_total: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "settings": {
                "resolution": settings.EDITING_RESOLUTION,
                "steps": settings.EDITING_STEPS,
                "guidance_scale": settings.EDITING_GUIDANCE,
                "seed": settings.EDITING_SEED,
                "task_metric_weight": settings.W_EDITING_TASK_METRIC,
                "clip_weight": settings.W_EDITING_CLIP,
                "clip_delta_scale": settings.EDITING_CLIP_DELTA_SCALE,
            },
            "tasks": [t.as_dict() for t in self.tasks],
            "robustness": self.robustness,
            "aggregate": {
                "mean_original": round(self.mean_original, 4),
                "mean_protected": round(self.mean_protected, 4),
                "mean_absolute_change": round(self.mean_absolute_change, 4),
                "mean_relative_change_pct": (
                    None if self.mean_relative_change_pct is None else round(self.mean_relative_change_pct, 1)
                ),
                "tasks_reduced": self.tasks_reduced,
                "tasks_total": self.tasks_total,
                # legacy key kept for compatibility (equals relative change)
                "mean_reduction_pct": (
                    None if self.mean_relative_change_pct is None else round(self.mean_relative_change_pct, 1)
                ),
            },
            "note": self.note,
        }


def _l2_norm_diff(a: np.ndarray, b: np.ndarray) -> float:
    b = _resize_to(b, a.shape[:2])
    diff = (a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=2))))


def _resize_to(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if arr.shape[:2] == size:
        return arr
    return np.asarray(
        Image.fromarray(arr.astype(np.uint8)).convert("RGB").resize((size[1], size[0]), Image.LANCZOS)
    )


def _apply_edit_transform(name: str, arr: np.ndarray) -> np.ndarray:
    """Real image transformations (identical to the recognition-robustness set)."""
    from ..robustness.tester import _apply_transform  # noqa: PLC0415

    return _apply_transform(name, arr)


class EditingBenchmark:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.manager = get_editing_manager(device)

    # ------------------------------------------------------------------
    def run(
        self,
        original_rgb: np.ndarray,
        protected_rgb: np.ndarray,
        face_boxes=None,
        person_boxes=None,
        task_ids: tuple[str, ...] | None = None,
        seeds: list[int] | None = None,
        prompt_variants: bool = False,
        strengths: list[float] | None = None,
        progress=None,
    ) -> EditingBenchmarkResult:
        if not settings.EDITING_ENABLED or not settings.EDITING_BENCHMARK_ENABLED:
            return EditingBenchmarkResult(available=False, note="AI-editing benchmark disabled by configuration.")

        tasks = tasks_for(task_ids if task_ids is not None else settings.EDITING_TASKS)
        if not tasks:
            return EditingBenchmarkResult(available=True, note="No editing tasks matched.")

        seeds = seeds or [settings.EDITING_SEED]
        try:
            clip = self.manager.get_clip(settings.EDITING_CLIP_REPO)
            processor = self.manager.clip_processor
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP scorer unavailable: %s", exc)
            return EditingBenchmarkResult(
                available=False,
                note=f"AI-editing benchmark unavailable: the CLIP scorer could not be loaded ({type(exc).__name__}).",
            )

        samples: list[tuple[EditingTask, str, str | None, dict]] = []  # (task, editor, mask_kind, fields)
        total_edits = 0

        # ---- editor 1: InstructPix2Pix (instruction-guided, held out) ----
        if settings.EDITING_IP2P_ENABLED:
            try:
                ip2p = self.manager.get_ip2p(settings.EDITING_IP2P_REPO)
                for task in tasks:
                    prompts = task.prompt_variants if prompt_variants and task.prompt_variants else [task.instruction]
                    for prompt in prompts:
                        for seed in seeds:
                            total_edits += 1
                for task in tasks:
                    prompts = task.prompt_variants if prompt_variants and task.prompt_variants else [task.instruction]
                    for prompt in prompts:
                        for seed in seeds:
                            if progress:
                                self._progress(progress, total_edits, f"instruction · {task.name}")
                            common = dict(
                                instruction=prompt, seed=seed,
                                resolution=settings.EDITING_RESOLUTION, num_inference_steps=settings.EDITING_STEPS,
                                guidance_scale=settings.EDITING_GUIDANCE,
                                image_guidance_scale=settings.EDITING_IMAGE_GUIDANCE,
                            )
                            out_orig = ip2p.edit(EditRequest(image_rgb=original_rgb, **common))
                            out_prot = ip2p.edit(EditRequest(image_rgb=protected_rgb, **common))
                            samples.append((task, "instruction", None, self._score(
                                clip, processor, task, None,
                                original_rgb, out_orig, protected_rgb, out_prot,
                                face_boxes, person_boxes,
                            )))
            except Exception as exc:  # noqa: BLE001
                logger.warning("InstructPix2Pix evaluation failed: %s", exc)

        # ---- editor 2: masked inpainting (real region masks) --------------
        if settings.EDITING_INPAINTING_ENABLED:
            try:
                inpaint = self.manager.get_editor(
                    "stable-diffusion-v1-5/stable-diffusion-inpainting", "inpainting"
                )
                mask_tasks = [t for t in tasks if t.mask_kind]
                for task in mask_tasks:
                    for prompt in ([task.instruction] if not prompt_variants or not task.prompt_variants else task.prompt_variants):
                        for seed in seeds:
                            total_edits += 1
                for task in mask_tasks:
                    mask = task_metrics.make_mask(
                        task.mask_kind, original_rgb.shape[0], original_rgb.shape[1], face_boxes, person_boxes
                    )
                    prompts = task.prompt_variants if prompt_variants and task.prompt_variants else [task.instruction]
                    for prompt in prompts:
                        for seed in seeds:
                            if progress:
                                self._progress(progress, total_edits, f"inpainting · {task.name}")
                            common = dict(
                                instruction=prompt, seed=seed,
                                resolution=settings.EDITING_RESOLUTION, num_inference_steps=settings.EDITING_STEPS,
                                guidance_scale=settings.EDITING_GUIDANCE,
                                image_guidance_scale=settings.EDITING_IMAGE_GUIDANCE,
                                mask=mask,
                            )
                            out_orig = inpaint.edit(EditRequest(image_rgb=original_rgb, **common))
                            out_prot = inpaint.edit(EditRequest(image_rgb=protected_rgb, **common))
                            samples.append((task, "inpainting", task.mask_kind, self._score(
                                clip, processor, task, mask,
                                original_rgb, out_orig, protected_rgb, out_prot,
                                face_boxes, person_boxes,
                            )))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Inpainting evaluation failed: %s", exc)

        # ---- editor 3: image-to-image (SD1.5, style/global tasks) ---------
        if settings.EDITING_IMG2IMG_ENABLED:
            try:
                img2img = self.manager.get_editor(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5", "image2image"
                )
                i2i_tasks = [t for t in tasks if t.id in settings.EDITING_IMG2IMG_TASKS]
                strength_values = strengths or [settings.EDITING_IMG2IMG_STRENGTH]
                for task in i2i_tasks:
                    for prompt in ([task.instruction] if not prompt_variants or not task.prompt_variants else task.prompt_variants):
                        for seed in seeds:
                            for _s in strength_values:
                                total_edits += 1
                for task in i2i_tasks:
                    prompts = task.prompt_variants if prompt_variants and task.prompt_variants else [task.instruction]
                    for prompt in prompts:
                        for seed in seeds:
                            for strength in strength_values:
                                if progress:
                                    self._progress(progress, total_edits, f"image-to-image s={strength} · {task.name}")
                                common = dict(
                                    instruction=prompt, seed=seed,
                                    resolution=settings.EDITING_RESOLUTION, num_inference_steps=settings.EDITING_STEPS,
                                    guidance_scale=settings.EDITING_GUIDANCE,
                                    image_guidance_scale=settings.EDITING_IMAGE_GUIDANCE,
                                    strength=strength,
                                )
                                out_orig = img2img.edit(EditRequest(image_rgb=original_rgb, **common))
                                out_prot = img2img.edit(EditRequest(image_rgb=protected_rgb, **common))
                                samples.append((task, "image2image", None, self._score(
                                    clip, processor, task, None,
                                    original_rgb, out_orig, protected_rgb, out_prot,
                                    face_boxes, person_boxes,
                                )))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Image-to-image evaluation failed: %s", exc)

        if not samples:
            return EditingBenchmarkResult(
                available=False,
                note="No editing model produced results on this hardware. "
                "See docs/editing-protection.md for model installation.",
            )

        return self._aggregate(samples)

    # ------------------------------------------------------------------
    def run_robustness(
        self,
        original_rgb: np.ndarray,
        protected_rgb: np.ndarray,
        face_boxes=None,
        person_boxes=None,
        task_ids: tuple[str, ...] | None = None,
        transforms: tuple[str, ...] | None = None,
        progress=None,
    ) -> list[dict]:
        """Measure **edit success after real transformations** of the protected image.

        For each transform: protected -> transform -> same AI editor -> edit success,
        compared against the untransformed original. This is the meaningful
        robustness question for editing protection (not embedding distance).
        """
        if not settings.EDITING_ENABLED or not settings.EDITING_BENCHMARK_ENABLED or not settings.EDITING_ROBUSTNESS_ENABLED:
            return []
        transforms = transforms or settings.EDITING_ROBUSTNESS_TRANSFORMS
        tasks = tasks_for(task_ids if task_ids is not None else settings.EDITING_ROBUSTNESS_TASKS)
        if not tasks:
            return []

        try:
            ip2p = self.manager.get_ip2p(settings.EDITING_IP2P_REPO)
            clip = self.manager.get_clip(settings.EDITING_CLIP_REPO)
            processor = self.manager.clip_processor
        except Exception as exc:  # noqa: BLE001
            logger.warning("Editing robustness unavailable: %s", exc)
            return []

        results: list[dict] = []
        total = len(transforms) * len(tasks)
        done = 0
        for tname in transforms:
            try:
                variant = _apply_edit_transform(tname, protected_rgb)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transform %s failed: %s", tname, exc)
                results.append({"transform": tname, "error": str(exc), "tasks": []})
                continue
            rows = []
            for task in tasks:
                done += 1
                if progress:
                    self._progress(progress, total, f"robustness · {tname}")
                common = dict(
                    instruction=task.instruction, seed=settings.EDITING_SEED,
                    resolution=settings.EDITING_RESOLUTION, num_inference_steps=settings.EDITING_STEPS,
                    guidance_scale=settings.EDITING_GUIDANCE,
                    image_guidance_scale=settings.EDITING_IMAGE_GUIDANCE,
                )
                out_orig = ip2p.edit(EditRequest(image_rgb=original_rgb, **common))
                out_var = ip2p.edit(EditRequest(image_rgb=variant, **common))
                s_orig = self._composite(clip, processor, task, None, original_rgb, out_orig, face_boxes, person_boxes)
                s_var = self._composite(clip, processor, task, None, variant, out_var, face_boxes, person_boxes)
                rows.append({
                    "task_id": task.id,
                    "name": task.name,
                    "success_original": round(s_orig, 4),
                    "success_after_transform": round(s_var, 4),
                })
            mean_orig = float(np.mean([r["success_original"] for r in rows])) if rows else 0.0
            mean_var = float(np.mean([r["success_after_transform"] for r in rows])) if rows else 0.0
            results.append({
                "transform": tname,
                "tasks": rows,
                "mean_success_original": round(mean_orig, 4),
                "mean_success_after_transform": round(mean_var, 4),
            })
        return results

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _score(
        self, clip, processor, task: EditingTask, mask: np.ndarray | None,
        in_orig: np.ndarray, out_orig: np.ndarray,
        in_prot: np.ndarray, out_prot: np.ndarray,
        face_boxes, person_boxes,
    ) -> dict:
        faces, persons = face_boxes or [], person_boxes or []

        # align resolutions: the editor output is the reference; inputs (and
        # masks) are resized to it so regions and masks line up pixel-for-pixel
        out_h, out_w = out_orig.shape[:2]
        if in_orig.shape[:2] != (out_h, out_w):
            in_orig = _resize_to(in_orig, (out_h, out_w))
        if in_prot.shape[:2] != (out_h, out_w):
            in_prot = _resize_to(in_prot, (out_h, out_w))
        if mask is not None and mask.shape[:2] != (out_h, out_w):
            from PIL import Image  # noqa: PLC0415

            mask_img = Image.fromarray((mask * 255.0).round().astype(np.uint8)).resize((out_w, out_h), Image.LANCZOS)
            mask = np.asarray(mask_img).astype(np.float32) / 255.0

        # primary: task-specific pixel metric in the edited region
        t_raw_orig, t_succ_orig = task_metrics.measure(task.metric, in_orig, out_orig, faces, persons, mask)
        t_raw_prot, t_succ_prot = task_metrics.measure(task.metric, in_prot, out_prot, faces, persons, mask)

        # auxiliary: CLIP semantic alignment
        emb_in_orig, emb_out_orig = self._emb_pair(clip, processor, in_orig, out_orig, task.target)
        emb_in_prot, emb_out_prot = self._emb_pair(clip, processor, in_prot, out_prot, task.target)
        clip_orig = _cos(emb_out_orig[0], emb_out_orig[1]) - _cos(emb_in_orig[0], emb_in_orig[1])
        clip_prot = _cos(emb_out_prot[0], emb_out_prot[1]) - _cos(emb_in_prot[0], emb_in_prot[1])

        scale = settings.EDITING_CLIP_DELTA_SCALE
        w_task, w_clip = settings.W_EDITING_TASK_METRIC, settings.W_EDITING_CLIP
        succ_orig = w_task * t_succ_orig + w_clip * _clip01(clip_orig / scale)
        succ_prot = w_task * t_succ_prot + w_clip * _clip01(clip_prot / scale)

        return {
            "task_metric_original": t_raw_orig,
            "task_metric_protected": t_raw_prot,
            "clip_delta_original": clip_orig,
            "clip_delta_protected": clip_prot,
            "success_original": succ_orig,
            "success_protected": succ_prot,
            "semantic_preservation_original": _cos(emb_in_orig[0], emb_out_orig[0]),
            "semantic_preservation_protected": _cos(emb_in_prot[0], emb_out_prot[0]),
            "edit_magnitude_original": _l2_norm_diff(in_orig, out_orig),
            "edit_magnitude_protected": _l2_norm_diff(in_prot, out_prot),
            "protected_region_change_original": self._region_change(in_orig, out_orig, faces),
            "protected_region_change_protected": self._region_change(in_prot, out_prot, faces),
        }

    def _composite(self, clip, processor, task, mask, img_in, img_out, face_boxes, person_boxes) -> float:
        """Single-image composite success (used by the robustness pass)."""
        faces, persons = face_boxes or [], person_boxes or []
        if img_in.shape[:2] != img_out.shape[:2]:
            img_in = _resize_to(img_in, img_out.shape[:2])
        _, t_succ = task_metrics.measure(task.metric, img_in, img_out, faces, persons, mask)
        emb_in, emb_out = self._emb_pair(clip, processor, img_in, img_out, task.target)
        clip_delta = _cos(emb_out[0], emb_out[1]) - _cos(emb_in[0], emb_in[1])
        scale = settings.EDITING_CLIP_DELTA_SCALE
        return settings.W_EDITING_TASK_METRIC * t_succ + settings.W_EDITING_CLIP * _clip01(clip_delta / scale)

    def _emb_pair(self, clip, processor, img_a: np.ndarray, img_b: np.ndarray, text: str):
        from PIL import Image  # noqa: PLC0415

        def _emb(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            with torch.no_grad():
                inputs = processor(text=[text], images=pil, return_tensors="pt", padding=True).to(self.device)
                out = clip(**inputs)
            return (
                out.image_embeds.detach().cpu().numpy().reshape(-1),
                out.text_embeds.detach().cpu().numpy().reshape(-1),
            )

        return _emb(img_a), _emb(img_b)

    # ------------------------------------------------------------------
    def _aggregate(self, samples) -> EditingBenchmarkResult:
        from collections import defaultdict

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for task, editor_type, mask_kind, fields in samples:
            groups[(editor_type, task.id)].append((task, editor_type, mask_kind, fields))

        results: list[TaskResult] = []
        for (editor_type, task_id), group in groups.items():
            task, _, mask_kind, _ = group[0]
            fields = [g[3] for g in group]
            mean = lambda key: float(np.mean([f[key] for f in fields]))  # noqa: E731
            succ_orig = mean("success_original")
            succ_prot = mean("success_protected")
            abs_change = succ_orig - succ_prot
            rel = _relative_change(succ_orig, succ_prot)
            results.append(TaskResult(
                task_id=task.id, name=task.name, editor_type=editor_type,
                instruction=task.instruction, target=task.target, mask_kind=mask_kind,
                task_metric_original=mean("task_metric_original"),
                task_metric_protected=mean("task_metric_protected"),
                clip_delta_original=mean("clip_delta_original"),
                clip_delta_protected=mean("clip_delta_protected"),
                success_original=succ_orig, success_protected=succ_prot,
                absolute_change=abs_change, relative_change_pct=rel,
                semantic_preservation_original=mean("semantic_preservation_original"),
                semantic_preservation_protected=mean("semantic_preservation_protected"),
                edit_magnitude_original=mean("edit_magnitude_original"),
                edit_magnitude_protected=mean("edit_magnitude_protected"),
                protected_region_change_original=mean("protected_region_change_original"),
                protected_region_change_protected=mean("protected_region_change_protected"),
                samples=len(fields),
                stats_original=SeedStats.of([f["success_original"] for f in fields]),
                stats_protected=SeedStats.of([f["success_protected"] for f in fields]),
                stats_absolute=SeedStats.of([f["success_original"] - f["success_protected"] for f in fields]),
            ))

        # stable order: by editor then task id
        editor_order = {"instruction": 0, "inpainting": 1, "image2image": 2}
        results.sort(key=lambda r: (editor_order.get(r.editor_type, 9), r.task_id))

        mean_orig = float(np.mean([r.success_original for r in results])) if results else 0.0
        mean_prot = float(np.mean([r.success_protected for r in results])) if results else 0.0
        mean_abs = float(np.mean([r.absolute_change for r in results])) if results else 0.0
        valid_rel = [r.relative_change_pct for r in results if r.relative_change_pct is not None]
        mean_rel = float(np.mean(valid_rel)) if valid_rel else None
        reduced = sum(1 for r in results if r.success_protected < r.success_original)

        return EditingBenchmarkResult(
            available=True,
            tasks=results,
            mean_original=mean_orig,
            mean_protected=mean_prot,
            mean_absolute_change=mean_abs,
            mean_relative_change_pct=mean_rel,
            tasks_reduced=reduced,
            tasks_total=len(results),
            note=(
                f"Multi-model benchmark over {len(results)} task/editor rows "
                f"({len(samples)} controlled edits: same prompt, same seed, same settings, "
                "only the input image changes). Success = "
                f"{settings.W_EDITING_TASK_METRIC:.0%} task-specific pixel metric + "
                f"{settings.W_EDITING_CLIP:.0%} CLIP semantic alignment; relative change is "
                "reported only when the original success is meaningful."
            ),
        )

    def _region_change(self, img_in: np.ndarray, img_out: np.ndarray, face_boxes) -> float:
        if not face_boxes:
            return 0.0
        img_out = _resize_to(img_out, img_in.shape[:2])
        h, w = img_in.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for box in face_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            if min(w, x2) > max(0, x1) and min(h, y2) > max(0, y1):
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1.0
        if mask.sum() < 1:
            return 0.0
        diff = (img_in.astype(np.float32) - img_out.astype(np.float32)) / 255.0
        return float(np.sqrt(np.mean(np.sum(diff * diff * mask[..., None], axis=2)))) / float(mask.mean())

    @staticmethod
    def _progress(progress, total: int, label: str) -> None:
        try:
            progress({"phase": "editing", "message": label})
        except Exception:  # noqa: BLE001
            pass


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

"""Reproducible AI-editing benchmark for AI Privacy Shield.

Runs the project's primary objective end to end: for every image, apply the
anti-diffusion editing-protection layer, then run a controlled editing
benchmark across an ensemble of local editors (InstructPix2Pix, masked
inpainting, SD1.5 image-to-image) with:

  * multiple seeds      (mean/median/std/min/max per task)
  * prompt variants     (same task re-phrased several ways)
  * multiple masks      (masked-inpainting attacks over different regions)
  * transformations     (edit success measured *after* real JPEG/resize/crop/...)

Everything except the input image is identical between the original and the
protected run. Outputs: results.json, results.csv, report.md, report.html.

The edit-success metric is the documented composite
    success = 0.6 * task-specific pixel metric + 0.4 * CLIP alignment
with the raw components reported. Percentages are only shown when meaningful.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402

from app.config import settings  # noqa: E402
from app.editing.manager import get_editing_manager  # noqa: E402
from app.editing.protector import apply_editing_protection  # noqa: E402
from app.evaluation.editing_benchmark import EditingBenchmark  # noqa: E402
from app.quality.metrics import compute_quality  # noqa: E402
from app.utils.imaging import load_and_normalize  # noqa: E402

PROJECT_ROOT = BACKEND.parent
DATASET_DIR = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_FIXTURES = sorted(
    list((DATASET_DIR).glob("*.jpg")) + list((DATASET_DIR).glob("*.png"))
    + list((PROJECT_ROOT / "tests" / "fixtures").glob("*.jpg"))
    + list((PROJECT_ROOT / "tests" / "fixtures").glob("*.png"))
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _env_record() -> dict:
    import diffusers  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": _device(),
        "gpu": _gpu_name(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rec["vram_mb"] = int(props.total_memory / 2**20)
        rec["cuda_version"] = torch.version.cuda
    return rec


def run_one(
    path: Path,
    task_ids,
    seeds: list[int],
    prompt_variants: bool,
    masks: tuple[str, ...] | None,
    transforms: tuple[str, ...] | None,
    strengths: list[float] | None = None,
) -> dict:
    record = load_and_normalize(path.read_bytes(), len(path.read_bytes()))
    original = record.array

    mgr = get_editing_manager(_device())

    # ---- 1) multi-model editing protection -----------------------------
    t0 = time.perf_counter()
    edit_protect = apply_editing_protection(original, _device(), progress=lambda _i: None)
    t_protect = time.perf_counter() - t0
    protected = edit_protect.protected if edit_protect.applied else original.copy()

    # ---- 2) editing benchmark (original vs protected) -------------------
    bench = EditingBenchmark(_device())
    t0 = time.perf_counter()
    result = bench.run(
        original,
        protected,
        task_ids=task_ids,
        seeds=seeds,
        prompt_variants=prompt_variants,
        strengths=strengths,
        progress=lambda _i: None,
    )
    t_bench = time.perf_counter() - t0

    # ---- 3) edit success under real transformations ----------------------
    t0 = time.perf_counter()
    robustness = bench.run_robustness(
        original, protected, task_ids=task_ids, transforms=transforms, progress=lambda _i: None
    )
    t_rob = time.perf_counter() - t0
    result.robustness = robustness  # included in as_dict()

    quality = compute_quality(original, protected, device=_device())

    return {
        "image": path.name,
        "dimensions": {"width": record.width, "height": record.height},
        "editing_protection": edit_protect.as_dict(),
        "editing_protection_seconds": round(t_protect, 1),
        "benchmark": result.as_dict(),
        "benchmark_seconds": round(t_bench, 1),
        "robustness_seconds": round(t_rob, 1),
        "visual_quality": {
            "ssim": round(quality.ssim, 4),
            "psnr_db": round(quality.psnr_db, 2),
            "perturbation_linf": round(quality.perturbation_linf, 4),
        },
    }


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def _rel(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}%"


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"


def csv_output(payload: dict, out_dir: Path) -> None:
    rows = []
    for img in payload["images"]:
        for t in img["benchmark"].get("tasks") or []:
            rows.append({
                "image": img["image"],
                "editor": t["editor_type"],
                "task": t["id"],
                "task_name": t["name"],
                "mask_kind": t.get("mask_kind") or "",
                "success_original": t["success_original"],
                "success_protected": t["success_protected"],
                "absolute_change": t["absolute_change"],
                "relative_change_pct": "" if t["relative_change_pct"] is None else t["relative_change_pct"],
                "task_metric_original": t["task_metric_original"],
                "task_metric_protected": t["task_metric_protected"],
                "clip_delta_original": t["clip_delta_original"],
                "clip_delta_protected": t["clip_delta_protected"],
                "samples": t["samples"],
                "ssim": img["visual_quality"]["ssim"],
            })
        for r in img["benchmark"].get("robustness") or []:
            for tr in r.get("tasks") or []:
                rows.append({
                    "image": img["image"],
                    "editor": f"robustness:{r['transform']}",
                    "task": tr["task_id"],
                    "task_name": tr["name"],
                    "success_original": tr["success_original"],
                    "success_protected": "",
                    "absolute_change": "",
                    "relative_change_pct": "",
                    "task_metric_original": "",
                    "task_metric_protected": "",
                    "clip_delta_original": "",
                    "clip_delta_protected": "",
                    "samples": "",
                    "ssim": img["visual_quality"]["ssim"],
                })
    if not rows:
        return
    fields = list(rows[0].keys())
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_report(payload: dict) -> str:
    lines = ["# AI PRIVACY SHIELD — AI-EDITING BENCHMARK", ""]
    env = payload["environment"]
    lines += [
        "## Environment",
        "",
        f"- Device: **{env['gpu']}** ({env['device']})" + (f", {env.get('vram_mb', '?')} MB VRAM" if env["device"] == "cuda" else ""),
        f"- Python {env['python']} · torch {env['torch']} · diffusers {env['diffusers']} · transformers {env['transformers']}",
        f"- Seeds: {payload['seeds']} · Prompt variants: {'yes' if payload['prompt_variants'] else 'no'} · Masks: {payload['masks']}",
        f"- Transformations: {payload['transformations']}",
        "",
    ]
    for img in payload["images"]:
        p = img["editing_protection"]
        q = img["visual_quality"]
        lines += [
            f"## {img['image']}",
            "",
            f"- Protection: {'applied' if p['applied'] else 'not applied'} · denoising error "
            f"+{p['loss_increase_pct']:.1f}% (verified +{p['verified_increase_pct']:.1f}% at full res) "
            f"· SSIM {q['ssim']:.3f} · PSNR {q['psnr_db']:.1f} dB · L∞ {q['perturbation_linf']:.4f}",
            "",
            "### Editing benchmark (same prompt, seed, settings — only the input changes)",
            "",
            "| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for t in img["benchmark"].get("tasks") or []:
            lines.append(
                f"| {t['editor_type']} | {t['name']} | {t.get('mask_kind') or '—'} "
                f"| {t['success_original']:.3f} | {t['success_protected']:.3f} "
                f"| {t['absolute_change']:+.3f} | {_rel(t['relative_change_pct'])} |"
            )
        agg = img["benchmark"].get("aggregate") or {}
        lines += [
            "",
            f"- **Aggregate:** mean success {agg.get('mean_original', 0):.3f} → "
            f"{agg.get('mean_protected', 0):.3f} (absolute change {agg.get('mean_absolute_change', 0):+.3f}"
            + (f", relative {agg.get('mean_relative_change_pct', 0):.1f}%"
               if agg.get("mean_relative_change_pct") is not None else "")
            + f"); {agg.get('tasks_reduced', 0)}/{agg.get('tasks_total', 0)} rows reduced",
            "",
        ]
        rob = img["benchmark"].get("robustness") or []
        if rob:
            lines += ["### Edit success after transformations (protected image)", "", "| Transform | Orig | After |", "| --- | ---: | ---: |"]
            for r in rob:
                if "error" in r:
                    lines.append(f"| {r['transform']} | error | — |")
                else:
                    lines.append(
                        f"| {r['transform']} | {r['mean_success_original']:.3f} | {r['mean_success_after_transform']:.3f} |"
                    )
            lines.append("")
    return "\n".join(lines)


def html_report(payload: dict) -> str:
    def _row(t, img_name):
        rel = "n/a" if t["relative_change_pct"] is None else f"{t['relative_change_pct']:.1f}%"
        cls = "good" if t["absolute_change"] > 0 else "weak"
        return (
            f"<tr><td>{img_name}</td><td>{t['editor_type']}</td><td>{t['name']}</td>"
            f"<td>{t.get('mask_kind') or '—'}</td>"
            f"<td>{t['success_original']:.3f}</td><td>{t['success_protected']:.3f}</td>"
            f"<td class='{cls}'>{t['absolute_change']:+.3f}</td><td>{rel}</td></tr>"
        )

    rows = "".join(
        _row(t, img["image"])
        for img in payload["images"]
        for t in (img["benchmark"].get("tasks") or [])
    )
    agg_rows = ""
    for img in payload["images"]:
        agg = img["benchmark"].get("aggregate") or {}
        rel = "n/a" if agg.get("mean_relative_change_pct") is None else f"{agg['mean_relative_change_pct']:.1f}%"
        agg_rows += (
            f"<tr><td>{img['image']}</td><td>{agg.get('mean_original', 0):.3f}</td>"
            f"<td>{agg.get('mean_protected', 0):.3f}</td><td>{agg.get('mean_absolute_change', 0):+.3f}</td>"
            f"<td>{rel}</td><td>{agg.get('tasks_reduced', 0)}/{agg.get('tasks_total', 0)}</td></tr>"
        )
    rob_rows = ""
    for img in payload["images"]:
        for r in img["benchmark"].get("robustness") or []:
            if "error" in r:
                rob_rows += f"<tr><td>{img['image']}</td><td>{r['transform']}</td><td colspan='2'>error</td></tr>"
            else:
                rob_rows += (
                    f"<tr><td>{img['image']}</td><td>{r['transform']}</td>"
                    f"<td>{r['mean_success_original']:.3f}</td><td>{r['mean_success_after_transform']:.3f}</td></tr>"
                )
    prot_rows = ""
    for img in payload["images"]:
        p = img["editing_protection"]
        q = img["visual_quality"]
        prot_rows += (
            f"<tr><td>{img['image']}</td><td>{'applied' if p['applied'] else 'not applied'}</td>"
            f"<td>+{p['loss_increase_pct']:.1f}%</td><td>+{p['verified_increase_pct']:.1f}%</td>"
            f"<td>{q['ssim']:.3f}</td><td>{q['psnr_db']:.1f} dB</td><td>{q['perturbation_linf']:.4f}</td></tr>"
        )
    env = payload["environment"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AI Privacy Shield — AI-Editing Benchmark</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
h1,h2{{border-bottom:1px solid #ddd;padding-bottom:.4rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.45rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f5f5f5}}
.good{{color:#0a7d33;font-weight:600}}
.weak{{color:#a33;font-weight:600}}
.meta{{font-size:.85rem;color:#555}}
</style></head><body>
<h1>AI Privacy Shield — AI-Editing Benchmark</h1>
<p class="meta">Generated {env['timestamp']} · {env['gpu']} ({env['device']})
{f', {env.get('vram_mb', '?')} MB VRAM' if env['device'] == 'cuda' else ''} · python {env['python']} ·
torch {env['torch']} · diffusers {env['diffusers']}<br>
seeds {payload['seeds']} · prompt variants {'yes' if payload['prompt_variants'] else 'no'} ·
masks {payload['masks']} · transformations {payload['transformations']}</p>
<h2>Multi-Model Editing Protection</h2>
<table><thead><tr><th>Image</th><th>Status</th><th>Denoising error</th><th>Verified (full res)</th>
<th>SSIM</th><th>PSNR</th><th>L∞</th></tr></thead><tbody>{prot_rows}</tbody></table>
<h2>Editing Benchmark (Original vs Protected)</h2>
<table><thead><tr><th>Image</th><th>Editor</th><th>Task</th><th>Mask</th><th>Orig Success</th>
<th>Prot Success</th><th>Absolute Δ</th><th>Relative Δ</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Aggregates per Image</h2>
<table><thead><tr><th>Image</th><th>Mean Orig</th><th>Mean Prot</th><th>Mean Abs Δ</th><th>Mean Rel Δ</th>
<th>Reduced</th></tr></thead><tbody>{agg_rows}</tbody></table>
<h2>Edit Success after Transformations</h2>
<table><thead><tr><th>Image</th><th>Transform</th><th>Orig Success</th><th>After Transform</th></tr></thead>
<tbody>{rob_rows}</tbody></table>
<p class="meta">Success = 60% task-specific pixel metric + 40% CLIP semantic alignment. Relative change is
reported only when the original success is meaningful (>= 0.02). "Reduced" = rows where protected
edit success dropped below the original.</p>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="image paths (default: data/benchmark/* + tests/fixtures/*)")
    parser.add_argument("--tasks", default=None, help="comma-separated task ids (default: configured)")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default: configured 42,7,1337)")
    parser.add_argument("--prompts", action="store_true", help="run prompt variants per task (slower)")
    parser.add_argument("--masks", default=None, help="comma-separated mask kinds for inpainting (default: configured)")
    parser.add_argument("--transformations", default=None, help="comma-separated transforms (default: configured)")
    parser.add_argument("--strengths", default=None, help="comma-separated img2img denoising strengths, e.g. 0.4,0.6,0.8")
    parser.add_argument("--out", default="results", help="output directory for json/csv/md/html")
    args = parser.parse_args()

    paths = [Path(p) for p in args.images] if args.images else DEFAULT_FIXTURES
    if not paths:
        print("No images found. Pass image paths or add files under data/benchmark/ or tests/fixtures/.")
        return 1

    task_ids = tuple(t.strip() for t in args.tasks.split(",")) if args.tasks else None
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else list(settings.EDITING_SEEDS)
    masks = tuple(m.strip() for m in args.masks.split(",") if m.strip()) if args.masks else settings.EDITING_MASK_KINDS
    transforms = (
        tuple(t.strip() for t in args.transformations.split(",") if t.strip())
        if args.transformations else settings.ROBUSTNESS_TRANSFORMS
    )
    strengths = (
        [float(s) for s in args.strengths.split(",") if s.strip()] if args.strengths else None
    )

    print("AI PRIVACY SHIELD — MULTI-MODEL EDITING BENCHMARK")
    print(f"  device: {_gpu_name()}  seeds: {seeds}  prompt variants: {args.prompts}")
    print(f"  masks: {masks}  transforms: {transforms}")

    images: list[dict] = []
    failed = 0
    for path in paths:
        print(f"\n=== {path.name} ===")
        try:
            img = run_one(path, task_ids, seeds, args.prompts, masks, transforms, strengths)
            p = img["editing_protection"]
            q = img["visual_quality"]
            print(f"  protection: applied={p['applied']} (denoise +{p['loss_increase_pct']:.1f}%), "
                  f"SSIM {q['ssim']:.3f}, PSNR {q['psnr_db']:.1f} dB")
            bench = img["benchmark"]
            if bench.get("tasks"):
                for t in bench["tasks"]:
                    rel = f"{t['relative_change_pct']:+.1f}%" if t["relative_change_pct"] is not None else "n/a"
                    print(
                        f"    [{t['editor_type']:>11}] {t['name']:<22} "
                        f"{t['success_original']:>6.3f} -> {t['success_protected']:>6.3f} "
                        f"(abs {t['absolute_change']:+.3f}, rel {rel})"
                    )
                agg = bench["aggregate"]
                rel = f", rel {agg['mean_relative_change_pct']:+.1f}%" if agg.get("mean_relative_change_pct") is not None else ""
                print(f"    aggregate: {agg['mean_original']:.3f} -> {agg['mean_protected']:.3f} "
                      f"(abs {agg['mean_absolute_change']:+.3f}{rel}), {agg['tasks_reduced']}/{agg['tasks_total']} reduced")
            else:
                print(f"  benchmark: {bench['note']}")
            rob = bench.get("robustness") or []
            for r in rob:
                if "error" in r:
                    print(f"    robustness {r['transform']}: error")
                else:
                    print(f"    robustness {r['transform']}: {r['mean_success_original']:.3f} -> "
                          f"{r['mean_success_after_transform']:.3f}")
            images.append(img)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failed += 1

    # aggregate stats across images (mean/median/std/min/max of absolute change)
    abs_changes = [
        t["absolute_change"]
        for img in images
        for t in (img["benchmark"].get("tasks") or [])
    ]
    aggregate_stats = None
    if abs_changes:
        aggregate_stats = {
            "mean": float(statistics.mean(abs_changes)),
            "median": float(statistics.median(abs_changes)),
            "std": float(statistics.stdev(abs_changes)) if len(abs_changes) > 1 else 0.0,
            "min": float(min(abs_changes)),
            "max": float(max(abs_changes)),
            "rows": len(abs_changes),
        }

    payload = {
        "environment": _env_record(),
        "seeds": seeds,
        "prompt_variants": args.prompts,
        "masks": list(masks),
        "transformations": list(transforms),
        "img2img_strengths": strengths or [settings.EDITING_IMG2IMG_STRENGTH],
        "tasks": task_ids or list(settings.EDITING_TASKS),
        "config": {
            "optimization_models": ["stable-diffusion-v1-5/stable-diffusion-v1-5 (anti-diffusion surrogate)"],
            "evaluation_models": [
                "timbrooks/instruct-pix2pix (held out)",
                "stable-diffusion-v1-5/stable-diffusion-inpainting",
                "stable-diffusion-v1-5/stable-diffusion-v1-5 (image-to-image)",
            ],
            "held_out_models": ["timbrooks/instruct-pix2pix"],
            "epsilon": settings.PERTURBATION_EPSILON,
            "surrogate_iters": settings.EDITING_SURROGATE_ITERS_GPU,
            "surrogate_resolution": settings.EDITING_SURROGATE_RESOLUTION,
            "edit_resolution": settings.EDITING_RESOLUTION,
            "edit_steps": settings.EDITING_STEPS,
            "guidance_scale": settings.EDITING_GUIDANCE,
            "task_metric_weight": settings.W_EDITING_TASK_METRIC,
            "clip_weight": settings.W_EDITING_CLIP,
        },
        "images": images,
        "aggregate_stats": aggregate_stats,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(md_report(payload), encoding="utf-8")
    (out_dir / "report.html").write_text(html_report(payload), encoding="utf-8")
    csv_output(payload, out_dir)

    print(f"\n{len(images)} succeeded, {failed} failed")
    if aggregate_stats:
        print("aggregate absolute change (all rows): "
              f"mean {aggregate_stats['mean']:+.3f}, median {aggregate_stats['median']:+.3f}, "
              f"std {aggregate_stats['std']:.3f}, min {aggregate_stats['min']:+.3f}, max {aggregate_stats['max']:+.3f}")
    print(f"report:   {out_dir / 'report.html'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

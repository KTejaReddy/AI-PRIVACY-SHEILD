"""Multi-family protection benchmark + adaptive red-team (research profile).

Runs the PhotoGuard-style multi-family engine end to end:

  1. PROTECT: optimize ONE perturbation against diffusion editing (A-D),
     identity-reference / face-swap (E-F) and vision encoders (I) using the
     production protection engine's multi-family PGD.
  2. RED-TEAM: outer rounds — probe each family with cheap surrogates, raise
     the weight of the weakest family, re-protect. Stops when gains saturate,
     the quality floor is hit, or the round budget is exhausted.
  3. EVALUATE the FINAL protected image per attack family:
       * direct editing (A/C/D): InstructPix2Pix (held out), masked
         inpainting (SD1.5), image-to-image (SD1.5), incl. edit success after
         real transformations;
       * identity-reference / face-swap (E/F): FaceNet (evaluation) + ArcFace
         (held out) face-embedding similarity;
       * vision encoders (I): CLIP ViT-L/14 + ResNet50 (held out)
         representation distance;
       * image-to-video (G): adapter registered — reported NOT TESTED rather
         than faked.

Outputs: results.json, results.csv, report.md, report.html. Every score is
the real measured value; relative percentages are only shown when the
underlying metric is meaningful (spec §20, §48).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("AIPS_PROFILE", "research")

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402

from app.config import settings  # noqa: E402
from app.attack_registry import families_report  # noqa: E402
from app.editing.manager import get_editing_manager  # noqa: E402
from app.editing.protector import apply_editing_protection  # noqa: E402
from app.evaluation.editing_benchmark import EditingBenchmark  # noqa: E402
from app.models.face_models import get_registry  # noqa: E402
from app.privacy.analyzer import PrivacyAnalyzer  # noqa: E402
from app.quality.metrics import compute_quality  # noqa: E402
from app.utils.imaging import load_and_normalize  # noqa: E402

PROJECT_ROOT = BACKEND.parent
DATASET_DIR = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_FIXTURES = sorted(
    list((DATASET_DIR).glob("*.jpg"))
    + list((DATASET_DIR).glob("*.png"))
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


# ---------------------------------------------------------------------------
# per-family probes (cheap, used by the red-team loop)
# ---------------------------------------------------------------------------
def _identity_probe(original, protected, face_boxes, device) -> float | None:
    """Mean face-embedding cosine similarity original->protected (lower=better)."""
    if not face_boxes:
        return None
    import numpy as np  # noqa: PLC0415

    from app.models.face_models import (  # noqa: PLC0415
        FACENET_INPUT_SIZE,
        l2_normalize,
    )

    reg = get_registry(device)
    facenet = reg._models.get("facenet_vggface2")
    if facenet is None or not facenet.info.loaded:
        return None
    sims = []
    for box in face_boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(original.shape[1], x1), min(original.shape[0], y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        def _emb(img):
            crop = img[y0:y1, x0:x1]
            import cv2  # noqa: PLC0415

            crop = cv2.resize(crop, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            return l2_normalize(facenet.embed_crops([crop], device=device)[0])

        e_orig = _emb(original)
        e_prot = _emb(protected)
        sims.append(float(np.dot(e_orig, e_prot)))
    if not sims:
        return None
    return float(np.mean(sims))


def _vision_probe(original, protected, device) -> dict:
    """Representation similarity for the vision-encoder family (lower=better)."""
    import numpy as np  # noqa: PLC0415

    from app.models.face_models import imagenet_preprocess, l2_normalize  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    reg = get_registry(device)
    out = {}
    for mid in ("mobilenet_v3_large", "resnet50"):
        model = reg._models.get(mid)
        if model is None or not model.info.loaded or model.torch_model is None:
            continue

        def _emb(img):
            r = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            inp = imagenet_preprocess(r)
            t = torch.from_numpy(inp).to(device)
            with torch.no_grad():
                e = model.torch_model(t).cpu().numpy()[0]
            return l2_normalize(e)

        out[mid] = float(np.dot(_emb(original), _emb(protected)))
    return out


def _denoise_probe(original, protected, device) -> float | None:
    """Denoising-error increase of the anti-diffusion surrogate (higher=better)."""
    mgr = get_editing_manager(device)
    sur = mgr.get_surrogate(resolution=settings.EDITING_SURROGATE_RESOLUTION, timestep=settings.EDITING_SURROGATE_TIMESTEP)
    if not sur.loaded:
        try:
            sur.load()
        except Exception:  # noqa: BLE001
            return None
    import torchvision.transforms.functional as TF  # noqa: PLC0415

    def _loss(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).half().to(device)
        t = TF.resize(t, (settings.EDITING_SURROGATE_RESOLUTION,) * 2, antialias=True)
        with torch.no_grad():
            return float(sur.denoising_loss(sur.encode(t)))

    before, after = _loss(original), _loss(protected)
    if before <= 1e-9:
        return None
    return (after - before) / before


def _probe_families(original, protected, face_boxes, device) -> dict:
    return {
        "diffusion": _denoise_probe(original, protected, device),
        "identity": _identity_probe(original, protected, face_boxes, device),
        "vision": _vision_probe(original, protected, device),
    }


def _weakest(probes: dict) -> tuple[str, float] | None:
    """Family with the least disruption (score to maximize)."""
    scored = {}
    if probes.get("diffusion") is not None:
        scored["diffusion"] = probes["diffusion"]
    if probes.get("identity") is not None:
        scored["identity"] = 1.0 - probes["identity"]
    vis = probes.get("vision") or {}
    if "mobilenet_v3_large" in vis:
        scored["vision"] = 1.0 - vis["mobilenet_v3_large"]
    if not scored:
        return None
    return min(scored.items(), key=lambda kv: kv[1])


# ---------------------------------------------------------------------------
# final per-family evaluation of the protected image
# ---------------------------------------------------------------------------
def _evaluate_identity(original, protected, face_boxes, device) -> list[dict]:
    """Face-embedding similarity per identity encoder (lower protected = better)."""
    if not face_boxes:
        return [{"family": "identity_reference", "note": "no face detected — not evaluated"}]
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from app.models.face_models import (  # noqa: PLC0415
        ARCFACE_INPUT_SIZE,
        FACENET_INPUT_SIZE,
        l2_normalize,
    )

    reg = get_registry(device)
    crops_orig, crops_prot = [], []
    for box in face_boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(original.shape[1], x1), min(original.shape[0], y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crops_orig.append(cv2.resize(original[y0:y1, x0:x1], (FACENET_INPUT_SIZE,) * 2, interpolation=cv2.INTER_AREA))
        crops_prot.append(cv2.resize(protected[y0:y1, x0:x1], (FACENET_INPUT_SIZE,) * 2, interpolation=cv2.INTER_AREA))

    if not crops_orig:
        return [{"family": "identity_reference", "note": "no face detected — not evaluated"}]

    rows = []
    for mid, label in (("facenet_casia", "FaceNet (CASIA-WebFace) — evaluation"),
                       ("arcface_mbf", "ArcFace (MobileFaceNet) — held out")):
        model = reg._models.get(mid)
        if model is None or not model.info.loaded:
            rows.append({"family": "identity_reference", "model": mid, "model_name": label,
                         "note": "model unavailable"})
            continue
        if model.onnx:
            size = ARCFACE_INPUT_SIZE
            o_crops = [cv2.resize(c, (size,) * 2, interpolation=cv2.INTER_AREA) for c in crops_orig]
            p_crops = [cv2.resize(c, (size,) * 2, interpolation=cv2.INTER_AREA) for c in crops_prot]
        else:
            o_crops, p_crops = crops_orig, crops_prot
        try:
            e_orig = [l2_normalize(e) for e in model.embed_crops(o_crops, device=device)]
            e_prot = [l2_normalize(e) for e in model.embed_crops(p_crops, device=device)]
            sims = [float(np.dot(a, b)) for a, b in zip(e_orig, e_prot)]
            rows.append({
                "family": "identity_reference",
                "model": mid,
                "model_name": label,
                "role": "evaluation" if mid == "facenet_casia" else "held_out",
                "similarity_original": round(float(np.mean(sims)), 4),
                "disruption": round(1.0 - float(np.mean(sims)), 4),
                "faces": len(sims),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"family": "identity_reference", "model": mid, "model_name": label,
                         "note": f"failed: {type(exc).__name__}"})
    return rows


def _evaluate_vision(original, protected, device) -> list[dict]:
    import numpy as np  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    from app.models.face_models import imagenet_preprocess, l2_normalize  # noqa: PLC0415

    reg = get_registry(device)
    rows = []
    for mid, label in (("mobilenet_v3_large", "MobileNetV3-Large — optimization"),
                       ("resnet50", "ResNet50 — held out")):
        model = reg._models.get(mid)
        if model is None or not model.info.loaded:
            rows.append({"family": "vision_encoder", "model": mid, "note": "model unavailable"})
            continue

        def _emb(arr):
            r = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
            inp = imagenet_preprocess(r)
            t = torch.from_numpy(inp).to(device)
            with torch.no_grad():
                return l2_normalize(model.torch_model(t).cpu().numpy()[0])

        try:
            sim = float(np.dot(_emb(original), _emb(protected)))
            rows.append({
                "family": "vision_encoder",
                "model": mid,
                "model_name": label,
                "role": "optimization" if mid == "mobilenet_v3_large" else "held_out",
                "similarity_original": round(sim, 4),
                "disruption": round(1.0 - sim, 4),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"family": "vision_encoder", "model": mid, "note": f"failed: {type(exc).__name__}"})

    # CLIP ViT-L/14 (VLM conditioning + vision encoder, evaluation)
    try:
        mgr = get_editing_manager(device)
        clip = mgr.get_clip()
        proc = mgr.clip_processor

        def _clip_emb(arr):
            from PIL import Image  # noqa: PLC0415

            inp = proc(images=Image.fromarray(arr), return_tensors="pt").to(device)
            with torch.no_grad():
                return clip.get_image_features(**inp)

        e_o = _clip_emb(original)
        e_p = _clip_emb(protected)
        sim = float(torch.nn.functional.cosine_similarity(e_o, e_p).item())
        rows.append({
            "family": "vlm_conditioning",
            "model": "openai/clip-vit-large-patch14",
            "model_name": "CLIP ViT-L/14 — evaluation",
            "role": "evaluation",
            "similarity_original": round(sim, 4),
            "disruption": round(1.0 - sim, 4),
        })
    except Exception as exc:  # noqa: BLE001
        rows.append({"family": "vlm_conditioning", "model": "openai/clip-vit-large-patch14",
                     "note": f"failed: {type(exc).__name__}"})
    return rows


# ---------------------------------------------------------------------------
# red-team protection (rounds)
# ---------------------------------------------------------------------------
def protect_with_redteam(original, face_boxes, device, rounds: int) -> dict:
    """Multi-family PGD + adaptive outer rounds. Returns protection history + final result."""
    weights = {"diffusion": 1.0, "identity": settings.EDITING_IDENTITY_WEIGHT, "vision": settings.EDITING_VISION_WEIGHT}
    history = []
    final = None
    prev_score = None
    for rnd in range(max(1, rounds)):
        t0 = time.perf_counter()
        result = apply_editing_protection(
            original, device, progress=lambda _i: None, face_boxes=face_boxes, weights=weights
        )
        if result.applied and result.protected is not None:
            protected = result.protected
        else:
            protected = original.copy()
        probes = _probe_families(original, protected, face_boxes, device)
        weak = _weakest(probes)
        history.append({
            "round": rnd + 1,
            "weights": dict(weights),
            "probes": {
                "diffusion_denoise_increase": probes.get("diffusion"),
                "identity_similarity": probes.get("identity"),
                "vision_similarity_mobilenet": (probes.get("vision") or {}).get("mobilenet_v3_large"),
            },
            "weakest_family": weak[0] if weak else None,
            "seconds": round(time.perf_counter() - t0, 1),
        })
        final = result
        # stopping conditions
        score = None
        parts = [p for p in (probes.get("diffusion"), 1.0 - probes["identity"] if probes.get("identity") is not None else None,
                             (1.0 - (probes.get("vision") or {}).get("mobilenet_v3_large")) if (probes.get("vision") or {}).get("mobilenet_v3_large") is not None else None)]
        parts = [p for p in parts if p is not None]
        if parts:
            score = float(statistics.mean(parts))
        if prev_score is not None and score is not None and (score - prev_score) < settings.RED_TEAM_MIN_GAIN:
            history[-1]["stopped"] = f"gain {score - prev_score:+.4f} below min gain {settings.RED_TEAM_MIN_GAIN}"
            break
        prev_score = score
        if weak is not None:
            fam_key = weak[0]
            weights[fam_key] = min(weights.get(fam_key, 1.0) * settings.RED_TEAM_WEIGHT_STEP, settings.RED_TEAM_MAX_WEIGHT)
        if rnd + 1 >= rounds:
            history[-1]["stopped"] = "round budget exhausted"
    return {"final": final, "history": history}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_one(path: Path, rounds: int, task_ids, seeds, masks, transforms, strengths) -> dict:
    record = load_and_normalize(path.read_bytes(), len(path.read_bytes()))
    original = record.array

    device = _device()
    reg = get_registry(device)
    if not any(m.info.loaded for m in reg.optimization_models):
        reg.load_all()

    # face boxes (drives identity-reference term + identity evaluation)
    analyzer = PrivacyAnalyzer()
    analysis = analyzer.analyze(record)
    face_boxes = [f.box for f in analysis.faces]

    # ---- multi-family protection with adaptive red-team ------------------
    rt = protect_with_redteam(original, face_boxes, device, rounds)
    edit_protect = rt["final"]
    protected = edit_protect.protected if edit_protect.applied and edit_protect.protected is not None else original.copy()
    quality = compute_quality(original, protected, device=device)

    # ---- final per-family evaluation -------------------------------------
    bench = EditingBenchmark(device)
    t0 = time.perf_counter()
    bench_result = bench.run(original, protected, task_ids=task_ids, seeds=seeds, strengths=strengths,
                             progress=lambda _i: None)
    t_bench = time.perf_counter() - t0
    t0 = time.perf_counter()
    bench_result.robustness = bench.run_robustness(original, protected, task_ids=task_ids, transforms=transforms,
                                                   progress=lambda _i: None)
    t_rob = time.perf_counter() - t0

    identity_rows = _evaluate_identity(original, protected, face_boxes, device)
    vision_rows = _evaluate_vision(original, protected, device)

    return {
        "image": path.name,
        "dimensions": {"width": record.width, "height": record.height},
        "face_count": analysis.face_count,
        "red_team": rt["history"],
        "editing_protection": edit_protect.as_dict(),
        "visual_quality": {
            "ssim": round(quality.ssim, 4),
            "psnr_db": round(quality.psnr_db, 2),
            "perturbation_linf": round(quality.perturbation_linf, 4),
        },
        "benchmark": bench_result.as_dict(),
        "benchmark_seconds": round(t_bench, 1),
        "robustness_seconds": round(t_rob, 1),
        "identity_reference": identity_rows,
        "vision_encoders": vision_rows,
    }


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def _rel(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}%"


def md_report(payload: dict) -> str:
    lines = ["# AI PRIVACY SHIELD — MULTI-FAMILY PROTECTION BENCHMARK", ""]
    env = payload["environment"]
    lines += ["## Environment", "",
              f"- Device: **{env['gpu']}** ({env['device']})"
              + (f", {env.get('vram_mb', '?')} MB VRAM" if env["device"] == "cuda" else ""),
              f"- Python {env['python']} · torch {env['torch']} · diffusers {env['diffusers']} · transformers {env['transformers']}",
              f"- Red-team rounds: {payload['red_team_rounds']} · Seeds: {payload['seeds']} · Masks: {payload['masks']} · Transforms: {payload['transformations']}", ""]
    for img in payload["images"]:
        p = img["editing_protection"]
        q = img["visual_quality"]
        lines += [f"## {img['image']}", "",
                  f"- Protection: {'applied' if p['applied'] else 'not applied'} · families: {', '.join(p.get('families') or [])} "
                  f"· denoising error +{p['loss_increase_pct']:.1f}% · SSIM {q['ssim']:.3f} · PSNR {q['psnr_db']:.1f} dB",
                  f"- Identity similarity (protected vs original): {p.get('identity_similarity_before')} → {p.get('identity_similarity_after')}",
                  ""]
        if img.get("red_team"):
            lines += ["### Adaptive red-team rounds", "", "| Round | diffusion↑ | identity sim | vision sim | weakest | weights |", "| --- | ---: | ---: | ---: | --- | --- |"]
            for r in img["red_team"]:
                pr = r["probes"]
                w = r["weights"]
                d_inc = pr["diffusion_denoise_increase"]
                i_sim = pr["identity_similarity"]
                v_sim = pr["vision_similarity_mobilenet"]
                lines.append(
                    f"| {r['round']} "
                    f"| {f'{d_inc:.3f}' if d_inc is not None else '—'} "
                    f"| {f'{i_sim:.3f}' if i_sim is not None else '—'} "
                    f"| {f'{v_sim:.3f}' if v_sim is not None else '—'} "
                    f"| {r['weakest_family'] or '—'} "
                    f"| d{w['diffusion']:.2f}/i{w['identity']:.2f}/v{w['vision']:.2f} |"
                )
            lines.append("")
        lines += ["### Direct editing (original vs protected)", "", "| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for t in img["benchmark"].get("tasks") or []:
            lines.append(f"| {t['editor_type']} | {t['name']} | {t.get('mask_kind') or '—'} | {t['success_original']:.3f} "
                         f"| {t['success_protected']:.3f} | {t['absolute_change']:+.3f} | {_rel(t['relative_change_pct'])} |")
        agg = img["benchmark"].get("aggregate") or {}
        lines += ["", f"- **Direct-editing aggregate:** {agg.get('mean_original', 0):.3f} → {agg.get('mean_protected', 0):.3f} "
                      f"(abs {agg.get('mean_absolute_change', 0):+.3f}); {agg.get('tasks_reduced', 0)}/{agg.get('tasks_total', 0)} rows reduced", ""]
        rob = img["benchmark"].get("robustness") or []
        if rob:
            lines += ["### Edit success after transformations", "", "| Transform | Orig | After |", "| --- | ---: | ---: |"]
            for r in rob:
                lines.append(f"| {r['transform']} | {r['mean_success_original']:.3f} | {r['mean_success_after_transform']:.3f} |" if "error" not in r else f"| {r['transform']} | error | — |")
            lines.append("")
        lines += ["### Identity-reference / face-swap (E-F)", "", "| Model | Role | Sim orig→prot | Disruption |", "| --- | --- | ---: | ---: |"]
        for r in img.get("identity_reference") or []:
            if "note" in r:
                lines.append(f"| {r.get('model_name', r.get('model', '?'))} | — | {r['note']} | — |")
            else:
                lines.append(f"| {r['model_name']} | {r['role']} | {r['similarity_original']} | {r['disruption']} |")
        lines += ["", "### Vision encoders (I) + VLM conditioning (H)", "", "| Model | Role | Sim orig→prot | Disruption |", "| --- | --- | ---: | ---: |"]
        for r in img.get("vision_encoders") or []:
            if "note" in r:
                lines.append(f"| {r.get('model_name', r.get('model', '?'))} | — | {r['note']} | — |")
            else:
                lines.append(f"| {r['model_name']} | {r['role']} | {r['similarity_original']} | {r['disruption']} |")
        lines += ["", "### Image-to-video (G)", "", "- NOT TESTED: modular adapter registered; local video-generation weights exceed this hardware. "
                  "Identity disruption (E/F) is the surrogate evidence.", ""]
    return "\n".join(lines)


def html_report(payload: dict) -> str:
    rows = ""
    for img in payload["images"]:
        for t in (img["benchmark"].get("tasks") or []):
            rel = _rel(t["relative_change_pct"])
            cls = "good" if t["absolute_change"] > 0 else "weak"
            rows += (f"<tr><td>{img['image']}</td><td>{t['editor_type']}</td><td>{t['name']}</td><td>{t.get('mask_kind') or '—'}</td>"
                     f"<td>{t['success_original']:.3f}</td><td>{t['success_protected']:.3f}</td>"
                     f"<td class='{cls}'>{t['absolute_change']:+.3f}</td><td>{rel}</td></tr>")
    id_rows = ""
    for img in payload["images"]:
        for r in (img.get("identity_reference") or []):
            if "note" in r:
                id_rows += f"<tr><td>{img['image']}</td><td>{r.get('model_name', r.get('model','?'))}</td><td colspan='2'>{r['note']}</td></tr>"
            else:
                id_rows += f"<tr><td>{img['image']}</td><td>{r['model_name']}</td><td>{r['similarity_original']}</td><td>{r['disruption']}</td></tr>"
    vis_rows = ""
    for img in payload["images"]:
        for r in (img.get("vision_encoders") or []):
            if "note" in r:
                vis_rows += f"<tr><td>{img['image']}</td><td>{r.get('model_name', r.get('model','?'))}</td><td colspan='2'>{r['note']}</td></tr>"
            else:
                vis_rows += f"<tr><td>{img['image']}</td><td>{r['model_name']}</td><td>{r['similarity_original']}</td><td>{r['disruption']}</td></tr>"
    prot_rows = ""
    for img in payload["images"]:
        p = img["editing_protection"]
        q = img["visual_quality"]
        prot_rows += (f"<tr><td>{img['image']}</td><td>{'applied' if p['applied'] else 'not applied'}</td>"
                      f"<td>{', '.join(p.get('families') or [])}</td><td>+{p['loss_increase_pct']:.1f}%</td>"
                      f"<td>{p.get('identity_similarity_before')} → {p.get('identity_similarity_after')}</td>"
                      f"<td>{q['ssim']:.3f}</td><td>{q['psnr_db']:.1f} dB</td></tr>")
    env = payload["environment"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AI Privacy Shield — Multi-Family Benchmark</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#222}}
h1,h2{{border-bottom:1px solid #ddd;padding-bottom:.4rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.45rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f5f5f5}}
.good{{color:#0a7d33;font-weight:600}}
.weak{{color:#a33;font-weight:600}}
.meta{{font-size:.85rem;color:#555}}
</style></head><body>
<h1>AI Privacy Shield — Multi-Family Protection Benchmark</h1>
<p class="meta">Generated {env['timestamp']} · {env['gpu']} ({env['device']}){f', {env.get('vram_mb','?')} MB VRAM' if env['device']=='cuda' else ''} · python {env['python']} · torch {env['torch']}<br>
red-team rounds {payload['red_team_rounds']} · seeds {payload['seeds']} · masks {payload['masks']} · transforms {payload['transformations']}</p>
<h2>Multi-Family Protection</h2>
<table><thead><tr><th>Image</th><th>Status</th><th>Families</th><th>Denoise↑</th><th>Identity sim</th><th>SSIM</th><th>PSNR</th></tr></thead>
<tbody>{prot_rows}</tbody></table>
<h2>Direct Editing (Original vs Protected)</h2>
<table><thead><tr><th>Image</th><th>Editor</th><th>Task</th><th>Mask</th><th>Orig</th><th>Prot</th><th>Abs Δ</th><th>Rel Δ</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Identity-Reference / Face-Swap (E-F)</h2>
<table><thead><tr><th>Image</th><th>Model</th><th>Sim orig→prot</th><th>Disruption</th></tr></thead><tbody>{id_rows}</tbody></table>
<h2>Vision Encoders (I) + VLM (H)</h2>
<table><thead><tr><th>Image</th><th>Model</th><th>Sim orig→prot</th><th>Disruption</th></tr></thead><tbody>{vis_rows}</tbody></table>
<p class="meta">Lower similarity = more disrupted reference usefulness. Direct-editing success = 60% task metric + 40% CLIP. Relative % only when the original is meaningful. Image-to-video (G): adapter registered, NOT TESTED on this hardware.</p>
</body></html>"""


def csv_output(payload: dict, out_dir: Path) -> None:
    rows = []
    for img in payload["images"]:
        for t in (img["benchmark"].get("tasks") or []):
            rows.append({"image": img["image"], "family": t["editor_type"], "task": t["id"],
                         "model": t["editor_type"], "role": "evaluation",
                         "metric": "edit_success", "original": t["success_original"],
                         "protected": t["success_protected"],
                         "absolute_change": t["absolute_change"]})
        for r in (img.get("identity_reference") or []):
            if "similarity_original" in r:
                rows.append({"image": img["image"], "family": "identity_reference", "task": "face_similarity",
                             "model": r["model"], "role": r["role"], "metric": "cosine_similarity",
                             "original": r["similarity_original"], "protected": r["similarity_original"],
                             "absolute_change": r["disruption"]})
        for r in (img.get("vision_encoders") or []):
            if "similarity_original" in r:
                rows.append({"image": img["image"], "family": r["family"], "task": "representation_similarity",
                             "model": r["model"], "role": r["role"], "metric": "cosine_similarity",
                             "original": r["similarity_original"], "protected": r["similarity_original"],
                             "absolute_change": r["disruption"]})
    if not rows:
        return
    fields = list(rows[0].keys())
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="image paths (default: data/benchmark/* + tests/fixtures/*)")
    parser.add_argument("--rounds", type=int, default=None, help="adaptive red-team rounds (default: research profile flag)")
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default: 42)")
    parser.add_argument("--masks", default=None, help="comma-separated mask kinds")
    parser.add_argument("--transformations", default=None, help="comma-separated transforms")
    parser.add_argument("--strengths", default=None, help="comma-separated img2img strengths")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--cache-dir", default="results/bench_cache", help="per-image result cache directory")
    parser.add_argument("--resume", action="store_true",
                        help="skip images whose per-image result is already cached")
    args = parser.parse_args()

    paths = [Path(p) for p in args.images] if args.images else DEFAULT_FIXTURES
    if not paths:
        print("No images found. Pass paths or add files under data/benchmark/ or tests/fixtures/.")
        return 1

    from app.attack_registry import load_profile  # noqa: PLC0415

    prof = load_profile("research")
    rounds = args.rounds if args.rounds is not None else int(prof.flags.get("red_team_rounds", 0))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else [42]
    task_ids = tuple(t.strip() for t in args.tasks.split(",")) if args.tasks else None
    masks = tuple(m.strip() for m in args.masks.split(",") if m.strip()) if args.masks else settings.EDITING_MASK_KINDS
    transforms = tuple(t.strip() for t in args.transformations.split(",") if t.strip()) if args.transformations else ("jpeg_compression", "resize")
    strengths = [float(s) for s in args.strengths.split(",") if s.strip()] if args.strengths else None

    print("AI PRIVACY SHIELD — MULTI-FAMILY PROTECTION BENCHMARK (research profile)")
    print(f"  device: {_gpu_name()}  red-team rounds: {rounds}  seeds: {seeds}")

    cache_dir = Path(args.cache_dir)

    def _summarize(img: dict) -> None:
        """Print the per-image summary (shared by fresh runs and cache loads)."""
        p = img["editing_protection"]
        q = img["visual_quality"]
        print(f"  protection: applied={p['applied']} families={p.get('families')} denoise+{p['loss_increase_pct']:.1f}% "
              f"SSIM {q['ssim']:.3f} PSNR {q['psnr_db']:.1f} dB", flush=True)
        for r in img.get("red_team") or []:
            pr = r["probes"]
            print(f"    round {r['round']}: diffusion {pr['diffusion_denoise_increase']}, "
                  f"identity sim {pr['identity_similarity']}, vision sim {pr['vision_similarity_mobilenet']}, "
                  f"weakest={r['weakest_family']} weights={r['weights']}", flush=True)
        bench = img["benchmark"]
        if bench.get("tasks"):
            agg = bench["aggregate"]
            print(f"  direct-editing: {agg['mean_original']:.3f} -> {agg['mean_protected']:.3f} "
                  f"(abs {agg['mean_absolute_change']:+.3f}), {agg['tasks_reduced']}/{agg['tasks_total']} reduced", flush=True)
        for r in img.get("identity_reference") or []:
            if "similarity_original" in r:
                print(f"  identity [{r['model_name']}]: sim {r['similarity_original']} disruption {r['disruption']}", flush=True)
        for r in img.get("vision_encoders") or []:
            if "similarity_original" in r:
                print(f"  vision [{r['model_name']}]: sim {r['similarity_original']} disruption {r['disruption']}", flush=True)

    images, failed, resumed = [], 0, 0
    for path in paths:
        print(f"\n=== {path.name} ===", flush=True)
        cache_file = cache_dir / f"{path.stem}.json"
        if args.resume and cache_file.exists():
            try:
                img = json.loads(cache_file.read_text(encoding="utf-8"))
                print(f"  resumed from cache ({cache_file.name})", flush=True)
                _summarize(img)
                images.append(img)
                resumed += 1
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  cache load failed ({type(exc).__name__}: {exc}); re-running", flush=True)
        try:
            img = run_one(path, rounds, task_ids, seeds, masks, transforms, strengths)
            _summarize(img)
            if args.resume:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(img, indent=2), encoding="utf-8")
            images.append(img)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed += 1

    abs_changes = [t["absolute_change"] for img in images for t in (img["benchmark"].get("tasks") or [])]
    agg_stats = None
    if abs_changes:
        agg_stats = {"mean": float(statistics.mean(abs_changes)), "median": float(statistics.median(abs_changes)),
                     "std": float(statistics.stdev(abs_changes)) if len(abs_changes) > 1 else 0.0,
                     "min": float(min(abs_changes)), "max": float(max(abs_changes)), "rows": len(abs_changes)}

    payload = {
        "environment": _env_record(),
        "profile": "research",
        "red_team_rounds": rounds,
        "seeds": seeds,
        "masks": list(masks),
        "transformations": list(transforms),
        "tasks": task_ids or list(settings.EDITING_TASKS),
        "families": families_report("research"),
        "config": {
            "optimization_models": ["SD1.5 anti-diffusion surrogate", "FaceNet (VGGFace2)", "MobileNetV3-Large"],
            "evaluation_models": ["InstructPix2Pix (held out)", "SD1.5 inpainting", "SD1.5 img2img",
                                  "FaceNet (CASIA)", "CLIP ViT-L/14"],
            "held_out_models": ["InstructPix2Pix", "ArcFace (MobileFaceNet)", "ResNet50"],
            "epsilon": settings.PERTURBATION_EPSILON,
            "surrogate_iters": settings.EDITING_SURROGATE_ITERS_GPU,
            "identity_weight": settings.EDITING_IDENTITY_WEIGHT,
            "vision_weight": settings.EDITING_VISION_WEIGHT,
        },
        "images": images,
        "aggregate_stats": agg_stats,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(md_report(payload), encoding="utf-8")
    (out_dir / "report.html").write_text(html_report(payload), encoding="utf-8")
    csv_output(payload, out_dir)

    print(f"\n{len(images)} succeeded, {failed} failed" + (f" ({resumed} resumed from cache)" if resumed else ""))
    if agg_stats:
        print(f"aggregate direct-editing absolute change: mean {agg_stats['mean']:+.3f}, median {agg_stats['median']:+.3f}, "
              f"std {agg_stats['std']:.3f}, min {agg_stats['min']:+.3f}, max {agg_stats['max']:+.3f}")
    print(f"report: {out_dir / 'report.html'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

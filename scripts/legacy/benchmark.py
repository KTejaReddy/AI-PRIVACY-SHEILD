"""Reproducible benchmark for the AI Privacy Shield protection pipeline.

Runs the full pipeline (analyze -> protect -> evaluate -> robustness ->
quality) over a set of local images and prints an honest before/after report:

    * per-model detection confidence (OpenCV SSD, MTCNN, HOG, Faster R-CNN)
    * per-model embedding / vision-feature similarity
    * transformation robustness
    * visual quality (SSIM, PSNR, perturbation norms)
    * hardware, seed, and optimizer configuration (reproducibility)

Every number comes from the pipeline's real computation. Nothing is faked;
if a model is unavailable its row says "not tested".

Usage (from the project root):

    python scripts/benchmark.py                         # default fixture set
    python scripts/benchmark.py photo1.jpg photo2.jpg   # custom images
    python scripts/benchmark.py --iterations 20         # override GPU iters

Exit code 0 on success, 1 if any image failed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.cleanup.manager import get_store  # noqa: E402
from app.config import settings  # noqa: E402
from app.processing.pipeline import run_pipeline  # noqa: E402

PROJECT_ROOT = BACKEND.parent
DEFAULT_FIXTURES = sorted(
    list((PROJECT_ROOT / "tests" / "fixtures").glob("*.jpg"))
    + list((PROJECT_ROOT / "tests" / "fixtures").glob("*.png"))
)


def _noop_emit(_event: dict) -> None:
    pass


def _fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(before, after) -> str:
    if before is None or after is None or before == 0:
        return "—"
    return f"{(after / before - 1) * 100:+.1f}%"


def run_one(path: Path, iterations: int | None) -> tuple[str, dict] | tuple[str, None]:
    store = get_store()
    session_id = store.create()
    try:
        raw = path.read_bytes()
        ext = path.suffix.lower() or ".jpg"
        store.save_upload(session_id, raw, extension=ext)
        if iterations is not None:
            settings.OPT_ITERATIONS_GPU = iterations
            settings.OPT_ITERATIONS_CPU = max(iterations // 2, 4)
        t0 = time.perf_counter()
        result = run_pipeline(session_id, _noop_emit)
        dt = time.perf_counter() - t0
        print(f"\n  pipeline wall time: {dt:.1f}s")
        return path.name, result
    except Exception as exc:  # noqa: BLE001
        print(f"\n  FAILED: {exc}")
        return path.name, None
    finally:
        store.delete(session_id)


def print_image_report(name: str, r: dict) -> None:
    print("=" * 74)
    print(f"IMAGE: {name}")
    print("=" * 74)

    faces = r.get("faces", {})
    persons = r.get("person_count", 0)
    print(f"faces detected: {len(r.get('faces', []))}   persons detected: {persons}   "
          f"protection applied: {r.get('protection_applied')}")

    # ---- AI perception test ---------------------------------------------
    perc = r.get("perception") or {}
    rows: list[tuple[str, str, str, str, str]] = []
    f = perc.get("faces") or {}
    if f.get("tested"):
        rows.append(("Face detector (OpenCV SSD)", _fmt(f.get("before")), _fmt(f.get("after")),
                     _pct(f.get("before"), f.get("after")), "confidence"))
    m = perc.get("faces_mtcnn") or {}
    if m.get("tested"):
        rows.append(("Face detector (MTCNN cascade)", _fmt(m.get("before")), _fmt(m.get("after")),
                     _pct(m.get("before"), m.get("after")), "confidence"))
    for block, label in ((perc.get("persons") or {}, "Person detector (HOG)"),
                         (perc.get("persons_neural") or {}, "Person detector (Faster R-CNN)")):
        if block.get("tested") and block.get("before") is not None:
            rows.append((label, _fmt(block.get("before")), _fmt(block.get("after")),
                         _pct(block.get("before"), block.get("after")), "score"))
    for eid, emb in (perc.get("embeddings") or {}).items():
        label = emb.get("display_name", eid)
        rows.append((f"Embedding ({label})", _fmt(emb.get("before")), _fmt(emb.get("after")),
                     _pct(emb.get("before"), emb.get("after")), "similarity"))

    print("\n  AI PERCEPTION TEST (before -> after)")
    print(f"  {'model':38s} {'before':>10s} {'after':>10s} {'change':>10s}  metric")
    print("  " + "-" * 72)
    for label, b, a, ch, metric in rows:
        print(f"  {label:38s} {b:>10s} {a:>10s} {ch:>10s}  {metric}")
    if not rows:
        print("  (no perception measurements available)")

    # ---- robustness ------------------------------------------------------
    rob = r.get("robustness") or {}
    print("\n  ROBUSTNESS TEST")
    transforms = rob.get("transforms") or {}
    if transforms:
        print(f"  {'transformation':28s} {'verdict':>9s} {'mean disruption':>18s}")
        print("  " + "-" * 60)
        for tname, t in transforms.items():
            print(f"  {str(tname):28s} {str(t.get('verdict')):>9s} "
                  f"{_fmt(t.get('mean'), 3):>18s}")
        print(f"  overall: {rob.get('overall')}   ({rob.get('note', '')})")
    else:
        print("  (robustness not tested)")

    # ---- visual quality --------------------------------------------------
    q = r.get("quality") or {}
    print("\n  VISUAL QUALITY")
    print(f"    SSIM: {_fmt(q.get('ssim'), 3)}")
    print(f"    PSNR: {_fmt(q.get('psnr_db'), 1)} dB")
    print(f"    perturbation L-inf: {_fmt(q.get('perturbation_linf'), 3)}  "
          f"L2: {_fmt(q.get('perturbation_l2'), 3)}  MAE: {_fmt(q.get('mae'), 4)}")

    # ---- protection details ----------------------------------------------
    print("\n  PROTECTION")
    p = r.get("protection") or {}
    print(f"    iterations: {p.get('iterations')}  early stop: {p.get('early_stopped')}")
    ref = p.get("refinement") or {}
    if ref.get("applied"):
        print(f"    refinement: {ref.get('iterations')} iters over {', '.join(ref.get('target_models', []))}")
        print(f"    face confidence (refinement): {_fmt(ref.get('face_confidence_before'))} -> "
              f"{_fmt(ref.get('face_confidence_after'))}")
        if ref.get("mtcnn_confidence_before") is not None:
            print(f"    MTCNN confidence (refinement): {_fmt(ref.get('mtcnn_confidence_before'))} -> "
                  f"{_fmt(ref.get('mtcnn_confidence_after'))}")
    msg = r.get("faces_message") or ""
    if msg:
        print(f"    note: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="image paths (default: tests/fixtures/*)")
    parser.add_argument("--iterations", type=int, default=None, help="override phase-1 GPU iterations")
    args = parser.parse_args()

    paths = [Path(p) for p in args.images] if args.images else DEFAULT_FIXTURES
    if not paths:
        print("No images found. Pass image paths or add fixtures under tests/fixtures/.")
        return 1
    for p in paths:
        if not p.exists():
            print(f"Image not found: {p}")
            return 1

    print("AI PRIVACY SHIELD — BENCHMARK")
    print(f"  seed: {settings.OPT_SEED}   epsilon: {settings.PERTURBATION_EPSILON:.3f} "
          f"({settings.PERTURBATION_EPSILON * 255:.0f}/255)")
    print(f"  phase-1 iterations: GPU {settings.OPT_ITERATIONS_GPU} / CPU {settings.OPT_ITERATIONS_CPU}")
    print(f"  refinement: max {settings.REFINE_MAX_ITERS_GPU} iters (GPU) / "
          f"{settings.REFINE_MAX_ITERS_CPU} (CPU), "
          f"det-attack share {settings.DET_ATTACK_FRACTION:.0%}")
    print(f"  quality floors: SSIM >= {settings.MIN_SSIM}, PSNR >= {settings.MIN_PSNR} dB")
    print(f"  device: {settings.DEVICE}")

    agg: list[dict] = []
    failed = 0
    for path in paths:
        name, result = run_one(path, args.iterations)
        if result is None:
            failed += 1
            continue
        agg.append(result)
        print_image_report(name, result)

    # ---- aggregate -------------------------------------------------------
    if len(agg) > 1:
        print("\n" + "=" * 74)
        print("AGGREGATE (mean across images)")
        print("=" * 74)
        keys = ["facenet_vggface2", "facenet_casia", "arcface_mbf",
                "mobilenet_v3_large", "resnet50"]
        for eid in keys:
            befores, afters = [], []
            for r in agg:
                e = (r.get("perception") or {}).get("embeddings", {}).get(eid)
                if e and e.get("before") is not None and e.get("after") is not None:
                    befores.append(e["before"])
                    afters.append(e["after"])
            if befores:
                mb, ma = sum(befores) / len(befores), sum(afters) / len(afters)
                print(f"  {eid:24s} similarity {mb:.3f} -> {ma:.3f} "
                      f"({(ma / mb - 1) * 100:+.1f}%)")
        ssims = [r.get("quality", {}).get("ssim") for r in agg]
        psnrs = [r.get("quality", {}).get("psnr_db") for r in agg]
        if all(s is not None for s in ssims):
            print(f"  {'SSIM':24s} {sum(ssims) / len(ssims):.3f}")
        if all(p is not None for p in psnrs):
            print(f"  {'PSNR (dB)':24s} {sum(psnrs) / len(psnrs):.1f}")

    print(f"\n{len(agg)} succeeded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

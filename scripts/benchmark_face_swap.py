"""Real face-swap benchmark — INSwapper attack on original vs protected sources.

This is the critical test for the face-swap/identity-reference protection:

    original source face  -> INSwapper -> swapped face  (identity transfer HIGH)
    protected source face -> INSwapper -> swapped face  (identity transfer LOWER?)

Everything else is identical: same target image, same detection, same model.
Identity transfer is measured with the ArcFace embedding family the swap
pipelines themselves use (buffalo_l w600k_r50 via insightface) AND with a
different-family encoder (FaceNet VGGFace2 via our registry) as a held-out
view. The protected image is also evaluated after JPEG / resize transforms.

The protection step is the same unified engine the production app runs
(apply_editing_protection), which requires the model registry to be loaded
(load_models) — otherwise the identity/face-swap terms silently no-op.

Checkpointing: the expensive GPU protection result (protected PNG + metadata)
is written per source; a re-run skips protection for sources already done and
only re-runs the cheap CPU swap phase. A crash never loses the whole run.

Model / license notes (see docs/face-swap-protection.md):
  * inswapper_128.onnx — insightface; research evaluation use, contact
    insightface for licensing/distribution.
  * buffalo_l (det + w600k_r50) — insightface (research use).
  * SimSwap — CC-BY-NC 4.0; weights hosted on Google Drive, often
    unreachable from build machines -> reported NOT TESTED when absent.

Usage:
    python scripts/benchmark_face_swap.py [--out results/face_swap]
                                          [--transforms jpeg_compression,resize]
                                          [--device cuda]
                                          [images ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AIPS_PROFILE", "research")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.editing.protector import apply_editing_protection  # noqa: E402
from app.models.face_models import get_registry, load_models  # noqa: E402
from app.quality.metrics import compute_quality  # noqa: E402
from app.utils.imaging import load_and_normalize  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = sorted((PROJECT_ROOT / "data" / "benchmark").glob("*.jpg")) + [
    PROJECT_ROOT / "tests" / "fixtures" / "einstein.jpg"
]

# insightface model root (buffalo_l + inswapper live under models/)
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface_models"
INSWAPPER_PATH = PROJECT_ROOT / "models" / "inswapper" / "inswapper_128.onnx"


def _load_rgb(path: Path) -> np.ndarray:
    record = load_and_normalize(path.read_bytes(), path.stat().st_size)
    return record.array


class SwapEnvironment:
    """Lazily-built insightface detection + INSwapper + identity encoders."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.face_app = None
        self.swapper = None
        self.registry = None

    def ensure(self):
        if self.face_app is not None:
            return
        import insightface  # noqa: PLC0415
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        # load the registry FIRST (research profile: FaceNet VGGFace2 + CASIA
        # identity encoders, ArcFace refinement, MobileNetV3, held-out ResNet50)
        # — apply_editing_protection reads these; without load_all() the
        # identity/face-swap protection terms silently do nothing.
        load_models()
        self.registry = get_registry("cpu")

        self.face_app = FaceAnalysis(
            name="buffalo_l", root=str(INSIGHTFACE_ROOT), providers=["CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_PATH), providers=["CPUExecutionProvider"]
        )

    # ---- helpers ----------------------------------------------------------
    def detect(self, img: np.ndarray):
        self.ensure()
        return self.face_app.get(img)

    def swap(self, img: np.ndarray, target_face, source_face) -> np.ndarray:
        self.ensure()
        return self.swapper.get(img, target_face, source_face, paste_back=True)

    def arcface_sim(self, a_emb: np.ndarray, b_emb: np.ndarray) -> float:
        a = a_emb / (np.linalg.norm(a_emb) + 1e-12)
        b = b_emb / (np.linalg.norm(b_emb) + 1e-12)
        return float(np.dot(a, b))

    def facenet_sim(self, img: np.ndarray, box, ref_emb: np.ndarray) -> float | None:
        """FaceNet VGGFace2 cosine sim (held-out encoder family)."""
        try:
            model = self.registry._models.get("facenet_vggface2")
            if model is None or not model.info.loaded:
                return None
            import cv2  # noqa: PLC0415

            x1, y1, x2, y2 = [int(round(v)) for v in box]
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
            emb = model.embed_crops([crop], "cpu")[0]
            return float(np.dot(emb, ref_emb))
        except Exception:  # noqa: BLE001
            return None

    def transform(self, img: np.ndarray, name: str) -> np.ndarray:
        from app.robustness.tester import _apply_transform  # noqa: PLC0415

        return np.asarray(_apply_transform(name, img.astype(np.uint8)))


def _apply_transform_safe(env: SwapEnvironment, img, name):
    try:
        return env.transform(img, name)
    except Exception:  # noqa: BLE001
        return img


def protect_source(
    env: SwapEnvironment,
    source_img: np.ndarray,
    source_path: Path,
    out_dir: Path,
) -> tuple[np.ndarray | None, dict | None]:
    """Protect one source with the unified engine; cache per-source artifacts.

    Returns ``(protected_img, meta)`` where ``meta`` carries the protection
    quality/identity metrics. If a cached protection exists on disk it is
    loaded instead of re-running the GPU step.
    """
    prot_dir = out_dir / "protected"
    prot_dir.mkdir(parents=True, exist_ok=True)
    meta_path = prot_dir / f"{source_path.stem}.json"
    png_path = prot_dir / f"{source_path.stem}.png"

    if meta_path.exists() and png_path.exists():
        try:
            import cv2  # noqa: PLC0415

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None and "quality" in meta:
                print(f"  (cached protection {source_path.name})", flush=True)
                return img, meta
        except Exception as exc:  # noqa: BLE001
            print(f"  (cache read failed: {exc}; re-protecting)", flush=True)

    faces = env.detect(source_img)
    if not faces:
        return None, None
    source_face = max(faces, key=lambda f: f.det_score)
    box = source_face.bbox  # xyxy
    face_boxes = [tuple(int(round(v)) for v in b) for b in [box]]

    t0 = time.perf_counter()
    prot_result = apply_editing_protection(
        source_img, env.device, progress=lambda _i: None, face_boxes=face_boxes
    )
    protect_seconds = time.perf_counter() - t0
    if not prot_result.applied or prot_result.protected is None:
        return None, {"error": prot_result.note}

    protected_img = prot_result.protected
    quality = compute_quality(source_img, protected_img, device="cpu")
    meta = {
        "source": source_path.name,
        "protect_seconds": round(protect_seconds, 1),
        "face_boxes": face_boxes,
        "quality": {"ssim": round(quality.ssim, 4), "psnr_db": round(quality.psnr_db, 2)},
        "identity_similarity_face_encoders": {
            "before": prot_result.identity_similarity_before,
            "after": prot_result.identity_similarity_after,
        },
        "arcface_similarity": {
            "before": prot_result.arcface_similarity_before,
            "after": prot_result.arcface_similarity_after,
        },
        "refinement": prot_result.identity_refinement,
        "families": prot_result.families,
    }
    # cache for resume
    try:
        import cv2  # noqa: PLC0415

        cv2.imwrite(str(png_path), cv2.cvtColor(protected_img, cv2.COLOR_RGB2BGR))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  (checkpoint write failed: {exc})", flush=True)
    return protected_img, meta


def run_swap_attack(
    env: SwapEnvironment,
    source_img: np.ndarray,
    protected_img: np.ndarray,
    targets: list[tuple[str, np.ndarray]],
    transforms: tuple[str, ...],
) -> list[dict]:
    """Swap original / protected / protected+transform into every target."""
    faces = env.detect(source_img)
    if not faces:
        return [{"error": "no source face detected"}]
    source_face = max(faces, key=lambda f: f.det_score)
    src_emb = source_face.normed_embedding

    box = source_face.bbox
    facenet_ref = None
    try:
        m = env.registry._models.get("facenet_vggface2")
        if m is not None and m.info.loaded:
            import cv2  # noqa: PLC0415

            x1, y1, x2, y2 = [int(round(v)) for v in box]
            crop = cv2.resize(source_img[y1:y2, x1:x2], (160, 160))
            facenet_ref = m.embed_crops([crop], "cpu")[0]
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for tname, target_img in targets:
        target_faces = env.detect(target_img)
        if not target_faces:
            continue
        target_face = max(target_faces, key=lambda f: f.det_score)

        variants = {"original": source_img, "protected": protected_img}
        for tr in transforms:
            variants[f"protected+{tr}"] = _apply_transform_safe(env, protected_img, tr)

        for vname, vimg in variants.items():
            try:
                out = env.swap(target_img.copy(), target_face, source_face_of(vimg, source_face, env))
                out_faces = env.detect(out)
                if not out_faces:
                    rows.append(
                        {"target": tname, "variant": vname, "swap_success": 0.0, "note": "no face in output"}
                    )
                    continue
                out_face = max(out_faces, key=lambda f: f.det_score)
                rows.append(
                    {
                        "target": tname,
                        "variant": vname,
                        "arcface_source_sim": env.arcface_sim(out_face.normed_embedding, src_emb),
                        "arcface_target_sim": env.arcface_sim(out_face.normed_embedding, target_face.normed_embedding),
                        "facenet_source_sim": env.facenet_sim(out, out_face.bbox, facenet_ref)
                        if facenet_ref is not None
                        else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"target": tname, "variant": vname, "error": str(exc)[:200]})
    return rows


def source_face_of(img: np.ndarray, ref_face, env: SwapEnvironment):
    """Return the face object of ``img`` nearest to the reference bbox."""
    faces = env.detect(img)
    if not faces:
        raise RuntimeError("no face detected in variant")
    return max(faces, key=lambda f: f.det_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real INSwapper face-swap benchmark")
    parser.add_argument("images", nargs="*", help="source images (default: benchmark set)")
    parser.add_argument("--out", default="results/face_swap")
    parser.add_argument("--transforms", default="jpeg_compression,resize")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images] if args.images else DEFAULT_IMAGES
    transforms = tuple(t for t in args.transforms.split(",") if t)

    # targets = the OTHER images in the set (multiple target identities)
    targets = []
    for p in image_paths:
        img = _load_rgb(p)
        targets.append((p.stem, img))

    env = SwapEnvironment(args.device)
    env.ensure()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for src in image_paths:
        ckpt_path = ckpt_dir / f"{src.stem}.json"
        if ckpt_path.exists():
            print(f"=== {src.name} (resumed from checkpoint) ===", flush=True)
            continue
        print(f"=== {src.name} ===", flush=True)
        img = _load_rgb(src)
        other = [(n, t) for (n, t) in targets if n != src.stem]
        protected_img, meta = protect_source(env, img, src, out_dir)
        if protected_img is None or meta is None:
            r = {"source": src.name, "error": (meta or {}).get("error", "protection failed")}
        else:
            try:
                rows = run_swap_attack(env, img, protected_img, other, transforms)
                r = dict(meta)
                r["rows"] = rows
            except Exception as exc:  # noqa: BLE001
                r = dict(meta)
                r["error"] = str(exc)[:300]
        ckpt_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"  -> {ckpt_path.name} written", flush=True)

    # merge ALL checkpoints (not just this invocation) so a partial re-run
    # never clobbers completed sources
    for ckpt in sorted(ckpt_dir.glob("*.json")):
        with ckpt.open(encoding="utf-8") as f:
            results.append(json.load(f))
    (out_dir / "results.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )

    # ---- honest report ----------------------------------------------------
    lines = ["# AI PRIVACY SHIELD — REAL FACE-SWAP BENCHMARK (INSwapper)\n"]
    lines.append(
        "Identity transfer = ArcFace (buffalo_l w600k_r50) cosine of the swapped output vs the "
        "SOURCE face — the metric swap pipelines themselves maximize. Lower on the "
        "protected source = the attack transfers less identity. FaceNet VGGFace2 is a "
        "held-out encoder family (never used by the swap).\n"
    )
    agg_rows = []
    for r in results:
        if "error" in r and "rows" not in r:
            lines.append(f"## {r['source']}\n\nERROR: {r['error']}\n")
            continue
        lines.append(f"## {r['source']}\n")
        lines.append(
            f"- Quality: SSIM {r['quality']['ssim']} · PSNR {r['quality']['psnr_db']} dB · "
            f"protect {r['protect_seconds']}s\n"
        )
        ide = r.get("identity_similarity_face_encoders") or {}
        arc = r.get("arcface_similarity") or {}
        if ide.get("before") is not None and ide.get("after") is not None:
            lines.append(
                f"- Face encoders sim: {ide['before']:.3f} -> {ide['after']:.3f} · "
            )
        else:
            lines.append("- Face encoders sim: not available (models failed to load)\n")
        if arc.get("before") is not None and arc.get("after") is not None:
            lines.append(
                f"  ArcFace w600k sim: {arc['before']:.3f} -> {arc['after']:.3f}\n"
            )
        else:
            lines.append("  ArcFace w600k sim: not available\n")
        if r.get("error"):
            lines.append(f"  ERROR (swap phase): {r['error']}\n")
        lines.append("\n| Target | Variant | ArcFace src sim | ArcFace tgt sim | FaceNet src sim |\n|---|---|---|---|---|\n")
        for row in r.get("rows", []):
            if "error" in row:
                lines.append(f"| {row.get('target')} | {row.get('variant')} | error: {row['error']} | | |\n")
                continue
            if "swap_success" in row:
                lines.append(f"| {row['target']} | {row['variant']} | (no face in output) | | |\n")
                continue
            fs = row["facenet_source_sim"]
            fs_cell = f"{fs:.3f}" if fs is not None else "—"
            lines.append(
                f"| {row['target']} | {row['variant']} | {row['arcface_source_sim']:.3f} | "
                f"{row['arcface_target_sim']:.3f} | "
                f"{fs_cell} |\n"
            )
            if row["variant"] == "original":
                agg_rows.append({"source": r["source"], "target": row["target"], "sim": row["arcface_source_sim"]})
            elif row["variant"] == "protected":
                agg_rows.append({"source": r["source"], "target": row["target"], "prot_sim": row["arcface_source_sim"]})

    # aggregate: protected vs original identity transfer
    pairs = {}
    for row in agg_rows:
        key = (row["source"], row["target"])
        pairs.setdefault(key, {})
        if "sim" in row:
            pairs[key]["orig"] = row["sim"]
        if "prot_sim" in row:
            pairs[key]["prot"] = row["prot_sim"]
    valid = [(k, v) for k, v in pairs.items() if "orig" in v and "prot" in v]
    if valid:
        diffs = [v["prot"] - v["orig"] for _, v in valid]
        reduced = sum(1 for d in diffs if d < -0.01)
        lines.append("\n## Aggregate (ArcFace identity transfer, original vs protected)\n")
        lines.append(
            f"- Pairs: {len(valid)} · mean change: {np.mean(diffs):+.3f} · "
            f"median change: {np.median(diffs):+.3f} · rows reduced (Δ < -0.01): {reduced}/{len(valid)}\n"
        )
        lines.append("| Source | Target | Orig | Prot | Δ |\n|---|---|---|---:|---:|\n")
        for (src, tgt), v in sorted(pairs.items()):
            if "orig" in v and "prot" in v:
                lines.append(f"| {src} | {tgt} | {v['orig']:.3f} | {v['prot']:.3f} | {v['prot'] - v['orig']:+.3f} |\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"results": results, "aggregate": {"pairs": list(pairs.values())}}, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport: {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

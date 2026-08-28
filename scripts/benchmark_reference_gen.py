"""AI PRIVACY SHIELD — REFERENCE-CONDITIONED GENERATION BENCHMARK (IP-Adapter FaceID SD1.5)

FAMILY E (identity/reference-conditioned generation):

    reference photo -> identity encoder -> NEW synthetic image of the same person

The attacker never edits the protected photo directly; they use it as an identity
REFERENCE to synthesize a new image. We measure how much recognizable identity
survives when the ORIGINAL vs PROTECTED photo is used as the reference.

Identity transfer = ArcFace (buffalo_l w600k_r50) cosine between the generated
face and the ORIGINAL reference embedding. FaceNet VGGFace2 is a held-out encoder
family.

Models (research profile only):
  * h94/IP-Adapter-FaceID SD1.5 (non-commercial license, research evaluation use)
  * stable-diffusion-v1-5 (local cache)
  * insightface buffalo_l (detection + recognition)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import os
import sys

os.environ.setdefault("AIPS_PROFILE", "research")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.editing.protector import apply_editing_protection  # noqa: E402
from app.models.face_models import load_models  # noqa: E402
from app.quality.metrics import compute_quality  # noqa: E402
from app.utils.imaging import load_and_normalize  # noqa: E402

DEFAULT_IMAGES = sorted((PROJECT_ROOT / "data" / "benchmark").glob("*.jpg")) + [
    PROJECT_ROOT / "tests" / "fixtures" / "einstein.jpg"
]

INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface_models"
FACEID_CKPT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--h94--IP-Adapter-FaceID"
    / "snapshots"
)
# Shared protected-image cache — the face-swap benchmark (benchmark_face_swap.py)
# writes its protected PNGs into results/face_swap/protected/, so reference
# generation reuses the SAME protections instead of recomputing them.
PROTECTED_CACHE = PROJECT_ROOT / "results" / "face_swap" / "protected"

PROMPTS = [
    "professional studio portrait photo of a person, sharp focus",
    "outdoor casual portrait photo of a person, natural light",
    "portrait photo of a person, dramatic lighting, cinematic",
]
SEED = 42
STEPS = 25


def _load_rgb(path: Path) -> np.ndarray:
    record = load_and_normalize(path.read_bytes(), path.stat().st_size)
    return record.array


def _find_snapshot() -> Path:
    snaps = sorted(FACEID_CKPT.glob("*/ip-adapter-faceid_sd15.bin"))
    if not snaps:
        raise RuntimeError("FaceID weights not downloaded (h94/IP-Adapter-FaceID)")
    return snaps[-1]


class RefGenEnv:
    """Lazily-built insightface detection + FaceID generation pipeline."""

    def __init__(self) -> None:
        self.face_app = None
        self.pipe = None
        self.ip_model = None
        self.registry = None

    def ensure(self):
        if self.face_app is not None:
            return
        import insightface  # noqa: PLC0415
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        # identity/reference terms read the model registry — must load it or
        # the protection silently skips the identity objective
        load_models()

        self.face_app = FaceAnalysis(
            name="buffalo_l", root=str(INSIGHTFACE_ROOT), providers=["CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.registry = None

    def detect(self, img: np.ndarray):
        self.ensure()
        return self.face_app.get(img)

    def embed(self, img: np.ndarray):
        """w600k_r50 embedding of the largest face, normalized."""
        faces = self.detect(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.det_score)
        return face.normed_embedding.astype(np.float32)

    def ensure_pipe(self, device: str = "cuda"):
        if self.pipe is not None:
            return
        import torch  # noqa: PLC0415
        from diffusers import StableDiffusionPipeline  # noqa: PLC0415
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # noqa: PLC0415

        self.ensure()
        # local cache has no safety_checker weights (the production surrogate
        # never needs them); disable the safety checker so the pipeline loads
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        snap = _find_snapshot()
        self.ip_model = IPAdapterFaceID(pipe, str(snap), device, lora_rank=128, num_tokens=4)

    def generate(self, ref_emb: np.ndarray, prompt: str, device: str = "cuda"):
        import torch  # noqa: PLC0415

        self.ensure_pipe(device)
        emb = torch.from_numpy(ref_emb.astype(np.float32)).unsqueeze(0).half()
        images = self.ip_model.generate(
            faceid_embeds=emb,
            prompt=prompt,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            scale=1.0,
            num_samples=1,
            seed=SEED,
            guidance_scale=7.5,
            num_inference_steps=STEPS,
            height=512,
            width=512,
        )
        return np.array(images[0])  # RGB uint8 HxWx3

    @staticmethod
    def sim(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def facenet_sim(self, img: np.ndarray, ref_emb: np.ndarray) -> float | None:
        """Held-out encoder family (FaceNet VGGFace2)."""
        try:
            from app.models.face_models import get_registry, facenet_preprocess  # noqa: PLC0415

            reg = get_registry("cpu")
            model = reg._models.get("facenet_vggface2")
            if model is None or not model.info.loaded:
                return None
            import cv2  # noqa: PLC0415

            faces = self.detect(img)
            if not faces:
                return None
            f = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
            crop = cv2.resize(img[y1:y2, x1:x2], (160, 160))
            emb = model.embed_crops([crop], "cpu")[0]
            return float(np.dot(emb, ref_emb))
        except Exception:  # noqa: BLE001
            return None


def protect_or_cache(source_img: np.ndarray, name: str) -> tuple[np.ndarray, dict, float]:
    """Protect via the unified engine; reuse results/protected/ cache when present."""
    PROTECTED_CACHE.mkdir(parents=True, exist_ok=True)
    cache_png = PROTECTED_CACHE / f"{name}.png"
    if cache_png.exists():
        record = load_and_normalize(cache_png.read_bytes(), cache_png.stat().st_size)
        q = compute_quality(source_img, record.array, device="cpu")
        return record.array, {"ssim": round(q.ssim, 4), "psnr_db": round(q.psnr_db, 2)}, 0.0

    from app.vision.face_detector import get_face_detector  # noqa: PLC0415

    det_faces = get_face_detector().detect(source_img)
    box = det_faces[0] if det_faces else None
    face_boxes = [tuple(int(round(v)) for v in b) for b in ([box] if box else [])]
    t0 = time.perf_counter()
    prot = apply_editing_protection(
        source_img, "cuda", progress=lambda _i: None, face_boxes=face_boxes
    )
    dt = time.perf_counter() - t0
    if not prot.applied or prot.protected is None:
        raise RuntimeError(prot.note)
    q = compute_quality(source_img, prot.protected, device="cpu")
    # cache the protected image for reuse by other benchmarks
    try:
        import cv2  # noqa: PLC0415

        cv2.imwrite(str(cache_png), cv2.cvtColor(prot.protected.astype(np.uint8), cv2.COLOR_RGB2BGR))
    except Exception:  # noqa: BLE001
        pass
    return prot.protected, {"ssim": round(q.ssim, 4), "psnr_db": round(q.psnr_db, 2)}, dt


def evaluate_source(env: RefGenEnv, img: np.ndarray, name: str) -> dict:
    faces = env.detect(img)
    if not faces:
        return {"error": "no source face detected", "source": name}
    orig_emb = env.embed(img)
    if orig_emb is None:
        return {"error": "no source face detected", "source": name}

    protected, quality, protect_seconds = protect_or_cache(img, name)
    prot_emb = env.embed(protected)
    if prot_emb is None:
        return {"error": "no face detected in protected image", "source": name}

    # FaceNet held-out reference on the ORIGINAL crop
    facenet_ref = None
    try:
        from app.models.face_models import get_registry  # noqa: PLC0415
        import cv2  # noqa: PLC0415

        reg = get_registry("cpu")
        m = reg._models.get("facenet_vggface2")
        if m is not None and m.info.loaded:
            f = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
            crop = cv2.resize(img[y1:y2, x1:x2], (160, 160))
            facenet_ref = m.embed_crops([crop], "cpu")[0]
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for prompt in PROMPTS:
        row = {"prompt": prompt[:40]}
        try:
            gen_orig = env.generate(orig_emb, prompt)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)[:200]
            rows.append(row)
            continue
        # identity transfer of the ORIGINAL-reference generation (baseline)
        gen_emb = env.embed(gen_orig)
        row["orig_arcface_sim"] = (
            round(env.sim(orig_emb, gen_emb), 4) if gen_emb is not None else None
        )
        _fs = env.facenet_sim(gen_orig, facenet_ref) if facenet_ref is not None else None
        row["orig_facenet_sim"] = round(_fs, 4) if _fs is not None else None
        # identity transfer when the PROTECTED photo is the reference
        try:
            gen_prot = env.generate(prot_emb, prompt)
        except Exception as exc:  # noqa: BLE001
            row["prot_error"] = str(exc)[:200]
            rows.append(row)
            continue
        gen_emb = env.embed(gen_prot)
        row["prot_arcface_sim"] = (
            round(env.sim(orig_emb, gen_emb), 4) if gen_emb is not None else None
        )
        _fs = env.facenet_sim(gen_prot, facenet_ref) if facenet_ref is not None else None
        row["prot_facenet_sim"] = round(_fs, 4) if _fs is not None else None
        rows.append(row)

    return {
        "source": name,
        "protect_seconds": round(protect_seconds, 1),
        "quality": quality,
        "reference_embedding_sim": {
            "before": round(env.sim(orig_emb, orig_emb), 4),
            "after": round(env.sim(orig_emb, prot_emb), 4),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="IP-Adapter FaceID reference-generation benchmark")
    parser.add_argument("images", nargs="*")
    parser.add_argument("--out", default="results/reference_gen")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images] if args.images else DEFAULT_IMAGES
    env = RefGenEnv()
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
        try:
            r = evaluate_source(env, _load_rgb(src), src.stem)
        except Exception as exc:  # noqa: BLE001
            r = {"source": src.stem, "error": str(exc)[:300]}
        ckpt_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    # merge ALL checkpoints (not just this invocation) so a partial re-run
    # never clobbers previously completed sources
    for ckpt in sorted(ckpt_dir.glob("*.json")):
        with ckpt.open(encoding="utf-8") as f:
            results.append(json.load(f))
    (out_dir / "results.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )

    lines = [
        "# AI PRIVACY SHIELD — REFERENCE-CONDITIONED GENERATION BENCHMARK (IP-Adapter FaceID SD1.5)\n",
        "\nReference photo -> identity encoder -> NEW synthetic image of the same person. "
        "Identity transfer = ArcFace (buffalo_l w600k_r50) cosine of the generated face vs the "
        "ORIGINAL reference embedding. FaceNet VGGFace2 is a held-out encoder family.\n",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"\n## {r['source']}\n\nERROR: {r['error']}\n")
            continue
        lines.append(f"\n## {r['source']}\n")
        lines.append(
            f"- Quality: SSIM {r['quality']['ssim']} · PSNR {r['quality']['psnr_db']} dB · "
            f"protect {r['protect_seconds']}s\n"
        )
        lines.append(
            f"- Reference embedding (ArcFace): "
            f"{r['reference_embedding_sim']['before']:.3f} -> {r['reference_embedding_sim']['after']:.3f}\n"
        )
        lines.append(
            "\n| Prompt | Orig ArcFace | Prot ArcFace | Δ | Orig FaceNet | Prot FaceNet |\n"
            "|---|---|---:|---:|---:|---:|---:|\n"
        )
        for row in r["rows"]:
            if "error" in row:
                lines.append(f"| {row['prompt']} | error: {row['error']} | | | | |\n")
                continue
            if "prot_error" in row:
                lines.append(f"| {row['prompt']} | {row.get('orig_arcface_sim')} | prot error: {row['prot_error']} | | | |\n")
                continue
            oa, pa = row["orig_arcface_sim"], row["prot_arcface_sim"]
            d = (pa - oa) if (oa is not None and pa is not None) else None
            dcell = f"{d:+.3f}" if d is not None else "—"
            lines.append(
                f"| {row['prompt']} | {oa} | {pa} | {dcell} | "
                f"{row['orig_facenet_sim']} | {row['prot_facenet_sim']} |\n"
            )

    # aggregate
    diffs, pairs = [], []
    for r in results:
        if "rows" not in r:
            continue
        for row in r["rows"]:
            oa, pa = row.get("orig_arcface_sim"), row.get("prot_arcface_sim")
            if oa is not None and pa is not None:
                diffs.append(pa - oa)
                pairs.append((r["source"], row["prompt"], oa, pa))
    if diffs:
        reduced = sum(1 for d in diffs if d < -0.01)
        lines.append("\n## Aggregate (identity transfer, original vs protected reference)\n")
        lines.append(
            f"- Generations: {len(diffs)} · mean Δ: {np.mean(diffs):+.3f} · "
            f"median Δ: {np.median(diffs):+.3f} · rows reduced (Δ < -0.01): {reduced}/{len(diffs)}\n"
        )
        lines.append("| Source | Prompt | Orig | Prot | Δ |\n|---|---|---|---:|---:|\n")
        for src, pr, oa, pa in pairs:
            lines.append(f"| {src} | {pr} | {oa} | {pa} | {pa - oa:+.3f} |\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nreport: {out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Download all local models required by AI Privacy Shield.

Usage (from the ``backend`` directory):

    python scripts/download_models.py            # everything
    python scripts/download_models.py --face     # OpenCV face detector only
    python scripts/download_models.py --arcface  # optional ArcFace verification model

Nothing here uses paid services. All models are publicly licensed research
models (see models/README.md for license notes).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODELS_DIR, settings  # noqa: E402

FACE_PROTO = "deploy.prototxt"
FACE_CAFFE = "res10_300x300_ssd_iter_140000.caffemodel"
ARCFACE_MODEL = "w600k_mbf.onnx"  # MobileFaceNet ArcFace, ~12 MB (buffalo_s pack)
ARCFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"


def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {label} already present: {dest}")
        return
    print(f"[download] {label} -> {dest}  ({url})")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Privacy-Shield/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
        print(f"[ok] {label} saved ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {label}: {exc}") from exc


def ensure_face_detector(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    _download(settings.FACE_DETECTOR_PROTO_URL, dest_dir / FACE_PROTO, "face detector prototxt")
    _download(settings.FACE_DETECTOR_CAFFE_URL, dest_dir / FACE_CAFFE, "face detector caffemodel")


def ensure_arcface(dest_dir: Path) -> None:
    """Download the buffalo_s pack and extract the MobileFaceNet ArcFace onnx."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / ARCFACE_MODEL
    if model_path.exists() and model_path.stat().st_size > 0:
        print(f"[skip] ArcFace already present: {model_path}")
        return
    zip_path = dest_dir / "buffalo_s.zip"
    _download(ARCFACE_URL, zip_path, "buffalo_s model pack")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            member = next(n for n in names if n.endswith(ARCFACE_MODEL))
            with zf.open(member) as src, open(model_path, "wb") as out:
                shutil.copyfileobj(src, out)
        zip_path.unlink()
        print(f"[ok] ArcFace extracted -> {model_path}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to extract ArcFace model: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AI Privacy Shield local models.")
    parser.add_argument("--face", action="store_true", help="only the OpenCV face detector")
    parser.add_argument("--arcface", action="store_true", help="only the ArcFace verification model")
    args = parser.parse_args()

    face_dir = settings.FACE_DETECTOR_DIR
    arcface_dir = MODELS_DIR / "arcface"

    try:
        if args.arcface:
            ensure_arcface(arcface_dir)
        elif args.face:
            ensure_face_detector(face_dir)
        else:
            ensure_face_detector(face_dir)
            ensure_arcface(arcface_dir)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nModel setup complete.")


if __name__ == "__main__":
    main()

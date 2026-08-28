"""Download the research face-swap / reference-generation models.

These models are ONLY used by the research benchmarks (scripts/benchmark_face_swap.py,
scripts/benchmark_reference_gen.py). The production application never loads them.

  * INSwapper (inswapper_128.onnx) — insightface; research evaluation use, contact
    insightface for licensing/distribution.
  * buffalo_l pack — insightface detection + ArcFace recognition (research use).
  * h94/IP-Adapter-FaceID SD1.5 adapter (non-commercial, research evaluation use).

Usage:  ./backend/.venv/Scripts/python scripts/download_face_swap_models.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INSWAPPER_URL = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- INSwapper -------------------------------------------------------
    inswapper = models_dir / "inswapper" / "inswapper_128.onnx"
    if not inswapper.exists():
        inswapper.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading INSwapper -> {inswapper}")
        import urllib.request

        urllib.request.urlretrieve(INSWAPPER_URL, inswapper)
    print(f"INSwapper: {'OK' if inswapper.exists() else 'MISSING'} ({inswapper.stat().st_size/1e6:.0f} MB)")

    # ---- buffalo_l -------------------------------------------------------
    root = models_dir / "insightface_models" / "models" / "buffalo_l"
    if not (root / "w600k_r50.onnx").exists():
        z = models_dir / "buffalo_l.zip"
        print(f"downloading buffalo_l -> {z}")
        import urllib.request

        urllib.request.urlretrieve(BUFFALO_L_URL, z)
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        z.unlink()
    files = sorted(p.name for p in root.glob("*.onnx"))
    print(f"buffalo_l: OK ({', '.join(files)})")

    # ---- IP-Adapter FaceID SD1.5 ----------------------------------------
    from huggingface_hub import snapshot_download

    snap = snapshot_download("h94/IP-Adapter-FaceID", allow_patterns=["ip-adapter-faceid_sd15.bin"])
    print(f"IP-Adapter-FaceID: OK ({snap})")


if __name__ == "__main__":
    main()

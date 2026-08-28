#!/usr/bin/env python
"""Check that the AI Privacy Shield backend environment is ready.

Usage (from the ``backend`` directory):

    .venv\\Scripts\\python scripts/check_environment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    print("AI Privacy Shield — environment check\n" + "-" * 50)
    ok = True

    try:
        import torch  # noqa: PLC0415

        print(f"[ok] torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[ok] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[warn] CUDA not available — CPU mode will be used (slower).")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] torch: {exc}")
        ok = False

    try:
        import cv2  # noqa: PLC0415

        print(f"[ok] OpenCV {cv2.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] opencv: {exc}")
        ok = False

    try:
        import facenet_pytorch  # noqa: PLC0415, F401

        print("[ok] facenet-pytorch")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] facenet-pytorch: {exc}")
        ok = False

    from app.config import settings  # noqa: PLC0415

    face_proto = settings.FACE_DETECTOR_DIR / "deploy.prototxt"
    face_caffe = settings.FACE_DETECTOR_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    if face_proto.exists() and face_caffe.exists():
        print("[ok] face detector models present")
    else:
        print("[warn] face detector models MISSING — run: python scripts/download_models.py")
        ok = False

    arcface = settings.MODELS_DIR / "arcface" / "w600k_mbf.onnx"
    if arcface.exists():
        print("[ok] ArcFace verification model present")
    else:
        print("[warn] ArcFace model missing (optional, verification only): "
              "python scripts/download_models.py --arcface")

    try:
        import onnxruntime  # noqa: PLC0415

        print(f"[ok] onnxruntime {onnxruntime.__version__} "
              f"providers={onnxruntime.get_available_providers()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] onnxruntime: {exc}")
        ok = False

    try:
        import rapidocr_onnxruntime  # noqa: PLC0415, F401

        print("[ok] RapidOCR (text detection) installed — models download on first use")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] RapidOCR not installed: {exc}")

    print("-" * 50)
    if ok:
        print("Environment looks ready. Start the backend with:\n"
              "    .venv\\Scripts\\python -m uvicorn app.main:app --port 8000")
    else:
        print("Environment has issues — see messages above and models/README.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()

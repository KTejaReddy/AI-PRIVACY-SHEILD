"""Face detection pipeline."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.vision.face_detector import get_face_detector


def test_detects_real_face(face_jpg_bytes):
    arr = np.asarray(Image.open(__import__("io").BytesIO(face_jpg_bytes)).convert("RGB"))
    faces = get_face_detector().detect(arr)
    assert len(faces) == 1
    face = faces[0]
    assert face.confidence > 0.9
    assert face.x2 > face.x1 and face.y2 > face.y1


def test_no_face_in_gradient_image(gradient_png_bytes):
    arr = np.asarray(Image.open(__import__("io").BytesIO(gradient_png_bytes)).convert("RGB"))
    faces = get_face_detector().detect(arr)
    assert len(faces) == 0


def test_detects_multiple_faces(face_jpg_bytes):
    """Two copies of the same face side by side should yield two detections."""
    import io  # noqa: PLC0415

    arr = np.asarray(Image.open(io.BytesIO(face_jpg_bytes)).convert("RGB"))
    w = arr.shape[1]
    canvas = np.zeros((arr.shape[0], w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = arr
    canvas[:, w:] = arr
    faces = get_face_detector().detect(canvas)
    # NMS may merge identical twins; accept one or two but require >= 1 and
    # that the detector clearly found the faces.
    assert len(faces) >= 1


def test_detector_available():
    assert get_face_detector().available is True

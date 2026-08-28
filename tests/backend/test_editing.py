"""AI-editing protection + benchmark — unit tests with fakes.

The heavy models (InstructPix2Pix, SD1.5 inpainting/img2img, CLIP) are never
loaded here; the benchmark run/scoring logic is exercised with fake editors,
and the availability gating / disabled-by-configuration paths are tested
directly. Task-specific pixel metrics are tested on synthetic images.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.editing.protector import EditingProtectionResult, apply_editing_protection
from app.editing.tasks import DEFAULT_TASKS, tasks_for
from app.evaluation import task_metrics
from app.evaluation.editing_benchmark import EditingBenchmark, _relative_change


# ---- tasks ----------------------------------------------------------------


def test_task_definitions_complete():
    ids = {t.id for t in DEFAULT_TASKS}
    assert len(ids) == len(DEFAULT_TASKS)  # unique
    for t in DEFAULT_TASKS:
        assert t.instruction and t.target and t.name
        assert t.metric in task_metrics.TASK_METRICS


def test_tasks_for_subset_preserves_order():
    out = tasks_for(("t05_sketch", "t01_shirt_color"))
    assert [t.id for t in out] == ["t05_sketch", "t01_shirt_color"]
    out = tasks_for(("t01_shirt_color", "nope", "t01_shirt_color"))
    assert [t.id for t in out] == ["t01_shirt_color"]


def test_tasks_for_none_returns_all():
    assert len(tasks_for(None)) == len(DEFAULT_TASKS)


# ---- task-specific metrics (synthetic images) ------------------------------


def _solid(color: tuple[int, int, int], size=64):
    return np.full((size, size, 3), color, dtype=np.uint8)


def test_shirt_color_metric_reds():
    faces = [(24, 8, 40, 24)]  # face at top
    gray_shirt = np.full((64, 64, 3), 120, dtype=np.uint8)
    red_shirt = gray_shirt.copy()
    red_shirt[24:48, 8:56] = (200, 40, 40)
    raw, success = task_metrics.measure("shirt_color", gray_shirt, red_shirt, faces, None)
    assert raw > 0.05  # redness increased
    assert success > 0.0
    # unchanged -> zero success
    _, success0 = task_metrics.measure("shirt_color", gray_shirt, gray_shirt, faces, None)
    assert success0 == pytest.approx(0.0, abs=1e-6)


def test_background_metric_changes_background():
    faces = [(20, 20, 44, 44)]
    in_img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = in_img.copy()
    out[0:20, 0:64] = (90, 160, 220)  # replace top band (background)
    _, success = task_metrics.measure("background", in_img, out, faces, None)
    assert success > 0.0
    # changing only the face region should NOT count as background change
    out2 = in_img.copy()
    out2[20:44, 20:44] = (255, 0, 0)
    _, success2 = task_metrics.measure("background", in_img, out2, faces, None)
    assert success2 < success


def test_hair_region_is_above_face():
    h, w = 100, 100
    box = task_metrics.hair_region(h, w, faces=[(30, 40, 70, 80)])
    x1, y1, x2, y2 = box
    assert y1 < 40  # starts above the face
    assert y2 <= 80
    assert x1 < 30 and x2 > 70  # wider than the face


def test_mask_kinds_produce_valid_masks():
    h, w = 100, 100
    for kind in task_metrics.MASK_KINDS:
        mask = task_metrics.make_mask(kind, h, w, faces=[(30, 30, 60, 60)], persons=[(20, 20, 80, 90)])
        assert mask.shape == (h, w)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0 and mask.max() <= 1.0
        assert mask.sum() > 0  # never an empty mask


def test_relative_change_is_none_when_unmeaningful():
    assert _relative_change(0.05, 0.02) == pytest.approx(60.0)
    assert _relative_change(0.01, 0.0) is None  # original success near zero


# ---- benchmark run with fake editors ---------------------------------------


class _FakeEmb:
    def __init__(self, arr):
        self.arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeClipOutput:
    def __init__(self, image_emb, text_emb):
        self.image_embeds = _FakeEmb(image_emb)
        self.text_embeds = _FakeEmb(text_emb)


class _FakeInputs(dict):
    def __init__(self, **kw):
        super().__init__(kw)

    def to(self, device):
        return self


class _FakeClip:
    """Deterministic 'CLIP': darker images are closer to the target embedding."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def __call__(self, **kwargs):
        images = kwargs["images"]
        arr = np.asarray(images).astype(np.float32)
        dark = 1.0 - float(arr.mean()) / 255.0
        img_emb = np.zeros((1, self.dim), dtype=np.float32)
        img_emb[0, 0] = dark
        text_emb = np.zeros((1, self.dim), dtype=np.float32)
        text_emb[0, 0] = 1.0
        return _FakeClipOutput(img_emb, text_emb)


class _FakeEditor:
    """Always 'edits' the input to a dark uniform image (closer to the target)."""

    def __init__(self, dark_value: int = 40):
        self.dark_value = dark_value

    def edit(self, request):
        return np.full_like(request.image_rgb, self.dark_value)


class _FakePassThroughEditor:
    """Returns the input unchanged — protection that changes nothing."""

    def edit(self, request):
        return request.image_rgb.copy()


class _FakeManager:
    def __init__(self, clip, editor):
        self.clip = clip
        self.editor = editor
        self.clip_processor = _FakeInputs

    def get_clip(self, *a, **k):
        return self.clip

    def get_ip2p(self, *a, **k):
        return self.editor

    def get_editor(self, *a, **k):
        return self.editor


def _gradient_image(size=64):
    x = np.linspace(0, 255, size, dtype=np.uint8)
    arr = np.stack([x, 255 - x, np.full(size, 128, dtype=np.uint8)], axis=-1)
    return np.repeat(arr[np.newaxis, :, :], size, axis=0).reshape(size, size, 3)


def test_run_protection_lowers_edit_success():
    """When the protected edit moves less toward the target, absolute change > 0."""
    bench = EditingBenchmark("cpu")
    bench.manager = _FakeManager(_FakeClip(), _FakeEditor())
    img = _gradient_image()
    # original is bright; the fake editor darkens it (success). The "protected"
    # image is already dark, so the same edit changes little (lower success).
    original = img
    protected = np.full_like(img, 60)

    result = bench.run(original, protected, task_ids=("t01_shirt_color",))
    assert result.available
    assert result.tasks
    t = result.tasks[0]
    assert t.success_original > t.success_protected
    assert t.absolute_change > 0
    assert t.relative_change_pct is not None
    assert result.tasks_reduced == len(result.tasks)  # every row reduced
    assert result.mean_absolute_change > 0


def test_run_honest_when_no_reduction():
    """A protection that changes nothing must NOT fabricate a positive change."""
    bench = EditingBenchmark("cpu")
    bench.manager = _FakeManager(_FakeClip(), _FakePassThroughEditor())
    img = _gradient_image()
    result = bench.run(img, img.copy(), task_ids=("t01_shirt_color",))
    t = result.tasks[0]
    assert t.absolute_change == pytest.approx(0.0, abs=1e-6)
    assert t.success_original == pytest.approx(t.success_protected, abs=1e-6)


def test_run_reports_raw_components():
    """Task-specific pixel metric and raw CLIP deltas are visible (not hidden)."""
    bench = EditingBenchmark("cpu")
    bench.manager = _FakeManager(_FakeClip(), _FakeEditor())
    img = _gradient_image()
    result = bench.run(img, np.full_like(img, 60), task_ids=("t01_shirt_color",))
    t = result.tasks[0]
    d = t.as_dict()
    for key in (
        "task_metric_original", "task_metric_protected",
        "clip_delta_original", "clip_delta_protected",
        "absolute_change", "success_original", "success_protected",
    ):
        assert key in d
    assert "relative_change_pct" in d
    assert d["mask_kind"] is None or isinstance(d["mask_kind"], str)


def test_run_multiple_editors_and_masks(monkeypatch):
    """All three editor types run; inpainting rows carry a real mask kind."""
    bench = EditingBenchmark("cpu")
    bench.manager = _FakeManager(_FakeClip(), _FakeEditor())
    result = bench.run(
        _gradient_image(), np.full_like(_gradient_image(), 60),
        face_boxes=[(20, 20, 44, 44)], person_boxes=[(10, 10, 54, 60)],
        task_ids=("t01_shirt_color", "t05_sketch"),
    )
    editors = {t.editor_type for t in result.tasks}
    assert "instruction" in editors
    assert "inpainting" in editors
    # t05 sketch is an img2img task
    i2i = [t for t in result.tasks if t.editor_type == "image2image"]
    assert i2i and i2i[0].task_id == "t05_sketch"
    # every inpainting row has a real mask kind (never a mock center mask)
    for t in result.tasks:
        if t.editor_type == "inpainting":
            assert t.mask_kind in task_metrics.MASK_KINDS


def test_robustness_measures_edit_success_after_transforms(monkeypatch):
    bench = EditingBenchmark("cpu")
    bench.manager = _FakeManager(_FakeClip(), _FakeEditor())
    monkeypatch.setattr(settings, "EDITING_ROBUSTNESS_ENABLED", True)
    rows = bench.run_robustness(
        _gradient_image(), _gradient_image(),
        task_ids=("t01_shirt_color",),
        transforms=("jpeg_compression", "resize"),
    )
    assert len(rows) == 2
    for r in rows:
        assert r["transform"] in ("jpeg_compression", "resize")
        assert len(r["tasks"]) == 1
        assert "success_original" in r["tasks"][0]
        assert "success_after_transform" in r["tasks"][0]


# ---- availability / disabled paths ----------------------------------------


def test_benchmark_disabled_by_config(monkeypatch):
    monkeypatch.setattr(settings, "EDITING_ENABLED", False)
    bench = EditingBenchmark("cpu")
    result = bench.run(_gradient_image(), _gradient_image())
    assert result.available is False
    assert "disabled" in result.note.lower()


def test_benchmark_unavailable_clip(monkeypatch):
    bench = EditingBenchmark("cpu")

    class _NoClip:
        def __init__(self, *a, **k):
            raise RuntimeError("clip not downloaded")

    bench.manager.get_clip = _NoClip
    result = bench.run(_gradient_image(), _gradient_image())
    assert result.available is False
    assert "unavailable" in result.note.lower()


def test_editing_available_maps_clip_key(monkeypatch):
    """available() reports CLIP under its repo id AND the legacy 'clip' alias."""
    bench = EditingBenchmark("cpu")
    av = bench.manager.available()
    assert av.get("clip") == av.get("openai/clip-vit-large-patch14")
    assert "openai/clip-vit-large-patch14" in av


def test_benchmark_no_matching_tasks(monkeypatch):
    monkeypatch.setattr(settings, "EDITING_BENCHMARK_ENABLED", True)
    bench = EditingBenchmark("cpu")
    result = bench.run(_gradient_image(), _gradient_image(), task_ids=("does_not_exist",))
    assert result.available is True
    assert result.tasks == []
    assert "No editing tasks" in result.note


def test_editing_protection_disabled_by_config(monkeypatch):
    monkeypatch.setattr(settings, "EDITING_ENABLED", False)
    result = apply_editing_protection(_gradient_image(), "cpu")
    assert result.applied is False
    assert "disabled" in result.note.lower()


def test_editing_protection_surrogate_unavailable(monkeypatch):
    """If the surrogate cannot load (no local SD1.5), the stage reports honestly."""
    monkeypatch.setattr(settings, "EDITING_ENABLED", True)
    monkeypatch.setattr(settings, "EDITING_SURROGATE_ENABLED", True)

    class _Boom:
        loaded = False
        model_id = "fake/sd15"

        def load(self):
            raise RuntimeError("missing model")

        def unload(self):
            pass

    result = apply_editing_protection(_gradient_image(), "cpu", surrogate=_Boom())
    assert result.applied is False
    assert "unavailable" in result.note.lower()


def test_editing_protection_result_serializable():
    r = EditingProtectionResult(applied=True, iterations=4, epsilon=0.01)
    d = r.as_dict()
    assert d["applied"] is True
    assert d["iterations"] == 4
    assert "objective" in d

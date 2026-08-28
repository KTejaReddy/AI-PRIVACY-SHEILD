"""Tests for the AI Attack Family Registry and the multi-family protector."""
from __future__ import annotations

import pytest

from app.attack_registry import (
    ATTACK_MODELS,
    FAMILY_INFO,
    AttackFamily,
    ModelRole,
    families_report,
    load_profile,
)


def test_families_a_through_i_are_registered():
    ids = {f.value for f in AttackFamily}
    assert ids == {
        "diffusion_editing",
        "inpainting",
        "instruction_editing",
        "image_to_image",
        "identity_reference",
        "face_swap",
        "image_to_video",
        "vlm_conditioning",
        "vision_encoder",
    }
    assert len(FAMILY_INFO) == 9


def test_key_models_registered_with_roles():
    assert ATTACK_MODELS["stable-diffusion-v1-5/stable-diffusion-v1-5"].role == ModelRole.OPTIMIZATION
    assert ATTACK_MODELS["timbrooks/instruct-pix2pix"].role == ModelRole.HELD_OUT
    assert ATTACK_MODELS["stable-diffusion-v1-5/stable-diffusion-inpainting"].role == ModelRole.EVALUATION
    assert ATTACK_MODELS["arcface_mbf"].role == ModelRole.HELD_OUT
    assert ATTACK_MODELS["resnet50"].role == ModelRole.HELD_OUT
    assert ATTACK_MODELS["facenet_vggface2"].role == ModelRole.OPTIMIZATION
    assert AttackFamily.DIFFUSION_EDITING.value in ATTACK_MODELS[
        "stable-diffusion-v1-5/stable-diffusion-v1-5"
    ].families


def test_production_profile_is_minimal():
    prof = load_profile("production")
    # only the protection engine, no benchmark editors
    assert "timbrooks/instruct-pix2pix" not in prof.active_models
    assert "stable-diffusion-v1-5/stable-diffusion-v1-5" in prof.active_models
    assert "facenet_vggface2" in prof.active_models
    assert prof.flags.get("editing_benchmark") is False


def test_research_profile_includes_benchmark_models():
    prof = load_profile("research")
    assert "timbrooks/instruct-pix2pix" in prof.active_models
    assert "stable-diffusion-v1-5/stable-diffusion-inpainting" in prof.active_models
    assert "openai/clip-vit-large-patch14" in prof.active_models
    assert prof.flags.get("editing_benchmark") is True
    assert prof.flags.get("red_team_rounds", 0) >= 1


def test_families_report_shape():
    rep = families_report("production")
    assert rep["profile"] == "production"
    fam_ids = {f["id"] for f in rep["families"]}
    assert "diffusion_editing" in fam_ids
    assert "identity_reference" in fam_ids
    for fam in rep["families"]:
        assert fam["name"]
        assert fam["mechanism"]
        assert isinstance(fam["models"], list)
        for m in fam["models"]:
            assert m["role"] in {"optimization", "evaluation", "held_out"}


def test_video_family_documented_unavailable_not_faked():
    m = ATTACK_MODELS["image_to_video_adapter"]
    assert m.local is False
    assert "NOT TESTED" in m.note or "not" in m.note.lower()


def test_multi_family_protection_result_reports_families_and_weights():
    from app.editing.protector import EditingProtectionResult

    r = EditingProtectionResult(
        applied=True,
        families=["diffusion_editing", "identity_reference", "face_swap", "vision_encoder"],
        weights={"diffusion": 1.0, "identity": 0.6, "vision": 0.35},
        identity_similarity_before=0.95,
        identity_similarity_after=0.5,
    )
    d = r.as_dict()
    assert d["families"] == r.families
    assert d["weights"]["identity"] == 0.6
    assert d["identity_similarity_before"] == 0.95
    assert "multi-family perturbation" in d["objective"]

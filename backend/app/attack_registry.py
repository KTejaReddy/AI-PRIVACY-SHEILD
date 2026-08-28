"""AI Attack Family Registry.

Classifies every protection-relevant model in the system by the *AI attack
family* it represents (spec §3, families A–I) and by its role in the
protection pipeline:

    OPTIMIZATION  -> participates in generating the perturbation
    EVALUATION    -> tests the perturbation, never part of its generation
    HELD_OUT      -> never exposed to the optimizer (transferability test)

The registry is the single source of truth for:

* which attack families exist and what they target,
* which representative local models cover each family,
* which models the *production* protection engine actually needs
  (``configs/production.yaml``) vs. the *research* benchmark environment
  (``configs/research.yaml``).

The registry deliberately does NOT model commercial products ("OpenAI
blocker", "Midjourney blocker", ...). AI systems are grouped by their
underlying manipulation mechanism, so new model families can be added by
registering a representative local adapter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .config import BACKEND_DIR

logger = logging.getLogger(__name__)

CONFIGS_DIR = BACKEND_DIR / "configs"


class AttackFamily(str, Enum):
    """The attack families this system targets (spec §3)."""

    DIFFUSION_EDITING = "diffusion_editing"          # A — latent diffusion editing pipelines
    INPAINTING = "inpainting"                        # B — masked inpainting / partial replacement
    INSTRUCTION_EDITING = "instruction_editing"      # C — image + text instruction -> edited image
    IMAGE_TO_IMAGE = "image_to_image"                # D — source image + prompt/style -> new image
    IDENTITY_REFERENCE = "identity_reference"        # E — reference photo -> identity-conditioned generation
    FACE_SWAP = "face_swap"                          # F — source face -> identity transfer
    IMAGE_TO_VIDEO = "image_to_video"                # G — reference image + motion -> synthetic video
    VLM_CONDITIONING = "vlm_conditioning"            # H — vision-language conditioned generation
    VISION_ENCODER = "vision_encoder"                # I — general learned visual representations


class ModelRole(str, Enum):
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    HELD_OUT = "held_out"


@dataclass(frozen=True)
class FamilyInfo:
    id: str
    name: str
    mechanism: str
    protection_target: str
    research_basis: str


FAMILY_INFO: dict[str, FamilyInfo] = {
    AttackFamily.DIFFUSION_EDITING.value: FamilyInfo(
        id=AttackFamily.DIFFUSION_EDITING.value,
        name="Diffusion image editing",
        mechanism="latent denoising conditioned on the input image",
        protection_target="diffusion/latent perturbation (anti-diffusion surrogate)",
        research_basis="PhotoGuard end-to-end diffusion attack",
    ),
    AttackFamily.INPAINTING.value: FamilyInfo(
        id=AttackFamily.INPAINTING.value,
        name="Masked inpainting",
        mechanism="masked denoising / partial-region replacement",
        protection_target="mask-robust protection (evaluated over real mask kinds)",
        research_basis="DiffusionGuard mask augmentation",
    ),
    AttackFamily.INSTRUCTION_EDITING.value: FamilyInfo(
        id=AttackFamily.INSTRUCTION_EDITING.value,
        name="Instruction-guided editing",
        mechanism="image + text instruction -> edited image",
        protection_target="instruction-edit disruption (held-out editor)",
        research_basis="EditShield instruction-edit perturbations",
    ),
    AttackFamily.IMAGE_TO_IMAGE.value: FamilyInfo(
        id=AttackFamily.IMAGE_TO_IMAGE.value,
        name="Image-to-image generation",
        mechanism="encode/noise/denoise from a source image + prompt",
        protection_target="image-to-image protection across denoising strengths",
        research_basis="PhotoGuard img2img attack family",
    ),
    AttackFamily.IDENTITY_REFERENCE.value: FamilyInfo(
        id=AttackFamily.IDENTITY_REFERENCE.value,
        name="Identity/reference-conditioned generation",
        mechanism="reference photo -> identity encoder -> new synthetic image",
        protection_target="reference/identity disruption (embedding-distance objective)",
        research_basis="identity-encoder surrogate for reference-conditioned pipelines",
    ),
    AttackFamily.FACE_SWAP.value: FamilyInfo(
        id=AttackFamily.FACE_SWAP.value,
        name="Face-swap / identity transfer",
        mechanism="source face -> target image/video identity transfer",
        protection_target="identity-source disruption (same embedding objective as E)",
        research_basis="face-embedding surrogate for swap pipelines",
    ),
    AttackFamily.IMAGE_TO_VIDEO.value: FamilyInfo(
        id=AttackFamily.IMAGE_TO_VIDEO.value,
        name="Image-to-video / face-video generation",
        mechanism="reference image + motion/target video -> synthetic video",
        protection_target="reference/video disruption (adapter registered; model too heavy for dev GPU)",
        research_basis="spec §17 — optional; documented unavailable rather than faked",
    ),
    AttackFamily.VLM_CONDITIONING.value: FamilyInfo(
        id=AttackFamily.VLM_CONDITIONING.value,
        name="Vision-language / multimodal conditioning",
        mechanism="image encoder + language-conditioned representations before generation",
        protection_target="VLM representation evaluation (CLIP ViT-L/14)",
        research_basis="CLIP-style encoders as conditioning backbones",
    ),
    AttackFamily.VISION_ENCODER.value: FamilyInfo(
        id=AttackFamily.VISION_ENCODER.value,
        name="General vision/image encoders",
        mechanism="learned visual representation used as conditioning or feature extractor",
        protection_target="feature disruption (optimization + held-out encoder)",
        research_basis="transferability via common representation disruption",
    ),
}


@dataclass(frozen=True)
class AttackModel:
    """One registered model and its attack-family / role classification."""

    id: str
    name: str
    families: tuple[str, ...]
    role: ModelRole
    architecture: str
    adapter: str
    license_note: str
    vram_estimate_mb: int
    local: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# The full model table. Every model the system can touch is classified here,
# including identity/vision models from the face registry and the editing
# models from the editing manager. This is what the pipeline reports as
# "attack families covered".
# ---------------------------------------------------------------------------
ATTACK_MODELS: dict[str, AttackModel] = {
    # --- diffusion editing / img2img / inpainting -------------------------
    "stable-diffusion-v1-5/stable-diffusion-v1-5": AttackModel(
        id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        name="Stable Diffusion 1.5 U-Net + VAE + text encoder",
        families=(
            AttackFamily.DIFFUSION_EDITING.value,
            AttackFamily.IMAGE_TO_IMAGE.value,
            AttackFamily.INPAINTING.value,
        ),
        role=ModelRole.OPTIMIZATION,
        architecture="latent diffusion (denoising U-Net)",
        adapter="editing.surrogate.AntiDiffusionSurrogate (differentiable)",
        license_note="CreativeML Open RAIL-M",
        vram_estimate_mb=4100,
        note="Differentiable anti-diffusion surrogate: the optimizer maximizes its denoising "
        "error, so downstream editors conditioned on the protected photo reconstruct the "
        "subject poorly (PhotoGuard-style end-to-end attack).",
    ),
    "stable-diffusion-v1-5/stable-diffusion-inpainting": AttackModel(
        id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        name="Stable Diffusion 1.5 Inpainting (9-channel U-Net)",
        families=(AttackFamily.INPAINTING.value,),
        role=ModelRole.EVALUATION,
        architecture="masked latent diffusion",
        adapter="editing.adapter.InpaintingTarget",
        license_note="CreativeML Open RAIL-M",
        vram_estimate_mb=2600,
        note="Real masked-inpainting evaluation over face/person-derived mask kinds "
        "(shirt, hair, background, person, irregular).",
    ),
    "timbrooks/instruct-pix2pix": AttackModel(
        id="timbrooks/instruct-pix2pix",
        name="InstructPix2Pix",
        families=(AttackFamily.INSTRUCTION_EDITING.value,),
        role=ModelRole.HELD_OUT,
        architecture="instruction-conditioned latent diffusion",
        adapter="editing.adapter.InstructPix2PixTarget",
        license_note="timbrooks/instruct-pix2pix (Apache-2.0)",
        vram_estimate_mb=2300,
        note="HELD OUT: never exposed to the optimizer. The strongest transferability "
        "evidence in the direct-editing family.",
    ),
    "stable-diffusion-v1-5/stable-diffusion-v1-5#img2img": AttackModel(
        id="stable-diffusion-v1-5/stable-diffusion-v1-5#img2img",
        name="SD1.5 image-to-image (evaluation)",
        families=(AttackFamily.IMAGE_TO_IMAGE.value,),
        role=ModelRole.EVALUATION,
        architecture="img2img latent diffusion",
        adapter="editing.adapter.Image2ImageTarget",
        license_note="CreativeML Open RAIL-M",
        vram_estimate_mb=2600,
        note="Style/attribute img2img evaluation across denoising strengths.",
    ),
    # --- identity-reference / face-swap (family E & F) --------------------
    "facenet_vggface2": AttackModel(
        id="facenet_vggface2",
        name="FaceNet (InceptionResnetV1, VGGFace2)",
        families=(AttackFamily.IDENTITY_REFERENCE.value, AttackFamily.FACE_SWAP.value),
        role=ModelRole.OPTIMIZATION,
        architecture="InceptionResnetV1 face embedding",
        adapter="models.face_models (differentiable)",
        license_note="facenet-pytorch pretrained weights (research use)",
        vram_estimate_mb=220,
        note="Differentiable identity encoder. The optimizer pushes the protected face's "
        "embedding away from the original, reducing its usefulness as a reference identity "
        "for reference-conditioned generation and face-swap pipelines.",
    ),
    "facenet_casia": AttackModel(
        id="facenet_casia",
        name="FaceNet (InceptionResnetV1, CASIA-WebFace)",
        families=(AttackFamily.IDENTITY_REFERENCE.value, AttackFamily.FACE_SWAP.value),
        role=ModelRole.EVALUATION,
        architecture="InceptionResnetV1 face embedding",
        adapter="models.face_models",
        license_note="facenet-pytorch pretrained weights (research use)",
        vram_estimate_mb=220,
        note="Second identity encoder for within-family transfer evaluation.",
    ),
    "arcface_mbf": AttackModel(
        id="arcface_mbf",
        name="ArcFace (MobileFaceNet, insightface)",
        families=(AttackFamily.IDENTITY_REFERENCE.value, AttackFamily.FACE_SWAP.value),
        role=ModelRole.HELD_OUT,
        architecture="MobileFaceNet + ArcFace head (ONNX)",
        adapter="models.face_models (onnxruntime)",
        license_note="insightface buffalo_s weights (research use)",
        vram_estimate_mb=90,
        note="HELD OUT: never used during optimization. Measures how well identity disruption "
        "transfers to a different identity encoder family.",
    ),
    # --- vision encoders (family I) ---------------------------------------
    "mobilenet_v3_large": AttackModel(
        id="mobilenet_v3_large",
        name="MobileNetV3-Large (ImageNet)",
        families=(AttackFamily.VISION_ENCODER.value,),
        role=ModelRole.OPTIMIZATION,
        architecture="MobileNetV3-Large global feature extractor",
        adapter="models.face_models (differentiable)",
        license_note="torchvision weights (BSD-3)",
        vram_estimate_mb=60,
        note="Differentiable global vision encoder: disrupts common visual representations "
        "image-wide, improving cross-pipeline transfer.",
    ),
    "resnet50": AttackModel(
        id="resnet50",
        name="ResNet50 (ImageNet) — held-out",
        families=(AttackFamily.VISION_ENCODER.value,),
        role=ModelRole.HELD_OUT,
        architecture="ResNet50 global feature extractor",
        adapter="models.face_models",
        license_note="torchvision weights (BSD-3)",
        vram_estimate_mb=210,
        note="HELD OUT vision encoder: transferability test for feature disruption.",
    ),
    # --- VLM conditioning (family H) --------------------------------------
    "openai/clip-vit-large-patch14": AttackModel(
        id="openai/clip-vit-large-patch14",
        name="CLIP ViT-L/14",
        families=(AttackFamily.VLM_CONDITIONING.value, AttackFamily.VISION_ENCODER.value),
        role=ModelRole.EVALUATION,
        architecture="contrastive vision-language encoder",
        adapter="editing.manager.get_clip",
        license_note="MIT (OpenAI)",
        vram_estimate_mb=1800,
        note="Evaluation only: auxiliary semantic scoring + vision-language representation "
        "distance. Not used to generate the perturbation.",
    ),
    # --- image-to-video (family G) — adapter only -------------------------
    "image_to_video_adapter": AttackModel(
        id="image_to_video_adapter",
        name="Image-to-video (modular adapter)",
        families=(AttackFamily.IMAGE_TO_VIDEO.value,),
        role=ModelRole.HELD_OUT,
        architecture="modular reference+video adapter (no model installed)",
        adapter="editing.video.VideoReferenceAdapter (stub)",
        license_note="n/a — no weights",
        vram_estimate_mb=0,
        local=False,
        note="Family registered with a modular adapter; local video-generation weights are "
        "far beyond the development GPU, so this family is reported as NOT TESTED rather "
        "than faked. Identity disruption (family E/F) is the surrogate evidence.",
    ),
}


# ---------------------------------------------------------------------------
# Profile loading (production vs research).
# ---------------------------------------------------------------------------
DEFAULT_PRODUCTION_YAML = """
profile: production
description: >
  Minimum models needed to GENERATE the one protection perturbation for a
  user upload. No benchmark models are loaded during normal processing.
target_families:
  - diffusion_editing
  - instruction_editing
  - inpainting
  - image_to_image
  - identity_reference
  - face_swap
  - vision_encoder
active_models:
  - stable-diffusion-v1-5/stable-diffusion-v1-5
  - facenet_vggface2
  - mobilenet_v3_large
flags:
  editing_benchmark: false
  editing_robustness: true
  red_team_rounds: 0
"""

DEFAULT_RESEARCH_YAML = """
profile: research
description: >
  All accessible evaluation/benchmark models. Loaded only by the research
  benchmark entry points (scripts/benchmark_*.py), never during normal user
  processing.
target_families:
  - diffusion_editing
  - inpainting
  - instruction_editing
  - image_to_image
  - identity_reference
  - face_swap
  - image_to_video
  - vlm_conditioning
  - vision_encoder
active_models:
  - stable-diffusion-v1-5/stable-diffusion-v1-5
  - stable-diffusion-v1-5/stable-diffusion-inpainting
  - timbrooks/instruct-pix2pix
  - stable-diffusion-v1-5/stable-diffusion-v1-5#img2img
  - facenet_vggface2
  - facenet_casia
  - arcface_mbf
  - mobilenet_v3_large
  - resnet50
  - openai/clip-vit-large-patch14
  - image_to_video_adapter
flags:
  editing_benchmark: true
  editing_robustness: true
  red_team_rounds: 2
"""


def _ensure_configs() -> None:
    """Write the default profile YAMLs if the configs dir is missing them."""
    if not CONFIGS_DIR.exists():
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("production.yaml", DEFAULT_PRODUCTION_YAML),
        ("research.yaml", DEFAULT_RESEARCH_YAML),
    ):
        path = CONFIGS_DIR / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            logger.info("Wrote default %s", path)


@dataclass
class Profile:
    name: str
    description: str
    target_families: list[str] = field(default_factory=list)
    active_models: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)


def load_profile(profile_name: str | None = None) -> Profile:
    """Load a profile from ``backend/configs/{name}.yaml`` (or defaults)."""
    _ensure_configs()
    name = (profile_name or "production").lower()
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = CONFIGS_DIR / name
    if not path.exists():
        logger.warning("Profile config %s not found; using production defaults.", path)
        raw = yaml.safe_load(DEFAULT_PRODUCTION_YAML)
    else:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Profile(
        name=str(raw.get("profile", "production")),
        description=str(raw.get("description", "")),
        target_families=list(raw.get("target_families", [])),
        active_models=list(raw.get("active_models", [])),
        flags=dict(raw.get("flags", {})),
    )


def families_report(profile_name: str | None = None) -> dict:
    """Human/API-facing summary of families + the models covering each one."""
    profile = load_profile(profile_name)
    families: list[dict] = []
    for fam_id in profile.target_families:
        info = FAMILY_INFO.get(fam_id)
        if info is None:
            continue
        models = [
            m
            for m in ATTACK_MODELS.values()
            if fam_id in m.families
        ]
        families.append(
            {
                "id": fam_id,
                "name": info.name,
                "mechanism": info.mechanism,
                "protection_target": info.protection_target,
                "research_basis": info.research_basis,
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "role": m.role.value,
                        "local": m.local,
                        "note": m.note,
                    }
                    for m in models
                ],
            }
        )
    return {
        "profile": profile.name,
        "description": profile.description,
        "families": families,
        "flags": profile.flags,
    }

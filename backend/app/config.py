"""Central configuration for AI Privacy Shield backend.

All tunable values live here. Environment variables use the ``AIPS_`` prefix
and are loaded from ``backend/.env`` if present (see ``.env.example``).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project layout
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = Path(os.getenv("AIPS_MODELS_DIR", PROJECT_ROOT / "models"))
TMP_DIR = Path(os.getenv("AIPS_TMP_DIR", BACKEND_DIR / ".tmp"))

load_dotenv(BACKEND_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    """Application settings (module-level singleton ``settings``)."""

    # ---- upload / image validation -------------------------------------
    MAX_UPLOAD_BYTES: int = _env_int("AIPS_MAX_UPLOAD_MB", 25) * 1024 * 1024
    MAX_IMAGE_DIMENSION: int = _env_int("AIPS_MAX_IMAGE_DIMENSION", 6000)
    ALLOWED_MIME_TYPES: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/gif",
        "image/tiff",
    )
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")

    # ---- hardware --------------------------------------------------------
    # "auto" -> cuda if available else cpu. Force with "cuda"/"cpu".
    DEVICE: str = os.getenv("AIPS_DEVICE", "auto").lower()



    # ---- face detection --------------------------------------------------
    FACE_DETECTOR_DIR: Path = MODELS_DIR / "face_detector"
    FACE_DETECTOR_PROTO_URL: str = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    )
    FACE_DETECTOR_CAFFE_URL: str = (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    FACE_CONFIDENCE: float = _env_float("AIPS_FACE_CONFIDENCE", 0.6)
    FACE_CROP_MARGIN: float = _env_float("AIPS_FACE_CROP_MARGIN", 2.2)  # box expansion factor
    FACE_REGION_DILATE: float = _env_float("AIPS_FACE_REGION_DILATE", 1.35)  # perturbation mask expansion

    # ---- person detection -------------------------------------------------
    # OpenCV's built-in HOG + linear SVM person detector (no model download).
    # Confidence is the raw SVM decision weight.
    PERSON_CONFIDENCE: float = _env_float("AIPS_PERSON_CONFIDENCE", 0.2)
    PERSON_REGION_DILATE: float = _env_float("AIPS_PERSON_REGION_DILATE", 1.15)  # mask expansion
    PERSON_REGION_WEIGHT: float = _env_float("AIPS_PERSON_REGION_WEIGHT", 0.6)  # relative perturbation strength
    CONTEXT_MASK_WEIGHT: float = _env_float("AIPS_CONTEXT_MASK_WEIGHT", 0.18)  # low-amplitude image-wide dither
    PERSON_DETECT_MAX_DIM: int = _env_int("AIPS_PERSON_DETECT_MAX_DIM", 720)  # downscale for HOG speed

    # ---- adversarial optimization ---------------------------------------
    PERTURBATION_EPSILON: float = _env_float("AIPS_EPSILON", 0.035)  # L-inf bound on perturbation (0..1 scale)
    # The face-detection/identity stage is SECONDARY to the multi-family
    # editing protection (the primary objective). It gets this fraction of the
    # total epsilon budget so the stacked perturbation stays visually tiny;
    # the rest of the budget is consumed by the anti-diffusion / identity /
    # vision editing terms (spec §13, Priority 2).
    FACE_PROTECT_EPSILON_FRACTION: float = _env_float("AIPS_FACE_PROTECT_EPSILON_FRACTION", 0.4)
    OPT_ITERATIONS_GPU: int = _env_int("AIPS_ITERATIONS_GPU", 70)
    OPT_ITERATIONS_CPU: int = _env_int("AIPS_ITERATIONS_CPU", 35)
    OPT_LR: float = _env_float("AIPS_OPT_LR", 0.03)
    OPT_MARGIN: float = _env_float("AIPS_OPT_MARGIN", 0.65)  # target L2 embedding distance (face identity)
    OPT_ROBUSTNESS_INTERVAL: int = _env_int("AIPS_ROBUSTNESS_INTERVAL", 1)  # eval transforms every N iters
    OPT_EARLY_STOP_PATIENCE: int = _env_int("AIPS_OPT_EARLY_STOP_PATIENCE", 10)  # stop after N iters w/o loss gain
    OPT_SEED: int = _env_int("AIPS_OPT_SEED", 0)

    # loss weights — multi-objective protection (centralized here)
    # Detection is the primary objective; identity is a supporting layer.
    W_IDENTITY: float = _env_float("AIPS_W_IDENTITY", 0.6)  # face-identity disruption (supporting)
    W_VISION: float = _env_float("AIPS_W_VISION", 0.7)  # global vision-feature disruption
    VISION_MARGIN: float = _env_float("AIPS_VISION_MARGIN", 0.55)  # target L2 distance for global features
    W_SSIM: float = _env_float("AIPS_W_SSIM", 0.10)
    W_MSE: float = _env_float("AIPS_W_MSE", 0.05)
    W_ROBUSTNESS: float = _env_float("AIPS_W_ROBUSTNESS", 0.60)
    W_PERTURBATION: float = _env_float("AIPS_W_PERTURBATION", 0.05)

    # ---- visual quality floors (candidates below these are rejected) -----
    MIN_SSIM: float = _env_float("AIPS_MIN_SSIM", 0.90)
    MIN_PSNR: float = _env_float("AIPS_MIN_PSNR", 30.0)

    # ---- detection-suppression objectives ---------------------------------
    W_FACE_DET: float = _env_float("AIPS_W_FACE_DET", 0.9)  # differentiable surrogate weight
    W_PERSON_DET: float = _env_float("AIPS_W_PERSON_DET", 0.30)
    DET_BLOB_DIRECTIONS_GPU: int = _env_int("AIPS_DET_BLOB_DIRECTIONS_GPU", 10)
    DET_BLOB_DIRECTIONS_CPU: int = _env_int("AIPS_DET_BLOB_DIRECTIONS_CPU", 4)
    DET_BLOB_COUNT: int = _env_int("AIPS_DET_BLOB_COUNT", 8)  # Gaussian blobs per direction
    DET_BLOB_SIGMA_MIN: int = _env_int("AIPS_DET_BLOB_SIGMA_MIN", 8)
    DET_BLOB_SIGMA_MAX: int = _env_int("AIPS_DET_BLOB_SIGMA_MAX", 28)
    DET_ATTACK_FRACTION: float = _env_float("AIPS_DET_ATTACK_FRACTION", 0.25)  # share of iters on pure detection
    # (the pure-detection attack buys a few % of extra face suppression at the
    # cost of halving embedding distances, so its budget is deliberately modest)

    # ---- black-box refinement (phase 2) ----------------------------------
    # Two normalized gradient estimates are combined: embedding disruption
    # (weight W_EMB) and detection suppression (weight W_DET). Detector
    # gradient is estimated on a downscaled image for speed.
    W_EMB: float = _env_float("AIPS_W_EMB", 0.8)
    W_DET: float = _env_float("AIPS_W_DET", 0.2)
    DET_TARGET_FACE: float = _env_float("AIPS_DET_TARGET_FACE", 0.35)  # push face score below this
    DET_TARGET_PERSON: float = _env_float("AIPS_DET_TARGET_PERSON", 0.0)  # push HOG weight below this
    DET_GRAD_INTERVAL_GPU: int = _env_int("AIPS_DET_GRAD_INTERVAL_GPU", 6)  # recompute detector gradient every N iters
    DET_GRAD_INTERVAL_CPU: int = _env_int("AIPS_DET_GRAD_INTERVAL_CPU", 8)
    DET_DIRECTIONS_GPU: int = _env_int("AIPS_DET_DIRECTIONS_GPU", 10)
    DET_DIRECTIONS_CPU: int = _env_int("AIPS_DET_DIRECTIONS_CPU", 5)
    # Per-detector weights inside the black-box detection loss. The neural
    # person detector and MTCNN cascade were previously omitted from the loss
    # (only OpenCV SSD + HOG were attacked), which left them nearly untouched.
    W_NEURAL_PERSON_DET: float = _env_float("AIPS_W_NEURAL_PERSON_DET", 0.40)
    W_MTCNN_DET: float = _env_float("AIPS_W_MTCNN_DET", 0.40)
    # Direction mix for the black-box detection attack: fractions of per-pixel
    # texture (moves MTCNN/CNNs), Gaussian blobs (survive downsampling), and
    # mid-frequency sinusoids (transferable) -- must sum to 1.0.
    DET_TEXTURE_FRACTION: float = _env_float("AIPS_DET_TEXTURE_FRACTION", 0.40)
    DET_BLOB_FRACTION: float = _env_float("AIPS_DET_BLOB_FRACTION", 0.40)
    # Differentiable surrogate targets. The O-Net face logit is what the
    # cascade reports as confidence; logit -1.2 ~ p=0.23 (fails O-Net's 0.7
    # threshold -> detection miss), so the surrogate pushes toward it while
    # the black-box phase verifies the real cascade result.
    ONET_LOGIT_TARGET: float = _env_float("AIPS_ONET_LOGIT_TARGET", -1.2)
    PNET_LOGIT_TARGET: float = _env_float("AIPS_PNET_LOGIT_TARGET", -2.0)
    RNET_LOGIT_TARGET: float = _env_float("AIPS_RNET_LOGIT_TARGET", -1.0)  # p=0.27, below the 0.7 R-Net threshold
    REFINE_DIRECTIONS_GPU: int = _env_int("AIPS_REFINE_DIRECTIONS_GPU", 12)
    REFINE_DIRECTIONS_CPU: int = _env_int("AIPS_REFINE_DIRECTIONS_CPU", 6)
    REFINE_MAX_ITERS_GPU: int = _env_int("AIPS_REFINE_MAX_ITERS_GPU", 60)
    REFINE_MAX_ITERS_CPU: int = _env_int("AIPS_REFINE_MAX_ITERS_CPU", 20)
    REFINE_TRANSFORM_INTERVAL: int = _env_int("AIPS_REFINE_TRANSFORM_INTERVAL", 6)  # real-transform eval every N iters

    # ---- vision encoder / neural person detector -------------------------
    VISION_ENCODER_ENABLED: bool = os.getenv("AIPS_VISION_ENCODER_ENABLED", "1").lower() in ("1", "true", "yes")
    VISION_OPT_ID: str = "mobilenet_v3_large"  # differentiable optimization surrogate
    VISION_HELD_OUT_ID: str = "resnet50"  # evaluation-only (transferability test)
    NEURAL_PERSON_CONFIDENCE: float = _env_float("AIPS_NEURAL_PERSON_CONFIDENCE", 0.5)
    NEURAL_PERSON_MAX_DIM: int = _env_int("AIPS_NEURAL_PERSON_MAX_DIM", 800)

    # ---- verification / robustness testing ------------------------------
    # Embeddings are L2-normalized; distances are in [0, 2].
    DISRUPT_PASS: float = _env_float("AIPS_DISRUPT_PASS", 0.70)
    DISRUPT_PARTIAL: float = _env_float("AIPS_DISRUPT_PARTIAL", 0.40)
    ROBUSTNESS_TRANSFORMS: tuple[str, ...] = (
        "jpeg_compression",
        "resize",
        "crop",
        "brightness",
        "contrast",
        "reencode",
    )

    # ---- AI-editing protection (primary objective) -----------------------
    # The optimizer maximizes the denoising reconstruction error of a local
    # Stable Diffusion surrogate (Photoguard-style anti-diffusion attack), so
    # an AI image editor conditioned on the protected photo reconstructs the
    # subject poorly. The InstructPix2Pix pipeline is the held-out *real*
    # editing benchmark: original vs protected through the same editor, same
    # prompt, same seed/steps/guidance — only the input image changes.
    EDITING_ENABLED: bool = os.getenv("AIPS_EDITING_ENABLED", "1").lower() in ("1", "true", "yes")
    EDITING_SURROGATE_ENABLED: bool = os.getenv("AIPS_EDITING_SURROGATE_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    # In the production profile the in-app benchmark is off by default (the
    # 3-editor benchmark is research tooling; see scripts/benchmark_*.py).
    # Set AIPS_EDITING_BENCHMARK_ENABLED explicitly to override.
    _BENCH_DEFAULT = "1" if os.getenv("AIPS_PROFILE", "production").lower() == "research" else "0"
    EDITING_BENCHMARK_ENABLED: bool = os.getenv("AIPS_EDITING_BENCHMARK_ENABLED", _BENCH_DEFAULT).lower() in (
        "1", "true", "yes"
    )
    EDITING_IP2P_REPO: str = os.getenv("AIPS_EDITING_IP2P_REPO", "timbrooks/instruct-pix2pix")
    EDITING_SD15_REPO: str = os.getenv(
        "AIPS_EDITING_SD15_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    EDITING_CLIP_REPO: str = os.getenv("AIPS_EDITING_CLIP_REPO", "openai/clip-vit-large-patch14")
    # benchmark inference settings (identical for original and protected)
    EDITING_RESOLUTION: int = _env_int("AIPS_EDITING_RESOLUTION", 512)
    EDITING_STEPS: int = _env_int("AIPS_EDITING_STEPS", 15)
    EDITING_GUIDANCE: float = _env_float("AIPS_EDITING_GUIDANCE", 7.5)
    EDITING_IMAGE_GUIDANCE: float = _env_float("AIPS_EDITING_IMAGE_GUIDANCE", 1.5)
    EDITING_SEED: int = _env_int("AIPS_EDITING_SEED", 42)  # fixed across original/protected
    # additional evaluation seeds for the benchmark script (comma-separated)
    EDITING_SEEDS: tuple[int, ...] = tuple(
        int(s) for s in os.getenv("AIPS_EDITING_SEEDS", "42,7,1337").split(",") if s.strip()
    )
    # editor selection for the in-app benchmark (all are evaluation-only)
    EDITING_IP2P_ENABLED: bool = os.getenv("AIPS_EDITING_IP2P_ENABLED", "1").lower() in ("1", "true", "yes")
    EDITING_INPAINTING_ENABLED: bool = os.getenv("AIPS_EDITING_INPAINTING_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    EDITING_IMG2IMG_ENABLED: bool = os.getenv("AIPS_EDITING_IMG2IMG_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    EDITING_IMG2IMG_TASKS: tuple[str, ...] = tuple(
        os.getenv("AIPS_EDITING_IMG2IMG_TASKS", "t02_background,t04_lighting,t05_sketch").split(",")
    )
    EDITING_IMG2IMG_STRENGTH: float = _env_float("AIPS_EDITING_IMG2IMG_STRENGTH", 0.6)
    # edit-success scoring: task-specific pixel metric (primary) + CLIP (auxiliary)
    W_EDITING_TASK_METRIC: float = _env_float("AIPS_W_EDITING_TASK_METRIC", 0.6)
    W_EDITING_CLIP: float = _env_float("AIPS_W_EDITING_CLIP", 0.4)
    # CLIP cosine deltas are typically 0.0-0.1; this maps a delta to the 0..1 scale
    EDITING_CLIP_DELTA_SCALE: float = _env_float("AIPS_EDITING_CLIP_DELTA_SCALE", 0.1)
    # editing-success-under-transform robustness (edit success measured after a
    # real JPEG/resize/crop/... on the protected image)
    EDITING_ROBUSTNESS_ENABLED: bool = os.getenv("AIPS_EDITING_ROBUSTNESS_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    EDITING_ROBUSTNESS_TRANSFORMS: tuple[str, ...] = tuple(
        os.getenv("AIPS_EDITING_ROBUSTNESS_TRANSFORMS", "jpeg_compression,resize").split(",")
    )
    EDITING_ROBUSTNESS_TASKS: tuple[str, ...] = tuple(
        os.getenv("AIPS_EDITING_ROBUSTNESS_TASKS", "t01_shirt_color,t02_background").split(",")
    )
    # mask kinds used by the masked-inpainting attack (comma-separated)
    EDITING_MASK_KINDS: tuple[str, ...] = tuple(
        os.getenv("AIPS_EDITING_MASK_KINDS", "shirt,hair,background,person,irregular").split(",")
    )
    # benchmark task subset (comma-separated ids; see editing/tasks.py for all)
    EDITING_TASKS: tuple[str, ...] = tuple(
        os.getenv("AIPS_EDITING_TASKS", "t01_shirt_color,t02_background,t03_hat,t05_sketch").split(",")
    )
    # anti-diffusion surrogate settings
    EDITING_SURROGATE_RESOLUTION: int = _env_int("AIPS_EDITING_SURROGATE_RESOLUTION", 256)
    EDITING_SURROGATE_TIMESTEP: int = _env_int("AIPS_EDITING_SURROGATE_TIMESTEP", 250)  # of 1000
    EDITING_SURROGATE_ITERS_GPU: int = _env_int("AIPS_EDITING_SURROGATE_ITERS_GPU", 8)
    EDITING_SURROGATE_ITERS_CPU: int = _env_int("AIPS_EDITING_SURROGATE_ITERS_CPU", 3)
    EDITING_SURROGATE_LR: float = _env_float("AIPS_EDITING_SURROGATE_LR", 0.05)
    # fraction of PERTURBATION_EPSILON budgeted to the anti-diffusion stage
    EDITING_SURROGATE_EPSILON_FRACTION: float = _env_float("AIPS_EDITING_SURROGATE_EPSILON_FRACTION", 0.5)
    EDITING_PROMPT: str = os.getenv("AIPS_EDITING_PROMPT", "a photo of a person")
    # hard quality floor: if the editing stage drops SSIM below this, the stage
    # is reverted (honest constraint — never visibly damage the image).
    EDITING_MIN_SSIM: float = _env_float("AIPS_EDITING_MIN_SSIM", 0.90)

    # ---- multi-family protection (spec: ONE perturbation) -----------------
    # The anti-diffusion PGD also optimizes identity-reference and vision-
    # encoder terms in the SAME perturbation, so a single protected image
    # targets families A-D (diffusion surrogate), E/F (identity reference /
    # face-swap via FaceNet) and I (vision encoders via MobileNetV3) at once.
    # Video (G) and VLM (H) are registered families with evaluation-only or
    # documented-unavailable adapters.
    EDITING_IDENTITY_ENABLED: bool = os.getenv("AIPS_EDITING_IDENTITY_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    # weight of the identity-reference loss relative to the (normalized)
    # diffusion-denosing term inside the multi-family PGD
    EDITING_IDENTITY_WEIGHT: float = _env_float("AIPS_EDITING_IDENTITY_WEIGHT", 1.0)
    # share of the PGD iterations run as PURE identity steps (no diffusion
    # gradient): with sign-SGD the image-wide diffusion gradient otherwise
    # dominates every pixel, so the face embedding barely moves. Dedicated
    # identity iterations guarantee the identity direction is applied to the
    # face region (the pixel-space analogue of ID-Eraser's identity-space
    # perturbation steps).
    EDITING_IDENTITY_ATTACK_FRACTION: float = _env_float("AIPS_EDITING_IDENTITY_ATTACK_FRACTION", 0.5)
    # ---- identity / face-swap protection upgrade (ID-Eraser/Phantom-style) ----
    # 1) A SECOND differentiable identity encoder (FaceNet CASIA) joins the PGD
    #    so the embedding objective does not overfit a single encoder
    #    (cross-encoder transfer; the old single-encoder objective barely
    #    moved on other encoders).
    EDITING_IDENTITY_CASIA_ENABLED: bool = os.getenv("AIPS_EDITING_IDENTITY_CASIA_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    # 2) Identity loss is also evaluated on JPEG-approx / brightness variants of
    #    the face crop (ID-Eraser interference-layer concept): the embedding
    #    disruption must survive re-encoding, not just the clean crop.
    EDITING_IDENTITY_TRANSFORMS: bool = os.getenv("AIPS_EDITING_IDENTITY_TRANSFORMS", "1").lower() in (
        "1", "true", "yes"
    )
    # 3) Soft elliptical emphasis over the face region (Phantom spatial
    #    constraint): perturbation is concentrated on identity-relevant facial
    #    regions instead of being spread uniformly.
    EDITING_IDENTITY_REGION_EMPHASIS: float = _env_float("AIPS_EDITING_IDENTITY_REGION_EMPHASIS", 0.3)
    # 4) In-place zeroth-order ArcFace (w600k_mbf) refinement of the SAME
    #    perturbation after the PGD. ArcFace is the identity-encoder family
    #    actual swap pipelines (SimSwap IIM, INSwapper) use; it is ONNX
    #    (black-box), so the refinement estimates gradients by finite
    #    differences over random directions — the identical δ keeps being
    #    optimized, never a second stacked perturbation.
    EDITING_IDENTITY_REFINE_ENABLED: bool = os.getenv("AIPS_EDITING_IDENTITY_REFINE_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    EDITING_IDENTITY_REFINE_ITERS_GPU: int = _env_int("AIPS_EDITING_IDENTITY_REFINE_ITERS_GPU", 14)
    EDITING_IDENTITY_REFINE_ITERS_CPU: int = _env_int("AIPS_EDITING_IDENTITY_REFINE_ITERS_CPU", 6)
    EDITING_IDENTITY_REFINE_DIRECTIONS_GPU: int = _env_int("AIPS_EDITING_IDENTITY_REFINE_DIRECTIONS_GPU", 8)
    EDITING_IDENTITY_REFINE_DIRECTIONS_CPU: int = _env_int("AIPS_EDITING_IDENTITY_REFINE_DIRECTIONS_CPU", 4)
    EDITING_IDENTITY_REFINE_GOAL: float = _env_float("AIPS_EDITING_IDENTITY_REFINE_GOAL", 0.9)  # ArcFace L2 target
    EDITING_VISION_ENABLED: bool = os.getenv("AIPS_EDITING_VISION_ENABLED", "1").lower() in (
        "1", "true", "yes"
    )
    EDITING_VISION_WEIGHT: float = _env_float("AIPS_EDITING_VISION_WEIGHT", 0.35)
    # adaptive red-team loop (research profile): protect -> probe families ->
    # raise the weight of the weakest family -> re-protect. Stops when gains
    # saturate, the quality floor is hit, or the round budget is exhausted.
    RED_TEAM_ROUNDS: int = _env_int("AIPS_RED_TEAM_ROUNDS", 0)
    RED_TEAM_MIN_GAIN: float = _env_float("AIPS_RED_TEAM_MIN_GAIN", 0.005)  # min family-score gain to continue
    RED_TEAM_MAX_WEIGHT: float = _env_float("AIPS_RED_TEAM_MAX_WEIGHT", 2.5)  # per-term weight cap
    RED_TEAM_WEIGHT_STEP: float = _env_float("AIPS_RED_TEAM_WEIGHT_STEP", 1.5)

        # ---- profile (production vs research) --------------------------------
    # production: protection engine only, no benchmark models loaded, fast
    # research:   full benchmark/red-team environment (scripts/benchmark_*)
    # The profile's declarative defaults live in configs/{profile}.yaml and
    # are applied by _apply_profile_defaults() below (env vars always win).
    PROFILE: str = os.getenv("AIPS_PROFILE", "production").lower()

    # ---- C2PA provenance (optional second defense layer) ------------------
    # C2PA is NOT an AI blocker: it cryptographically binds a provenance
    # manifest (who processed the image, when, content hash) to the protected
    # file so tampering/editing is detectable. Platforms may strip it — the
    # adversarial perturbation remains the primary layer. A local self-signed
    # keypair is generated on first use so the feature works offline; real
    # deployments should point AIPS_C2PA_KEY/CERT at their own signing key.
    C2PA_ENABLED: bool = os.getenv("AIPS_C2PA_ENABLED", "1").lower() in ("1", "true", "yes")
    C2PA_KEY_PATH: str = os.getenv("AIPS_C2PA_KEY", "")  # PEM private key (optional)
    C2PA_CERT_PATH: str = os.getenv("AIPS_C2PA_CERT", "")  # PEM cert chain (optional)
    C2PA_KEY_DIR: Path = BACKEND_DIR / "data" / "c2pa"  # self-generated keypair lives here

    # ---- sensitive content analysis -------------------------------------
    OCR_MAX_DIMENSION: int = _env_int("AIPS_OCR_MAX_DIMENSION", 1280)
    OCR_ENABLED: bool = os.getenv("AIPS_OCR_ENABLED", "1").lower() in ("1", "true", "yes")
    SENSITIVE_REGION_BLUR: int = _env_int("AIPS_SENSITIVE_REGION_BLUR", 25)

    # ---- output ----------------------------------------------------------
    OUTPUT_FORMAT: str = os.getenv("AIPS_OUTPUT_FORMAT", "png").lower()  # png or jpeg (jpeg is lossy)
    JPEG_QUALITY: int = _env_int("AIPS_JPEG_QUALITY", 95)

    # ---- server ----------------------------------------------------------
    HOST: str = os.getenv("AIPS_HOST", "127.0.0.1")
    PORT: int = _env_int("AIPS_PORT", 8000)
    CORS_ORIGINS: tuple[str, ...] = tuple(
        os.getenv("AIPS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    )
    DEBUG: bool = os.getenv("AIPS_DEBUG", "0").lower() in ("1", "true", "yes")

    # ---- session / temp lifecycle ---------------------------------------
    SESSION_TTL_SECONDS: int = _env_int("AIPS_SESSION_TTL_SECONDS", 1800)
    JANITOR_INTERVAL_SECONDS: int = _env_int("AIPS_JANITOR_INTERVAL_SECONDS", 300)


def _apply_profile_defaults() -> None:
    """Apply the active profile's ``configs/{profile}.yaml`` flags as env
    defaults BEFORE ``Settings`` is built (explicit env vars always win).

    This makes the declarative production/research split (spec §21) the
    source of truth for profile behaviour while keeping every knob
    overridable per deployment.
    """
    profile = os.getenv("AIPS_PROFILE", "production").lower()
    path = BACKEND_DIR / "configs" / f"{profile}.yaml"
    if not path.exists():
        return
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - never fail startup over a config file
        return
    flags = data.get("flags") or {}
    env_map = {
        "editing_benchmark": "AIPS_EDITING_BENCHMARK_ENABLED",
        "editing_robustness": "AIPS_EDITING_ROBUSTNESS_ENABLED",
        "red_team_rounds": "AIPS_RED_TEAM_ROUNDS",
        "editing_surrogate": "AIPS_EDITING_SURROGATE_ENABLED",
    }
    for key, env in env_map.items():
        if key in flags and os.getenv(env) is None:
            os.environ[env] = str(flags[key]).lower()


_apply_profile_defaults()

settings = Settings()


def ensure_dirs() -> None:
    """Create directories the application needs at runtime."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    (TMP_DIR / "sessions").mkdir(parents=True, exist_ok=True)


def resolve_device(force: str | None = None) -> str:
    """Resolve the torch compute device: 'cuda' when available, else 'cpu'."""
    requested = (force or settings.DEVICE).lower()
    if requested in ("cuda", "gpu"):
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:  # pragma: no cover - torch always present here
            return "cpu"
    if requested in ("cpu", "mps"):
        return "cpu"
    # auto
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"

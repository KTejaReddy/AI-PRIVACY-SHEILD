"""Surrogate / verification models.

Face-embedding models:
  Optimization surrogates (differentiable):
    * InceptionResnetV1 trained on VGGFace2   (``facenet-pytorch``)
    * InceptionResnetV1 trained on CASIA-WebFace (``facenet-pytorch``)
  Verification (non-differentiable, evaluation + black-box refinement):
    * MobileFaceNet ArcFace (ONNX) from insightface ``buffalo_s``

Global vision-encoder models (mode="global", embed the whole image):
  Optimization surrogate (differentiable):
    * MobileNetV3-Large (ImageNet) — broad vision-feature disruption
  Held-out evaluation model (never optimized against):
    * ResNet50 (ImageNet) — transferability test

All weights are public research weights. Facenet weights auto-download on
first use from the ``facenet-pytorch`` GitHub release; torchvision weights
auto-download from the PyTorch model zoo; ArcFace is downloaded by
``scripts/download_models.py``. ``mode`` distinguishes per-face crops
("face") from whole-image embeddings ("global").
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import settings

logger = logging.getLogger(__name__)

FACENET_INPUT_SIZE = 160
ARCFACE_INPUT_SIZE = 112

# ---- preprocessing -------------------------------------------------------


def facenet_preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    """160x160 RGB crop -> float32 [1,3,160,160] in FaceNet input space."""
    crop = crop_rgb.astype(np.float32)
    crop = (crop - 127.5) / 128.0  # same normalization the package uses
    crop = np.transpose(crop, (2, 0, 1))[None, ...]
    return crop


def arcface_preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    """112x112 RGB crop -> float32 [1,3,112,112] in ArcFace input space (BGR, mean-sub)."""
    bgr = crop_rgb[..., ::-1].astype(np.float32)
    bgr = (bgr - 127.5) / 128.0
    bgr = np.transpose(bgr, (2, 0, 1))[None, ...]
    return bgr


# ImageNet normalization used by the global vision encoders (torchvision).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    """RGB image (any size) -> float32 [1,3,224,224] in ImageNet input space.

    The crop-resize is done by the caller; this normalizes and adds the batch
    dim only, matching torchvision's ``Normalize`` transform.
    """
    x = crop_rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def facenet_preprocess_torch(x) -> torch.Tensor:
    """[1,3,H,W] tensor in [0,1] -> FaceNet input space (matches numpy path)."""
    return (x * 255.0 - 127.5) / 128.0


def imagenet_preprocess_torch(x) -> torch.Tensor:
    """[1,3,H,W] tensor in [0,1] -> ImageNet input space (matches numpy path)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


# ---- model definitions ----------------------------------------------------


@dataclass
class FaceModelInfo:
    id: str
    display_name: str
    kind: str  # "optimization" | "verification"
    architecture: str
    license_note: str
    loaded: bool = False
    error: str | None = None
    device: str | None = None


@dataclass
class FaceModel:
    """Uniform wrapper so the rest of the code treats every model the same."""

    info: FaceModelInfo
    torch_model: object | None = None
    onnx_session: object | None = None
    input_size: int = FACENET_INPUT_SIZE
    preprocess: object = facenet_preprocess
    onnx: bool = False
    mode: str = "face"  # "face" -> embed face crops; "global" -> embed the whole image
    preprocess_torch: object = facenet_preprocess_torch  # differentiable path for the optimizer
    _embed: object | None = field(default=None, init=False)

    def embed_crops(self, crops: list[np.ndarray], device: str = "cpu") -> list[np.ndarray]:
        """Embed a list of square RGB crops. Returns list of L2-normalized 512-d vectors."""
        if self.onnx:
            return self._embed_onnx(crops)
        return self._embed_torch(crops, device)

    def _embed_torch(self, crops: list[np.ndarray], device: str) -> list[np.ndarray]:
        import torch  # noqa: PLC0415

        # The model lives on its own device (e.g. cuda); the caller's ``device``
        # hint is not always reliable (benchmarks pass "cpu"). Send the input
        # to wherever the weights actually are.
        params = list(self.torch_model.parameters())
        model_device = params[0].device if params else torch.device("cpu")
        batch = np.concatenate([self.preprocess(c) for c in crops], axis=0)
        tensor = torch.from_numpy(batch).to(model_device)
        with torch.no_grad():
            emb = self.torch_model(tensor).cpu().numpy()
        return [l2_normalize(e) for e in emb]

    def _embed_onnx(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        import onnxruntime as ort  # noqa: PLC0415

        results: list[np.ndarray] = []
        for crop in crops:
            inp = self.preprocess(crop)
            out = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inp})[0]
            results.append(l2_normalize(out[0]))
        return results


# ---- registry -------------------------------------------------------------


class FaceModelRegistry:
    """Lazy-loading registry of surrogate / verification face models."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._lock = threading.Lock()
        self._models: dict[str, FaceModel] = {}
        self._load_state: dict[str, str] = {}  # id -> "pending"|"loading"|"ok"|"error"

    # ------------------------------------------------------------------
    @property
    def optimization_models(self) -> list[FaceModel]:
        return [m for m in self._models.values() if m.info.kind == "optimization"]

    @property
    def verification_models(self) -> list[FaceModel]:
        """Models used for verification/testing: every loaded model."""
        return [m for m in self._models.values() if m.info.loaded]

    def status(self) -> list[dict]:
        return [m.info.__dict__ | {"state": self._load_state.get(m.info.id, "pending")} for m in self._models.values()]

    def describe(self) -> dict:
        return {
            "device": self.device,
            "models": self.status(),
        }

    # ------------------------------------------------------------------
    def _declare(self, info: FaceModelInfo) -> FaceModel:
        model = FaceModel(info=info)
        self._models[info.id] = model
        self._load_state[info.id] = "pending"
        return model

    def load_all(self) -> None:
        """Load the models declared for the active profile (called once).

        Production profile: the differentiable optimization surrogates the
        unified protection engine actually consumes (FaceNet VGGFace2 +
        CASIA identity encoders, MobileNetV3 vision term, ArcFace for the
        in-place black-box identity refinement) plus the SD1.5 surrogate on
        demand. Held-out models (ResNet50, …) load only in the research
        profile — the production app must not become a local AI laboratory.
        """
        self._declare(
            FaceModelInfo(
                id="facenet_vggface2",
                display_name="FaceNet (VGGFace2)",
                kind="optimization",
                architecture="InceptionResnetV1",
                license_note="facenet-pytorch pretrained weights (research use)",
            )
        )
        # A SECOND differentiable identity encoder: the single-encoder
        # objective overfits (the old FaceNet-only term barely moved other
        # encoders), so the unified PGD pushes VGGFace2 AND CASIA. This is a
        # core production component of the face-swap/identity-reference
        # protection (ID-Eraser/Phantom-style multi-encoder objective).
        if settings.EDITING_IDENTITY_CASIA_ENABLED:
            self._declare(
                FaceModelInfo(
                    id="facenet_casia",
                    display_name="FaceNet (CASIA-WebFace)",
                    kind="optimization",
                    architecture="InceptionResnetV1",
                    license_note="facenet-pytorch pretrained weights (research use)",
                )
            )
        # ArcFace w600k_mbf is the identity-encoder family real swap pipelines
        # use (SimSwap IIM / INSwapper). Used for the in-place black-box
        # identity refinement of the SAME perturbation.
        if settings.EDITING_IDENTITY_REFINE_ENABLED:
            self._declare(
                FaceModelInfo(
                    id="arcface_mbf",
                    display_name="ArcFace (MobileFaceNet, w600k)",
                    kind="optimization",
                    architecture="MobileFaceNet + ArcFace head (ONNX, black-box)",
                    license_note="insightface buffalo_s weights (research use)",
                )
            )
        if settings.VISION_ENCODER_ENABLED:
            self._declare(
                FaceModelInfo(
                    id=settings.VISION_OPT_ID,
                    display_name="MobileNetV3-Large (ImageNet)",
                    kind="optimization",
                    architecture="MobileNetV3-Large (global vision encoder)",
                    license_note="torchvision pretrained weights (BSD-3, research use)",
                )
            )
        if settings.PROFILE == "research":
            # held-out vision encoder (research only)
            if settings.VISION_ENCODER_ENABLED:
                self._declare(
                    FaceModelInfo(
                        id=settings.VISION_HELD_OUT_ID,
                        display_name="ResNet50 (ImageNet) — held-out",
                        kind="verification",
                        architecture="ResNet50 (global vision encoder, evaluation only)",
                        license_note="torchvision pretrained weights (BSD-3, research use)",
                    )
                )
        for model in list(self._models.values()):
            self._load_one(model)

    def _load_one(self, model: FaceModel) -> None:
        if model.info.loaded:
            return
        with self._lock:
            if self._load_state.get(model.info.id) in ("loading", "ok"):
                return
            self._load_state[model.info.id] = "loading"
        try:
            if model.info.id.startswith("facenet_"):
                self._load_facenet(model)
            elif model.info.id == "arcface_mbf":
                self._load_arcface(model)
            elif model.info.id in (settings.VISION_OPT_ID, settings.VISION_HELD_OUT_ID):
                self._load_vision(model)
            model.info.loaded = True
            model.info.device = self.device
            self._load_state[model.info.id] = "ok"
            logger.info("Loaded model %s", model.info.id)
        except Exception as exc:  # noqa: BLE001
            model.info.error = str(exc)
            self._load_state[model.info.id] = "error"
            logger.warning("Model %s failed to load: %s", model.info.id, exc)

    def _load_facenet(self, model: FaceModel) -> None:
        import torch  # noqa: PLC0415
        from facenet_pytorch import InceptionResnetV1  # noqa: PLC0415

        pretrained = "vggface2" if model.info.id == "facenet_vggface2" else "casia-webface"
        net = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        model.torch_model = net
        model.input_size = FACENET_INPUT_SIZE
        model.preprocess = facenet_preprocess

    def _load_vision(self, model: FaceModel) -> None:
        """Load a torchvision feature extractor (global mode, ImageNet weights)."""
        import torch  # noqa: PLC0415
        from torchvision import models as tv_models  # noqa: PLC0415

        if model.info.id == settings.VISION_OPT_ID:
            weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT
            net = tv_models.mobilenet_v3_large(weights=weights)
            net.classifier = torch.nn.Identity()
            out_dim = 1280
        else:
            weights = tv_models.ResNet50_Weights.DEFAULT
            net = tv_models.resnet50(weights=weights)
            net.fc = torch.nn.Identity()
            out_dim = 2048

        net = net.eval().to(model.info.device or self.device)
        # Keep the raw net: the optimizer needs gradients, and the evaluation
        # path (embed_crops) already wraps its forward in torch.no_grad().
        model.torch_model = net
        model.input_size = 224
        model.preprocess = imagenet_preprocess
        model.preprocess_torch = imagenet_preprocess_torch
        model.mode = "global"
        logger.info(
            "Vision encoder %s ready (global mode, %d-d features)", model.info.id, out_dim
        )

    def _load_arcface(self, model: FaceModel) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        from ..config import MODELS_DIR  # noqa: PLC0415

        path = MODELS_DIR / "arcface" / "w600k_mbf.onnx"
        if not path.exists():
            raise FileNotFoundError(
                "ArcFace model not found. Run: python scripts/download_models.py --arcface"
            )
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(path), sess_options, providers=["CPUExecutionProvider"])
        model.onnx_session = sess
        model.onnx = True
        model.input_size = ARCFACE_INPUT_SIZE
        model.preprocess = arcface_preprocess


_registry: FaceModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry(device: str | None = None) -> FaceModelRegistry:
    """Get the process-wide model registry, creating it with the resolved device."""
    global _registry  # noqa: PLW0603
    with _registry_lock:
        if _registry is None:
            from ..config import resolve_device  # noqa: PLC0415

            _registry = FaceModelRegistry(device=resolve_device(device))
        return _registry


def load_models() -> FaceModelRegistry:
    reg = get_registry()
    reg.load_all()
    return reg

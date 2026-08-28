"""Editing model manager.

The target GPU has ~4.3 GB VRAM. The SD1.5 surrogate needs ~4.1 GB peak and
InstructPix2Pix + CLIP need ~2.3 GB each — so only **one heavy family** may
reside on the GPU at a time. This manager enforces that, and also offloads the
face-registry models (FaceNet / MobileNetV3) to CPU while a heavy editing
model is resident, restoring them afterwards.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    id: str
    type: str  # 'instruction', 'inpainting', 'image2image', 'surrogate'
    local_path: str
    version: str
    role: str  # 'optimization', 'evaluation', 'held-out'
    vram_estimate_mb: int
    loaded: bool = False

class EditingModelRegistry:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}

    def register(self, info: ModelInfo):
        self.models[info.id] = info

    def get_info(self, model_id: str) -> Optional[ModelInfo]:
        return self.models.get(model_id)

    def get_by_role(self, role: str) -> list[ModelInfo]:
        return [m for m in self.models.values() if m.role == role]
        
    def get_by_type(self, type: str) -> list[ModelInfo]:
        return [m for m in self.models.values() if m.type == type]

class EditingModelManager:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._lock = threading.Lock()
        
        self.registry = EditingModelRegistry()
        # Register known models
        self.registry.register(ModelInfo(
            id="timbrooks/instruct-pix2pix",
            type="instruction",
            local_path="timbrooks/instruct-pix2pix",
            version="1.0",
            role="held-out",  # never seen by the optimizer; real editing benchmark
            vram_estimate_mb=2300
        ))
        self.registry.register(ModelInfo(
            id="stable-diffusion-v1-5/stable-diffusion-inpainting",
            type="inpainting",
            local_path="stable-diffusion-v1-5/stable-diffusion-inpainting",
            version="1.5",
            role="evaluation",  # masked inpainting evaluation (real model)
            vram_estimate_mb=2600
        ))
        self.registry.register(ModelInfo(
            id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            type="surrogate",
            local_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
            version="1.5",
            role="optimization",  # differentiable anti-diffusion surrogate
            vram_estimate_mb=4100
        ))
        # The same SD1.5 repo also serves as an image-to-image *evaluation*
        # editor (style-transfer attack).
        self.registry.register(ModelInfo(
            id="stable-diffusion-v1-5/stable-diffusion-v1-5#img2img",
            type="image2image",
            local_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
            version="1.5",
            role="evaluation",
            vram_estimate_mb=2600
        ))
        
        # We will hold instances of loaded models keyed by their repo_id/id
        self._loaded_models: Dict[str, Any] = {}
        
        self._clip = None
        self._clip_processor = None
        self._registry_offloaded = False

    # ------------------------------------------------------------------
    # availability
    # ------------------------------------------------------------------
    @staticmethod
    def _local_path(repo_id: str) -> bool:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415

        try:
            for r in scan_cache_dir().repos:
                if r.repo_id == repo_id:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def available(self) -> dict:
        """Report which editing-model families are usable (all local)."""
        av = {}
        for m_id in self.registry.models:
            av[m_id] = self._local_path(m_id)
        clip_id = "openai/clip-vit-large-patch14"
        av[clip_id] = self._local_path(clip_id)
        av["clip"] = av[clip_id]  # legacy alias
        return av

    def surrogate_available(self) -> bool:
        """True when the anti-diffusion surrogate (SD1.5) is local.

        This is the ONLY model the *production* protection engine needs —
        the InstructPix2Pix / inpainting / CLIP models are research
        benchmark tooling and must not gate user processing.
        """
        a = self.available()
        return a.get("stable-diffusion-v1-5/stable-diffusion-v1-5", False)

    def editing_available(self) -> bool:
        """True when every benchmark editor/scorer is local (research only)."""
        a = self.available()
        return (
            a.get("timbrooks/instruct-pix2pix", False)
            and a.get("stable-diffusion-v1-5/stable-diffusion-v1-5", False)
            and a.get("openai/clip-vit-large-patch14", False)
            and a.get("stable-diffusion-v1-5/stable-diffusion-inpainting", False)
        )

    # ------------------------------------------------------------------
    # registry offload (free VRAM for the heavy models)
    # ------------------------------------------------------------------
    def offload_registry(self, registry) -> None:
        if self.device != "cuda" or self._registry_offloaded:
            return
        moved = 0
        for model in registry._models.values():
            tm = getattr(model, "torch_model", None)
            if tm is not None and next(tm.parameters(), None) is not None and next(tm.parameters()).is_cuda:
                tm.to("cpu")
                moved += 1
        if moved:
            torch.cuda.empty_cache()
            logger.info("Offloaded %d registry model(s) to CPU for the editing stage.", moved)
        self._registry_offloaded = True

    def restore_registry(self, registry) -> None:
        if self.device != "cuda" or not self._registry_offloaded:
            return
        moved = 0
        for model in registry._models.values():
            tm = getattr(model, "torch_model", None)
            if tm is not None and next(tm.parameters(), None) is not None and next(tm.parameters()).is_cpu:
                try:
                    tm.to(self.device)
                    moved += 1
                except Exception:  # noqa: BLE001
                    logger.warning("Could not restore %s to GPU.", model.info.id, exc_info=True)
        self._registry_offloaded = False
        if moved:
            logger.info("Restored %d registry model(s) to GPU.", moved)

    # ------------------------------------------------------------------
    # one-at-a-time loading
    # ------------------------------------------------------------------
    
    def get_editor(self, repo_id: str, editor_type: str = "instruction"):
        from .adapter import InstructPix2PixTarget, InpaintingTarget, Image2ImageTarget  # noqa: PLC0415

        with self._lock:
            key = repo_id + ("#img2img" if editor_type == "image2image" else "")
            if key not in self._loaded_models:
                self._ensure_only(key)
                if editor_type == "instruction":
                    target = InstructPix2PixTarget(repo_id=repo_id, device=self.device)
                elif editor_type == "inpainting":
                    target = InpaintingTarget(repo_id=repo_id, device=self.device)
                elif editor_type == "image2image":
                    target = Image2ImageTarget(repo_id=repo_id, device=self.device)
                else:
                    raise ValueError(f"Unknown editor type: {editor_type}")
                self._loaded_models[key] = target
                info = self.registry.get_info(key)
                if info:
                    info.loaded = True
            return self._loaded_models[key]

    def get_ip2p(self, repo_id: str | None = None):
        return self.get_editor(repo_id or "timbrooks/instruct-pix2pix", "instruction")

    def get_surrogate(self, repo_id: str | None = None, **kwargs):
        from .surrogate import AntiDiffusionSurrogate  # noqa: PLC0415

        repo_id = repo_id or "stable-diffusion-v1-5/stable-diffusion-v1-5"
        with self._lock:
            if repo_id not in self._loaded_models:
                self._ensure_only(repo_id)
                target = AntiDiffusionSurrogate(repo_id=repo_id, device=self.device, **kwargs)
                self._loaded_models[repo_id] = target
                info = self.registry.get_info(repo_id)
                if info:
                    info.loaded = True
            return self._loaded_models[repo_id]

    def get_clip(self, repo_id: str | None = None):
        with self._lock:
            if self._clip is None:
                self._ensure_only("clip")
                from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

                self._clip = CLIPModel.from_pretrained(
                    repo_id or "openai/clip-vit-large-patch14",
                    torch_dtype=torch.float16,
                    local_files_only=True,
                ).to(self.device)
                self._clip_processor = CLIPProcessor.from_pretrained(
                    repo_id or "openai/clip-vit-large-patch14", local_files_only=True
                )
            return self._clip

    @property
    def clip_processor(self):
        return self._clip_processor

    def _ensure_only(self, active_id: str) -> None:
        """Release every other heavy family before loading ``active_id``."""
        to_remove = []
        for loaded_id, model in self._loaded_models.items():
            if loaded_id != active_id:
                model.unload()
                to_remove.append(loaded_id)
                info = self.registry.get_info(loaded_id)
                if info:
                    info.loaded = False
        
        for loaded_id in to_remove:
            del self._loaded_models[loaded_id]
            
        if active_id != "clip" and self._clip is not None:
            self._clip = None
            self._clip_processor = None
            
        if to_remove or active_id != "clip":
            import gc  # noqa: PLC0415
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def unload_all(self) -> None:
        with self._lock:
            for loaded_id, model in self._loaded_models.items():
                model.unload()
                info = self.registry.get_info(loaded_id)
                if info:
                    info.loaded = False
            self._loaded_models.clear()
            
            self._clip = None
            self._clip_processor = None
            import gc  # noqa: PLC0415

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()


_manager: dict[str, EditingModelManager] = {}


def get_editing_manager(device: str = "cuda") -> EditingModelManager:
    if device not in _manager:
        _manager[device] = EditingModelManager(device)
    return _manager[device]

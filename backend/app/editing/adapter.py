"""Editing-model adapters.

``BaseEditingTarget`` is the common interface; ``InstructPix2PixTarget`` wraps
the local InstructPix2Pix pipeline (instruction-guided image editing). 
``InpaintingTarget`` wraps a masked diffusion inpainting pipeline.
``Image2ImageTarget`` wraps an image-to-image pipeline for style transformation.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditRequest:
    image_rgb: np.ndarray  # HxWx3 uint8
    instruction: str
    seed: int
    resolution: int
    num_inference_steps: int
    guidance_scale: float
    image_guidance_scale: float
    mask: Optional[np.ndarray] = None # HxW float or uint8 mask for inpainting
    strength: Optional[float] = None # Strength for img2img


class BaseEditingTarget(ABC):
    """Common interface for local editing models."""

    model_id: str = ""

    @abstractmethod
    def edit(self, request: EditRequest) -> np.ndarray:
        """Run one edit. Returns HxWx3 uint8 RGB."""

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory (call before loading another model family)."""

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "adapter": type(self).__name__,
        }


class InstructPix2PixTarget(BaseEditingTarget):
    """Instruction-guided editing via the local InstructPix2Pix pipeline."""

    model_id = "timbrooks/instruct-pix2pix"

    def __init__(self, repo_id: str | None = None, device: str = "cuda") -> None:
        self.repo_id = repo_id or self.model_id
        self.device = device
        self._pipe = None
        self._generator = None

    def _ensure(self) -> None:
        if self._pipe is None:
            from diffusers import StableDiffusionInstructPix2PixPipeline  # noqa: PLC0415

            self._pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
            )
            if self.device == "cuda":
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe = self._pipe.to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            self._generator = torch.Generator(device=self.device).manual_seed(0)
            logger.info("InstructPix2Pix loaded (%s).", self.repo_id)

    def edit(self, request: EditRequest) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        self._ensure()
        img = Image.fromarray(request.image_rgb).convert("RGB")
        res = request.resolution
        if img.size != (res, res):
            img = img.resize((res, res), Image.LANCZOS)
        
        gen = torch.Generator(device=self.device).manual_seed(request.seed)
        out = self._pipe(
            prompt=request.instruction,
            image=img,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            image_guidance_scale=request.image_guidance_scale,
            generator=gen,
        ).images[0]
        return np.asarray(out.convert("RGB")).astype(np.uint8)

    def unload(self) -> None:
        if self._pipe is not None:
            try:
                if self.device == "cuda" and hasattr(self._pipe, "_all_hooks"):
                    self._pipe._all_hooks = []
                self._pipe = None
                self._generator = None
                import gc  # noqa: PLC0415

                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to unload InstructPix2Pix.", exc_info=True)
                self._pipe = None


class InpaintingTarget(BaseEditingTarget):
    """Masked inpainting editing via Stable Diffusion Inpainting pipeline.

    Only the U-Net differs from SD1.5 (9 input channels for image+mask); the
    VAE, text encoder, tokenizer and scheduler are shared with the already-
    local SD1.5 base repo. This keeps the download to a single unet file.
    """

    model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    base_repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def __init__(self, repo_id: str | None = None, device: str = "cuda") -> None:
        self.repo_id = repo_id or self.model_id
        self.device = device
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is None:
            from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel  # noqa: PLC0415

            unet = UNet2DConditionModel.from_pretrained(
                self.repo_id,
                subfolder="unet",
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=True,
            )
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.base_repo_id,
                unet=unet,
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
            )
            if self.device == "cuda":
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe = self._pipe.to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            logger.info("Inpainting model loaded (%s + %s).", self.base_repo_id, self.repo_id)

    def edit(self, request: EditRequest) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415
        
        if request.mask is None:
            raise ValueError("Mask must be provided for InpaintingTarget")

        self._ensure()
        img = Image.fromarray(request.image_rgb).convert("RGB")
        mask_arr = request.mask
        if mask_arr.dtype != np.float32:
            mask_arr = mask_arr.astype(np.float32)
        if mask_arr.min() < 0.0 or mask_arr.max() > 1.0:
            mask_arr = np.clip(mask_arr, 0.0, 1.0)
        mask_img = Image.fromarray((mask_arr * 255).round().astype(np.uint8)).convert("L")
        
        res = request.resolution
        if img.size != (res, res):
            img = img.resize((res, res), Image.LANCZOS)
            mask_img = mask_img.resize((res, res), Image.LANCZOS)
            
        gen = torch.Generator(device=self.device).manual_seed(request.seed)
        out = self._pipe(
            prompt=request.instruction,
            image=img,
            mask_image=mask_img,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=gen,
        ).images[0]
        return np.asarray(out.convert("RGB")).astype(np.uint8)

    def unload(self) -> None:
        if self._pipe is not None:
            try:
                if self.device == "cuda" and hasattr(self._pipe, "_all_hooks"):
                    self._pipe._all_hooks = []
                self._pipe = None
                import gc  # noqa: PLC0415
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to unload Inpainting model.", exc_info=True)
                self._pipe = None


class Image2ImageTarget(BaseEditingTarget):
    """Image-to-image editing for style transfer."""

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5" 

    def __init__(self, repo_id: str | None = None, device: str = "cuda") -> None:
        self.repo_id = repo_id or self.model_id
        self.device = device
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is None:
            from diffusers import StableDiffusionImg2ImgPipeline  # noqa: PLC0415

            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
            )
            if self.device == "cuda":
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe = self._pipe.to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            logger.info("Img2Img model loaded (%s).", self.repo_id)

    def edit(self, request: EditRequest) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        self._ensure()
        img = Image.fromarray(request.image_rgb).convert("RGB")
        res = request.resolution
        if img.size != (res, res):
            img = img.resize((res, res), Image.LANCZOS)
            
        strength = request.strength if request.strength is not None else 0.5
            
        gen = torch.Generator(device=self.device).manual_seed(request.seed)
        out = self._pipe(
            prompt=request.instruction,
            image=img,
            strength=strength,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=gen,
        ).images[0]
        return np.asarray(out.convert("RGB")).astype(np.uint8)

    def unload(self) -> None:
        if self._pipe is not None:
            try:
                if self.device == "cuda" and hasattr(self._pipe, "_all_hooks"):
                    self._pipe._all_hooks = []
                self._pipe = None
                import gc  # noqa: PLC0415
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to unload Img2Img model.", exc_info=True)
                self._pipe = None

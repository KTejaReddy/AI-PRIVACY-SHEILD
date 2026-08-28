"""Differentiable anti-diffusion surrogate (Photoguard-style objective).

The core idea (Salman et al., "Raising the Cost of Malicious AI-powered Image
Editing", Photoguard): a diffusion editor reconstructs its conditioning image
through a VAE + U-Net denoising loop. If the *latent* of the protected photo
is perturbed so that the U-Net's denoising reconstruction error is maximized,
the editor's reconstruction of the protected subject degrades — while the
decoded image stays perceptually identical to a human.

Photoguard performs the attack in VAE latent space, not pixel space: pixel
space L-inf attacks are visible, but a small latent perturbation decodes to a
clean-looking image and survives the editor's own VAE re-encode, because the
editor uses the same VAE. This surrogate therefore exposes:

    encode(x) -> z                 (VAE encoder, torch)
    denoising_loss(z) -> scalar    (MSE between U-Net's predicted and the
                                    fixed seeded noise; differentiable in z)
    decode(z) -> x                 (VAE decoder)

Noise and timestep are fixed per call (seeded) so the objective is
deterministic. SD1.5 U-Net/VAE/text-encoder run in fp16 and are released
after the optimization stage to respect the ~4.3 GB VRAM budget.
"""
from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger(__name__)


class AntiDiffusionSurrogate:
    """Wraps SD1.5 VAE + U-Net + text encoder for the latent-space attack."""

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def __init__(
        self,
        repo_id: str | None = None,
        device: str = "cuda",
        resolution: int = 256,
        timestep: int = 250,
        prompt: str = "a photo of a person",
    ) -> None:
        self.repo_id = repo_id or self.model_id
        self.device = device
        self.resolution = resolution
        self.timestep = timestep
        self.prompt = prompt
        self._lock = threading.Lock()
        self._loaded: dict = {}
        self._prompt_emb: torch.Tensor | None = None
        self._fixed_noise: torch.Tensor | None = None

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return bool(self._loaded)

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel  # noqa: PLC0415
            from transformers import CLIPTextModel, CLIPTokenizer  # noqa: PLC0415

            vae = AutoencoderKL.from_pretrained(
                self.repo_id, subfolder="vae", torch_dtype=torch.float16, local_files_only=True
            ).to(self.device)
            unet = UNet2DConditionModel.from_pretrained(
                self.repo_id, subfolder="unet", torch_dtype=torch.float16, local_files_only=True
            ).to(self.device)
            text_enc = CLIPTextModel.from_pretrained(
                self.repo_id, subfolder="text_encoder", torch_dtype=torch.float16, local_files_only=True
            ).to(self.device)
            tokenizer = CLIPTokenizer.from_pretrained(self.repo_id, subfolder="tokenizer", local_files_only=True)
            scheduler = DDPMScheduler.from_pretrained(self.repo_id, subfolder="scheduler", local_files_only=True)
            self._loaded = {
                "vae": vae,
                "unet": unet,
                "text_encoder": text_enc,
                "tokenizer": tokenizer,
                "scheduler": scheduler,
            }
            tokens = tokenizer([self.prompt], return_tensors="pt").to(self.device)
            with torch.no_grad():
                self._prompt_emb = text_enc(tokens.input_ids)[0]
            lat_h = self.resolution // 8
            self._fixed_noise = torch.randn(
                1, 4, lat_h, lat_h, dtype=torch.float16, device=self.device,
                generator=torch.Generator(self.device).manual_seed(0),
            )
            logger.info("Anti-diffusion surrogate loaded (%s).", self.repo_id)

    def unload(self) -> None:
        with self._lock:
            self._loaded = {}
            self._prompt_emb = None
            self._fixed_noise = None
            import gc  # noqa: PLC0415

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """VAE-encode ``image`` ([1,3,H,W] fp16 in [0,1]) -> latent z (scaled)."""
        vae = self._loaded["vae"]
        return vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE-decode latent z (scaled) -> image tensor in [0,1] (fp16)."""
        vae = self._loaded["vae"]
        return (vae.decode(latent / vae.config.scaling_factor).sample).clamp(0.0, 1.0)

    def denoising_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """MSE between the U-Net's predicted and the fixed noise for ``latent``.

        ``latent`` is the *scaled* latent ([1,4,h,w]). The scheduler's
        ``add_noise`` expects unscaled latents, so the scaling is inverted
        inside (matching how the pipeline itself consumes latents).
        """
        unet = self._loaded["unet"]
        scheduler = self._loaded["scheduler"]
        vae = self._loaded["vae"]

        unscaled = latent / vae.config.scaling_factor
        noise = self._fixed_noise
        t = torch.tensor([self.timestep], device=self.device)
        noisy = scheduler.add_noise(unscaled, noise, t)
        pred = unet(noisy, t, encoder_hidden_states=self._prompt_emb).sample
        return torch.nn.functional.mse_loss(pred, noise)

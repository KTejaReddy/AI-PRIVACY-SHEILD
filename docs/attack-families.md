# AI Attack Family Registry

AI Privacy Shield groups AI systems by their **underlying manipulation
mechanism**, not by vendor. The engine targets representative accessible
local models from each family, so new model families can be added by
registering a representative adapter — never by adding a "blocker" for a
specific commercial product.

The registry lives in `backend/app/attack_registry.py` and is the single
source of truth for which families exist, which models cover them, and each
model's role (`optimization` / `evaluation` / `held_out`).

## Families A–I

| ID | Family | Mechanism | Protection target | Research basis |
| --- | --- | --- | --- | --- |
| `diffusion_editing` | Diffusion image editing | latent denoising | diffusion/latent perturbation | PhotoGuard end-to-end attack |
| `inpainting` | Masked inpainting | masked denoising | mask-robust protection | DiffusionGuard mask augmentation |
| `instruction_editing` | Instruction-guided editing | image + text conditioning | instruction-edit disruption | EditShield |
| `image_to_image` | Image-to-image | encode/noise/denoise | img2img protection | PhotoGuard img2img |
| `identity_reference` | Reference-conditioned generation | reference encoder → new image | reference/identity disruption | identity-encoder surrogate |
| `face_swap` | Identity transfer | source face → target | identity-source disruption | face-embedding surrogate |
| `image_to_video` | Image/video generation | reference + motion | reference/video disruption | spec §17 (adapter, not tested) |
| `vlm_conditioning` | Vision-language conditioning | image + language rep | VLM representation eval | CLIP-style backbones |
| `vision_encoder` | General vision encoders | learned representation | feature disruption | transferability |

## Representative local models and roles

| Model | Family(ies) | Role | Why |
| --- | --- | --- | --- |
| SD1.5 U-Net + VAE + text (anti-diffusion surrogate) | diffusion_editing, inpainting, image_to_image | **optimization** | differentiable denoising-error objective |
| FaceNet (VGGFace2) | identity_reference, face_swap | **optimization** | differentiable identity embedding |
| MobileNetV3-Large | vision_encoder | **optimization** | differentiable global feature extractor |
| SD1.5 inpainting U-Net | inpainting | evaluation | real masked-inpainting tests |
| SD1.5 image-to-image | image_to_image | evaluation | style/attribute edits |
| FaceNet (CASIA-WebFace) | identity_reference, face_swap | evaluation | within-family transfer |
| CLIP ViT-L/14 | vlm_conditioning, vision_encoder | evaluation | semantic/representation scoring |
| InstructPix2Pix | instruction_editing | **held out** | never seen by the optimizer |
| ArcFace (MobileFaceNet) | identity_reference, face_swap | **held out** | cross-identity-encoder transfer |
| ResNet50 | vision_encoder | **held out** | cross-encoder transfer |
| image-to-video adapter (stub) | image_to_video | held out | documented unavailable |

**Optimization models generate the perturbation. Evaluation models test it.
Held-out models never participate in generation** — they are the evidence
that protection transfers beyond what the optimizer saw.

## One perturbation, many families

All optimization-model objectives are summed into a **single PGD loop** in
`backend/app/editing/protector.py`:

```text
TOTAL_LOSS =
    diffusion denoising-error (variants: resize/brightness/contrast/crop/JPEG-approx/noise)
  + W_IDENTITY * face-embedding distance to the original identity
  + W_VISION   * global-representation distance to the original
```

The result is one protected image — never one image per AI. Weights are
centralized in `backend/app/config.py` and can be overridden per call by the
adaptive red-team loop (`scripts/benchmark_protection.py`).

## Reporting

The pipeline reports `families` in every result payload: which families the
current profile targets, and which models cover them. The honest claim is:

> *protected against the tested attack families in the benchmark* —

never *"no AI can edit this image"*.

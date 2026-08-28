# AI-Editing Protection

> **Primary objective:** a human sees and uses the protected photo normally,
> while a **tested** AI image editor has substantially reduced ability to use
> that photo as a source for unauthorized editing.

Recognition protection (face detection / identity embedding disruption) remains
as a secondary layer. This document describes the editing objective: why it
exists, the threat model, the models used, how the perturbation is generated,
how the benchmark is scored, and what is *not* guaranteed.

---

## 1. Why editing protection is the primary objective

Face-recognition protection makes *recognition* fail. That is useful, but a
photo is most often misused by **AI image editors** — instruction-guided,
image-to-image, inpainting — which can change a shirt, the background, add an
accessory, or restyle a photo. The threat is the *edit*, not the *label*.

A photo can be completely recognizable to humans **and** recognition models
yet still be a perfect conditioning input for an editor. So the primary
objective is measured directly: does the same editor, given the same
instruction, succeed less on the protected photo than on the original?

## 2. Threat model

The application protects against **tested, local editing workflows** across
three editor families:

| Editor | Attack type | Role |
|---|---|---|
| InstructPix2Pix | instruction-guided editing | **held out** — never seen by the optimizer |
| SD1.5 inpainting (9-channel U-Net) | masked inpainting with real region masks | evaluation |
| SD1.5 image-to-image | text-guided regeneration / style transfer | evaluation |

The optimizer additionally targets the class of diffusion editors that
reconstruct their conditioning image through a VAE + U-Net denoising loop
(SD1.5-family surrogate). It makes **no claim** about every possible
generative model — different architectures may be unaffected.

Benchmark tasks (controlled, reproducible, measurable — each has a
region-aware pixel metric, see §5):

| Task | Instruction | Primary metric (region) |
|---|---|---|
| Change shirt color | "Make the shirt red." | redness change in the shirt region |
| Change background | "Make the background a beach." | structural change in the background mask |
| Add a hat | "Add a hat." | structural change in the hair/top-of-head region |
| Change lighting | "Make the lighting warm sunset light." | warm-light shift (mean R − B) |
| Convert to pencil sketch | "Make it a pencil sketch." | saturation drop + edge-density rise |
| Change hairstyle | "Change the hairstyle." | structural change in the hair region |

Only tasks the local editors can realistically perform are included.

## 3. Models

| Model | Role | Loaded where |
|---|---|---|
| **SD1.5 U-Net + VAE + text encoder** | *Optimization surrogate* — differentiable anti-diffusion objective | local, fp16, CPU-offloaded |
| **InstructPix2Pix** | *Held-out evaluation editor* — instruction benchmark | local, fp16, CPU-offloaded |
| **SD1.5 inpainting U-Net** | *Evaluation editor* — masked-inpainting attacks (VAE/text shared with SD1.5) | local, fp16, CPU-offloaded |
| **SD1.5 image-to-image** | *Evaluation editor* — style-transfer attacks | local, fp16, CPU-offloaded |
| **CLIP ViT-L/14** | *Auxiliary scorer* — semantic alignment only | local, fp16 |

InstructPix2Pix is **not** used during optimization — it is held out, so the
benchmark measures transferability to a real editor the optimizer never saw.
The inpainting and image-to-image editors are evaluation-only too.

## 4. How the perturbation is generated

Photoguard-style anti-diffusion attack ("Raising the Cost of Malicious
AI-powered Image Editing", Salman et al.):

```
denoising error(x) = MSE( UNet( VAE-encode(x) + noise(t) ), noise )
protected = x + δ,  where δ maximizes denoising error
```

* A small perturbation δ is optimized with a projected gradient loop so that
  the U-Net's **denoising reconstruction error** on the protected photo is
  maximized. A diffusion editor conditions on the photo through the same
  VAE/U-Net machinery, so a higher denoising error means the editor's
  reconstruction of the protected subject is worse.
* **Transformation-aware:** each PGD step maximizes the error over several
  differentiable proxies of common transforms — downscale/upscale (resize),
  brightness, contrast, center crop, an 8×8 block-averaging JPEG approximation,
  and additive noise — so the perturbation is not tuned to one clean image.
* **High-frequency projection:** the raw gradient through the VAE encoder is
  low-frequency (smooth) and therefore visible. Each PGD step keeps only the
  high-frequency component of δ, so the perturbation stays imperceptible
  (measured SSIM ≈ 0.98–0.99 at the default budget) while still moving the
  denoising loss.
* **Bounded:** ‖δ‖∞ ≤ `PERTURBATION_EPSILON × EDITING_SURROGATE_EPSILON_FRACTION`
  (default 0.035 × 0.5 = 0.0175 ≈ 4.5/255).
* **Quality floor:** the stage is **reverted** if SSIM vs. the input drops
  below `EDITING_MIN_SSIM` (0.90). The image is never visibly damaged to win
  the attack.
* Deterministic: fixed timestep (`EDITING_SURROGATE_TIMESTEP`, default 250 of
  1000), fixed seeded noise, fixed prompt conditioning ("a photo of a person").

## 5. How editing success is measured

For every task the **exact same edit** runs on the original and the protected
photo — same editor, instruction, seed, resolution, inference steps, guidance,
mask. Only the input image changes.

**Primary evidence is a task-specific pixel metric** measured in the region
the edit should affect (`app/evaluation/task_metrics.py`), e.g. redness in the
shirt region, structural change in the background mask, or saturation drop for
a sketch. These are deterministic, documented, and normalized to [0, 1].

**CLIP is only an auxiliary check**:

```
composite success = W_TASK × task_metric(0..1)
                  + W_CLIP × clip( (cos(CLIP(out), target) − cos(CLIP(in), target)) / scale, 0, 1 )
```

with defaults W_TASK = 0.6, W_CLIP = 0.4, scale = 0.1. Both raw components
(the pixel metric and the signed CLIP delta) are reported in the JSON so
nothing is hidden.

**Change reporting is honest:**

* `absolute_change = success_original − success_protected` — always reported.
* `relative_change_pct` — reported **only when meaningful** (original success
  ≥ 0.02). When the underlying metric is near zero or crosses zero, the
  relative percentage is omitted (`n/a`) instead of printing an inflated
  number like "112%".

Per-task secondary metrics (unchanged): `semantic_preservation`
(cos(CLIP(in), CLIP(out))), `edit_magnitude` (normalized L2 pixel change),
`protected_region_change` (L2 change inside the face boxes).

### Seed / prompt / mask robustness

The benchmark script runs every task over **multiple seeds** (default 42, 7,
1337) and reports mean/median/std/min/max per task, plus **prompt variants**
(same task re-phrased 3 ways, `--prompts`) and **multiple masks** for the
inpainting attack (shirt, hair, background, person, irregular). The in-app
benchmark stays fast (single seed, canonical prompt) but runs all three
editors.

### Editing success after transformations

`run_robustness()` applies **real** transforms to the protected image
(JPEG q70, resize, crop, brightness, contrast, re-encode) and then runs the
editing attack on the transformed image — the meaningful robustness question
for editing protection, not embedding distance.

## 6. Where it runs in the pipeline

```
analyze → faces → sensitive → protect (recognition) → treat
        → editing (anti-diffusion perturbation, transformation-aware)
        → test (quality, robustness, perception — measured on the final image)
        → editing_benchmark (original vs protected: IP2P + inpainting + img2img)
        → robustness (edit success after real transforms on the protected image)
        → finalize → cleanup
```

The heavy editing models are loaded **one at a time** and released after each
stage; the face-registry models are offloaded to CPU while a heavy editing
model is resident, so everything fits in ~4.3 GB VRAM (with shared-memory
spill under peaks). If the models are not downloaded, the stages report
"unavailable on this hardware" instead of failing silently.

## 7. Reproducibility

```bash
# download the local models once (~11 GB, see docs section 8)
python scripts/download_editing_models.py

# full benchmark over data/benchmark/* + tests/fixtures/*
python scripts/benchmark_protection.py

# custom runs
python scripts/benchmark_protection.py photo1.jpg photo2.jpg --out results/
python scripts/benchmark_protection.py --tasks t01_shirt_color,t05_sketch --seeds 42,7,1337
python scripts/benchmark_protection.py --prompts --masks shirt,hair,irregular
python scripts/benchmark_protection.py --transformations jpeg_compression,resize,crop
python scripts/benchmark_protection.py --strengths 0.4,0.6,0.8   # img2img denoising-strength sweep
```

It writes `results.json`, `results.csv`, `report.md` and `report.html` with
environment (GPU/VRAM/Python/torch/diffusers), seeds, prompt-variant flag,
masks, transformations, optimizer config, per-task results, per-image
aggregates, and cross-image statistics.

## 8. Hardware

| | Requirement |
|---|---|
| GPU (recommended) | NVIDIA with ≥ 4 GB VRAM (CUDA); peaks use shared memory |
| CPU-only | Works but each edit takes minutes; reduce `EDITING_RESOLUTION` |
| RAM | ≥ 8 GB recommended (models stream through CPU offload) |
| Disk | ~11 GB for all model families (SD1.5 ~7 GB, IP2P ~5 GB, inpainting U-Net ~1.7 GB, CLIP ~2 GB) |

## 9. What this does NOT guarantee

* No claim of "no AI can edit this image" — different architectures may ignore
  the perturbation.
* Protection is measured against **tested** editors (InstructPix2Pix,
  SD1.5 inpainting, SD1.5 img2img) and the surrogate (SD1.5); it does not
  transfer to every model.
* A weak result is reported honestly (a task where protected success rose is
  shown with its real negative change).
* Human-perceived quality is preserved by constraint, but a human can always
  zoom in; the perturbation is bounded and imperceptible, not invisible to
  pixel-peeping.

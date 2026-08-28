# Robustness

## Goal

The perturbation must not be tuned to a single untouched file. Common
real-world image handling — JPEG re-encoding, resizing, cropping, brightness and
contrast tweaks, re-uploads — must not silently erase the protection. The same
applies to the **AI-editing** perturbation: a protected photo that only works
when served byte-identical is not real protection.

## Robustness of the editing protection

Editing protection has its own robustness path — and it measures the right
thing: **edit success after the transform**, not embedding distance.

* `EditingBenchmark.run_robustness()` applies real transformations (JPEG q70,
  resize, crop, brightness, contrast, re-encode — the same transform set as the
  recognition test) to the **protected** image, then runs the same AI editor
  with the same instruction/seed/settings on the transformed image and reports
  the edit success. If JPEG erases the protection, the report shows
  protected-success 0.03 → after-JPEG 0.12 — it is not hidden.
* The anti-diffusion optimizer is itself **transformation-aware**: every PGD
  step maximizes the denoising error over differentiable proxies of resize,
  brightness, contrast, center-crop and an 8×8 block-averaging JPEG
  approximation (see `app/editing/protector.py`), so the perturbation is not
  tuned to one byte-identical file.

## How robustness enters the optimization

Robustness is a **first-class objective in both phases**, not just a post-hoc
test.

**Phase 1 (differentiable).** Every iteration re-embeds the candidate through
differentiable approximations of the common transforms and penalizes any
variant whose embedding distance falls below the target:

- scale (0.75× down/up), center-crop (90%), gamma (0.9), 2% translation,
  ±1% noise, contrast ×1.15, brightness ×0.9, Gaussian blur, and a
  differentiable DCT-block JPEG surrogate (orthonormal 8×8 DCT with
  per-block quantization — validated to match real JPEG's behavior, identity
  on a clean round-trip).

**Phase 2 (black-box).** Real, non-differentiable transforms are applied to
candidate images during refinement: JPEG-70, 0.75× resize, 10% center crop,
brightness ×0.9, contrast ×1.15, PNG+JPEG-85 re-encode. Three of the six are
sampled per gradient call (rotating) so every transform is seen every two
iterations while cost stays bounded.

**Transform sets are reproducible**: the same seed, budget, and image give the
same trajectory.

## Why not just test after?

Post-hoc testing alone cannot *improve* the perturbation — it only reports
failures. Optimizing through (approximate) transforms during generation is what
moves the measured numbers; the post-hoc robustness test then verifies the
result independently with real encoders.

## Measured limits (honest)

With ε ≈ 9/255 and SSIM ≥ 0.9, embedding disruption that is strong on the
clean image degrades under lossy transforms:

| transform | measured mean L2 distance (test portrait) |
|---|---|
| clean (no transform) | ~0.55–0.75 |
| JPEG quality 70 | ~0.34–0.38 |
| resize 0.75× | ~0.34–0.38 |
| center crop 10% | ~0.51–0.56 |
| brightness ×0.9 | ~0.39–0.43 |
| contrast ×1.15 | ~0.40–0.45 |
| PNG + JPEG-85 re-encode | ~0.33–0.38 |

The PARTIAL threshold is 0.40 (worst-model rule). JPEG, resize and re-encode
sit just below it; crop and contrast hover around it. Raising the perturbation
budget above the quality floors is the main lever for stronger post-processing
survival; the system deliberately keeps quality high instead.

## Threat model for transformations

The perturbation is designed for **common digital handling** (the transforms
above). It is *not* designed for: physical re-photography, screenshots with
heavy downscaling, or adversarial pre-processing (e.g., a system trained to
strip adversarial noise). Those are outside the threat model (see
`docs/limitations.md`).

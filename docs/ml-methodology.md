# ML methodology

## Problem statement

The **primary** objective is editing protection: given a photograph, produce a
visually near-identical image that a human uses normally while a **tested** AI
image editor has substantially reduced ability to use it as a source for
unauthorized editing. Recognition protection (face detectors, person
detectors, identity embedders, image encoders) remains as a secondary layer.
Both attacks are imperceptible by design: perturbations are bounded
(`‖δ‖∞ ≤ ε`, default 9/255 for recognition, ≈ 4.5/255 for the editing stage)
and candidates violating SSIM/PSNR floors are rejected.

## Anti-diffusion editing objective (Photoguard-style)

A diffusion editor (e.g., InstructPix2Pix) conditions on the photo through a
VAE + U-Net denoising loop. If the photo's denoising reconstruction error is
maximized, the editor's reconstruction of the subject degrades:

```
L_edit(δ) = − MSE( UNet( VAE-encode(original + δ) + fixed_noise(t) ), fixed_noise )
```

Maximized by projected gradient descent with a **high-frequency projection**
(the raw gradient through the VAE encoder is smooth and visible; each step
keeps only the high-frequency component of δ) and bounded at
`PERTURBATION_EPSILON × EDITING_SURROGATE_EPSILON_FRACTION`. A hard SSIM
floor reverts the stage if it becomes visible. Deterministic by construction:
fixed timestep (250/1000), seeded noise, fixed prompt conditioning.

**Held-out evaluation:** the optimizer only ever sees the SD1.5 surrogate;
InstructPix2Pix runs the real benchmark and CLIP ViT-L/14 scores it. If the
protected photo reduces edit success on that held-out editor, the protection
transfers beyond the surrogate. Task instructions, seeds, resolution, steps
and guidance are identical for original and protected; only the input image
changes.

## Recognition objective (secondary layer)

## Optimization: what is optimized, and against what

`protected = original + δ`, with `δ = mask ⊙ δ₀`, `|δ₀|∞ ≤ ε`. The soft region
mask concentrates the budget on faces (weight 1.0) and persons (0.6), with a
small controlled context dither (0.18) so detection suppression has context to
work with and no hard mask boundary is visible.

### Phase 1 — differentiable multi-objective (white-box)

Surrogates (all differentiable, loaded from local weights):

| model | role | mode |
|---|---|---|
| FaceNet Inception-ResNet v1 (VGGFace2) | identity disruption | face crops |
| FaceNet Inception-ResNet v1 (CASIA-WebFace) | identity disruption | face crops |
| MobileNetV3-Large (ImageNet) | vision-feature disruption | whole image |
| MTCNN P-Net / R-Net / O-Net | face-detection suppression | face crops at cascade scales |

Objective (Adam, projected onto the ε-ball):

```
L = W_IDENTITY · relu(margin − ‖e_adv − e_orig‖₂)          (face embeddings)
  + W_VISION   · relu(VISION_MARGIN − ‖f_adv − f_orig‖₂)    (global features)
  + W_FACE_DET · Σ relu(logit_stage − target_logit_stage)    (MTCNN P/R/O-Net)
  + W_SSIM · (1 − SSIM) + W_MSE · MSE                        (perceptual)
  + W_ROBUSTNESS · Σ_m,var relu(target − ‖e(var(adv)) − e_orig‖₂)
  + W_PERTURBATION · ‖δ‖₂
```

Key design decisions:

- **Logit-space detection loss.** P-Net/R-Net/O-Net probabilities are
  saturated near 1.0 for a clear face; a perturbation moves a saturated
  probability by ~1e-4, so raw-probability losses give ~zero gradient. Taking
  `log(p/(1−p))` linearizes exactly those changes (d logit/dp = 1/(p(1−p)) ≈
  330 at p=0.997). The cascade's crops are matched exactly (square-ified box,
  no margin) so the surrogate transfers to the real cascade.
- **Vision-feature disruption is a primary layer, not an afterthought.** The
  whole-image MobileNetV3 embedding is pushed past `VISION_MARGIN` (0.55),
  which is what the ResNet50 held-out evaluation measures.
- **Transformation-aware from the start.** Differentiable approximations of
  scale, center-crop, gamma, translate, noise, contrast, brightness, blur and
  a DCT-block JPEG surrogate are sampled **every iteration**, so the
  perturbation is optimized to survive re-encoding, not just to work on the
  clean file.

### Phase 2 — zeroth-order black-box refinement

Non-differentiable targets join: ArcFace (ONNX), ResNet50 (held-out),
OpenCV SSD, the real MTCNN cascade, HOG, and Faster R-CNN.

Gradients are estimated by central finite differences over **structured**
random directions inside the mask:

- per-pixel texture (light Gaussian smoothing) — moves CNNs the most
  (measured: MTCNN −0.016 at ±16/255),
- Gaussian blobs — low-frequency, survive downsampling,
- mid-frequency sinusoids — transfer across interpolation.

The detection loss is a weighted sum of logit-space log-sum-exp over each
detector's overlapping box scores, so every tested detector is an optimization
target. Embedding floors (0.60) apply to **every** model — a model already
above the floor was previously excluded from the loss, letting the detection
attack drag it back toward the original embeddings; now the embedding gradient
acts as a preserving force. A short embedding-repair pass runs after the
detection attack, and real transformations are part of the embedding objective
during refinement (3 of 6 sampled per iteration, rotating).

## Metric definitions (used consistently everywhere)

- **Embedding distance**: L2 norm of the difference of L2-normalized
  embeddings. Cosine similarity *s* relates by `d = √(2 − 2s)`; both are in
  [0, 2] / [−1, 1]. The robustness tester reports distances; the perception
  panel reports similarities. They are **not** interchangeable — a distance of
  0.40 ≈ similarity 0.92.
- **Detector confidence / score**: the maximum box score (detector-specific
  post-softmax probability for SSD/MTCNN, class score for Faster R-CNN, SVM
  decision weight for HOG) overlapping each target box, averaged over boxes.
  It is an *aggregated detection score*, not a calibrated probability of a
  specific person being present.
- **Change %**: `(after/before − 1) × 100`, computed from the real
  measurements; `—` when the before value is zero or the test did not run.
- **Visual quality**: SSIM (structural similarity), PSNR (dB), perturbation
  L∞ / L2 / MAE, computed on the actual images.

## Reproducibility

The report and benchmark include: random seed, ε, phase-1/phase-2 iteration
budgets, detection-attack fraction, quality floors, model list, hardware
(GPU/CPU), and processing time. `scripts/benchmark.py` re-runs the whole
pipeline and prints a per-image and aggregate report; nothing in it is
hard-coded.

## Honest scope

The results are *model- and transformation-dependent*. The SSD detector is
measured near-invariant at this budget (see limitations); robustness under
lossy transforms lands just below the PARTIAL threshold for face embeddings.
These are reported as measured, never hidden and never faked.

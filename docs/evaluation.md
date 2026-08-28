# Evaluation

## AI-editing benchmark (the primary objective)

`app/evaluation/editing_benchmark.py` measures **AI editing success before vs
after protection** — the central success criterion of the project. For every
controlled task the *exact same* edit runs on the original and on the
protected photo: same editor, same instruction, same seed, same resolution,
same inference steps, same guidance (and same mask for inpainting). Only the
input image changes. Three local editors are used, all evaluation-only:
InstructPix2Pix (**held out** from optimization), SD1.5 masked inpainting
(real region masks derived from the detected face/person boxes — never a mock
mask), and SD1.5 image-to-image.

### Edit-success scoring

A single raw CLIP cosine is not evidence that an edit happened. The primary
score is a **task-specific, region-aware pixel metric** (see
`app/evaluation/task_metrics.py`): redness change in the shirt region,
structural change in the background mask, hair/top-of-head region change for
hats and hairstyles, warm-light shift, or saturation drop + edge-density rise
for sketches. CLIP cosine alignment is only an auxiliary check:

```
composite success = W_TASK × task_metric(0..1)
                  + W_CLIP × clip( (cos(CLIP(out), target) − cos(CLIP(in), target)) / scale, 0, 1 )
```

defaults: W_TASK = 0.6, W_CLIP = 0.4, scale = 0.1. Both raw components are
reported in the JSON so nothing is hidden.

Change reporting is honest:

- `absolute_change = success_original − success_protected` — always reported.
- `relative_change_pct` — reported **only when meaningful** (original success
  ≥ 0.02); otherwise `n/a`. This removes the old practice of printing an
  inflated percentage (e.g. "112%") when the underlying metric crossed zero.
- Per-task stats over seeds (mean/median/std/min/max) are reported when the
  benchmark runs with multiple seeds.

Per-task secondary metrics:

| metric | definition | meaning |
|---|---|---|
| semantic_preservation | cos(CLIP(in), CLIP(out)) | how much of the input survived the edit (unintended change) |
| edit_magnitude | normalized L2 pixel change in→out | how large the edit was |
| protected_region_change | normalized L2 pixel change inside the face box(es) | whether the protected subject changed |

Aggregates: mean original/protected success, mean absolute change, mean
relative change (over valid rows only), and the count of task/editor rows
whose protected success dropped.

### Editing success after transformations

`run_robustness()` applies **real** transformations (JPEG q70, resize, crop,
brightness, contrast, re-encode) to the protected image and then runs the
editing attack on the transformed image. This measures whether the protection
survives common re-encodings — the meaningful robustness question for editing
protection (not embedding distance).

### Benchmark settings (reproducible)

Resolution, inference steps, guidance scale, image-guidance scale, seed,
mask kinds, transformation list, and the scoring weights are identical for
original and protected and are printed in the report. `scripts/benchmark_protection.py`
runs the full pipeline (protection + benchmark + transform robustness) over
any image set and writes `results.json`, `results.csv`, `report.md` and
`report.html` with environment (GPU/VRAM/Python/torch/diffusers), seeds,
prompt-variant flag, masks, transforms, optimizer config, per-task results,
per-image aggregates, and cross-image statistics.

## AI perception test (the "AI Perception Test" panel)

`app/evaluation/perception.py` measures real before/after values for every
system the attack targets, on the actual uploaded image:

| row | detector / model | metric | optimization role |
|---|---|---|---|
| Face (OpenCV SSD) | res10_300x300 SSD | max box confidence over each face box | black-box target |
| Face (MTCNN) | P-Net/R-Net/O-Net cascade | max box confidence over each face box | differentiable surrogate + black-box target |
| Person (HOG) | HOG + linear SVM | max decision weight over each person box | black-box target |
| Person (Faster R-CNN) | ResNet-50 FPN, COCO | max class score over each person box | black-box target |
| Embedding (FaceNet VGGFace2) | Inception-ResNet v1 | cosine similarity, face crops | differentiable surrogate |
| Embedding (FaceNet CASIA) | Inception-ResNet v1 | cosine similarity, face crops | differentiable surrogate |
| Embedding (ArcFace) | w600k_mbf ONNX | cosine similarity, face crops | black-box target |
| Embedding (MobileNetV3) | ImageNet | cosine similarity, whole image | differentiable surrogate |
| Embedding (ResNet50) | ImageNet | cosine similarity, whole image | **held-out** — never optimized against |

Each row shows `before → after → change %`, or `—` when the test did not run
(model unavailable, no face/person detected). The panel is explicit that
results are model- and transformation-dependent.

### Why both optimization and held-out models

FaceNet VGGFace2, FaceNet CASIA and MobileNetV3 are optimized against — the
perturbation is *tuned* to them. ArcFace and ResNet50 are **not** used by the
optimizer; their measured changes test transferability: if the protection
moves a held-out model too, it is less likely to be mere overfitting to the
surrogate family.

## Robustness test (the "Robustness Test" section)

`app/robustness/tester.py` re-embeds the original and the **transformed**
protected image through every loaded model and reports, per transform, the
mean L2 embedding distance and a verdict:

- **PASS** if every model keeps a distance ≥ `DISRUPT_PASS` (0.70),
- **PARTIAL** if every model stays ≥ `DISRUPT_PARTIAL` (0.40),
- **FAIL** otherwise (worst-model rule: one weak model fails the transform).

Transforms (real, applied with common parameters):

| transform | parameters |
|---|---|
| JPEG compression | quality 70 |
| Resize | 0.75× down then back (area → linear) |
| Crop | center 10% crop, back to full size |
| Brightness | × 0.9 |
| Contrast | × 1.15 |
| Re-encode | PNG round-trip + JPEG quality 85 |

Known, measured limitation: under this imperceptible budget the face-embedding
distances after lossy transforms land just below the PARTIAL threshold; the
report says so rather than hiding it (see `docs/limitations.md`).

## Visual quality

SSIM, PSNR, perturbation L∞ / L2 and MAE are computed on the actual images
(`app/quality/metrics.py`). The optimizer enforces floors (`MIN_SSIM` 0.90,
`MIN_PSNR` 30 dB); candidates below them are rejected during refinement.

## Benchmark mode

`scripts/benchmark_protection.py` runs the multi-family protection over one or more local images
and prints, per image: the perception table, robustness table, visual quality,
protection details (iterations, early stop, refinement targets, detector
confidences), plus an aggregate across images. Usage:

```bash
python scripts/benchmark_protection.py                     # default fixture set
python scripts/benchmark_protection.py photo1.jpg photo2.jpg
python scripts/benchmark_protection.py --rounds 2 --seeds 42 # red-team rounds + seeds
```

The header prints seed, ε, iteration budgets, quality floors, and device so a
run can be reproduced.

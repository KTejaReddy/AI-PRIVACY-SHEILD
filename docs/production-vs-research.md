# Production vs Research: two modes, one engine

AI Privacy Shield separates the **user-facing protection engine** from the
**research/benchmark environment**. The user uploads one photo and receives
one protected photo; they never choose models, never see benchmark tables,
and never need to understand machine learning. All the evaluation machinery
lives in research mode.

## Profiles

| | `production` (default) | `research` |
| --- | --- | --- |
| Config | `backend/configs/production.yaml` | `backend/configs/research.yaml` |
| Purpose | generate ONE protected photo per upload | benchmark, red-team, transfer tests |
| Models loaded | protection engine only (SD1.5 surrogate, FaceNet, MobileNetV3) | protection engine + every evaluation/held-out model |
| Benchmark editors | never loaded | InstructPix2Pix, SD1.5 inpainting, SD1.5 img2img, CLIP |
| In-app editing benchmark | off | on |
| Adaptive red-team | off | on (`scripts/benchmark_protection.py`) |
| Env var | `AIPS_PROFILE=production` | `AIPS_PROFILE=research` |

## Production pipeline (what a user runs)

```text
UPLOAD
  → validate → analyze (faces/persons/sensitive regions)
  → recognition protection (secondary layer)
  → MULTI-FAMILY protection (one perturbation: diffusion + identity + vision)
  → quality validation (SSIM/PSNR floor; stage reverts if violated)
  → metadata sanitization
  → Original | Protected | Difference  →  COPY / DOWNLOAD
  → temporary data cleared
```

No model selection. No protection modes. The result page shows the three
images, visual-quality numbers, the targeted attack families, and honest
positioning. Research tables are collapsed behind "Technical details".

## Research mode

```bash
python scripts/benchmark_protection.py [images...] \
    --rounds 2 --seeds 42 --tasks t01_shirt_color,t02_background \
    --masks shirt,hair,background --transformations jpeg_compression,resize \
    --out results/mfam
```

What it does per image:

1. **Protect** — multi-family PGD (diffusion + identity + vision), the same
   engine production uses.
2. **Red-team** — outer rounds: probe each family with cheap surrogates,
   raise the weight of the weakest family, re-protect; stop when gains
   saturate, the quality floor is hit, or the round budget is exhausted.
3. **Evaluate the final image per family** — direct editing (held-out
   InstructPix2Pix, inpainting, img2img), identity-reference / face-swap
   (FaceNet + held-out ArcFace), vision encoders (MobileNetV3 + held-out
   ResNet50 + CLIP), edit success after real transformations.
4. Writes `results.json`, `results.csv`, `report.md`, `report.html`.

Families that cannot run on this hardware (image-to-video) are reported as
**NOT TESTED**, never faked.

## Why the split matters

- **Footprint:** the production app never loads benchmark editors (~6 GB of
  models stay unloaded during normal use).
- **Honesty:** benchmark numbers come from a reproducible research command,
  not from the user-facing UI, so a marketing-looking score can never be
  mistaken for a product guarantee.
- **Extensibility:** adding a new attack family = register a model +
  adapter in `attack_registry.py` and (if it should run during protection)
  a loss term; the research benchmark picks it up automatically.

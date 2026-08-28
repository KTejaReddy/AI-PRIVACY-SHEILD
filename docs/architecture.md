# Architecture

## Overview

AI Privacy Shield is a **local-first** web application. The browser is the
front door; all machine learning runs in a local FastAPI backend on the user's
own machine. No image is ever written to a database, sent to a third-party
service, or kept after the session ends.

```
USER DEVICE
│
├── Browser (React + Vite)
│     ├── upload / validation
│     ├── SSE pipeline progress
│     ├── ORIGINAL | PROTECTED | DIFFERENCE comparison
│     ├── protection report
│     ├── copy / download actions
│     └── client-side cleanup (object URLs, blobs, buffers)
│
└── Local backend (FastAPI + PyTorch + OpenCV + ONNX Runtime)
      ├── POST /api/upload        validate + stage into a temp session dir
      ├── GET  /api/process/{id}  SSE stream: analyze → protect → test → result
      ├── POST /api/cleanup/{id}  delete session temp files (idempotent)
      ├── GET  /api/health        hardware + installed models
      └── GET  /api/models        model registry detail
```

## Processing pipeline

The backend pipeline (`app/processing/pipeline.py`) drives everything and
emits a real stage event for each step:

1. **Image validation** (`app/utils/imaging.py`) — magic-byte sniffing, MIME
   check, size limits, dimension caps, decode + normalize to RGB.
2. **Privacy analysis** (`app/privacy/analyzer.py`) —
   - Face detection — two independent detector families: OpenCV DNN SSD
     (`app/vision/face_detector.py`, res10_300x300) and the MTCNN cascade
     (`app/vision/mtcnn_face_detector.py`, P-Net/R-Net/O-Net). Both support
     0 / 1 / many faces and both are evaluation targets.
   - Person detection — two families: OpenCV's built-in HOG + linear SVM
     full-body detector (`app/vision/person_detector.py`, no download) and a
     neural Faster R-CNN ResNet-50 detector (`app/vision/neural_person_detector.py`,
     torchvision COCO weights). Person boxes widen the protection mask and
     feed the detection-suppression objective.
   - Sensitive content (`app/vision/sensitive.py`) — QR/barcode decoding
     (PyZBar-compatible detector via OpenCV), optional local OCR
     (RapidOCR/ONNX) with PII regex classification (phone, email, ID-like
     patterns), metadata analysis.
3. **Unified protection engine** (`app/editing/protector.py`) — the ONLY
   protection stage. ONE multi-family perturbation is optimized in a single
   PGD loop so AI image editors, inpainting/instruction pipelines and
   reference-conditioned generators conditioned on the protected photo
   reconstruct the subject poorly, while the face's identity embedding and
   the image's global representation are pushed away from the originals (see
   `docs/editing-protection.md`, `docs/attack-families.md`). The legacy
   stacked recognition stage lives in `backend/legacy/adversarial/`.
4. **Robustness testing** (`app/robustness/tester.py`, research profile) —
   re-evaluates the protected image under JPEG compression, resize, crop,
   brightness, contrast, and re-encoding; each transform is PASS / PARTIAL /
   FAIL based on measured embedding distances.
5. **AI perception test** (`app/evaluation/perception.py`, research profile)
   — measures real before/after values for the AI systems the attack
   targets: face-detector confidence, person-detector weight, and per-model
   face embedding similarity.
6. **AI-editing benchmark** (`app/evaluation/editing_benchmark.py`, research
   profile) — runs the original and the protected photo through **three**
   local editors (InstructPix2Pix held out from optimization, SD1.5 masked
   inpainting with real region masks, SD1.5 image-to-image) with identical
   instruction/seed/steps/guidance/mask. Edit success is scored primarily by
   task-specific pixel metrics (region-aware) with CLIP as an auxiliary
   check, and **edit success is also measured after real transformations**
   (JPEG/resize/…) of the protected image.
7. **Sensitive-region treatment** — sensitive regions (QR codes, OCR-flagged
   text, detected document-like areas) are blurred where practical.
8. **Metadata sanitization** (`app/metadata/sanitizer.py`) — EXIF, GPS, XMP,
   IPTC and all other metadata removed from the output image.
9. **C2PA provenance** (`app/metadata/provenance.py`) — optionally embeds a
   cryptographically signed C2PA manifest (content hash, operation, timestamp)
   into the protected PNG. A second, non-AI-blocking layer.
10. **Quality measurement** (`app/quality/metrics.py`) — PSNR, SSIM, MSE and
    perturbation norms computed on the actual images.
11. **Output** — lossless PNG by default (or high-quality JPEG), no debug
    overlays, no difference-map artifacts.
12. **Cleanup** (`app/cleanup/manager.py`) — session temp dir, upload bytes and
    pipeline buffers are deleted; a janitor sweeps stale sessions.

## Data lifecycle

- Uploads are written to `backend/.tmp/sessions/<id>/` with random session
  ids, never to a database, and never logged.
- After the SSE stream finishes, the pipeline deletes its own temp dir. The
  `POST /api/cleanup/{id}` endpoint is a belt-and-braces idempotent cleanup,
  and the browser also revokes its object URLs and clears its image buffers
  after copy/download.
- The janitor removes sessions older than `SESSION_TTL_SECONDS` (default 30
  minutes) so a crashed client cannot leak files.

## The adversarial protection engine

Conceptually: `protected = original + perturbation`, where the perturbation is
the solution of a constrained optimization problem.

**Weighted region mask.** The perturbation support is no longer face-only. A
per-pixel weight map gives faces the full budget (1.0), person bodies a strong
share (`PERSON_REGION_WEIGHT`, default 0.6), and the remaining context a small
controlled dither (`CONTEXT_MASK_WEIGHT`, default 0.18). Detection disruption
often needs surrounding context, while the visual budget stays concentrated on
the person.

**Phase 1 — white-box, multi-model** (differentiable surrogates):
- Surrogates: FaceNet Inception-ResNet v1 (VGGFace2) and FaceNet (CASIA),
  loaded via `facenet-pytorch`, plus a global vision encoder (MobileNetV3-Large,
  ImageNet weights) that pushes **general visual features** away, not just
  identity embeddings. All are differentiable end to end.
- Loss = `W_IDENTITY · identity_loss + W_VISION · vision_loss
  + W_FACE_DET · mtcnn_surrogate_loss + W_SSIM · ssim_penalty
  + W_MSE · mse_penalty + W_ROBUSTNESS · robustness_loss
  + W_PERTURBATION · perturbation_penalty`.
- The identity term pushes each face's embedding **away** from its original
  embedding beyond a target margin; the vision term does the same for
  whole-image features. The robustness term re-embeds differentiable
  transformed variants (scale, crop, gamma, translate, noise, contrast,
  brightness, blur, JPEG surrogate) **every iteration** so the perturbation
  transfers to real JPEG/resize/brightness-like distortions.
- **Face-detection suppression is now a first-class differentiable objective**:
  the MTCNN P-Net, R-Net, and O-Net stages are exposed as differentiable
  surrogates. Their scores are pushed in **logit space** (`log(p/(1-p))`) at
  the exact pyramid scales and square-ified crop geometry the cascade uses —
  saturated probabilities pinned near 1.0 would otherwise give ~zero gradient.
  O-Net's probability *is* the confidence the cascade reports, and R-Net's
  0.7-threshold decision deletes proposals, so these surrogates attack the
  reported confidence directly.
- A projected-gradient update (projected Adam) keeps the perturbation inside
  the L-infinity bound `ε` (default 0.035 → ≈ 9/255 per channel, weighted by
  the region mask) and the output clamped to valid pixel range.

**Phase 2 — black-box refinement** (zeroth order):
- Non-differentiable targets join in: ArcFace (w600k_mbf, ONNX Runtime),
  ResNet50 (held-out vision encoder), the OpenCV SSD face detector, the real
  MTCNN cascade, HOG, and Faster R-CNN. Central finite differences over
  structured random directions (per-pixel texture, Gaussian blobs, sinusoids)
  estimate each objective's gradient; sign-SGD applies them inside the mask.
- The detection loss is **logit-space LSE** over each detector's overlapping
  box scores (with `W_FACE_DET`, `W_MTCNN_DET`, `W_PERSON_DET`,
  `W_NEURAL_PERSON_DET` weights) so saturated confidence contributes real
  gradient signal.
- **Embedding floors are enforced for every model** (0.60), not just weak
  ones: a model already above the floor used to be excluded from the loss,
  which let the detection attack drag it back toward the original embeddings.
  A short embedding-repair pass runs after the detection attack, and the
  attack itself blends in a low-weight embedding gradient.
- Real transformations (JPEG, resize, crop, brightness, contrast, re-encode)
  are part of the embedding objective during refinement, not only a
  post-hoc test.
- CPU mode automatically uses reduced iterations and query counts.

**Phase 3 — multi-family editing protection** (`app/editing/protector.py`):

- One perturbation, several families: a single PGD loop combines the
  **diffusion** term (SD1.5 U-Net + VAE + text encoder, fp16, loaded only for
  this stage; the attack maximizes the **denoising reconstruction error** —
  the mechanism by which a diffusion editor's reconstruction of the subject
  degrades), the **identity-reference / face-swap** term (FaceNet VGGFace2
  embedding of each detected face pushed away from the original identity) and
  the **vision-encoder** term (MobileNetV3 global representation pushed
  away). Weights are configurable (`EDITING_IDENTITY_WEIGHT`,
  `EDITING_VISION_WEIGHT`) and can be overridden by the adaptive red-team
  loop.
- **Transformation-aware:** each PGD step maximizes the error over
  differentiable proxies of resize, brightness, contrast, center-crop and an
  8×8 block-averaging JPEG approximation, so the perturbation is not tuned to
  one byte-identical file.
- **High-frequency projection:** the raw gradient through the VAE encoder is
  low-frequency and visible; each PGD step keeps only the high-frequency
  component of δ, keeping the perturbation imperceptible (SSIM ≈ 0.98–0.99).
- Bounded at `PERTURBATION_EPSILON × EDITING_SURROGATE_EPSILON_FRACTION`
  (≈ 4.5/255) with a hard SSIM floor (`EDITING_MIN_SSIM`); the stage is
  reverted if the floor is violated.
- The benchmark editors (InstructPix2Pix, inpainting, img2img) are **all
  held out / evaluation-only** — never used by the optimizer — so the
  measured editing reduction tests transferability.

All numbers the UI shows (embedding distances, SSIM, PSNR, robustness
PASS/PARTIAL/FAIL, editing-success reduction) come from these computations —
nothing is hard-coded.

## Attack family registry and profiles

`app/attack_registry.py` classifies every model by AI attack family (A–I) and
role (`optimization` / `evaluation` / `held_out`) — see
`docs/attack-families.md`. Two deployment profiles are loaded from
`backend/configs/`:

- **production.yaml** — the protection engine only (SD1.5 surrogate, FaceNet,
  MobileNetV3). The user app loads nothing else; the in-app editing benchmark
  is off and heavy benchmark editors are never loaded.
- **research.yaml** — every evaluation/held-out model. Used by
  `scripts/benchmark_protection.py`, which also runs the **adaptive red-team
  loop**: probe each family, raise the weight of the weakest family,
  re-protect, stop on saturation/quality/compute budget.

See `docs/production-vs-research.md` for the split rationale.

## Verification

`app/robustness/tester.py` re-embeds the original and protected images through
every loaded verification model, computes L2 distances on normalized
embeddings (range [0, 2]) and classifies each model and each transform:

- distance ≥ `DISRUPT_PASS` (0.70) → PASS
- distance ≥ `DISRUPT_PARTIAL` (0.40) → PARTIAL
- else → FAIL

The same numbers power the report, the demo panel, and the honest
"protection depends on tested models" disclaimer.

## AI-editing benchmark

`app/evaluation/editing_benchmark.py` runs the **primary** objective: for each
controlled task, the exact same edit (same editor, instruction, seed,
resolution, steps, guidance, mask) is run on the original and the protected
photo. Three local editors are used, all evaluation-only:

- **InstructPix2Pix** — instruction-guided (held out from optimization),
- **SD1.5 masked inpainting** — masks derived from the detected face/person
  boxes (shirt, hair, background, person, irregular), never a mock mask,
- **SD1.5 image-to-image** — style/global regeneration.

Edit success = `W_TASK × task-specific pixel metric` (region-aware, e.g.
redness in the shirt region or background-region change) `+ W_CLIP × CLIP`
semantic alignment. The report shows original → protected success, absolute
change (always) and relative change (only when meaningful), plus
semantic preservation, edit magnitude, protected-region change, per-seed
stats, and edit-success-under-transform robustness. Failures (e.g. a task
where protected success rose) are shown with their real values.

## AI perception test

`app/evaluation/perception.py` runs the protected image back through the
**target systems** and reports real before/after values:

- face-detector confidence — OpenCV SSD **and** MTCNN cascade (mean over
  detected face boxes),
- person-detector score — HOG **and** Faster R-CNN (mean over detected
  person boxes),
- per-model embedding cosine similarity (face models on face crops, vision
  encoders on the whole image).

Rows are marked "not tested" when a detector or model is unavailable, and the
UI explains that results are model- and transformation-dependent. VLM
semantic evaluation is not enabled in this configuration (a local vision-
language model is not bundled); the UI states this explicitly.

## Frontend

- `App.tsx` is the state machine: `landing → uploading → analyzing →
  protecting → testing → result → cleanup complete`.
- `services/api.ts` — SSE client that renders real pipeline stages.
- `services/cleanup.ts` — the client-side cleanup manager: revokes object
  URLs, releases blobs and buffers, resets state after copy/download/start-new.
- `components/DifferenceCanvas.tsx` — draws `|protected − original|` on a
  canvas with an amplification slider (1×–100×) done in the browser, so the
  protected image itself is never modified.
- `components/DemoPanel.tsx` — developer-only panel (disabled by default) that
  shows raw model-by-model and transform-by-transform numbers.
- `utils/imageValidation.ts` — client-side validation mirroring the backend's
  (magic bytes, MIME, size).

## Security notes

- No shell execution, no path traversal (session ids are generated server-side
  and validated), no image logging, no analytics with image data.
- Uploads are validated by content (magic bytes), never by extension alone.
- CORS restricted to the local dev origins by default.
- Errors shown to the user are friendly; technical details only reach logs in
  debug mode.

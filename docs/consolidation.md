# Final Consolidation Build

This document records the consolidation pass that turned the project from an
experimental collection of protection stages and benchmark tooling into ONE
clean production pipeline.

## 1. Final production architecture

```
                    USER PHOTO
                        │
                        ▼
               Privacy Analyzer (faces, persons, QR/OCR/PII, metadata)
                        │
                        ▼
             Unified Protection Engine  (app/editing/protector.py)
                        │
      ONE perturbation δ = PGD(diffusion + identity + vision + transforms)
                        │
                        ▼
                Sensitive-region treatment
                        │
                        ▼
                Quality validation (SSIM/PSNR floor)
                        │
                        ▼
              Metadata sanitization + C2PA provenance
                        │
                        ▼
                COPY / DOWNLOAD  →  TEMP DATA CLEAR
```

There is exactly **one protection stage**. `protected = original + δ`, where
δ is a single perturbation optimized in one PGD loop against:

| Objective | Model | Attack families |
|---|---|---|
| Diffusion denoising error ↑ | SD1.5 U-Net surrogate | A diffusion editing, B inpainting, C instruction editing, D img2img |
| Identity embedding distance ↑ | FaceNet VGGFace2 | E identity-reference, F face-swap |
| Global vision feature distance ↑ | MobileNetV3-Large | I vision encoders |
| Transform robustness | differentiable resize/brightness/contrast/crop/JPEG-approx/noise variants | all of the above after re-encoding |

Nothing is stacked on top of the perturbation. The previous two-stage design
(face-protection pass **plus** editing pass) stacked independent perturbations
and measurably degraded quality (production SSIM ~0.94 vs ~0.98 for the same
algorithm alone); the identity objective is now part of the one PGD.

## 2. Files deleted / moved / added

**Moved (kept on disk, out of the app):**

- `backend/app/adversarial/` → `backend/legacy/adversarial/` — the old
  stacked face-protection engine (white-box + black-box detection suppression).
  No script or module in the app imports it anymore. Its shared box/crop
  helpers were extracted to `backend/app/utils/boxes.py` for the perception /
  robustness evaluation modules.
- `scripts/benchmark.py`, `scripts/benchmark_full.py`,
  `scripts/benchmark_editing.py` → `scripts/legacy/` — superseded by
  `scripts/benchmark_protection.py` (multi-family protection + adaptive
  red-team + resumable per-image cache).

**Deleted:**

- `tests/backend/test_adversarial.py` (its subject is the legacy module).
  Coverage for the consolidated path lives in the rewritten
  `tests/backend/test_pipeline.py` (unified stage, quality, provenance,
  cleanup).

**Added:**

- `backend/app/metadata/provenance.py` — C2PA provenance layer.
- `backend/app/utils/boxes.py` — shared box/crop helpers.
- `tests/backend/test_provenance.py` — C2PA + profile-flag tests.
- `docs/consolidation.md` — this document.

**Rewritten:**

- `backend/app/processing/pipeline.py` — one unified protection stage;
  perception/robustness gated to the research profile; C2PA at finalize;
  honest protection summary.
- `frontend/src/components/ResultView.tsx` + `ProtectionReport.tsx` — the
  production UI is now Upload → Protect → Original|Protected|Difference →
  Copy/Download, with all technical material collapsed behind a single
  "Technical details" toggle.

## 3. External research reused — methods and licenses

The system implements the *published methodologies* of the cited research
lines with our own differentiable implementations (no external repository
code was vendored into the app). Upstream references:

| Research | Method used here | Upstream repo | License |
|---|---|---|---|
| PhotoGuard | end-to-end anti-diffusion PGD (denoising-error maximization) | github.com/madrylab/photoguard | MIT |
| DiffusionGuard | mask-robust protection over real region masks; transform robustness | authors' repo (DiffusionGuard) | verify before reuse |
| EditShield | instruction-edit disruption via held-out editor evaluation | authors' repo (EditShield) | verify before reuse |
| ID-Eraser | identity-space perturbation for face-swap defense (family E/F objective) | authors' repo (ID-Eraser) | verify before reuse |
| Phantom | identity-shifted targets + face-region masking for face-swap defense | authors' repo (Phantom) | verify before reuse |
| C2PA | provenance manifest (SDK, not model code) | github.com/contentauth/c2pa-rs | Apache-2.0 / MIT |

The face-swap / identity-reference defense is implemented as the
identity-space perturbation idea of ID-Eraser/Phantom inside the unified PGD:

- **Dual encoder identity term** — FaceNet VGGFace2 + FaceNet CASIA both push
  the face embedding away from the original identity, so a face-swap or
  reference encoder that relies on either embedding family sees a moved
  identity.
- **Transform-robust identity** — the identity loss is evaluated on
  differentiable JPEG-approx / resize / brightness variants of the protected
  image, so the identity shift survives re-encoding.
- **Phantom-style spatial constraint** — an elliptical identity-region mask
  concentrates perturbation on identity-relevant facial area (soft edges, no
  visible boundaries) instead of a raw rectangle; a weaker global component
  covers hair/body/background.
- **ID-Eraser-style in-place refinement** — after the PGD, a small zeroth-order
  refinement drives the ArcFace (w600k_r50) embedding of the SAME δ further
  from the original identity, mimicking ID-Eraser's identity-space step
  without a second independent perturbation. The refinement is a continuation
  of the one optimizer, not a stacked pass.

Full ID-Eraser/Phantom reproduction (identity-shifted reconstruction
networks) is not deployed because those pipelines need additional generative
models beyond this machine's budget; the family is honestly reported as
covered by the identity-space objective plus a REAL face-swap benchmark
(`scripts/benchmark_face_swap.py` — INSwapper) and reference-generation
benchmark (`scripts/benchmark_reference_gen.py` — IP-Adapter FaceID SD1.5)
that measure actual identity-transfer reduction, not just embedding deltas.
Image-to-video (family G) remains `NOT TESTED` — a modular adapter is
registered, no heavy weights are installed, and nothing is faked.

## 5b. Research-only face-swap / reference evaluation

The following are research-only (`requirements-dev.txt` + downloaded models;
the production app never loads them):

| Component | Model | Role | License |
|---|---|---|---|
| Face swap | INSwapper (`inswapper_128.onnx`) | real face-swap attack (evaluation) | insightface; contact for redistribution |
| Face detection + recognition | insightface `buffalo_l` (scrfd_10g_bnkps + w600k_r50) | swap/reference identity metric | MIT for code; model weights research use |
| Reference generation | IP-Adapter FaceID SD1.5 (`h94/IP-Adapter-FaceID`) | identity/reference-conditioned generation (evaluation) | non-commercial (research use) |

`scripts/download_face_swap_models.py` fetches the weights; the benchmarks
cache protected outputs in `results/protected/` so the face-swap and
reference benchmarks share one protection pass.

## 4. Dependency audit

Production `backend/requirements.txt` now contains only what the unified
engine needs:

- **Kept:** fastapi/uvicorn/SSE, numpy, Pillow, opencv, torch/torchvision,
  facenet-pytorch (FaceNet identity term + MTCNN), diffusers/transformers/
  accelerate/hf_transfer (SD1.5 surrogate), onnxruntime + rapidocr (sensitive
  OCR), **c2pa-python + cryptography (C2PA provenance)**.
- **Shared with research:** every production dependency is also used by the
  research benchmark; there were no pure-research packages to remove.
- **Dev-only** (`requirements-dev.txt`): pytest, httpx, qrcode.

The old recognition stage pulled in the same core deps, so nothing was a
dedicated legacy dependency — the consolidation win is *model loading*, not
package removal: production now loads only FaceNet-VGGFace2 + MobileNetV3 +
the SD1.5 surrogate; CASIA / ArcFace / ResNet50 load only in the research
profile (`configs/research.yaml`).

## 5. C2PA status

**Implemented and tested.** `app/metadata/provenance.py` embeds a
self-signed C2PA manifest (claim generator "AI Privacy Shield", operation
`c2pa.placed`, content SHA-256, timestamp) into the protected PNG using
c2pa-python / c2pa-rs. A local Ed25519 keypair + C2PA-profile-compliant leaf
certificate is generated on first use (`backend/data/c2pa/`); real
deployments point `AIPS_C2PA_KEY` / `AIPS_C2PA_CERT` at their own signing key.

Honest caveats, per the spec:

- C2PA is **not** an AI blocker and is explicitly documented as a second
  defense layer. If a platform strips C2PA metadata, the adversarial
  perturbation remains the primary protection.
- The default self-signed key proves internal consistency (the file was
  produced by this app and not tampered with since), not identity trust. For
  verifiable-issuer provenance, supply your own signing certificate.
- If the dependency is missing or signing fails, the original bytes are
  returned untouched with an honest status — the user's image is never
  silently dropped.

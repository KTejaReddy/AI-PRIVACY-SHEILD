# AI Privacy Shield

> **Imperceptible protection that reduces unauthorized AI image editing and
> synthetic-media generation while keeping your photo visually normal.**

AI Privacy Shield is a **local-first**, privacy-focused image protection
platform. It uses adversarial machine learning to produce **one protected
photo per upload** — a single imperceptible perturbation optimized
simultaneously against several *AI attack families*: diffusion-based editing,
masked inpainting, instruction-guided editing, image-to-image generation,
identity/reference-conditioned generation, face-swap identity transfer, and
general vision encoders. The system combines multi-family adversarial
optimization (see `docs/attack-families.md`), transformation robustness
testing, sensitive-region analysis, metadata sanitization, and automatic
temporary-data cleanup. Users compare the original, protected, and amplified
difference images and copy or download the protected result.

**Positioning — read this first.** This project does **not** claim to make
images "AI-proof", "100% unrecognizable", or safe against every recognition
system on Earth. Its claim is narrow and honest:

> **Maximum practical resistance to the tested AI manipulation families.**
> Protection is evaluated against representative local models from each attack
> family and a fixed set of transformations. Effectiveness depends on the
> tested threat model — never a claim that "no AI can edit this image".

---

## 1. Problem statement

Automated facial recognition, image-embedding systems, and **AI image editors**
are cheap to run and increasingly ubiquitous. Once a photo with your face,
location metadata, or a readable QR/ID enters a database, it can be matched,
cross-referenced, tracked — and edited by generative models — often without
your knowledge. Blurring or pixelating destroys the photo; doing nothing
surrenders it. There is a middle path: alter the image *just enough* that a
human still sees the same photograph while the **tested AI systems** —
recognition models **and image editors** — can no longer use it reliably.

## 2. Proposed solution

A local-first web app with a real ML pipeline:

1. Upload a photo (never stored permanently, never leaves your machine).
2. Detect faces, persons, and other sensitive content (QR codes, text/PII).
3. Run the **unified protection engine**: ONE Photoguard-style perturbation is
   optimized in a single PGD loop against the anti-diffusion surrogate
   (families A–D: diffusion editing, inpainting, instruction editing,
   img2img), the FaceNet identity encoder on the detected faces (families
   E–F: identity-reference generation, face-swap) and the MobileNetV3 vision
   encoder (family I). See `docs/attack-families.md`. This is the only
   protection stage — nothing is stacked on top of it.
4. Measure the result on the real protected image: the **AI-editing
   benchmark** (original vs protected through InstructPix2Pix — held out from
   optimization — plus SD1.5 inpainting and img2img, with task-specific +
   CLIP edit-success), the **AI Perception Test**, transformation robustness,
   and visual quality.
5. Sanitize metadata, blur flagged sensitive regions, embed an optional C2PA
   provenance manifest, and hand back a clean output the user can copy or
   download.
6. Immediately clear all application-owned temporary image data.

## 3. How adversarial protection works

```
protected = original + one_multi_family_perturbation
```

**Unified protection engine** (multi-family, Photoguard-style):

The perturbation is the solution of a constrained optimization problem, not
random noise:

- **One perturbation, several families:** a single PGD loop maximizes the
  SD1.5 U-Net's **denoising reconstruction error** on the protected photo — a
  diffusion editor conditions on the photo through the same VAE/U-Net
  machinery, so a higher denoising error means a worse reconstruction of the
  protected subject — while simultaneously pushing the protected face's
  **identity embedding** (FaceNet, on the detected face boxes) and the image's
  **global representation** (MobileNetV3) away from the originals, reducing
  the photo's usefulness as a reference identity for generation/face-swap
  pipelines. The same perturbation is evaluated against every transform
  variant (resize/brightness/contrast/crop/JPEG-approx/noise) so it survives
  common re-encodings.
- Each PGD step applies a **high-frequency projection** (the raw gradient
  through the VAE encoder is smooth and visible), so the perturbation stays
  imperceptible (SSIM ≈ 0.98–0.99) while still moving the loss.
- Bounded at ≈ 4.5/255 with a hard SSIM floor; the stage is reverted if it
  becomes visible.
- **No stacked stages.** Earlier builds applied a separate face-protection
  pass *on top of* the editing perturbation, which degraded quality (SSIM
  dropped to ~0.94). The identity objective is part of the one PGD, so
  production quality now matches the benchmark (~0.98). The old recognition
  stage lives in `backend/legacy/adversarial/` for reference.
- **Multi-editor evaluation:** the benchmark runs **three** local editors —
  InstructPix2Pix (**held out** from optimization), SD1.5 masked inpainting
  (real region masks from the detected face/person boxes) and SD1.5
  image-to-image — on the original and the protected photo with identical
  instruction, seed, resolution, steps and guidance. Edit success is scored
  primarily by a **task-specific pixel metric** (redness in the shirt region,
  background-region change, saturation drop for sketches, …) with CLIP as an
  auxiliary check. Absolute change is always reported; relative percentages
  only when meaningful (no inflated "112%" numbers).

Every displayed number — embedding distances, SSIM, PSNR, robustness
PASS/PARTIAL/FAIL, editing-success reduction — is computed, not fabricated.
Embedding distances are L2 on normalized vectors, so they live in [0, 2].
See `docs/editing-protection.md` for the full methodology.

## 4. Features

- **AI Attack Family Registry** (`backend/app/attack_registry.py`): models are
  grouped by manipulation mechanism (diffusion editing, inpainting, instruction
  editing, img2img, identity-reference, face-swap, image-to-video, VLM,
  vision encoders) with optimization/evaluation/held-out roles
- **Production vs research separation**: the user app runs the protection
  engine only (`backend/configs/production.yaml`); benchmark models and the
  adaptive red-team live in research mode (`research.yaml`,
  `scripts/benchmark_protection.py`) — see `docs/production-vs-research.md`
- Real face detection (0 / 1 / many faces) with **two independent detector
  families** (OpenCV SSD + MTCNN cascade) driving the identity objective
- **Unified protection (the only stage):** one Photoguard-style PGD that
  targets diffusion editing, inpainting, instruction editing, img2img,
  identity-reference/face-swap and vision encoders at once
- Transformation-aware optimization (differentiable variants every iteration)
  plus robustness testing with real measurements
- **AI-editing benchmark (research profile):** original vs protected through
  all three local editors with task-specific success metrics, seed/prompt/mask
  robustness in `scripts/benchmark_protection.py`, and **edit success measured
  after real transformations** (JPEG/resize/…) on the protected image
- **AI Perception Test** panel: real before/after face-confidence (SSD +
  MTCNN), person-confidence (HOG + Faster R-CNN), and per-model
  embedding-similarity numbers, including a **held-out** ResNet50 encoder
  (research profile only)
- **C2PA provenance:** optional cryptographically signed manifest bound to the
  protected file (a second, non-AI-blocking layer — see `docs/privacy.md`)
- Amplified difference visualization with a 1×–100× slider (browser-side,
  never modifies the protected image)
- Sensitive-region analysis: QR/barcodes, experimental local OCR with PII
  pattern detection, blur treatment
- Metadata sanitization (EXIF/GPS/XMP/IPTC removed from every output)
- Copy-to-clipboard and download with automatic app-side cleanup
- GPU (CUDA) detection with automatic CPU fallback
- Developer demo panel (hidden by default) with raw per-model / per-transform
  numbers
- Local-first: no cloud AI, no paid APIs, no image database

## 5. Technology stack

| Layer | Technology |
| --- | --- |
| Frontend | React 18 + TypeScript + Vite, canvas-based difference rendering, SSE |
| Backend | Python 3.10–3.12, FastAPI, uvicorn, SSE (sse-starlette) |
| ML | PyTorch (CUDA/CPU), facenet-pytorch, torchvision (Faster R-CNN, MobileNetV3, ResNet50), ONNX Runtime + InsightFace ArcFace, **diffusers (SD1.5 anti-diffusion surrogate + InstructPix2Pix editor), transformers (CLIP ViT-L/14 scorer)**, accelerate (CPU offload) |
| Vision | OpenCV DNN face detection, MTCNN, HOG, Faster R-CNN, QR/barcode decoding, RapidOCR (optional) |
| Imaging | Pillow, NumPy |
| Tests | pytest (backend), Vitest + Testing Library (frontend) |

## 6. Project structure

```
ai-privacy-shield/
├── frontend/                 React + Vite UI
│   ├── src/
│   │   ├── components/       upload zone, progress, result, report, demo panel…
│   │   ├── services/         SSE api client, client-side cleanup manager
│   │   ├── utils/            validation, difference-map math
│   │   └── styles.css
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/              FastAPI routes (upload/process/cleanup/health)
│   │   ├── editing/          the unified protection engine: tasks, adapters, anti-diffusion surrogate, multi-family PGD protector
│   │   ├── attack_registry.py  AI attack-family registry + profile YAMLs
│   │   ├── evaluation/       AI perception test + AI-editing benchmark (before/after measurements)
│   │   ├── vision/           face + person detection, sensitive content
│   │   ├── privacy/          analyzer (orchestrates analysis)
│   │   ├── robustness/       transformation + model verification
│   │   ├── metadata/         metadata sanitization
│   │   ├── quality/          PSNR / SSIM / perturbation metrics
│   │   ├── cleanup/          session store + janitor
│   │   ├── models/           model registry (FaceNet, ArcFace)
│   │   ├── processing/       pipeline orchestrator
│   │   └── config.py
│   ├── scripts/              model download, environment check
│   └── requirements.txt
├── models/                   local model weights (see models/README.md)
├── tests/backend/            pytest suite
├── docs/                     architecture, setup, privacy, limitations
├── scripts/                  setup / run scripts (.sh + .ps1)
└── .env.example
```

## 7. Installation

**Prerequisites:** Python 3.10–3.12, Node.js 18+, ~4 GB disk. GPU optional.

```bash
# One-shot (Git Bash / Linux / macOS)
bash scripts/setup.sh

# or native Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

### Manual

```bash
# backend
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# models
python scripts/download_models.py
python scripts/check_environment.py

# frontend
cd ../frontend
npm install
```

### CUDA (NVIDIA GPU)

On Windows/Linux, `pip install torch` normally resolves to the CUDA wheel. If
you end up on CPU-only torch, force it:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The backend auto-detects CUDA and reports "GPU acceleration: available";
without CUDA it runs in "CPU mode: active" with reduced-but-functional
settings. It never crashes for lack of a GPU.

## 8. Running

```bash
# terminal 1 — backend (http://127.0.0.1:8000)
bash scripts/run_backend.sh          # or: powershell -File scripts/run_backend.ps1

# terminal 2 — frontend (http://localhost:5173)
bash scripts/run_frontend.sh         # or: powershell -File scripts/run_frontend.ps1
```

Open **http://localhost:5173**, upload a photo, and watch the real pipeline
stages: analyze → faces → protect → test → sanitize → result. Compare
ORIGINAL | PROTECTED | DIFFERENCE, drag the amplification slider, review the
protection report, then **Copy** or **Download**. The app clears its
temporary image data afterwards.

Set `AIPS_PROFILE=research` (in `backend/.env`) to enable the in-app
3-editor benchmark and red-team machinery; the default `production` profile
keeps user processing fast and benchmark models unloaded.

## 9. Testing

```bash
# backend
cd backend && .venv/Scripts/python.exe -m pytest        # or: source .venv/bin/activate && pytest

# frontend
cd frontend && npm test

# typecheck + production build
cd frontend && npm run build

# research benchmark (multi-family protection + adaptive red-team) — requires
# the local editing models (download once: python scripts/download_editing_models.py)
# resumable per-image cache: --resume skips already-completed images
python scripts/benchmark_protection.py --rounds 2 --seeds 42 --out results/
python scripts/benchmark_protection.py --resume   # finish a partial run
# options: --seeds 42,7,1337 --transformations jpeg_compression,resize,crop \
#          --tasks t01_shirt_color,t05_sketch --out results/

# real face-swap benchmark (INSwapper) + reference-generation benchmark
# (IP-Adapter FaceID SD1.5) — research-only models, download once:
python scripts/download_face_swap_models.py
python scripts/benchmark_face_swap.py --transforms jpeg_compression,resize
python scripts/benchmark_reference_gen.py
```

`scripts/benchmark_protection.py` runs the multi-family protection over an
image set and reports per-image and aggregate edit-success reduction, visual
quality, per-family transfer (identity/vision), held-out editors and
transformation robustness — the honest research evaluation of the unified
engine.

- Fast backend tests: upload validation, metrics, face detector, sensitive
  content, metadata. (≈10 s)
- ML backend tests: real optimization runs (bounded perturbation, valid
  output, quality floors, embedding disruption) and the full pipeline
  end-to-end including temp cleanup. These take minutes on first run.
- Frontend tests: upload validation, difference-map math, cleanup manager,
  result rendering with copy/download behavior.

## 10. Configuration

Copy `.env.example` → `backend/.env`. All knobs are optional. Key ones:
`AIPS_DEVICE` (auto/cuda/cpu), `AIPS_PROFILE` (production/research),
`AIPS_EPSILON` (perturbation bound), `AIPS_ITERATIONS_GPU/CPU`,
`AIPS_OCR_ENABLED`, `AIPS_OUTPUT_FORMAT` (png/jpeg),
`AIPS_SESSION_TTL_SECONDS`. See `.env.example` for the full list and
`docs/setup.md` for details.

## 11. Privacy architecture

- **Local-first:** the browser talks only to the local backend. No image data
  ever leaves the machine.
- **No image database:** temporary files live under `backend/.tmp/sessions/`
  with random ids and are deleted when processing finishes, by explicit
  cleanup, or by a janitor (30-min TTL) if the client crashes.
- **Copy:** after the image reaches the OS clipboard, the app clears its own
  state — it claims only "Application-owned temporary image data is cleared
  after copy", never that the clipboard is emptied.
- **Download:** after the file is on the device, the app revokes object URLs
  and clears its buffers — "Temporary copies held by this application are
  automatically cleared after download."
- **No logging of image contents**, no analytics, no third-party API calls.
- See `docs/privacy.md` for the full data lifecycle table.

## 12. Models

| Model | Role | Differentiable | License |
| --- | --- | --- | --- |
| OpenCV DNN face detector (res10 SSD) | face detection + perception test | n/a | BSD (OpenCV) |
| MTCNN cascade (P/R/O-Net) | face detection + differentiable surrogate | P/R/O-Net: yes | MIT (facenet-pytorch) |
| OpenCV HOG + SVM person detector | person detection + perception test | n/a | BSD (OpenCV) |
| Faster R-CNN ResNet-50 (COCO) | person detection + perception test | n/a | BSD (torchvision) |
| FaceNet Inception-ResNet v1 — VGGFace2 | optimization surrogate | yes | research |
| FaceNet Inception-ResNet v1 — CASIA | optimization surrogate | yes | research |
| InsightFace ArcFace (w600k_mbf) | verification + black-box refinement | no (ONNX) | non-commercial research |
| MobileNetV3-Large (ImageNet) | vision-feature optimization surrogate | yes | BSD (torchvision) |
| ResNet50 (ImageNet) | **held-out** vision encoder (transferability) | n/a | BSD (torchvision) |
| **Stable Diffusion v1.5** (U-Net + VAE + text encoder) | **anti-diffusion editing surrogate** (primary objective) | yes | CreativeML OpenRAIL-M |
| **InstructPix2Pix** | **held-out editing benchmark** (never optimized against) | n/a (eval) | Apache 2.0 / CC-BY-NC (weights: research) |
| **CLIP ViT-L/14** | edit-success scorer | n/a (eval) | MIT |

Details, download commands, and license notes: `models/README.md`. The
`/api/health` endpoint and the UI report exactly which models are installed
and tested. If ArcFace is absent, verification uses the remaining models and
the report says so.

## 13. API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | service, hardware (cuda/cpu), installed models, OCR status |
| `GET /api/models` | model registry detail |
| `POST /api/upload` | validate + stage an upload → `session_id` |
| `GET /api/process/{id}` | SSE stream of pipeline stages + final result |
| `POST /api/cleanup/{id}` | idempotent temp cleanup |

Uploads are validated by content (magic bytes), not extension; oversized and
malformed files are rejected with friendly errors.

## 14. Difference visualization

`difference = |protected − original|`, amplified by a 1×–100× slider and drawn
on a canvas in the browser. The amplification is **display only** — the
protected image you copy/download is untouched by it.

## 15. Data lifecycle summary

| What | Where | When it goes away |
| --- | --- | --- |
| Original + protected preview | browser state | copy / download / start-new / failure |
| Object URLs, blobs, buffers | browser | `cleanup.ts` after transfer |
| Upload bytes + intermediates | `backend/.tmp/sessions/<id>/` | pipeline end / cleanup endpoint / janitor |
| Model tensors | GPU/CPU memory | per-stage |
| Anything else | — | never created |

## 16. Known limitations

- Protection is evaluated against the tested surrogates and transformations
  only; other recognition systems may be less affected. No system can
  guarantee against every AI.
- The L∞ bound keeps images natural but caps how much disruption is possible.
- Person detection uses OpenCV HOG — a full-body detector that often misses
  portraits, seated, or occluded people; the report says what it found.
- VLM semantic protection is not enabled in this configuration; the UI states
  this explicitly.
- OCR/PII detection is experimental (can miss or misread text).
- CPU mode uses reduced iterations (weaker, faster).
- See `docs/limitations.md` for the full threat model.

## 17. Threat model

Defends against automated matching by recognition models similar to the
tested surrogates and against common re-compression/resizing. Out of scope:
adaptive attackers, physical re-photography, human recognition, and
obliterating the photo's context (clothing, tattoos, background).

## 18. Ethical / legal

Protect only content you are entitled to process. Protecting someone else's
photograph may require consent in your jurisdiction. This is a personal
privacy tool — not a way to defeat lawful identity verification. It also does
not anonymize context outside the face.

## 19. Screenshots

To be added — run the app and the result view shows the ORIGINAL / PROTECTED /
DIFFERENCE comparison, the protection report, and the copy/download actions.

## 20. Troubleshooting

See `docs/setup.md` — covers CUDA detection, model download failures, port
conflicts, and slow CPU processing.

---

## Demo / developer panel

Append `?demo=1` to the app URL (or set the `ai-privacy-shield-demo`
localStorage flag to `1`) to reveal the developer panel with raw per-model and
per-transform numbers, embedding distances, SSIM/PSNR, and perturbation
magnitudes. Hidden from the normal UI.

## License / disclaimer

This project is a research prototype. The ML components use weights with
research / non-commercial terms (see `models/README.md`); verify they fit your
use before shipping. No warranty of protection against any specific
recognition system is implied or provided.

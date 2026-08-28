# Setup

Requires **Python 3.10–3.12** and **Node.js 18+**. The backend works on CPU
with reduced-but-functional settings; an NVIDIA GPU with CUDA gives the full
iteration budget.

## One-shot setup

```bash
# Git Bash / macOS / Linux
bash scripts/setup.sh

# Native Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

This creates `backend/.venv`, installs backend and frontend dependencies,
downloads the models, and prints an environment report.

## Manual setup

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt      # tests only
```

> **CUDA note:** on Windows and Linux, plain `pip install torch` resolves to
> the CUDA-enabled wheel when a driver is present. If you get a CPU-only torch
> and you have an NVIDIA GPU, force the CUDA build:
>
> ```bash
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Models

```bash
cd backend
python scripts/download_models.py        # face detector + ArcFace + FaceNet cache
python scripts/check_environment.py      # verify what is installed
```

- The OpenCV face detector is required.
- ArcFace is optional; without it the verification phase uses the FaceNet
  models only, and the black-box refinement phase is skipped.
- FaceNet weights download automatically on first use (torch cache).

### 3. Frontend

```bash
cd frontend
npm install
```

## Run

```bash
# terminal 1 — backend
bash scripts/run_backend.sh              # or run_backend.ps1
# API at http://127.0.0.1:8000  ·  health: http://127.0.0.1:8000/api/health

# terminal 2 — frontend
bash scripts/run_frontend.sh             # or run_frontend.ps1
# UI at http://localhost:5173
```

## Tests

```bash
# backend (fast suite ~10 s, ML suite several minutes on first run)
cd backend && .venv/Scripts/python.exe -m pytest

# frontend
cd frontend && npm test

# typecheck + production build
cd frontend && npm run build
```

## Configuration

Copy `.env.example` to `backend/.env` and edit. Every value is optional;
defaults are production-safe. Notable ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AIPS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `AIPS_EPSILON` | `0.035` | L∞ perturbation bound (0–1 scale) |
| `AIPS_ITERATIONS_GPU` / `AIPS_ITERATIONS_CPU` | 70 / 35 | optimization budget |
| `AIPS_OCR_ENABLED` | `1` | local OCR for text/PII detection |
| `AIPS_OUTPUT_FORMAT` | `png` | `png` (lossless) or `jpeg` |
| `AIPS_SESSION_TTL_SECONDS` | 1800 | temp-session expiry |

## Troubleshooting

- **CUDA not detected** — run `python -c "import torch; print(torch.cuda.is_available())"`.
  If `False` with an NVIDIA GPU, reinstall torch from the cu121 index (above)
  and confirm `nvidia-smi` works.
- **Face detection downloads fail** — the OpenCV model URLs are occasionally
  slow; re-run `download_models.py`, it resumes.
- **Port 8000 / 5173 in use** — set `AIPS_PORT` / Vite port, or stop the other
  process. `backend/.tmp` sessions from a crashed run are swept by the janitor.
- **Slow CPU processing** — lower `AIPS_ITERATIONS_CPU` and `AIPS_EPSILON` for
  faster (weaker) protection; the UI shows "CPU mode: active".

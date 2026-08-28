#!/usr/bin/env bash
# Start the AI Privacy Shield local processing backend (FastAPI + uvicorn).
#
# Usage:  bash scripts/run_backend.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/backend"

# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

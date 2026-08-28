#!/usr/bin/env bash
# One-time setup for AI Privacy Shield.
# Creates the backend venv, installs backend + frontend dependencies, and
# downloads the local models.
#
# Usage:  bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/4] Backend virtual environment"
cd backend
if [ ! -d .venv ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -r requirements-dev.txt -q
cd "$ROOT"

echo "==> [2/4] Frontend dependencies"
cd frontend
npm install --no-audit --no-fund
cd "$ROOT"

echo "==> [3/4] Models"
cd backend
python scripts/download_models.py
cd "$ROOT"

echo "==> [4/4] Environment check"
cd backend
python scripts/check_environment.py || true
cd "$ROOT"

echo
echo "Setup complete. Start the backend and frontend with:"
echo "  bash scripts/run_backend.sh"
echo "  bash scripts/run_frontend.sh"

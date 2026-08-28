#!/usr/bin/env bash
# Start the AI Privacy Shield frontend (Vite dev server).
#
# Usage:  bash scripts/run_frontend.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/frontend"

exec npm run dev

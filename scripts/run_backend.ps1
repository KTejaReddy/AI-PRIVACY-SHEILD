# Start the AI Privacy Shield local processing backend (FastAPI + uvicorn).
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $Root "backend")
& ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# One-time setup for AI Privacy Shield (native Windows PowerShell).
# Creates the backend venv, installs backend + frontend dependencies, and
# downloads the local models.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "==> [1/4] Backend virtual environment"
Push-Location (Join-Path $Root "backend")
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
& ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt -q
Pop-Location

Write-Host "==> [2/4] Frontend dependencies"
Push-Location (Join-Path $Root "frontend")
npm install --no-audit --no-fund
Pop-Location

Write-Host "==> [3/4] Models"
Push-Location (Join-Path $Root "backend")
& ".venv\Scripts\python.exe" scripts\download_models.py
Pop-Location

Write-Host "==> [4/4] Environment check"
Push-Location (Join-Path $Root "backend")
& ".venv\Scripts\python.exe" scripts\check_environment.py
Pop-Location

Write-Host ""
Write-Host "Setup complete. Start the backend and frontend with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1"

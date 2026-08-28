# Start the AI Privacy Shield frontend (Vite dev server).
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $Root "frontend")
npm run dev

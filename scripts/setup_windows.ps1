$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "SYNTRA setup" -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11 or newer first."
}

py -3.11 --version
if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[dev]"

New-Item -ItemType Directory -Force -Path "data\raw", "data\processed", "checkpoints" | Out-Null

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "1. Add permitted source files to data\raw"
Write-Host "2. Run: .\.venv\Scripts\python.exe -m syntra.cli prepare"
Write-Host "3. Run: .\.venv\Scripts\python.exe -m syntra.cli train --config config\default.json"

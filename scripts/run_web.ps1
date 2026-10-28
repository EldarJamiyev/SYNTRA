$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Frontend = Join-Path $Root "frontend"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}
if (-not (Test-Path (Join-Path $Frontend "package.json"))) {
    throw "Frontend files are missing."
}

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$Root'; & '$Python' -m syntra.cli serve --host 127.0.0.1 --port 8000"
)

if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Push-Location $Frontend
    npm install
    Pop-Location
}

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$Frontend'; npm run dev -- --host 127.0.0.1"
)

Write-Host "Backend and frontend terminals launched. Open http://127.0.0.1:5173" -ForegroundColor Green

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js was not found. Install Node.js 20 or newer first."
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11 or newer was not found. Install Python before starting SYNTRA."
}
if (-not (Test-Path (Join-Path $Root "node_modules"))) {
    npm install
}
$TorchReady = python -c "import torch; print(torch.__version__)" 2>$null
if (-not $TorchReady) {
    Write-Warning "PyTorch is not installed in this Python environment. Run: python -m pip install -r requirements.txt"
    Write-Warning "The GUI and workspace service will start, but native model training/inference will remain offline."
} else {
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command", "Set-Location '$Root'; npm run ai:server"
    )
}
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command", "Set-Location '$Root'; npm run server"
)
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command", "Set-Location '$Root'; npm run dev"
)

Write-Host "SYNTRA services launched." -ForegroundColor Green
Write-Host "GUI: http://127.0.0.1:3000"
Write-Host "Local workspace API: http://127.0.0.1:3001/local-api/health"
Write-Host "Native AI API: http://127.0.0.1:3002/health (when PyTorch is installed)"

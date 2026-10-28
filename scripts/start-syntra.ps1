$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js was not found. Install Node.js 20 or newer first."
}
if (-not (Test-Path (Join-Path $Root "node_modules"))) {
    npm install
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

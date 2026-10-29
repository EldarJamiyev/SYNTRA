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

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = (Get-Command python).Source
}

# Inspect installed package metadata without importing torch. Importing torch can
# write a NumPy warning to stderr, which older PowerShell treats as a terminating error.
$RuntimeCheck = & $Python -c "import importlib.util; print('ready' if importlib.util.find_spec('torch') and importlib.util.find_spec('numpy') else 'missing')"
if ($LASTEXITCODE -ne 0 -or "$RuntimeCheck".Trim() -ne "ready") {
    Write-Warning "SYNTRA's local Python dependencies are missing."
    Write-Warning "Run 'npm run ai:install' and then start SYNTRA again."
} else {
    $PythonDirectory = Split-Path -Parent $Python
    $env:PATH = $PythonDirectory + ";" + $env:PATH
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
Write-Host "Native AI API: http://127.0.0.1:3002/health (when PyTorch and NumPy are installed)"

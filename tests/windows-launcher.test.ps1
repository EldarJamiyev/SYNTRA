$ErrorActionPreference = "Stop"

$launcher = Get-Content (Join-Path $PSScriptRoot "..\scripts\start-syntra.ps1") -Raw
if ($launcher -match "(?i)ollama|gemini|SYNTRA_MODEL_PROVIDER|SYNTRA_LOCAL_MODEL") {
    throw "The Windows launcher must not probe or configure external model providers."
}
if (-not $launcher.Contains("scripts\setup_syntra_env.py")) {
    throw "The launcher must prepare the isolated local PyTorch environment."
}
if (-not $launcher.Contains("checkpoints\syntra-brain.pt")) {
    throw "The launcher must check the standalone brain checkpoint."
}
if (-not $launcher.Contains("local PyTorch brain")) {
    throw "The launcher must identify the standalone local PyTorch brain."
}

Write-Host "Windows launcher local-brain startup behavior tests passed." -ForegroundColor Green

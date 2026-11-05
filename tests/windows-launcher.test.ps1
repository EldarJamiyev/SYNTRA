$ErrorActionPreference = "Stop"

$launcherPath = Join-Path $PSScriptRoot "..\scripts\start-syntra.ps1"
$launcher = Get-Content -LiteralPath $launcherPath -Raw

if ($launcher -match "(?i)SYNTRA_MODEL_PROVIDER|SYNTRA_LOCAL_MODEL|SYNTRA_EMBED_MODEL|SYNTRA_REVIEW_MODEL|11434|UsePretrainedModel|launcher-helpers") {
    throw "The Windows launcher must not contain alternate AI providers or model selection."
}
foreach ($required in @(
    "scripts\setup_syntra_env.py",
    "requirements.lock.txt",
    "syntra_ai.cli",
    "Preparing the isolated native AI environment",
    "Checking the local model",
    "data\checkpoints\syntra.pt",
    "Ensure-Service 3002",
    "Ensure-Service 3001",
    "Ensure-Service 3000"
)) {
    if (-not $launcher.Contains($required)) {
        throw "The Windows launcher is missing its native-only startup requirement: $required"
    }
}
if ($launcher -notmatch 'First-run local training is required') {
    throw "A clean installation must train SYNTRA's own checkpoint before chat starts."
}
if ($launcher -notmatch 'CHECKPOINT_OK') {
    throw "The launcher must validate the native checkpoint before starting chat."
}

Write-Host "Windows launcher native-only startup checks passed." -ForegroundColor Green

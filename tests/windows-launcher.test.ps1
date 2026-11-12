$ErrorActionPreference = "Stop"

$launcherPath = Join-Path $PSScriptRoot "..\scripts\start-syntra.ps1"
$launcher = Get-Content -LiteralPath $launcherPath -Raw

if ($launcher -match "(?i)SYNTRA_LOCAL_MODEL|SYNTRA_EMBED_MODEL|SYNTRA_REVIEW_MODEL|11434|UsePretrainedModel|launcher-helpers") {
    throw "The Windows launcher must not contain legacy model selectors or alternate-provider launcher helpers."
}
if ($launcher -notmatch "SYNTRA_MODEL_PROVIDER" -or $launcher -notmatch "openai-compatible") {
    throw "The Windows launcher must expose the explicit, opt-in hybrid provider mode."
}
if ($launcher -notmatch "\^20\.19\.0 or >=22\.12\.0") {
    throw "The Windows launcher must enforce the runtime range required by Vite 8."
}
foreach ($required in @(
    "scripts\setup_syntra_env.py",
    "local/model-provider.mjs",
    "openai-compatible",
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

Write-Host "Windows launcher native-first and hybrid startup checks passed." -ForegroundColor Green

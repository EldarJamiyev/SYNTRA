$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\scripts\launcher-helpers.ps1")

$catalog = @(
    [pscustomobject]@{ name = "qwen-coder:14b"; size = 500 },
    [pscustomobject]@{ name = "small-model:latest"; size = 200 },
    [pscustomobject]@{ name = "provider-cloud:latest"; size = 100 }
)

$automatic = Get-SyntraOllamaSelection -Models $catalog -RamBytes 1000
if ($automatic.Status -ne "available" -or $automatic.Models.Count -ne 2) {
    throw "Automatic selection must include installed local models that fit the conservative RAM threshold and exclude cloud entries."
}

$explicit = Get-SyntraOllamaSelection -Models $catalog -RequestedModel "qwen-coder:14b" -RamBytes 1000
if ($explicit.Status -ne "available" -or $explicit.Models.Count -ne 1 -or $explicit.Models[0].name -ne "qwen-coder:14b") {
    throw "An explicitly selected installed local model must be selected exactly."
}

$missing = Get-SyntraOllamaSelection -Models $catalog -RequestedModel "missing:latest" -RamBytes 1000
if ($missing.Status -ne "not_installed") {
    throw "A missing explicit model must be distinguished from an automatic model-selection fallback."
}

$tooLarge = Get-SyntraOllamaSelection -Models $catalog -RequestedModel "qwen-coder:14b" -RamBytes 500
if ($tooLarge.Status -ne "memory_limit") {
    throw "An explicitly selected model above the 70% RAM threshold must report a memory-limit status."
}

$cloudOnly = Get-SyntraOllamaSelection -Models @([pscustomobject]@{ name = "provider-cloud:latest"; size = 100 }) -RamBytes 1000
if ($cloudOnly.Status -ne "no_local_models") {
    throw "Cloud-only catalog entries must not be selected as local Ollama models."
}

Write-Host "Windows launcher model-selection behavior tests passed." -ForegroundColor Green

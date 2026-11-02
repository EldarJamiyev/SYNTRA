function Get-SyntraOllamaSelection {
    [CmdletBinding()]
    param(
        [AllowNull()]
        [object[]]$Models,
        [string]$RequestedModel = "",
        [double]$RamBytes = 0
    )

    $requested = if ($RequestedModel) { $RequestedModel.Trim() } else { "" }
    $localModels = @($Models | Where-Object {
        $_ -and $_.name -and ([string]$_.name -notmatch '(?i)(?:^|[:._-])cloud(?:$|[:._-])')
    })

    $selected = $localModels
    if ($requested) {
        $selected = @($localModels | Where-Object { [string]$_.name -eq $requested })
        if ($selected.Count -eq 0) {
            return [pscustomobject]@{ Status = "not_installed"; Models = @() }
        }
    }

    if ($RamBytes -gt 0) {
        $withinMemory = @($selected | Where-Object {
            $size = $_.size
            if ($null -eq $size -or [string]::IsNullOrWhiteSpace([string]$size)) {
                return $true
            }
            try {
                return ([double]$size -le ($RamBytes * 0.70))
            } catch {
                # Unknown size metadata is not treated as proof that a model is too large.
                return $true
            }
        })
        if ($requested -and $withinMemory.Count -eq 0) {
            return [pscustomobject]@{ Status = "memory_limit"; Models = @() }
        }
        $selected = $withinMemory
    }

    if ($selected.Count -eq 0) {
        return [pscustomobject]@{ Status = "no_local_models"; Models = @() }
    }
    return [pscustomobject]@{ Status = "available"; Models = @($selected) }
}

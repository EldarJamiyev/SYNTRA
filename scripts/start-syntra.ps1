$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Endpoint([string]$Uri, [string]$ExpectedContent = "") {
    try {
        $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -ne 200) { return $false }
        if ($ExpectedContent -and -not ([string]$response.Content).Contains($ExpectedContent)) { return $false }
        return $true
    } catch {
        return $false
    }
}

function Get-EndpointDiagnostic([string]$Uri) {
    try {
        $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 3 -UseBasicParsing
        $body = [string]$response.Content
        if ($body.Length -gt 240) { $body = $body.Substring(0, 240) + "..." }
        return "Last response: HTTP $([int]$response.StatusCode). Body: $body"
    } catch {
        $status = "no HTTP response"
        $body = [string]$_.Exception.Message
        try {
            if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
                $status = "HTTP $([int]$_.Exception.Response.StatusCode)"
            }
        } catch {}
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            $body = [string]$_.ErrorDetails.Message
        }
        $body = $body -replace '[\r\n]+', ' '
        if ($body.Length -gt 240) { $body = $body.Substring(0, 240) + "..." }
        return "Last response: $status. Detail: $body"
    }
}

function Wait-Endpoint([string]$Uri, [int]$TimeoutSeconds = 45, [string]$ExpectedContent = "") {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-Endpoint $Uri $ExpectedContent) { return }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)

    $diagnostic = Get-EndpointDiagnostic $Uri
    $listenerDetails = @()
    try {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort ([uri]$Uri).Port -ErrorAction SilentlyContinue)
        foreach ($listener in $listeners) {
            try {
                $processInfo = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$listener.OwningProcess) -ErrorAction Stop
                $listenerDetails += "PID $($listener.OwningProcess): $($processInfo.Name) $($processInfo.CommandLine)"
            } catch {
                $listenerDetails += "PID $($listener.OwningProcess): process details unavailable"
            }
        }
    } catch {}
    $owner = if ($listenerDetails.Count) { " Port owner: " + ($listenerDetails -join " | ") } else { " No listening process was found on the target port." }
    $marker = if ($ExpectedContent) { " Expected page marker: '$ExpectedContent'." } else { "" }
    throw "Service did not become ready at $Uri.$marker $diagnostic$owner Check the Vite service window for its startup error."
}

function Ensure-Service([int]$Port, [string]$HealthUrl, [string]$NpmScript, [string]$ExpectedContent = "") {
    if (Test-Endpoint $HealthUrl $ExpectedContent) {
        Write-Host "Already running: $NpmScript ($HealthUrl)" -ForegroundColor Green
        return
    }

    $listeners = @()
    try {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    } catch {
        $listeners = @()
    }
    if ($listeners.Count -gt 0) {
        $restartedStaleVite = $false
        foreach ($listener in $listeners) {
            try {
                $processInfo = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$listener.OwningProcess) -ErrorAction Stop
                $commandLine = [string]$processInfo.CommandLine
                if ($commandLine -and $commandLine.Contains($Root) -and $commandLine -match '(?i)\\vite\\|vite\.js') {
                    Write-Host "Replacing the old SYNTRA frontend process so the current chat UI loads..." -ForegroundColor Yellow
                    Stop-Process -Id ([int]$listener.OwningProcess) -Force -ErrorAction Stop
                    $restartedStaleVite = $true
                }
            } catch {}
        }
        if ($restartedStaleVite) {
            $releaseDeadline = (Get-Date).AddSeconds(8)
            do {
                Start-Sleep -Milliseconds 350
                $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
            } while ($listeners.Count -gt 0 -and (Get-Date) -lt $releaseDeadline)
        }
        if ($listeners.Count -gt 0) {
            throw "Port $Port is already in use by a process that does not match the current SYNTRA app. Close the other application or server on port $Port, then double-click Start-SYNTRA.bat again."
        }
    }

    Write-Host "Starting $NpmScript ..."
    $command = if ($NpmScript -eq "dev") { "npm run dev -- --clearScreen false" } else { "npm run $NpmScript" }
    Start-Process -FilePath $env:ComSpec -ArgumentList @("/k", $command) -WorkingDirectory $Root
    if ($NpmScript -eq "dev") {
        Write-Host "The Vite service window will show the frontend startup log. The launcher checks for the SYNTRA Chat page before continuing."
    }
    Wait-Endpoint $HealthUrl 45 $ExpectedContent
    Write-Host "Ready: $NpmScript" -ForegroundColor Green
}

try {
    Write-Host "SYNTRA - one-click local startup" -ForegroundColor Green
    Write-Host "Project: $Root"

    Write-Step "Checking prerequisites"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw "Node.js is missing. Install Node.js 20 or newer from https://nodejs.org/ and double-click Start-SYNTRA.bat again."
    }
    $nodeVersionText = (& node --version 2>&1 | Out-String).Trim()
    $nodeMatch = [regex]::Match($nodeVersionText, '^v?(\d+)\.')
    if ($LASTEXITCODE -ne 0 -or -not $nodeMatch.Success) {
        throw "Could not read the Node.js version. Reinstall Node.js 20 or newer."
    }
    $nodeMajor = [int]$nodeMatch.Groups[1].Value
    if ($nodeMajor -lt 20) {
        throw "SYNTRA needs Node.js 20 or newer. Installed version: $nodeVersionText"
    }

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python is missing. Install Python 3.11 or newer from https://www.python.org/downloads/ and enable Add Python to PATH."
    }
    $pythonVersionText = (& python --version 2>&1 | Out-String).Trim()
    $pythonMatch = [regex]::Match($pythonVersionText, 'Python\s+(\d+)\.(\d+)')
    if ($LASTEXITCODE -ne 0 -or -not $pythonMatch.Success) {
        throw "Could not read the Python version. Install Python 3.11 or newer."
    }
    $pythonMajor = [int]$pythonMatch.Groups[1].Value
    $pythonMinor = [int]$pythonMatch.Groups[2].Value
    if (($pythonMajor -lt 3) -or ($pythonMajor -eq 3 -and $pythonMinor -lt 11)) {
        throw "SYNTRA needs Python 3.11 or newer. Installed version: $pythonVersionText"
    }
    Write-Host "Node.js: $nodeVersionText" -ForegroundColor Green
    Write-Host "$pythonVersionText" -ForegroundColor Green

    Write-Step "Preparing the web application"
    $ProjectLock = Join-Path $Root "package-lock.json"
    $InstalledLock = Join-Path $Root "node_modules\.package-lock.json"
    if (-not (Test-Path -LiteralPath $ProjectLock -PathType Leaf)) {
        throw "package-lock.json is missing. Run git pull in the SYNTRA repository folder before starting the application."
    }

    $requiredDependenciesMissing = (
        -not (Test-Path (Join-Path $Root "node_modules\vite\bin\vite.js")) -or
        -not (Test-Path (Join-Path $Root "node_modules\react\package.json")) -or
        -not (Test-Path (Join-Path $Root "node_modules\react-markdown\package.json")) -or
        -not (Test-Path (Join-Path $Root "node_modules\remark-gfm\package.json"))
    )
    $dependencyInputs = @(
        Get-Item -LiteralPath (Join-Path $Root "package.json")
        Get-Item -LiteralPath $ProjectLock
    )
    $latestDependencyInput = ($dependencyInputs | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc
    $lockNeedsRefresh = -not (Test-Path -LiteralPath $InstalledLock -PathType Leaf)
    if (-not $lockNeedsRefresh) {
        $lockNeedsRefresh = $latestDependencyInput -gt (Get-Item -LiteralPath $InstalledLock).LastWriteTimeUtc
    }

    if ($requiredDependenciesMissing -or $lockNeedsRefresh) {
        Write-Host "Installing the exact dependency tree from package-lock.json with npm ci..."
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed. Fix the npm error shown above and run Start-SYNTRA.bat again." }
    } else {
        Write-Host "JavaScript dependencies match the committed npm lockfile." -ForegroundColor Green
    }

    Write-Step "Preparing the standalone PyTorch brain environment"
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    Write-Host "Checking the Python environment against requirements.lock.txt..."
    & python (Join-Path $Root "scripts\setup_syntra_env.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Python setup or lock validation failed. Check the installation error above, then run Start-SYNTRA.bat again."
    }

    $runtimeCheck = & $VenvPython -c "import torch, numpy; assert torch.version.cuda is None, 'SYNTRA requires CPU-only PyTorch'; import brain; print('READY PyTorch=' + torch.__version__ + ' NumPy=' + numpy.__version__ + ' CUDA=' + str(torch.version.cuda))" 2>&1
    if ($LASTEXITCODE -ne 0 -or (($runtimeCheck | Out-String) -notmatch 'READY PyTorch=.*CUDA=None')) {
        $runtimeCheck | ForEach-Object { Write-Host $_ }
        throw "The local Python brain failed its environment health check. Run scripts/setup_syntra_env.py and inspect the error."
    }
    $runtimeCheck | ForEach-Object { Write-Host $_ -ForegroundColor Green }

    $BrainCheckpoint = Join-Path $Root "checkpoints\syntra-brain.pt"
    if (Test-Path -LiteralPath $BrainCheckpoint -PathType Leaf) {
        Write-Host "A standalone local PyTorch brain checkpoint is available." -ForegroundColor Green
    } else {
        Write-Warning "No trained local PyTorch brain checkpoint exists yet. Start SYNTRA, open Active workspace, and select 'Train local brain on this workspace'."
    }

    Write-Step "Starting the local services"
    Ensure-Service 3001 "http://127.0.0.1:3001/local-api/health" "server"
    Ensure-Service 3000 "http://127.0.0.1:3000/" "dev" "SYNTRA Chat"

    Write-Step "Opening SYNTRA"
    Start-Process "http://127.0.0.1:3000"
    Write-Host ""
    Write-Host "SYNTRA is ready at http://127.0.0.1:3000" -ForegroundColor Green
    Write-Host "Keep the service windows open while using SYNTRA."
    Write-Host "To launch it next time, double-click Start-SYNTRA.bat again."
} catch {
    Write-Host ""
    Write-Host "SYNTRA could not start: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Do not close this window before copying the error if you need help." -ForegroundColor Yellow
}
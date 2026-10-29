$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Endpoint([string]$Uri) {
    try {
        $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 2 -UseBasicParsing
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Wait-Endpoint([string]$Uri, [int]$TimeoutSeconds = 45) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-Endpoint $Uri) { return }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw "Service did not become ready at $Uri. Check its open service window for the actual error."
}

function Ensure-Service([int]$Port, [string]$HealthUrl, [string]$NpmScript) {
    if (Test-Endpoint $HealthUrl) {
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
        throw "Port $Port is already in use, but SYNTRA did not respond at $HealthUrl. Close the other app using port $Port, then double-click Start-SYNTRA.bat again."
    }

    Write-Host "Starting $NpmScript ..."
    Start-Process -FilePath $env:ComSpec -ArgumentList @("/k", "npm run $NpmScript") -WorkingDirectory $Root
    Wait-Endpoint $HealthUrl
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
    if (-not (Test-Path (Join-Path $Root "node_modules\vite\bin\vite.js")) -or -not (Test-Path (Join-Path $Root "node_modules\react\package.json"))) {
        npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed. Fix the npm error shown above and run Start-SYNTRA.bat again." }
    } else {
        Write-Host "JavaScript dependencies are already installed." -ForegroundColor Green
    }

    Write-Step "Preparing the isolated local AI environment"
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    $pythonReady = $false
    if (Test-Path $VenvPython) {
        $dependencyCheck = & $VenvPython -c "import importlib.util; print('READY' if importlib.util.find_spec('torch') and importlib.util.find_spec('numpy') else 'MISSING')" 2>&1
        $pythonReady = ($LASTEXITCODE -eq 0 -and (($dependencyCheck | Out-String).Trim() -eq "READY"))
    }

    if (-not $pythonReady) {
        Write-Host "Installing or repairing PyTorch and NumPy. This is needed only when dependencies are missing."
        & python (Join-Path $Root "scripts\setup_syntra_env.py")
        if ($LASTEXITCODE -ne 0) {
            throw "Python setup failed. Check the installation error above, then double-click Start-SYNTRA.bat again."
        }
    }

    $runtimeCheck = & $VenvPython -c "import torch, numpy; print('READY PyTorch=' + torch.__version__ + ' NumPy=' + numpy.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0 -or (($runtimeCheck | Out-String) -notmatch 'READY PyTorch=')) {
        $runtimeCheck | ForEach-Object { Write-Host $_ }
        throw "The local Python runtime still cannot import PyTorch and NumPy. Run npm run ai:install and inspect the error."
    }
    $runtimeCheck | ForEach-Object { Write-Host $_ -ForegroundColor Green }

    Write-Step "Checking the local model"
    $Checkpoint = Join-Path $Root "data\checkpoints\syntra.pt"
    $CheckpointMetadata = $Checkpoint + ".json"
    $checkpointValid = $false
    if (Test-Path $Checkpoint) {
        $validationOutput = & $VenvPython -c "from syntra_ai.runtime import _read_checkpoint; _read_checkpoint(); print('CHECKPOINT_OK')" 2>&1
        $checkpointValid = ($LASTEXITCODE -eq 0 -and (($validationOutput | Out-String) -match 'CHECKPOINT_OK'))
        if (-not $checkpointValid) {
            $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
            Write-Warning "The saved model checkpoint is invalid. It will be backed up before a clean rebuild."
            Move-Item -LiteralPath $Checkpoint -Destination ($Checkpoint + ".invalid-" + $stamp)
            if (Test-Path $CheckpointMetadata) {
                Move-Item -LiteralPath $CheckpointMetadata -Destination ($CheckpointMetadata + ".invalid-" + $stamp)
            }
        }
    }

    $targetSteps = 1000
    $targetContext = 1024
    $resumeTraining = $false
    $stepsToRun = $targetSteps

    if ($checkpointValid) {
        $statusOutput = & $VenvPython -m syntra_ai.cli status 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Could not read the existing model status. Checkpoint was preserved; see the error above."
        }
        $modelStatus = (($statusOutput | Out-String) | ConvertFrom-Json)
        $currentSteps = [int]$modelStatus.trained_steps
        $currentContext = $modelStatus.context_tokens
        if ((Test-Path $CheckpointMetadata) -and (($currentContext -eq 1024) -or ($currentContext -eq 256)) -and $currentSteps -lt $targetSteps) {
            $targetContext = [int]$currentContext
            $resumeTraining = $true
            $stepsToRun = $targetSteps - $currentSteps
            Write-Host "Resuming interrupted first-run training at step $currentSteps of $targetSteps." -ForegroundColor Yellow
        } else {
            Write-Host "Existing valid checkpoint found (saved steps: $currentSteps). Skipping first-run training." -ForegroundColor Green
        }
    }

    if (-not $checkpointValid -or $resumeTraining) {
        Write-Host ""
        Write-Host "First-run local training is required because SYNTRA ships without pretrained weights." -ForegroundColor Yellow
        Write-Host "Training uses source files in this repository. Progress appears below; checkpoints are saved periodically."
        $trainArgs = @("-m", "syntra_ai.cli", "train", "--root", $Root, "--seq-len", "$targetContext", "--steps", "$stepsToRun", "--batch-size", "2")
        if ($resumeTraining) { $trainArgs += "--resume" }

        $trainingOutput = @()
        & $VenvPython @trainArgs 2>&1 | Tee-Object -Variable trainingOutput
        $trainingExit = $LASTEXITCODE

        if ($trainingExit -ne 0 -and -not $resumeTraining -and (($trainingOutput | Out-String) -match '80/20 split needs more than')) {
            Write-Warning "The source corpus is too small for a 1,024-token context. Retrying with a smaller 256-token context."
            $trainingOutput = @()
            $fallbackArgs = @("-m", "syntra_ai.cli", "train", "--root", $Root, "--seq-len", "256", "--steps", "$targetSteps", "--batch-size", "2")
            & $VenvPython @fallbackArgs 2>&1 | Tee-Object -Variable trainingOutput
            $trainingExit = $LASTEXITCODE
        }

        if ($trainingExit -ne 0) {
            throw "Initial local training failed. The error is shown above. The launcher will keep any saved checkpoints; fix that error and double-click Start-SYNTRA.bat again."
        }
    }

    $validationOutput = & $VenvPython -c "from syntra_ai.runtime import _read_checkpoint; _read_checkpoint(); print('CHECKPOINT_OK')" 2>&1
    if ($LASTEXITCODE -ne 0 -or (($validationOutput | Out-String) -notmatch 'CHECKPOINT_OK')) {
        $validationOutput | ForEach-Object { Write-Host $_ }
        throw "SYNTRA could not validate the model checkpoint after setup."
    }

    Write-Step "Starting the local services"
    Ensure-Service 3002 "http://127.0.0.1:3002/health" "ai:server"
    Ensure-Service 3001 "http://127.0.0.1:3001/local-api/health" "server"
    Ensure-Service 3000 "http://127.0.0.1:3000/" "dev"

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
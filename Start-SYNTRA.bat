@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0scripts\start-syntra.ps1" (
  echo ERROR: scripts\start-syntra.ps1 was not found.
  echo Run git pull in your SYNTRA folder, then try again.
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0scripts\start-syntra.ps1"
endlocal

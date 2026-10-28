"""Create an isolated local Python environment for SYNTRA and install its runtime."""
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
if sys.version_info < (3, 11):
    raise SystemExit("SYNTRA requires Python 3.11 or newer; current interpreter is " + sys.version.split()[0])

if sys.platform == "win32":
    VENV_PYTHON = VENV / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV / "bin" / "python"

if not VENV_PYTHON.exists():
    print("Creating local environment at", VENV, flush=True)
    venv.EnvBuilder(with_pip=True, clear=VENV.exists()).create(VENV)

print("Installing SYNTRA Python requirements into", VENV, flush=True)
subprocess.run(
    [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
    cwd=ROOT,
    check=True,
)
print("SYNTRA local Python environment is ready.", flush=True)

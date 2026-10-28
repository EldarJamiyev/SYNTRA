"""Run a supported SYNTRA Python module using .venv when it exists."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MODULES = {"server": "syntra_ai.server", "cli": "syntra_ai.cli"}


def selected_python() -> Path:
    if sys.platform == "win32":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        print("Usage: python scripts/run_syntra_module.py server|cli [arguments...]", file=sys.stderr)
        return 2
    module = MODULES[sys.argv[1]]
    command = [str(selected_python()), "-m", module, *sys.argv[2:]]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

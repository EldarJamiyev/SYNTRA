"""Run a supported SYNTRA Python module using .venv when it exists."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MODULES = {"server": "syntra_ai.server", "cli": "syntra_ai.cli", "brain-train": "brain.train"}
SCRIPTS = {"benchmark-model": ROOT / "tests" / "benchmark_engine" / "runner.py"}


def selected_python() -> Path:
    if sys.platform == "win32":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_syntra_module.py server|cli|brain-train|benchmark-model [arguments...]", file=sys.stderr)
        return 2
    python = str(selected_python())
    if sys.argv[1] in MODULES:
        command = [python, "-m", MODULES[sys.argv[1]], *sys.argv[2:]]
    elif sys.argv[1] in SCRIPTS:
        command = [python, str(SCRIPTS[sys.argv[1]]), *sys.argv[2:]]
    else:
        print("Unknown SYNTRA Python command: " + sys.argv[1], file=sys.stderr)
        return 2
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

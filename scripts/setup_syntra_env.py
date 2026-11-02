"""Create and maintain SYNTRA's isolated, hash-locked CPU Python environment."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
LOCKFILE = ROOT / "requirements.lock.txt"
LOCK_STAMP = VENV / ".syntra-requirements-lock.sha256"

if sys.platform == "win32":
    VENV_PYTHON = VENV / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV / "bin" / "python"


def lock_fingerprint(lockfile: Path = LOCKFILE) -> str:
    """Return the SHA-256 digest used to bind a venv to the committed lock."""
    return hashlib.sha256(lockfile.read_bytes()).hexdigest()


def environment_is_current(
    python_executable: Path,
    stamp_file: Path,
    expected_fingerprint: str,
    run: Callable = subprocess.run,
) -> bool:
    """Check the lock fingerprint, CPU-only import, and installed dependency consistency."""
    if not python_executable.is_file() or not stamp_file.is_file():
        return False
    try:
        if stamp_file.read_text(encoding="utf-8").strip() != expected_fingerprint:
            return False
        smoke = run(
            [
                str(python_executable),
                "-c",
                "import numpy, torch; assert torch.version.cuda is None",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if smoke.returncode != 0:
            return False
        consistency = run(
            [str(python_executable), "-m", "pip", "check"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return consistency.returncode == 0
    except (OSError, UnicodeError):
        return False


def main() -> int:
    if sys.version_info < (3, 12):
        raise SystemExit(
            "SYNTRA requires Python 3.12 or newer; current interpreter is "
            + sys.version.split()[0]
        )
    if not LOCKFILE.is_file():
        raise SystemExit(
            "SYNTRA Python dependency lockfile is missing: "
            + str(LOCKFILE)
            + ". Run git pull and retry setup."
        )

    expected_fingerprint = lock_fingerprint()
    if environment_is_current(VENV_PYTHON, LOCK_STAMP, expected_fingerprint):
        print("SYNTRA Python environment matches the committed lockfile.", flush=True)
        return 0

    if VENV.exists():
        print(
            "The existing Python environment is stale or invalid. "
            "Recreating it from the committed lockfile.",
            flush=True,
        )
        shutil.rmtree(VENV)

    print("Creating isolated environment at", VENV, flush=True)
    venv.EnvBuilder(with_pip=True).create(VENV)

    print("Installing hash-verified SYNTRA Python dependencies into", VENV, flush=True)
    subprocess.run(
        [
            str(VENV_PYTHON), "-m", "pip", "install",
            "--require-hashes", "-r", str(LOCKFILE),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "check"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            str(VENV_PYTHON), "-c",
            "import numpy, torch; "
            "assert torch.version.cuda is None, 'Expected CPU-only PyTorch'; "
            "print('NumPy', numpy.__version__, 'PyTorch', torch.__version__, "
            "'CUDA', torch.version.cuda)",
        ],
        cwd=ROOT,
        check=True,
    )
    LOCK_STAMP.write_text(expected_fingerprint + "\n", encoding="utf-8")
    print("SYNTRA local CPU Python environment is ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

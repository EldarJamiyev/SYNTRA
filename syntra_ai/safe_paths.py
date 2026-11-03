"""Canonical path validation for local native-model corpus inputs.

The loopback HTTP API is reachable by local browser contexts, so request paths
must not be allowed to select arbitrary files on the host. Corpus locations are
restricted to the repository and explicitly approved local directories.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALLOWED_ROOTS_ENV = "SYNTRA_ALLOWED_CORPUS_ROOTS"
_MAX_PATH_CHARS = 4096


def _approved_roots() -> tuple[Path, ...]:
    roots = [PROJECT_ROOT]
    configured = os.environ.get(_ALLOWED_ROOTS_ENV, "")
    for item in configured.split(os.pathsep):
        value = item.strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise ValueError(
                f"{_ALLOWED_ROOTS_ENV} entries must be absolute directory paths."
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError(
                f"An {_ALLOWED_ROOTS_ENV} directory does not exist or cannot be resolved."
            ) from None
        if resolved == Path(resolved.anchor) or not resolved.is_dir():
            raise ValueError(
                f"{_ALLOWED_ROOTS_ENV} entries must be existing non-root directories."
            )
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _resolve_approved_path(value: str | Path, *, expected: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"The selected {expected} path must be a string.")
    raw = os.fspath(value)
    if not raw or len(raw) > _MAX_PATH_CHARS or "\x00" in raw:
        raise ValueError(f"The selected {expected} path is invalid.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError(f"The selected {expected} path must be absolute.")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"The selected {expected} path does not exist or cannot be resolved.") from None

    roots = _approved_roots()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(
            f"The selected {expected} path is outside the SYNTRA project and approved local data directories."
        )
    return resolved


def resolve_corpus_root(value: str | Path) -> Path:
    """Return an existing, approved directory for training or evaluation."""
    resolved = _resolve_approved_path(value, expected="corpus")
    if resolved == Path(resolved.anchor) or not resolved.is_dir():
        raise ValueError("The selected corpus path must be an existing non-root directory.")
    return resolved


def resolve_instruction_dataset(value: str | Path) -> Path:
    """Return an existing, approved JSONL file for supervised fine-tuning."""
    resolved = _resolve_approved_path(value, expected="fine-tuning dataset")
    if not resolved.is_file():
        raise ValueError("The fine-tuning dataset must be an existing regular file.")
    if resolved.suffix.lower() != ".jsonl":
        raise ValueError("Fine-tuning datasets must use the .jsonl file extension.")
    return resolved

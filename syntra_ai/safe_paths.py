"""Canonical path validation for local native-model corpus inputs.

Untrusted paths are canonicalized with realpath and checked against explicit
approved roots before a Path object or filesystem operation is created.
"""
from pathlib import Path
import os
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALLOWED_ROOTS_ENV = "SYNTRA_ALLOWED_CORPUS_ROOTS"
_MAX_PATH_CHARS = 4096
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.jsonl", re.IGNORECASE)


def _is_filesystem_root(value: str) -> bool:
    return os.path.dirname(value) == value


def is_path_within(candidate: str | Path, root: str | Path) -> bool:
    """Compare canonical paths using a component boundary, not a string prefix."""
    try:
        candidate_real = os.path.normcase(os.path.realpath(os.fspath(candidate), strict=True))
        root_real = os.path.normcase(os.path.realpath(os.fspath(root), strict=True))
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    root_prefix = root_real.rstrip("/\\") + os.sep
    return candidate_real == root_real or candidate_real.startswith(root_prefix)


def _approved_roots() -> tuple[str, ...]:
    project_root = os.path.realpath(os.fspath(PROJECT_ROOT), strict=True)
    roots = [project_root]
    configured = os.environ.get(_ALLOWED_ROOTS_ENV, "")
    for item in configured.split(os.pathsep):
        value = item.strip()
        if not value:
            continue
        if len(value) > _MAX_PATH_CHARS or "\x00" in value or not os.path.isabs(value):
            raise ValueError(f"{_ALLOWED_ROOTS_ENV} entries must be absolute directory paths.")
        try:
            resolved = os.path.realpath(value, strict=True)
        except (OSError, RuntimeError):
            raise ValueError(
                f"An {_ALLOWED_ROOTS_ENV} directory does not exist or cannot be resolved."
            ) from None
        if _is_filesystem_root(resolved) or not os.path.isdir(resolved):
            raise ValueError(
                f"{_ALLOWED_ROOTS_ENV} entries must be existing non-root directories."
            )
        normalized = os.path.normcase(resolved)
        if all(os.path.normcase(root) != normalized for root in roots):
            roots.append(resolved)
    return tuple(roots)


def _resolve_approved_path(value: str | Path, *, expected: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"The selected {expected} path must be a string.")
    raw = os.fspath(value)
    if not raw or len(raw) > _MAX_PATH_CHARS or "\x00" in raw:
        raise ValueError(f"The selected {expected} path is invalid.")
    if not os.path.isabs(raw):
        raise ValueError(f"The selected {expected} path must be absolute.")
    try:
        # realpath resolves traversal and symlinks before the containment guard.
        resolved = os.path.realpath(raw, strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"The selected {expected} path does not exist or cannot be resolved.") from None

    roots = _approved_roots()
    if not any(is_path_within(resolved, root) for root in roots):
        raise ValueError(
            f"The selected {expected} path is outside the SYNTRA project and approved local data directories."
        )
    return Path(resolved)


def resolve_corpus_root(value: str | Path) -> Path:
    """Return an existing, approved directory for training or evaluation."""
    resolved = os.fspath(_resolve_approved_path(value, expected="corpus"))
    if _is_filesystem_root(resolved) or not os.path.isdir(resolved):
        raise ValueError("The selected corpus path must be an existing non-root directory.")
    return Path(resolved)


def resolve_instruction_dataset(value: str | Path) -> Path:
    """Return an existing, approved JSONL file for supervised fine-tuning."""
    resolved = os.fspath(_resolve_approved_path(value, expected="fine-tuning dataset"))
    if not os.path.isfile(resolved):
        raise ValueError("The fine-tuning dataset must be an existing regular file.")
    if os.path.splitext(resolved)[1].lower() != ".jsonl":
        raise ValueError("Fine-tuning datasets must use the .jsonl file extension.")
    return Path(resolved)


def resolve_curriculum_output_path(
    value: str | Path | None,
    workspace_root: str | Path,
) -> Path:
    """Resolve a JSONL output beneath the project or selected workspace only."""
    if value is None:
        default_parent = os.path.join(os.fspath(PROJECT_ROOT), "data", "curricula")
        os.makedirs(default_parent, exist_ok=True)
        parent = os.path.realpath(default_parent, strict=True)
        filename = "workspace-instructions.jsonl"
        allowed_roots = (os.fspath(PROJECT_ROOT), os.fspath(workspace_root))
    else:
        raw = os.fspath(value)
        if not raw or len(raw) > _MAX_PATH_CHARS or "\x00" in raw:
            raise ValueError("Curriculum output path is invalid.")
        requested = raw if os.path.isabs(raw) else os.path.join(os.fspath(PROJECT_ROOT), raw)
        filename = os.path.basename(requested)
        parent_input = os.path.dirname(requested) or os.curdir
        try:
            parent = os.path.realpath(parent_input, strict=True)
        except (OSError, RuntimeError):
            raise ValueError("Curriculum output directory must already exist and be resolvable.") from None
        allowed_roots = (os.fspath(PROJECT_ROOT), os.fspath(workspace_root))

    if not _OUTPUT_NAME.fullmatch(filename):
        raise ValueError("Curriculum output filename must be a simple .jsonl filename.")
    if not os.path.isdir(parent):
        raise ValueError("Curriculum output directory must be an existing directory.")
    if not any(is_path_within(parent, root) for root in allowed_roots):
        raise ValueError("Curriculum output directory must be inside the active workspace or SYNTRA project directory.")

    target = os.path.join(parent, filename)
    # Do not follow a pre-existing output symlink, even when its name is allowed.
    if os.path.islink(target):
        raise ValueError("Curriculum output must not be a symbolic link.")
    return Path(target)

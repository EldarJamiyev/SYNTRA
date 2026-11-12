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


def _matches_path_boundary(candidate: str, root: str) -> bool:
    candidate_norm = os.path.normcase(os.path.normpath(candidate))
    root_norm = os.path.normcase(os.path.normpath(root))
    root_prefix = root_norm.rstrip("/\\") + os.sep
    return candidate_norm == root_norm or candidate_norm.startswith(root_prefix)


def is_path_within(candidate: str | Path, root: str | Path) -> bool:
    """Compare canonical paths using a component boundary, not a string prefix."""
    try:
        candidate_real = os.path.realpath(os.fspath(candidate), strict=True)
        root_real = os.path.realpath(os.fspath(root), strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return _matches_path_boundary(candidate_real, root_real)


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
        # Canonicalize untrusted input before the local containment test.
        resolved = os.path.realpath(raw, strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"The selected {expected} path does not exist or cannot be resolved.") from None

    approved = False
    for trusted_root in _approved_roots():
        root_real = os.path.realpath(trusted_root, strict=True)
        candidate_norm = os.path.normcase(os.path.normpath(resolved))
        root_norm = os.path.normcase(os.path.normpath(root_real))
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        # Keep this canonical prefix check local to the path sanitizer so the
        # validated value remains visibly constrained before any later sink.
        if candidate_norm == root_norm or candidate_norm.startswith(root_prefix):
            approved = True
            break
    if not approved:
        raise ValueError(
            f"The selected {expected} path is outside the SYNTRA project and approved local data directories."
        )
    return Path(resolved)


def resolve_corpus_root(value: str | Path) -> Path:
    """Return an existing, approved directory for training or evaluation."""
    resolved = os.path.normcase(os.path.normpath(os.path.realpath(
        os.fspath(_resolve_approved_path(value, expected="corpus")), strict=True
    )))
    for trusted_root in _approved_roots():
        root_norm = os.path.normcase(os.path.normpath(os.path.realpath(trusted_root, strict=True)))
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if resolved == root_norm:
            if _is_filesystem_root(root_norm) or not os.path.isdir(root_norm):
                raise ValueError("The selected corpus path must be an existing approved non-root directory.")
            return Path(root_norm)
        if resolved.startswith(root_prefix):
            break
    else:
        raise ValueError("The selected corpus path must be an existing approved non-root directory.")
    if _is_filesystem_root(resolved) or not os.path.isdir(resolved):
        raise ValueError("The selected corpus path must be an existing approved non-root directory.")
    return Path(resolved)

def resolve_instruction_dataset(value: str | Path) -> Path:
    """Return an existing, approved JSONL file for supervised fine-tuning."""
    resolved = os.path.normcase(os.path.normpath(os.path.realpath(
        os.fspath(_resolve_approved_path(value, expected="fine-tuning dataset")), strict=True
    )))
    for trusted_root in _approved_roots():
        root_norm = os.path.normcase(os.path.normpath(os.path.realpath(trusted_root, strict=True)))
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if resolved.startswith(root_prefix):
            break
    else:
        raise ValueError("The fine-tuning dataset must be an existing regular file inside an approved root.")
    if not os.path.isfile(resolved):
        raise ValueError("The fine-tuning dataset must be an existing regular file inside an approved root.")
    if os.path.splitext(resolved)[1].lower() != ".jsonl":
        raise ValueError("Fine-tuning datasets must use the .jsonl file extension.")
    return Path(resolved)

def resolve_curriculum_output_path(
    value: str | Path | None,
    workspace_root: str | Path,
) -> Path:
    """Resolve a JSONL output beneath the project or selected workspace only."""
    try:
        workspace_real = os.path.normcase(os.path.normpath(os.path.realpath(os.fspath(workspace_root), strict=True)))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ValueError("Curriculum workspace root does not exist or cannot be resolved.") from None
    approved_workspace = False
    for trusted_root in _approved_roots():
        root_norm = os.path.normcase(os.path.normpath(os.path.realpath(trusted_root, strict=True)))
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if workspace_real == root_norm or workspace_real.startswith(root_prefix):
            approved_workspace = True
            break
    if not approved_workspace or _is_filesystem_root(workspace_real):
        raise ValueError("Curriculum workspace root must be inside an approved non-root directory.")

    if value is None:
        default_parent = os.path.join(os.fspath(PROJECT_ROOT), "data", "curricula")
        filename = "workspace-instructions.jsonl"
        allowed_roots = (os.fspath(PROJECT_ROOT), workspace_real)
        parent_input = default_parent
    else:
        raw = os.fspath(value)
        if not raw or len(raw) > _MAX_PATH_CHARS or "\x00" in raw:
            raise ValueError("Curriculum output path is invalid.")
        requested = raw if os.path.isabs(raw) else os.path.join(os.fspath(PROJECT_ROOT), raw)
        filename = os.path.basename(requested)
        parent_input = os.path.dirname(requested) or os.curdir
        allowed_roots = (os.fspath(PROJECT_ROOT), workspace_real)

    if not _OUTPUT_NAME.fullmatch(filename):
        raise ValueError("Curriculum output filename must be a simple .jsonl filename.")
    try:
        parent = os.path.normcase(os.path.normpath(os.path.realpath(parent_input, strict=False)))
        trusted_roots = tuple(
            os.path.normcase(os.path.normpath(os.path.realpath(root, strict=True)))
            for root in allowed_roots
        )
    except (OSError, RuntimeError):
        raise ValueError("Curriculum output directory cannot be resolved safely.") from None

    for root_norm in trusted_roots:
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if parent == root_norm or parent.startswith(root_prefix):
            break
    else:
        raise ValueError("Curriculum output directory must be inside the active workspace or SYNTRA project directory.")


    target = os.path.normcase(os.path.normpath(os.path.join(parent, filename)))
    for root_norm in trusted_roots:
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if target.startswith(root_prefix):
            break
    else:
        raise ValueError("Curriculum output must be a simple file inside an approved directory and must not be a symbolic link.")
    if os.path.islink(target):
        raise ValueError("Curriculum output must be a simple file inside an approved directory and must not be a symbolic link.")
    return Path(target)

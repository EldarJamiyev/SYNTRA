"""Content-addressed identity for the raw-model benchmark protocol and fixtures."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

MAX_PROTOCOL_FILES = 20_000
MAX_PROTOCOL_FILE_BYTES = 128 * 1024 * 1024
MAX_PROTOCOL_TOTAL_BYTES = 512 * 1024 * 1024


class EvaluationProvenanceError(ValueError):
    """The evaluation inputs are unsafe, incomplete, or changed while hashing."""


def _identity_tuple(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _hash_regular_file(
    path: Path,
    repository_root: str | Path | None = None,
) -> tuple[str, int]:
    """Hash a bounded regular file while binding its bytes to its canonical path.

    The optional repository root is used by the protocol walker to reject a
    parent-directory replacement that redirects the same inode outside the
    approved repository. Descriptor identity alone cannot detect that case.
    """
    path = Path(path)
    if path.is_symlink():
        raise EvaluationProvenanceError("Evaluation protocol must not contain symbolic links: " + str(path))
    try:
        before = path.lstat()
        canonical_before = path.resolve(strict=True)
        approved_root = (
            Path(repository_root).resolve(strict=True)
            if repository_root is not None
            else None
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationProvenanceError(
            "Cannot inspect evaluation input " + str(path) + ": " + str(exc)
        ) from None

    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_PROTOCOL_FILE_BYTES:
        raise EvaluationProvenanceError("Evaluation input must be a bounded regular file: " + str(path))

    def is_inside_approved_root(candidate: Path) -> bool:
        if approved_root is None:
            return True
        try:
            candidate.relative_to(approved_root)
            return True
        except ValueError:
            return False

    if not is_inside_approved_root(canonical_before):
        raise EvaluationProvenanceError(
            "Evaluation input escapes the approved repository root: " + str(path)
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if _identity_tuple(opened) != _identity_tuple(before):
                raise EvaluationProvenanceError("Evaluation input changed while opening: " + str(path))
            digest = hashlib.sha256()
            total = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
                if total > MAX_PROTOCOL_FILE_BYTES:
                    raise EvaluationProvenanceError(
                        "Evaluation input exceeded its size limit while hashing: " + str(path)
                    )
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            current = path.lstat()
            canonical_after = path.resolve(strict=True)
    except EvaluationProvenanceError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationProvenanceError(
            "Cannot hash evaluation input " + str(path) + ": " + str(exc)
        ) from None

    if (
        not stat.S_ISREG(current.st_mode)
        or _identity_tuple(before) != _identity_tuple(after)
        or _identity_tuple(before) != _identity_tuple(current)
        or canonical_before != canonical_after
        or not is_inside_approved_root(canonical_after)
        or total != before.st_size
    ):
        raise EvaluationProvenanceError(
            "Evaluation input changed while hashing or its path identity was replaced/redirected: "
            + str(path)
        )
    return digest.hexdigest(), total

def evaluation_protocol_identity(repository_root: str | Path, engine_root: str | Path) -> dict[str, Any]:
    """Hash every checked-in input that defines benchmark prompts and test outcomes.

    Container image digests remain separately recorded in each benchmark report.
    """
    repository = Path(repository_root).resolve(strict=True)
    engine = Path(engine_root).resolve(strict=True)
    required = (
        engine / "suite.v1.json",
        engine / "runner.py",
        engine / "environments.py",
        engine / "apply_patch_bridge.mjs",
        engine / "typescript_test_runner.mjs",
        engine / "evaluation_provenance.py",
        repository / "package-lock.json",
        repository / "local",
        engine / "fixtures",
        engine / "hidden",
    )
    files: list[Path] = []

    def reject_walk_error(error: OSError) -> None:
        raise EvaluationProvenanceError("Cannot enumerate evaluation protocol inputs: " + str(error)) from None

    for item in required:
        if item.is_symlink():
            raise EvaluationProvenanceError("Evaluation protocol input must not be a symbolic link: " + str(item))
        try:
            resolved = item.resolve(strict=True)
        except OSError as exc:
            raise EvaluationProvenanceError("Required evaluation protocol input is missing: " + str(item)) from None
        try:
            resolved.relative_to(repository)
        except ValueError:
            raise EvaluationProvenanceError("Evaluation protocol input escapes the repository: " + str(item)) from None
        if item.is_dir():
            for current, directories, filenames in os.walk(item, followlinks=False, onerror=reject_walk_error):
                current_path = Path(current)
                directories.sort()
                filenames.sort()
                for name in directories:
                    child = current_path / name
                    if child.is_symlink():
                        raise EvaluationProvenanceError("Evaluation protocol directory must not contain symbolic links: " + str(child))
                for name in filenames:
                    files.append(current_path / name)
        elif item.is_file():
            files.append(item)
        else:
            raise EvaluationProvenanceError("Evaluation protocol input is not a regular file or directory: " + str(item))

    entries = []
    total_bytes = 0
    for path in sorted(files, key=lambda value: value.relative_to(repository).as_posix()):
        if len(entries) >= MAX_PROTOCOL_FILES:
            raise EvaluationProvenanceError("Evaluation protocol exceeds its file-count limit.")
        digest, size = _hash_regular_file(path, repository_root=repository)
        total_bytes += size
        if total_bytes > MAX_PROTOCOL_TOTAL_BYTES:
            raise EvaluationProvenanceError("Evaluation protocol exceeds its aggregate size limit.")
        entries.append({
            "path": path.relative_to(repository).as_posix(),
            "bytes": size,
            "sha256": digest,
        })
    if not entries:
        raise EvaluationProvenanceError("Evaluation protocol contains no files.")
    manifest = json.dumps(
        {"schema_version": 1, "files": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(manifest).hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }

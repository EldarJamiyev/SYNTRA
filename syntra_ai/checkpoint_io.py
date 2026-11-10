"""Race-resistant loading for native SYNTRA PyTorch checkpoint files.

Checkpoint validation and deserialization must operate on the same bytes. A
pathname checked with stat() and later passed to torch.load() is not a stable
identity: another process can replace the path between those operations.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
import stat
from tempfile import TemporaryFile
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from .resources import checkpoint_load_budget_bytes


MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


def _preserve_final_path(path: str | os.PathLike[str]) -> Path:
    """Resolve parent components while preserving the final component for O_NOFOLLOW."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def load_torch_checkpoint_snapshot(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = MAX_CHECKPOINT_BYTES,
) -> dict[str, Any]:
    """Deserialize a bounded, stable regular-file snapshot with weights-only loading.

    The source must remain the same regular file, with unchanged size and
    timestamps, from the initial no-follow stat through the final path check.
    torch.load() receives the verified temporary snapshot, never the mutable
    source pathname.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load a SYNTRA checkpoint.")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Checkpoint byte limit must be a positive integer.")

    source = _preserve_final_path(path)
    descriptor: int | None = None
    try:
        before = os.stat(source, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise ValueError("Checkpoint path must identify a non-empty regular file.")
        memory_budget = checkpoint_load_budget_bytes()
        effective_max_bytes = min(max_bytes, memory_budget) if memory_budget is not None else max_bytes
        if before.st_size > effective_max_bytes:
            if memory_budget is not None and memory_budget < max_bytes and before.st_size > memory_budget:
                available_gib = (memory_budget / 0.40) / (1024 ** 3)
                budget_gib = memory_budget / (1024 ** 3)
                raise ValueError(
                    f"Checkpoint file is {before.st_size / (1024 ** 3):.2f} GiB, above the "
                    f"40% checkpoint-load budget ({budget_gib:.2f} GiB of {available_gib:.2f} GiB "
                    "currently available RAM). Close memory-heavy applications or use a smaller "
                    "local checkpoint before loading it."
                )
            raise ValueError(f"Checkpoint exceeds the {max_bytes:,}-byte limit.")

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        with TemporaryFile(mode="w+b") as snapshot:
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = None
                opened = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _file_identity(opened) != _file_identity(before)
                ):
                    raise ValueError("Checkpoint changed while it was being opened.")
                if opened.st_size < 1 or opened.st_size > effective_max_bytes:
                    if memory_budget is not None and memory_budget < max_bytes and opened.st_size > memory_budget:
                        raise ValueError("Checkpoint exceeds the 40% checkpoint-load budget for currently available RAM.")
                    raise ValueError(f"Checkpoint exceeds the {max_bytes:,}-byte limit.")

                total = 0
                while True:
                    chunk = handle.read(min(_READ_CHUNK_BYTES, effective_max_bytes - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > effective_max_bytes:
                        if memory_budget is not None and memory_budget < max_bytes:
                            raise ValueError("Checkpoint exceeds the 40% checkpoint-load budget for currently available RAM.")
                        raise ValueError(f"Checkpoint exceeds the {max_bytes:,}-byte limit.")
                    snapshot.write(chunk)

                after = os.fstat(handle.fileno())
                current = os.stat(source, follow_symlinks=False)
                if (
                    not stat.S_ISREG(current.st_mode)
                    or _file_identity(opened) != _file_identity(after)
                    or _file_identity(opened) != _file_identity(current)
                    or total != after.st_size
                ):
                    raise ValueError("Checkpoint changed while it was being read.")

            snapshot.flush()
            snapshot.seek(0)
            try:
                payload = torch.load(snapshot, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                raise ValueError(f"Cannot load local SYNTRA checkpoint: {exc}") from None

    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot read local SYNTRA checkpoint safely: {exc}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must contain a JSON-like dictionary.")
    return payload


__all__ = ["MAX_CHECKPOINT_BYTES", "load_torch_checkpoint_snapshot"]

"""Atomic, versioned checkpoint management for the standalone SYNTRA brain."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from io import BytesIO
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

import torch

from .model import TransformerConfig
from .tokenizer import BytePairTokenizer

CHECKPOINT_FORMAT = "syntra-brain-checkpoint"
CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024
_METADATA_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _checkpoint_identity(path: Path) -> tuple[str, int]:
    """Hash one stable regular checkpoint without following a final symlink."""
    descriptor: int | None = None
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise ValueError("Checkpoint must be a non-empty regular file.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ValueError("Checkpoint changed while it was being opened.")
            digest = hashlib.sha256()
            total = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(handle.fileno())
            if (
                (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or opened.st_size != after.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
                or total != after.st_size
            ):
                raise ValueError("Checkpoint changed while its identity was being calculated.")
            return digest.hexdigest(), total
    except OSError as exc:
        raise ValueError(f"Cannot verify checkpoint identity: {exc}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise


def checkpoint_metadata_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path).expanduser().resolve()
    return target.with_suffix(target.suffix + ".json")


def save_checkpoint(
    path: str | os.PathLike[str],
    payload: dict[str, Any],
) -> Path:
    """Write a complete training state atomically, then update its JSON summary."""
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Refusing to save an unsupported SYNTRA brain checkpoint.")
    required = {
        "version",
        "config",
        "tokenizer",
        "model_state_dict",
        "optimizer_state_dict",
        "step",
        "epoch",
        "dataset_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError("Checkpoint is missing required fields: " + ", ".join(sorted(missing)))
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version.")
    config = TransformerConfig.from_dict(payload["config"])
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint model and tokenizer vocabularies do not match.")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("Checkpoint step must be a non-negative integer.")
    if type(payload["epoch"]) is not int or payload["epoch"] < 0:
        raise ValueError("Checkpoint epoch must be a non-negative integer.")

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise

    checkpoint_sha256, checkpoint_bytes = _checkpoint_identity(destination)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "checkpoint": str(destination),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "step": payload["step"],
        "epoch": payload["epoch"],
        "best_validation_loss": payload.get("best_validation_loss"),
        "train_loss": payload.get("train_loss"),
        "validation_loss": payload.get("validation_loss"),
        "dataset_sha256": payload["dataset_sha256"],
        "dataset_files": payload.get("dataset_files", []),
        "training_data_format": payload.get("training_data_format", "raw-token-stream-v1"),
        "instruction_example_count": int(payload.get("instruction_example_count", 0)),
        "training_context_blocks": int(payload.get("training_context_blocks", 0)),
        "config": config.to_dict(),
        "vocab_size": tokenizer.vocab_size,
        "parameter_count": int(payload.get("parameter_count", 0)),
        "estimated_training_memory_bytes": int(payload.get("estimated_training_memory_bytes", 0)),
        "updated_at": payload.get("updated_at"),
    }
    _atomic_write_json(checkpoint_metadata_path(destination), metadata)
    return destination


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = MAX_CHECKPOINT_BYTES,
) -> dict[str, Any]:
    """Load only tensor/primitive checkpoint data; never unpickle arbitrary objects."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Checkpoint byte limit must be a positive integer.")
    source = Path(path).expanduser()
    descriptor: int | None = None
    try:
        before = os.stat(source, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise ValueError("Checkpoint path must identify a non-empty regular file.")
        if before.st_size > max_bytes:
            raise ValueError(f"Checkpoint must be between 1 and {max_bytes:,} bytes.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_size < 1
                or opened.st_size > max_bytes
            ):
                raise ValueError("Checkpoint changed while it was being opened.")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = handle.read(min(1024 * 1024, max_bytes - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Checkpoint exceeds the {max_bytes:,}-byte limit.")
            after = os.fstat(handle.fileno())
            current = os.stat(source, follow_symlinks=False)
            identity = (opened.st_dev, opened.st_ino)
            if (
                identity != (after.st_dev, after.st_ino)
                or identity != (current.st_dev, current.st_ino)
                or not stat.S_ISREG(current.st_mode)
                or opened.st_size != after.st_size
                or opened.st_size != current.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
                or total != after.st_size
            ):
                raise ValueError("Checkpoint changed while it was being read.")
        payload = torch.load(BytesIO(b"".join(chunks)), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot load local SYNTRA checkpoint: {exc}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("The selected file is not a supported SYNTRA brain checkpoint.")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("The SYNTRA brain checkpoint version is not supported.")
    required = {
        "config",
        "tokenizer",
        "model_state_dict",
        "optimizer_state_dict",
        "step",
        "epoch",
        "dataset_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError("Checkpoint is missing required fields: " + ", ".join(sorted(missing)))
    config = TransformerConfig.from_dict(payload["config"])
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint model and tokenizer vocabularies do not match.")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("Checkpoint step is invalid.")
    if type(payload["epoch"]) is not int or payload["epoch"] < 0:
        raise ValueError("Checkpoint epoch is invalid.")
    if not isinstance(payload["model_state_dict"], dict):
        raise ValueError("Checkpoint model weights are malformed.")
    if not isinstance(payload["optimizer_state_dict"], dict):
        raise ValueError("Checkpoint optimizer state is malformed.")
    if not isinstance(payload["dataset_sha256"], str) or len(payload["dataset_sha256"]) != 64:
        raise ValueError("Checkpoint dataset fingerprint is malformed.")
    return payload


def read_checkpoint_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read metadata; reject a modern sidecar that no longer matches its checkpoint."""
    target = Path(path).expanduser().resolve()
    metadata_path = checkpoint_metadata_path(target)
    try:
        if metadata_path.is_symlink() or not metadata_path.is_file():
            return {}
        if metadata_path.stat().st_size > 128 * 1024:
            return {}
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        # Older checkpoints remain readable, but their sidecars are explicitly legacy
        # and are not treated as content-verified.
        if "checkpoint_sha256" not in value and "checkpoint_bytes" not in value:
            return value
        expected_hash = value.get("checkpoint_sha256")
        expected_bytes = value.get("checkpoint_bytes")
        if (
            not isinstance(expected_hash, str)
            or _METADATA_SHA256.fullmatch(expected_hash) is None
            or type(expected_bytes) is not int
            or expected_bytes < 1
            or value.get("checkpoint") != str(target)
        ):
            return {}
        actual_hash, actual_bytes = _checkpoint_identity(target)
        if actual_hash != expected_hash or actual_bytes != expected_bytes:
            return {}
        return value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "MAX_CHECKPOINT_BYTES",
    "checkpoint_metadata_path",
    "save_checkpoint",
    "load_checkpoint",
    "read_checkpoint_metadata",
]

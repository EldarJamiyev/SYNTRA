"""Atomic, versioned checkpoint management for the standalone SYNTRA brain."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .model import TransformerConfig
from .tokenizer import BytePairTokenizer

CHECKPOINT_FORMAT = "syntra-brain-checkpoint"
CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024


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
    except OSError:
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
    except OSError:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise

    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "checkpoint": str(destination),
        "step": payload["step"],
        "epoch": payload["epoch"],
        "best_validation_loss": payload.get("best_validation_loss"),
        "train_loss": payload.get("train_loss"),
        "validation_loss": payload.get("validation_loss"),
        "dataset_sha256": payload["dataset_sha256"],
        "dataset_files": payload.get("dataset_files", []),
        "config": config.to_dict(),
        "vocab_size": tokenizer.vocab_size,
        "parameter_count": int(payload.get("parameter_count", 0)),
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
    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("Checkpoint path must identify a regular file.")
        size = resolved.stat().st_size
        if size < 1 or size > max_bytes:
            raise ValueError(f"Checkpoint must be between 1 and {max_bytes:,} bytes.")
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError(f"Cannot load local SYNTRA checkpoint: {exc}") from None

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
    """Read the lightweight status file without deserializing model tensors."""
    metadata_path = checkpoint_metadata_path(path)
    try:
        if not metadata_path.is_file():
            return {}
        if metadata_path.stat().st_size > 128 * 1024:
            return {}
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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

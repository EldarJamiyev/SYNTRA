"""Checkpoint-backed tokenizer, resident inference cache, context counting, embeddings, and fresh evaluation."""
from __future__ import annotations
from codecs import getincrementaldecoder
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Iterator
import hashlib
import json
import math
import os
import stat

try:
    import torch
    from torch.nn import functional as F
    from .model import SyntraTransformer
    _no_grad = torch.no_grad()
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    SyntraTransformer = None  # type: ignore[assignment]
    def _no_grad(fn):  # type: ignore[misc]
        return fn

from .config import ModelConfig
from .compute_lock import compute_operation
from .checkpoint_io import checkpoint_file_identity, load_torch_checkpoint_snapshot
from .tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer
from .training import (
    SOURCE_VALIDATION_STRATEGY,
    _atomic_json,
    _batch,
    _encode_document_partition,
    _estimate_loss,
    _partition_source_documents,
    choose_device,
    select_inference_checkpoint,
)
from .data import MAX_CORPUS_TOKENS, load_corpus
from .resources import validate_inference_memory_budget
from .safe_paths import resolve_corpus_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "syntra.pt"
FALLBACK_BRAIN_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "syntra-brain.pt"

_CACHE_LOCK = RLock()
_RESIDENT_CACHE: dict[str, tuple[tuple[Any, ...], Any, dict, ModelConfig, ByteBPETokenizer, Any]] = {}
# Per-checkpoint single-flight locks prevent concurrent requests from constructing
# duplicate resident models; reference counts keep the lock map bounded.
_RESIDENT_LOAD_LOCKS: dict[str, tuple[Any, int]] = {}
_TOKENIZER_CACHE: dict[str, tuple[tuple[Any, ...], ModelConfig, ByteBPETokenizer, int]] = {}
_TOKENIZER_LOAD_LOCKS: dict[str, tuple[Any, int]] = {}
_TOKENIZER_METADATA_MAX_BYTES = 4 * 1024 * 1024


class ContextWindowExceededError(ValueError):
    """A prompt cannot be preserved inside the selected checkpoint context."""

    def __init__(self, prompt_tokens: int, context_window_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.context_window_tokens = context_window_tokens
        self.usable_context_tokens = max(0, context_window_tokens - 1)
        self.tokens_over_context = max(0, prompt_tokens - self.usable_context_tokens)
        super().__init__(
            f"Prompt requires {prompt_tokens} tokens, but checkpoint context "
            f"{context_window_tokens} can preserve at most {self.usable_context_tokens} prompt tokens."
        )


def _checkpoint_stat(path: Path):
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("Checkpoint path must identify a regular file, not a symbolic link or directory.")
    return info


def _file_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = _checkpoint_stat(path)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_stat_identity_record(path: Path) -> dict[str, int]:
    return checkpoint_file_identity(path)


def checkpoint_path() -> Path:
    configured = os.environ.get("SYNTRA_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().absolute()
    default_resolved = DEFAULT_CHECKPOINT.expanduser().absolute()
    if default_resolved.is_file():
        return default_resolved
    fallback_resolved = FALLBACK_BRAIN_CHECKPOINT.expanduser().absolute()
    if fallback_resolved.is_file():
        return fallback_resolved
    return default_resolved


def _convert_brain_state_dict(state_dict: dict) -> dict:
    """Translate standalone brain.LocalTransformer parameter keys to SyntraTransformer keys."""
    converted = {}
    for key, tensor in state_dict.items():
        mapped = key
        mapped = mapped.replace(".norm_attention.", ".ln_1.")
        mapped = mapped.replace(".attention.qkv.", ".attn.qkv.")
        mapped = mapped.replace(".attention.projection.", ".attn.proj.")
        mapped = mapped.replace(".norm_feed_forward.", ".ln_2.")
        mapped = mapped.replace(".feed_forward.layers.", ".mlp.")
        mapped = mapped.replace(".feed_forward.w_", ".mlp.w_")
        if mapped.startswith("language_model_head."):
            mapped = "lm_head." + mapped[len("language_model_head."):]
        converted[mapped] = tensor
    return converted


def _normalize_checkpoint_payload(payload: dict) -> dict:
    """Unify syntra-transformer-v1 and syntra-brain-checkpoint formats transparently."""
    if not isinstance(payload, dict):
        raise ValueError("The selected file is not a SYNTRA Transformer checkpoint.")
    fmt = payload.get("format")
    if fmt == "syntra-transformer-v1":
        return payload
    if fmt == "syntra-brain-checkpoint":
        raw_config = dict(payload.get("config") or {})
        if "seq_len" not in raw_config and "block_size" in raw_config:
            raw_config["seq_len"] = raw_config["block_size"]
        raw_tokenizer = dict(payload.get("tokenizer") or {})
        if raw_tokenizer.get("format") == "syntra-local-byte-bpe-v1":
            raw_tokenizer["format"] = "syntra-byte-bpe-v1"
        raw_state = payload.get("model_state_dict") or payload.get("model_state")
        if not isinstance(raw_state, dict):
            raise ValueError("Checkpoint model weights are malformed.")
        normalized = dict(payload)
        normalized["format"] = "syntra-transformer-v1"
        normalized["source_format"] = "syntra-brain-checkpoint"
        normalized["config"] = raw_config
        normalized["tokenizer"] = raw_tokenizer
        normalized["model_state"] = _convert_brain_state_dict(raw_state)
        if "optimizer_state_dict" in payload and "optimizer_state" not in normalized:
            normalized["optimizer_state"] = payload["optimizer_state_dict"]
        return normalized
    raise ValueError("The selected file is not a SYNTRA Transformer checkpoint.")


def _read_checkpoint(
    path: str | Path | None = None,
    *,
    validate_tokenizer_vocab: bool = True,
) -> tuple[Path, dict, ModelConfig, ByteBPETokenizer]:
    target = Path(path).expanduser().absolute() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")
    load_target = select_inference_checkpoint(target)

    def load_validated(source: Path) -> tuple[dict, ModelConfig, ByteBPETokenizer]:
        raw_payload = load_torch_checkpoint_snapshot(source)
        payload = _normalize_checkpoint_payload(raw_payload)
        config = ModelConfig.from_dict(payload["config"])
        tokenizer = ByteBPETokenizer.from_dict(payload.get("tokenizer"))
        if validate_tokenizer_vocab and tokenizer.vocab_size != config.vocab_size:
            raise ValueError("Checkpoint tokenizer vocabulary does not match the model architecture.")
        return payload, config, tokenizer

    try:
        saved, config, tokenizer = load_validated(load_target)
    except MemoryError:
        raise
    except Exception:
        if load_target == target:
            raise
        # A missing, stale or damaged lightweight companion must never make the
        # original resumable checkpoint unusable.
        saved, config, tokenizer = load_validated(target)

    if load_target != target:
        # A companion is valid only for the exact regular source checkpoint from
        # which it was produced. Preserve the final path component so symlinks
        # cannot be normalized away before this no-follow identity check.
        source_identity = _file_stat_identity_record(target)
        if saved.get("source_checkpoint_identity") != source_identity:
            saved, config, tokenizer = load_validated(target)
        if _file_stat_identity_record(target) != source_identity:
            raise ValueError("The source checkpoint changed while its inference companion was being loaded.")
    return target, saved, config, tokenizer


def _resident_cache_signature(target: Path, device_name: str) -> tuple[tuple[Any, ...], Path]:
    load_target = select_inference_checkpoint(target)
    try:
        # Include device, inode, size, mtime and ctime for both the resumable
        # checkpoint and selected inference companion. Promotion uses atomic
        # replacement, which can preserve size/mtime while changing file identity.
        signature = (
            _file_stat_identity(target),
            str(load_target),
            _file_stat_identity(load_target),
            str(device_name),
        )
    except OSError:
        signature = (
            (-1, -1, -1, -1, -1),
            str(load_target),
            (-1, -1, -1, -1, -1),
            str(device_name),
        )
    return signature, load_target


def _tokenizer_cache_signature(target: Path) -> tuple[Any, ...]:
    """Bind tokenizer cache entries to source, companion and sidecar file identities."""
    missing_identity = (-1, -1, -1, -1, -1)
    try:
        source_identity = _file_stat_identity(target)
    except OSError:
        source_identity = missing_identity

    load_target = select_inference_checkpoint(target)
    try:
        companion_identity = _file_stat_identity(load_target)
    except OSError:
        companion_identity = missing_identity

    metadata_path = target.with_suffix(target.suffix + ".json")
    try:
        metadata_identity = _file_stat_identity(metadata_path)
    except OSError:
        metadata_identity = missing_identity
    return (source_identity, str(load_target), companion_identity, metadata_identity)


def _read_identity_bound_tokenizer_metadata(
    target: Path,
    expected_checkpoint_identity: dict[str, int],
) -> tuple[ModelConfig, ByteBPETokenizer, int] | None:
    """Read tokenizer metadata only when it is a stable sidecar for this checkpoint."""
    metadata_path = target.with_suffix(target.suffix + ".json")
    descriptor: int | None = None

    def stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def reject_constant(value: str):
        raise ValueError(f"Non-standard JSON constant {value} in checkpoint metadata.")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key in checkpoint metadata.")
            result[key] = value
        return result

    try:
        before = os.stat(metadata_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > _TOKENIZER_METADATA_MAX_BYTES
        ):
            return None

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(metadata_path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or stat_identity(opened) != stat_identity(before):
                return None

            content = handle.read(_TOKENIZER_METADATA_MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
            current = os.stat(metadata_path, follow_symlinks=False)
            if (
                len(content) > _TOKENIZER_METADATA_MAX_BYTES
                or stat_identity(opened) != stat_identity(after)
                or stat_identity(opened) != stat_identity(current)
                or len(content) != opened.st_size
            ):
                return None

        metadata = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        if not isinstance(metadata, dict):
            return None
        if metadata.get("checkpoint_identity") != expected_checkpoint_identity:
            return None

        raw_config = metadata.get("config")
        raw_tokenizer = metadata.get("tokenizer_data")
        step = metadata.get("step")
        if not isinstance(raw_config, dict) or not isinstance(raw_tokenizer, dict):
            return None
        if type(step) is not int or step < 0:
            return None

        config = ModelConfig.from_dict(raw_config)
        tokenizer = ByteBPETokenizer.from_dict(raw_tokenizer)
        if tokenizer.vocab_size != config.vocab_size:
            return None
        if _file_stat_identity_record(target) != expected_checkpoint_identity:
            return None
        return config, tokenizer, step
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError, json.JSONDecodeError):
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _get_tokenizer_bundle(
    path: str | Path | None = None,
) -> tuple[Path, ModelConfig, ByteBPETokenizer, int]:
    """Load only tokenizer/config metadata without constructing or retaining a model."""
    target = Path(path).expanduser().absolute() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")

    cache_key = str(target)
    def cached_bundle(signature_value: tuple[Any, ...]):
        with _CACHE_LOCK:
            cached_value = _TOKENIZER_CACHE.get(cache_key)
        if cached_value is None or cached_value[0] != signature_value:
            return None
        if _tokenizer_cache_signature(target) != signature_value:
            raise RuntimeError("The checkpoint changed while tokenizer metadata was being retrieved; retry the request.")
        return cached_value

    signature = _tokenizer_cache_signature(target)
    cached = cached_bundle(signature)
    if cached is not None:
        return target, cached[1], cached[2], cached[3]

    with _CACHE_LOCK:
        lock_entry = _TOKENIZER_LOAD_LOCKS.get(cache_key)
        if lock_entry is None:
            load_lock = Lock()
            _TOKENIZER_LOAD_LOCKS[cache_key] = (load_lock, 1)
        else:
            load_lock, users = lock_entry
            _TOKENIZER_LOAD_LOCKS[cache_key] = (load_lock, users + 1)

    load_lock.acquire()
    try:
        signature = _tokenizer_cache_signature(target)
        cached = cached_bundle(signature)
        if cached is not None:
            return target, cached[1], cached[2], cached[3]

        checkpoint_identity = _file_stat_identity_record(target)
        metadata_bundle = _read_identity_bound_tokenizer_metadata(target, checkpoint_identity)
        if metadata_bundle is not None:
            config, tokenizer, trained_steps = metadata_bundle
        else:
            _loaded_target, saved, config, tokenizer = _read_checkpoint(target)
            trained_steps = int(saved.get("step", 0))
            # Do not keep the deserialized state dictionary alive after extracting
            # the small tokenizer/config bundle; token counting does not need weights.
            del saved

        signature_after = _tokenizer_cache_signature(target)
        if signature_after != signature:
            raise RuntimeError("The checkpoint changed while tokenizer metadata was loading; retry the request.")

        with _CACHE_LOCK:
            if len(_TOKENIZER_CACHE) >= 4 and cache_key not in _TOKENIZER_CACHE:
                oldest_key = next(iter(_TOKENIZER_CACHE))
                _TOKENIZER_CACHE.pop(oldest_key, None)
            _TOKENIZER_CACHE[cache_key] = (signature, config, tokenizer, trained_steps)
        return target, config, tokenizer, trained_steps
    finally:
        load_lock.release()
        with _CACHE_LOCK:
            entry = _TOKENIZER_LOAD_LOCKS.get(cache_key)
            if entry is not None and entry[0] is load_lock:
                if entry[1] <= 1:
                    _TOKENIZER_LOAD_LOCKS.pop(cache_key, None)
                else:
                    _TOKENIZER_LOAD_LOCKS[cache_key] = (load_lock, entry[1] - 1)


def _get_resident_bundle(path: str | Path | None = None, device_name: str = "auto") -> tuple[Path, SyntraTransformer, dict, ModelConfig, ByteBPETokenizer, torch.device]:
    target = Path(path).expanduser().absolute() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")

    cache_key = str(target)
    signature, _load_target = _resident_cache_signature(target, device_name)
    with _CACHE_LOCK:
        cached = _RESIDENT_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            _, model, saved, config, tokenizer, device = cached
            return target, model, saved, config, tokenizer, device

        lock_entry = _RESIDENT_LOAD_LOCKS.get(cache_key)
        if lock_entry is None:
            load_lock = Lock()
            _RESIDENT_LOAD_LOCKS[cache_key] = (load_lock, 1)
        else:
            load_lock, users = lock_entry
            _RESIDENT_LOAD_LOCKS[cache_key] = (load_lock, users + 1)

    load_lock.acquire()
    try:
        # Another request may have loaded this checkpoint while we waited.
        # Re-resolve the inference companion and fingerprint after acquiring the
        # per-path lock, then re-check the cache before doing expensive work.
        signature, _load_target = _resident_cache_signature(target, device_name)
        with _CACHE_LOCK:
            cached = _RESIDENT_CACHE.get(cache_key)
            if cached is not None and cached[0] == signature:
                _, model, saved, config, tokenizer, device = cached
                return target, model, saved, config, tokenizer, device

        target, saved, config, tokenizer = _read_checkpoint(target)
        device = choose_device(device_name)
        validate_inference_memory_budget(
            config, batch_size=1, sequence_length=config.seq_len, device=device
        )
        model = SyntraTransformer(config).to(device)
        model.load_state_dict(saved["model_state"])
        model.eval()

        # Never publish a model loaded from bytes whose checkpoint identity is
        # no longer the identity observed before deserialization/model creation.
        signature_after, _load_target_after = _resident_cache_signature(target, device_name)
        if signature_after != signature:
            raise RuntimeError(
                "The checkpoint changed while its resident model was loading; retry the request."
            )

        with _CACHE_LOCK:
            if len(_RESIDENT_CACHE) >= 4 and cache_key not in _RESIDENT_CACHE:
                oldest_key = next(iter(_RESIDENT_CACHE))
                _RESIDENT_CACHE.pop(oldest_key, None)
            _RESIDENT_CACHE[cache_key] = (
                signature, model, saved, config, tokenizer, device
            )
        return target, model, saved, config, tokenizer, device
    finally:
        load_lock.release()
        with _CACHE_LOCK:
            entry = _RESIDENT_LOAD_LOCKS.get(cache_key)
            if entry is not None and entry[0] is load_lock:
                if entry[1] <= 1:
                    _RESIDENT_LOAD_LOCKS.pop(cache_key, None)
                else:
                    _RESIDENT_LOAD_LOCKS[cache_key] = (load_lock, entry[1] - 1)

def clear_resident_cache() -> None:
    with _CACHE_LOCK:
        _RESIDENT_CACHE.clear()
        _TOKENIZER_CACHE.clear()


def _checkpoint_identity(path: str | Path | None = None) -> str | None:
    """Identify the current full checkpoint and its selected inference companion."""
    source = Path(path or checkpoint_path()).expanduser().absolute()
    try:
        selected = select_inference_checkpoint(source)
        parts = []
        for candidate in dict.fromkeys((source, selected)):
            info = _checkpoint_stat(candidate)
            parts.append(
                f"{candidate}\0{info.st_dev}\0{info.st_ino}\0{info.st_size}\0{info.st_mtime_ns}\0{info.st_ctime_ns}"
            )
    except OSError:
        return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _resident_cache_matches_checkpoint(path: str | Path) -> bool:
    """Report a resident model only while its cached files match the current checkpoint."""
    target = Path(path).expanduser().absolute()
    try:
        selected = select_inference_checkpoint(target)
        source_identity = _file_stat_identity(target)
        selected_identity = _file_stat_identity(selected)
    except OSError:
        return False

    with _CACHE_LOCK:
        cached = _RESIDENT_CACHE.get(str(target))
    if cached is None:
        return False

    signature = cached[0]
    current_files = (source_identity, str(selected), selected_identity)
    return isinstance(signature, tuple) and signature[:3] == current_files


def _reconcile_checkpoint_status_metadata(
    checkpoint: Path,
    metadata_path: Path,
    metadata: dict,
    expected_identity: dict[str, int],
) -> dict:
    """Recover progress metadata from a verified, resumable native training checkpoint."""
    payload = load_torch_checkpoint_snapshot(checkpoint)
    current_identity = _file_stat_identity_record(checkpoint)
    if current_identity != expected_identity:
        raise RuntimeError("The checkpoint changed while status metadata was being reconciled.")

    training_mode = payload.get("training_mode")
    if training_mode == "supervised-instruction-finetuning":
        instruction_fingerprint = payload.get("instruction_dataset_sha256")
        instruction_bytes = payload.get("instruction_dataset_bytes")
        supported = (
            payload.get("finetuning_resume_schema") == "syntra-sft-resume-v1"
            and isinstance(payload.get("optimizer_state"), dict)
            and isinstance(payload.get("finetuning_rng_state"), dict)
            and isinstance(payload.get("finetuning_settings"), dict)
            and isinstance(instruction_fingerprint, str)
            and len(instruction_fingerprint) == 64
            and all(char in "0123456789abcdef" for char in instruction_fingerprint)
            and type(instruction_bytes) is int
            and instruction_bytes > 0
        )
    elif training_mode == "causal-source-pretraining":
        source_fingerprint = payload.get("training_data_sha256")
        supported = (
            isinstance(payload.get("optimizer_state"), dict)
            and isinstance(payload.get("rng_state"), dict)
            and isinstance(payload.get("training_settings"), dict)
            and isinstance(source_fingerprint, str)
            and len(source_fingerprint) == 64
            and all(char in "0123456789abcdef" for char in source_fingerprint)
            and payload.get("validation_strategy") == SOURCE_VALIDATION_STRATEGY
        )
    else:
        supported = False

    if not supported:
        raise ValueError("The checkpoint does not contain a supported resumable native training status record.")

    refreshed = dict(metadata)
    fields = (
        "config",
        "step",
        "train_loss",
        "validation_loss",
        "training_mode",
        "training_history",
        "requested_config",
        "training_settings",
        "training_data_sha256",
        "learning_rate",
        "training_scheduler",
        "minimum_learning_rate",
        "warmup_steps",
        "decay_steps",
        "gradient_clip_norm",
        "gradient_norm_before_clip",
        "corpus_files",
        "corpus_tokens",
        "training_documents",
        "validation_documents",
        "duplicate_documents_excluded",
        "token_budget_documents_excluded",
        "validation_strategy",
        "instruction_examples",
        "instruction_context_windows",
        "instruction_dataset_quality",
        "instruction_dataset_sha256",
        "instruction_dataset_bytes",
        "finetuning_resume_schema",
        "finetuning_settings",
        "finetuning_optimizer_scope",
        "finetuning_stage_base_step",
        "finetuning_stage_steps",
        "finetuning_stage_target_steps",
        "finetuning_validation_stage_step",
        "finetuning_remaining_steps",
        "finetuning_resume_supported",
        "finetuning_learning_rate",
        "finetuning_gradient_norm_before_clip",
    )
    for field in fields:
        if field in payload:
            refreshed[field] = payload[field]

    config = payload.get("config")
    if isinstance(config, dict):
        refreshed["context_tokens"] = config.get("seq_len") or config.get("block_size")
    tokenizer = payload.get("tokenizer")
    if isinstance(tokenizer, dict):
        refreshed["tokenizer_vocab_size"] = tokenizer.get("vocab_size")
        refreshed["tokenizer_data"] = tokenizer
    model_state = payload.get("model_state")
    if isinstance(model_state, dict) and torch is not None:
        refreshed["parameter_count"] = sum(
            tensor.numel() for tensor in model_state.values() if torch.is_tensor(tensor)
        )
    refreshed["checkpoint_identity"] = current_identity
    refreshed["status_reconciled_from_checkpoint"] = True
    _atomic_json(metadata_path, refreshed)
    if _file_stat_identity_record(checkpoint) != current_identity:
        raise RuntimeError("The checkpoint advanced again while its status metadata was being repaired.")
    return refreshed


def status() -> dict:
    path = checkpoint_path()
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = {}
    metadata_error = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint metadata must contain a JSON object.")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            metadata = {}
            metadata_error = str(exc)
    current_checkpoint_identity = None
    try:
        _checkpoint_stat(path)
        checkpoint_exists = True
        current_checkpoint_identity = _file_stat_identity_record(path)
    except OSError:
        checkpoint_exists = False

    checkpoint_metadata_stale: bool | None = None
    checkpoint_metadata_reconciled = False
    checkpoint_metadata_identity_verified = False
    checkpoint_metadata_warning = None
    saved_identity = metadata.get("checkpoint_identity")
    if checkpoint_exists:
        has_saved_identity = isinstance(saved_identity, dict)
        identity_matches = has_saved_identity and saved_identity == current_checkpoint_identity
        checkpoint_metadata_stale = (
            False if identity_matches else True if has_saved_identity else None
        )
        checkpoint_metadata_identity_verified = identity_matches
        if not identity_matches:
            training_mode = metadata.get("training_mode")
            resumable_metadata_contract = (
                training_mode == "causal-source-pretraining"
                or (
                    training_mode == "supervised-instruction-finetuning"
                    and metadata.get("finetuning_resume_schema") == "syntra-sft-resume-v1"
                )
            )
            if not resumable_metadata_contract:
                checkpoint_metadata_warning = (
                    "Checkpoint metadata has no verified file identity or recognized recovery contract."
                    if not has_saved_identity else
                    "Checkpoint file identity differs from metadata, and its training mode is not recoverable."
                )
            else:
                try:
                    metadata = _reconcile_checkpoint_status_metadata(
                        path,
                        metadata_path,
                        metadata,
                        current_checkpoint_identity,
                    )
                    checkpoint_metadata_stale = False
                    checkpoint_metadata_reconciled = True
                    checkpoint_metadata_identity_verified = True
                except Exception as exc:
                    checkpoint_metadata_warning = (
                        "Checkpoint progress metadata is unverified or stale and could not be reconciled: "
                        + str(exc)
                    )

    raw_config = metadata.get("config") or {}
    context_tokens = metadata.get("context_tokens") or raw_config.get("seq_len") or raw_config.get("block_size")
    vocab_size = metadata.get("tokenizer_vocab_size") or metadata.get("vocab_size") or raw_config.get("vocab_size")
    history_error = None
    try:
        from .training_history import validate_training_history
        training_history = validate_training_history(metadata.get("training_history"))
    except ValueError as exc:
        training_history = []
        history_error = str(exc)
    if metadata_error is not None and history_error is None:
        history_error = "Checkpoint metadata is unavailable or malformed: " + metadata_error
    return {
        "service": "SYNTRA native local Transformer",
        "backend": "PyTorch",
        "checkpoint": str(path),
        "checkpoint_identity": _checkpoint_identity(path) if checkpoint_exists else None,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_metadata_stale": checkpoint_metadata_stale,
        "checkpoint_metadata_reconciled": checkpoint_metadata_reconciled,
        "checkpoint_metadata_identity_verified": checkpoint_metadata_identity_verified,
        "checkpoint_metadata_warning": checkpoint_metadata_warning,
        "trained_steps": int(metadata.get("step", 0)),
        "parameter_count": int(metadata.get("parameter_count", 0)),
        "config": raw_config if raw_config else None,
        "train_loss": metadata.get("train_loss"),
        "validation_loss": metadata.get("validation_loss"),
        "best_validation_loss": metadata.get("best_validation_loss"),
        "learning_rate": metadata.get("learning_rate"),
        "training_scheduler": metadata.get("training_scheduler"),
        "minimum_learning_rate": metadata.get("minimum_learning_rate"),
        "warmup_steps": metadata.get("warmup_steps"),
        "decay_steps": metadata.get("decay_steps"),
        "gradient_clip_norm": metadata.get("gradient_clip_norm"),
        "gradient_norm_before_clip": metadata.get("gradient_norm_before_clip"),
        "finetuning_settings": metadata.get("finetuning_settings"),
        "finetuning_learning_rate": metadata.get("finetuning_learning_rate"),
        "finetuning_gradient_norm_before_clip": metadata.get("finetuning_gradient_norm_before_clip"),
        "finetuning_resume_schema": metadata.get("finetuning_resume_schema"),
        "finetuning_optimizer_scope": metadata.get("finetuning_optimizer_scope"),
        "finetuning_stage_base_step": metadata.get("finetuning_stage_base_step"),
        "finetuning_stage_steps": metadata.get("finetuning_stage_steps"),
        "finetuning_stage_target_steps": metadata.get("finetuning_stage_target_steps"),
        "finetuning_validation_stage_step": metadata.get("finetuning_validation_stage_step"),
        "finetuning_remaining_steps": metadata.get("finetuning_remaining_steps"),
        "finetuning_resume_supported": metadata.get("finetuning_resume_supported"),
        "training_history": training_history,
        "training_history_points": len(training_history),
        "training_history_valid": history_error is None,
        "training_history_error": history_error,
        "corpus_files": metadata.get("corpus_files", len(metadata.get("dataset_files", [])) if isinstance(metadata.get("dataset_files"), list) else 0),
        "corpus_tokens": metadata.get("corpus_tokens", 0),
        "training_documents": metadata.get("training_documents", 0),
        "validation_documents": metadata.get("validation_documents", 0),
        "duplicate_documents_excluded": metadata.get("duplicate_documents_excluded", 0),
        "token_budget_documents_excluded": metadata.get("token_budget_documents_excluded", 0),
        "validation_strategy": metadata.get("validation_strategy"),
        "tokenizer": metadata.get("tokenizer", "byte-BPE" if checkpoint_exists else None),
        "tokenizer_vocab_size": vocab_size,
        "context_tokens": context_tokens,
        "training_mode": metadata.get("training_mode", "causal-source-pretraining"),
        "pretrained_weights": False,
        "external_ai_api": False,
        "resident_in_memory": _resident_cache_matches_checkpoint(path),
        "capabilities": {
            "unifiedBrainCheckpoint": True,
            "residentCache": True,
            "nativeEmbeddings": True,
            "streamingGeneration": True,
            "ropeSupport": True,
            "rmsnormSupport": True,
            "swigluSupport": True,
        },
        "note": "Local byte-BPE Transformer. Generation requires a trained checkpoint; reported loss is not a coding benchmark."
    }


def load_model(path: str | Path | None = None, device_name: str = "auto"):
    _, model, saved, _, _, device = _get_resident_bundle(path=path, device_name=device_name)
    return model, saved, device


def tokenize_text(text: str, path: str | Path | None = None) -> dict:
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")
    if len(text.encode("utf-8")) > 300_000:
        raise ValueError("Text exceeds the 300 KB token-count API limit.")
    target, config, tokenizer, trained_steps = _get_tokenizer_bundle(path=path)
    count = len(tokenizer.encode(text))
    return {
        "token_count": count,
        "context_window_tokens": config.seq_len,
        "vocabulary_size": tokenizer.vocab_size,
        "tokenizer": "byte-BPE",
        "tokens_over_context": max(0, count - (config.seq_len - 1)),
        "checkpoint": str(target),
        "trained_steps": trained_steps,
    }


@_no_grad
@compute_operation("inference")
def embed_texts(
    texts: list[str],
    path: str | Path | None = None,
    device_name: str = "auto",
) -> dict:
    """Compute dense semantic embeddings using the native SYNTRA Transformer."""
    if not isinstance(texts, list) or not texts or len(texts) > 64:
        raise ValueError("input must be a non-empty list of at most 64 strings.")
    for item in texts:
        if not isinstance(item, str) or len(item.encode("utf-8")) > 100_000:
            raise ValueError("Each text item must be a string of at most 100 KB.")
    device_name = _normalize_device_name(device_name)
    identity_before = _checkpoint_identity(path)
    target, model, saved, config, tokenizer, device = _get_resident_bundle(path=path, device_name=device_name)
    vectors: list[list[float]] = []
    limit = config.seq_len - 1
    for item in texts:
        encoded = tokenizer.encode(item)[:limit]
        token_ids = [BOS_ID, *encoded]
        tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        vec = model.embed_tokens(tensor)[0].detach().cpu().tolist()
        vectors.append([round(float(v), 5) for v in vec])
    identity_after = _checkpoint_identity(path)
    if identity_before is None or identity_after != identity_before:
        raise RuntimeError("The native checkpoint changed during embedding generation; retry the request.")
    return {
        "embeddings": vectors,
        "dimensions": config.n_embd,
        "model": "syntra-native-transformer",
        "checkpoint": str(target),
        "checkpoint_identity": identity_before,
        "trained_steps": int(saved.get("step", 0)),
        "local_only": True,
    }


def _encode_prompt_within_context(
    prompt: str, tokenizer: ByteBPETokenizer, config: ModelConfig
) -> list[int]:
    """Encode the complete prompt or fail; never silently discard source context."""
    token_ids = tokenizer.encode(prompt)
    usable_context_tokens = config.seq_len - 1  # one position is consumed by BOS
    if len(token_ids) > usable_context_tokens:
        raise ContextWindowExceededError(len(token_ids), config.seq_len)
    return token_ids


def _token_bytes_table(tokenizer: ByteBPETokenizer) -> list[bytes]:
    table = [bytes((token_id,)) for token_id in range(256)] + [b"", b""]
    for left, right in tokenizer.merges:
        if left >= len(table) or right >= len(table):
            raise ValueError("Tokenizer merge table contains an invalid dependency.")
        table.append(table[left] + table[right])
    return table


def _normalize_device_name(device_name: Any) -> str:
    """Validate local device selection before opening a checkpoint or allocating a model."""
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps.")
    normalized = device_name.strip().lower()
    try:
        choose_device(normalized)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Requested device is unavailable: " + str(exc)) from None
    return normalized


def _normalize_generation_options(
    max_new_tokens: Any,
    temperature: Any,
    top_k: Any,
    top_p: Any,
    repetition_penalty: Any,
    device_name: Any,
) -> tuple[int, float, int, float, float, str]:
    """Validate generation controls before checkpoint loading or inference allocation."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    def finite_number(name: str, value: Any, minimum: float, maximum: float, *, exclusive_min: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}.")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}.") from None
        lower_invalid = normalized <= minimum if exclusive_min else normalized < minimum
        if not math.isfinite(normalized) or lower_invalid or normalized > maximum:
            comparator = "exclusive" if exclusive_min else "inclusive"
            raise ValueError(
                f"{name} must be a finite number in the {comparator} range "
                + f"{minimum}..{maximum}."
            )
        return normalized

    normalized_temperature = finite_number("temperature", temperature, 0.0, 2.0)
    normalized_top_p = finite_number("top_p", top_p, 0.0, 1.0, exclusive_min=True)
    normalized_repetition_penalty = finite_number(
        "repetition_penalty", repetition_penalty, 1.0, 2.0
    )
    normalized_device = _normalize_device_name(device_name)
    return (
        max_new_tokens,
        normalized_temperature,
        top_k,
        normalized_top_p,
        normalized_repetition_penalty,
        normalized_device,
    )


@compute_operation("inference")
def stream_generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device_name: str = "auto",
    path: str | Path | None = None,
) -> Iterator[dict]:
    """Stream token fragments and metadata incrementally from the resident native Transformer."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("A non-empty text prompt is required.")
    if len(prompt.encode("utf-8")) > 200_000:
        raise ValueError("Prompt exceeds 200 KB.")
    (
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        device_name,
    ) = _normalize_generation_options(
        max_new_tokens, temperature, top_k, top_p, repetition_penalty, device_name
    )
    target, model, saved, config, tokenizer, device = _get_resident_bundle(path=path, device_name=device_name)
    full_prompt_ids = _encode_prompt_within_context(prompt, tokenizer, config)
    prompt_ids = full_prompt_ids
    tokens = torch.tensor([[BOS_ID, *prompt_ids]], dtype=torch.long, device=device)
    token_bytes = _token_bytes_table(tokenizer)
    decoder = getincrementaldecoder("utf-8")("replace")
    generated_ids: list[int] = []
    full_text_pieces: list[str] = []

    context = tokens[:, -config.seq_len:]
    # Streaming runs outside a normal function call boundary: keep the model
    # forward explicitly out of autograd so its KV cache cannot retain graphs.
    with torch.no_grad():
        logits, _, past_key_values = model(context, use_cache=True)
    effective_top_k = max(1, min(int(top_k), config.vocab_size))

    for step in range(max_new_tokens):
        next_logits = logits[:, -1, :].float().clone()
        next_logits[:, BOS_ID] = float("-inf")
        if repetition_penalty > 1.0 and generated_ids:
            unique_ids = torch.tensor(sorted(set(generated_ids)), dtype=torch.long, device=device)
            unique_ids = unique_ids[unique_ids != EOS_ID]
            if unique_ids.numel() > 0:
                selected = next_logits[0, unique_ids]
                next_logits[0, unique_ids] = torch.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
        if temperature == 0.0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if effective_top_k < next_logits.shape[-1]:
                threshold = torch.topk(next_logits, effective_top_k, dim=-1).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False
                scatter_mask = torch.zeros_like(remove_mask).scatter(1, sorted_indices, remove_mask)
                next_logits = next_logits.masked_fill(scatter_mask, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        token_id = int(next_token[0, 0].item())
        tokens = torch.cat((tokens, next_token), dim=1)
        if token_id == EOS_ID:
            break
        generated_ids.append(token_id)
        delta = decoder.decode(token_bytes[token_id], final=False)
        if delta:
            full_text_pieces.append(delta)
            yield {
                "delta": delta,
                "generated_tokens": len(generated_ids),
                "done": False,
            }
        if step + 1 >= max_new_tokens:
            break
        cache_length = past_key_values[0][0].shape[-2]
        if cache_length < config.seq_len:
            with torch.no_grad():
                logits, _, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        else:
            context = tokens[:, -config.seq_len:]
            with torch.no_grad():
                logits, _, past_key_values = model(context, use_cache=True)

    tail = decoder.decode(b"", final=True)
    if tail:
        full_text_pieces.append(tail)
        yield {"delta": tail, "generated_tokens": len(generated_ids), "done": False}

    yield {
        "delta": "",
        "text": "".join(full_text_pieces),
        "prompt_tokens": len(full_prompt_ids),
        "prompt_tokens_used": len(prompt_ids),
        "prompt_truncated": len(prompt_ids) != len(full_prompt_ids),
        "context_window_tokens": config.seq_len,
        "tokenizer": "byte-BPE",
        "generated_tokens": len(generated_ids),
        "checkpoint": str(target),
        "trained_steps": int(saved.get("step", 0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "is_model_output": True,
        "done": True,
        "warning": "Raw neural-model output. Structured patch workflows must validate it; inspect and test generated code before applying it.",
    }


@_no_grad
@compute_operation("inference")
def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    device_name: str = "auto",
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    path: str | Path | None = None,
) -> dict:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("A non-empty text prompt is required.")
    if len(prompt.encode("utf-8")) > 200_000:
        raise ValueError("Prompt exceeds 200 KB.")
    (
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        device_name,
    ) = _normalize_generation_options(
        max_new_tokens, temperature, top_k, top_p, repetition_penalty, device_name
    )
    target, model, saved, config, tokenizer, device = _get_resident_bundle(path=path, device_name=device_name)
    full_prompt_ids = _encode_prompt_within_context(prompt, tokenizer, config)
    prompt_ids = full_prompt_ids
    tokens = torch.tensor([[BOS_ID, *prompt_ids]], dtype=torch.long, device=device)
    effective_top_k = max(1, min(int(top_k), config.vocab_size))
    output = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=effective_top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )[0].tolist()
    generated_ids = [tok for tok in output[len(tokens[0]):] if tok != EOS_ID]
    return {
        "text": tokenizer.decode(generated_ids),
        "prompt_tokens": len(full_prompt_ids),
        "prompt_tokens_used": len(prompt_ids),
        "prompt_truncated": len(prompt_ids) != len(full_prompt_ids),
        "context_window_tokens": config.seq_len,
        "tokenizer": "byte-BPE",
        "generated_tokens": len(generated_ids),
        "checkpoint": str(target),
        "trained_steps": int(saved.get("step", 0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "is_model_output": True,
        "warning": "Raw neural-model output. Structured patch workflows must validate it; inspect and test generated code before applying it."
    }


@_no_grad
@compute_operation("inference")
def evaluate(
    root: str | Path,
    batches: int = 16,
    device_name: str = "auto",
    max_tokens: int = 2_000_000,
    checkpoint: str | Path | None = None,
) -> dict:
    """Run a fresh sampled loss pass on a newly supplied local corpus.

    This is a new measurement, but not an independent benchmark if that corpus was
    also included in training. Task-level code benchmarks remain separate.
    """
    if type(batches) is not int or not 1 <= batches <= 1000:
        raise ValueError("batches must be an integer between 1 and 1,000.")
    if type(max_tokens) is not int or not 1024 <= max_tokens <= MAX_CORPUS_TOKENS:
        raise ValueError(f"max_tokens must be an integer between 1,024 and {MAX_CORPUS_TOKENS:,}.")
    device_name = _normalize_device_name(device_name)
    corpus_root = resolve_corpus_root(root)
    target, model, saved, config, tokenizer, device = _get_resident_bundle(path=checkpoint, device_name=device_name)
    corpus = load_corpus(
        [str(corpus_root)],
        max_tokens=max_tokens,
        include_tokens=False,
    )
    partition = _partition_source_documents(corpus.documents, max_tokens=max_tokens)
    document_split = _encode_document_partition(
        partition,
        tokenizer,
        max_tokens=max_tokens,
        seq_len=config.seq_len,
    )
    validation_data = torch.tensor(document_split.validation_tokens, dtype=torch.long)
    fresh_loss = _estimate_loss(
        model, validation_data, batch_size=1, seq_len=config.seq_len, device=device, batches=batches
    )
    return {
        "checkpoint": str(target),
        "trained_steps": int(saved.get("step", 0)),
        "evaluation_performed": True,
        "evaluation_kind": "fresh sampled next-token cross-entropy on selected local corpus",
        "validation_loss": fresh_loss,
        "perplexity": math.exp(min(fresh_loss, 80.0)),
        "corpus_files": len(corpus.files),
        "evaluation_tokens": len(document_split.validation_tokens),
        "training_documents": len(document_split.training_document_hashes),
        "validation_documents": len(document_split.validation_document_hashes),
        "duplicate_documents_excluded": document_split.duplicate_documents_excluded,
        "token_budget_documents_excluded": document_split.token_budget_documents_excluded,
        "validation_strategy": SOURCE_VALIDATION_STRATEGY,
        "batches": batches,
        "context_window_tokens": config.seq_len,
        "tokenizer": "byte-BPE",
        "note": "This measures next-token prediction, not patch correctness. If this corpus was used for training, it is not an independent benchmark.",
    }

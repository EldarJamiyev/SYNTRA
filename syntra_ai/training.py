"""Neural training loop with corpus-trained BPE tokenization and local checkpoints."""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Callable
import json
import math
import os
import random
import tempfile

try:
    import torch
    from .model import SyntraTransformer
except ImportError:
    torch = None  # type: ignore[assignment]
    SyntraTransformer = None  # type: ignore[assignment]

from .config import ModelConfig
from .checkpoint_io import load_torch_checkpoint_snapshot
from .data import MAX_CORPUS_TOKENS, load_corpus
from .resources import validate_training_memory_budget
from .tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer
from .training_history import append_training_history, validate_training_history

ProgressCallback = Callable[[dict], None]
LEGACY_SOURCE_TRAINING_SCHEDULER = "linear-warmup-v1"
SOURCE_TRAINING_SCHEDULER = "global-cosine-warmup-v2"
SOURCE_SCHEDULER_WARMUP_STEPS = 100
SOURCE_SCHEDULER_DECAY_STEPS = 10_000
SOURCE_MINIMUM_LEARNING_RATE = 1e-5
SOURCE_GRADIENT_CLIP_NORM = 1.0
SOURCE_VALIDATION_STRATEGY = "sha256-grouped-document-holdout-v1"
TRAINING_FRACTION = 0.8
VALIDATION_FRACTION = 0.2


@dataclass(frozen=True)
class DocumentPartition:
    """Deterministic, whole-document train/validation membership."""

    training_documents: tuple[str, ...]
    validation_documents: tuple[str, ...]
    training_document_hashes: tuple[str, ...]
    validation_document_hashes: tuple[str, ...]
    duplicate_documents_excluded: int
    token_budget_documents_excluded: int


@dataclass(frozen=True)
class DocumentTokenSplit:
    """Encoded whole-document token streams for causal training and validation."""

    training_tokens: list[int]
    validation_tokens: list[int]
    training_document_hashes: tuple[str, ...]
    validation_document_hashes: tuple[str, ...]
    duplicate_documents_excluded: int
    token_budget_documents_excluded: int


def _partition_source_documents(
    documents: list[str],
    *,
    max_tokens: int,
) -> DocumentPartition:
    """Deduplicate and deterministically assign complete documents to 80/20 splits.

    UTF-8 byte length is a conservative token-budget bound for the byte-BPE
    tokenizer. The tokenizer is learned only from the training partition.
    """
    if type(max_tokens) is not int or max_tokens < 4:
        raise ValueError("max_tokens must be an integer of at least four.")

    unique: dict[str, str] = {}
    duplicate_count = 0
    for document in documents:
        if not isinstance(document, str):
            raise ValueError("Every source document must be valid UTF-8 text.")
        raw = document.encode("utf-8")
        if not raw.strip():
            continue
        digest = sha256(raw).hexdigest()
        if digest in unique:
            duplicate_count += 1
        else:
            unique[digest] = document

    selected: list[tuple[str, str, int]] = []
    used_bytes = 0
    budget_excluded = 0
    for digest in sorted(unique):
        document = unique[digest]
        byte_length = len(document.encode("utf-8"))
        cost = byte_length + 2  # BOS and EOS.
        if cost > max_tokens - used_bytes:
            budget_excluded += 1
            continue
        selected.append((digest, document, byte_length))
        used_bytes += cost

    if len(selected) < 2:
        raise ValueError(
            "Document-isolated validation requires at least two independent source documents "
            "after exact-content deduplication and the configured token budget."
        )

    # Largest documents are placed first; hashes make all ties deterministic.
    ordered = sorted(selected, key=lambda item: (-item[2], item[0]))
    training: list[tuple[str, str, int]] = [ordered[0]]
    validation: list[tuple[str, str, int]] = []
    training_weight = ordered[0][2] + 2
    validation_weight = 0

    for item in ordered[1:]:
        digest, document, byte_length = item
        weight = byte_length + 2
        training_fill = training_weight / TRAINING_FRACTION
        validation_fill = validation_weight / VALIDATION_FRACTION
        if validation_fill <= training_fill:
            validation.append((digest, document, byte_length))
            validation_weight += weight
        else:
            training.append((digest, document, byte_length))
            training_weight += weight

    training.sort(key=lambda item: item[0])
    validation.sort(key=lambda item: item[0])
    return DocumentPartition(
        training_documents=tuple(item[1] for item in training),
        validation_documents=tuple(item[1] for item in validation),
        training_document_hashes=tuple(item[0] for item in training),
        validation_document_hashes=tuple(item[0] for item in validation),
        duplicate_documents_excluded=duplicate_count,
        token_budget_documents_excluded=budget_excluded,
    )


def _encode_document_partition(
    partition: DocumentPartition,
    tokenizer: ByteBPETokenizer,
    *,
    max_tokens: int,
    seq_len: int,
) -> DocumentTokenSplit:
    """Encode the already separated documents without truncating any document."""
    def encode(documents: tuple[str, ...]) -> list[int]:
        output: list[int] = []
        for document in documents:
            encoded = [BOS_ID, *tokenizer.encode(document), EOS_ID]
            if len(output) + len(encoded) > max_tokens:
                raise RuntimeError("Encoded document tokens exceeded the validated raw-byte token budget.")
            output.extend(encoded)
        return output

    training_tokens = encode(partition.training_documents)
    validation_tokens = encode(partition.validation_documents)
    if len(training_tokens) <= seq_len + 1 or len(validation_tokens) <= seq_len + 1:
        raise ValueError(
            "Document-isolated validation does not contain enough independent tokens for the configured context. "
            f"Training: {len(training_tokens)} tokens across {len(partition.training_documents)} documents; "
            f"validation: {len(validation_tokens)} tokens across {len(partition.validation_documents)} documents; "
            f"context: {seq_len}. Add independent source files or reduce the context length."
        )
    return DocumentTokenSplit(
        training_tokens=training_tokens,
        validation_tokens=validation_tokens,
        training_document_hashes=partition.training_document_hashes,
        validation_document_hashes=partition.validation_document_hashes,
        duplicate_documents_excluded=partition.duplicate_documents_excluded,
        token_budget_documents_excluded=partition.token_budget_documents_excluded,
    )


def _scheduled_learning_rate(
    base_learning_rate: float,
    global_step: int,
    minimum_learning_rate: float = SOURCE_MINIMUM_LEARNING_RATE,
    warmup_steps: int = SOURCE_SCHEDULER_WARMUP_STEPS,
    decay_steps: int = SOURCE_SCHEDULER_DECAY_STEPS,
) -> float:
    """Warm up from the configured floor, then cosine-decay to that floor."""
    for name, value in (
        ("base_learning_rate", base_learning_rate),
        ("minimum_learning_rate", minimum_learning_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(name + " must be a finite positive number.")
    if minimum_learning_rate > base_learning_rate:
        raise ValueError("minimum_learning_rate cannot exceed base_learning_rate.")
    if type(global_step) is not int or global_step < 1:
        raise ValueError("global_step must be a positive integer.")
    if type(warmup_steps) is not int or warmup_steps < 1:
        raise ValueError("warmup_steps must be a positive integer.")
    if type(decay_steps) is not int or decay_steps <= warmup_steps:
        raise ValueError("decay_steps must be greater than warmup_steps.")
    base = float(base_learning_rate)
    floor = float(minimum_learning_rate)
    if global_step <= warmup_steps:
        return floor + (base - floor) * (global_step / warmup_steps)
    progress = min(
        (global_step - warmup_steps) / (decay_steps - warmup_steps),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (base - floor) * cosine


def _legacy_scheduled_learning_rate(base_learning_rate: float, global_step: int) -> float:
    """Preserve the pre-cosine application trainer's linear warmup trajectory."""
    if (
        isinstance(base_learning_rate, bool)
        or not isinstance(base_learning_rate, (int, float))
        or not math.isfinite(base_learning_rate)
        or base_learning_rate <= 0
    ):
        raise ValueError("base_learning_rate must be a finite positive number.")
    if type(global_step) is not int or global_step < 1:
        raise ValueError("global_step must be a positive integer.")
    return float(base_learning_rate) * min(
        1.0,
        global_step / SOURCE_SCHEDULER_WARMUP_STEPS,
    )


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot find a CUDA device.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS was requested but is not available on this machine.")
    return device


def _training_data_fingerprint(corpus, max_tokens: int, config: ModelConfig, tokenizer: ByteBPETokenizer) -> str:
    """Bind a training checkpoint to the exact selected corpus, limits, config and tokenizer."""
    payload = {
        "schema": "syntra-document-heldout-source-corpus-v2",
        "validation_strategy": SOURCE_VALIDATION_STRATEGY,
        "source_sha256": corpus.source_sha256,
        "max_tokens": max_tokens,
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _rng_state(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: object, device: torch.device) -> None:
    if not isinstance(state, dict):
        raise ValueError("Resume checkpoint does not contain random-generator state.")
    cpu_state = state.get("cpu")
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() < 1
    ):
        raise ValueError("Resume checkpoint CPU random-generator state is malformed.")
    try:
        torch.set_rng_state(cpu_state.cpu())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Resume checkpoint CPU random-generator state is incompatible: {exc}") from None

    cuda_states = state.get("cuda")
    if device.type == "cuda" and torch.cuda.is_available():
        if not isinstance(cuda_states, list) or not cuda_states:
            raise ValueError("Resume checkpoint is missing CUDA random-generator state for the selected device.")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Resume checkpoint CUDA random-generator state does not match the available device count.")
        if any(
            not isinstance(item, torch.Tensor)
            or item.dtype != torch.uint8
            or item.ndim != 1
            or item.numel() < 1
            for item in cuda_states
        ):
            raise ValueError("Resume checkpoint CUDA random-generator state is malformed.")
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Resume checkpoint CUDA random-generator state is incompatible: {exc}") from None


def _batch(data, batch_size: int, seq_len: int, device):
    if data.numel() <= seq_len + 1:
        raise ValueError("The tokenized corpus split is too small for the configured context length.")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[start:start + seq_len] for start in starts])
    y = torch.stack([data[start + 1:start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)


def _estimate_loss(model, data, batch_size, seq_len, device, batches=8) -> float:
    was_training = model.training
    model.eval()
    values = []
    try:
        with torch.no_grad():
            for _ in range(batches):
                x, y = _batch(data, batch_size, seq_len, device)
                _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Validation produced a non-finite loss.")
                values.append(float(loss.item()))
    finally:
        # Evaluation may share a resident model with live inference. Restore the
        # caller's mode instead of unconditionally enabling dropout afterward.
        model.train(was_training)
    return sum(values) / len(values)


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename where the platform exposes directory fsync."""
    if os.name == "nt":
        return
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support directory fsync; file data itself is flushed.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    """Flush a complete tensor checkpoint before atomically replacing its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: dict) -> None:
    """Write status metadata atomically so readers see either old or complete new JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            content = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if os.path.exists(temporary):
            os.unlink(temporary)


INFERENCE_CHECKPOINT_FIELDS = (
    "format", "config", "tokenizer", "model_state", "step", "created_by",
    "device_at_training", "corpus_files", "corpus_tokens", "train_loss",
    "validation_loss", "training_mode", "instruction_examples", "training_history",
    "instruction_dataset_sha256", "instruction_dataset_bytes",
    "training_documents", "validation_documents", "duplicate_documents_excluded",
    "token_budget_documents_excluded", "validation_strategy", "training_settings",
    "learning_rate", "gradient_norm_before_clip", "finetuning_settings",
    "finetuning_learning_rate", "finetuning_gradient_norm_before_clip",
    "finetuning_stage_steps", "finetuning_resume_schema",
    "finetuning_optimizer_scope", "finetuning_stage_base_step",
    "finetuning_stage_target_steps", "finetuning_validation_stage_step",
    "finetuning_remaining_steps", "finetuning_resume_supported",
)


def inference_checkpoint_path(checkpoint: str | Path) -> Path:
    target = Path(checkpoint).expanduser().absolute()
    return target.with_name(target.stem + ".inference" + target.suffix)


def select_inference_checkpoint(checkpoint: str | Path) -> Path:
    """Select a recent companion candidate; runtime validates its source-file identity."""
    target = Path(checkpoint).expanduser().absolute()
    companion = inference_checkpoint_path(target)
    try:
        if companion.is_file() and companion.stat().st_mtime_ns >= target.stat().st_mtime_ns:
            return companion
    except OSError:
        pass
    return target


def _save_inference_checkpoint(checkpoint: str | Path, payload: dict) -> Path:
    target = Path(checkpoint).expanduser().absolute()
    companion = inference_checkpoint_path(target)
    inference_payload = {
        key: payload[key]
        for key in INFERENCE_CHECKPOINT_FIELDS
        if key in payload
    }
    source_stat = target.stat()
    inference_payload["source_checkpoint_mtime_ns"] = source_stat.st_mtime_ns
    inference_payload["source_checkpoint_identity"] = {
        "device": source_stat.st_dev,
        "inode": source_stat.st_ino,
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "ctime_ns": source_stat.st_ctime_ns,
    }
    _atomic_checkpoint(companion, inference_payload)
    return companion


def _encode_documents(documents: list[str], tokenizer: ByteBPETokenizer, max_tokens: int) -> list[int]:
    tokens: list[int] = []
    for document in documents:
        encoded = [BOS_ID, *tokenizer.encode(document), EOS_ID]
        remaining = max_tokens - len(tokens)
        if remaining <= 0:
            break
        tokens.extend(encoded[:remaining])
    if not tokens:
        raise ValueError("The selected corpus produced no tokens with this tokenizer.")
    return tokens


def _same_architecture(requested: ModelConfig, stored: ModelConfig) -> bool:
    # Compare the complete canonical configuration. Omitting vocabulary or
    # architectural flags can silently ignore a requested resume configuration.
    return requested.to_dict() == stored.to_dict()


def train(
    roots: list[str],
    checkpoint: str | Path,
    steps: int = 1000,
    batch_size: int = 4,
    config: ModelConfig | None = None,
    learning_rate: float = 3e-4,
    max_tokens: int = 2_000_000,
    device_name: str = "auto",
    seed: int = 1337,
    resume: bool = False,
    callback: ProgressCallback | None = None,
) -> dict:
    if steps < 1 or steps > 1_000_000:
        raise ValueError("steps must be between 1 and 1,000,000.")
    if batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size must be between 1 and 128.")
    if not 1e-6 <= learning_rate <= 1e-2:
        raise ValueError("learning_rate must be between 1e-6 and 1e-2.")
    if max_tokens < 1024 or max_tokens > MAX_CORPUS_TOKENS:
        raise ValueError(f"max_tokens must be between 1,024 and {MAX_CORPUS_TOKENS:,}.")
    requested_config = config
    effective_scheduler = SOURCE_TRAINING_SCHEDULER
    effective_minimum_learning_rate = min(
        SOURCE_MINIMUM_LEARNING_RATE,
        float(learning_rate),
    )
    effective_warmup_steps = SOURCE_SCHEDULER_WARMUP_STEPS
    effective_decay_steps = SOURCE_SCHEDULER_DECAY_STEPS
    effective_gradient_clip_norm = SOURCE_GRADIENT_CLIP_NORM
    checkpoint_path = Path(checkpoint).expanduser().absolute()
    device = choose_device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    saved = None
    corpus = None
    if resume and checkpoint_path.exists():
        # The snapshot loader binds the hard byte ceiling and available-RAM budget
        # to the same descriptor whose verified bytes are passed to torch.load().
        saved = load_torch_checkpoint_snapshot(checkpoint_path)
        if saved.get("format") != "syntra-transformer-v1":
            raise ValueError("Resume checkpoint is not a supported SYNTRA Transformer checkpoint.")
        stored_config = ModelConfig.from_dict(saved["config"])
        if requested_config is not None:
            saved_requested_config = saved.get("requested_config")
            if isinstance(saved_requested_config, dict):
                try:
                    original_request = ModelConfig.from_dict(saved_requested_config)
                except (TypeError, ValueError):
                    raise ValueError("Resume checkpoint contains an invalid original model configuration.") from None
                if not _same_architecture(requested_config, original_request):
                    raise ValueError("Resume architecture differs from the original requested checkpoint configuration.")
            else:
                # Older checkpoints only stored the tokenizer's realized vocabulary.
                # Compare every other architectural field without treating the
                # corpus-dependent BPE vocabulary reduction as a user config change.
                comparable_request = replace(requested_config, vocab_size=stored_config.vocab_size)
                if not _same_architecture(comparable_request, stored_config):
                    raise ValueError("Resume architecture differs from the saved checkpoint configuration.")
        requested_config_payload = saved.get("requested_config")
        if not isinstance(requested_config_payload, dict):
            requested_config_payload = None
        config = stored_config
        config.validate()
        validate_training_memory_budget(config, batch_size, device)
        corpus = load_corpus(roots, max_tokens=max_tokens, include_tokens=False)
        document_partition = _partition_source_documents(corpus.documents, max_tokens=max_tokens)
        tokenizer = ByteBPETokenizer.from_dict(saved.get("tokenizer"))
        if tokenizer.vocab_size != config.vocab_size:
            if "tokenizer" in saved:
                raise ValueError("Checkpoint tokenizer vocabulary does not match the model architecture.")
            # Legacy byte-only checkpoints remain loadable for continued byte training.
            tokenizer = ByteBPETokenizer.empty()
    else:
        config = requested_config or ModelConfig()
        config.validate()
        requested_config_payload = config.to_dict()
        validate_training_memory_budget(config, batch_size, device)
        corpus = load_corpus(roots, max_tokens=max_tokens, include_tokens=False)
        document_partition = _partition_source_documents(corpus.documents, max_tokens=max_tokens)
        # The validation documents must not influence the learned BPE vocabulary.
        tokenizer = ByteBPETokenizer.train(
            document_partition.training_documents,
            vocab_size=config.vocab_size,
        )
        config = replace(config, vocab_size=tokenizer.vocab_size)

    config.validate()
    document_split = _encode_document_partition(
        document_partition,
        tokenizer,
        max_tokens=max_tokens,
        seq_len=config.seq_len,
    )
    corpus_tokens = len(document_split.training_tokens) + len(document_split.validation_tokens)
    training_data_sha256 = _training_data_fingerprint(corpus, max_tokens, config, tokenizer)
    training_history = validate_training_history(saved.get("training_history") if saved is not None else None)
    if saved is not None:
        if saved.get("training_mode") != "causal-source-pretraining":
            raise ValueError(
                "The checkpoint was not produced by causal source pretraining. "
                "Resume it with its matching training workflow or start a new pretraining checkpoint."
            )
        if saved.get("training_data_sha256") != training_data_sha256:
            raise ValueError(
                "The selected source corpus, token budget, tokenizer, or document-isolated validation "
                "strategy differs from the checkpoint's training corpus. Select the original source set and settings or "
                "start a new checkpoint; token-offset validation checkpoints are not compatible."
            )
        if not isinstance(saved.get("optimizer_state"), dict):
            raise ValueError("Resume checkpoint is missing its optimizer state.")
        if not isinstance(saved.get("rng_state"), dict):
            raise ValueError("Resume checkpoint is missing its random-generator state.")
        saved_settings = saved.get("training_settings")
        required_settings = {"batch_size", "learning_rate", "seed", "max_tokens"}
        if not isinstance(saved_settings, dict) or not required_settings.issubset(saved_settings):
            raise ValueError("Resume checkpoint is missing complete training settings; start a new run.")
        expected_settings = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        changed_settings = [
            name
            for name, expected in expected_settings.items()
            if type(saved_settings.get(name)) not in (int, float)
            or saved_settings.get(name) != expected
        ]
        if changed_settings:
            raise ValueError(
                "Training resume settings differ from the checkpoint: "
                + ", ".join(changed_settings)
                + ". Use the original settings or start a new checkpoint."
            )

        # Pre-scheduler application checkpoints used a linear 100-step warmup
        # followed by a constant rate. Preserve that trajectory on resume.
        saved_scheduler = saved_settings.get(
            "scheduler",
            LEGACY_SOURCE_TRAINING_SCHEDULER,
        )
        if saved_scheduler == LEGACY_SOURCE_TRAINING_SCHEDULER:
            effective_scheduler = LEGACY_SOURCE_TRAINING_SCHEDULER
            effective_minimum_learning_rate = None
        elif saved_scheduler == SOURCE_TRAINING_SCHEDULER:
            required_scheduler_settings = {
                "minimum_learning_rate",
                "warmup_steps",
                "decay_steps",
                "gradient_clip_norm",
            }
            if not required_scheduler_settings.issubset(saved_settings):
                raise ValueError("The cosine-scheduler checkpoint is missing scheduler or gradient-clip settings.")
            expected_scheduler_settings = {
                "minimum_learning_rate": effective_minimum_learning_rate,
                "warmup_steps": effective_warmup_steps,
                "decay_steps": effective_decay_steps,
                "gradient_clip_norm": effective_gradient_clip_norm,
            }
            changed_scheduler_settings = [
                name
                for name, expected in expected_scheduler_settings.items()
                if (
                    type(saved_settings.get(name)) not in (int, float)
                    or isinstance(saved_settings.get(name), bool)
                    or saved_settings.get(name) != expected
                )
            ]
            if changed_scheduler_settings:
                raise ValueError(
                    "Training resume settings differ from the checkpoint: "
                    + ", ".join(changed_scheduler_settings)
                    + ". Use the original settings or start a new checkpoint."
                )
            effective_scheduler = SOURCE_TRAINING_SCHEDULER
        else:
            raise ValueError("Checkpoint contains an unsupported source-training scheduler; start a new run.")

    train_data = torch.tensor(document_split.training_tokens, dtype=torch.long)
    val_data = torch.tensor(document_split.validation_tokens, dtype=torch.long)
    start_step = 0
    model = SyntraTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    if saved is not None:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        _restore_rng_state(saved["rng_state"], device)
        start_step = int(saved.get("step", 0))

    started = monotonic()
    last_train_loss = None
    last_val_loss = None
    last_learning_rate = None
    last_gradient_norm = None

    def report(status: str, step: int, message: str = ""):
        current = {
            "status": status,
            "step": step,
            "max_steps": start_step + steps,
            "train_loss": last_train_loss,
            "validation_loss": last_val_loss,
            "learning_rate": last_learning_rate,
            "training_scheduler": effective_scheduler,
            "minimum_learning_rate": effective_minimum_learning_rate,
            "gradient_clip_norm": effective_gradient_clip_norm,
            "gradient_norm_before_clip": last_gradient_norm,
            "device": str(device),
            "elapsed_seconds": round(monotonic() - started, 2),
            "message": message,
            "checkpoint": str(checkpoint_path),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "corpus_files": len(corpus.files),
            "corpus_tokens": corpus_tokens,
            "training_documents": len(document_split.training_document_hashes),
            "validation_documents": len(document_split.validation_document_hashes),
            "duplicate_documents_excluded": document_split.duplicate_documents_excluded,
            "token_budget_documents_excluded": document_split.token_budget_documents_excluded,
            "validation_strategy": SOURCE_VALIDATION_STRATEGY,
            "tokenizer": "byte-BPE",
            "vocabulary_size": tokenizer.vocab_size,
            "context_tokens": config.seq_len,
            "training_history_points": len(training_history),
        }
        if callback:
            callback(current)
        return current

    report(
        "running",
        start_step,
        "Training randomly initialized Transformer weights with train-only byte-BPE and document-isolated validation."
        if start_step == 0 else "Resuming the saved BPE tokenizer/model/optimizer checkpoint.",
    )
    model.train()
    for step in range(start_step + 1, start_step + steps + 1):
        if effective_scheduler == SOURCE_TRAINING_SCHEDULER:
            learning_rate_now = _scheduled_learning_rate(
                learning_rate,
                step,
                minimum_learning_rate=effective_minimum_learning_rate,
                warmup_steps=effective_warmup_steps,
                decay_steps=effective_decay_steps,
            )
        else:
            learning_rate_now = _legacy_scheduled_learning_rate(learning_rate, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_now
        last_learning_rate = learning_rate_now
        x, y = _batch(train_data, batch_size, config.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            effective_gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm).item():
            raise RuntimeError("Gradient norm became non-finite before clipping.")
        last_gradient_norm = float(gradient_norm.item())
        optimizer.step()
        last_train_loss = float(loss.item())

        if step == start_step + 1 or step % 25 == 0 or step == start_step + steps:
            last_val_loss = _estimate_loss(model, val_data, batch_size, config.seq_len, device, batches=4)
            training_history = append_training_history(
                training_history,
                step=step,
                phase="causal-source-pretraining",
                train_loss=last_train_loss,
                validation_loss=last_val_loss,
            )
            payload = {
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "created_by": "SYNTRA from-scratch trainer",
                "device_at_training": str(device),
                "corpus_files": len(corpus.files),
                "corpus_tokens": corpus_tokens,
                "training_documents": len(document_split.training_document_hashes),
                "validation_documents": len(document_split.validation_document_hashes),
                "duplicate_documents_excluded": document_split.duplicate_documents_excluded,
                "token_budget_documents_excluded": document_split.token_budget_documents_excluded,
                "validation_strategy": SOURCE_VALIDATION_STRATEGY,
                "training_data_sha256": training_data_sha256,
                "requested_config": requested_config_payload or config.to_dict(),
                "learning_rate": last_learning_rate,
                "gradient_norm_before_clip": last_gradient_norm,
                "training_settings": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "scheduler": effective_scheduler,
                    "minimum_learning_rate": effective_minimum_learning_rate,
                    "warmup_steps": effective_warmup_steps,
                    "decay_steps": effective_decay_steps,
                    "gradient_clip_norm": effective_gradient_clip_norm,
                },
                "rng_state": _rng_state(device),
                "train_loss": last_train_loss,
                "validation_loss": last_val_loss,
                "training_mode": "causal-source-pretraining",
                "training_history": training_history,
            }
            _atomic_checkpoint(checkpoint_path, payload)
            _save_inference_checkpoint(checkpoint_path, payload)
            metadata = {
                "checkpoint": str(checkpoint_path),
                "step": step,
                "config": config.to_dict(),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "train_loss": last_train_loss,
                "validation_loss": last_val_loss,
                "learning_rate": last_learning_rate,
                "training_scheduler": effective_scheduler,
                "minimum_learning_rate": effective_minimum_learning_rate,
                "warmup_steps": effective_warmup_steps,
                "decay_steps": effective_decay_steps,
                "gradient_clip_norm": effective_gradient_clip_norm,
                "gradient_norm_before_clip": last_gradient_norm,
                "corpus_files": len(corpus.files),
                "corpus_tokens": corpus_tokens,
                "training_documents": len(document_split.training_document_hashes),
                "validation_documents": len(document_split.validation_document_hashes),
                "duplicate_documents_excluded": document_split.duplicate_documents_excluded,
                "token_budget_documents_excluded": document_split.token_budget_documents_excluded,
                "validation_strategy": SOURCE_VALIDATION_STRATEGY,
                "tokenizer": "byte-BPE",
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "context_tokens": config.seq_len,
                "created_by": "SYNTRA from-scratch trainer",
                "pretrained_weights": False,
                "training_history": training_history,
                "training_history_points": len(training_history),
            }
            metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
            _atomic_json(metadata_path, metadata)
            report(
                "running" if step < start_step + steps else "completed",
                step,
                "Checkpoint saved; document-isolated held-out BPE-token validation loss measured.",
            )

    return report("completed", start_step + steps, "Training finished. Model, tokenizer, optimizer, and measured metrics saved locally.")

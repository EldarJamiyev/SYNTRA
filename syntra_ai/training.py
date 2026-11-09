"""Neural training loop with corpus-trained BPE tokenization and local checkpoints."""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Callable
import json
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
from .data import MAX_CORPUS_TOKENS, load_corpus
from .resources import validate_checkpoint_load_budget, validate_training_memory_budget
from .tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer
from .training_history import append_training_history, validate_training_history

ProgressCallback = Callable[[dict], None]


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
        "schema": "syntra-causal-source-corpus-v1",
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
    "validation_loss", "training_mode", "instruction_examples", "training_history"
)


def inference_checkpoint_path(checkpoint: str | Path) -> Path:
    target = Path(checkpoint).expanduser().resolve()
    return target.with_name(target.stem + ".inference" + target.suffix)


def select_inference_checkpoint(checkpoint: str | Path) -> Path:
    """Use the lightweight companion only if it was written after the full checkpoint."""
    target = Path(checkpoint).expanduser().resolve()
    companion = inference_checkpoint_path(target)
    try:
        if companion.is_file() and companion.stat().st_mtime_ns >= target.stat().st_mtime_ns:
            return companion
    except OSError:
        pass
    return target


def _save_inference_checkpoint(checkpoint: str | Path, payload: dict) -> Path:
    target = Path(checkpoint).expanduser().resolve()
    companion = inference_checkpoint_path(target)
    inference_payload = {
        key: payload[key]
        for key in INFERENCE_CHECKPOINT_FIELDS
        if key in payload
    }
    inference_payload["source_checkpoint_mtime_ns"] = target.stat().st_mtime_ns
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
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    device = choose_device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    saved = None
    corpus = None
    if resume and checkpoint_path.exists():
        # Bound the on-disk checkpoint before deserializing its tensor and optimizer state.
        validate_checkpoint_load_budget(checkpoint_path)
        # Load checkpoint tensors on CPU first so the training memory preflight can run before
        # copying model and optimizer state onto a potentially constrained device.
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        corpus = load_corpus(roots, max_tokens=max_tokens)
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
        corpus = load_corpus(roots, max_tokens=max_tokens)
        tokenizer = ByteBPETokenizer.train(corpus.documents, vocab_size=config.vocab_size)
        config = replace(config, vocab_size=tokenizer.vocab_size)

    config.validate()
    if corpus is None:
        corpus = load_corpus(roots, max_tokens=max_tokens)
    token_ids = _encode_documents(corpus.documents, tokenizer, max_tokens=max_tokens)
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
                "The selected source corpus, token budget, or tokenizer differs from the checkpoint's "
                "training corpus. Select the original source set and settings or start a new checkpoint."
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
    split = int(len(token_ids) * 0.8)
    if split <= config.seq_len + 1 or len(token_ids) - split <= config.seq_len + 1:
        raise ValueError(
            f"Corpus produced {len(token_ids)} BPE tokens. With context={config.seq_len}, "
            f"the 80/20 split needs more than {5 * (config.seq_len + 1)} tokens. "
            "Use a larger local source corpus, reduce --seq-len, or lower the UI context setting."
        )
    train_data = torch.tensor(token_ids[:split], dtype=torch.long)
    val_data = torch.tensor(token_ids[split:], dtype=torch.long)
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

    def report(status: str, step: int, message: str = ""):
        current = {
            "status": status,
            "step": step,
            "max_steps": start_step + steps,
            "train_loss": last_train_loss,
            "validation_loss": last_val_loss,
            "device": str(device),
            "elapsed_seconds": round(monotonic() - started, 2),
            "message": message,
            "checkpoint": str(checkpoint_path),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "corpus_files": len(corpus.files),
            "corpus_tokens": len(token_ids),
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
        "Training randomly initialized Transformer weights with corpus-trained byte-BPE tokens."
        if start_step == 0 else "Resuming the saved BPE tokenizer/model/optimizer checkpoint.",
    )
    model.train()
    for step in range(start_step + 1, start_step + steps + 1):
        learning_rate_now = learning_rate * min(1.0, step / 100.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_now
        x, y = _batch(train_data, batch_size, config.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                "corpus_tokens": len(token_ids),
                "training_data_sha256": training_data_sha256,
                "requested_config": requested_config_payload or config.to_dict(),
                "training_settings": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "max_tokens": max_tokens,
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
                "corpus_files": len(corpus.files),
                "corpus_tokens": len(token_ids),
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
                "Checkpoint saved; sampled held-out BPE-token validation loss measured.",
            )

    return report("completed", start_step + steps, "Training finished. Model, tokenizer, optimizer, and measured metrics saved locally.")

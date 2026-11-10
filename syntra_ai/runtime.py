"""Checkpoint-backed tokenizer, resident inference cache, context counting, embeddings, and fresh evaluation."""
from __future__ import annotations
from codecs import getincrementaldecoder
from pathlib import Path
from threading import RLock
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
from .checkpoint_io import load_torch_checkpoint_snapshot
from .tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer
from .training import (
    _batch, _encode_documents, _estimate_loss, choose_device, select_inference_checkpoint,
)
from .data import MAX_CORPUS_TOKENS, load_corpus
from .resources import validate_checkpoint_load_budget, validate_inference_memory_budget
from .safe_paths import resolve_corpus_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "syntra.pt"
FALLBACK_BRAIN_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "syntra-brain.pt"

_CACHE_LOCK = RLock()
_RESIDENT_CACHE: dict[str, tuple[tuple[Any, ...], Any, dict, ModelConfig, ByteBPETokenizer, Any]] = {}


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
    info = _checkpoint_stat(path)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


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


def _read_checkpoint(path: str | Path | None = None) -> tuple[Path, dict, ModelConfig, ByteBPETokenizer]:
    target = Path(path).expanduser().absolute() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")
    load_target = select_inference_checkpoint(target)

    def load_validated(source: Path) -> tuple[dict, ModelConfig, ByteBPETokenizer]:
        validate_checkpoint_load_budget(source)
        raw_payload = load_torch_checkpoint_snapshot(source)
        payload = _normalize_checkpoint_payload(raw_payload)
        config = ModelConfig.from_dict(payload["config"])
        tokenizer = ByteBPETokenizer.from_dict(payload.get("tokenizer"))
        if tokenizer.vocab_size != config.vocab_size:
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


def _get_resident_bundle(path: str | Path | None = None, device_name: str = "auto") -> tuple[Path, SyntraTransformer, dict, ModelConfig, ByteBPETokenizer, torch.device]:
    target = Path(path).expanduser().absolute() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")
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
        signature = ((-1, -1, -1, -1, -1), str(load_target), (-1, -1, -1, -1, -1), str(device_name))

    cache_key = str(target)
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
    with _CACHE_LOCK:
        if len(_RESIDENT_CACHE) >= 4:
            oldest_key = next(iter(_RESIDENT_CACHE))
            _RESIDENT_CACHE.pop(oldest_key, None)
        _RESIDENT_CACHE[cache_key] = (signature, model, saved, config, tokenizer, device)
    return target, model, saved, config, tokenizer, device


def clear_resident_cache() -> None:
    with _CACHE_LOCK:
        _RESIDENT_CACHE.clear()


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
    try:
        _checkpoint_stat(path)
        checkpoint_exists = True
    except OSError:
        checkpoint_exists = False
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
        "trained_steps": int(metadata.get("step", 0)),
        "parameter_count": int(metadata.get("parameter_count", 0)),
        "config": raw_config if raw_config else None,
        "train_loss": metadata.get("train_loss"),
        "validation_loss": metadata.get("validation_loss"),
        "best_validation_loss": metadata.get("best_validation_loss"),
        "training_history": training_history,
        "training_history_points": len(training_history),
        "training_history_valid": history_error is None,
        "training_history_error": history_error,
        "corpus_files": metadata.get("corpus_files", len(metadata.get("dataset_files", [])) if isinstance(metadata.get("dataset_files"), list) else 0),
        "corpus_tokens": metadata.get("corpus_tokens", 0),
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
    target, _, saved, config, tokenizer, _ = _get_resident_bundle(path=path, device_name="cpu")
    count = len(tokenizer.encode(text))
    return {
        "token_count": count,
        "context_window_tokens": config.seq_len,
        "vocabulary_size": tokenizer.vocab_size,
        "tokenizer": "byte-BPE",
        "tokens_over_context": max(0, count - (config.seq_len - 1)),
        "checkpoint": str(target),
        "trained_steps": int(saved.get("step", 0)),
    }


@_no_grad
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


@_no_grad
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
    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096.")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")
    target, model, saved, config, tokenizer, device = _get_resident_bundle(path=path, device_name=device_name)
    full_prompt_ids = _encode_prompt_within_context(prompt, tokenizer, config)
    prompt_ids = full_prompt_ids
    tokens = torch.tensor([[BOS_ID, *prompt_ids]], dtype=torch.long, device=device)
    token_bytes = _token_bytes_table(tokenizer)
    decoder = getincrementaldecoder("utf-8")("replace")
    generated_ids: list[int] = []
    full_text_pieces: list[str] = []

    context = tokens[:, -config.seq_len:]
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
            logits, _, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        else:
            context = tokens[:, -config.seq_len:]
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
    if batches < 1 or batches > 1000:
        raise ValueError("batches must be between 1 and 1,000.")
    if max_tokens < 1024 or max_tokens > MAX_CORPUS_TOKENS:
        raise ValueError(f"max_tokens must be between 1,024 and {MAX_CORPUS_TOKENS:,}.")
    corpus_root = resolve_corpus_root(root)
    target, model, saved, config, tokenizer, device = _get_resident_bundle(path=checkpoint, device_name=device_name)
    corpus = load_corpus([str(corpus_root)], max_tokens=max_tokens)
    token_ids = _encode_documents(corpus.documents, tokenizer, max_tokens=max_tokens)
    split = int(len(token_ids) * 0.8)
    if split <= config.seq_len + 1 or len(token_ids) - split <= config.seq_len + 1:
        raise ValueError(
            f"The selected corpus produced {len(token_ids)} tokens. A fresh 80/20 evaluation "
            f"with context={config.seq_len} needs more than {5 * (config.seq_len + 1)} tokens."
        )
    validation_data = torch.tensor(token_ids[split:], dtype=torch.long)
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
        "evaluation_tokens": len(token_ids) - split,
        "batches": batches,
        "context_window_tokens": config.seq_len,
        "tokenizer": "byte-BPE",
        "note": "This measures next-token prediction, not patch correctness. If this corpus was used for training, it is not an independent benchmark.",
    }

"""Checkpoint-backed tokenizer, inference, context counting, and fresh evaluation."""
from pathlib import Path
import json
import math
import os

import torch

from .config import ModelConfig
from .model import BOS_ID, SyntraTransformer
from .tokenizer import ByteBPETokenizer
from .training import (
    _batch, _encode_documents, _estimate_loss, choose_device, select_inference_checkpoint,
)
from .data import MAX_CORPUS_TOKENS, load_corpus
from .resources import validate_checkpoint_load_budget, validate_inference_memory_budget

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "syntra.pt"


def checkpoint_path() -> Path:
    return Path(os.environ.get("SYNTRA_CHECKPOINT", str(DEFAULT_CHECKPOINT))).expanduser().resolve()


def _read_checkpoint(path: str | Path | None = None) -> tuple[Path, dict, ModelConfig, ByteBPETokenizer]:
    target = Path(path).expanduser().resolve() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")
    load_target = select_inference_checkpoint(target)

    def load_validated(source: Path) -> tuple[dict, ModelConfig, ByteBPETokenizer]:
        validate_checkpoint_load_budget(source)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if payload.get("format") != "syntra-transformer-v1":
            raise ValueError("The selected file is not a SYNTRA Transformer checkpoint.")
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
        try:
            source_mtime = target.stat().st_mtime_ns
        except OSError:
            source_mtime = -1
        if saved.get("source_checkpoint_mtime_ns") != source_mtime:
            saved, config, tokenizer = load_validated(target)
    return target, saved, config, tokenizer


def status() -> dict:
    path = checkpoint_path()
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    checkpoint_exists = path.is_file()
    return {
        "service": "SYNTRA native local Transformer",
        "backend": "PyTorch",
        "checkpoint": str(path),
        "checkpoint_exists": checkpoint_exists,
        "trained_steps": int(metadata.get("step", 0)),
        "parameter_count": int(metadata.get("parameter_count", 0)),
        "config": metadata.get("config"),
        "train_loss": metadata.get("train_loss"),
        "validation_loss": metadata.get("validation_loss"),
        "corpus_files": metadata.get("corpus_files", 0),
        "corpus_tokens": metadata.get("corpus_tokens", 0),
        "tokenizer": metadata.get("tokenizer", "byte-BPE" if checkpoint_exists else None),
        "tokenizer_vocab_size": metadata.get("tokenizer_vocab_size", (metadata.get("config") or {}).get("vocab_size")),
        "context_tokens": metadata.get("context_tokens", (metadata.get("config") or {}).get("seq_len")),
        "training_mode": metadata.get("training_mode", "causal-source-pretraining"),
        "pretrained_weights": False,
        "external_ai_api": False,
        "note": "Local byte-BPE Transformer. Generation requires a trained checkpoint; reported loss is not a coding benchmark."
    }


def load_model(path: str | Path | None = None, device_name: str = "auto"):
    target, saved, config, tokenizer = _read_checkpoint(path)
    device = choose_device(device_name)
    validate_inference_memory_budget(
        config, batch_size=1, sequence_length=config.seq_len, device=device
    )
    model = SyntraTransformer(config).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved, device


def tokenize_text(text: str, path: str | Path | None = None) -> dict:
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")
    if len(text.encode("utf-8")) > 300_000:
        raise ValueError("Text exceeds the 300 KB token-count API limit.")
    target, saved, config, tokenizer = _read_checkpoint(path)
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


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    device_name: str = "auto",
) -> dict:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("A non-empty text prompt is required.")
    if len(prompt.encode("utf-8")) > 200_000:
        raise ValueError("Prompt exceeds 200 KB.")
    model, saved, device = load_model(device_name=device_name)
    config = ModelConfig.from_dict(saved["config"])
    tokenizer = ByteBPETokenizer.from_dict(saved.get("tokenizer"))
    full_prompt_ids = tokenizer.encode(prompt)
    # Keep BOS plus the most recent prompt context; report truncation rather than hiding it.
    prompt_limit = config.seq_len - 1
    prompt_ids = full_prompt_ids[-prompt_limit:] if len(full_prompt_ids) > prompt_limit else full_prompt_ids
    tokens = torch.tensor([[BOS_ID, *prompt_ids]], dtype=torch.long, device=device)
    output = model.generate(tokens, max_new_tokens, temperature, top_k)[0].tolist()
    generated_ids = output[len(tokens[0]):]
    return {
        "text": tokenizer.decode(generated_ids),
        "prompt_tokens": len(full_prompt_ids),
        "prompt_tokens_used": len(prompt_ids),
        "prompt_truncated": len(prompt_ids) != len(full_prompt_ids),
        "context_window_tokens": config.seq_len,
        "tokenizer": "byte-BPE",
        "generated_tokens": len(generated_ids),
        "checkpoint": str(checkpoint_path()),
        "trained_steps": int(saved.get("step", 0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "is_model_output": True,
        "warning": "Raw neural-model output. Structured patch workflows must validate it; inspect and test generated code before applying it."
    }


@torch.no_grad()
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
    model, saved, device = load_model(path=checkpoint, device_name=device_name)
    config = ModelConfig.from_dict(saved["config"])
    tokenizer = ByteBPETokenizer.from_dict(saved.get("tokenizer"))
    corpus = load_corpus([str(Path(root).expanduser().resolve())], max_tokens=max_tokens)
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
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint else str(checkpoint_path()),
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

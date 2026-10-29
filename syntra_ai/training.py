"""Neural training loop with corpus-trained BPE tokenization and local checkpoints."""
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Callable
import json
import os
import random
import tempfile

import torch

from .config import ModelConfig
from .data import MAX_CORPUS_TOKENS, load_corpus
from .model import BOS_ID, EOS_ID, SyntraTransformer
from .resources import validate_training_memory_budget
from .tokenizer import ByteBPETokenizer

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


def _batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    if data.numel() <= seq_len + 1:
        raise ValueError("The tokenized corpus split is too small for the configured context length.")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[start:start + seq_len] for start in starts])
    y = torch.stack([data[start + 1:start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(model, data, batch_size, seq_len, device, batches=8) -> float:
    model.eval()
    values = []
    for _ in range(batches):
        x, y = _batch(data, batch_size, seq_len, device)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Validation produced a non-finite loss.")
        values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    return (
        requested.seq_len == stored.seq_len
        and requested.n_layer == stored.n_layer
        and requested.n_head == stored.n_head
        and requested.n_embd == stored.n_embd
        and requested.dropout == stored.dropout
    )


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
        # Load checkpoint tensors on CPU first so the memory preflight can run before
        # copying model and optimizer state onto a potentially constrained device.
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("format") != "syntra-transformer-v1":
            raise ValueError("Resume checkpoint is not a supported SYNTRA Transformer checkpoint.")
        stored_config = ModelConfig.from_dict(saved["config"])
        if requested_config is not None and not _same_architecture(requested_config, stored_config):
            raise ValueError("Resume architecture differs from the saved checkpoint configuration.")
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
        validate_training_memory_budget(config, batch_size, device)
        corpus = load_corpus(roots, max_tokens=max_tokens)
        tokenizer = ByteBPETokenizer.train(corpus.documents, vocab_size=config.vocab_size)
        config = replace(config, vocab_size=tokenizer.vocab_size)

    config.validate()
    if corpus is None:
        corpus = load_corpus(roots, max_tokens=max_tokens)
    token_ids = _encode_documents(corpus.documents, tokenizer, max_tokens=max_tokens)
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
        if saved.get("optimizer_state"):
            optimizer.load_state_dict(saved["optimizer_state"])
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
                "train_loss": last_train_loss,
                "validation_loss": last_val_loss,
                "training_mode": "causal-source-pretraining",
            }
            _atomic_checkpoint(checkpoint_path, payload)
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
            }
            metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            report(
                "running" if step < start_step + steps else "completed",
                step,
                "Checkpoint saved; sampled held-out BPE-token validation loss measured.",
            )

    return report("completed", start_step + steps, "Training finished. Model, tokenizer, optimizer, and measured metrics saved locally.")

"""Actual gradient-descent training and checkpoint management."""
from pathlib import Path
from time import monotonic
from typing import Callable
import json
import os
import random
import tempfile

import torch

from .config import ModelConfig
from .data import load_corpus
from .model import SyntraTransformer

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
        raise ValueError("The corpus split is too small for the configured context length.")
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


def train(
    roots: list[str],
    checkpoint: str | Path,
    steps: int = 1000,
    batch_size: int = 8,
    config: ModelConfig | None = None,
    learning_rate: float = 3e-4,
    max_tokens: int = 5_000_000,
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
    config = config or ModelConfig()
    config.validate()
    device = choose_device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    corpus = load_corpus(roots, max_tokens=max_tokens)
    split = int(len(corpus.token_ids) * 0.8)
    if split <= config.seq_len + 1 or len(corpus.token_ids) - split <= config.seq_len + 1:
        raise ValueError(
            f"Corpus has {len(corpus.token_ids)} tokens; the 80/20 split must leave at least "
            f"{config.seq_len + 2} tokens on each side. Add more source files or lower --seq-len."
        )
    train_data = torch.tensor(corpus.token_ids[:split], dtype=torch.long)
    val_data = torch.tensor(corpus.token_ids[split:], dtype=torch.long)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    start_step = 0
    model = SyntraTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        saved_config = ModelConfig.from_dict(saved["config"])
        if saved_config != config:
            raise ValueError("Resume configuration does not match the checkpoint architecture.")
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
            "corpus_tokens": len(corpus.token_ids),
        }
        if callback:
            callback(current)
        return current

    report("running", start_step, "Training neural transformer weights from random initialization." if start_step == 0 else "Resuming local checkpoint training.")
    model.train()
    for step in range(start_step + 1, start_step + steps + 1):
        learning_rate_now = learning_rate * min(1.0, step / 100.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_now
        x, y = _batch(train_data, batch_size, config.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_loss = float(loss.item())

        if step == start_step + 1 or step % 25 == 0 or step == start_step + steps:
            last_val_loss = _estimate_loss(model, val_data, batch_size, config.seq_len, device, batches=4)
            payload = {
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "created_by": "SYNTRA from-scratch trainer",
                "device_at_training": str(device),
                "corpus_files": len(corpus.files),
                "corpus_tokens": len(corpus.token_ids),
                "train_loss": last_train_loss,
                "validation_loss": last_val_loss,
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
                "corpus_tokens": len(corpus.token_ids),
                "created_by": "SYNTRA from-scratch trainer",
                "pretrained_weights": False,
            }
            metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            report("running" if step < start_step + steps else "completed", step, "Checkpoint saved; validation loss measured on held-out source bytes.")

    final = report("completed", start_step + steps, "Training finished. Checkpoint and measured metrics saved locally.")
    return final

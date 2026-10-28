"""Training from random initialization with validation and resumable checkpoints."""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from syntra.config import SyntraConfig
from syntra.data import read_token_stream
from syntra.model import SyntraModel


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Use device='auto' or 'cpu'. "
                "The integrated Intel Arc GPU is not supported by this CUDA path."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_batch(
    data: torch.Tensor, config: SyntraConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - config.block_size - 1
    if max_start < 1:
        raise ValueError(
            "Not enough tokens for this block_size. Add data or reduce block_size."
        )
    starts = torch.randint(0, max_start, (config.batch_size,))
    inputs = torch.stack(
        [data[int(start) : int(start) + config.block_size] for start in starts]
    )
    targets = torch.stack(
        [
            data[int(start) + 1 : int(start) + config.block_size + 1]
            for start in starts
        ]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: SyntraModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: SyntraConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for label, data in (("train", train_data), ("validation", val_data)):
        losses = []
        for _ in range(config.eval_batches):
            x, y = _get_batch(data, config, device)
            _, loss = model(x, y)
            if loss is not None:
                losses.append(float(loss.item()))
        result[label] = sum(losses) / max(1, len(losses))
    model.train()
    return result


def _save_checkpoint(
    path: Path,
    model: SyntraModel,
    optimizer: torch.optim.Optimizer,
    config: SyntraConfig,
    step: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "validation_loss": validation_loss,
        "created_unix": time.time(),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    config: SyntraConfig,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    config.validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    device = choose_device(config.device)
    print("SYNTRA training device: " + str(device))
    if device.type == "cpu":
        print("CPU training selected. Expect slower throughput than a dedicated training GPU.")

    tokens = read_token_stream(config.data_file)
    minimum = max(2 * config.block_size + 2, 64)
    if len(tokens) < minimum:
        raise ValueError(
            "The corpus contains " + str(len(tokens)) + " tokens; at least "
            + str(minimum) + " are required for train/validation splits. Add corpus data or lower block_size."
        )
    split_at = int(len(tokens) * 0.9)
    train_data, val_data = tokens[:split_at], tokens[split_at:]
    if len(val_data) <= config.block_size + 1:
        raise ValueError("Validation split is too small; add more source text.")

    model = SyntraModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    first_step = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Resume checkpoint not found: " + str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        saved_config = SyntraConfig.from_mapping(checkpoint["config"])
        architecture_keys = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")
        mismatches = [
            key for key in architecture_keys
            if getattr(saved_config, key) != getattr(config, key)
        ]
        if mismatches:
            raise ValueError("Resume checkpoint architecture differs for: " + ", ".join(mismatches))
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        first_step = int(checkpoint.get("step", 0))
        print("Resuming at step " + str(first_step))

    output_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.csv"
    new_metrics_file = not metrics_path.exists() or first_step == 0
    best_val = float("inf")
    if not new_metrics_file and metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            values = [float(row["validation_loss"]) for row in rows if row.get("validation_loss")]
            if values:
                best_val = min(values)
        except (OSError, ValueError, KeyError):
            pass

    with metrics_path.open("a" if not new_metrics_file else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "train_loss", "validation_loss", "learning_rate", "seconds"],
        )
        if new_metrics_file:
            writer.writeheader()
        for step in range(first_step, config.max_steps):
            started = time.time()
            progress = step / max(1, config.max_steps)
            learning_rate = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            model.train()
            x, y = _get_batch(train_data, config, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Training loss was not computed.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            should_evaluate = (step + 1) % config.eval_interval == 0 or step == first_step
            if should_evaluate or step == config.max_steps - 1:
                losses = estimate_loss(model, train_data, val_data, config, device)
                row = {
                    "step": step + 1,
                    "train_loss": round(losses["train"], 6),
                    "validation_loss": round(losses["validation"], 6),
                    "learning_rate": learning_rate,
                    "seconds": round(time.time() - started, 3),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    "step " + str(step + 1) + "/" + str(config.max_steps)
                    + " | train=" + format(losses["train"], ".4f")
                    + " | val=" + format(losses["validation"], ".4f")
                    + " | lr=" + format(learning_rate, ".2e")
                )
                _save_checkpoint(
                    output_dir / "latest.pt", model, optimizer, config, step + 1,
                    losses["validation"],
                )
                if losses["validation"] < best_val:
                    best_val = losses["validation"]
                    _save_checkpoint(
                        output_dir / "best.pt", model, optimizer, config, step + 1, best_val
                    )

    print("Training finished. Latest checkpoint: " + str(output_dir / "latest.pt"))
    print("Metrics: " + str(metrics_path))
    return output_dir / "latest.pt"

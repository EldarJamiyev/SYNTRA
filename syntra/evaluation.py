"""Checkpoint loading and basic held-out next-token evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from syntra.config import SyntraConfig
from syntra.data import read_token_stream
from syntra.model import SyntraModel
from syntra.training import _get_batch, choose_device


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    requested_device: str = "auto",
) -> tuple[SyntraModel, SyntraConfig, torch.device, dict[str, Any]]:
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(source))
    device = choose_device(requested_device)
    # Checkpoints are expected to be created by SYNTRA locally. Do not load untrusted files.
    payload = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload or "config" not in payload:
        raise ValueError("Unsupported checkpoint format.")
    config = SyntraConfig.from_mapping(payload["config"])
    model = SyntraModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, device, payload


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_file: str | Path = "data/processed/tokens.bin",
) -> dict[str, float | int | str]:
    model, config, device, payload = load_checkpoint(checkpoint_path)
    tokens = read_token_stream(data_file)
    if len(tokens) < 2 * config.block_size + 2:
        raise ValueError("The evaluation stream is too small for a held-out split.")
    val_data = tokens[int(len(tokens) * 0.9):]
    if len(val_data) <= config.block_size + 1:
        raise ValueError("The held-out split is too small.")
    losses: list[float] = []
    for _ in range(max(20, config.eval_batches)):
        x, y = _get_batch(val_data, config, device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    return {
        "checkpoint": str(checkpoint_path),
        "step": int(payload.get("step", 0)),
        "evaluation_batches": len(losses),
        "cross_entropy": mean_loss,
        "perplexity": math.exp(min(mean_loss, 30.0)),
    }

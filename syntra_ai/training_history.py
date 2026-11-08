"""Bounded, validated training-loss history shared by native training workflows."""

from __future__ import annotations

import math
from typing import Any

MAX_TRAINING_HISTORY_POINTS = 512


def _normalize_loss(value: Any, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Training loss history point {index} has an invalid {field}.")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Training loss history point {index} has an invalid {field}.") from None
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"Training loss history point {index} has an invalid {field}.")
    return normalized


def validate_training_history(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize saved loss points without inventing missing history."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 100_000:
        raise ValueError("Training loss history must be a bounded list.")

    points: list[dict[str, Any]] = []
    previous_step = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Training loss history point {index} must be an object.")
        step = item.get("step")
        phase = item.get("phase")
        train_loss = item.get("train_loss")
        validation_loss = item.get("validation_loss")
        if type(step) is not int or step < 1 or step <= previous_step:
            raise ValueError("Training loss history steps must be positive and strictly increasing.")
        if not isinstance(phase, str) or not phase.strip() or len(phase) > 64:
            raise ValueError(f"Training loss history point {index} has an invalid phase.")
        normalized_train = _normalize_loss(train_loss, "training loss", index)
        normalized_validation = (
            None if validation_loss is None
            else _normalize_loss(validation_loss, "validation loss", index)
        )
        points.append({
            "step": step,
            "phase": phase,
            "train_loss": normalized_train,
            "validation_loss": normalized_validation,
        })
        previous_step = step

    return points[-MAX_TRAINING_HISTORY_POINTS:]


def append_training_history(
    history: Any,
    *,
    step: int,
    phase: str,
    train_loss: float,
    validation_loss: float | None,
) -> list[dict[str, Any]]:
    """Append one measured checkpoint evaluation and retain the newest bounded window."""
    points = validate_training_history(history)
    candidate = validate_training_history([{
        "step": step,
        "phase": phase,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }])[0]
    if points and candidate["step"] <= points[-1]["step"]:
        raise ValueError("New training loss history step must follow the saved history.")
    return [*points, candidate][-MAX_TRAINING_HISTORY_POINTS:]

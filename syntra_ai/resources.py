"""Conservative RAM/VRAM guardrails for local training."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig

SAFETY_FRACTION = 0.70
CHECKPOINT_LOAD_FRACTION = 0.40
INFERENCE_SAFETY_FRACTION = 0.70
WORKSPACE_RESERVE_BYTES = 256 * 1024 * 1024
INFERENCE_RESERVE_BYTES = 128 * 1024 * 1024


def estimate_training_memory_bytes(config: ModelConfig, batch_size: int) -> int:
    """Estimate a conservative working set; this is a guardrail, not an exact profiler."""
    config.validate()
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    embd = config.n_embd
    # Tied token/language-model embeddings are counted only once.
    parameter_count = (
        config.vocab_size * embd
        + config.seq_len * embd
        + config.n_layer * (12 * embd * embd + 13 * embd)
        + 2 * embd
    )
    parameter_and_optimizer = parameter_count * 16  # FP32 weights, gradients and Adam moments.
    activations = batch_size * config.seq_len * embd * config.n_layer * 10 * 4
    attention_workspace = batch_size * config.n_head * config.seq_len * config.seq_len * config.n_layer * 4
    return int(
        (parameter_and_optimizer + activations + attention_workspace) * 1.25
        + WORKSPACE_RESERVE_BYTES
    )


def _system_available_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError):
            return None
        return None

    try:
        # Linux reports MemAvailable, a better estimate than free pages alone.
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if available_pages > 0 and page_size > 0:
            return available_pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return None


def available_memory_bytes(device: Any) -> int | None:
    selected = torch.device(device)
    if selected.type == "cuda":
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(selected)
            return int(free_bytes)
        except (RuntimeError, ValueError, TypeError):
            pass
    return _system_available_memory_bytes()


def validate_training_memory_budget(
    config: ModelConfig,
    batch_size: int,
    device: Any,
    available_bytes: int | None = None,
) -> int:
    """Reject clearly oversized training settings before data/model allocation."""
    estimate = estimate_training_memory_bytes(config, batch_size)
    available = available_memory_bytes(device) if available_bytes is None else int(available_bytes)
    if available is None or available <= 0:
        # Do not guess when the platform does not expose an available-memory estimate.
        return estimate

    safe_budget = int(available * SAFETY_FRACTION)
    if estimate > safe_budget:
        estimate_gib = estimate / (1024 ** 3)
        available_gib = available / (1024 ** 3)
        budget_gib = safe_budget / (1024 ** 3)
        raise ValueError(
            f"Estimated training working set is about {estimate_gib:.1f} GiB, above "
            f"the 70% safety budget ({budget_gib:.1f} GiB of {available_gib:.1f} GiB "
            f"currently available memory on {torch.device(device)}). Reduce --seq-len, "
            "--batch-size, --layers, or --embedding, or close other applications. "
            "This is a conservative estimate, not a guarantee against every out-of-memory error."
        )
    return estimate

def validate_checkpoint_load_budget(
    checkpoint_path: str | Path,
    available_bytes: int | None = None,
) -> int:
    """Reject a checkpoint file that could consume most of currently free system RAM."""
    target = Path(checkpoint_path).expanduser()
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ValueError("Cannot inspect checkpoint file size before loading: " + str(exc)) from exc
    if size <= 0:
        raise ValueError("The selected checkpoint file is empty.")

    available = _system_available_memory_bytes() if available_bytes is None else int(available_bytes)
    if available is not None and available > 0:
        budget = int(available * CHECKPOINT_LOAD_FRACTION)
        if size > budget:
            size_gib = size / (1024 ** 3)
            available_gib = available / (1024 ** 3)
            budget_gib = budget / (1024 ** 3)
            raise ValueError(
                f"Checkpoint file is {size_gib:.2f} GiB, above the 40% checkpoint-load budget "
                f"({budget_gib:.2f} GiB of {available_gib:.2f} GiB currently available RAM). "
                "Close memory-heavy applications or use a smaller local checkpoint before loading it."
            )
    return size


def estimate_inference_memory_bytes(
    config: ModelConfig,
    batch_size: int = 1,
    sequence_length: int | None = None,
) -> int:
    """Estimate model weights, activations and conservative attention workspace for inference."""
    config.validate()
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    length = config.seq_len if sequence_length is None else sequence_length
    if not isinstance(length, int) or length < 1 or length > config.seq_len:
        raise ValueError("sequence_length must be between 1 and the configured context length.")

    embd = config.n_embd
    parameter_count = (
        config.vocab_size * embd
        + config.seq_len * embd
        + config.n_layer * (12 * embd * embd + 13 * embd)
        + 2 * embd
    )
    weights = parameter_count * 4
    activations = batch_size * length * embd * config.n_layer * 6 * 4
    attention_workspace = batch_size * config.n_head * length * length * config.n_layer * 4
    return int((weights + activations + attention_workspace) * 1.20 + INFERENCE_RESERVE_BYTES)


def validate_inference_memory_budget(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    device: Any,
    available_bytes: int | None = None,
) -> int:
    """Reject clearly oversized inference settings before allocating the live model."""
    estimate = estimate_inference_memory_bytes(config, batch_size, sequence_length)
    available = available_memory_bytes(device) if available_bytes is None else int(available_bytes)
    if available is None or available <= 0:
        return estimate

    safe_budget = int(available * INFERENCE_SAFETY_FRACTION)
    if estimate > safe_budget:
        estimate_gib = estimate / (1024 ** 3)
        available_gib = available / (1024 ** 3)
        budget_gib = safe_budget / (1024 ** 3)
        raise ValueError(
            f"Estimated inference working set is about {estimate_gib:.1f} GiB, above "
            f"the 70% safety budget ({budget_gib:.1f} GiB of {available_gib:.1f} GiB "
            f"available memory on {torch.device(device)}). Reduce --seq-len, --layers, or "
            "--embedding, or close other applications. This is conservative; actual memory "
            "use depends on the selected PyTorch attention backend."
        )
    return estimate


"""Validated configuration shared by training, inference, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class SyntraConfig:
    seed: int = 42
    vocab_size: int = 260
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    grad_clip: float = 1.0
    num_threads: int = 6
    device: str = "auto"
    data_file: str = "data/processed/tokens.bin"
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> "SyntraConfig":
        if self.vocab_size != 260:
            raise ValueError("The built-in byte tokenizer requires vocab_size=260.")
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8.")
        if self.n_layer < 1:
            raise ValueError("n_layer must be at least 1.")
        if self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_head and n_embd must be positive.")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("eval_interval and eval_batches must be positive.")
        if self.grad_clip <= 0 or self.num_threads < 1:
            raise ValueError("grad_clip and num_threads must be positive.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SyntraConfig":
        allowed = cls.__dataclass_fields__.keys()
        config = cls(**{key: values[key] for key in allowed if key in values})
        return config.validate()

    @classmethod
    def load(cls, path: str | Path) -> "SyntraConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration root must be a JSON object.")
        return cls.from_mapping(values)

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

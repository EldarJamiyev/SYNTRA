"""Local checkpoint loading, status, generation, and evaluation."""
from pathlib import Path
import json
import math

import torch

from .config import ModelConfig
from .model import BOS_ID, EOS_ID, SyntraTransformer
from .training import choose_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "syntra.pt"


def checkpoint_path() -> Path:
    import os
    return Path(os.environ.get("SYNTRA_CHECKPOINT", str(DEFAULT_CHECKPOINT))).expanduser().resolve()


def status() -> dict:
    path = checkpoint_path()
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return {
        "service": "SYNTRA native local Transformer",
        "backend": "PyTorch",
        "checkpoint": str(path),
        "checkpoint_exists": path.is_file(),
        "trained_steps": int(metadata.get("step", 0)),
        "parameter_count": int(metadata.get("parameter_count", 0)),
        "config": metadata.get("config"),
        "train_loss": metadata.get("train_loss"),
        "validation_loss": metadata.get("validation_loss"),
        "corpus_files": metadata.get("corpus_files", 0),
        "corpus_tokens": metadata.get("corpus_tokens", 0),
        "training_mode": metadata.get("training_mode", "causal-source-pretraining"),
        "instruction_examples": metadata.get("instruction_examples", 0),
        "pretrained_weights": False,
        "external_ai_api": False,
        "note": "Generation is unavailable until a local checkpoint has actually been trained."
    }


def load_model(path: str | Path | None = None, device_name: str = "auto"):
    target = Path(path).expanduser().resolve() if path else checkpoint_path()
    if not target.is_file():
        raise FileNotFoundError("No trained SYNTRA checkpoint exists at " + str(target) + ". Train a local model first.")
    device = choose_device(device_name)
    saved = torch.load(target, map_location=device, weights_only=True)
    if saved.get("format") != "syntra-transformer-v1":
        raise ValueError("The selected file is not a SYNTRA Transformer checkpoint.")
    config = ModelConfig.from_dict(saved["config"])
    model = SyntraTransformer(config).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved, device


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
    prompt_bytes = list(prompt.encode("utf-8"))
    tokens = torch.tensor([[BOS_ID, *prompt_bytes]], dtype=torch.long, device=device)
    output = model.generate(tokens, max_new_tokens, temperature, top_k)[0].tolist()
    generated_ids = output[len(tokens[0]):]
    generated_bytes = bytes(token for token in generated_ids if 0 <= token <= 255)
    return {
        "text": generated_bytes.decode("utf-8", errors="replace"),
        "prompt_tokens": len(prompt_bytes),
        "generated_tokens": len(generated_ids),
        "checkpoint": str(checkpoint_path()),
        "trained_steps": int(saved.get("step", 0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "is_model_output": True,
        "warning": "Raw model continuation. Inspect and test generated code before applying it."
    }


@torch.no_grad()
def evaluate(device_name: str = "auto") -> dict:
    # Validation loss is measured against a held-out portion of the local corpus
    # during the last checkpoint save. The original corpus is not silently re-read.
    _model, saved, _device = load_model(device_name=device_name)
    val_loss = saved.get("validation_loss")
    train_loss = saved.get("train_loss")
    if val_loss is None:
        raise ValueError("This checkpoint contains no held-out validation measurement.")
    measured_loss = float(val_loss)
    return {
        "checkpoint": str(checkpoint_path()),
        "step": int(saved.get("step", 0)),
        "train_loss": train_loss,
        "validation_loss": measured_loss,
        "cross_entropy": measured_loss,
        "perplexity": math.exp(min(measured_loss, 80.0)),
        "measurement": "Held-out corpus split measured and saved during training",
        "pretrained_weights": False,
    }

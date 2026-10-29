"""Supervised prompt/completion fine-tuning with the checkpoint's exact BPE tokenizer."""
from pathlib import Path
from time import monotonic
from typing import Callable
import json

import torch

from .config import ModelConfig
from .model import BOS_ID, EOS_ID, SyntraTransformer
from .resources import validate_training_memory_budget
from .runtime import checkpoint_path
from .tokenizer import ByteBPETokenizer
from .training import _atomic_checkpoint, choose_device


MAX_INSTRUCTION_EXAMPLES = 10_000
MAX_INSTRUCTION_TOKENS = 3_000_000
MAX_INSTRUCTION_FILE_BYTES = 100 * 1024 * 1024
MAX_INSTRUCTION_RECORD_BYTES = 1024 * 1024
MAX_INSTRUCTION_FIELD_BYTES = 512 * 1024


def load_instruction_examples(
    dataset_path: str | Path,
    seq_len: int,
    tokenizer: ByteBPETokenizer | None = None,
    max_examples: int = MAX_INSTRUCTION_EXAMPLES,
    max_tokens: int = MAX_INSTRUCTION_TOKENS,
) -> list[dict]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Instruction dataset must be an existing local JSONL file.")
    if path.stat().st_size > MAX_INSTRUCTION_FILE_BYTES:
        raise ValueError(f"Instruction JSONL is limited to {MAX_INSTRUCTION_FILE_BYTES // (1024 * 1024)} MiB.")
    if not isinstance(seq_len, int) or seq_len < 16 or seq_len > 16384:
        raise ValueError("Instruction context must be between 16 and 16,384 tokens.")
    if not isinstance(max_examples, int) or max_examples < 2 or max_examples > MAX_INSTRUCTION_EXAMPLES:
        raise ValueError(f"max_examples must be between 2 and {MAX_INSTRUCTION_EXAMPLES:,}.")
    if not isinstance(max_tokens, int) or max_tokens < 1024 or max_tokens > MAX_INSTRUCTION_TOKENS:
        raise ValueError(f"max_tokens must be between 1,024 and {MAX_INSTRUCTION_TOKENS:,}.")
    tokenizer = tokenizer or ByteBPETokenizer.empty()
    examples = []
    total_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES:
                raise ValueError(f"Instruction record on line {line_number} exceeds the 1 MiB record limit.")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str) or not isinstance(item.get("completion"), str):
                raise ValueError(f"Line {line_number} must contain string fields named prompt and completion.")
            prompt_text, completion_text = item["prompt"], item["completion"]
            if not prompt_text.strip() or not completion_text.strip():
                continue
            if (len(prompt_text.encode("utf-8")) > MAX_INSTRUCTION_FIELD_BYTES
                or len(completion_text.encode("utf-8")) > MAX_INSTRUCTION_FIELD_BYTES):
                raise ValueError(f"Prompt/completion on line {line_number} exceeds the 512 KiB field limit.")
            prompt = tokenizer.encode(prompt_text)
            completion = tokenizer.encode(completion_text)
            completion = completion[:max(8, seq_len // 2)]
            prompt_limit = max(0, seq_len - len(completion) - 2)
            prompt = prompt[-prompt_limit:] if prompt_limit else []
            sequence = [BOS_ID, *prompt, *completion, EOS_ID]
            input_ids = sequence[:-1]
            labels = sequence[1:]
            # labels 0..len(prompt)-1 target the prompt; completion and EOS remain supervised.
            for position in range(min(len(prompt), len(labels))):
                labels[position] = -100
            example_tokens = len(input_ids) + len(labels)
            if total_tokens + example_tokens > max_tokens:
                raise ValueError(
                    f"Instruction dataset exceeds the {max_tokens:,}-token safety cap at line {line_number}. "
                    "Reduce max_examples or shorten the prompt/completion examples."
                )
            examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "prompt_tokens": len(prompt),
                "completion_tokens": len(completion),
                "line": line_number,
            })
            total_tokens += example_tokens
            if len(examples) >= max_examples:
                break
    if len(examples) < 2:
        raise ValueError("At least two non-empty prompt/completion examples are required for train/validation splits.")
    return examples


def _instruction_batch(examples: list[dict], batch_size: int, device: torch.device):
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    chosen = [examples[index] for index in indices]
    length = max(len(example["input_ids"]) for example in chosen)
    inputs, targets = [], []
    for example in chosen:
        padding = length - len(example["input_ids"])
        inputs.append(example["input_ids"] + [EOS_ID] * padding)
        targets.append(example["labels"] + [-100] * padding)
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


@torch.no_grad()
def _instruction_loss(model, examples: list[dict], device: torch.device, batch_size: int, batches: int = 4) -> float:
    model.eval()
    values = []
    for _ in range(batches):
        x, y = _instruction_batch(examples, batch_size, device)
        _, loss = model(x, y)
        if loss is not None and torch.isfinite(loss):
            values.append(float(loss.item()))
    model.train()
    if not values:
        raise ValueError("Validation batch contained no supervised completion tokens.")
    return sum(values) / len(values)


def finetune(
    dataset_path: str | Path,
    steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
    checkpoint: str | Path | None = None,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    if steps < 1 or steps > 1_000_000:
        raise ValueError("steps must be between 1 and 1,000,000.")
    if batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size must be between 1 and 128.")
    if not 1e-6 <= learning_rate <= 1e-2:
        raise ValueError("learning_rate must be between 1e-6 and 1e-2.")
    checkpoint_path_value = Path(checkpoint).expanduser().resolve() if checkpoint else checkpoint_path()
    if not checkpoint_path_value.is_file():
        raise FileNotFoundError("Train a SYNTRA checkpoint from local source code before instruction fine-tuning.")
    device = choose_device(device_name)
    # Read checkpoint tensors on CPU first; run the same resource guard used by
    # pretraining before moving weights, gradients, and optimizer state to an accelerator.
    saved = torch.load(checkpoint_path_value, map_location="cpu", weights_only=True)
    if saved.get("format") != "syntra-transformer-v1":
        raise ValueError("Selected checkpoint is not a SYNTRA Transformer checkpoint.")
    config = ModelConfig.from_dict(saved["config"])
    config.validate()
    validate_training_memory_budget(config, batch_size, device)
    tokenizer = ByteBPETokenizer.from_dict(saved.get("tokenizer"))
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint tokenizer vocabulary does not match the model architecture.")
    model = SyntraTransformer(config).to(device)
    model.load_state_dict(saved["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    examples = load_instruction_examples(dataset_path, config.seq_len, tokenizer)
    split = max(1, int(len(examples) * 0.9))
    if split >= len(examples):
        split = len(examples) - 1
    training_examples, validation_examples = examples[:split], examples[split:]
    base_step = int(saved.get("step", 0))
    started = monotonic()
    last_train_loss = None
    last_val_loss = None

    def report(status: str, step: int, message: str) -> dict:
        value = {
            "status": status,
            "step": step,
            "max_steps": base_step + steps,
            "train_loss": last_train_loss,
            "validation_loss": last_val_loss,
            "device": str(device),
            "elapsed_seconds": round(monotonic() - started, 2),
            "message": message,
            "checkpoint": str(checkpoint_path_value),
            "training_mode": "supervised-instruction-finetuning",
            "instruction_examples": len(examples),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "context_tokens": config.seq_len,
        }
        if callback:
            callback(value)
        return value

    model.train()
    report("running", base_step, f"Supervised fine-tuning started with {len(examples)} local prompt/completion examples.")
    for step in range(base_step + 1, base_step + steps + 1):
        x, y = _instruction_batch(training_examples, batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Training loss is invalid; check the instruction dataset.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_loss = float(loss.item())
        if step == base_step + 1 or step % 25 == 0 or step == base_step + steps:
            last_val_loss = _instruction_loss(model, validation_examples, device, batch_size, batches=2)
            payload = {
                **saved,
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "training_mode": "supervised-instruction-finetuning",
                "instruction_examples": len(examples),
                "train_loss": last_train_loss,
                "validation_loss": last_val_loss,
            }
            _atomic_checkpoint(checkpoint_path_value, payload)
            metadata_path = checkpoint_path_value.with_suffix(checkpoint_path_value.suffix + ".json")
            metadata = {
                "checkpoint": str(checkpoint_path_value), "step": step, "config": config.to_dict(),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "train_loss": last_train_loss, "validation_loss": last_val_loss,
                "tokenizer": "byte-BPE", "tokenizer_vocab_size": tokenizer.vocab_size,
                "context_tokens": config.seq_len,
                "training_mode": "supervised-instruction-finetuning", "instruction_examples": len(examples),
                "pretrained_weights": False,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            report("running" if step < base_step + steps else "completed", step, "Checkpoint saved with measured supervised validation loss.")
    return report("completed", base_step + steps, "Supervised fine-tuning finished; the local BPE-tokenized checkpoint was updated.")

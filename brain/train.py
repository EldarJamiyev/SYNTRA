"""Train SYNTRA's standalone Transformer exclusively on selected local text files.

Example:
    python -m brain.train --dataset dataset.txt --epochs 3 --device auto
    python -m brain.train --dataset .\corpus\python.txt --dataset .\corpus\docs.txt --resume
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Iterable
import unicodedata

import torch

from .checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from .model import LocalTransformer, TransformerConfig, select_device
from .tokenizer import BOS_ID, EOS_ID, BytePairTokenizer

DEFAULT_DATASET = Path("dataset.txt")
DEFAULT_CHECKPOINT = Path("checkpoints") / "syntra-brain.pt"
MAX_DATASET_BYTES = 256 * 1024 * 1024
MAX_INSTRUCTION_RECORD_BYTES = 1024 * 1024
MAX_PACKED_CONTEXT_TOKENS = 10_000_000
ProgressCallback = Callable[[dict], None]


SOURCE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".scala", ".sh", ".ps1", ".bat",
    ".sql", ".html", ".htm", ".css", ".scss", ".sass", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".rst", ".txt",
    ".xml", ".graphql", ".gql", ".proto", ".vue", ".svelte", ".ipynb",
})
SKIP_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules", "vendor",
    "dist", "build", "target", "coverage", ".next", ".nuxt", ".turbo",
    ".cache", ".idea", ".vscode", "site-packages", "checkpoints", "artifacts",
    "logs", ".syntra",
})
SENSITIVE_SUFFIXES = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".p7b", ".p7c", ".jks", ".crt", ".cer", ".der",
    ".keystore", ".sqlite", ".db", ".dump", ".bak", ".log",
})
MAX_DATASET_FILES = 20_000


def _is_sensitive_name(name: str) -> bool:
    lower = name.casefold()
    if lower == ".env" or lower.startswith(".env."):
        return True
    if lower in {
        "credentials", "credentials.json", "secrets", "secrets.json",
        "id_rsa", "id_ed25519", "known_hosts", "authorized_keys",
        "access_token", "access_token.txt", "refresh_token", "token.txt",
    }:
        return True
    if any(marker in lower for marker in ("secret", "credential", "password", "private_key", "private-key")):
        return True
    if Path(lower).suffix in SENSITIVE_SUFFIXES:
        return True
    return False


def _iter_local_source_files(source: Path):
    """Yield approved text files in a file or project folder; never follow symlinks."""
    resolved_source = source.resolve(strict=True)
    if resolved_source.is_file():
        if _is_sensitive_name(resolved_source.name):
            raise ValueError(f"Refusing to train on an obviously sensitive file: {resolved_source.name}")
        yield resolved_source, resolved_source.name, True
        return
    if not resolved_source.is_dir():
        raise ValueError(f"Dataset path must be a local file or directory: {source}")

    root = resolved_source

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current_path = Path(current)
        directory_names[:] = sorted(
            name for name in directory_names
            if name.casefold() not in SKIP_DIRECTORIES
            and name.casefold() not in {"secrets", "private", "credentials", "certs", "certificates"}
            and not _is_sensitive_name(name)
            and not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            candidate = current_path / filename
            if candidate.is_symlink() or _is_sensitive_name(filename):
                continue
            if candidate.suffix.casefold() not in SOURCE_EXTENSIONS:
                continue
            resolved = candidate.resolve(strict=True)
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if not resolved.is_file():
                continue
            yield resolved, relative.as_posix(), False


def _read_local_datasets(
    paths: Iterable[str | os.PathLike[str]],
    max_dataset_bytes: int,
    exclude_paths: set[str] | None = None,
) -> tuple[str, str, list[str], int, str]:
    """Read bounded local text/code files and produce a stable corpus fingerprint."""
    documents: list[str] = []
    names: list[str] = []
    digest = sha256()
    total_bytes = 0
    file_count = 0
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            source = path.resolve(strict=True)
            for resolved, relative_name, explicitly_selected in _iter_local_source_files(source):
                if exclude_paths and str(resolved) in exclude_paths:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                if file_count >= MAX_DATASET_FILES:
                    raise ValueError(f"Local training data exceeds the {MAX_DATASET_FILES:,}-file limit.")
                size = resolved.stat().st_size
                if size < 1:
                    continue
                if total_bytes + size > max_dataset_bytes:
                    raise ValueError(
                        f"Selected datasets exceed the {max_dataset_bytes:,}-byte local corpus limit."
                    )
                raw = resolved.read_bytes()
                if total_bytes + len(raw) > max_dataset_bytes:
                    raise ValueError(
                        f"Selected datasets exceed the {max_dataset_bytes:,}-byte local corpus limit."
                    )
                try:
                    source_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    if explicitly_selected:
                        raise ValueError(f"Dataset must contain valid UTF-8 text: {resolved}") from None
                    continue

                total_bytes += len(raw)
                file_count += 1
                relative_bytes = relative_name.encode("utf-8")
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
                documents.append(f"\n# SYNTRA_SOURCE_FILE: {relative_name}\n{source_text}")
                names.append(str(resolved))
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError(f"Cannot read local dataset {path}: {exc}") from None

    if not documents or not any(document.strip() for document in documents):
        raise ValueError(
            "No readable UTF-8 source files were found. Select a text/code file or a project folder."
        )

    combined = "\n\n".join(documents)
    split_at = int(len(combined) * 0.9)
    # Prefer a source-file line boundary while keeping both splits non-empty.
    line_boundary = combined.rfind("\n", 0, split_at)
    if line_boundary > 0 and len(combined) - line_boundary > 1:
        split_at = line_boundary + 1
    train_text = combined[:split_at]
    validation_text = combined[split_at:]
    if not train_text.strip() or not validation_text.strip():
        raise ValueError("The local corpus is too small to create separate training and validation splits.")
    return train_text, validation_text, names, total_bytes, digest.hexdigest()



def _read_instruction_jsonl_datasets(
    paths: Iterable[str | os.PathLike[str]],
    max_dataset_bytes: int,
) -> tuple[list[dict[str, str]], list[str], int, str]:
    """Read structured prompt/response JSONL files; leave ordinary text for the legacy reader."""
    records: list[dict[str, str]] = []
    names: list[str] = []
    digest = sha256()
    total_bytes = 0
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            source = path.resolve(strict=True)
            for resolved, relative_name, _explicitly_selected in _iter_local_source_files(source):
                if resolved.suffix.casefold() != ".jsonl" or resolved in seen:
                    continue
                seen.add(resolved)
                raw = resolved.read_bytes()
                if not raw:
                    continue
                if total_bytes + len(raw) > max_dataset_bytes:
                    raise ValueError(
                        f"Structured instruction datasets exceed the {max_dataset_bytes:,}-byte local corpus limit."
                    )
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    # The general reader will raise for an explicitly selected invalid UTF-8 file.
                    continue

                file_records: list[dict[str, str]] = []
                structured = True
                nonempty_lines = 0
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    nonempty_lines += 1
                    if len(line.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES:
                        raise ValueError(
                            f"Instruction JSONL record in {relative_name}:{line_number} exceeds "
                            f"{MAX_INSTRUCTION_RECORD_BYTES:,} bytes."
                        )
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        structured = False
                        break
                    if not isinstance(item, dict):
                        structured = False
                        break
                    fields = {str(key).casefold(): value for key, value in item.items()}
                    instruction = fields.get("instruction", fields.get("prompt"))
                    response = fields.get("response", fields.get("completion", fields.get("output")))
                    if (
                        not isinstance(instruction, str)
                        or not isinstance(response, str)
                        or not instruction.strip()
                        or not response.strip()
                    ):
                        structured = False
                        break
                    if (
                        len(instruction.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES
                        or len(response.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES
                    ):
                        raise ValueError(
                            f"Instruction/response field in {relative_name}:{line_number} exceeds "
                            f"{MAX_INSTRUCTION_RECORD_BYTES:,} bytes."
                        )
                    file_records.append({
                        "prompt": instruction.strip(),
                        "completion": response.strip(),
                        "source_path": relative_name,
                        "source_line": str(line_number),
                    })

                # A JSONL file is treated as structured only when every non-blank
                # record has a supported prompt/response shape. Otherwise it is
                # retained as ordinary text by the backwards-compatible reader.
                if not structured or not nonempty_lines or not file_records:
                    continue

                total_bytes += len(raw)
                names.append(str(resolved))
                relative_bytes = relative_name.encode("utf-8")
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
                records.extend(file_records)
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError(f"Cannot read local instruction dataset {path}: {exc}") from None

    return records, names, total_bytes, digest.hexdigest()


def _split_instruction_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split by normalized instruction groups to prevent repeated-prompt leakage."""
    if not records:
        return [], []
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        normalized = " ".join(unicodedata.normalize("NFKC", record["prompt"]).casefold().split())
        group_id = sha256(normalized.encode("utf-8")).hexdigest()
        groups.setdefault(group_id, []).append(record)
    if len(groups) < 2:
        raise ValueError(
            "Structured instruction training needs at least two distinct prompts "
            "so validation can remain separate from training."
        )
    validation_count = max(1, int(len(groups) * 0.1))
    validation_count = min(validation_count, len(groups) - 1)
    ranked = sorted(
        groups,
        key=lambda key: sha256(("syntra-brain-instruction-split-v1:" + key).encode("utf-8")).hexdigest(),
    )
    validation_groups = set(ranked[:validation_count])
    training = [
        item for item in records
        if sha256(
            " ".join(unicodedata.normalize("NFKC", item["prompt"]).casefold().split()).encode("utf-8")
        ).hexdigest() not in validation_groups
    ]
    validation = [
        item for item in records
        if sha256(
            " ".join(unicodedata.normalize("NFKC", item["prompt"]).casefold().split()).encode("utf-8")
        ).hexdigest() in validation_groups
    ]
    if not training or not validation:
        raise ValueError("The instruction dataset did not produce non-empty train and validation sets.")
    return training, validation


def _append_context_block(
    input_ids: list[int],
    target_ids: list[int],
    block_size: int,
    input_blocks: list[list[int]],
    target_blocks: list[list[int]],
) -> None:
    if not input_ids or len(input_ids) != len(target_ids):
        return
    if len(input_ids) > block_size:
        raise ValueError("Internal context block exceeds the configured block size.")
    padding = block_size - len(input_ids)
    input_blocks.append(input_ids + [EOS_ID] * padding)
    target_blocks.append(target_ids + [-100] * padding)


def _raw_context_blocks(
    text: str,
    tokenizer: BytePairTokenizer,
    block_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pack legacy raw text into fixed, shuffled causal context blocks."""
    if not text.strip():
        return [], []
    token_ids = [BOS_ID, *tokenizer.encode(text), EOS_ID]
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, max(0, len(token_ids) - 1), block_size):
        window = token_ids[start:start + block_size + 1]
        if len(window) < 2:
            continue
        x, y = window[:-1], window[1:]
        if y:
            _append_context_block(x, y, block_size, inputs, targets)
        if (len(inputs) * block_size) > MAX_PACKED_CONTEXT_TOKENS:
            raise ValueError("Packed raw context exceeds the SYNTRA training token safety limit.")
    return inputs, targets


def _instruction_context_blocks(
    records: list[dict[str, str]],
    tokenizer: BytePairTokenizer,
    block_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pack prompt/response examples; mask prompt targets and supervise response plus EOS only."""
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for record in records:
        prompt = "Instruction:\\n" + record["prompt"].strip() + "\\nResponse:\\n"
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(record["completion"])
        if not response_ids:
            continue
        sequence = [BOS_ID, *prompt_ids, *response_ids, EOS_ID]
        first_response_target = 1 + len(prompt_ids)
        # Start at the final context window before the response when instructions
        # are longer than the model window. Subsequent blocks overlap by one token.
        start = max(0, first_response_target - block_size)
        while start < len(sequence) - 1:
            window = sequence[start:start + block_size + 1]
            if len(window) < 2:
                break
            x, y = window[:-1], window[1:]
            for index in range(len(y)):
                target_position = start + index + 1
                if target_position < first_response_target:
                    y[index] = -100
            if any(token != -100 for token in y):
                _append_context_block(x, y, block_size, inputs, targets)
            if (len(inputs) * block_size) > MAX_PACKED_CONTEXT_TOKENS:
                raise ValueError("Packed instruction contexts exceed the SYNTRA training token safety limit.")
            start += block_size
    return inputs, targets


def _stack_context_blocks(
    inputs: list[list[int]],
    targets: list[list[int]],
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not inputs or len(inputs) != len(targets):
        raise ValueError(f"The {label} data did not produce any supervised context blocks.")
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )

def _batch(
    data: torch.Tensor,
    chunk_indices: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = (chunk_indices * block_size).tolist()
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_validation_loss(
    model: LocalTransformer,
    validation_data: torch.Tensor,
    block_size: int,
    device: torch.device,
    batches: int,
) -> float:
    model.eval()
    values: list[float] = []
    available_chunks = max(1, (validation_data.numel() - 1) // block_size)
    for _ in range(batches):
        count = min(8, available_chunks)
        starts = torch.randint(0, available_chunks, (count,), device="cpu")
        x, y = _batch(validation_data, starts, block_size, device)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError("Validation produced a non-finite loss.")
        values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def _rng_state(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None, device: torch.device) -> None:
    if not isinstance(state, dict):
        return
    cpu_state = state.get("cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state)
    cuda_states = state.get("cuda")
    if device.type == "cuda" and torch.cuda.is_available() and isinstance(cuda_states, list):
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except (RuntimeError, ValueError):
            # Device topology may have changed; the saved model/optimizer remain valid.
            pass


def train(
    dataset_paths: Iterable[str | os.PathLike[str]] = (DEFAULT_DATASET,),
    *,
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    config: TransformerConfig | None = None,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    tokenizer_vocab_size: int | None = None,
    tokenizer_bytes: int = 100_000,
    max_dataset_bytes: int = MAX_DATASET_BYTES,
    evaluation_batches: int = 8,
    save_every_steps: int = 100,
    device_name: str = "auto",
    seed: int = 1337,
    resume: bool = False,
    callback: ProgressCallback | None = None,
) -> dict:
    """Train or resume a local model, saving complete optimizer/model state."""
    if type(epochs) is not int or not 1 <= epochs <= 100_000:
        raise ValueError("epochs must be an integer between 1 and 100,000.")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be an integer between 1 and 256.")
    if not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 1e-2:
        raise ValueError("learning_rate must be between 1e-6 and 1e-2.")
    if not math.isfinite(weight_decay) or not 0 <= weight_decay <= 1:
        raise ValueError("weight_decay must be between 0 and 1.")
    if type(evaluation_batches) is not int or not 1 <= evaluation_batches <= 1000:
        raise ValueError("evaluation_batches must be between 1 and 1,000.")
    if type(save_every_steps) is not int or not 1 <= save_every_steps <= 100_000:
        raise ValueError("save_every_steps must be between 1 and 100,000.")
    if type(max_dataset_bytes) is not int or not 1024 <= max_dataset_bytes <= MAX_DATASET_BYTES:
        raise ValueError(f"max_dataset_bytes must be between 1,024 and {MAX_DATASET_BYTES:,}.")
    if type(tokenizer_bytes) is not int or not 1024 <= tokenizer_bytes <= 10 * 1024 * 1024:
        raise ValueError("tokenizer_bytes must be between 1,024 and 10,485,760.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4,294,967,295.")

    paths = tuple(dataset_paths)
    if not paths:
        raise ValueError("At least one local dataset file is required.")
    train_text, validation_text, dataset_files, dataset_bytes, dataset_hash = _read_local_datasets(
        paths, max_dataset_bytes
    )
    target = Path(checkpoint_path).expanduser().resolve()
    device = select_device(device_name)
    requested_config = config or TransformerConfig(
        vocab_size=tokenizer_vocab_size or 512,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    requested_config.validate()
    desired_vocab = tokenizer_vocab_size or requested_config.vocab_size
    if not 258 <= desired_vocab <= 8192:
        raise ValueError("tokenizer_vocab_size must be between 258 and 8,192.")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    saved: dict | None = None
    if resume:
        saved = load_checkpoint(target)
        if saved["dataset_sha256"] != dataset_hash:
            raise ValueError(
                "The selected dataset differs from the checkpoint's training corpus. "
                "Start a new checkpoint or use the original dataset to resume."
            )
        stored_config = TransformerConfig.from_dict(saved["config"])
        if config is not None:
            comparable_requested = replace(requested_config, vocab_size=stored_config.vocab_size)
            if comparable_requested != stored_config:
                raise ValueError("Requested model architecture differs from the saved checkpoint.")
        requested_config = stored_config
        tokenizer = BytePairTokenizer.from_dict(saved["tokenizer"])
    else:
        tokenizer = BytePairTokenizer.train_from_text(
            train_text,
            vocab_size=desired_vocab,
            max_training_bytes=tokenizer_bytes,
        )
        requested_config = replace(requested_config, vocab_size=tokenizer.vocab_size)

    if requested_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary and local tokenizer vocabulary do not match.")

    train_ids = torch.tensor(
        [BOS_ID, *tokenizer.encode(train_text), EOS_ID],
        dtype=torch.long,
    )
    validation_ids = torch.tensor(
        [BOS_ID, *tokenizer.encode(validation_text), EOS_ID],
        dtype=torch.long,
    )
    block_size = requested_config.block_size
    if train_ids.numel() <= block_size + 1 or validation_ids.numel() <= block_size + 1:
        raise ValueError(
            f"Training and validation splits must each contain more than {block_size + 1} tokens. "
            f"Received {train_ids.numel()} training and {validation_ids.numel()} validation tokens."
        )

    chunk_count = (train_ids.numel() - 1) // block_size
    if chunk_count < 1:
        raise ValueError("The training split does not contain a complete context block.")
    steps_per_epoch = math.ceil(chunk_count / batch_size)
    model = LocalTransformer(requested_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_epoch = 0
    start_step = 0
    best_validation_loss: float | None = None
    last_train_loss: float | None = None
    last_validation_loss: float | None = None
    epoch_order: torch.Tensor | None = None
    next_batch_index = 0

    if saved is not None:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        _restore_rng_state(saved.get("rng_state"), device)
        start_epoch = int(saved["epoch"])
        start_step = int(saved["step"])
        best_validation_loss = saved.get("best_validation_loss")
        last_train_loss = saved.get("train_loss")
        last_validation_loss = saved.get("validation_loss")
        raw_order = saved.get("epoch_order")
        if isinstance(raw_order, torch.Tensor) and raw_order.numel() == chunk_count:
            epoch_order = raw_order.to(dtype=torch.long, device="cpu")
            next_batch_index = int(saved.get("next_batch_index", 0))
            if not 0 <= next_batch_index <= steps_per_epoch:
                raise ValueError("Checkpoint resume position is invalid.")
        elif raw_order is not None:
            raise ValueError("Checkpoint data-order state does not match the current corpus.")

    parameter_count = model.parameter_count
    started = perf_counter()
    target_epoch = start_epoch + epochs
    total_steps_this_run = epochs * steps_per_epoch
    current_step_in_run = 0

    def payload_for_save(completed_epochs: int, order: torch.Tensor | None, next_index: int) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "config": requested_config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": start_step + current_step_in_run,
            "epoch": completed_epochs,
            "epoch_order": order,
            "next_batch_index": next_index,
            "best_validation_loss": best_validation_loss,
            "train_loss": last_train_loss,
            "validation_loss": last_validation_loss,
            "dataset_sha256": dataset_hash,
            "dataset_files": dataset_files,
            "dataset_bytes": dataset_bytes,
            "training_settings": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "steps_per_epoch": steps_per_epoch,
            },
            "rng_state": _rng_state(device),
            "parameter_count": parameter_count,
            "updated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        }

    def report(status: str, completed_epochs: int, message: str) -> dict:
        elapsed = max(perf_counter() - started, 1e-9)
        result = {
            "status": status,
            "checkpoint": str(target),
            "device": str(device),
            "step": start_step + current_step_in_run,
            "epoch": completed_epochs,
            "target_epoch": target_epoch,
            "steps_per_epoch": steps_per_epoch,
            "train_loss": last_train_loss,
            "validation_loss": last_validation_loss,
            "best_validation_loss": best_validation_loss,
            "tokens_per_second": round(current_step_in_run * batch_size * block_size / elapsed, 2),
            "elapsed_seconds": round(elapsed, 2),
            "parameter_count": parameter_count,
            "dataset_files": len(dataset_files),
            "dataset_bytes": dataset_bytes,
            "training_tokens": int(train_ids.numel()),
            "validation_tokens": int(validation_ids.numel()),
            "vocab_size": tokenizer.vocab_size,
            "message": message,
        }
        if callback:
            callback(result)
        return result

    report(
        "running",
        start_epoch,
        "Resuming local model and optimizer state." if saved else "Training a randomly initialized Transformer on local UTF-8 data.",
    )
    model.train()
    while start_epoch < target_epoch:
        if epoch_order is None:
            epoch_order = torch.randperm(chunk_count, device="cpu")
            next_batch_index = 0

        for batch_index in range(next_batch_index, steps_per_epoch):
            begin = batch_index * batch_size
            indices = epoch_order[begin : begin + batch_size]
            if indices.numel() == 0:
                continue
            x, y = _batch(train_ids, indices, block_size, device)

            current_step_in_run += 1
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            progress = current_step_in_run / max(total_steps_this_run, 1)
            warmup = min(1.0, current_step_in_run / max(10, min(total_steps_this_run, 100)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            scheduled_lr = learning_rate * warmup * cosine
            for group in optimizer.param_groups:
                group["lr"] = max(scheduled_lr, learning_rate * 0.05)
            optimizer.step()
            last_train_loss = float(loss.detach().item())
            next_batch_index = batch_index + 1

            should_evaluate = (
                current_step_in_run == 1
                or current_step_in_run % save_every_steps == 0
                or next_batch_index == steps_per_epoch
            )
            if should_evaluate:
                last_validation_loss = _estimate_validation_loss(
                    model,
                    validation_ids,
                    block_size,
                    device,
                    evaluation_batches,
                )
                if best_validation_loss is None or last_validation_loss < best_validation_loss:
                    best_validation_loss = last_validation_loss

            should_save = (
                current_step_in_run == 1
                or current_step_in_run % save_every_steps == 0
                or next_batch_index == steps_per_epoch
            )
            if should_save:
                save_checkpoint(
                    target,
                    payload_for_save(start_epoch, epoch_order, next_batch_index),
                )
                report(
                    "running",
                    start_epoch,
                    "Checkpoint saved; training and held-out validation metrics updated.",
                )

        start_epoch += 1
        epoch_order = None
        next_batch_index = 0
        # Epoch boundary is always a durable resume point, even when save_every_steps is large.
        save_checkpoint(target, payload_for_save(start_epoch, None, 0))
        report(
            "running" if start_epoch < target_epoch else "completed",
            start_epoch,
            "Completed one full shuffled pass through the training token blocks.",
        )

    return report("completed", start_epoch, "Training finished. Model, tokenizer, optimizer, progress, and validation metrics are saved locally.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain.train",
        description="Train SYNTRA's local Transformer from randomly initialized weights.",
    )
    parser.add_argument("--dataset", action="append", default=None, help="Local UTF-8 text/code file or project directory; repeat to add sources. Defaults to dataset.txt.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Local checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=1, help="Additional full passes through shuffled training blocks.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-bytes", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--max-dataset-bytes", type=int, default=MAX_DATASET_BYTES)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true", help="Resume the exact dataset, model, optimizer, and epoch position from the checkpoint.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.width,
        dropout=args.dropout,
    )
    try:
        result = train(
            dataset_paths=args.dataset or (DEFAULT_DATASET,),
            checkpoint_path=args.checkpoint,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            tokenizer_vocab_size=args.vocab_size,
            tokenizer_bytes=args.tokenizer_bytes,
            max_dataset_bytes=args.max_dataset_bytes,
            evaluation_batches=args.evaluation_batches,
            save_every_steps=args.save_every_steps,
            device_name=args.device,
            seed=args.seed,
            resume=args.resume,
            callback=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"SYNTRA training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

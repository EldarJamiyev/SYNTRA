"""Supervised prompt/completion fine-tuning with the checkpoint's exact BPE tokenizer."""
from __future__ import annotations
from pathlib import Path
from time import monotonic
from typing import Any, Callable
import hashlib
import json
import math
import os
import re
import stat
import unicodedata

try:
    import torch
    from .model import SyntraTransformer
except ImportError:
    torch = None  # type: ignore[assignment]
    SyntraTransformer = None  # type: ignore[assignment]

from .config import ModelConfig
from .data import load_corpus
from .resources import validate_checkpoint_load_budget, validate_training_memory_budget
from .runtime import _normalize_checkpoint_payload, checkpoint_path, clear_resident_cache
from .safe_paths import (
    PROJECT_ROOT, _approved_roots, resolve_corpus_root,
    resolve_instruction_dataset, resolve_curriculum_output_path,
)
from .tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer
from .training import (
    _atomic_checkpoint, _atomic_json, _save_inference_checkpoint, choose_device, select_inference_checkpoint,
    append_training_history, validate_training_history,
)


MAX_INSTRUCTION_EXAMPLES = 10_000
MAX_INSTRUCTION_WINDOWS = 100_000
MAX_INSTRUCTION_TOKENS = 3_000_000
MAX_INSTRUCTION_FILE_BYTES = 100 * 1024 * 1024
MAX_INSTRUCTION_RECORD_BYTES = 1024 * 1024
MAX_INSTRUCTION_FIELD_BYTES = 512 * 1024
INSTRUCTION_VALIDATION_FRACTION = 0.1
INSTRUCTION_SPLIT_SEED = "syntra-instruction-split-v1"


def _normalized_instruction_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_instruction_examples(
    dataset_path: str | Path,
    seq_len: int,
    tokenizer: ByteBPETokenizer | None = None,
    max_examples: int = MAX_INSTRUCTION_EXAMPLES,
    max_tokens: int = MAX_INSTRUCTION_TOKENS,
) -> list[dict]:
    # Direct library callers receive the same approved-root protection as HTTP
    # and CLI entry points; a caller cannot bypass the dataset path boundary.
    path = os.path.realpath(
        os.fspath(resolve_instruction_dataset(dataset_path)), strict=True
    )
    # Keep a local canonical-root guard adjacent to the file-open sink. The
    # resolver already enforces policy; this repeat check makes the constraint
    # explicit where this function opens user-selected JSONL data.
    path_norm = os.path.normcase(os.path.normpath(path))
    approved_dataset = False
    for trusted_root in _approved_roots():
        root_norm = os.path.normcase(os.path.normpath(
            os.path.realpath(trusted_root, strict=True)
        ))
        root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
        if path_norm.startswith(root_prefix):
            approved_dataset = True
            break
    if not approved_dataset:
        raise ValueError("The instruction dataset must remain inside an approved local root.")
    if type(seq_len) is not int or seq_len < 16 or seq_len > 16384:
        raise ValueError("Instruction context must be between 16 and 16,384 tokens.")
    if type(max_examples) is not int or max_examples < 2 or max_examples > MAX_INSTRUCTION_EXAMPLES:
        raise ValueError(f"max_examples must be between 2 and {MAX_INSTRUCTION_EXAMPLES:,}.")
    if type(max_tokens) is not int or max_tokens < 1024 or max_tokens > MAX_INSTRUCTION_TOKENS:
        raise ValueError(f"max_tokens must be between 1,024 and {MAX_INSTRUCTION_TOKENS:,}.")
    tokenizer = tokenizer or ByteBPETokenizer.empty()
    examples: list[dict] = []
    total_tokens = 0
    source_record_count = 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            file_info = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_info.st_mode):
                raise ValueError("Instruction dataset must be an existing regular file.")
            if file_info.st_size > MAX_INSTRUCTION_FILE_BYTES:
                raise ValueError(
                    f"Instruction JSONL is limited to {MAX_INSTRUCTION_FILE_BYTES // (1024 * 1024)} MiB."
                )
            raw = handle.read(MAX_INSTRUCTION_FILE_BYTES + 1)
            after = os.fstat(handle.fileno())
            if (
                len(raw) > MAX_INSTRUCTION_FILE_BYTES
                or (file_info.st_dev, file_info.st_ino) != (after.st_dev, after.st_ino)
                or file_info.st_size != after.st_size
                or file_info.st_mtime_ns != after.st_mtime_ns
                or file_info.st_ctime_ns != after.st_ctime_ns
            ):
                raise ValueError("Instruction dataset changed while it was being read or exceeded its file-size limit.")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"Cannot safely read instruction dataset: {exc}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    try:
        dataset_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Instruction dataset must contain valid UTF-8 JSONL.") from None

    for line_number, line in enumerate(dataset_text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES:
            raise ValueError(f"Instruction record on line {line_number} exceeds the 1 MiB record limit.")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("prompt"), str)
            or not isinstance(item.get("completion"), str)
        ):
            raise ValueError(f"Line {line_number} must contain string fields named prompt and completion.")
        prompt_text, completion_text = item["prompt"], item["completion"]
        if not prompt_text.strip() or not completion_text.strip():
            continue
        if (
            len(prompt_text.encode("utf-8")) > MAX_INSTRUCTION_FIELD_BYTES
            or len(completion_text.encode("utf-8")) > MAX_INSTRUCTION_FIELD_BYTES
        ):
            raise ValueError(f"Prompt/completion on line {line_number} exceeds the 512 KiB field limit.")

        source_record_count += 1
        if source_record_count > max_examples:
            raise ValueError(
                f"Instruction dataset exceeds the {max_examples:,}-source-record safety limit."
            )

        normalized_prompt = _normalized_instruction_text(prompt_text)
        normalized_completion = _normalized_instruction_text(completion_text)
        prompt_group_id = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
        split_group_value = item.get("split_group")
        if split_group_value is not None and (
            not isinstance(split_group_value, str)
            or len(split_group_value) > 2048
        ):
            raise ValueError(f"split_group on line {line_number} must be a string of at most 2,048 characters.")
        split_group_id = (
            hashlib.sha256(_normalized_instruction_text(split_group_value).encode("utf-8")).hexdigest()
            if isinstance(split_group_value, str) and split_group_value.strip()
            else None
        )

        prompt = tokenizer.encode(prompt_text)
        completion = tokenizer.encode(completion_text)
        # A prompt is the conditioning contract for its completion. Silently
        # keeping only its tail can remove system constraints, task details,
        # or required output formats. Fail closed and require the dataset
        # producer/operator to shorten the prompt or select a larger model.
        if len(prompt) > seq_len - 1:
            raise ValueError(
                f"Prompt on line {line_number} requires {len(prompt)} tokens, "
                f"but the selected checkpoint can retain at most {seq_len - 1} "
                "prompt tokens after reserving BOS. The prompt was not truncated; "
                "shorten it or use a checkpoint with a larger context window."
            )
        sequence = [BOS_ID, *prompt, *completion, EOS_ID]
        first_response_target = 1 + len(prompt)
        source_example_id = hashlib.sha256(
            (normalized_prompt + "\0" + normalized_completion).encode("utf-8")
        ).hexdigest()
        source_row_id = hashlib.sha256(
            (source_example_id + "\0line:" + str(line_number)).encode("utf-8")
        ).hexdigest()

        stride = max(1, seq_len // 2)
        start = max(0, first_response_target - seq_len)
        last_supervised_target = first_response_target - 1
        chunk_index = 0
        while start < len(sequence) - 1 and last_supervised_target < len(sequence) - 1:
            window = sequence[start:start + seq_len + 1]
            if len(window) < 2:
                break
            input_ids = window[:-1]
            labels = window[1:]
            prompt_target_count = 0
            overlap_target_count = 0
            for position in range(len(labels)):
                target_position = start + position + 1
                if target_position < first_response_target:
                    prompt_target_count += 1
                    labels[position] = -100
                elif target_position <= last_supervised_target:
                    overlap_target_count += 1
                    labels[position] = -100

            if any(token != -100 for token in labels):
                if len(examples) >= MAX_INSTRUCTION_WINDOWS:
                    raise ValueError(
                        f"Instruction dataset exceeds the {MAX_INSTRUCTION_WINDOWS:,}-context-window safety limit."
                    )
                example_tokens = len(input_ids) + len(labels)
                if total_tokens + example_tokens > max_tokens:
                    raise ValueError(
                        f"Instruction dataset exceeds the {max_tokens:,}-token safety cap at line {line_number}. "
                        "Reduce max_examples or shorten the prompt/completion examples."
                    )
                chunk_example_id = hashlib.sha256(
                    (source_example_id + "\0" + str(chunk_index)).encode("utf-8")
                ).hexdigest()
                examples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "prompt_tokens": prompt_target_count,
                    "completion_tokens": sum(token != -100 for token in labels),
                    "context_overlap_tokens": overlap_target_count,
                    "prompt_group_id": prompt_group_id,
                    "split_group_id": split_group_id,
                    "source_example_id": source_example_id,
                    "source_row_id": source_row_id,
                    "example_id": chunk_example_id,
                    "line": line_number,
                    "chunk_index": chunk_index,
                })
                total_tokens += example_tokens
                last_supervised_target = start + len(labels)
            start += stride
            chunk_index += 1

    if not examples:
        raise ValueError("Instruction dataset contains no non-empty prompt/completion examples.")
    return examples


def _source_record_index(examples: list[dict]) -> dict[str, dict[str, Any]]:
    """Collapse overlapping context windows back to their originating JSONL records."""
    records: dict[str, dict[str, Any]] = {}
    for example in examples:
        example_id = example.get("example_id")
        if not isinstance(example_id, str) or len(example_id) != 64:
            raise ValueError("Instruction context window is missing its exact example hash.")
        row_id = example.get("source_row_id", example_id)
        source_id = example.get("source_example_id", example_id)
        if not isinstance(row_id, str) or len(row_id) != 64:
            raise ValueError("Instruction context window has a malformed source-row identifier.")
        if not isinstance(source_id, str) or len(source_id) != 64:
            raise ValueError("Instruction context window has a malformed source-example identifier.")
        record = {
            "source_example_id": source_id,
            "prompt_group_id": example["prompt_group_id"],
            "split_group_id": example.get("split_group_id"),
        }
        previous = records.get(row_id)
        if previous is not None and previous != record:
            raise ValueError("Context windows from one source row contain inconsistent record metadata.")
        records[row_id] = record
    return records


def split_instruction_examples(
    examples: list[dict],
    validation_fraction: float = INSTRUCTION_VALIDATION_FRACTION,
    seed: str = INSTRUCTION_SPLIT_SEED,
) -> tuple[list[dict], list[dict], dict]:
    """Split duplicate prompts and explicitly related curriculum records together."""
    if not isinstance(examples, list) or len(examples) < 2:
        raise ValueError("At least two instruction context windows are required for a train/validation split.")
    if not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5.")
    if not isinstance(seed, str) or not seed.strip() or len(seed) > 128:
        raise ValueError("Instruction split seed must be a non-empty string of at most 128 characters.")

    parents = list(range(len(examples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    owners: dict[str, int] = {}
    record_keys: list[tuple[str, ...]] = []
    source_group_ids: set[str] = set()

    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError("Instruction examples must be dictionaries returned by load_instruction_examples.")
        prompt_group_id = example.get("prompt_group_id")
        split_group_id = example.get("split_group_id")
        if not isinstance(prompt_group_id, str) or len(prompt_group_id) != 64:
            raise ValueError("Instruction example is missing its normalized prompt-group hash.")
        if split_group_id is not None and (
            not isinstance(split_group_id, str) or len(split_group_id) != 64
        ):
            raise ValueError("Instruction example has a malformed source-group hash.")

        keys = ["prompt:" + prompt_group_id]
        if split_group_id:
            keys.append("source:" + split_group_id)
            source_group_ids.add(split_group_id)
        record_keys.append(tuple(keys))
        for key in keys:
            previous = owners.get(key)
            if previous is None:
                owners[key] = index
            else:
                union(index, previous)

    source_records = _source_record_index(examples)
    component_members: dict[int, list[int]] = {}
    component_keys: dict[int, set[str]] = {}
    for index, keys in enumerate(record_keys):
        root = find(index)
        component_members.setdefault(root, []).append(index)
        component_keys.setdefault(root, set()).update(keys)

    groups: dict[str, list[dict]] = {}
    for root, members in component_members.items():
        material = "\0".join(sorted(component_keys[root]))
        group_id = hashlib.sha256(material.encode("utf-8")).hexdigest()
        groups[group_id] = [examples[index] for index in members]

    if len(groups) < 2:
        raise ValueError(
            "At least two distinct normalized prompts and two independent prompt/source groups are required; "
            "related curriculum examples must remain together in one split."
        )

    validation_group_count = max(1, int(len(groups) * validation_fraction))
    validation_group_count = min(validation_group_count, len(groups) - 1)
    ranked_groups = sorted(
        groups,
        key=lambda group_id: hashlib.sha256((seed + ":" + group_id).encode("utf-8")).hexdigest(),
    )
    validation_group_ids = set(ranked_groups[:validation_group_count])
    training_examples = [
        example
        for group_id, members in groups.items()
        if group_id not in validation_group_ids
        for example in members
    ]
    validation_examples = [
        example
        for group_id, members in groups.items()
        if group_id in validation_group_ids
        for example in members
    ]
    if not training_examples or not validation_examples:
        raise ValueError("Instruction split did not produce non-empty train and validation sets.")

    training_row_ids = {
        item.get("source_row_id", item["example_id"]) for item in training_examples
    }
    validation_row_ids = {
        item.get("source_row_id", item["example_id"]) for item in validation_examples
    }
    training_records = {key: source_records[key] for key in training_row_ids}
    validation_records = {key: source_records[key] for key in validation_row_ids}

    unique_prompts = len({item["prompt_group_id"] for item in source_records.values()})
    source_record_count = len(source_records)
    unique_source_examples = len({
        item["source_example_id"] for item in source_records.values()
    })
    duplicate_prompts = source_record_count - unique_prompts
    exact_duplicates = source_record_count - unique_source_examples
    training_prompt_ids = {item["prompt_group_id"] for item in training_records.values()}
    validation_prompt_ids = {item["prompt_group_id"] for item in validation_records.values()}
    prompt_leakage_groups = len(training_prompt_ids.intersection(validation_prompt_ids))
    validation_fraction_actual = len(validation_records) / source_record_count
    warnings = []
    if source_record_count < 100:
        warnings.append("Small dataset: fewer than 100 source records; validation loss is weak evidence of generalization.")
    if unique_prompts < 20:
        warnings.append("Low prompt diversity: fewer than 20 distinct normalized prompts.")
    if duplicate_prompts:
        warnings.append(
            f"{duplicate_prompts} source record(s) reuse an existing normalized prompt; all copies stay in one split."
        )
    if source_group_ids:
        warnings.append(
            "Curriculum source groups are kept intact across validation to reduce same-symbol/source leakage."
        )
    if exact_duplicates:
        warnings.append(
            f"{exact_duplicates} exact prompt/completion duplicate(s) may overweight repeated records during training."
        )
    if validation_fraction_actual < 0.05 or validation_fraction_actual > 0.3:
        warnings.append(
            "The prompt/source-group constraint makes the actual validation fraction differ substantially from the requested fraction."
        )

    quality = {
        "example_count": source_record_count,
        "context_window_count": len(examples),
        "unique_prompt_count": unique_prompts,
        "source_group_count": len(source_group_ids),
        "independent_split_group_count": len(groups),
        "repeated_prompt_example_count": duplicate_prompts,
        "exact_duplicate_example_count": exact_duplicates,
        "training_examples": len(training_records),
        "validation_examples": len(validation_records),
        "training_context_windows": len(training_examples),
        "validation_context_windows": len(validation_examples),
        "training_prompt_groups": len(training_prompt_ids),
        "validation_prompt_groups": len(validation_prompt_ids),
        "training_data_groups": len(groups) - len(validation_group_ids),
        "validation_data_groups": len(validation_group_ids),
        "validation_fraction": round(validation_fraction_actual, 4),
        "requested_validation_fraction": float(validation_fraction),
        "split_strategy": "deterministic-sha256-prompt-and-source-groups-v2",
        "split_seed": seed,
        "prompt_leakage_groups": prompt_leakage_groups,
        "warnings": warnings,
    }
    return training_examples, validation_examples, quality


def audit_instruction_dataset(
    dataset_path: str | Path,
    seq_len: int = 4096,
    tokenizer: ByteBPETokenizer | None = None,
    max_examples: int = MAX_INSTRUCTION_EXAMPLES,
    max_tokens: int = MAX_INSTRUCTION_TOKENS,
) -> dict:
    """Return a read-only structural audit of a local prompt/completion JSONL file."""
    tokenizer_basis = "checkpoint-tokenizer" if tokenizer is not None else "byte-fallback"
    try:
        examples = load_instruction_examples(
            dataset_path, seq_len, tokenizer=tokenizer,
            max_examples=max_examples, max_tokens=max_tokens
        )
        source_records = _source_record_index(examples)
    except (OSError, ValueError) as error:
        return {
            "valid": False, "split_available": False, "example_count": 0,
            "tokenizer_basis": tokenizer_basis, "error": str(error), "warnings": [],
        }

    unique_prompt_count = len({item["prompt_group_id"] for item in source_records.values()})
    unique_example_count = len({
        item["source_example_id"] for item in source_records.values()
    })
    repeated_prompt_count = len(source_records) - unique_prompt_count
    exact_duplicate_count = len(source_records) - unique_example_count
    warnings = []
    if len(source_records) < 100:
        warnings.append("Small dataset: fewer than 100 source records; validation loss is weak evidence of generalization.")
    if unique_prompt_count < 20:
        warnings.append("Low prompt diversity: fewer than 20 distinct normalized prompts.")
    if repeated_prompt_count:
        warnings.append(
            f"{repeated_prompt_count} source record(s) reuse an existing normalized prompt; all copies stay in one split."
        )
    if exact_duplicate_count:
        warnings.append(
            f"{exact_duplicate_count} exact prompt/completion duplicate(s) may overweight repeated source records during training."
        )

    try:
        _, _, split_quality = split_instruction_examples(examples)
    except ValueError as error:
        warnings.append(str(error))
        return {
            "valid": True,
            "split_available": False,
            "example_count": len(source_records),
            "context_window_count": len(examples),
            "unique_prompt_count": unique_prompt_count,
            "repeated_prompt_example_count": repeated_prompt_count,
            "exact_duplicate_example_count": exact_duplicate_count,
            "training_examples": 0,
            "validation_examples": 0,
            "training_context_windows": 0,
            "validation_context_windows": 0,
            "validation_fraction": 0.0,
            "split_strategy": "deterministic-sha256-prompt-and-source-groups-v2",
            "split_seed": INSTRUCTION_SPLIT_SEED,
            "prompt_leakage_groups": None,
            "tokenizer_basis": tokenizer_basis,
            "tokenization_note": "Byte-level fallback used for audit unless a checkpoint tokenizer is supplied; fine-tuning uses the checkpoint tokenizer.",
            "error": str(error),
            "warnings": warnings,
        }
    return {
        "valid": True, "split_available": True, **split_quality,
        "tokenizer_basis": tokenizer_basis,
        "tokenization_note": "Byte-level fallback used for audit unless a checkpoint tokenizer is supplied; fine-tuning uses the checkpoint tokenizer.",
    }


def _instruction_batch_from_examples(chosen: list[dict], device: Any):
    if not chosen:
        raise ValueError("An instruction micro-batch must contain at least one example.")
    length = max(len(example["input_ids"]) for example in chosen)
    inputs, targets = [], []
    for example in chosen:
        padding = length - len(example["input_ids"])
        inputs.append(example["input_ids"] + [EOS_ID] * padding)
        targets.append(example["labels"] + [-100] * padding)
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def _sample_instruction_examples(examples: list[dict], batch_size: int) -> list[dict]:
    if not examples:
        raise ValueError("Instruction training requires at least one example.")
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    return [examples[index] for index in indices]


def _instruction_batch(examples: list[dict], batch_size: int, device: Any):
    return _instruction_batch_from_examples(
        _sample_instruction_examples(examples, batch_size),
        device,
    )


def _instruction_loss(model, examples: list[dict], device: Any, batch_size: int, batches: int = 4) -> float:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    try:
        with torch.no_grad():
            for _ in range(batches):
                x, y = _instruction_batch(examples, batch_size, device)
                valid_count = int((y != -100).sum().item())
                if valid_count < 1:
                    raise ValueError("Validation batch contained no supervised completion tokens.")
                _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss).item():
                    raise RuntimeError("Validation produced a non-finite instruction loss.")
                weighted_loss += float(loss.item()) * valid_count
                supervised_tokens += valid_count
    finally:
        # Keep this helper safe for both training-time validation and standalone
        # evaluation of a model that may be resident in the inference process.
        model.train(was_training)
    if supervised_tokens < 1:
        raise ValueError("Validation did not process any supervised completion tokens.")
    return weighted_loss / supervised_tokens


def finetune(
    dataset_path: str | Path,
    steps: int = 1000,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
    checkpoint: str | Path | None = None,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    if type(steps) is not int or steps < 1 or steps > 1_000_000:
        raise ValueError("steps must be an integer between 1 and 1,000,000.")
    if type(batch_size) is not int or batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size must be an integer between 1 and 128.")
    if type(gradient_accumulation_steps) is not int or not 1 <= gradient_accumulation_steps <= 128:
        raise ValueError("gradient_accumulation_steps must be an integer between 1 and 128.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 1e-6 <= learning_rate <= 1e-2
    ):
        raise ValueError("learning_rate must be a finite number between 1e-6 and 1e-2.")
    checkpoint_path_value = Path(checkpoint).expanduser().resolve() if checkpoint else checkpoint_path()
    if not checkpoint_path_value.is_file():
        raise FileNotFoundError("Train a SYNTRA checkpoint from local source code before instruction fine-tuning.")
    device = choose_device(device_name)
    load_target = select_inference_checkpoint(checkpoint_path_value)
    validate_checkpoint_load_budget(load_target)
    # Read model-only checkpoint tensors on CPU first; run the resource guard before
    # moving weights, gradients, and optimizer state to an accelerator.
    raw_saved = torch.load(load_target, map_location="cpu", weights_only=True)
    saved = _normalize_checkpoint_payload(raw_saved)
    config = ModelConfig.from_dict(saved["config"])
    config.validate()
    # Accumulation keeps only one physical micro-batch and its activations on
    # the device at a time; do not budget it as one combined physical batch.
    estimated_training_memory_bytes = validate_training_memory_budget(
        config, batch_size, device
    )
    tokenizer = ByteBPETokenizer.from_dict(saved.get("tokenizer"))
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint tokenizer vocabulary does not match the model architecture.")
    model = SyntraTransformer(config).to(device)
    model.load_state_dict(saved["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    examples = load_instruction_examples(dataset_path, config.seq_len, tokenizer)
    training_examples, validation_examples, dataset_quality = split_instruction_examples(examples)
    base_step = int(saved.get("step", 0))
    training_history = validate_training_history(saved.get("training_history"))
    from .hardware_telemetry import (
        device_memory_sample, process_peak_rss_bytes, reset_device_peak_stats,
    )

    started = monotonic()
    reset_device_peak_stats(device, torch)
    last_train_loss = None
    last_val_loss = None
    supervised_tokens_processed = 0
    last_step_latency_ms = None
    step_latency_total_ms = 0.0
    peak_device_memory_bytes = None
    device_memory_metric = "not_sampled"

    def report(status: str, step: int, message: str) -> dict:
        value = {
            "status": status,
            "step": step,
            "max_steps": base_step + steps,
            "train_loss": last_train_loss,
            "validation_loss": last_val_loss,
            "device": str(device),
            "elapsed_seconds": round(monotonic() - started, 2),
            "tokens_per_second": round(supervised_tokens_processed / max(monotonic() - started, 1e-9), 2),
            "supervised_tokens_processed": supervised_tokens_processed,
            "step_latency_ms": last_step_latency_ms,
            "mean_step_latency_ms": round(step_latency_total_ms / max(1, step - base_step), 3),
            "peak_rss_bytes": process_peak_rss_bytes(),
            "peak_rss_scope": "process_lifetime_high_water",
            "peak_device_memory_bytes": peak_device_memory_bytes,
            "throughput_scope": "stage_wall_clock_including_validation_and_checkpoint_io",
            "device_memory_metric": device_memory_metric,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "estimated_training_memory_bytes": estimated_training_memory_bytes,
            "message": message,
            "checkpoint": str(checkpoint_path_value),
            "training_mode": "supervised-instruction-finetuning",
            "instruction_examples": dataset_quality["example_count"],
            "instruction_context_windows": len(examples),
            "instruction_dataset_quality": dataset_quality,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "context_tokens": config.seq_len,
            "training_history_points": len(training_history),
        }
        if callback:
            callback(value)
        return value

    model.train()
    report(
        "running", base_step,
        f"Supervised fine-tuning started with {len(training_examples)} training and "
        f"{len(validation_examples)} validation context windows across "
        f"{dataset_quality['training_examples']} training and "
        f"{dataset_quality['validation_examples']} validation source records. "
        + ("Dataset warnings: " + " ".join(dataset_quality["warnings"]) if dataset_quality["warnings"] else "No basic dataset-quality warnings.")
    )
    for step in range(base_step + 1, base_step + steps + 1):
        from time import perf_counter

        step_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss_total = 0.0
        # Keep only Python token lists for the accumulation group in host memory.
        # Materialize each tensor pair on the selected device immediately before
        # forward/backward, then release it before allocating the next micro-batch.
        sampled_micro_batches = [
            _sample_instruction_examples(training_examples, batch_size)
            for _micro_step in range(gradient_accumulation_steps)
        ]
        micro_token_counts = [
            sum(sum(token != -100 for token in example["labels"]) for example in chosen)
            for chosen in sampled_micro_batches
        ]
        batch_supervised_tokens = sum(micro_token_counts)
        if batch_supervised_tokens < 1:
            raise RuntimeError("SFT optimizer step contains no supervised response/EOS targets.")
        for chosen, micro_tokens in zip(sampled_micro_batches, micro_token_counts):
            if micro_tokens < 1:
                raise RuntimeError("SFT micro-batch contains no supervised response/EOS targets.")
            x, y = _instruction_batch_from_examples(chosen, device)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training loss is invalid; check the instruction dataset.")
            weighted_loss_total += float(loss.detach().item()) * micro_tokens
            (loss * (micro_tokens / batch_supervised_tokens)).backward()
            sampled_memory, device_memory_metric = device_memory_sample(device, torch)
            if sampled_memory is not None:
                peak_device_memory_bytes = max(peak_device_memory_bytes or 0, sampled_memory)
            del x, y, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        last_train_loss = weighted_loss_total / max(1, batch_supervised_tokens)
        supervised_tokens_processed += batch_supervised_tokens
        last_step_latency_ms = (perf_counter() - step_started) * 1000.0
        step_latency_total_ms += last_step_latency_ms
        sampled_memory, device_memory_metric = device_memory_sample(device, torch)
        if sampled_memory is not None:
            peak_device_memory_bytes = max(peak_device_memory_bytes or 0, sampled_memory)
        if step == base_step + 1 or step % 25 == 0 or step == base_step + steps:
            last_val_loss = _instruction_loss(model, validation_examples, device, batch_size, batches=2)
            training_history = append_training_history(
                training_history,
                step=step,
                phase="supervised-instruction-finetuning",
                train_loss=last_train_loss,
                validation_loss=last_val_loss,
            )
            payload = {
                **saved,
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "training_mode": "supervised-instruction-finetuning",
                "instruction_examples": dataset_quality["example_count"],
                "training_history": training_history,
                "instruction_context_windows": len(examples),
                "instruction_dataset_quality": dataset_quality,
                "train_loss": last_train_loss,
                "validation_loss": last_val_loss,
            }
            _atomic_checkpoint(checkpoint_path_value, payload)
            _save_inference_checkpoint(checkpoint_path_value, payload)
            clear_resident_cache()
            metadata_path = checkpoint_path_value.with_suffix(checkpoint_path_value.suffix + ".json")
            metadata = {
                "checkpoint": str(checkpoint_path_value), "step": step, "config": config.to_dict(),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "train_loss": last_train_loss, "validation_loss": last_val_loss,
                "tokenizer": "byte-BPE", "tokenizer_vocab_size": tokenizer.vocab_size,
                "context_tokens": config.seq_len,
                "training_mode": "supervised-instruction-finetuning",
                "instruction_examples": dataset_quality["example_count"],
                "instruction_context_windows": len(examples),
                "instruction_dataset_quality": dataset_quality,
                "pretrained_weights": False,
                "training_history": training_history,
                "training_history_points": len(training_history),
            }
            _atomic_json(metadata_path, metadata)
            report("running" if step < base_step + steps else "completed", step, "Checkpoint saved with measured supervised validation loss.")
    return report("completed", base_step + steps, "Supervised fine-tuning finished; the local BPE-tokenized checkpoint was updated.")


def synthesize_repository_curriculum(
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    max_examples: int = 240,
) -> dict:
    """Compatibility entry point for the dedicated repository curriculum engine."""
    from .curriculum import synthesize_repository_curriculum as generate_curriculum

    return generate_curriculum(root=root, output_path=output_path, max_examples=max_examples)

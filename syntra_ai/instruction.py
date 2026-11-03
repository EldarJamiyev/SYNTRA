"""Supervised prompt/completion fine-tuning with the checkpoint's exact BPE tokenizer."""
from __future__ import annotations
from pathlib import Path
from time import monotonic
from typing import Any, Callable
import hashlib
import json
import re
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
from .safe_paths import PROJECT_ROOT, resolve_corpus_root
from .tokenizer import BOS_ID, EOS_ID, ByteBPETokenizer
from .training import (
    _atomic_checkpoint, _save_inference_checkpoint, choose_device, select_inference_checkpoint,
)


MAX_INSTRUCTION_EXAMPLES = 10_000
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
            normalized_prompt = _normalized_instruction_text(prompt_text)
            normalized_completion = _normalized_instruction_text(completion_text)
            prompt_group_id = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
            example_id = hashlib.sha256(
                (normalized_prompt + "\\0" + normalized_completion).encode("utf-8")
            ).hexdigest()
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
                "prompt_group_id": prompt_group_id,
                "example_id": example_id,
                "line": line_number,
            })
            total_tokens += example_tokens
            if len(examples) >= max_examples:
                break
    if len(examples) < 1:
        raise ValueError("Instruction dataset contains no non-empty prompt/completion examples.")
    return examples


def split_instruction_examples(
    examples: list[dict],
    validation_fraction: float = INSTRUCTION_VALIDATION_FRACTION,
    seed: str = INSTRUCTION_SPLIT_SEED,
) -> tuple[list[dict], list[dict], dict]:
    """Split by normalized prompt groups so repeated prompts cannot leak across validation."""
    if not isinstance(examples, list) or len(examples) < 2:
        raise ValueError("At least two instruction examples are required for a train/validation split.")
    if not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5.")
    if not isinstance(seed, str) or not seed.strip() or len(seed) > 128:
        raise ValueError("Instruction split seed must be a non-empty string of at most 128 characters.")

    groups: dict[str, list[dict]] = {}
    example_ids = []
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Instruction examples must be dictionaries returned by load_instruction_examples.")
        prompt_group_id = example.get("prompt_group_id")
        example_id = example.get("example_id")
        if not isinstance(prompt_group_id, str) or len(prompt_group_id) != 64:
            raise ValueError("Instruction example is missing its normalized prompt-group hash.")
        if not isinstance(example_id, str) or len(example_id) != 64:
            raise ValueError("Instruction example is missing its exact example hash.")
        groups.setdefault(prompt_group_id, []).append(example)
        example_ids.append(example_id)

    if len(groups) < 2:
        raise ValueError(
            "At least two distinct normalized prompts are required; repeated copies of one prompt "
            "cannot form an independent validation set."
        )

    validation_group_count = max(1, int(len(groups) * validation_fraction))
    validation_group_count = min(validation_group_count, len(groups) - 1)
    ranked_groups = sorted(
        groups,
        key=lambda group_id: hashlib.sha256((seed + ":" + group_id).encode("utf-8")).hexdigest()
    )
    validation_group_ids = set(ranked_groups[:validation_group_count])
    training_examples = [example for example in examples if example["prompt_group_id"] not in validation_group_ids]
    validation_examples = [example for example in examples if example["prompt_group_id"] in validation_group_ids]
    if not training_examples or not validation_examples:
        raise ValueError("Instruction split did not produce non-empty train and validation sets.")

    unique_examples = len(set(example_ids))
    unique_prompts = len(groups)
    duplicate_prompts = len(examples) - unique_prompts
    exact_duplicates = len(examples) - unique_examples
    validation_fraction_actual = len(validation_examples) / len(examples)
    warnings = []
    if len(examples) < 100:
        warnings.append("Small dataset: fewer than 100 examples; validation loss is weak evidence of generalization.")
    if unique_prompts < 20:
        warnings.append("Low prompt diversity: fewer than 20 distinct normalized prompts.")
    if duplicate_prompts:
        warnings.append(
            f"{duplicate_prompts} example(s) reuse an existing normalized prompt; all copies are kept in one split."
        )
    if exact_duplicates:
        warnings.append(
            f"{exact_duplicates} exact prompt/completion duplicate(s) may overweight repeated examples during training."
        )
    if validation_fraction_actual < 0.05 or validation_fraction_actual > 0.3:
        warnings.append(
            "The prompt-group constraint makes the actual validation fraction differ substantially from the requested fraction."
        )

    quality = {
        "example_count": len(examples),
        "unique_prompt_count": unique_prompts,
        "repeated_prompt_example_count": duplicate_prompts,
        "exact_duplicate_example_count": exact_duplicates,
        "training_examples": len(training_examples),
        "validation_examples": len(validation_examples),
        "training_prompt_groups": len({item["prompt_group_id"] for item in training_examples}),
        "validation_prompt_groups": len(validation_group_ids),
        "validation_fraction": round(validation_fraction_actual, 4),
        "requested_validation_fraction": float(validation_fraction),
        "split_strategy": "deterministic-sha256-prompt-groups",
        "split_seed": seed,
        "prompt_leakage_groups": 0,
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
    except (OSError, ValueError) as error:
        return {
            "valid": False, "split_available": False, "example_count": 0,
            "tokenizer_basis": tokenizer_basis, "error": str(error), "warnings": [],
        }

    unique_prompt_count = len({item["prompt_group_id"] for item in examples})
    unique_example_count = len({item["example_id"] for item in examples})
    repeated_prompt_count = len(examples) - unique_prompt_count
    exact_duplicate_count = len(examples) - unique_example_count
    warnings = []
    if len(examples) < 100:
        warnings.append("Small dataset: fewer than 100 examples; validation loss is weak evidence of generalization.")
    if unique_prompt_count < 20:
        warnings.append("Low prompt diversity: fewer than 20 distinct normalized prompts.")
    if repeated_prompt_count:
        warnings.append(f"{repeated_prompt_count} example(s) reuse an existing normalized prompt; all copies stay in one split.")
    if exact_duplicate_count:
        warnings.append(f"{exact_duplicate_count} exact prompt/completion duplicate(s) may overweight repeated examples during training.")

    try:
        _, _, split_quality = split_instruction_examples(examples)
    except ValueError as error:
        warnings.append(str(error))
        return {
            "valid": True, "split_available": False, "example_count": len(examples),
            "unique_prompt_count": unique_prompt_count,
            "repeated_prompt_example_count": repeated_prompt_count,
            "exact_duplicate_example_count": exact_duplicate_count,
            "training_examples": 0, "validation_examples": 0, "validation_fraction": 0.0,
            "split_strategy": "deterministic-sha256-prompt-groups",
            "split_seed": INSTRUCTION_SPLIT_SEED,
            "prompt_leakage_groups": 0 if unique_prompt_count < 2 else None,
            "tokenizer_basis": tokenizer_basis,
            "tokenization_note": "Byte-level fallback used for audit unless a checkpoint tokenizer is supplied; fine-tuning uses the checkpoint tokenizer.",
            "error": str(error), "warnings": warnings,
        }
    return {
        "valid": True, "split_available": True, **split_quality,
        "tokenizer_basis": tokenizer_basis,
        "tokenization_note": "Byte-level fallback used for audit unless a checkpoint tokenizer is supplied; fine-tuning uses the checkpoint tokenizer.",
    }

def _instruction_batch(examples: list[dict], batch_size: int, device: Any):
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    chosen = [examples[index] for index in indices]
    length = max(len(example["input_ids"]) for example in chosen)
    inputs, targets = [], []
    for example in chosen:
        padding = length - len(example["input_ids"])
        inputs.append(example["input_ids"] + [EOS_ID] * padding)
        targets.append(example["labels"] + [-100] * padding)
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def _instruction_loss(model, examples: list[dict], device: Any, batch_size: int, batches: int = 4) -> float:
    with torch.no_grad():
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
    load_target = select_inference_checkpoint(checkpoint_path_value)
    validate_checkpoint_load_budget(load_target)
    # Read model-only checkpoint tensors on CPU first; run the resource guard before
    # moving weights, gradients, and optimizer state to an accelerator.
    raw_saved = torch.load(load_target, map_location="cpu", weights_only=True)
    saved = _normalize_checkpoint_payload(raw_saved)
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
    training_examples, validation_examples, dataset_quality = split_instruction_examples(examples)
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
            "instruction_dataset_quality": dataset_quality,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "context_tokens": config.seq_len,
        }
        if callback:
            callback(value)
        return value

    model.train()
    report(
        "running", base_step,
        f"Supervised fine-tuning started with {len(training_examples)} training and "
        f"{len(validation_examples)} validation examples. "
        + ("Dataset warnings: " + " ".join(dataset_quality["warnings"]) if dataset_quality["warnings"] else "No basic dataset-quality warnings.")
    )
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
                "training_mode": "supervised-instruction-finetuning", "instruction_examples": len(examples),
                "instruction_dataset_quality": dataset_quality,
                "pretrained_weights": False,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            report("running" if step < base_step + steps else "completed", step, "Checkpoint saved with measured supervised validation loss.")
    return report("completed", base_step + steps, "Supervised fine-tuning finished; the local BPE-tokenized checkpoint was updated.")


_DECLARATION_PATTERNS = (
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)"),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)"),
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)"),
    re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)"),
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)"),
)


def _extract_symbol_blocks(rel_path: str, content: str) -> list[dict]:
    lines = content.splitlines()
    blocks: list[dict] = []
    for idx, line in enumerate(lines):
        for pattern in _DECLARATION_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1)
            params = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
            end_idx = min(len(lines), idx + 28)
            snippet_lines = [lines[idx]]
            brace_depth = line.count("{") - line.count("}")
            for cursor in range(idx + 1, end_idx):
                current = lines[cursor]
                if any(p.match(current) for p in _DECLARATION_PATTERNS) and len(snippet_lines) >= 4:
                    break
                snippet_lines.append(current)
                brace_depth += current.count("{") - current.count("}")
                if brace_depth <= 0 and current.strip().endswith("}") and len(snippet_lines) >= 3:
                    break
            body_text = "\n".join(snippet_lines).strip()
            if len(body_text) >= 32:
                blocks.append({
                    "name": name,
                    "params": params,
                    "line": idx + 1,
                    "signature": line.strip(),
                    "body": body_text[:1800],
                    "path": rel_path,
                })
            break
    return blocks


def synthesize_repository_curriculum(
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    max_examples: int = 240,
) -> dict:
    """Synthesize a high-diversity, deduplicated instruction-tuning JSONL curriculum from local code."""
    if not isinstance(max_examples, int) or max_examples < 4 or max_examples > MAX_INSTRUCTION_EXAMPLES:
        raise ValueError(f"max_examples must be between 4 and {MAX_INSTRUCTION_EXAMPLES:,}.")
    corpus_root = resolve_corpus_root(root or PROJECT_ROOT)
    if output_path:
        raw_out = Path(output_path).expanduser()
        if not raw_out.is_absolute():
            raw_out = (PROJECT_ROOT / raw_out).resolve()
        else:
            raw_out = raw_out.resolve()
    else:
        raw_out = (PROJECT_ROOT / "data" / "curricula" / "workspace-instructions.jsonl").resolve()
    if raw_out.suffix.lower() != ".jsonl":
        raise ValueError("Synthesized curriculum output path must end with .jsonl.")
    if not (raw_out == corpus_root or raw_out.is_relative_to(corpus_root) or raw_out.is_relative_to(PROJECT_ROOT)):
        raise ValueError("Curriculum output path must reside inside the active workspace or SYNTRA project directory.")

    corpus = load_corpus([corpus_root])
    documents, files = corpus.documents, corpus.files
    seen_prompts: set[str] = set()
    records: list[dict[str, str]] = []

    def add_pair(prompt_text: str, completion_text: str) -> None:
        if len(records) >= max_examples:
            return
        prompt_clean = prompt_text.strip()
        completion_clean = completion_text.strip()
        if len(prompt_clean) < 16 or len(completion_clean) < 16:
            return
        norm_prompt = _normalized_instruction_text(prompt_clean)
        if norm_prompt in seen_prompts:
            return
        seen_prompts.add(norm_prompt)
        records.append({"prompt": prompt_clean, "completion": completion_clean})

    for doc, file_str in zip(documents, files):
        if len(records) >= max_examples:
            break
        try:
            rel_path = Path(file_str).resolve().relative_to(corpus_root).as_posix()
        except ValueError:
            rel_path = Path(file_str).name
        content = doc.strip()
        if not content:
            continue
        lines = content.splitlines()
        blocks = _extract_symbol_blocks(rel_path, content)

        if blocks:
            symbol_list = ", ".join(f"`{b['name']}` (line {b['line']})" for b in blocks[:8])
            preview = "\n".join(lines[:min(18, len(lines))])
            add_pair(
                f"SYNTRA WORKSPACE QUERY\nSummarize the primary declarations and structure of `{rel_path}`.",
                f"`{rel_path}` defines {len(blocks)} primary symbol(s): {symbol_list}.\n\nModule header excerpt:\n```\n{preview}\n```",
            )

        for block in blocks:
            if len(records) >= max_examples:
                break
            add_pair(
                f"SYNTRA CODE IMPLEMENTATION\nFile: {rel_path}\nSymbol: {block['name']}\nWrite the implementation starting from `{block['signature']}`.",
                block["body"],
            )
            add_pair(
                f"SYNTRA SYMBOL COMPREHENSION\nExplain the signature and implementation of `{block['name']}` in `{rel_path}`.",
                f"In `{rel_path}` (line {block['line']}), `{block['signature']}` implements `{block['name']}`"
                + (f" accepting parameters ({block['params']})" if block["params"] else "")
                + f":\n```\n{block['body']}\n```",
            )
            block_lines = block["body"].splitlines()
            if len(block_lines) >= 3:
                mid = max(1, len(block_lines) // 2)
                anchor = "\n".join(block_lines[:mid])
                target = "\n".join(block_lines[:min(len(block_lines), mid + 3)])
                patch_obj = {
                    "summary": f"Complete `{block['name']}` implementation in {rel_path}",
                    "edits": [
                        {
                            "path": rel_path,
                            "find": anchor,
                            "replace": target,
                            "reason": f"Ensure `{block['name']}` includes the subsequent logic in {rel_path}.",
                        }
                    ],
                }
                add_pair(
                    f"SYNTRA_AGENT_PATCH_V1\nTASK: Complete the implementation of `{block['name']}` in `{rel_path}`.\nSOURCE ANCHOR ({rel_path}):\n{anchor}",
                    json.dumps(patch_obj, ensure_ascii=False),
                )

    if len(records) < 2:
        for idx, doc in enumerate(documents[:max_examples]):
            header, _, content = doc.partition("\n")
            rel_path = header.replace("FILE:", "").strip() if header.startswith("FILE:") else f"file-{idx + 1}"
            snippet = "\n".join(content.strip().splitlines()[:24])
            if snippet:
                add_pair(
                    f"Provide the opening source excerpt for `{rel_path}` (section {idx + 1}).",
                    snippet,
                )

    if len(records) < 2:
        raise ValueError("Could not extract at least two distinct instruction examples from the selected corpus.")

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    audit = audit_instruction_dataset(raw_out)
    return {
        "dataset_path": str(raw_out),
        "workspace_root": str(corpus_root),
        "source_files": len(files),
        "example_count": len(records),
        "audit": audit,
    }


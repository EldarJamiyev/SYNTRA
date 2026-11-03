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
import subprocess
import sys
from time import perf_counter, time_ns
from typing import Callable, Iterable

import torch
from torch.nn import functional as F

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




def _read_instruction_records(
    paths: Iterable[str | os.PathLike[str]],
    max_dataset_bytes: int,
) -> tuple[list[dict[str, str]], list[str], int, str]:
    """Read prepared JSONL records or synthesize records from approved project roots."""
    from .corpus import build_instruction_corpus
    from syntra_ai.safe_paths import resolve_corpus_root, resolve_instruction_dataset

    records: list[dict[str, str]] = []
    files: list[str] = []
    total_bytes = 0
    digest = sha256()
    seen_sources: set[str] = set()

    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError(f"Instruction corpus path does not exist: {candidate}") from None
        if resolved.is_dir():
            remaining = max_dataset_bytes - total_bytes
            if remaining < 1024:
                raise ValueError("Selected instruction datasets exceed the configured corpus byte limit.")
            corpus = build_instruction_corpus(
                [resolve_corpus_root(resolved)],
                max_total_bytes=remaining,
            )
            records.extend(corpus.records)
            files.extend(corpus.files)
            total_bytes += corpus.source_bytes
            digest.update(corpus.sha256.encode("ascii"))
            continue

        dataset = resolve_instruction_dataset(resolved)
        size = dataset.stat().st_size
        if size < 1 or size > max_dataset_bytes or total_bytes + size > max_dataset_bytes:
            raise ValueError("Prepared instruction JSONL exceeds the configured corpus byte limit.")
        raw = dataset.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"Instruction dataset must be valid UTF-8: {dataset}") from None
        digest.update(str(dataset).encode("utf-8"))
        digest.update(raw)
        total_bytes += len(raw)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            if len(records) >= 50_000:
                raise ValueError("Instruction dataset exceeds the 50,000-record limit.")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {dataset.name}:{line_number}: {exc.msg}") from None
            if not isinstance(item, dict):
                raise ValueError(f"Instruction record at {dataset.name}:{line_number} must be a JSON object.")
            prompt = item.get("prompt")
            completion = item.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(f"Instruction record at {dataset.name}:{line_number} requires string prompt/completion fields.")
            prompt, completion = prompt.strip(), completion.strip()
            if not prompt or not completion or len(prompt) > 32_000 or len(completion) > 32_000:
                raise ValueError(f"Instruction record at {dataset.name}:{line_number} is empty or exceeds the 32,000-character limit.")
            source = item.get("source")
            if not isinstance(source, str) or not source.strip():
                source = f"{dataset.name}#record-{line_number}"
            records.append({"prompt": prompt, "completion": completion, "source": source[:512]})
        files.append(str(dataset))

    if len(records) < 2:
        raise ValueError("Instruction training requires at least two valid prompt/completion records.")
    if len(records) > 50_000:
        raise ValueError("Instruction corpus exceeds the 50,000-record limit.")
    return records, files, total_bytes, digest.hexdigest()


def _split_instruction_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep all records from a source file in one split to reduce source leakage."""
    groups: dict[str, list[dict[str, str]]] = {}
    for index, record in enumerate(records):
        key = str(record.get("source") or f"record-{index}")
        groups.setdefault(key, []).append(record)
    keys = sorted(groups)
    if len(keys) >= 2:
        validation_count = max(1, len(keys) // 10)
        validation_keys = set(keys[-validation_count:])
        train_records = [item for key in keys if key not in validation_keys for item in groups[key]]
        validation_records = [item for key in keys if key in validation_keys for item in groups[key]]
    else:
        boundary = max(1, min(len(records) - 1, int(len(records) * 0.9)))
        train_records, validation_records = records[:boundary], records[boundary:]
    if not train_records or not validation_records:
        raise ValueError("Instruction dataset cannot provide non-empty training and held-out validation splits.")
    return train_records, validation_records


def _pack_instruction_records(
    records: list[dict[str, str]],
    tokenizer: BytePairTokenizer,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack prompt/completion examples; prompt tokens are masked from the loss."""
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    supervised_tokens = 0
    for record in records:
        prompt_ids = tokenizer.encode(record["prompt"])
        completion_ids = tokenizer.encode(record["completion"])
        if not completion_ids:
            continue
        completion_limit = max(1, block_size * 2 // 3)
        completion_ids = completion_ids[:completion_limit]
        prompt_budget = max(1, block_size - 1 - len(completion_ids))
        if len(prompt_ids) > prompt_budget:
            head = max(1, prompt_budget // 2)
            tail = max(0, prompt_budget - head)
            prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
        sequence = [BOS_ID, *prompt_ids, *completion_ids, EOS_ID]
        sequence = sequence[:block_size + 1]
        prompt_end_position = 1 + len(prompt_ids)
        row_input = sequence[:-1]
        row_target = [
            token if position + 1 > prompt_end_position else -100
            for position, token in enumerate(sequence[1:])
        ]
        if not any(token != -100 for token in row_target):
            continue
        supervised_tokens += sum(token != -100 for token in row_target)
        padding = block_size - len(row_input)
        row_input.extend([0] * padding)
        row_target.extend([-100] * padding)
        inputs.append(row_input)
        targets.append(row_target)
    if not inputs:
        raise ValueError("Instruction corpus produced no completion tokens inside the configured context window.")
    return (
        torch.tensor(inputs, dtype=torch.int32),
        torch.tensor(targets, dtype=torch.int32),
        supervised_tokens,
    )


def _instruction_batch(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_inputs = input_data.index_select(0, indices.to(dtype=torch.long))
    selected_targets = target_data.index_select(0, indices.to(dtype=torch.long))
    return (
        selected_inputs.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda"),
        selected_targets.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda"),
    )


@torch.no_grad()
def _estimate_instruction_validation_loss(
    model: LocalTransformer,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    device: torch.device,
    batch_size: int,
    batches: int,
) -> float:
    model.eval()
    count = input_data.shape[0]
    values: list[tuple[float, int]] = []
    order = torch.randperm(count, device="cpu")
    cursor = 0
    for _ in range(batches):
        if cursor >= count:
            cursor = 0
            order = torch.randperm(count, device="cpu")
        indices = order[cursor:min(count, cursor + batch_size)]
        cursor += len(indices)
        x, y = _instruction_batch(input_data, target_data, indices, device)
        logits, _ = model(x)
        valid_tokens = int((y != -100).sum().item())
        if valid_tokens < 1:
            continue
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=-100,
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Instruction validation produced a non-finite cross-entropy loss.")
        values.append((float(loss.item()), valid_tokens))
    model.train()
    if not values:
        raise RuntimeError("Instruction validation contained no supervised completion tokens.")
    total_tokens = sum(count for _, count in values)
    return sum(loss * count for loss, count in values) / total_tokens


def _run_epoch_benchmark(
    checkpoint_path: Path,
    epoch: int,
    task_ids: Iterable[str],
    timeout_seconds: int,
) -> dict:
    """Run the real coding-agent fixtures against the just-saved local checkpoint."""
    import re

    tasks = tuple(task_ids)
    if not tasks or len(tasks) > 4 or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", item) for item in tasks):
        return {"status": "skipped", "error": "Benchmark task selection is invalid.", "task_count": 0}
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 3600:
        return {"status": "skipped", "error": "Benchmark timeout must be between 60 and 3,600 seconds.", "task_count": len(tasks)}
    root = Path(__file__).resolve().parent.parent
    report_dir = root / "benchmark-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase4-epoch-{epoch}-{time_ns()}.json"
    command = [
        "npm", "run", "benchmark", "--",
        *[part for task in tasks for part in ("--task", task)],
        "--output", str(report_path),
    ]
    if os.name == "nt":
        command = ["cmd.exe", "/d", "/s", "/c", subprocess.list2cmdline(command)]
    environment = os.environ.copy()
    environment["SYNTRA_BRAIN_CHECKPOINT"] = str(checkpoint_path)
    environment["SYNTRA_BENCHMARK_NATIVE_ONLY"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "task_count": len(tasks),
            "completed_task_count": 0,
            "task_completion_rate": 0.0,
            "report_path": None,
            "error": (str(exc) or type(exc).__name__)[:1000],
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        return {
            "status": "completed",
            "task_count": int(summary["taskCount"]),
            "completed_task_count": int(summary["completedTaskCount"]),
            "task_completion_rate": float(summary["taskCompletionRate"]),
            "patch_validity_rate": float(summary["patchValidityRate"]),
            "verification_pass_rate": float(summary["verificationPassRate"]),
            "report_path": str(report_path),
            "process_exit_code": int(completed.returncode),
            "error": None if completed.returncode == 0 else (completed.stderr.strip()[-1000:] or "One or more fixture tasks did not pass."),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "status": "failed",
            "task_count": len(tasks),
            "completed_task_count": 0,
            "task_completion_rate": 0.0,
            "patch_validity_rate": 0.0,
            "verification_pass_rate": 0.0,
            "report_path": str(report_path) if report_path.exists() else None,
            "process_exit_code": int(completed.returncode),
            "error": (completed.stderr.strip()[-1000:] or completed.stdout.strip()[-1000:] or "Benchmark did not produce a valid report."),
        }


def train(
    dataset_paths: Iterable[str | os.PathLike[str]] = (DEFAULT_DATASET,),
    *,
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    config: TransformerConfig | None = None,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.05,
    tokenizer_vocab_size: int | None = None,
    tokenizer_bytes: int = 1_000_000,
    max_dataset_bytes: int = 128 * 1024 * 1024,
    evaluation_batches: int = 4,
    save_every_steps: int = 50,
    device_name: str = "auto",
    seed: int = 1337,
    resume: bool = False,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 10,
    min_learning_rate_ratio: float = 0.1,
    gradient_clip_norm: float = 1.0,
    instruction_mode: bool = False,
    run_benchmarks: bool = False,
    benchmark_tasks: Iterable[str] = ("utility-clamp", "python-slugify"),
    benchmark_timeout_seconds: int = 900,
    callback: ProgressCallback | None = None,
) -> dict:
    """Train or resume the local Transformer with next-token or masked instruction loss."""
    if type(epochs) is not int or not 1 <= epochs <= 100_000:
        raise ValueError("epochs must be an integer between 1 and 100,000.")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be an integer between 1 and 256.")
    if type(gradient_accumulation_steps) is not int or not 1 <= gradient_accumulation_steps <= 1024:
        raise ValueError("gradient_accumulation_steps must be between 1 and 1,024.")
    if type(warmup_steps) is not int or not 0 <= warmup_steps <= 100_000:
        raise ValueError("warmup_steps must be between 0 and 100,000.")
    if not math.isfinite(min_learning_rate_ratio) or not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be between 0 and 1.")
    if not math.isfinite(gradient_clip_norm) or not 0.1 <= gradient_clip_norm <= 10:
        raise ValueError("gradient_clip_norm must be between 0.1 and 10.")
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

    paths = tuple(paths_item for paths_item in dataset_paths)
    if not paths:
        raise ValueError("At least one local dataset file or project root is required.")
    if instruction_mode:
        records, dataset_files, dataset_bytes, dataset_hash = _read_instruction_records(paths, max_dataset_bytes)
        train_records, validation_records = _split_instruction_records(records)
        tokenizer_text = "\n".join(
            record["prompt"] + "\n" + record["completion"] for record in train_records
        )
        raw_train_text = raw_validation_text = None
    else:
        raw_train_text, raw_validation_text, dataset_files, dataset_bytes, dataset_hash = _read_local_datasets(
            paths, max_dataset_bytes
        )
        tokenizer_text = raw_train_text
        train_records = validation_records = []

    target = Path(checkpoint_path).expanduser().resolve()
    device = select_device(device_name)
    requested_config = config or TransformerConfig(
        vocab_size=tokenizer_vocab_size or 1024,
        block_size=512,
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
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
    elif device.type == "cpu":
        torch.set_num_threads(max(1, min(torch.get_num_threads(), os.cpu_count() or 1)))

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
        if bool(saved.get("instruction_mode", False)) != bool(instruction_mode):
            raise ValueError("Checkpoint training mode differs from the requested instruction-mode setting.")
        tokenizer = BytePairTokenizer.from_dict(saved["tokenizer"])
    else:
        tokenizer = BytePairTokenizer.train_from_text(
            tokenizer_text,
            vocab_size=desired_vocab,
            max_training_bytes=tokenizer_bytes,
        )
        requested_config = replace(requested_config, vocab_size=tokenizer.vocab_size)

    if requested_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary and local tokenizer vocabulary do not match.")

    instruction_train_inputs = instruction_train_targets = None
    instruction_validation_inputs = instruction_validation_targets = None
    if instruction_mode:
        instruction_train_inputs, instruction_train_targets, training_tokens = _pack_instruction_records(
            train_records, tokenizer, requested_config.block_size
        )
        instruction_validation_inputs, instruction_validation_targets, validation_tokens = _pack_instruction_records(
            validation_records, tokenizer, requested_config.block_size
        )
        chunk_count = instruction_train_inputs.shape[0]
        validation_ids = None
        train_ids = None
    else:
        train_ids = torch.tensor([BOS_ID, *tokenizer.encode(raw_train_text), EOS_ID], dtype=torch.long)
        validation_ids = torch.tensor([BOS_ID, *tokenizer.encode(raw_validation_text), EOS_ID], dtype=torch.long)
        block_size = requested_config.block_size
        if train_ids.numel() <= block_size + 1 or validation_ids.numel() <= block_size + 1:
            raise ValueError(
                f"Training and validation splits must each contain more than {block_size + 1} tokens. "
                f"Received {train_ids.numel()} training and {validation_ids.numel()} validation tokens."
            )
        chunk_count = (train_ids.numel() - 1) // block_size
        training_tokens = int(train_ids.numel())
        validation_tokens = int(validation_ids.numel())
    if chunk_count < 1:
        raise ValueError("The training split does not contain a complete context block.")

    steps_per_epoch = math.ceil(chunk_count / batch_size)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation_steps)
    model = LocalTransformer(requested_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_epoch = 0
    start_step = 0
    best_validation_loss: float | None = None
    last_train_loss: float | None = None
    last_validation_loss: float | None = None
    last_benchmark: dict | None = None
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
        last_benchmark = saved.get("benchmark")
        raw_order = saved.get("epoch_order")
        if isinstance(raw_order, torch.Tensor) and raw_order.numel() == chunk_count:
            epoch_order = raw_order.to(dtype=torch.long, device="cpu")
            next_batch_index = int(saved.get("next_batch_index", 0))
            if not 0 <= next_batch_index <= steps_per_epoch:
                raise ValueError("Checkpoint resume position is invalid.")
            if next_batch_index % gradient_accumulation_steps and next_batch_index != steps_per_epoch:
                raise ValueError("Checkpoint resume position is not an optimizer-step boundary.")
        elif raw_order is not None:
            raise ValueError("Checkpoint data-order state does not match the current corpus.")

    parameter_count = model.parameter_count
    started = perf_counter()
    target_epoch = start_epoch + epochs
    total_optimizer_steps = epochs * optimizer_steps_per_epoch
    current_step_in_run = 0
    last_learning_rate = learning_rate

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
            "benchmark": last_benchmark,
            "instruction_mode": instruction_mode,
            "dataset_sha256": dataset_hash,
            "dataset_files": dataset_files,
            "dataset_bytes": dataset_bytes,
            "training_settings": {
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "min_learning_rate_ratio": min_learning_rate_ratio,
                "gradient_clip_norm": gradient_clip_norm,
                "seed": seed,
                "steps_per_epoch": steps_per_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "instruction_mode": instruction_mode,
                "benchmark_tasks": list(benchmark_tasks),
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
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else ("Apple MPS" if device.type == "mps" else "CPU"),
            "step": start_step + current_step_in_run,
            "epoch": completed_epochs,
            "target_epoch": target_epoch,
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": last_learning_rate,
            "train_loss": last_train_loss,
            "validation_loss": last_validation_loss,
            "best_validation_loss": best_validation_loss,
            "tokens_per_second": round(current_step_in_run * batch_size * gradient_accumulation_steps * requested_config.block_size / elapsed, 2),
            "elapsed_seconds": round(elapsed, 2),
            "parameter_count": parameter_count,
            "dataset_files": len(dataset_files),
            "dataset_bytes": dataset_bytes,
            "training_tokens": training_tokens,
            "validation_tokens": validation_tokens,
            "vocab_size": tokenizer.vocab_size,
            "instruction_mode": instruction_mode,
            "benchmark": last_benchmark,
            "message": message,
        }
        if callback:
            callback(result)
        return result

    report(
        "running",
        start_epoch,
        "Resuming local model and optimizer state." if saved else (
            "Training local instruction examples with prompt tokens masked from cross-entropy loss."
            if instruction_mode else "Training a randomly initialized Transformer on local UTF-8 data."
        ),
    )
    model.train()
    while start_epoch < target_epoch:
        if epoch_order is None:
            epoch_order = torch.randperm(chunk_count, device="cpu")
            next_batch_index = 0

        for batch_index in range(next_batch_index, steps_per_epoch):
            begin = batch_index * batch_size
            indices = epoch_order[begin:begin + batch_size]
            if indices.numel() == 0:
                continue
            if instruction_mode:
                x, y = _instruction_batch(
                    instruction_train_inputs, instruction_train_targets, indices, device
                )
            else:
                x, y = _batch(train_ids, indices, requested_config.block_size, device)

            accumulation_group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
            accumulation_group_size = min(
                gradient_accumulation_steps, steps_per_epoch - accumulation_group_start
            )
            if batch_index == accumulation_group_start:
                optimizer.zero_grad(set_to_none=True)

            if instruction_mode:
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=-100,
                )
            else:
                _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("Training cross-entropy loss became non-finite.")
            (loss / accumulation_group_size).backward()
            last_train_loss = float(loss.detach().item())
            next_batch_index = batch_index + 1
            optimizer_update = (
                next_batch_index % gradient_accumulation_steps == 0
                or next_batch_index == steps_per_epoch
            )
            if not optimizer_update:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            current_step_in_run += 1
            progress = current_step_in_run / max(total_optimizer_steps, 1)
            warmup_scale = min(1.0, current_step_in_run / warmup_steps) if warmup_steps else 1.0
            cosine_progress = min(progress, 1.0)
            cosine_scale = min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            )
            last_learning_rate = learning_rate * warmup_scale * cosine_scale
            for group in optimizer.param_groups:
                group["lr"] = last_learning_rate
            optimizer.step()

            should_evaluate = (
                current_step_in_run == 1
                or current_step_in_run % save_every_steps == 0
                or next_batch_index == steps_per_epoch
            )
            if should_evaluate:
                if instruction_mode:
                    last_validation_loss = _estimate_instruction_validation_loss(
                        model, instruction_validation_inputs, instruction_validation_targets,
                        device, batch_size, evaluation_batches,
                    )
                else:
                    last_validation_loss = _estimate_validation_loss(
                        model, validation_ids, requested_config.block_size, device, evaluation_batches
                    )
                if best_validation_loss is None or last_validation_loss < best_validation_loss:
                    best_validation_loss = last_validation_loss

            should_save = (
                current_step_in_run == 1
                or current_step_in_run % save_every_steps == 0
                or next_batch_index == steps_per_epoch
            )
            if should_save:
                save_checkpoint(target, payload_for_save(start_epoch, epoch_order, next_batch_index))
                report("running", start_epoch, "Checkpoint saved; cross-entropy and held-out validation metrics updated.")

        start_epoch += 1
        epoch_order = None
        next_batch_index = 0
        save_checkpoint(target, payload_for_save(start_epoch, None, 0))
        report(
            "running" if start_epoch < target_epoch else "completed",
            start_epoch,
            "Completed one full shuffled pass through the local training examples.",
        )
        if run_benchmarks:
            last_benchmark = _run_epoch_benchmark(
                target, start_epoch, benchmark_tasks, benchmark_timeout_seconds
            )
            save_checkpoint(target, payload_for_save(start_epoch, None, 0))
            report(
                "running" if start_epoch < target_epoch else "completed",
                start_epoch,
                "Epoch fixture evaluation " + str(last_benchmark.get("status", "unknown"))
                + "; measured task completion " + str(last_benchmark.get("task_completion_rate", "n/a")) + ".",
            )

    return report(
        "completed",
        start_epoch,
        "Training finished. Model, tokenizer, optimizer, validation metrics, and local fixture results are saved.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain.train",
        description="Train SYNTRA's standalone local Transformer from local source or prepared instruction JSONL.",
    )
    parser.add_argument("--dataset", action="append", default=None, help="Approved UTF-8 source directory or instruction JSONL; repeat to add sources.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Local checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--tokenizer-bytes", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--save-every-steps", type=int, default=50)
    parser.add_argument("--max-dataset-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--instruction-mode", action="store_true")
    parser.add_argument("--benchmark-each-epoch", action="store_true")
    parser.add_argument("--benchmark-task", action="append", default=None)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true", help="Resume against the exact same local corpus and training mode.")
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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            min_learning_rate_ratio=args.min_lr_ratio,
            gradient_clip_norm=args.gradient_clip_norm,
            instruction_mode=args.instruction_mode,
            run_benchmarks=args.benchmark_each_epoch,
            benchmark_tasks=args.benchmark_task or ("utility-clamp", "python-slugify"),
            benchmark_timeout_seconds=args.benchmark_timeout_seconds,
            callback=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"SYNTRA training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

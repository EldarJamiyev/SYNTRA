"""Train SYNTRA's standalone Transformer exclusively on selected local text files.

Example:
    python -m brain.train --dataset dataset.txt --epochs 3 --device auto
    python -m brain.train --dataset .\corpus\python.txt --dataset .\corpus\docs.txt --resume
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import stat
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
MAX_INSTRUCTION_RECORDS = 10_000
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
    "logs", ".syntra", ".ssh", ".aws", ".azure", ".kube", ".docker",
    ".gnupg", ".terraform",
})
SENSITIVE_SUFFIXES = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".p7b", ".p7c", ".jks", ".crt", ".cer", ".der",
    ".keystore", ".sqlite", ".db", ".dump", ".bak", ".log",
})
MAX_DATASET_FILES = 20_000
TRAINING_SCHEDULER = "global-cosine-warmup-v1"
SCHEDULER_WARMUP_STEPS = 100
SCHEDULER_DECAY_STEPS = 10_000


def _read_bounded_regular_file(
    path: Path,
    max_bytes: int,
    *,
    label: str,
) -> bytes:
    """Read one unchanged regular file without following a final-component symlink."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError(f"{label} byte budget is invalid.")
    descriptor: int | None = None
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        if before.st_size > max_bytes:
            raise ValueError(
                f"{label} exceeds the remaining {max_bytes:,}-byte local dataset budget: {path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_size > max_bytes
            ):
                raise ValueError(f"{label} changed while it was being opened: {path}")
            raw = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
            if (
                len(raw) > max_bytes
                or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or opened.st_size != after.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
            ):
                raise ValueError(f"{label} changed while it was being read or exceeded its byte limit: {path}")
            return raw
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"Cannot read local dataset {path}: {exc}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _is_sensitive_name(name: str) -> bool:
    lower = name.casefold()
    if lower == ".env" or lower.startswith(".env."):
        return True
    if lower in {
        "credentials", "credentials.json", "secrets", "secrets.json",
        "id_rsa", "id_ed25519", "known_hosts", "authorized_keys",
        "access_token", "access_token.txt", "refresh_token", "token.txt",
        ".npmrc", ".pypirc", ".netrc", ".git-credentials",
        "dockerconfigjson", ".dockerconfigjson", "service-account.json",
        "serviceaccount.json", "kubeconfig",
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
    sensitive_root_components = {
        "secrets", "credentials", "private", "private_keys", "certs", "certificates",
        ".ssh", ".aws", ".azure", ".kube", ".docker", ".gnupg", ".terraform",
    }
    path_components = {part.casefold() for part in resolved_source.parts}
    selected_sensitive_root = (
        resolved_source.is_dir()
        and resolved_source.name.casefold() in sensitive_root_components
    )
    if path_components.intersection(sensitive_root_components) and not selected_sensitive_root:
        raise ValueError("Refusing to train from a path inside a sensitive directory.")
    if resolved_source.is_file():
        if _is_sensitive_name(resolved_source.name):
            raise ValueError(f"Refusing to train on an obviously sensitive file: {resolved_source.name}")
        if path_components.intersection(SKIP_DIRECTORIES):
            raise ValueError("Refusing to train from a file inside a generated or excluded directory.")
        yield resolved_source, resolved_source.name, True
        return
    if not resolved_source.is_dir():
        raise ValueError(f"Dataset path must be a local file or directory: {source}")

    if (
        resolved_source.name.casefold() in SKIP_DIRECTORIES
        or path_components.intersection(SKIP_DIRECTORIES)
        or resolved_source.name.casefold() in sensitive_root_components
        or _is_sensitive_name(resolved_source.name)
    ):
        raise ValueError("Refusing to train on a sensitive or generated directory root.")

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
    allow_empty: bool = False,
    require_holdout: bool = True,
    max_files: int = MAX_DATASET_FILES,
    preloaded_files: dict[str, tuple[str, bytes, bool]] | None = None,
    already_scanned_files: int = 0,
) -> tuple[str, str, list[str], int, str]:
    """Read bounded local text/code files and produce a stable corpus fingerprint."""
    if type(max_files) is not int or not 0 <= max_files <= MAX_DATASET_FILES:
        raise ValueError(f"max_files must be between 0 and {MAX_DATASET_FILES:,}.")
    documents: list[tuple[str, str]] = []
    names: list[str] = []
    digest = sha256()
    total_bytes = 0
    newly_read_bytes = 0
    file_count = already_scanned_files
    seen: set[Path] = set()
    preloaded_files = preloaded_files or {}
    if type(already_scanned_files) is not int or not 0 <= already_scanned_files <= max_files:
        raise ValueError("already_scanned_files is outside the local dataset file limit.")

    for source_index, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser()
        try:
            source = path.resolve(strict=True)
            for resolved, relative_name, explicitly_selected in _iter_local_source_files(source):
                if source_index > 0:
                    relative_name = f"selected-root-{source_index + 1}/{relative_name}"
                if exclude_paths and str(resolved) in exclude_paths:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                preloaded = preloaded_files.get(str(resolved))
                if preloaded is None:
                    if file_count >= max_files:
                        raise ValueError(f"Local training data exceeds the {max_files:,}-file limit.")
                    if newly_read_bytes >= max_dataset_bytes:
                        raise ValueError(
                            f"Selected datasets exceed the {max_dataset_bytes:,}-byte local corpus limit."
                        )
                    raw = _read_bounded_regular_file(
                        resolved,
                        max_dataset_bytes - newly_read_bytes,
                        label="training dataset",
                    )
                    newly_read_bytes += len(raw)
                    source_explicitly_selected = explicitly_selected
                    file_count += 1
                else:
                    preloaded_name, raw, source_explicitly_selected = preloaded
                    if preloaded_name != relative_name:
                        raise ValueError("Preloaded dataset path metadata does not match the selected source path.")
                if not raw:
                    continue
                try:
                    source_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    if source_explicitly_selected:
                        raise ValueError(f"Dataset must contain valid UTF-8 text: {resolved}") from None
                    continue

                total_bytes += len(raw)
                relative_bytes = relative_name.encode("utf-8")
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
                documents.append((
                    relative_name,
                    f"\n# SYNTRA_SOURCE_FILE: {relative_name}\n{source_text}",
                ))
                names.append(str(resolved))
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError(f"Cannot read local dataset {path}: {exc}") from None

    if not documents or not any(document.strip() for _, document in documents):
        if allow_empty:
            return "", "", names, total_bytes, digest.hexdigest()
        raise ValueError(
            "No readable UTF-8 source files were found. Select a text/code file or a project folder."
        )

    if len(documents) >= 2:
        # Keep every source file wholly inside one split. Splitting a concatenated
        # repository at a character offset leaks definitions and shared file context
        # across training and validation. The stable path hash makes the holdout
        # reproducible when the same approved source set is selected again.
        ranked_documents = sorted(
            documents,
            key=lambda item: sha256(
                ("syntra-raw-file-holdout-v1:" + item[0]).encode("utf-8")
            ).hexdigest(),
        )
        validation_count = min(
            len(ranked_documents) - 1,
            max(1, math.ceil(len(ranked_documents) * 0.1)),
        )
        validation_documents = ranked_documents[:validation_count]
        training_documents = ranked_documents[validation_count:]
        train_text = "\n\n".join(document for _, document in training_documents)
        validation_text = "\n\n".join(document for _, document in validation_documents)
    else:
        # A single explicitly selected file cannot be held out by source identity.
        # Retain the historical line-boundary split for this legacy use case.
        combined = documents[0][1]
        split_at = int(len(combined) * 0.9)
        line_boundary = combined.rfind("\n", 0, split_at)
        if line_boundary > 0 and len(combined) - line_boundary > 1:
            split_at = line_boundary + 1
        train_text = combined[:split_at]
        validation_text = combined[split_at:]

    if not train_text.strip() or not validation_text.strip():
        if require_holdout:
            raise ValueError("The local corpus is too small to create separate training and validation splits.")
        return train_text, validation_text, names, total_bytes, digest.hexdigest()
    return train_text, validation_text, names, total_bytes, digest.hexdigest()



def _read_instruction_jsonl_datasets(
    paths: Iterable[str | os.PathLike[str]],
    max_dataset_bytes: int,
) -> tuple[list[dict[str, str]], list[str], int, str, int, int, dict[str, tuple[str, bytes, bool]]]:
    """Read structured prompt/response JSONL files; leave ordinary text for the legacy reader."""
    records: list[dict[str, str]] = []
    names: list[str] = []
    digest = sha256()
    total_bytes = 0
    scan_bytes = 0
    seen: set[Path] = set()
    fallback_files: dict[str, tuple[str, bytes, bool]] = {}

    for source_index, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser()
        try:
            source = path.resolve(strict=True)
            for resolved, relative_name, explicitly_selected in _iter_local_source_files(source):
                if resolved.suffix.casefold() != ".jsonl" or resolved in seen:
                    continue
                if source_index > 0:
                    relative_name = f"selected-root-{source_index + 1}/{relative_name}"
                seen.add(resolved)
                if len(seen) > MAX_DATASET_FILES:
                    raise ValueError(f"Local training data exceeds the {MAX_DATASET_FILES:,}-file limit.")
                if scan_bytes >= max_dataset_bytes:
                    raise ValueError(
                        f"Selected datasets exceed the {max_dataset_bytes:,}-byte local corpus limit."
                    )
                raw = _read_bounded_regular_file(
                    resolved,
                    max_dataset_bytes - scan_bytes,
                    label="instruction dataset",
                )
                scan_bytes += len(raw)
                if not raw:
                    continue
                file_records: list[dict[str, str]] = []
                schema_confirmed = False
                plain_text_fallback = False
                nonempty_lines = 0
                for line_number, raw_line in enumerate(BytesIO(raw), start=1):
                    record_bytes = raw_line.rstrip(b"\r\n")
                    if not record_bytes.strip():
                        continue
                    nonempty_lines += 1
                    try:
                        line = record_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        if schema_confirmed:
                            raise ValueError(
                                f"Malformed UTF-8 instruction record in {relative_name}:{line_number}."
                            ) from None
                        plain_text_fallback = True
                        break

                    # A generic JSONL file or plain-text file remains eligible
                    # for raw-token fallback until an instruction schema is seen.
                    if not line.lstrip().startswith("{"):
                        if schema_confirmed:
                            raise ValueError(
                                f"Mixed instruction and raw-text records in {relative_name}:{line_number}."
                            )
                        plain_text_fallback = True
                        break
                    if len(record_bytes) > MAX_INSTRUCTION_RECORD_BYTES:
                        raise ValueError(
                            f"Instruction JSONL record in {relative_name}:{line_number} exceeds "
                            f"{MAX_INSTRUCTION_RECORD_BYTES:,} bytes."
                        )
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if schema_confirmed:
                            raise ValueError(
                                f"Invalid instruction JSON in {relative_name}:{line_number}: {exc.msg}"
                            ) from None
                        plain_text_fallback = True
                        break
                    if not isinstance(item, dict):
                        if schema_confirmed:
                            raise ValueError(
                                f"Instruction JSONL record in {relative_name}:{line_number} must be an object."
                            )
                        plain_text_fallback = True
                        break

                    fields = {str(key).casefold(): value for key, value in item.items()}
                    prompt_present = "instruction" in fields or "prompt" in fields
                    response_present = any(key in fields for key in ("response", "completion", "output"))
                    if not schema_confirmed and not (prompt_present or response_present):
                        plain_text_fallback = True
                        break
                    if not prompt_present or not response_present:
                        raise ValueError(
                            f"Incomplete instruction schema in {relative_name}:{line_number}; "
                            "each record must contain Instruction/Prompt and Response/Completion."
                        )
                    instruction = fields.get("instruction", fields.get("prompt"))
                    response = fields.get("response", fields.get("completion", fields.get("output")))
                    if (
                        not isinstance(instruction, str)
                        or not isinstance(response, str)
                        or not instruction.strip()
                        or not response.strip()
                    ):
                        raise ValueError(
                            f"Instruction/response fields in {relative_name}:{line_number} "
                            "must be non-empty strings."
                        )
                    if (
                        len(instruction.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES
                        or len(response.encode("utf-8")) > MAX_INSTRUCTION_RECORD_BYTES
                    ):
                        raise ValueError(
                            f"Instruction/response field in {relative_name}:{line_number} exceeds "
                            f"{MAX_INSTRUCTION_RECORD_BYTES:,} bytes."
                        )
                    if len(records) + len(file_records) >= MAX_INSTRUCTION_RECORDS:
                        raise ValueError(
                            f"Structured instruction datasets exceed the {MAX_INSTRUCTION_RECORDS:,}-record safety limit."
                        )
                    split_group = fields.get("split_group", "")
                    if split_group is None:
                        split_group = ""
                    if (
                        not isinstance(split_group, str)
                        or len(split_group) > 2048
                    ):
                        raise ValueError(
                            f"split_group in {relative_name}:{line_number} must be a string of at most 2,048 characters."
                        )
                    schema_confirmed = True
                    file_records.append({
                        "prompt": instruction.strip(),
                        "completion": response.strip(),
                        "source_path": relative_name,
                        "source_line": str(line_number),
                        "split_group": split_group.strip(),
                    })

                # Generic JSONL and ordinary text keep their legacy raw-token
                # path. Once one supported instruction record is recognized,
                # malformed or mixed subsequent records fail instead of silently
                # training raw source text as if it were valid supervision.
                if plain_text_fallback or not nonempty_lines or not schema_confirmed or not file_records:
                    fallback_files[str(resolved)] = (relative_name, raw, explicitly_selected)
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
            raise ValueError(f"Cannot read local dataset {path}: {exc}") from None

    return records, names, total_bytes, digest.hexdigest(), scan_bytes, len(seen), fallback_files


def _split_instruction_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split prompt duplicates and curriculum-related source groups together."""
    if not records:
        return [], []

    parents = list(range(len(records)))

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
    prompt_ids: set[str] = set()
    for index, record in enumerate(records):
        normalized_prompt = " ".join(
            unicodedata.normalize("NFKC", record["prompt"]).casefold().split()
        )
        prompt_id = sha256(normalized_prompt.encode("utf-8")).hexdigest()
        prompt_ids.add(prompt_id)
        keys = ["prompt:" + prompt_id]

        split_group = record.get("split_group")
        if isinstance(split_group, str) and split_group.strip():
            normalized_group = " ".join(
                unicodedata.normalize("NFKC", split_group).casefold().split()
            )
            if normalized_group:
                keys.append("source:" + sha256(normalized_group.encode("utf-8")).hexdigest())

        record_keys.append(tuple(keys))
        for key in keys:
            previous = owners.get(key)
            if previous is None:
                owners[key] = index
            else:
                union(index, previous)

    component_keys: dict[int, set[str]] = {}
    component_members: dict[int, list[int]] = {}
    for index, keys in enumerate(record_keys):
        root = find(index)
        component_keys.setdefault(root, set()).update(keys)
        component_members.setdefault(root, []).append(index)

    groups: dict[str, list[dict[str, str]]] = {}
    for root, members in component_members.items():
        stable_material = "\\0".join(sorted(component_keys[root]))
        group_id = sha256(stable_material.encode("utf-8")).hexdigest()
        groups[group_id] = [records[index] for index in members]

    if len(groups) < 2:
        raise ValueError(
            "Structured instruction training needs at least two independent prompt/source groups "
            "so validation remains separate from related curriculum examples."
        )
    validation_count = max(1, int(len(groups) * 0.1))
    validation_count = min(validation_count, len(groups) - 1)
    ranked = sorted(
        groups,
        key=lambda key: sha256(
            ("syntra-brain-instruction-split-v2:" + key).encode("utf-8")
        ).hexdigest(),
    )
    validation_groups = set(ranked[:validation_count])
    training = [
        item for group_id, items in groups.items()
        if group_id not in validation_groups
        for item in items
    ]
    validation = [
        item for group_id, items in groups.items()
        if group_id in validation_groups
        for item in items
    ]
    if not training or not validation:
        raise ValueError("The instruction dataset did not produce independent train and validation sets.")
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
    max_context_tokens: int = MAX_PACKED_CONTEXT_TOKENS,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pack legacy raw text into fixed, shuffled causal context blocks."""
    if not text.strip():
        return [], []
    token_ids = [BOS_ID, *tokenizer.encode(text), EOS_ID]
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, max(0, len(token_ids) - 1), block_size):
        window = token_ids[start:start + block_size + 1]
        # Preserve the historical raw-stream chunk count and resume ordering:
        # only complete blocks are used for legacy checkpoints.
        if len(window) != block_size + 1:
            continue
        x, y = window[:-1], window[1:]
        if (len(inputs) + 1) * block_size > max_context_tokens:
            raise ValueError("Packed raw context exceeds the remaining SYNTRA training token budget.")
        _append_context_block(x, y, block_size, inputs, targets)
    return inputs, targets


def _instruction_context_blocks(
    records: list[dict[str, str]],
    tokenizer: BytePairTokenizer,
    block_size: int,
    max_context_tokens: int = MAX_PACKED_CONTEXT_TOKENS,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pack prompt/response examples; mask prompt targets and supervise response plus EOS only."""
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for record in records:
        prompt = "Instruction:\n" + record["prompt"].strip() + "\nResponse:\n"
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
                if (len(inputs) + 1) * block_size > max_context_tokens:
                    raise ValueError("Packed instruction contexts exceed the remaining SYNTRA training token budget.")
                _append_context_block(x, y, block_size, inputs, targets)
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
    target_blocks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim == 2:
        if target_blocks is None or target_blocks.shape != data.shape:
            raise ValueError("Prepacked context blocks require a matching target-block tensor.")
        indices = chunk_indices.to(dtype=torch.long, device="cpu")
        return (
            data.index_select(0, indices).to(device),
            target_blocks.index_select(0, indices).to(device),
        )
    if data.ndim != 1 or target_blocks is not None:
        raise ValueError("Legacy token streams must be one-dimensional and cannot include target blocks.")
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
    validation_targets: torch.Tensor | None = None,
) -> float:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    supervised_targets = 0
    try:
        if validation_data.ndim == 2:
            if validation_targets is None or validation_targets.shape != validation_data.shape:
                raise ValueError("Prepacked validation contexts require matching target blocks.")
            available_chunks = validation_data.shape[0]
        else:
            available_chunks = max(1, (validation_data.numel() - 1) // block_size)
        for _ in range(batches):
            count = min(8, available_chunks)
            starts = torch.randint(0, available_chunks, (count,), device="cpu")
            x, y = _batch(validation_data, starts, block_size, device, validation_targets)
            valid_count = int((y != -100).sum().item())
            if valid_count < 1:
                raise RuntimeError("Validation batch contains no supervised target tokens.")
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("Validation produced a non-finite loss.")
            weighted_loss += float(loss.item()) * valid_count
            supervised_targets += valid_count
    finally:
        model.train(was_training)
    if supervised_targets < 1:
        raise RuntimeError("Validation did not process any supervised target tokens.")
    return weighted_loss / supervised_targets


def _scheduled_learning_rate(base_learning_rate: float, global_step: int) -> float:
    """Return a resume-invariant learning rate based only on checkpointed global step."""
    if (
        isinstance(base_learning_rate, bool)
        or not isinstance(base_learning_rate, (int, float))
        or not math.isfinite(base_learning_rate)
        or base_learning_rate <= 0
    ):
        raise ValueError("base_learning_rate must be a finite positive number.")
    if type(global_step) is not int or global_step < 1:
        raise ValueError("global_step must be a positive integer.")
    warmup = min(1.0, global_step / SCHEDULER_WARMUP_STEPS)
    progress = min(global_step / SCHEDULER_DECAY_STEPS, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(float(base_learning_rate) * warmup * cosine, float(base_learning_rate) * 0.05)


def _validate_resume_order(
    raw_order: object,
    raw_next_batch_index: object,
    chunk_count: int,
    steps_per_epoch: int,
) -> tuple[torch.Tensor | None, int]:
    """Reject corrupt or incompatible saved shuffle state before resuming training."""
    if raw_order is None:
        if raw_next_batch_index is not None and (
            type(raw_next_batch_index) is not int or raw_next_batch_index != 0
        ):
            raise ValueError("Checkpoint resume cursor is non-zero without a saved epoch permutation.")
        return None, 0

    if not isinstance(raw_order, torch.Tensor):
        raise ValueError("Checkpoint epoch order must be a tensor.")
    if raw_order.dtype != torch.long or raw_order.ndim != 1 or raw_order.numel() != chunk_count:
        raise ValueError("Checkpoint epoch order shape or dtype does not match the current training blocks.")
    if chunk_count < 1 or int(raw_order.min().item()) < 0 or int(raw_order.max().item()) >= chunk_count:
        raise ValueError("Checkpoint epoch order contains an out-of-range training block index.")
    if not torch.equal(torch.sort(raw_order.cpu()).values, torch.arange(chunk_count, dtype=torch.long)):
        raise ValueError("Checkpoint epoch order is not a complete, duplicate-free permutation.")

    if type(raw_next_batch_index) is not int or not 0 <= raw_next_batch_index <= steps_per_epoch:
        raise ValueError("Checkpoint resume position is invalid.")
    return raw_order.cpu(), raw_next_batch_index


def _rng_state(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: object, device: torch.device) -> None:
    """Restore exact checkpointed generators or reject an unsafe continuation."""
    if not isinstance(state, dict):
        raise ValueError("Resume checkpoint is missing its random-generator state.")
    cpu_state = state.get("cpu")
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() < 1
    ):
        raise ValueError("Resume checkpoint CPU random-generator state is malformed.")
    try:
        torch.set_rng_state(cpu_state.cpu())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Resume checkpoint CPU random-generator state is incompatible: {exc}") from None

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_states = state.get("cuda")
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Resume checkpoint CUDA random-generator state does not match the available device count."
            )
        if any(
            not isinstance(item, torch.Tensor)
            or item.dtype != torch.uint8
            or item.ndim != 1
            or item.numel() < 1
            for item in cuda_states
        ):
            raise ValueError("Resume checkpoint CUDA random-generator state is malformed.")
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Resume checkpoint CUDA random-generator state is incompatible: {exc}") from None


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

    (
        instruction_records,
        instruction_files,
        instruction_bytes,
        instruction_hash,
        instruction_scan_bytes,
        instruction_scan_files,
        raw_jsonl_fallbacks,
    ) = _read_instruction_jsonl_datasets(paths, max_dataset_bytes)
    structured_paths = set(instruction_files)
    remaining_read_bytes = max_dataset_bytes - instruction_scan_bytes
    train_text, validation_text, raw_files, raw_bytes, raw_hash = _read_local_datasets(
        paths,
        remaining_read_bytes,
        exclude_paths=structured_paths,
        allow_empty=bool(instruction_records),
        require_holdout=not bool(instruction_records),
        max_files=MAX_DATASET_FILES,
        preloaded_files=raw_jsonl_fallbacks,
        already_scanned_files=instruction_scan_files,
    )

    if instruction_records:
        training_records, validation_records = _split_instruction_records(instruction_records)
        if train_text.strip():
            training_data_format = "mixed-raw-and-instruction-jsonl-v1"
        else:
            training_data_format = "instruction-jsonl-masked-v1"
        # The source-group split policy is part of the effective training corpus.
        # Bump this fingerprint domain so old prompt-only-split checkpoints cannot
        # silently resume against a different train/validation assignment.
        dataset_hash = sha256(
            ("syntra-brain-corpus-v3:source-group-split-v2:" + raw_hash + ":" + instruction_hash).encode("ascii")
        ).hexdigest()
    else:
        training_records, validation_records = [], []
        training_data_format = "raw-token-stream-v1"
        # Preserve historical dataset fingerprints for unchanged raw-text training runs.
        dataset_hash = raw_hash

    dataset_files = list(dict.fromkeys([*raw_files, *instruction_files]))
    dataset_bytes = raw_bytes + instruction_bytes
    if not train_text.strip() and not training_records:
        raise ValueError("The selected local sources produced no usable training content.")
    if not validation_text.strip() and not validation_records:
        raise ValueError("The selected local sources produced no independent validation content.")

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

    training_instruction_text = [
        "Instruction:\n" + item["prompt"] + "\nResponse:\n" + item["completion"]
        for item in training_records
    ]
    tokenizer_training_text = "\n\n".join(
        item for item in [train_text, *training_instruction_text] if item.strip()
    )

    saved: dict | None = None
    saved_training_settings: dict | None = None
    if resume:
        saved = load_checkpoint(target)
        if saved["dataset_sha256"] != dataset_hash:
            raise ValueError(
                "The selected dataset differs from the checkpoint's training corpus. "
                "Start a new checkpoint or use the original dataset to resume."
            )
        saved_data_format = saved.get("training_data_format", "raw-token-stream-v1")
        if saved_data_format != training_data_format:
            raise ValueError(
                "The selected dataset format differs from the checkpoint's training data format. "
                "Start a new checkpoint rather than resuming across raw-text and supervised JSONL modes."
            )
        settings = saved.get("training_settings")
        required_settings = {
            "batch_size",
            "learning_rate",
            "weight_decay",
            "seed",
            "steps_per_epoch",
            "evaluation_batches",
            "save_every_steps",
            "scheduler",
        }
        if isinstance(settings, dict) and required_settings.issubset(settings):
            expected_settings = {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "evaluation_batches": evaluation_batches,
                "save_every_steps": save_every_steps,
                "scheduler": TRAINING_SCHEDULER,
            }
            changed_settings = [
                name for name, expected in expected_settings.items()
                if type(settings.get(name)) is not type(expected)
                or settings.get(name) != expected
            ]
            if changed_settings:
                raise ValueError(
                    "Training resume settings differ from the checkpoint: "
                    + ", ".join(changed_settings)
                    + ". Use the original settings or start a new checkpoint."
                )
            saved_training_settings = settings
        else:
            # Older checkpoints can be resumed at a clean epoch boundary only.
            # Mid-epoch cursors cannot be interpreted safely without saved batch settings.
            if isinstance(saved.get("epoch_order"), torch.Tensor):
                raise ValueError(
                    "The checkpoint lacks the batch settings required for a safe mid-epoch resume. "
                    "Finish from a compatible checkpoint or start a new training run."
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
            tokenizer_training_text,
            vocab_size=desired_vocab,
            max_training_bytes=tokenizer_bytes,
        )
        requested_config = replace(requested_config, vocab_size=tokenizer.vocab_size)

    if requested_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary and local tokenizer vocabulary do not match.")

    block_size = requested_config.block_size
    remaining_context_tokens = MAX_PACKED_CONTEXT_TOKENS
    train_input_rows, train_target_rows = _raw_context_blocks(
        train_text, tokenizer, block_size, remaining_context_tokens
    )
    remaining_context_tokens -= len(train_input_rows) * block_size

    validation_input_rows, validation_target_rows = _raw_context_blocks(
        validation_text, tokenizer, block_size, remaining_context_tokens
    )
    remaining_context_tokens -= len(validation_input_rows) * block_size

    instruction_train_inputs, instruction_train_targets = _instruction_context_blocks(
        training_records, tokenizer, block_size, remaining_context_tokens
    )
    remaining_context_tokens -= len(instruction_train_inputs) * block_size

    instruction_validation_inputs, instruction_validation_targets = _instruction_context_blocks(
        validation_records, tokenizer, block_size, remaining_context_tokens
    )
    train_input_rows.extend(instruction_train_inputs)
    train_target_rows.extend(instruction_train_targets)
    validation_input_rows.extend(instruction_validation_inputs)
    validation_target_rows.extend(instruction_validation_targets)

    train_inputs, train_targets = _stack_context_blocks(
        train_input_rows, train_target_rows, "training"
    )
    validation_inputs, validation_targets = _stack_context_blocks(
        validation_input_rows, validation_target_rows, "validation"
    )
    chunk_count = train_inputs.shape[0]
    if chunk_count < 1:
        raise ValueError("The training corpus does not contain a supervised context block.")
    steps_per_epoch = math.ceil(chunk_count / batch_size)
    if saved_training_settings is not None:
        saved_steps_per_epoch = saved_training_settings.get("steps_per_epoch")
        if type(saved_steps_per_epoch) is not int or saved_steps_per_epoch != steps_per_epoch:
            raise ValueError(
                "The training dataset now produces a different number of batches per epoch. "
                "Start a new checkpoint rather than resuming with a changed data-order mapping."
            )
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
        epoch_order, next_batch_index = _validate_resume_order(
            raw_order,
            saved.get("next_batch_index", 0),
            chunk_count,
            steps_per_epoch,
        )

    parameter_count = model.parameter_count
    started = perf_counter()
    target_epoch = start_epoch + epochs
    current_step_in_run = 0
    supervised_tokens_processed = 0

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
            "training_data_format": training_data_format,
            "instruction_example_count": len(instruction_records),
            "training_context_blocks": chunk_count,
            "dataset_files": dataset_files,
            "dataset_bytes": dataset_bytes,
            "training_settings": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "steps_per_epoch": steps_per_epoch,
                "evaluation_batches": evaluation_batches,
                "save_every_steps": save_every_steps,
                "scheduler": TRAINING_SCHEDULER,
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
            "tokens_per_second": round(supervised_tokens_processed / elapsed, 2),
            "supervised_tokens_processed": supervised_tokens_processed,
            "throughput_basis": "supervised_target_tokens",
            "elapsed_seconds": round(elapsed, 2),
            "parameter_count": parameter_count,
            "dataset_files": len(dataset_files),
            "dataset_bytes": dataset_bytes,
            "training_tokens": int((train_targets != -100).sum().item()),
            "validation_tokens": int((validation_targets != -100).sum().item()),
            "training_context_blocks": int(train_inputs.shape[0]),
            "validation_context_blocks": int(validation_inputs.shape[0]),
            "instruction_examples": len(instruction_records),
            "training_data_format": training_data_format,
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
            x, y = _batch(train_inputs, indices, block_size, device, train_targets)
            batch_supervised_tokens = int((y != -100).sum().item())
            if batch_supervised_tokens < 1:
                raise RuntimeError("Training batch contains no supervised target tokens.")

            current_step_in_run += 1
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            global_step = start_step + current_step_in_run
            scheduled_lr = _scheduled_learning_rate(learning_rate, global_step)
            for group in optimizer.param_groups:
                group["lr"] = scheduled_lr
            optimizer.step()
            last_train_loss = float(loss.detach().item())
            supervised_tokens_processed += batch_supervised_tokens
            next_batch_index = batch_index + 1

            should_evaluate = (
                global_step == 1
                or global_step % save_every_steps == 0
                or next_batch_index == steps_per_epoch
            )
            if should_evaluate:
                last_validation_loss = _estimate_validation_loss(
                    model,
                    validation_inputs,
                    block_size,
                    device,
                    evaluation_batches,
                    validation_targets,
                )
                if best_validation_loss is None or last_validation_loss < best_validation_loss:
                    best_validation_loss = last_validation_loss

            should_save = (
                global_step == 1
                or global_step % save_every_steps == 0
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

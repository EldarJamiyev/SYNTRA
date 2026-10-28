"""Local-only source-corpus preparation and token-stream readers."""

from __future__ import annotations

from array import array
from pathlib import Path
from typing import Iterable
import sys

import torch

from syntra.tokenizer import ByteTokenizer

SUPPORTED_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go",
    ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb", ".swift",
    ".html", ".css", ".scss", ".sql", ".json", ".md", ".txt", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".sh", ".ps1", ".bat",
}
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
    "target", ".next", ".idea", ".pytest_cache",
}


def iter_source_files(raw_dir: str | Path) -> Iterable[Path]:
    root = Path(raw_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("Raw data directory does not exist: " + str(root))
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES or path.name == "Dockerfile":
            yield path


def prepare_corpus(
    raw_dir: str | Path = "data/raw",
    output_file: str | Path = "data/processed/tokens.bin",
    *,
    max_file_bytes: int = 2_000_000,
    max_total_bytes: int = 50_000_000,
) -> dict[str, int | str]:
    """Encode local source files into little-endian unsigned 16-bit token IDs."""
    tokenizer = ByteTokenizer()
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token_ids = array("H")
    source_bytes = 0
    file_count = 0
    skipped_count = 0

    for path in iter_source_files(raw_dir):
        size = path.stat().st_size
        if size <= 0 or size > max_file_bytes or source_bytes + size > max_total_bytes:
            skipped_count += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped_count += 1
            continue
        if not text.strip():
            skipped_count += 1
            continue
        token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        token_ids.append(14)  # newline byte separates adjacent files
        source_bytes += size
        file_count += 1

    if file_count == 0:
        raise ValueError(
            "No usable source files were found in " + str(raw_dir)
            + ". Add permitted text/code files under that directory first."
        )
    if token_ids.itemsize != 2:
        raise RuntimeError("This Python platform does not provide 16-bit array('H').")
    if sys.byteorder != "little":
        token_ids.byteswap()
    with destination.open("wb") as handle:
        token_ids.tofile(handle)

    return {
        "files": file_count,
        "skipped": skipped_count,
        "source_bytes": source_bytes,
        "tokens": len(token_ids),
        "output": str(destination),
    }


def read_token_stream(path: str | Path) -> torch.Tensor:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            "Token stream not found: " + str(source) + ". Run the prepare command first."
        )
    content = source.read_bytes()
    if len(content) % 2 != 0:
        raise ValueError("Token stream is malformed: byte length must be even.")
    values = array("H")
    values.frombytes(content)
    if sys.byteorder != "little":
        values.byteswap()
    if not values:
        raise ValueError("Token stream is empty.")
    if min(values) < 0 or max(values) >= 260:
        raise ValueError("Token stream contains token IDs outside the 260-token vocabulary.")
    return torch.tensor(values, dtype=torch.long)

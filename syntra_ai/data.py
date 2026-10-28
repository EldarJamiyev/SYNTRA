"""Safe local source corpus discovery and byte-level tokenization."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os

from .model import BOS_ID, EOS_ID

TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
    ".html", ".css", ".scss", ".sql", ".json", ".md", ".mdx", ".txt", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".sh", ".ps1", ".bat", ".go", ".rs", ".java",
    ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb", ".swift", ".vue",
    ".svelte", ".astro", ".graphql", ".gql", ".dockerfile"
}
IGNORED_DIRS = {
    ".git", ".syntra", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "coverage",
    ".next", ".nuxt", ".idea", ".vscode", "target", "vendor", "obj"
}
SENSITIVE_DIRS = {
    "secrets", "credentials", "private_keys", "certificates", ".ssh", ".aws", ".azure"
}
SENSITIVE_NAMES = {
    ".env", ".env.local", ".env.production", ".npmrc", ".pypirc", "credentials",
    "credentials.json", "secrets.json", "id_rsa", "id_ed25519", "known_hosts"
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".keystore", ".secret"}


@dataclass
class Corpus:
    token_ids: list[int]
    files: list[str]
    source_bytes: int
    skipped_files: int


def _safe_file(path: Path, max_file_bytes: int) -> bool:
    name = path.name.lower()
    if path.is_symlink() or name in SENSITIVE_NAMES:
        return False
    if name.startswith(".env.") and name != ".env.example":
        return False
    if name.startswith(("secret.", "secrets.", "credential.", "credentials.")) or ".secret." in name:
        return False
    if any(parent.name.lower() in SENSITIVE_DIRS for parent in path.parents):
        return False
    if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES and name not in {".gitignore", "dockerfile", "makefile"}:
        return False
    try:
        return path.is_file() and path.stat().st_size <= max_file_bytes
    except OSError:
        return False


def load_corpus(
    roots: Iterable[str | Path],
    max_files: int = 10000,
    max_file_bytes: int = 256 * 1024,
    max_tokens: int = 5_000_000,
) -> Corpus:
    tokens: list[int] = []
    files: list[str] = []
    total_bytes = 0
    skipped = 0
    seen: set[str] = set()
    if max_files < 1 or max_tokens < 1024:
        raise ValueError("max_files must be positive and max_tokens must be at least 1024.")
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            raise ValueError("Every corpus root must be an absolute path: " + str(raw_root))
        if not root.exists() or not root.is_dir():
            raise ValueError("Corpus root is not a directory: " + str(root))
        root = root.resolve()
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(current)
            dirnames[:] = [
                name for name in dirnames
                if name.lower() not in IGNORED_DIRS and name.lower() not in SENSITIVE_DIRS
                and not name.startswith(".") and not (base / name).is_symlink()
            ]
            for filename in sorted(filenames):
                if len(files) >= max_files or len(tokens) >= max_tokens:
                    break
                path = base / filename
                if not _safe_file(path, max_file_bytes):
                    skipped += 1
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    raw = path.read_bytes()
                    text = raw.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    skipped += 1
                    continue
                if not text.strip():
                    skipped += 1
                    continue
                encoded = [BOS_ID, *raw, EOS_ID]
                remaining = max_tokens - len(tokens)
                tokens.extend(encoded[:remaining])
                files.append(resolved)
                total_bytes += len(raw)
                if len(tokens) >= max_tokens:
                    break
            if len(files) >= max_files or len(tokens) >= max_tokens:
                break
    if not tokens:
        raise ValueError("No readable text/source files were found in the selected corpus root(s).")
    return Corpus(tokens, files, total_bytes, skipped)

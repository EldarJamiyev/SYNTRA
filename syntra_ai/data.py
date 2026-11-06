"""Safe local source corpus discovery and byte-level tokenization."""
from dataclasses import dataclass
from typing import Iterable
import os
import stat

from .tokenizer import BOS_ID, EOS_ID
from .safe_paths import is_path_within, resolve_corpus_root

MAX_CORPUS_FILES = 10_000
MAX_CORPUS_FILE_BYTES = 1024 * 1024
MAX_CORPUS_TOKENS = 10_000_000

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
    ".next", ".nuxt", ".idea", ".vscode", "target", "vendor", "obj",
    "checkpoints", "runs", "wandb", "mlruns", "lightning_logs", "tensorboard", "logs"
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
    documents: list[str]
    token_count: int = 0


def _read_safe_file(path: str, root: str, max_file_bytes: int) -> tuple[str, bytes, str] | None:
    """Read one bounded regular file after canonical containment and link checks."""
    try:
        absolute = os.path.abspath(path)
        resolved = os.path.realpath(path, strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    # The candidate must resolve beneath the already-approved root. Comparing
    # abspath to realpath also rejects file or parent symlinks.
    if not is_path_within(resolved, root):
        return None
    if os.path.normcase(absolute) != os.path.normcase(resolved):
        return None

    relative_parts = os.path.relpath(resolved, root).split(os.sep)
    name = os.path.basename(resolved).lower()
    if any(part.lower() in SENSITIVE_DIRS for part in relative_parts[:-1]):
        return None
    if name in SENSITIVE_NAMES:
        return None
    if name.startswith(".env.") and name != ".env.example":
        return None
    if name.startswith(("secret.", "secrets.", "credential.", "credentials.")) or ".secret." in name:
        return None
    if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        return None
    if os.path.splitext(name)[1].lower() not in TEXT_SUFFIXES and name not in {".gitignore", "dockerfile", "makefile"}:
        return None

    descriptor = None
    try:
        before = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_file_bytes:
            return None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_size > max_file_bytes):
                return None
            raw = handle.read(max_file_bytes + 1)
            after = os.fstat(handle.fileno())
            if (
                len(raw) > max_file_bytes
                or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or opened.st_size != after.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
            ):
                return None
        if len(raw) > max_file_bytes:
            return None
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if not text.strip():
        return None
    return resolved, raw, text


def load_corpus(
    roots: Iterable[str | os.PathLike[str]],
    max_files: int = 10000,
    max_file_bytes: int = 256 * 1024,
    max_tokens: int = 5_000_000,
    include_tokens: bool = True,
) -> Corpus:
    if type(include_tokens) is not bool:
        raise ValueError("include_tokens must be a boolean.")
    tokens: list[int] = []
    token_count = 0
    files: list[str] = []
    total_bytes = 0
    skipped = 0
    documents: list[str] = []
    seen: set[str] = set()
    if max_files < 1 or max_files > MAX_CORPUS_FILES:
        raise ValueError(f"max_files must be between 1 and {MAX_CORPUS_FILES:,}.")
    if max_file_bytes < 1 or max_file_bytes > MAX_CORPUS_FILE_BYTES:
        raise ValueError(f"max_file_bytes must be between 1 and {MAX_CORPUS_FILE_BYTES:,}.")
    if max_tokens < 1024 or max_tokens > MAX_CORPUS_TOKENS:
        raise ValueError(f"max_tokens must be between 1,024 and {MAX_CORPUS_TOKENS:,}.")

    for raw_root in roots:
        # Canonicalize and approve the root before traversal. os.walk and all
        # descendant paths are derived from this bounded, trusted directory.
        root = os.fspath(resolve_corpus_root(raw_root))
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            try:
                base = os.path.realpath(current, strict=True)
            except (OSError, RuntimeError):
                dirnames[:] = []
                continue
            if not is_path_within(base, root):
                dirnames[:] = []
                continue

            safe_directories = []
            for name in dirnames:
                if (name.lower() in IGNORED_DIRS or name.lower() in SENSITIVE_DIRS
                    or name.startswith(".")):
                    continue
                child = os.path.join(base, name)
                try:
                    child_absolute = os.path.abspath(child)
                    child_real = os.path.realpath(child, strict=True)
                    if (os.path.normcase(child_absolute) != os.path.normcase(child_real)
                        or not is_path_within(child_real, root)
                        or not os.path.isdir(child_real)):
                        continue
                except (OSError, RuntimeError, ValueError):
                    continue
                safe_directories.append(name)
            dirnames[:] = safe_directories

            for filename in sorted(filenames):
                if len(files) >= max_files or token_count >= max_tokens:
                    break
                candidate = os.path.join(base, filename)
                result = _read_safe_file(candidate, root, max_file_bytes)
                if result is None:
                    skipped += 1
                    continue
                resolved, raw, text = result
                normalized_key = os.path.normcase(resolved)
                if normalized_key in seen:
                    continue
                seen.add(normalized_key)
                remaining = max_tokens - token_count
                added_tokens = min(len(raw) + 2, remaining)
                if include_tokens and added_tokens:
                    # Materialize only the retained prefix instead of allocating a
                    # second full Python-int list for a large source file.
                    append_bos = min(1, added_tokens)
                    if append_bos:
                        tokens.append(BOS_ID)
                    retained_bytes = min(len(raw), added_tokens - append_bos)
                    if retained_bytes:
                        tokens.extend(raw[:retained_bytes])
                    if added_tokens > append_bos + retained_bytes:
                        tokens.append(EOS_ID)
                token_count += added_tokens
                files.append(resolved)
                documents.append(text)
                total_bytes += len(raw)
                if token_count >= max_tokens:
                    break
            if len(files) >= max_files or token_count >= max_tokens:
                break
    if not files:
        raise ValueError("No readable text/source files were found in the selected corpus root(s).")
    return Corpus(tokens, files, total_bytes, skipped, documents, token_count)

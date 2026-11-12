"""Safe local source corpus discovery and byte-level tokenization."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import os
import stat

from .tokenizer import BOS_ID, EOS_ID
from .safe_paths import _approved_roots, resolve_corpus_root

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
    "checkpoints", "runs", "wandb", "mlruns", "lightning_logs", "tensorboard", "logs",
    ".kube", ".docker", ".gnupg", ".terraform"
}
SENSITIVE_DIRS = {
    "secrets", "credentials", "private_keys", "certificates",
    ".ssh", ".aws", ".azure", ".kube", ".docker", ".gnupg", ".terraform"
}
SENSITIVE_NAMES = {
    ".env", ".env.local", ".env.production", ".npmrc", ".pypirc", ".netrc",
    ".git-credentials", ".dockerconfigjson", "credentials", "credentials.json",
    "secrets.json", "id_rsa", "id_ed25519", "known_hosts", "authorized_keys",
    "access_token", "access_token.txt", "refresh_token", "token.txt",
    "service-account.json", "serviceaccount.json", "kubeconfig"
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
    source_sha256: str = ""


def _read_safe_file(path: str, root: str, max_file_bytes: int) -> tuple[str, bytes, str] | None:
    """Read one bounded regular file after canonical containment and link checks."""
    try:
        absolute = os.path.abspath(path)
        resolved = os.path.realpath(path, strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    # Canonicalize both paths and perform the containment comparison in
    # this function, immediately before file-system access. This rejects
    # traversal and symlinks without depending on a custom sanitizer call.
    root_real = os.path.realpath(root, strict=True)
    candidate_norm = os.path.normcase(os.path.normpath(resolved))
    root_norm = os.path.normcase(os.path.normpath(root_real))
    root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
    if not (candidate_norm == root_norm or candidate_norm.startswith(root_prefix)):
        return None
    if os.path.normcase(absolute) != os.path.normcase(resolved):
        return None

    # The path has passed the canonical root-prefix guard above.
    relative_parts = os.path.relpath(resolved, root_real).split(os.sep)
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
            # The descriptor checks above prove that the opened inode did not
            # mutate while reading. They do not prove that the selected path
            # still names that inode: a concurrent rename/replace can leave the
            # descriptor stable while changing the file associated with this
            # corpus path and its provenance digest.
            current = os.stat(resolved, follow_symlinks=False)
            current_path = os.path.realpath(resolved, strict=True)
            if (
                len(raw) > max_file_bytes
                or os.path.normcase(os.path.normpath(current_path))
                != os.path.normcase(os.path.normpath(resolved))
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                or opened.st_size != after.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
                or opened.st_size != current.st_size
                or opened.st_mtime_ns != current.st_mtime_ns
                or opened.st_ctime_ns != current.st_ctime_ns
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
    exclude_paths: Iterable[str | os.PathLike[str]] = (),
) -> Corpus:
    if type(include_tokens) is not bool:
        raise ValueError("include_tokens must be a boolean.")
    try:
        excluded_paths = []
        for value in exclude_paths:
            raw_exclusion = os.fspath(value)
            if not os.path.isabs(raw_exclusion):
                raise ValueError("Corpus exclusion paths must be absolute.")
            excluded_paths.append(os.path.normcase(os.path.realpath(raw_exclusion, strict=False)))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Corpus exclusion paths are invalid: " + str(exc)) from None
    tokens: list[int] = []
    token_count = 0
    files: list[str] = []
    total_bytes = 0
    skipped = 0
    documents: list[str] = []
    digest = sha256()
    seen: set[str] = set()
    if max_files < 1 or max_files > MAX_CORPUS_FILES:
        raise ValueError(f"max_files must be between 1 and {MAX_CORPUS_FILES:,}.")
    if max_file_bytes < 1 or max_file_bytes > MAX_CORPUS_FILE_BYTES:
        raise ValueError(f"max_file_bytes must be between 1 and {MAX_CORPUS_FILE_BYTES:,}.")
    if max_tokens < 1024 or max_tokens > MAX_CORPUS_TOKENS:
        raise ValueError(f"max_tokens must be between 1,024 and {MAX_CORPUS_TOKENS:,}.")

    for raw_root in roots:
        # resolve_corpus_root applies the application policy; repeat the
        # canonical approved-root containment check here at the traversal sink.
        root_path = resolve_corpus_root(raw_root)
        root = os.path.realpath(os.fspath(root_path), strict=True)
        root_norm = os.path.normcase(os.path.normpath(root))
        approved_root = False
        for trusted_root in _approved_roots():
            trusted_norm = os.path.normcase(os.path.normpath(
                os.path.realpath(trusted_root, strict=True)
            ))
            trusted_prefix = trusted_norm if trusted_norm.endswith(os.sep) else trusted_norm + os.sep
            if root_norm == trusted_norm or root_norm.startswith(trusted_prefix):
                approved_root = True
                break
        if not approved_root:
            raise ValueError("Refusing to scan a corpus root outside approved directories.")
        root_path = Path(root)
        root_parts = {part.casefold() for part in root_path.parts}
        root_name = root_path.name.casefold()
        if (
            root_name in IGNORED_DIRS
            or root_name in SENSITIVE_DIRS
            or root_parts.intersection(IGNORED_DIRS)
            or root_parts.intersection(SENSITIVE_DIRS)
        ):
            raise ValueError("Refusing to scan a sensitive or generated corpus root.")
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            try:
                base = os.path.realpath(current, strict=True)
            except (OSError, RuntimeError):
                dirnames[:] = []
                continue
            base_norm = os.path.normcase(os.path.normpath(base))
            root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
            if not (base_norm == root_norm or base_norm.startswith(root_prefix)):
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
                    child_norm = os.path.normcase(os.path.normpath(child_real))
                    root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
                    if (
                        os.path.normcase(child_absolute) != os.path.normcase(child_real)
                        or not (child_norm == root_norm or child_norm.startswith(root_prefix))
                        or not os.path.isdir(child_real)
                    ):
                        continue
                except (OSError, RuntimeError, ValueError):
                    continue
                safe_directories.append(name)
            dirnames[:] = safe_directories

            for filename in sorted(filenames):
                if len(files) >= max_files or token_count >= max_tokens:
                    break
                candidate = os.path.join(base, filename)
                try:
                    candidate_real = os.path.realpath(candidate, strict=True)
                except (OSError, RuntimeError):
                    skipped += 1
                    continue
                normalized_candidate = os.path.normcase(os.path.normpath(candidate_real))
                excluded_candidate = False
                for excluded in excluded_paths:
                    excluded_norm = os.path.normcase(os.path.normpath(excluded))
                    excluded_prefix = excluded_norm if excluded_norm.endswith(os.sep) else excluded_norm + os.sep
                    if normalized_candidate == excluded_norm or normalized_candidate.startswith(excluded_prefix):
                        excluded_candidate = True
                        break
                if excluded_candidate:
                    skipped += 1
                    continue
                result = _read_safe_file(candidate, root, max_file_bytes)
                if result is None:
                    skipped += 1
                    continue
                resolved, raw, text = result
                normalized_key = os.path.normcase(resolved)
                if normalized_key in seen:
                    continue
                seen.add(normalized_key)
                path_bytes = normalized_key.encode("utf-8")
                digest.update(len(path_bytes).to_bytes(8, "big"))
                digest.update(path_bytes)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
                remaining = max_tokens - token_count
                if not include_tokens and len(raw) + 2 > remaining:
                    skipped += 1
                    continue
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
    return Corpus(tokens, files, total_bytes, skipped, documents, token_count, digest.hexdigest())

"""Source-grounded multi-file agent reasoning curriculum synthesis.

Pairs are formed only from local import/reference relationships and exact,
unique repair anchors in both files. The generated rationale is a compact
task-localization trace, not a hidden model transcript.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .data import load_corpus
from .polyglot import (
    LANGUAGE_SUFFIXES, SUPPORTED_LANGUAGES, _code_mask, language_for_path,
    safe_repair_examples,
)
from .safe_paths import PROJECT_ROOT, resolve_corpus_root, resolve_curriculum_output_path
from .curriculum import _safe_none_patch_examples, _write_jsonl_atomically, _module_metadata

MAX_REASONING_EXAMPLES = 5_000
MAX_PAIRED_SOURCE_CHARS = 12_000
MAX_REASONING_SCAN_FILES = 10_000
_PREFERRED_SUFFIXES = {
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx", ".mts", ".cts"),
}
_JS_IMPORT = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*|\brequire\s*\()\s*["'](?P<specifier>\.{1,2}/[^"']+)["']"""
)
_GO_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_]\w*)")
_GO_TEST = re.compile(r"_test\.go$")
_RUST_MOD = re.compile(r"(?m)^\s*(?:pub\s+)?mod\s+(?P<module>[A-Za-z_]\w*)\s*;")
_RUST_USE = re.compile(r"(?m)^\s*use\s+(?:crate|self|super)::(?P<path>[A-Za-z_][\w:]*)")


def _hidden_benchmark_exclusions(roots: list[Path], supplied: Iterable[str | Path]) -> list[str | Path]:
    excluded: list[str | Path] = list(supplied)
    for root in roots:
        hidden = root / "tests" / "benchmark_engine" / "hidden"
        if hidden.exists():
            excluded.append(hidden)
    return excluded


def _safe_repairs(path: str, source: str) -> list[dict[str, str]]:
    if language_for_path(path) == "python":
        return _safe_none_patch_examples(path, source)
    return safe_repair_examples(path, source)


def _relative_module_candidates(
    relative_path: str,
    specifier: str,
    language: str,
) -> list[str]:
    parent = posixpath.dirname(relative_path)
    base = posixpath.normpath(posixpath.join(parent, specifier))
    suffixes = _PREFERRED_SUFFIXES[language]
    candidates = [base] if PurePosixPath(base).suffix else [base + suffix for suffix in suffixes]
    candidates.extend(posixpath.join(base, "index" + suffix) for suffix in suffixes)
    return candidates


def _python_module_candidates(relative_path: str, module: str | None, level: int) -> list[str]:
    """Resolve local Python import names to sibling modules without loading them."""
    current = PurePosixPath(relative_path)
    parts = list(current.parent.parts)
    if level:
        for _ in range(max(0, level - 1)):
            if not parts:
                return []
            parts.pop()
        module_parts = module.split(".") if module else []
        base_parts = parts + module_parts
    else:
        if not module:
            return []
        base_parts = module.split(".")
    if not base_parts:
        return []
    base = posixpath.join(*base_parts)
    return [base + ".py", base + ".pyi", posixpath.join(base, "__init__.py")]


def _related_targets(path: str, source: str, language: str, same_language_paths: set[str]) -> list[str]:
    targets: list[str] = []
    code_mask = _code_mask(source)
    if language == "python":
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, TypeError):
            return []
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    for alias in node.names:
                        candidates.extend(_python_module_candidates(path, alias.name, node.level))
                else:
                    candidates.extend(_python_module_candidates(path, node.module, node.level))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    candidates.extend(_python_module_candidates(path, alias.name, 0))
            for candidate in candidates:
                if candidate in same_language_paths and candidate != path:
                    targets.append(candidate)
                    break
    elif language in {"javascript", "typescript"}:
        for match in _JS_IMPORT.finditer(source):
            if not code_mask[match.start()]:
                continue
            for candidate in _relative_module_candidates(
                path, match.group("specifier"), language
            ):
                if candidate in same_language_paths and candidate != path:
                    targets.append(candidate)
                    break
    elif language == "go":
        package = _GO_PACKAGE.search(source)
        if package and _GO_TEST.search(path):
            prefix = posixpath.dirname(path) + "/"
            for candidate in sorted(same_language_paths):
                if candidate != path and candidate.startswith(prefix) and not _GO_TEST.search(candidate):
                    targets.append(candidate)
    elif language == "rust":
        modules = [
            match.group("module") for match in _RUST_MOD.finditer(source)
            if code_mask[match.start()]
        ]
        modules.extend(
            match.group("path").split("::")[0]
            for match in _RUST_USE.finditer(source) if code_mask[match.start()]
        )
        for module in modules:
            candidates = [
                posixpath.join(posixpath.dirname(path), module + ".rs"),
                posixpath.join(posixpath.dirname(path), module, "mod.rs"),
            ]
            for candidate in candidates:
                if candidate in same_language_paths and candidate != path:
                    targets.append(candidate)
                    break
    return sorted(set(targets))


def _referenced_export(
    exports: list[str],
    importing_source: str,
    language: str,
) -> str | None:
    if language == "python":
        try:
            tree = ast.parse(importing_source)
        except (SyntaxError, ValueError, TypeError):
            return None
        public = set(exports)
        imported_names: dict[str, str] = {}
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in public:
                        imported_names[alias.asname or alias.name] = alias.name
                    elif node.module is None:
                        imported_modules.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.asname or alias.name.split(".")[0])
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in imported_names:
                return imported_names[node.id]
            if (
                isinstance(node, ast.Attribute)
                and node.attr in public
                and isinstance(node.value, ast.Name)
                and node.value.id in imported_modules
            ):
                return node.attr
        return None

    code_mask = _code_mask(importing_source)
    for name in sorted(set(exports), key=lambda item: (-len(item), item)):
        if not name or name.startswith("_"):
            continue
        for match in re.finditer(rf"\b{re.escape(name)}\b", importing_source):
            if code_mask[match.start()]:
                return name
    return None


def synthesize_multifile_reasoning_curriculum(
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    max_examples: int = 120,
    languages: Iterable[str] = SUPPORTED_LANGUAGES,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Generate paired-source localization traces with two exact verified edits."""
    if type(max_examples) is not int or not 1 <= max_examples <= MAX_REASONING_EXAMPLES:
        raise ValueError(f"max_examples must be between 1 and {MAX_REASONING_EXAMPLES:,}.")
    selected_languages = tuple(sorted(set(languages)))
    if not selected_languages or any(item not in SUPPORTED_LANGUAGES for item in selected_languages):
        raise ValueError("languages must select one or more supported SYNTRA curriculum languages.")

    if root is None:
        configured = [
            item.strip() for item in os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS", "").split(os.pathsep)
            if item.strip()
        ]
        candidates: list[str | Path] = [PROJECT_ROOT, *configured]
    else:
        candidates = [root]
    roots: list[Path] = []
    seen_roots: set[str] = set()
    for candidate in candidates:
        approved = resolve_corpus_root(candidate)
        key = os.path.normcase(os.fspath(approved))
        if key not in seen_roots:
            roots.append(approved)
            seen_roots.add(key)
    if not roots:
        raise ValueError("No approved local repository roots were configured.")

    excluded = _hidden_benchmark_exclusions(roots, exclude_paths)
    corpus = load_corpus(
        roots, include_tokens=False, exclude_paths=excluded,
        max_files=MAX_REASONING_SCAN_FILES,
    )
    if len(corpus.files) != len(corpus.documents):
        raise RuntimeError("The bounded source reader returned inconsistent file/document counts.")

    by_root: dict[int, dict[str, dict[str, Any]]] = {index: {} for index in range(len(roots))}
    for source_path, source_text in sorted(
        zip(corpus.files, corpus.documents),
        key=lambda item: os.path.normcase(item[0]),
    ):
        source_real = os.path.realpath(os.fspath(source_path), strict=True)
        source_norm = os.path.normcase(os.path.normpath(source_real))
        matched_root: int | None = None
        relative: str | None = None
        for index, selected_root in enumerate(roots):
            root_real = os.path.realpath(os.fspath(selected_root), strict=True)
            root_norm = os.path.normcase(os.path.normpath(root_real))
            root_prefix = root_norm if root_norm.endswith(os.sep) else root_norm + os.sep
            if source_norm == root_norm or source_norm.startswith(root_prefix):
                relative = Path(os.path.relpath(source_real, root_real)).as_posix()
                matched_root = index
                break
        if matched_root is None or relative is None:
            raise ValueError("A reasoning source file escaped all approved curriculum roots.")
        language = language_for_path(relative)
        if language not in selected_languages or len(source_text) > MAX_PAIRED_SOURCE_CHARS:
            continue
        repairs = _safe_repairs(relative, source_text)
        if not repairs:
            continue
        metadata = _module_metadata(relative, source_text)
        by_root[matched_root][relative] = {
            "source": source_text,
            "language": language,
            "repairs": repairs,
            "exports": metadata.get("exports", []),
        }

    records: list[dict[str, str]] = []
    considered_pairs = 0
    for root_index, files in by_root.items():
        # Patch paths are applied relative to the active workspace (root zero).
        # Other approved roots are read-only training sources and cannot safely
        # be represented as edits against a different active workspace.
        if root_index != 0:
            continue
        paths = set(files)
        for consumer_path in sorted(paths):
            consumer = files[consumer_path]
            language = consumer["language"]
            for implementation_path in _related_targets(
                consumer_path, consumer["source"], language, paths
            ):
                implementation = files[implementation_path]
                if implementation["language"] != language:
                    continue
                if language == "go":
                    package_consumer = _GO_PACKAGE.search(consumer["source"])
                    package_implementation = _GO_PACKAGE.search(implementation["source"])
                    if (
                        not _GO_TEST.search(consumer_path)
                        or not package_consumer or not package_implementation
                        or package_consumer.group(1) != package_implementation.group(1)
                    ):
                        continue
                export = _referenced_export(
                    implementation.get("exports", []), consumer["source"], language
                )
                if export is None:
                    continue
                considered_pairs += 1
                left_repair = sorted(
                    implementation["repairs"],
                    key=lambda item: (
                        item["line"], item.get("rule", "python-none-sentinel"), item["before"]
                    ),
                )[0]
                right_repair = sorted(
                    consumer["repairs"],
                    key=lambda item: (
                        item["line"], item.get("rule", "python-none-sentinel"), item["before"]
                    ),
                )[0]
                pair_paths = sorted((implementation_path, consumer_path))
                pair_id = hashlib.sha256(
                    (str(root_index) + "\0" + "\0".join(pair_paths)).encode("utf-8")
                ).hexdigest()
                fence = chr(96) * 3
                source_blocks = (
                    f"FILE: {implementation_path}\n"
                    f"{fence}{language}\n{implementation['source'].rstrip()}\n{fence}\n\n"
                    f"FILE: {consumer_path}\n"
                    f"{fence}{language}\n{consumer['source'].rstrip()}\n{fence}"
                )
                localization = (
                    "<thought>\n"
                    f"1. Resolve the consumer's local dependency to {implementation_path}.\n"
                    f"2. Locate the referenced declaration {export or 'in the related local module'} and confirm its implementation context.\n"
                    f"3. Verify the unique repair anchor in {implementation_path} and the separate unique anchor in {consumer_path}; preserve all unrelated source.\n"
                    "4. Emit one exact edit for each file using repository-relative paths.\n"
                    "</thought>"
                )
                payload = {
                    "format": "SYNTRA_AGENT_PATCH_V1",
                    "summary": "Apply two source-verified repairs across the related implementation and consumer files.",
                    "edits": [
                        {
                            "path": implementation_path, "find": left_repair["before"],
                            "replace": left_repair["after"],
                            "reason": left_repair.get(
                                "reason",
                                "Use identity comparison when checking the Python None singleton.",
                            ),
                        },
                        {
                            "path": consumer_path, "find": right_repair["before"],
                            "replace": right_repair["after"],
                            "reason": right_repair.get(
                                "reason",
                                "Use identity comparison when checking the Python None singleton.",
                            ),
                        },
                    ],
                }
                prompt = (
                    "Task: inspect these two related local files and produce a multi-file "
                    "SYNTRA_AGENT_PATCH_V1 patch. First localize the implementation and its "
                    "consumer, then verify each exact unique source anchor. Do not invent "
                    "changes, modify unrelated code, or omit either file.\n\n" + source_blocks
                )
                completion = localization + "\n" + json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                )
                records.append({
                    "prompt": prompt, "completion": completion,
                    "kind": "multi_file_reasoning", "source_path": implementation_path,
                    "split_group": f"multi-file:{pair_id}",
                })

    unique: dict[str, dict[str, str]] = {}
    for record in records:
        key = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
        unique.setdefault(key, record)
    ordered = [unique[key] for key in sorted(unique)][:max_examples]
    if not ordered:
        return {
            "dataset_path": None, "workspace_roots": [str(item) for item in roots],
            "source_files": len(corpus.files), "example_count": 0,
            "candidate_relationships": considered_pairs,
            "examples_by_type": {"multi_file_reasoning": 0},
            "warnings": [
                "No related local file pair contained a mechanically verified repair anchor in both files; no reasoning examples were fabricated and no output file was written."
            ],
        }

    target = resolve_curriculum_output_path(output_path, roots[0])
    _write_jsonl_atomically(target, ordered)
    from .instruction import audit_instruction_dataset
    audit = audit_instruction_dataset(target)
    return {
        "dataset_path": str(target), "workspace_roots": [str(item) for item in roots],
        "source_files": len(corpus.files), "paired_candidates": considered_pairs,
        "example_count": len(ordered),
        "examples_by_type": {"multi_file_reasoning": len(ordered)},
        "completion_supervision": "thought-block, patch-json, and EOS are completion tokens; prompt tokens remain masked by the native SFT loader.",
        "audit": audit, "warnings": list(audit.get("warnings", [])),
    }


__all__ = ["synthesize_multifile_reasoning_curriculum"]

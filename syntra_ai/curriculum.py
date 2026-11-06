"""Evidence-grounded repository curriculum synthesis for SYNTRA.

Uses the standard library and SYNTRA's bounded local source reader. It never
calls a hosted model or invents source code. Repair examples are emitted only
for a narrow, mechanically checked Python rule.
"""
from __future__ import annotations

import ast
import hashlib
import heapq
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .data import load_corpus
from .safe_paths import PROJECT_ROOT, resolve_corpus_root, resolve_curriculum_output_path

MAX_INSTRUCTION_EXAMPLES = 10_000
MAX_SYMBOL_CHARS = 12_000
MAX_MODULE_CHARS = 8_000
MAX_IMPORTS = 80
MAX_DECLARATIONS = 120
MAX_CURRICULUM_CANDIDATE_BYTES = 128 * 1024 * 1024
MAX_CURRICULUM_RECORD_BYTES = 1024 * 1024
MAX_CURRICULUM_OUTPUT_BYTES = 128 * 1024 * 1024

_DECLARATION = re.compile(
    r"^\s*(?:(?:export|default|pub|public|async|static|abstract|declare)\s+)*"
    r"(?:function|class|interface|type|enum|def|fn|func|struct|trait|impl)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)"
)
_IMPORT = re.compile(r"^\s*(?:import\s|from\s.+\simport\s|#include\s|use\s)")
_ARROW_DECLARATION = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=")
_NONE_COMPARE_LEFT = re.compile(
    r"^(?P<left>.+?)(?P<space_left>\s*)(?P<operator>==|!=)(?P<space_right>\s*)None$",
    re.DOTALL,
)
_NONE_COMPARE_RIGHT = re.compile(
    r"^None(?P<space_left>\s*)(?P<operator>==|!=)(?P<space_right>\s*)(?P<right>.+)$",
    re.DOTALL,
)


def _normalised_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _line_span(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[max(0, start - 1):max(start - 1, end)]).strip()


def _python_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
    source: str,
    prefix: str = "",
) -> dict[str, Any]:
    decorator_lines = [item.lineno for item in getattr(node, "decorator_list", [])]
    start_line = min(decorator_lines + [node.lineno])
    end_line = int(getattr(node, "end_lineno", node.lineno))
    body_start = node.body[0].lineno if getattr(node, "body", None) else node.lineno
    signature = _line_span(lines, node.lineno, max(node.lineno, body_start - 1))
    if not signature:
        signature = lines[node.lineno - 1].strip()
    body = _line_span(lines, start_line, end_line) or ast.get_source_segment(source, node) or lines[node.lineno - 1]
    truncated = len(body) > MAX_SYMBOL_CHARS
    name = f"{prefix}.{node.name}" if prefix else node.name
    return {
        "name": name,
        "short_name": node.name,
        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
        "signature": signature[:1200],
        # Never label a partial source excerpt as a complete implementation.
        "body": "" if truncated else body,
        "truncated": truncated,
        "start_line": start_line,
        "end_line": end_line,
        "docstring": (ast.get_docstring(node, clean=False) or "")[:1200],
    }


def _parse_python_module(relative_path: str, source: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(source, filename=relative_path, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return None

    lines = source.splitlines()
    imports: list[str] = []
    symbols: list[dict[str, Any]] = []
    module_values: list[str] = []
    explicit_exports: list[str] | None = None

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.get_source_segment(source, node)
            if text:
                imports.append(text.strip()[:1200])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(_python_symbol(node, lines, source))
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        member = _python_symbol(child, lines, source, prefix=node.name)
                        member["is_member"] = True
                        symbols.append(member)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [
                target.id for target in targets
                if isinstance(target, ast.Name) and target.id != "__all__"
            ]
            module_values.extend(name for name in names if not name.startswith("_"))
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                try:
                    parsed = ast.literal_eval(node.value) if node.value is not None else None
                    if isinstance(parsed, (list, tuple, set)) and all(isinstance(item, str) for item in parsed):
                        explicit_exports = sorted(parsed) if isinstance(parsed, set) else list(parsed)
                except (ValueError, TypeError):
                    explicit_exports = None

    if explicit_exports is not None:
        exports = explicit_exports[:MAX_DECLARATIONS]
    else:
        exports = [
            *[
                item["name"] for item in symbols
                if not item.get("is_member") and not item["short_name"].startswith("_")
            ],
            *module_values,
        ][:MAX_DECLARATIONS]

    full_docstring = ast.get_docstring(tree, clean=False) or ""
    return {
        "docstring": full_docstring[:MAX_MODULE_CHARS],
        "docstring_truncated": len(full_docstring) > MAX_MODULE_CHARS,
        "imports": imports[:MAX_IMPORTS],
        "imports_truncated": len(imports) > MAX_IMPORTS,
        "symbols": symbols[:MAX_DECLARATIONS],
        "symbols_truncated": len(symbols) > MAX_DECLARATIONS,
        "exports": exports,
        "exports_truncated": len(exports) >= MAX_DECLARATIONS,
        "declarations": [item["name"] for item in symbols][:MAX_DECLARATIONS],
    }


def _logical_brace_deltas(lines: list[str]) -> list[int]:
    """Count structural braces while ignoring quoted text and source comments."""
    deltas: list[int] = []
    in_block_comment = False
    quote: str | None = None
    escaped = False

    for line in lines:
        delta = 0
        index = 0
        while index < len(line):
            char = line[index]
            next_pair = line[index:index + 2]

            if in_block_comment:
                if next_pair == "*/":
                    in_block_comment = False
                    index += 2
                else:
                    index += 1
                continue

            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue

            if next_pair == "//" or char == "#":
                break
            if next_pair == "/*":
                in_block_comment = True
                index += 2
                continue
            if char in ("'", '"') or char == chr(96):
                # Rust lifetimes and labels use apostrophes without a closing
                # quote on the same line. Do not treat those as string literals.
                if char == "'":
                    has_closing_quote = False
                    cursor = index + 1
                    escaped_quote = False
                    while cursor < len(line):
                        current = line[cursor]
                        if escaped_quote:
                            escaped_quote = False
                        elif current == "\\":
                            escaped_quote = True
                        elif current == "'":
                            has_closing_quote = True
                            break
                        cursor += 1
                    if not has_closing_quote:
                        index += 1
                        continue
                quote = char
                index += 1
                continue

            if char == "{":
                delta += 1
            elif char == "}":
                delta -= 1
            index += 1
        deltas.append(delta)
    return deltas


def _extract_logical_symbols(source: str) -> list[dict[str, Any]]:
    """Extract bounded declarations while distinguishing module exports from members."""
    lines = source.splitlines()
    # Keep the enclosing brace depth at each declaration so a method body closes
    # at its own parent scope rather than being mistaken for a truncated module function.
    starts: list[tuple[int, str, str, int, bool]] = []
    brace_deltas = _logical_brace_deltas(lines)
    brace_depth = 0
    for index, line in enumerate(lines):
        match = _DECLARATION.match(line) or _ARROW_DECLARATION.match(line)
        indentation = len(line) - len(line.lstrip())
        if match:
            is_member = brace_depth > 0 or indentation > 0
            starts.append((index, match.group(1), line.strip(), brace_depth, is_member))
        brace_depth = max(0, brace_depth + brace_deltas[index])

    symbols: list[dict[str, Any]] = []
    for position, (start, name, signature, parent_depth, is_member) in enumerate(starts[:MAX_DECLARATIONS]):
        next_start = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        hard_end = start + 300
        end = min(next_start, hard_end)
        truncated = next_start > hard_end
        local_brace_depth = parent_depth + brace_deltas[start]
        if local_brace_depth > parent_depth:
            closed = False
            for cursor in range(start + 1, end):
                local_brace_depth += brace_deltas[cursor]
                if local_brace_depth <= parent_depth:
                    end = cursor + 1
                    closed = True
                    break
            if not closed:
                truncated = True
        else:
            base_indent = len(lines[start]) - len(lines[start].lstrip())
            ended_at_dedent = False
            for cursor in range(start + 1, end):
                current = lines[cursor]
                if current.strip() and len(current) - len(current.lstrip()) <= base_indent:
                    end = cursor
                    ended_at_dedent = True
                    break
            if not ended_at_dedent and next_start > hard_end:
                truncated = True
        body = "\n".join(lines[start:end]).strip()
        if len(body) < 12:
            continue
        truncated = truncated or len(body) > MAX_SYMBOL_CHARS
        symbols.append({
            "name": name,
            "short_name": name,
            "kind": "declaration",
            "signature": signature[:1200],
            "body": "" if truncated else body,
            "truncated": truncated,
            "is_member": is_member,
            "is_container": bool(re.match(r"^(?:(?:pub|public)\s+)*impl\b", signature.strip())),
            "start_line": start + 1,
            "end_line": max(start + 1, end),
            "docstring": "",
        })
    return symbols


def _module_metadata(relative_path: str, source: str) -> dict[str, Any]:
    suffix = Path(relative_path).suffix.casefold()
    parsed = _parse_python_module(relative_path, source) if suffix in {".py", ".pyi"} else None
    lines = source.splitlines()
    if parsed is not None:
        imports = parsed["imports"]
        symbols = parsed["symbols"]
        docstring = parsed["docstring"]
        exports = parsed["exports"]
        declarations = parsed["declarations"]
        metadata_truncated = any((
            parsed["docstring_truncated"],
            parsed["imports_truncated"],
            parsed["symbols_truncated"],
            parsed["exports_truncated"],
        ))
    else:
        all_imports = [line.strip() for line in lines if _IMPORT.match(line)]
        declaration_count = sum(
            1 for line in lines
            if _DECLARATION.match(line) or _ARROW_DECLARATION.match(line)
        )
        all_symbols = _extract_logical_symbols(source)
        imports = all_imports[:MAX_IMPORTS]
        symbols = all_symbols[:MAX_DECLARATIONS]
        raw_docstring = (
            "\n".join(lines[:min(18, len(lines))])
            if lines and lines[0].lstrip().startswith(('"', "'", "/*", "//", "#"))
            else ""
        )
        docstring = raw_docstring[:MAX_MODULE_CHARS]
        exports = [
            item["name"] for item in symbols
            if (
                not item.get("is_member")
                and not item.get("is_container")
                and not item["short_name"].startswith("_")
            )
        ][:MAX_DECLARATIONS]
        declarations = [item["name"] for item in symbols][:MAX_DECLARATIONS]
        metadata_truncated = (
            len(all_imports) > MAX_IMPORTS
            or declaration_count > MAX_DECLARATIONS
            or len(raw_docstring) > MAX_MODULE_CHARS
        )

    structure = [
        f"Module: {relative_path}",
        "Purpose / module documentation:",
        docstring if docstring else "(No module docstring was found.)",
        "",
        "Imports and dependencies:",
        *(imports if imports else ["(No top-level imports were found.)"]),
        "",
        "Structural exports:",
        *([f"- {name}" for name in exports] if exports else ["(No public exports were identified.)"]),
        "",
        "Declarations and signatures:",
    ]
    if symbols:
        structure.extend(
            f"- {item['kind']} {item['name']} — {item['signature']}"
            for item in symbols[:MAX_DECLARATIONS]
        )
    else:
        structure.append("(No supported function, class, or declaration blocks were identified.)")

    full_structure = "\n".join(structure)
    structure_truncated = metadata_truncated or len(full_structure) > MAX_MODULE_CHARS * 3
    if structure_truncated:
        marker = "[PARTIAL STRUCTURAL SUMMARY: source declarations or metadata exceeded curriculum limits; omitted items are unknown.]"
        full_structure = full_structure[:MAX_MODULE_CHARS * 3 - len(marker)].rstrip() + "\n" + marker
    return {
        "imports": imports,
        "symbols": symbols,
        "exports": exports,
        "declarations": declarations,
        "structure": full_structure,
        "structure_truncated": structure_truncated,
    }


def _safe_none_patch_examples(relative_path: str, source: str) -> list[dict[str, str]]:
    """Create patches only for AST-verified Python comparisons against None."""
    if Path(relative_path).suffix.casefold() not in {".py", ".pyi"}:
        return []
    try:
        tree = ast.parse(source, filename=relative_path, type_comments=True)
    except (SyntaxError, ValueError, TypeError):
        return []

    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        operator = node.ops[0]
        if not isinstance(operator, (ast.Eq, ast.NotEq)):
            continue
        left_is_none = isinstance(node.left, ast.Constant) and node.left.value is None
        right = node.comparators[0]
        right_is_none = isinstance(right, ast.Constant) and right.value is None
        if left_is_none == right_is_none:
            continue
        before = ast.get_source_segment(source, node)
        if (
            not before
            or len(before) > 12_000
            or source.count(before) != 1
        ):
            # The live agent rejects ambiguous or oversized exact anchors.
            # Do not emit a curriculum repair task it cannot apply safely.
            continue
        if right_is_none:
            match = _NONE_COMPARE_LEFT.fullmatch(before.strip())
            if not match:
                continue
            keyword = "is" if isinstance(operator, ast.Eq) else "is not"
            after = match.group("left").rstrip() + " " + keyword + (match.group("space_right") or " ") + "None"
        else:
            match = _NONE_COMPARE_RIGHT.fullmatch(before.strip())
            if not match:
                continue
            keyword = "is" if isinstance(operator, ast.Eq) else "is not"
            after = "None " + keyword + (match.group("space_right") or " ") + match.group("right").lstrip()
        if len(after) > 12_000 or before == after or (before, after) in seen:
            continue
        seen.add((before, after))
        examples.append({"before": before, "after": after, "line": str(node.lineno)})
    return examples


def _fsync_output_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_jsonl_atomically(target: Path, records: list[dict[str, str]]) -> None:
    """Write a size-bounded JSONL dataset atomically and durably."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("Curriculum output must not be a symbolic link.")
    temporary_path: str | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".syntra-curriculum-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = handle.name
            for record in records:
                encoded = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                if len(encoded) > MAX_CURRICULUM_RECORD_BYTES:
                    raise ValueError(
                        f"Curriculum record exceeds the {MAX_CURRICULUM_RECORD_BYTES:,}-byte output record limit."
                    )
                if total_bytes + len(encoded) > MAX_CURRICULUM_OUTPUT_BYTES:
                    raise ValueError(
                        f"Curriculum output exceeds the {MAX_CURRICULUM_OUTPUT_BYTES:,}-byte dataset limit."
                    )
                handle.write(encoded)
                total_bytes += len(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if target.is_symlink():
            raise ValueError("Curriculum output must not be a symbolic link.")
        os.replace(temporary_path, target)
        temporary_path = None
        _fsync_output_directory(target.parent)
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def synthesize_repository_curriculum(
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    max_examples: int = 240,
) -> dict[str, Any]:
    """Generate JSONL prompt/completion records from an approved local repository."""
    if type(max_examples) is not int or not 4 <= max_examples <= MAX_INSTRUCTION_EXAMPLES:
        raise ValueError(f"max_examples must be between 4 and {MAX_INSTRUCTION_EXAMPLES:,}.")

    if root is None:
        configured_roots = [
            item.strip()
            for item in os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS", "").split(os.pathsep)
            if item.strip()
        ]
        root_candidates: list[str | Path] = [PROJECT_ROOT, *configured_roots]
    else:
        root_candidates = [root]

    corpus_roots: list[Path] = []
    seen_roots: set[str] = set()
    for candidate in root_candidates:
        approved = resolve_corpus_root(candidate)
        key = os.path.normcase(os.fspath(approved))
        if key not in seen_roots:
            corpus_roots.append(approved)
            seen_roots.add(key)
    if not corpus_roots:
        raise ValueError("No approved local repository roots were configured.")

    output_root = corpus_roots[0]
    target = resolve_curriculum_output_path(output_path, output_root)
    corpus = load_corpus(corpus_roots, include_tokens=False)
    if len(corpus.files) != len(corpus.documents):
        raise RuntimeError("The bounded source reader returned inconsistent file/document counts.")

    records: list[dict[str, str]] = []
    seen_prompts: set[str] = set()
    counts = {
        "module_comprehension": 0,
        "symbol_implementation": 0,
        "symbol_comprehension": 0,
        "anchor_patch_repair": 0,
    }
    module_count = 0
    symbol_count = 0
    patch_candidate_count = 0

    # Keep a bounded, deterministic candidate reservoir per instruction type.
    # This prevents a large first module from consuming the entire dataset cap.
    category_pools: dict[str, list[tuple[int, str, int, dict[str, str]]]] = {
        kind: [] for kind in counts
    }
    per_file_symbol_limit = max(
        2,
        min(8, (max_examples + max(1, len(corpus.files)) - 1) // max(1, len(corpus.files))),
    )
    per_file_patch_limit = 2
    candidate_sequence = 0
    candidate_pool_bytes = 0
    candidate_sizes: dict[str, int] = {}
    candidate_budget_skips = 0

    def offer_candidate(
        kind: str,
        relative_path: str,
        selector: str,
        split_group: str,
        prompt: str,
        completion: str,
    ) -> None:
        nonlocal candidate_sequence, candidate_pool_bytes, candidate_budget_skips
        prompt = prompt.strip()
        completion = completion.strip()
        if len(prompt) < 20 or len(completion) < 16:
            return
        normalised = _normalised_prompt(prompt)
        identity = f"{relative_path}\\0{selector}\\0{normalised}"
        rank = int(hashlib.sha256((kind + "\\0" + identity).encode("utf-8")).hexdigest(), 16)
        candidate = {
            "prompt": prompt,
            "completion": completion,
            "kind": kind,
            "source_path": relative_path,
            "split_group": split_group,
        }
        heap = category_pools[kind]
        if identity in candidate_sizes:
            return
        candidate_size = len(
            (json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        )
        if candidate_size > MAX_CURRICULUM_CANDIDATE_BYTES:
            candidate_budget_skips += 1
            return

        candidate_sequence += 1
        entry = (-rank, identity, candidate_sequence, candidate)
        if len(heap) < max_examples:
            if candidate_pool_bytes + candidate_size > MAX_CURRICULUM_CANDIDATE_BYTES:
                candidate_budget_skips += 1
                return
            heapq.heappush(heap, entry)
            candidate_sizes[identity] = candidate_size
            candidate_pool_bytes += candidate_size
        elif rank < -heap[0][0]:
            removed = heap[0]
            removed_size = candidate_sizes.get(removed[1], 0)
            if candidate_pool_bytes - removed_size + candidate_size > MAX_CURRICULUM_CANDIDATE_BYTES:
                candidate_budget_skips += 1
                return
            heapq.heapreplace(heap, entry)
            candidate_sizes.pop(removed[1], None)
            candidate_sizes[identity] = candidate_size
            candidate_pool_bytes = candidate_pool_bytes - removed_size + candidate_size

    source_documents = sorted(
        zip(corpus.files, corpus.documents),
        key=lambda item: os.path.normcase(item[0]),
    )
    for source_path, source_text in source_documents:
        resolved = Path(source_path).resolve()
        relative_path = None
        for approved_root in corpus_roots:
            try:
                relative_path = resolved.relative_to(approved_root).as_posix()
                break
            except ValueError:
                continue
        if relative_path is None:
            raise ValueError("A source file escaped all approved curriculum roots.")
        if not source_text.strip():
            continue

        module_count += 1
        metadata = _module_metadata(relative_path, source_text)
        structure = metadata["structure"]
        module_split_group = f"module:{relative_path}"
        offer_candidate(
            "module_comprehension",
            relative_path,
            "module-purpose-exports",
            f"module:{relative_path}",
            (
                f"Instruction: Explain the purpose and structural exports of module {relative_path}. "
                "Describe its documented purpose, imports, public exports, and declared symbols."
            ),
            structure,
        )

        symbol_count += len(metadata["symbols"])
        eligible_symbols = [
            item for item in metadata["symbols"]
            if not item.get("truncated")
        ]
        eligible_symbols = sorted(
            eligible_symbols,
            key=lambda item: hashlib.sha256(
                f"{relative_path}\\0{item['name']}".encode("utf-8")
            ).hexdigest(),
        )[:per_file_symbol_limit]
        for symbol in eligible_symbols:
            signature = symbol["signature"] or symbol["name"]
            source_block = symbol["body"]
            offer_candidate(
                "symbol_implementation",
                relative_path,
                symbol["name"],
                module_split_group,
                (
                    f"Instruction: Write a clean implementation of symbol {symbol['name']} "
                    f"with its signature from module {relative_path}.\n"
                    f"Signature:\n{signature}\n"
                    "Use the repository's observed implementation as the source of truth."
                ),
                source_block,
            )
            explanation = [
                f"Symbol: {symbol['name']}",
                f"Module: {relative_path}",
                f"Kind: {symbol['kind']}",
                f"Lines: {symbol['start_line']}-{symbol['end_line']}",
                f"Signature: {signature}",
            ]
            if symbol["docstring"]:
                explanation.extend(["Documentation:", symbol["docstring"]])
            explanation.extend(["Exact source implementation:", source_block])
            offer_candidate(
                "symbol_comprehension",
                relative_path,
                symbol["name"],
                module_split_group,
                (
                    f"Instruction: Explain the role, signature, and implementation of "
                    f"symbol {symbol['name']} in {relative_path}."
                ),
                "\n".join(explanation),
            )

        file_patch_count = 0
        for patch in _safe_none_patch_examples(relative_path, source_text):
            patch_candidate_count += 1
            if file_patch_count >= per_file_patch_limit:
                continue
            file_patch_count += 1
            patch_payload = {
                "format": "SYNTRA_AGENT_PATCH_V1",
                "summary": "Use identity comparison when checking the None sentinel.",
                "edits": [{
                    "path": relative_path,
                    "find": patch["before"],
                    "replace": patch["after"],
                    "reason": (
                        "None is a singleton sentinel. Python code should use identity "
                        "comparison (is / is not) for this check."
                    ),
                }],
            }
            offer_candidate(
                "anchor_patch_repair",
                relative_path,
                f"none-comparison-line-{patch['line']}",
                module_split_group,
                (
                    "Instruction: Apply a SYNTRA_AGENT_PATCH_V1 change to fix the "
                    f"non-identity None comparison in {relative_path} at line {patch['line']}.\n"
                    f"Exact source anchor:\n{patch['before']}\n"
                    "Return a structured search/replace patch using the repository-relative path."
                ),
                json.dumps(patch_payload, ensure_ascii=False),
            )

    def add_pair(kind: str, candidate: dict[str, str]) -> bool:
        if len(records) >= max_examples:
            return False
        prompt = candidate["prompt"].strip()
        completion = candidate["completion"].strip()
        normalised = _normalised_prompt(prompt)
        if normalised in seen_prompts:
            return False
        seen_prompts.add(normalised)
        records.append(candidate)
        counts[kind] += 1
        return True

    ordered_pools = {
        kind: [
            entry[3]
            for entry in sorted(heap, key=lambda entry: (-entry[0], entry[1], entry[2]))
        ]
        for kind, heap in category_pools.items()
    }
    pool_positions = {kind: 0 for kind in counts}
    category_order = (
        "module_comprehension",
        "symbol_implementation",
        "symbol_comprehension",
        "anchor_patch_repair",
    )
    # Interleave types and source-ranked candidates so module order cannot starve
    # later files or rarer repair examples.
    while len(records) < max_examples:
        made_progress = False
        for kind in category_order:
            candidates = ordered_pools[kind]
            while pool_positions[kind] < len(candidates):
                candidate = candidates[pool_positions[kind]]
                pool_positions[kind] += 1
                if add_pair(kind, candidate):
                    made_progress = True
                    break
            if len(records) >= max_examples:
                break
        if not made_progress:
            break

    if len(records) < 2:
        raise ValueError(
            "The approved repository did not yield two distinct, source-grounded instruction examples."
        )

    _write_jsonl_atomically(target, records)

    # Lazy import avoids a cycle: instruction.py re-exports this function.
    from .instruction import audit_instruction_dataset

    audit = audit_instruction_dataset(target)
    represented_source_files = len({item["source_path"] for item in records})
    warnings: list[str] = []
    if not counts["anchor_patch_repair"]:
        warnings.append(
            "No supported, mechanically grounded patch pattern was found; no synthetic repair examples were fabricated."
        )
    if candidate_budget_skips:
        warnings.append(
            f"{candidate_budget_skips:,} candidate(s) were excluded by the "
            f"{MAX_CURRICULUM_CANDIDATE_BYTES:,}-byte in-memory candidate budget."
        )
    if len(records) >= max_examples and represented_source_files < module_count:
        warnings.append(
            f"The {max_examples:,}-example curriculum cap selected {represented_source_files:,} of "
            f"{module_count:,} source modules; increase --max-examples for broader coverage."
        )
    source_token_limit = 5_000_000
    source_file_limit = 10_000
    if corpus.token_count >= source_token_limit:
        warnings.append(
            f"Repository scanning reached the {source_token_limit:,}-token source limit; the corpus may be incomplete."
        )
    if len(corpus.files) >= source_file_limit:
        warnings.append(
            f"Repository scanning reached the {source_file_limit:,}-file source limit; additional files may not be included."
        )
    return {
        "dataset_path": str(target),
        "workspace_root": str(output_root),
        "workspace_roots": [str(item) for item in corpus_roots],
        "source_files": len(corpus.files),
        "module_count": module_count,
        "represented_source_files": represented_source_files,
        "source_tokens": corpus.token_count,
        "symbol_count": symbol_count,
        "patch_candidate_count": patch_candidate_count,
        "example_count": len(records),
        "examples_by_type": counts,
        "selection_strategy": "deterministic-source-ranked-round-robin",
        "audit": audit,
        "warnings": warnings,
    }


__all__ = ["synthesize_repository_curriculum"]

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
from typing import Any, Iterable

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
MIN_AST_SYMBOL_QUALITY_SCORE = 1

_DECLARATION = re.compile(
    r"^\s*(?:(?:export|default|pub|public|async|static|abstract|declare)\s+)*"
    r"(?:function|class|interface|type|enum|def|fn|fun|func|struct|trait|impl)\s+"
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


_BUILTIN_CALL_NAMES = {
    "abs", "all", "any", "bool", "bytes", "dict", "enumerate", "filter",
    "float", "getattr", "hasattr", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "object", "open", "print",
    "property", "range", "repr", "reversed", "round", "set", "setattr",
    "slice", "sorted", "staticmethod", "classmethod", "str", "sum", "super",
    "tuple", "type", "zip",
}


def _is_trivial_python_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> bool:
    """Reject empty stubs and generated property getter/setter boilerplate."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return True
    if isinstance(node, ast.ClassDef):
        return all(
            isinstance(item, ast.Pass)
            or (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and item.value.value is Ellipsis)
            for item in body
        )
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is Ellipsis:
        return True
    if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Attribute):
        return (
            isinstance(statement.value.value, ast.Name)
            and statement.value.value.id in {"self", "cls"}
        )
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = statement.value
        argument_names = {
            argument.arg
            for argument in (
                list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else []
            )
        }
        return (
            len(targets) == 1
            and isinstance(targets[0], ast.Attribute)
            and isinstance(targets[0].value, ast.Name)
            and targets[0].value.id in {"self", "cls"}
            and isinstance(value, ast.Name)
            and value.id in argument_names
        )
    return False


def _python_symbol_quality(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> dict[str, Any]:
    """Score documented, typed and logically substantive AST declarations."""
    body = list(node.body)
    has_docstring = bool(ast.get_docstring(node, clean=False))
    if has_docstring and body and isinstance(body[0], ast.Expr):
        body = body[1:]

    annotation_count = 0
    member_count = 0
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = (
            list(node.args.posonlyargs) + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        annotation_count = sum(argument.annotation is not None for argument in arguments)
        annotation_count += int(node.args.vararg is not None and node.args.vararg.annotation is not None)
        annotation_count += int(node.args.kwarg is not None and node.args.kwarg.annotation is not None)
        annotation_count += int(node.returns is not None)
    elif isinstance(node, ast.ClassDef):
        member_count = sum(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not _is_trivial_python_symbol(item)
            for item in body
        )
        annotation_count = sum(isinstance(item, ast.AnnAssign) and item.annotation is not None for item in body)

    logic_nodes = [
        item for statement in body for item in ast.walk(statement)
        if not isinstance(item, (ast.Load, ast.Store, ast.Del, ast.Name, ast.Constant, ast.arg, ast.Expr))
    ]
    control_flow = sum(
        isinstance(item, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match))
        for item in logic_nodes
    )
    calls = sum(isinstance(item, ast.Call) for item in logic_nodes)
    operations = sum(
        isinstance(item, (
            ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ))
        for item in logic_nodes
    )
    statement_count = (
        sum(
            not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            or not _is_trivial_python_symbol(item)
            for item in body
        )
        if isinstance(node, ast.ClassDef)
        else len(body)
    )
    score = (
        (2 if has_docstring else 0)
        + min(3, annotation_count)
        + min(2, member_count)
        + (2 if control_flow else 0)
        + (1 if calls else 0)
        + (1 if operations else 0)
        + (1 if statement_count > 1 else 0)
    )
    return {
        "score": score,
        "has_docstring": has_docstring,
        "annotation_count": annotation_count,
        "member_count": member_count,
        "control_flow_count": control_flow,
        "call_count": calls,
        "operation_count": operations,
        "boilerplate": _is_trivial_python_symbol(node),
    }


class _CanonicalSymbolNames(ast.NodeTransformer):
    """Normalize local identifiers while retaining operators, attributes and called APIs."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "__symbol__"
        self._strip_docstring(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "__symbol__"
        self._strip_docstring(node)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = "__class__"
        self._strip_docstring(node)
        return self.generic_visit(node)

    @staticmethod
    def _strip_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = node.body[1:]

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "__arg__"
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in _BUILTIN_CALL_NAMES and not node.id.isupper() and not node.id.startswith("__"):
            node.id = "__name__"
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name):
            # Retain called API names; they carry behavior, unlike local variable names.
            node.args = [self.visit(item) for item in node.args]
            node.keywords = [self.visit(item) for item in node.keywords]
            return node
        return self.generic_visit(node)


def _python_symbol_fingerprint(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            body = body[1:]
    normalized = ast.Module(body=body, type_ignores=[])
    normalized = _CanonicalSymbolNames().visit(normalized)
    ast.fix_missing_locations(normalized)
    structural = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(structural.encode("utf-8")).hexdigest()


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
    quality = _python_symbol_quality(node)
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
        "quality_score": quality["score"],
        "quality": quality,
        "structural_fingerprint": _python_symbol_fingerprint(node),
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

    def append_class_members(node: ast.ClassDef, prefix: str) -> None:
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                nested = _python_symbol(child, lines, source, prefix=prefix)
                nested["is_member"] = True
                symbols.append(nested)
                append_class_members(child, f"{prefix}.{child.name}")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                member = _python_symbol(child, lines, source, prefix=prefix)
                member["is_member"] = True
                symbols.append(member)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.get_source_segment(source, node)
            if text:
                imports.append(text.strip()[:1200])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(_python_symbol(node, lines, source))
            if isinstance(node, ast.ClassDef):
                append_class_members(node, node.name)
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
                    index += 1
                    continue
                if char == "\\":
                    escaped = True
                    index += 1
                    continue
                if len(quote) == 3:
                    if line[index:index + 3] == quote:
                        quote = None
                        index += 3
                    else:
                        index += 1
                    continue
                if char == quote:
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
                delimiter = line[index:index + 3]
                if delimiter in ("'''", '"""'):
                    quote = delimiter
                    index += 3
                    continue
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
    exclude_paths: Iterable[str | Path] = (),
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
    corpus = load_corpus(corpus_roots, include_tokens=False, exclude_paths=exclude_paths)
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
    seen_structural_symbols: set[str] = set()
    quality_statistics = {
        "quality_scored_symbols": 0,
        "low_quality_symbols_skipped": 0,
        "boilerplate_symbols_skipped": 0,
        "near_duplicate_symbols_skipped": 0,
    }

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
        source_root: Path | None = None
        source_root_index: int | None = None
        for root_index, approved_root in enumerate(corpus_roots):
            try:
                relative_path = resolved.relative_to(approved_root).as_posix()
                source_root = approved_root
                source_root_index = root_index
                break
            except ValueError:
                continue
        if relative_path is None or source_root is None or source_root_index is None:
            raise ValueError("A source file escaped all approved curriculum roots.")
        is_active_workspace_source = (
            os.path.normcase(os.fspath(source_root))
            == os.path.normcase(os.fspath(output_root))
        )
        if not is_active_workspace_source:
            # Different approved roots may contain the same relative filename.
            # Namespace read-only roots to avoid source_path and split-group collisions.
            relative_path = f"approved-root-{source_root_index + 1}/{relative_path}"
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
        eligible_symbols = []
        for item in metadata["symbols"]:
            if item.get("truncated"):
                continue
            quality = item.get("quality")
            fingerprint = item.get("structural_fingerprint")
            if isinstance(quality, dict):
                quality_statistics["quality_scored_symbols"] += 1
                if quality.get("boilerplate") is True:
                    quality_statistics["boilerplate_symbols_skipped"] += 1
                    continue
                score = quality.get("score")
                if type(score) is not int or score < MIN_AST_SYMBOL_QUALITY_SCORE:
                    quality_statistics["low_quality_symbols_skipped"] += 1
                    continue
                if isinstance(fingerprint, str) and fingerprint:
                    if fingerprint in seen_structural_symbols:
                        quality_statistics["near_duplicate_symbols_skipped"] += 1
                        continue
                    seen_structural_symbols.add(fingerprint)
            eligible_symbols.append(item)
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
        # Patch paths are relative to the active workspace and are applied by
        # the local agent there. Other approved roots are read-only curriculum
        # sources; never train a repair action that could target an unrelated
        # same-named file in the active workspace.
        patch_examples = (
            _safe_none_patch_examples(relative_path, source_text)
            if is_active_workspace_source
            else []
        )
        for patch in patch_examples:
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
            "No supported, mechanically grounded patch pattern was found in the active workspace; no synthetic repair examples were fabricated."
        )
    if len(corpus_roots) > 1:
        warnings.append(
            "Anchor-patch examples are restricted to the active output workspace; additional approved roots contribute read-only comprehension and implementation examples."
        )
    if quality_statistics["boilerplate_symbols_skipped"]:
        warnings.append(
            f"{quality_statistics['boilerplate_symbols_skipped']:,} empty or trivial AST getter/setter/stub symbol(s) were excluded."
        )
    if quality_statistics["low_quality_symbols_skipped"]:
        warnings.append(
            f"{quality_statistics['low_quality_symbols_skipped']:,} AST symbol(s) below the minimum structural quality score were excluded."
        )
    if quality_statistics["near_duplicate_symbols_skipped"]:
        warnings.append(
            f"{quality_statistics['near_duplicate_symbols_skipped']:,} structurally equivalent AST symbol(s) were deduplicated across the approved repositories."
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
        "quality_statistics": quality_statistics,
        "minimum_ast_symbol_quality_score": MIN_AST_SYMBOL_QUALITY_SCORE,
        "audit": audit,
        "warnings": warnings,
    }


__all__ = ["synthesize_repository_curriculum"]

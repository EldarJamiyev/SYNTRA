"""Bounded, dependency-free structural curation for JavaScript, TypeScript, Go and Rust."""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Any

LANGUAGE_SUFFIXES = {
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx", ".mts", ".cts"},
    "go": {".go"},
    "rust": {".rs"},
}
SUPPORTED_LANGUAGES = ("python", "javascript", "typescript", "go", "rust")
MAX_POLYGLOT_SYMBOL_CHARS = 12_000
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_STRING_TOKEN = r"""(?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
_TEMPLATE_TOKEN = r"|(?P<template>" + re.escape(chr(96)) + r"(?:\\.|[^" + re.escape(chr(96)) + r"\\])*" + re.escape(chr(96)) + r")"
_TOKEN = re.compile(
    _STRING_TOKEN + _TEMPLATE_TOKEN
    + r"""|(?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)"""
    + r"""|(?P<number>(?:0[xX][0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?))"""
    + r"""|(?P<operator>===|!==|=>|==|!=|<=|>=|&&|\|\||\?\?|::|->|\+=|-=|\*=|/=|\+\+|--|.)"""
)
_KEYWORDS = set("""
abstract as async await break case catch class const continue debugger default delete do
else enum export extends false finally for from function get if implements import in
instanceof interface let new null of package private protected public return set static
super switch this throw true try typeof var void while with yield type declare namespace
func go goto map chan range select defer fallthrough struct fn mut pub crate mod use impl
trait match loop move ref self Self where dyn unsafe box err nil Option Some None
""".split())
_TYPE_WORDS = set("""
any unknown never string number boolean bigint symbol object void undefined null
int int8 int16 int32 int64 uint uint8 uint16 uint32 uint64 uintptr byte rune
bool error float32 float64 complex64 complex128 usize isize char str
""".split())


def language_for_path(relative_path: str) -> str | None:
    suffix = PurePosixPath(relative_path).suffix.casefold()
    if suffix in {".py", ".pyi"}:
        return "python"
    for language, suffixes in LANGUAGE_SUFFIXES.items():
        if suffix in suffixes:
            return language
    return None


def _code_mask(source: str) -> list[bool]:
    """Mark code positions while ignoring comments and quoted literals."""
    mask = [False] * len(source)
    quote: str | None = None
    escaped = False
    block_comment = False
    line_comment = False
    index = 0
    while index < len(source):
        char = source[index]
        pair = source[index:index + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if pair == "*/":
                block_comment = False
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
        if pair == "//":
            line_comment = True
            index += 2
            continue
        if pair == "/*":
            block_comment = True
            index += 2
            continue
        if char in ("'", '"', chr(96)):
            quote = char
            index += 1
            continue
        mask[index] = True
        index += 1
    return mask


def _strip_comments(source: str) -> str:
    """Remove comments without changing offsets or quoted source literals."""
    chars = list(source)
    quote: str | None = None
    escaped = False
    block_comment = False
    line_comment = False
    index = 0
    while index < len(source):
        char = source[index]
        pair = source[index:index + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
            elif char != "\r":
                chars[index] = " "
            index += 1
            continue
        if block_comment:
            if pair == "*/":
                chars[index] = chars[index + 1] = " "
                block_comment = False
                index += 2
            else:
                if char != "\n":
                    chars[index] = " "
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
        if pair == "//":
            chars[index] = chars[index + 1] = " "
            line_comment = True
            index += 2
            continue
        if pair == "/*":
            chars[index] = chars[index + 1] = " "
            block_comment = True
            index += 2
            continue
        if char in ("'", '"', chr(96)):
            quote = char
        index += 1
    return "".join(chars)


def _leading_documentation(lines: list[str], start_line: int) -> str:
    """Read at most 20 contiguous documentation/comment lines above a symbol."""
    cursor = max(0, start_line - 2)
    collected: list[str] = []
    while cursor >= 0 and len(collected) < 20:
        stripped = lines[cursor].strip()
        if not stripped:
            if collected:
                break
            cursor -= 1
            continue
        if stripped.startswith(("///", "//!", "/**", "/*", "*", "//")) or stripped.endswith("*/"):
            collected.append(stripped)
            cursor -= 1
            continue
        break
    return "\n".join(reversed(collected))


def _parameter_names(signature: str) -> list[str]:
    """Extract uncomplicated parameter identifiers in their declared order."""
    opening = signature.find("(")
    if opening < 0:
        return []
    depth = 0
    quote: str | None = None
    closing = -1
    for index in range(opening, len(signature)):
        char = signature[index]
        if quote is not None:
            if char == quote and (index == 0 or signature[index - 1] != "\\"):
                quote = None
            continue
        if char in ("'", '"', chr(96)):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing < 0:
        return []
    parameters = signature[opening + 1:closing]
    segments: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(parameters):
        if char in ("[", "(", "{", "<"):
            depth += 1
        elif char in ("]", ")", "}", ">"):
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            segments.append(parameters[start:index])
            start = index + 1
    segments.append(parameters[start:])
    result: list[str] = []
    for segment in segments:
        match = _IDENT.search(segment.strip().lstrip("*&"))
        if match:
            name = match.group(0)
            if name not in {"self", "this", "mut", "ref"} and name not in result:
                result.append(name)
    return result[:64]


def _polyglot_fingerprint(language: str, name: str, signature: str, body: str) -> str:
    """Normalize local identifiers while preserving operators, calls and members."""
    source = _strip_comments(body)
    tokens = [match.group(0) for match in _TOKEN.finditer(source)]
    names = {value: f"__arg_{index}__" for index, value in enumerate(_parameter_names(signature))}
    next_local = len(names)
    normalized: list[str] = []
    for index, token in enumerate(tokens):
        if token.isspace():
            continue
        if not _IDENT.fullmatch(token):
            normalized.append(token)
            continue
        previous = tokens[index - 1] if index else ""
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == name:
            normalized.append("__symbol__")
        elif token in _KEYWORDS or token in _TYPE_WORDS:
            normalized.append(token)
        elif previous in {".", "->"}:
            normalized.append(token)
        elif following == "(":
            normalized.append(token)
        elif token[:1].isupper():
            normalized.append(token)
        else:
            if token not in names:
                names[token] = f"__local_{next_local}__"
                next_local += 1
            normalized.append(names[token])
    return hashlib.sha256((language + "\0" + " ".join(normalized)).encode("utf-8")).hexdigest()


def _is_trivial_block(body: str) -> bool:
    code_source = _strip_comments(body)
    block = re.search(r"\{(.*)\}\s*$", code_source, re.DOTALL)
    if block:
        code_source = block.group(1)
    code = re.sub(r"\s+", "", code_source).strip("{};")
    if not code:
        return True
    if re.fullmatch(
        r"(?:pass|\.\.\.|todo!\(\)|unimplemented!\(\)|return;?|panic!\([\"'][^\"']*[\"']\);?)",
        code,
    ):
        return True
    if re.fullmatch(r"return(?:this|self|[A-Za-z_]\w*)\.[A-Za-z_$]\w*;?", code):
        return True
    if re.fullmatch(r"return[A-Za-z_$][A-Za-z0-9_$]*;?", code):
        return True
    if re.fullmatch(
        r"(?:this|self|[A-Za-z_]\w*)\.[A-Za-z_$]\w*=[A-Za-z_$]\w*;?",
        code,
    ):
        return True
    return False


def _quality(language: str, symbol: dict[str, Any], source_lines: list[str]) -> dict[str, Any]:
    body = symbol.get("body", "")
    signature = symbol.get("signature", "")
    start_line = int(symbol.get("start_line", 1))
    docs = _leading_documentation(source_lines, start_line)
    has_docstring = bool(docs.strip()) or bool(
        re.match(r"\s*/\*\*|^\s*///|^\s*//!", body)
    )
    code = _strip_comments(body)
    control_flow = len(re.findall(
        r"\b(?:if|else\s+if|for|while|switch|match|try|catch|select)\b", code
    ))
    call_count = len(re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\(", code))
    if symbol.get("name"):
        call_count = max(0, call_count - 1)
    operations = len(re.findall(
        r"===|!==|==|!=|<=|>=|&&|\|\||\?\?|(?:\+\+|--|\+=|-=|\*=|/=)|(?<![=<>!])(?:[+*/%<>])",
        code,
    ))
    statements = [
        line.strip() for line in code.splitlines()
        if line.strip() and line.strip() not in {"{", "}", "};"}
    ]
    typed_parameters = 0
    if language in {"javascript", "typescript"}:
        typed_parameters = len(re.findall(
            r"\b[A-Za-z_$][\w$]*\s*:\s*[\w$.[\]<>|& ]+", signature
        ))
        if re.search(r"\)\s*:\s*[\w$.[\]<>|&]+", signature):
            typed_parameters += 1
    elif language == "go":
        match = re.search(r"\((.*)\)", signature)
        if match:
            typed_parameters = sum(
                1 for item in match.group(1).split(",")
                if len(_IDENT.findall(item)) >= 2
            )
        if re.search(r"\)\s*(?:\([^)]*\)|\*?[A-Za-z_]\w*)", signature):
            typed_parameters += 1
    elif language == "rust":
        typed_parameters = len(re.findall(r"\b[A-Za-z_]\w*\s*:\s*[^,)=]+", signature))
        if re.search(r"\)\s*->\s*[^({]+", signature):
            typed_parameters += 1
    if language in {"typescript", "rust"}:
        typed_parameters += len(re.findall(
            r"(?m)^\s*[A-Za-z_$][A-Za-z0-9_$]*\??\s*:\s*[^;{},\n]+[;,]?",
            code,
        ))
    elif language == "go":
        typed_parameters += len(re.findall(
            r"(?m)^\s*[A-Za-z_]\w*\s+[\w*.\[\]]+\s*$",
            code,
        ))
    member_count = 0
    if symbol.get("is_container") or symbol.get("kind") in {"class", "interface", "struct", "trait", "impl"}:
        member_count = len(re.findall(
            r"(?m)^\s*(?:(?:pub|public|private|protected|static|async)\s+)*(?:fn|func|function|def)\s+[A-Za-z_$]\w*",
            body,
        ))
    boilerplate = _is_trivial_block(body)
    score = (
        (2 if has_docstring else 0)
        + min(3, typed_parameters)
        + min(2, member_count)
        + (2 if control_flow else 0)
        + (1 if call_count else 0)
        + (1 if operations else 0)
        + (1 if len(statements) > 1 else 0)
    )
    return {
        "language": language, "score": score, "has_docstring": has_docstring,
        "annotation_count": typed_parameters, "member_count": member_count,
        "control_flow_count": control_flow, "call_count": call_count,
        "operation_count": operations, "statement_count": len(statements),
        "boilerplate": boilerplate,
    }


def enrich_logical_symbols(
    relative_path: str,
    source: str,
    symbols: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach quality signals and fingerprints to complete non-Python blocks."""
    language = language_for_path(relative_path)
    if language is None or language == "python":
        return symbols
    lines = source.splitlines()
    enriched: list[dict[str, Any]] = []
    for original in symbols:
        item = dict(original)
        body = item.get("body", "")
        if not item.get("truncated") and body and len(body) <= MAX_POLYGLOT_SYMBOL_CHARS:
            quality = _quality(language, item, lines)
            item["quality"] = quality
            item["quality_score"] = quality["score"]
            item["structural_fingerprint"] = _polyglot_fingerprint(
                language, str(item.get("short_name") or item.get("name") or ""),
                str(item.get("signature") or ""), body,
            )
        enriched.append(item)
    return enriched


def _matching_brace(source: str, opening: int, code_mask: list[bool]) -> int:
    depth = 0
    for index in range(opening, len(source)):
        if not code_mask[index]:
            continue
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def safe_repair_examples(relative_path: str, source: str) -> list[dict[str, str]]:
    """Return only narrow, exact, semantics-conservative language-specific repairs."""
    language = language_for_path(relative_path)
    if language not in LANGUAGE_SUFFIXES:
        return []
    examples: list[dict[str, str]] = []
    code_mask = _code_mask(source)

    if language in {"javascript", "typescript"}:
        pattern = re.compile(
            r"""(?P<expression>typeof\s+(?:\([^()\n]+\)|[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*))"""
            r"""(?P<space>\s*)(?<![=!])(?P<operator>==|!=)(?![=])"""
            r"""(?P<space_after>\s*)(?P<literal>["'](?:string|number|boolean|bigint|symbol|undefined|function|object)["'])"""
        )
        for match in pattern.finditer(source):
            if not code_mask[match.start("expression")] or not code_mask[match.start("operator")]:
                continue
            before = match.group(0)
            if source.count(before) != 1:
                continue
            after = (
                match.group("expression") + match.group("space")
                + match.group("operator") + "=" + match.group("space_after")
                + match.group("literal")
            )
            examples.append({
                "before": before, "after": after,
                "line": str(source.count("\n", 0, match.start()) + 1),
                "rule": "typeof-strict-equality", "language": language,
                "summary": "Use strict equality for a JavaScript typeof comparison.",
                "reason": "typeof always returns a string, so strict equality avoids coercive comparison without changing the tested primitive-type result.",
            })

    elif language == "go":
        signature_pattern = re.compile(
            r"(?m)^\s*func\s+(?:\([^)\n]*\)\s*)?[A-Za-z_]\w*\s*"
            r"\([^)\n]*\)\s*\(\s*\*?[A-Za-z_]\w*(?:\[[^\]]+\])?\s*,\s*error\s*\)\s*\{"
        )
        for signature in signature_pattern.finditer(source):
            if not code_mask[signature.start()]:
                continue
            opening = source.find("{", signature.start(), signature.end())
            closing = _matching_brace(source, opening, code_mask)
            if opening < 0 or closing < 0:
                continue
            body = source[opening + 1:closing]
            guard = re.search(
                r"if\s+(?P<error>[A-Za-z_]\w*)\s*!=\s*nil\s*\{\s*return\s+nil\s*\}",
                body,
            )
            if not guard:
                continue
            before = "return nil"
            anchor = opening + 1 + guard.start()
            offset = source.find(before, anchor, opening + 1 + guard.end())
            if offset < 0 or source.count(before) != 1:
                continue
            examples.append({
                "before": before, "after": f"return nil, {guard.group('error')}",
                "line": str(source.count("\n", 0, offset) + 1),
                "rule": "go-error-return-value", "language": language,
                "summary": "Return the encountered error with the nil result.",
                "reason": "The function returns a result and an error; the guarded failure path must return both declared values.",
            })

    elif language == "rust":
        signature_pattern = re.compile(
            r"\bfn\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)\s*"
            r"->\s*Option\s*<\s*&[A-Za-z_]\w*\s*>\s*\{"
        )
        for signature in signature_pattern.finditer(source):
            if not code_mask[signature.start()]:
                continue
            params = signature.group("params")
            slice_match = re.search(r"\b(?P<slice>[A-Za-z_]\w*)\s*:\s*&\s*\[", params)
            index_match = re.search(r"\b(?P<index>[A-Za-z_]\w*)\s*:\s*usize\b", params)
            if not slice_match or not index_match:
                continue
            slice_name = slice_match.group("slice")
            index_name = index_match.group("index")
            opening = source.find("{", signature.start(), signature.end())
            closing = _matching_brace(source, opening, code_mask)
            if opening < 0 or closing < 0:
                continue
            body = source[opening + 1:closing]
            if re.search(
                rf"\bif\s+{re.escape(index_name)}\s*(?:>=|>)\s*{re.escape(slice_name)}\.len\s*\(",
                body,
            ):
                continue
            access = re.compile(
                rf"Some\(\s*&{re.escape(slice_name)}\s*\[\s*{re.escape(index_name)}\s*\]\s*\)"
            )
            for match in access.finditer(body):
                before = match.group(0)
                absolute = opening + 1 + match.start()
                if source.count(before) != 1:
                    continue
                examples.append({
                    "before": before, "after": f"{slice_name}.get({index_name})",
                    "line": str(source.count("\n", 0, absolute) + 1),
                    "rule": "rust-option-bounds-checked-access", "language": language,
                    "summary": "Use Option-returning slice access instead of unchecked indexing.",
                    "reason": "The function returns Option<&T>; slice.get returns that type and safely produces None for an out-of-range index.",
                })
    return examples


__all__ = [
    "LANGUAGE_SUFFIXES", "SUPPORTED_LANGUAGES", "enrich_logical_symbols",
    "language_for_path", "safe_repair_examples",
]

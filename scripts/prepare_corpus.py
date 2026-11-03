#!/usr/bin/env python3
"""Prepare an instruction-following corpus from explicitly approved local roots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from brain.corpus import build_instruction_corpus
from syntra_ai.safe_paths import PROJECT_ROOT, resolve_corpus_root


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Collect approved local source and create a grounded SYNTRA instruction JSONL corpus."
    )
    command.add_argument("--root", action="append", required=True,
                         help="Absolute directory inside the repository or SYNTRA_ALLOWED_CORPUS_ROOTS. Repeat to add roots.")
    command.add_argument("--output", default=str(PROJECT_ROOT / "data" / "corpus" / "phase4-instructions.jsonl"),
                         help="Output JSONL path inside the repository or an approved local root.")
    command.add_argument("--manifest", default=None,
                         help="Optional manifest path; defaults to <output>.manifest.json.")
    command.add_argument("--max-files", type=int, default=10_000)
    command.add_argument("--max-file-bytes", type=int, default=512 * 1024)
    command.add_argument("--max-total-bytes", type=int, default=128 * 1024 * 1024)
    command.add_argument("--chunk-chars", type=int, default=900)
    return command


def write_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resolve the parent after creation, reject symlink redirection, and require
    # it to be within the project or an explicitly approved corpus directory.
    parent = resolve_corpus_root(destination.parent.resolve(strict=True))
    target = parent / destination.name
    if target.is_symlink():
        raise ValueError("Refusing to overwrite a symlink corpus output.")
    fd, temporary_name = tempfile.mkstemp(prefix=".syntra-corpus-", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main() -> int:
    args = parser().parse_args()
    roots = [resolve_corpus_root(root) for root in args.root]
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve(strict=False)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else Path(str(output) + ".manifest.json")
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_path = manifest_path.resolve(strict=False)

    corpus = build_instruction_corpus(
        roots,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        chunk_chars=args.chunk_chars,
    )
    jsonl = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for record in corpus.records)
    manifest = {
        "schema_version": 1,
        "format": "syntra-instruction-jsonl-v1",
        "examples": len(corpus.records),
        "source_files": len(corpus.files),
        "source_bytes": corpus.source_bytes,
        "skipped_files": corpus.skipped_files,
        "source_sha256": corpus.sha256,
        "roots": [str(root) for root in roots],
        "files": corpus.files,
        "normalization": "UTF-8, line-ending normalization, Markdown boilerplate filtering, deterministic source-derived prompts",
    }
    write_atomic(output, jsonl)
    write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "prepared",
        "output": str(output),
        "manifest": str(manifest_path),
        "examples": len(corpus.records),
        "source_files": len(corpus.files),
        "source_bytes": corpus.source_bytes,
        "skipped_files": corpus.skipped_files,
        "source_sha256": corpus.sha256,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

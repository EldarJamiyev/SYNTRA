#!/usr/bin/env python
"""Compare two content-addressed raw native-model benchmark reports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPOSITORY_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from benchmark_engine import runner  # noqa: E402

MAX_INPUT_BYTES = 10 * 1024 * 1024


def read_report(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    descriptor = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size < 1 or before.st_size > MAX_INPUT_BYTES:
            raise runner.BenchmarkError("Raw benchmark report must be a bounded regular non-symlink file.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            opened = os.fstat(handle.fileno())
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            if not stat.S_ISREG(opened.st_mode) or identity != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns
            ):
                raise runner.BenchmarkError("Raw benchmark report changed while it was opened.")
            raw = handle.read(MAX_INPUT_BYTES + 1)
            after = os.fstat(handle.fileno())
            current = path.lstat()
            current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            if (
                len(raw) > MAX_INPUT_BYTES
                or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or identity != current_identity
            ):
                raise runner.BenchmarkError("Raw benchmark report changed while it was being read.")
        try:
            parsed = runner._strict_json_loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise runner.BenchmarkError("Raw benchmark report is invalid JSON: " + str(exc)) from None
        return runner.validate_raw_benchmark_report(parsed)
    except runner.BenchmarkError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise runner.BenchmarkError("Cannot safely read raw benchmark report " + str(path) + ": " + str(exc)) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_comparison(path_value: str, comparison: dict[str, Any]) -> Path:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute():
        requested = REPOSITORY_ROOT / requested
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    try:
        parent.relative_to(REPOSITORY_ROOT.resolve(strict=True))
    except ValueError:
        raise runner.BenchmarkError("Comparison output must remain inside the SYNTRA repository.") from None
    target = parent / requested.name
    if not requested.name or requested.name in {".", ".."}:
        raise runner.BenchmarkError("Comparison output must name a file.")
    if target.is_symlink() or target.exists():
        raise runner.BenchmarkError("Comparison output already exists or is a symbolic link; choose a new path.")
    encoded = (json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_INPUT_BYTES:
        raise runner.BenchmarkError("Comparison output exceeds the 10 MiB limit.")
    temporary = parent / ("." + target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard-link creates the final name atomically and fails
        # when another process already owns it; unlike replace(), it cannot clobber.
        os.link(temporary, target)
        return target
    except FileExistsError:
        raise runner.BenchmarkError("Comparison output was created concurrently; choose a new path.") from None
    except OSError as exc:
        raise runner.BenchmarkError("Could not create comparison output safely: " + str(exc)) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two raw SYNTRA PyTorch benchmark reports without hiding protocol, denominator, or hardware differences."
    )
    parser.add_argument("--baseline", required=True, help="Path to the baseline raw-model report JSON.")
    parser.add_argument("--candidate", required=True, help="Path to the candidate raw-model report JSON.")
    parser.add_argument("--output", help="Optional comparison JSON path inside the SYNTRA repository.")
    return parser.parse_args(argv)


def format_metric(metric: dict[str, Any]) -> str:
    before, after = metric["baseline"], metric["candidate"]
    display = lambda value: "unavailable" if value is None else str(round(value, 4))
    values = display(before) + " → " + display(after)
    delta = metric["relative_change_percent"]
    change = "n/a" if delta is None else (("+" if delta > 0 else "") + str(delta) + "%")
    return metric["key"] + ": " + values + " | " + change + " | " + metric["direction"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = read_report(args.baseline)
        candidate = read_report(args.candidate)
        comparison = runner.compare_raw_benchmark_reports(baseline, candidate)
        print("SYNTRA raw-model benchmark comparison")
        print("Baseline: " + comparison["baseline"]["run_id"] + " · " + comparison["baseline"]["checkpoint_sha256"])
        print("Candidate: " + comparison["candidate"]["run_id"] + " · " + comparison["candidate"]["checkpoint_sha256"])
        print("Evaluation protocol: " + comparison["evaluation_protocol_sha256"])
        print("Quality outcome: " + comparison["quality_status"])
        print("Hardware-controlled performance: " + ("yes" if comparison["environment"]["hardware_comparable"] else "no"))
        print("")
        for metric in comparison["metrics"]:
            print(format_metric(metric))
        if comparison["warnings"]:
            print("")
            print("Warnings:")
            for warning in comparison["warnings"]:
                print("- " + warning)
        if args.output:
            output = write_comparison(args.output, comparison)
            print("")
            print("Comparison JSON: " + str(output))
        return 0
    except (runner.BenchmarkError, OSError, RuntimeError, ValueError) as exc:
        print("SYNTRA raw-model benchmark comparison: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

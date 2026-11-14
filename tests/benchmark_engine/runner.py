"""Held-out raw-native-model benchmark. Hidden evaluation code is not model-visible."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
ENGINE_ROOT = Path(__file__).resolve().parent
SUITE_PATH = ENGINE_ROOT / "suite.v1.json"
try:
    from .evaluation_provenance import evaluation_protocol_identity
except ImportError:
    from evaluation_provenance import evaluation_protocol_identity
TYPESCRIPT_ROOT = REPOSITORY_ROOT / "node_modules" / "typescript"
TYPESCRIPT_TEST_RUNNER = ENGINE_ROOT / "typescript_test_runner.mjs"

try:
    from .environments import SUPPORTED_LANGUAGES, expected_test_filename, resolve_environment
except ImportError:
    from environments import SUPPORTED_LANGUAGES, expected_test_filename, resolve_environment
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "benchmarks" / "latest_run.json"
MAX_SOURCE_FILES = 48
MAX_SOURCE_FILE_BYTES = 128 * 1024
MAX_PROMPT_BYTES = 192 * 1024
MAX_PATCH_BYTES = 256 * 1024
MAX_COMMAND_OUTPUT_BYTES = 256 * 1024
MAX_TRAINING_HISTORY_POINTS = 512
LOW_TRAINING_LOSS_THRESHOLD = 1.0
COMMAND_TIMEOUT_SECONDS = 30


class BenchmarkError(RuntimeError):
    """A fail-closed benchmark configuration, evidence, or execution error."""


class ContextWindowExceeded(BenchmarkError):
    """A measured prompt exceeded the checkpoint context before generation."""

    def __init__(self, prompt_tokens: int, context_window_tokens: int, tokens_over_context: int):
        self.prompt_tokens = prompt_tokens
        self.context_window_tokens = context_window_tokens
        self.tokens_over_context = tokens_over_context
        self.context_utilization = round(prompt_tokens / context_window_tokens, 4)
        super().__init__(
            "Prompt exceeds checkpoint context by "
            + str(tokens_over_context) + " tokens ("
            + str(prompt_tokens) + "/" + str(context_window_tokens)
            + "; utilization=" + str(self.context_utilization) + ")."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_within(parent: Path, target: Path) -> bool:
    try:
        target.relative_to(parent)
        return True
    except ValueError:
        return False


def _strict_json_loads(value: str) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = item
        return result
    def reject_constant(item):
        raise ValueError("Invalid JSON number: " + item)
    return json.loads(value, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def load_suite(path: Path = SUITE_PATH) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not _is_within(ENGINE_ROOT.resolve(), resolved) or not resolved.is_file() or resolved.stat().st_size > 512 * 1024:
        raise BenchmarkError("Benchmark suite must be a bounded regular file inside tests/benchmark_engine.")
    suite = _strict_json_loads(resolved.read_text(encoding="utf-8"))
    validate_suite(suite)
    return suite


def validate_suite(suite: Any) -> None:
    if not isinstance(suite, dict) or suite.get("schema_version") != 1:
        raise BenchmarkError("Benchmark suite must be a schema_version 1 object.")
    if not isinstance(suite.get("version"), str) or not re.fullmatch(r"\d+\.\d+\.\d+", suite["version"]):
        raise BenchmarkError("Benchmark suite version must use semantic version format.")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 100:
        raise BenchmarkError("Benchmark suite must define between 1 and 100 tasks.")
    if suite.get("task_count") != len(tasks):
        raise BenchmarkError("Benchmark suite task_count must match the actual task list.")
    ids = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise BenchmarkError("Each benchmark task must be an object.")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", task_id) or task_id in ids:
            raise BenchmarkError("Benchmark task identifiers must be unique kebab-case strings.")
        ids.add(task_id)
        if not isinstance(task.get("goal"), str) or not task["goal"].strip() or len(task["goal"]) > 5000:
            raise BenchmarkError("Task " + task_id + " has an invalid implementation goal.")
        for key, prefix in (("workspace", "fixtures/"), ("hidden_evaluation", "hidden/")):
            value = task.get(key)
            parts = PurePosixPath(value).parts if isinstance(value, str) else ()
            if not value or not value.startswith(prefix) or "\\" in value or any(part in {"", ".", ".."} for part in parts):
                raise BenchmarkError("Task " + task_id + " has an unsafe " + key + " path.")
        language = task.get("language")
        if language not in SUPPORTED_LANGUAGES:
            raise BenchmarkError("Task " + task_id + " has an unsupported benchmark language.")
        for key, phase in (("target_test", "target"), ("regression_test", "regression")):
            value = task.get(key)
            if not isinstance(value, str) or value != expected_test_filename(language, phase):
                raise BenchmarkError("Task " + task_id + " has an invalid hidden test filename for " + language + ".")
        expected = task.get("expected_changed_paths")
        if not isinstance(expected, list) or not 1 <= len(expected) <= 6:
            raise BenchmarkError("Task " + task_id + " must declare expected changed paths.")
        for value in expected:
            if not isinstance(value, str) or not value or "\\" in value or Path(value).is_absolute():
                raise BenchmarkError("Task " + task_id + " contains an unsafe changed path.")
            pure = PurePosixPath(value)
            if (
                pure.is_absolute() or pure.as_posix() != value
                or re.match(r"^[A-Za-z]:", value)
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                raise BenchmarkError("Task " + task_id + " contains an unsafe changed path.")
        if len(set(expected)) != len(expected):
            raise BenchmarkError("Task " + task_id + " contains duplicate changed paths.")


def load_checkpoint_evidence(checkpoint: Path) -> dict[str, Any]:
    requested = checkpoint.expanduser()
    if requested.is_symlink():
        raise BenchmarkError("The selected native checkpoint must not be a symbolic link.")
    target = requested.resolve(strict=True)
    if not target.is_file():
        raise BenchmarkError("The selected native checkpoint must be a regular file.")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata_path = target.with_suffix(target.suffix + ".json")
    metadata, metadata_error = {}, None
    if metadata_path.exists():
        try:
            if metadata_path.is_symlink() or not metadata_path.is_file() or metadata_path.stat().st_size > 5 * 1024 * 1024:
                raise ValueError("sidecar must be a bounded regular file")
            metadata = _strict_json_loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("sidecar must contain a JSON object")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            metadata_error, metadata = str(exc), {}

    from syntra_ai.training_history import validate_training_history
    history_error = None
    try:
        history = validate_training_history(metadata.get("training_history"))
    except ValueError as exc:
        history, history_error = [], str(exc)

    def finite_number(key):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return normalized if math.isfinite(normalized) else None

    step = metadata.get("step")
    if type(step) is not int or step < 0:
        step = None
    instruction_dataset_sha256 = metadata.get("instruction_dataset_sha256")
    instruction_dataset_bytes = metadata.get("instruction_dataset_bytes")
    if instruction_dataset_sha256 is not None and (
        not isinstance(instruction_dataset_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", instruction_dataset_sha256) is None
    ):
        raise BenchmarkError("Checkpoint sidecar contains a malformed SFT dataset SHA-256.")
    if instruction_dataset_bytes is not None and (
        type(instruction_dataset_bytes) is not int or instruction_dataset_bytes < 0
    ):
        raise BenchmarkError("Checkpoint sidecar contains a malformed SFT dataset byte count.")
    if (instruction_dataset_sha256 is None) != (instruction_dataset_bytes is None):
        raise BenchmarkError("Checkpoint sidecar contains incomplete SFT dataset provenance.")
    return {
        "path": str(target), "sha256": digest.hexdigest(), "step": step,
        "train_loss": finite_number("train_loss"), "validation_loss": finite_number("validation_loss"),
        "best_validation_loss": finite_number("best_validation_loss"),
        "training_mode": metadata.get("training_mode"),
        "instruction_dataset_sha256": metadata.get("instruction_dataset_sha256"),
        "instruction_dataset_bytes": metadata.get("instruction_dataset_bytes"),
        "history": history,
        "history_available": bool(history), "history_error": history_error, "metadata_error": metadata_error,
    }
def _copy_fixture_workspace(task: dict[str, Any], parent: Path) -> Path:
    fixture_input = ENGINE_ROOT / task["workspace"]
    if fixture_input.is_symlink():
        raise BenchmarkError("Fixture workspace root must not be a symbolic link.")
    fixture_root = fixture_input.resolve(strict=True)
    fixture_base = (ENGINE_ROOT / "fixtures").resolve(strict=True)
    if not _is_within(fixture_base, fixture_root) or not fixture_root.is_dir():
        raise BenchmarkError("Task fixture is missing or escapes the fixtures directory.")
    workspace = parent / "workspace"
    workspace.mkdir(mode=0o755)
    for current, directories, filenames in os.walk(fixture_root, followlinks=False):
        current_path = Path(current)
        destination_dir = workspace / current_path.relative_to(fixture_root)
        destination_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        for name in directories:
            if (current_path / name).is_symlink():
                raise BenchmarkError("Symlink directory is not allowed in benchmark fixtures.")
        for name in filenames:
            source = current_path / name
            if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_SOURCE_FILE_BYTES:
                raise BenchmarkError("Fixture contains an unsafe or oversized source file.")
            shutil.copyfile(source, destination_dir / name)
    return workspace


def collect_sources(workspace: Path) -> list[tuple[str, str]]:
    root = workspace.resolve(strict=True)
    result, total = [], 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(name for name in directories if name not in {".git", "__pycache__", ".venv", "venv", "node_modules"} and not (current_path / name).is_symlink())
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {
                ".py", ".pyi", ".toml", ".md", ".json", ".ts", ".tsx", ".mts", ".lock",
                ".js", ".mjs", ".go", ".rs", ".mod", ".sum",
            }:
                continue
            size = path.stat().st_size
            total += size
            if size > MAX_SOURCE_FILE_BYTES or len(result) >= MAX_SOURCE_FILES or total > MAX_PROMPT_BYTES:
                raise BenchmarkError("Fixture exceeds the bounded source-context budget.")
            result.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    if not result:
        raise BenchmarkError("Fixture workspace contains no supported source files.")
    return result


def build_prompt(task: dict[str, Any], source_files: list[tuple[str, str]]) -> str:
    sections = [
        "You are implementing a narrowly scoped change in an unfamiliar repository.",
        "Use only the source files shown below as evidence. Do not assume unseen tests, dependencies, or APIs.",
        "Return exactly one JSON object and no prose or Markdown fences.",
        'Required schema: {"format":"SYNTRA_AGENT_PATCH_V1","summary":"short summary","edits":[{"path":"relative/path.py","find":"exact existing source text","replace":"replacement source text","reason":"why this change is needed"}]}',
        "Treat the listed source files as one connected repository. Preserve sibling-module imports, generic contracts, concurrency invariants, and lifetimes across files.",
        "Return every required file edit in this single JSON response. Use one distinct edit per path, exact source anchors, and no partial multi-file changes.",
        "Do not edit tests, add files, or include shell commands.",
        "", "TASK:", task["goal"].strip(), "", "SOURCE FILES:",
    ]
    for relative, content in source_files:
        sections.extend([f"FILE: {relative}", "----- SOURCE BEGIN -----", content, "----- SOURCE END -----", ""])
    prompt = "\n".join(sections)
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise BenchmarkError("Rendered benchmark prompt exceeds the maximum prompt size.")
    return prompt


def parse_patch_output(raw_output: str) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip() or len(raw_output.encode("utf-8")) > MAX_PATCH_BYTES:
        raise BenchmarkError("Native model returned an empty or oversized patch response.")
    try:
        patch = _strict_json_loads(raw_output.strip())
    except (ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkError("Model output is not strict JSON or contains trailing output: " + str(exc)) from None
    if not isinstance(patch, dict) or set(patch) != {"format", "summary", "edits"} or patch.get("format") != "SYNTRA_AGENT_PATCH_V1":
        raise BenchmarkError("Patch must contain exactly the SYNTRA_AGENT_PATCH_V1 format, summary, and edits.")
    if not isinstance(patch["summary"], str) or not patch["summary"].strip() or len(patch["summary"]) > 400:
        raise BenchmarkError("Patch summary is invalid.")
    edits = patch["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 6:
        raise BenchmarkError("Patch must contain between 1 and 6 distinct file edits.")
    seen_paths = set()
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or set(edit) != {"path", "find", "replace", "reason"}:
            raise BenchmarkError("Patch edit " + str(index) + " has an invalid schema.")
        if any(not isinstance(edit.get(key), str) for key in ("path", "find", "replace", "reason")):
            raise BenchmarkError("Patch edit fields must be strings.")
        relative = edit["path"]
        pure = PurePosixPath(relative)
        if (
            not relative or pure.is_absolute() or pure.as_posix() != relative
            or "\\" in relative or re.match(r"^[A-Za-z]:", relative)
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise BenchmarkError("Patch edit paths must be safe relative repository paths.")
        if relative in seen_paths:
            raise BenchmarkError("A multi-file patch may edit each path only once.")
        seen_paths.add(relative)
        if not edit["find"] or len(edit["find"]) > 12_000 or len(edit["replace"]) > 12_000:
            raise BenchmarkError("Exact patch anchors and replacements are limited to 12,000 characters.")
        if len(edit["replace"].encode("utf-8")) > 40 * 1024 or not edit["reason"].strip() or len(edit["reason"]) > 500:
            raise BenchmarkError("Patch replacement or reason exceeds SYNTRA's patch limits.")
    return patch


def apply_patch(workspace: Path, patch: dict[str, Any], expected_paths: list[str]) -> list[str]:
    """Delegate validation and candidate writing to SYNTRA's existing patch implementation."""
    bridge = ENGINE_ROOT / "apply_patch_bridge.mjs"
    payload_file = workspace.parent / ("patch-" + uuid.uuid4().hex + ".json")
    payload = {"patch": patch, "expected_paths": expected_paths}
    try:
        with payload_file.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        outcome = run_command(
            ["node", str(bridge), str(workspace), str(payload_file)],
            cwd=REPOSITORY_ROOT, env=dict(os.environ), timeout_seconds=20,
        )
        if outcome["returncode"] != 0:
            raise BenchmarkError("SYNTRA patch validator rejected the candidate: " + outcome["stderr"][-1500:])
        response = _strict_json_loads(outcome["stdout"])
        changed = response.get("changed_paths") if isinstance(response, dict) else None
        if not isinstance(changed, list) or sorted(changed) != sorted(expected_paths):
            raise BenchmarkError("Existing patch validator returned an unexpected changed-path set.")
        return sorted(changed)
    finally:
        if payload_file.exists():
            payload_file.unlink()


def _runtime_environment(language: str, workspace: Path, *, in_container: bool) -> dict[str, str]:
    if in_container:
        env = {"HOME": "/tmp", "TMPDIR": "/build"}
        if language == "python":
            env.update({"PYTHONPATH": "/workspace", "PYTHONDONTWRITEBYTECODE": "1", "SYNTRA_BENCHMARK_WORKSPACE": "/workspace"})
        elif language == "javascript":
            env["SYNTRA_BENCHMARK_WORKSPACE"] = "/workspace"
        elif language == "typescript":
            env["SYNTRA_TYPESCRIPT_ROOT"] = "/opt/typescript"
        elif language == "go":
            env.update({
                "GOTOOLCHAIN": "local", "GOPROXY": "off", "GOSUMDB": "off",
                "GOFLAGS": "-mod=readonly", "GOCACHE": "/build/go-cache",
                "GOTMPDIR": "/build", "GOMODCACHE": "/build/go-mod-cache",
            })
        elif language == "rust":
            env.update({
                "CARGO_HOME": "/tmp/cargo-home", "CARGO_TARGET_DIR": "/build/cargo-target",
                "CARGO_NET_OFFLINE": "true", "CARGO_INCREMENTAL": "0",
            })
        return env

    env = dict(os.environ)
    runtime_root = workspace.parent / (".syntra-runtime-" + language)
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    env["HOME"] = str(runtime_root)
    if language == "python":
        env.update({
            "SYNTRA_BENCHMARK_WORKSPACE": str(workspace.resolve()),
            "PYTHONPATH": str(workspace.resolve()),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
    elif language == "javascript":
        env["SYNTRA_BENCHMARK_WORKSPACE"] = str(workspace.resolve())
    elif language == "typescript":
        if not TYPESCRIPT_ROOT.is_dir():
            raise BenchmarkError("The locked local TypeScript dependency is missing; install the repository's npm dependencies before evaluating TypeScript tasks.")
        env["SYNTRA_TYPESCRIPT_ROOT"] = str(TYPESCRIPT_ROOT.resolve())
    elif language == "go":
        for name in ("go-cache", "go-tmp", "go-mod-cache"):
            (runtime_root / name).mkdir(mode=0o700, exist_ok=True)
        env.update({
            "GOTOOLCHAIN": "local", "GOPROXY": "off", "GOSUMDB": "off",
            "GOFLAGS": "-mod=readonly", "GOCACHE": str(runtime_root / "go-cache"),
            "GOTMPDIR": str(runtime_root), "GOMODCACHE": str(runtime_root / "go-mod-cache"),
        })
    elif language == "rust":
        env.setdefault("RUSTUP_HOME", os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup")))
        (runtime_root / "cargo-home").mkdir(mode=0o700, exist_ok=True)
        (runtime_root / "cargo-target").mkdir(mode=0o700, exist_ok=True)
        env.update({
            "CARGO_HOME": str(runtime_root / "cargo-home"),
            "CARGO_TARGET_DIR": str(runtime_root / "cargo-target"),
            "CARGO_NET_OFFLINE": "true", "CARGO_INCREMENTAL": "0",
        })
    return env


def run_environment_command(
    *,
    language: str,
    isolation: dict[str, Any],
    workspace: Path,
    argv: list[str],
    run_id: str,
    phase: str,
    hidden_dir: Path | None = None,
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a fixed language command in the selected host or networkless Docker boundary."""
    if language not in SUPPORTED_LANGUAGES:
        raise BenchmarkError("Unsupported benchmark language: " + language)
    if isolation["mode"] == "host-opt-in":
        env = _runtime_environment(language, workspace, in_container=False)
        return run_command(argv, cwd=workspace, env=env, timeout_seconds=timeout_seconds)

    docker = shutil.which("docker")
    if not docker:
        raise BenchmarkError("Docker disappeared after environment preflight.")
    image_id = isolation.get("image_id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise BenchmarkError("Docker execution requires the immutable image ID captured during local preflight.")
    if language == "typescript" and (not TYPESCRIPT_ROOT.is_dir() or not TYPESCRIPT_TEST_RUNNER.is_file()):
        raise BenchmarkError("The local TypeScript compiler and harness must be present before offline evaluation.")
    name = "syntra-bench-" + run_id[-8:] + "-" + language + "-" + phase + "-" + uuid.uuid4().hex[:6]
    command = [
        docker, "run", "--rm", "--pull=never", "--name", name,
        "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=256m,uid=65534,gid=65534,mode=700",
        "--tmpfs", "/build:rw,nosuid,nodev,size=512m,uid=65534,gid=65534,mode=700",
        "--pids-limit", "96", "--memory", "768m", "--cpus", "1",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--mount", f"type=bind,src={workspace.resolve()},dst=/workspace,readonly",
    ]
    if hidden_dir is not None:
        if language == "rust":
            command.extend(["--mount", f"type=bind,src={hidden_dir.resolve()},dst=/workspace/tests,readonly"])
        else:
            command.extend(["--mount", f"type=bind,src={hidden_dir.resolve()},dst=/evaluation,readonly"])
    if language == "typescript":
        command.extend([
            "--mount", f"type=bind,src={TYPESCRIPT_ROOT.resolve()},dst=/opt/typescript,readonly",
            "--mount", f"type=bind,src={TYPESCRIPT_TEST_RUNNER.resolve()},dst=/opt/typescript_test_runner.mjs,readonly",
        ])
    for key, value in _runtime_environment(language, workspace, in_container=True).items():
        command.extend(["--env", key + "=" + value])
    # Execute the exact object inspected during preflight, not a mutable tag.
    command.extend(["--workdir", "/workspace", image_id, *argv])
    return run_command(
        command, cwd=workspace, env=dict(os.environ), timeout_seconds=timeout_seconds,
        cleanup_command=[docker, "kill", name],
    )


def syntax_check(
    workspace: Path,
    paths: list[str],
    *,
    language: str = "python",
    isolation: dict[str, Any] | None = None,
    run_id: str = "syntax-check",
) -> dict[str, Any]:
    checked = []
    try:
        root = workspace.resolve(strict=True)
        for relative in paths:
            pure = PurePosixPath(relative)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or "\\" in relative:
                raise BenchmarkError("Invalid syntax-check path.")
            target = (root / Path(*pure.parts)).resolve(strict=True)
            if not _is_within(root, target):
                raise BenchmarkError("Syntax-check path escapes the task workspace.")
            checked.append(relative)
        if language == "python":
            for relative in checked:
                target = root / Path(*PurePosixPath(relative).parts)
                if target.suffix.lower() not in {".py", ".pyi"}:
                    raise BenchmarkError("No supported Python syntax check for " + relative)
                ast.parse(target.read_text(encoding="utf-8"), filename=relative, type_comments=True)
            return {"passed": True, "checked_paths": checked, "error": None}
        if isolation is None:
            raise BenchmarkError("A language-specific syntax check requires a selected runtime environment.")
        if language == "javascript":
            for index, relative in enumerate(checked):
                source_path = (
                    ("/workspace/" if isolation["mode"] == "docker-no-network" else str(root) + os.sep)
                    + relative.replace("/", os.sep)
                )
                outcome = run_environment_command(
                    language=language, isolation=isolation, workspace=root,
                    argv=["node", "--check", source_path], run_id=run_id,
                    phase="syntax-" + str(index), timeout_seconds=30,
                )
                if outcome["returncode"] != 0 or outcome["timed_out"] or outcome["output_limited"]:
                    diagnostic = (outcome["stderr"] + "\n" + outcome["stdout"]).strip()
                    return {"passed": False, "checked_paths": checked, "error": diagnostic[-1000:] or "Node.js syntax check failed."}
            return {"passed": True, "checked_paths": checked, "error": None}
        if language == "typescript":
            compiler = "/opt/typescript/bin/tsc" if isolation["mode"] == "docker-no-network" else str(TYPESCRIPT_ROOT / "bin" / "tsc")
            source_paths = [
                ("/workspace/" if isolation["mode"] == "docker-no-network" else str(root) + os.sep)
                + relative.replace("/", os.sep)
                for relative in checked
            ]
            command = [
                "node", compiler, "--noEmit", "--strict", "--target", "ES2022",
                "--module", "ESNext", "--moduleResolution", "Bundler", "--skipLibCheck",
                "--pretty", "false", *source_paths,
            ]
        elif language == "go":
            command = ["go", "test", "./..."]
        elif language == "rust":
            manifest = "/workspace/Cargo.toml" if isolation["mode"] == "docker-no-network" else str(root / "Cargo.toml")
            command = ["cargo", "check", "--offline", "--locked", "--manifest-path", manifest]
        else:
            raise BenchmarkError("No syntax compiler is configured for " + language + ".")
        outcome = run_environment_command(
            language=language, isolation=isolation, workspace=root, argv=command,
            run_id=run_id, phase="syntax", timeout_seconds=60,
        )
        passed = outcome["returncode"] == 0 and not outcome["timed_out"] and not outcome["output_limited"]
        diagnostic = (outcome["stderr"] + "\n" + outcome["stdout"]).strip()
        return {
            "passed": passed, "checked_paths": checked,
            "error": None if passed else diagnostic[-1000:] or "Language compiler exited unsuccessfully.",
        }
    except (BenchmarkError, SyntaxError, UnicodeError, OSError, RuntimeError, ValueError) as exc:
        return {"passed": False, "checked_paths": checked, "error": str(exc)[:1000]}


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        else:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], timeout_seconds: int = COMMAND_TIMEOUT_SECONDS, output_limit: int = MAX_COMMAND_OUTPUT_BYTES, cleanup_command: list[str] | None = None) -> dict[str, Any]:
    try:
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, close_fds=True, start_new_session=(os.name != "nt"), creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
    except OSError as exc:
        return {"returncode": None, "timed_out": False, "output_limited": False, "stdout": "", "stderr": str(exc), "elapsed_ms": 0}
    lock, limited = threading.Lock(), threading.Event()
    output_bytes = 0
    stdout_chunks, stderr_chunks = [], []

    def drain(stream, chunks):
        nonlocal output_bytes
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                with lock:
                    available = max(0, output_limit - output_bytes)
                    if available:
                        chunks.append(chunk[:available])
                        output_bytes += min(available, len(chunk))
                    if len(chunk) > available:
                        limited.set()
                if limited.is_set():
                    _terminate_process(process)
                    break
        except (OSError, ValueError):
            pass

    started = time.perf_counter()
    threads = [threading.Thread(target=drain, args=(process.stdout, stdout_chunks), daemon=True), threading.Thread(target=drain, args=(process.stderr, stderr_chunks), daemon=True)]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
    for thread in threads:
        thread.join(timeout=3)
    if process.poll() is None:
        _terminate_process(process)
        process.wait(timeout=3)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    if cleanup_command and (timed_out or limited.is_set()):
        subprocess.run(cleanup_command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    if limited.is_set():
        stderr += "\n[benchmark output limit reached; process terminated]"
    return {"returncode": process.returncode, "timed_out": timed_out, "output_limited": limited.is_set(), "stdout": stdout, "stderr": stderr, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def prepare_isolation(mode: str, image: str) -> dict[str, Any]:
    if mode == "host":
        return {"mode": "host-opt-in", "image": None, "image_id": None, "warning": "Generated code executes with the host OS user's permissions; this is not sandboxed."}
    docker = shutil.which("docker")
    if not docker:
        raise BenchmarkError("Docker is unavailable. Install Docker or explicitly pass --isolation host --allow-host-execution.")
    inspection = run_command([docker, "image", "inspect", image, "--format", "{{.Id}}"], cwd=REPOSITORY_ROOT, env=dict(os.environ), timeout_seconds=10)
    image_id = inspection["stdout"].strip()
    if inspection["returncode"] != 0 or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise BenchmarkError("Docker image " + image + " is not present locally or did not resolve to a valid immutable image ID. Images are never pulled; explicitly opt into host execution only when you accept its risk.")
    return {"mode": "docker-no-network", "image": image, "image_id": image_id, "warning": "Docker still trusts the Docker daemon and host kernel; it is not a VM boundary."}


def run_hidden_test(
    *,
    isolation: dict[str, Any],
    workspace: Path,
    hidden_dir: Path,
    test_name: str,
    run_id: str,
    phase: str,
    language: str = "python",
) -> dict[str, Any]:
    if test_name != expected_test_filename(language, phase):
        raise BenchmarkError("Hidden test filename does not match its language and evaluation phase.")
    hidden_input = hidden_dir
    if hidden_input.is_symlink():
        raise BenchmarkError("Hidden evaluation directory must not be a symbolic link.")
    hidden_root = hidden_input.resolve(strict=True)
    test_input = hidden_root / test_name
    if test_input.is_symlink():
        raise BenchmarkError("Hidden evaluation test must not be a symbolic link.")
    test_file = test_input.resolve(strict=True)
    hidden_base = (ENGINE_ROOT / "hidden").resolve(strict=True)
    if not _is_within(hidden_base, hidden_root) or not _is_within(hidden_root, test_file) or not test_file.is_file():
        raise BenchmarkError("Hidden test must be a regular file inside its declared hidden directory.")

    if language == "python":
        argv = (
            [sys.executable, "-B", str(test_file)]
            if isolation["mode"] == "host-opt-in"
            else ["python", "-B", "/evaluation/" + test_name]
        )
        return run_environment_command(
            language=language, isolation=isolation, workspace=workspace, argv=argv,
            run_id=run_id, phase=phase, hidden_dir=hidden_root, timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        )

    if language == "javascript":
        test_path = str(test_file) if isolation["mode"] == "host-opt-in" else "/evaluation/" + test_name
        return run_environment_command(
            language=language, isolation=isolation, workspace=workspace,
            argv=["node", test_path], run_id=run_id, phase=phase,
            hidden_dir=hidden_root, timeout_seconds=60,
        )

    if language == "typescript":
        if isolation["mode"] == "host-opt-in":
            scratch = workspace.parent / ("typescript-evaluation-" + uuid.uuid4().hex)
            try:
                staged_workspace = scratch / "workspace"
                staged_evaluation = scratch / "evaluation"
                build_directory = scratch / "build"
                shutil.copytree(workspace, staged_workspace)
                staged_evaluation.mkdir(parents=True, mode=0o755)
                shutil.copyfile(test_file, staged_evaluation / test_name)
                return run_environment_command(
                    language=language, isolation=isolation, workspace=staged_workspace,
                    argv=["node", str(TYPESCRIPT_TEST_RUNNER), str(staged_evaluation / test_name), str(build_directory)],
                    run_id=run_id, phase=phase, timeout_seconds=60,
                )
            finally:
                if scratch.exists():
                    shutil.rmtree(scratch, ignore_errors=True)
        return run_environment_command(
            language=language, isolation=isolation, workspace=workspace,
            argv=[
                "node", "/opt/typescript_test_runner.mjs",
                "/evaluation/" + test_name, "/tmp/syntra-typescript-" + run_id[-8:] + "-" + phase,
            ],
            run_id=run_id, phase=phase, hidden_dir=hidden_root, timeout_seconds=60,
        )

    if language == "go":
        test_path = str(test_file) if isolation["mode"] == "host-opt-in" else "/evaluation/" + test_name
        return run_environment_command(
            language=language, isolation=isolation, workspace=workspace,
            argv=["go", "run", test_path], run_id=run_id, phase=phase,
            hidden_dir=hidden_root, timeout_seconds=60,
        )

    if language == "rust":
        target_name = Path(test_name).stem
        if isolation["mode"] == "host-opt-in":
            scratch = workspace.parent / ("rust-evaluation-" + uuid.uuid4().hex)
            try:
                shutil.copytree(workspace, scratch)
                tests_dir = scratch / "tests"
                if tests_dir.exists():
                    shutil.rmtree(tests_dir)
                tests_dir.mkdir(mode=0o755)
                shutil.copyfile(test_file, tests_dir / test_name)
                return run_environment_command(
                    language=language, isolation=isolation, workspace=scratch,
                    argv=[
                        "cargo", "test", "--offline", "--locked", "--manifest-path",
                        str(scratch / "Cargo.toml"), "--test", target_name,
                    ],
                    run_id=run_id, phase=phase, timeout_seconds=90,
                )
            finally:
                if scratch.exists():
                    shutil.rmtree(scratch, ignore_errors=True)
        return run_environment_command(
            language=language, isolation=isolation, workspace=workspace,
            argv=[
                "cargo", "test", "--offline", "--locked", "--manifest-path",
                "/workspace/Cargo.toml", "--test", target_name,
            ],
            run_id=run_id, phase=phase, hidden_dir=hidden_root, timeout_seconds=90,
        )

    raise BenchmarkError("No hidden-test executor is configured for " + language + ".")


def _expected_baseline_failure(result: dict[str, Any]) -> bool:
    output = result["stdout"] + "\n" + result["stderr"]
    assertion = (
        "SYNTRA_BENCHMARK_ASSERTION_FAILED" in output
        or "AssertionError" in output
        or "FAIL: test_" in output
        or "ERROR: test_" in output
        or "FAILED (failures=" in output
        or "FAILED (errors=" in output
        or "panicked at" in output
        or re.search(r"test .* FAILED", output) is not None
    )
    return result["returncode"] not in (None, 0) and not result["timed_out"] and not result["output_limited"] and assertion


def _generate_patch(runtime: Any, prompt: str, checkpoint: Path, device: str, max_new_tokens: int) -> dict[str, Any]:
    prompt_info = runtime.tokenize_text(prompt, path=checkpoint)
    if not isinstance(prompt_info, dict):
        raise BenchmarkError("Native tokenizer returned malformed context evidence.")
    prompt_tokens = prompt_info.get("token_count")
    context_window = prompt_info.get("context_window_tokens")
    tokens_over_context = prompt_info.get("tokens_over_context")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise BenchmarkError("Native tokenizer did not return a valid prompt token count.")
    if type(context_window) is not int or context_window < 2:
        raise BenchmarkError("Native tokenizer did not return a valid context-window size.")
    if type(tokens_over_context) is not int or tokens_over_context < 0:
        raise BenchmarkError("Native tokenizer did not return valid context-overflow evidence.")
    measured_overflow = max(tokens_over_context, prompt_tokens - (context_window - 1), 0)
    if measured_overflow > 0:
        raise ContextWindowExceeded(prompt_tokens, context_window, measured_overflow)
    pieces, terminal = [], None
    started = time.perf_counter()
    for event in runtime.stream_generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_k=1, top_p=1.0, repetition_penalty=1.0, device_name=device, path=checkpoint):
        if not isinstance(event, dict):
            raise BenchmarkError("Native generation yielded an invalid event.")
        delta = event.get("delta")
        if isinstance(delta, str) and delta:
            pieces.append(delta)
        if event.get("done") is True:
            terminal = event
    elapsed = (time.perf_counter() - started) * 1000
    if terminal is None or terminal.get("is_model_output") is not True:
        raise BenchmarkError("Native model stream ended without a verified terminal event.")
    output = "".join(pieces)
    if terminal.get("text") != output:
        raise BenchmarkError("Native model terminal text does not match streamed deltas.")
    return {
        "text": output, "prompt_tokens": prompt_tokens,
        "context_window_tokens": context_window,
        "context_utilization": round(prompt_tokens / context_window, 4),
        "generated_tokens": int(terminal.get("generated_tokens", 0)),
        "generation_ms": round(elapsed, 2),
        "prompt_truncated": bool(terminal.get("prompt_truncated", False)),
    }


def _finish_result(result: dict[str, Any], started: float) -> dict[str, Any]:
    result["execution_speed"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result
def run_task(task: dict[str, Any], *, runtime: Any, checkpoint: Path, device: str, max_new_tokens: int, isolation: dict[str, Any], run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = {
        "id": task["id"], "language": task["language"], "category": task.get("category", "general"), "title": task.get("title", task["id"]),
        "environment_image": isolation.get("image"), "status": "failed", "baseline": {}, "baseline_validated": False,
        "patch_correctness": False, "task_completion": False,
        "regression_prevention": None, "changed_paths": [], "error": None,
        "execution_speed": {"total_ms": 0, "generation_ms": None, "patch_apply_ms": None, "syntax_check_ms": None, "target_test_ms": None, "regression_test_ms": None, "prompt_tokens": 0, "context_window_tokens": None, "context_utilization": None, "context_tokens_over_limit": 0, "context_status": "not-measured", "generated_tokens": 0, "output_tokens_per_second": None},
    }
    with tempfile.TemporaryDirectory(prefix="syntra-benchmark-" + task["id"] + "-") as temp:
        workspace = _copy_fixture_workspace(task, Path(temp))
        hidden_input = ENGINE_ROOT / task["hidden_evaluation"]
        if hidden_input.is_symlink():
            raise BenchmarkError("Hidden evaluation directory must not be a symbolic link.")
        hidden_dir = hidden_input.resolve(strict=True)
        hidden_base = (ENGINE_ROOT / "hidden").resolve(strict=True)
        if not _is_within(hidden_base, hidden_dir) or not hidden_dir.is_dir():
            raise BenchmarkError("Task hidden evaluation directory is invalid.")
        baseline_target = run_hidden_test(isolation=isolation, workspace=workspace, hidden_dir=hidden_dir, test_name=task["target_test"], run_id=run_id, phase="baseline-target", language=task["language"])
        baseline_regression = run_hidden_test(isolation=isolation, workspace=workspace, hidden_dir=hidden_dir, test_name=task["regression_test"], run_id=run_id, phase="baseline-regression", language=task["language"])
        result["baseline"] = {
            "target_expected_to_fail": _expected_baseline_failure(baseline_target),
            "target_exit_code": baseline_target["returncode"], "target_elapsed_ms": baseline_target["elapsed_ms"],
            "regression_passed": baseline_regression["returncode"] == 0 and not baseline_regression["timed_out"] and not baseline_regression["output_limited"],
            "regression_exit_code": baseline_regression["returncode"], "regression_elapsed_ms": baseline_regression["elapsed_ms"],
        }
        if not result["baseline"]["target_expected_to_fail"]:
            result["status"], result["error"] = "invalid-fixture", "Target test did not fail with an assertion before patching."
            return _finish_result(result, started)
        if not result["baseline"]["regression_passed"]:
            result["status"], result["error"] = "invalid-fixture", "Regression test failed before patching."
            return _finish_result(result, started)
        result["baseline_validated"] = True

        try:
            prompt = build_prompt(task, collect_sources(workspace))
            generation = _generate_patch(runtime, prompt, checkpoint, device, max_new_tokens)
        except ContextWindowExceeded as exc:
            speed = result["execution_speed"]
            speed["prompt_tokens"] = exc.prompt_tokens
            speed["context_window_tokens"] = exc.context_window_tokens
            speed["context_utilization"] = exc.context_utilization
            speed["context_tokens_over_limit"] = exc.tokens_over_context
            speed["context_status"] = "exceeded"
            result["status"], result["error"] = "context-exceeded", str(exc)[:1500]
            return _finish_result(result, started)
        except (BenchmarkError, OSError, RuntimeError, ValueError) as exc:
            result["status"], result["error"] = "generation-error", str(exc)[:1500]
            return _finish_result(result, started)
        if generation["prompt_truncated"]:
            result["status"], result["error"] = "generation-error", "Native generation reported prompt truncation."
            return _finish_result(result, started)
        result["execution_speed"]["generation_ms"] = generation["generation_ms"]
        result["execution_speed"]["prompt_tokens"] = generation["prompt_tokens"]
        result["execution_speed"]["context_window_tokens"] = generation["context_window_tokens"]
        result["execution_speed"]["context_utilization"] = generation["context_utilization"]
        result["execution_speed"]["context_status"] = "within-window"
        result["execution_speed"]["context_tokens_over_limit"] = 0
        result["execution_speed"]["generated_tokens"] = generation["generated_tokens"]
        seconds = generation["generation_ms"] / 1000
        result["execution_speed"]["output_tokens_per_second"] = round(generation["generated_tokens"] / seconds, 2) if seconds > 0 else None
        try:
            patch = parse_patch_output(generation["text"])
        except BenchmarkError as exc:
            result["status"], result["error"] = "invalid-patch", str(exc)
            return _finish_result(result, started)

        patch_started = time.perf_counter()
        try:
            changed = apply_patch(workspace, patch, task["expected_changed_paths"])
        except (BenchmarkError, OSError, UnicodeError, RuntimeError, ValueError) as exc:
            result["status"], result["error"] = "patch-rejected", str(exc)[:1500]
            result["execution_speed"]["patch_apply_ms"] = round((time.perf_counter() - patch_started) * 1000, 2)
            return _finish_result(result, started)
        result["execution_speed"]["patch_apply_ms"] = round((time.perf_counter() - patch_started) * 1000, 2)
        result["changed_paths"] = changed
        syntax_started = time.perf_counter()
        syntax = syntax_check(workspace, changed, language=task["language"], isolation=isolation, run_id=run_id)
        result["execution_speed"]["syntax_check_ms"] = round((time.perf_counter() - syntax_started) * 1000, 2)
        if not syntax["passed"]:
            result["status"], result["error"] = "syntax-failed", syntax["error"]
            return _finish_result(result, started)
        result["patch_correctness"] = True

        try:
            target_test = run_hidden_test(isolation=isolation, workspace=workspace, hidden_dir=hidden_dir, test_name=task["target_test"], run_id=run_id, phase="target", language=task["language"])
            regression_test = run_hidden_test(isolation=isolation, workspace=workspace, hidden_dir=hidden_dir, test_name=task["regression_test"], run_id=run_id, phase="regression", language=task["language"])
        except (BenchmarkError, OSError, RuntimeError, ValueError) as exc:
            result["status"], result["error"] = "test-execution-error", str(exc)[:1500]
            return _finish_result(result, started)
        result["execution_speed"]["target_test_ms"] = target_test["elapsed_ms"]
        target_passed = target_test["returncode"] == 0 and not target_test["timed_out"] and not target_test["output_limited"]
        result["task_completion"] = target_passed
        result["execution_speed"]["regression_test_ms"] = regression_test["elapsed_ms"]
        regression_passed = regression_test["returncode"] == 0 and not regression_test["timed_out"] and not regression_test["output_limited"]
        result["regression_prevention"] = regression_passed
        result["test_evidence"] = {
            "target_exit_code": target_test["returncode"], "target_timed_out": target_test["timed_out"],
            "target_output_limited": target_test["output_limited"], "target_stdout": target_test["stdout"][-3000:],
            "target_stderr": target_test["stderr"][-3000:], "regression_exit_code": regression_test["returncode"],
            "regression_timed_out": regression_test["timed_out"], "regression_output_limited": regression_test["output_limited"],
            "regression_stdout": regression_test["stdout"][-3000:], "regression_stderr": regression_test["stderr"][-3000:],
        }
        if target_passed and regression_passed:
            result["status"] = "completed"
        elif not target_passed and not regression_passed:
            result["status"] = "target-and-regression-failed"
        elif not target_passed:
            result["status"] = "target-failed"
        else:
            result["status"] = "regression-failed"
    return _finish_result(result, started)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    valid = [item for item in results if item.get("baseline_validated") is True]
    valid_count = len(valid)
    completed = sum(item["task_completion"] is True for item in valid)
    generation_attempted = [
        item for item in valid
        if item.get("execution_speed", {}).get("context_status") == "within-window"
    ]
    generation_completed = sum(item["task_completion"] is True for item in generation_attempted)
    generation_completion_rate = (
        round(generation_completed / len(generation_attempted), 4)
        if generation_attempted else None
    )
    full = sum(item["task_completion"] is True and item["regression_prevention"] is True for item in valid)
    patch_count = sum(item["patch_correctness"] is True for item in valid)
    measured = [item for item in valid if item["regression_prevention"] is not None]
    regression_count = sum(item["regression_prevention"] is True for item in measured)
    times = [float(item["execution_speed"]["total_ms"]) for item in valid]
    tokens = sum(int(item["execution_speed"]["generated_tokens"] or 0) for item in valid)
    generation_seconds = sum(float(item["execution_speed"]["generation_ms"]) / 1000 for item in valid if item["execution_speed"]["generation_ms"] is not None)
    language_breakdown = {}
    context_measurements = []
    context_exceeded_count = 0
    for item in results:
        language = item.get("language", "unknown")
        entry = language_breakdown.setdefault(language, {
            "task_count": 0,
            "baseline_valid_task_count": 0,
            "task_completion_count": 0,
            "patch_correct_count": 0,
            "regression_prevention_measured_count": 0,
            "regression_prevented_count": 0,
            "fully_verified_task_count": 0,
            "task_completion_rate": None,
            "patch_correctness_rate": None,
            "regression_prevention_rate": None,
            "fully_verified_task_rate": None,
            "context_measurement_count": 0,
            "context_exceeded_task_count": 0,
            "max_context_utilization": None,
        })
        entry["task_count"] += 1
        speed = item.get("execution_speed", {})
        prompt_tokens = speed.get("prompt_tokens")
        context_window = speed.get("context_window_tokens")
        utilization = speed.get("context_utilization")
        if (
            type(prompt_tokens) is int and prompt_tokens >= 0
            and type(context_window) is int and context_window >= 2
            and isinstance(utilization, (int, float)) and not isinstance(utilization, bool)
            and math.isfinite(float(utilization))
        ):
            measurement = {
                "language": language,
                "prompt_tokens": prompt_tokens,
                "context_window_tokens": context_window,
                "utilization": float(utilization),
                "status": speed.get("context_status", "measured"),
            }
            context_measurements.append(measurement)
            entry["context_measurement_count"] += 1
            previous_max = entry["max_context_utilization"]
            entry["max_context_utilization"] = (
                measurement["utilization"] if previous_max is None
                else max(previous_max, measurement["utilization"])
            )
        if speed.get("context_status") == "exceeded":
            entry["context_exceeded_task_count"] += 1
            context_exceeded_count += 1
        if item.get("baseline_validated") is True:
            entry["baseline_valid_task_count"] += 1
            if item.get("task_completion") is True:
                entry["task_completion_count"] += 1
            if item.get("patch_correctness") is True:
                entry["patch_correct_count"] += 1
            regression_result = item.get("regression_prevention")
            if regression_result is not None:
                entry["regression_prevention_measured_count"] += 1
                if regression_result is True:
                    entry["regression_prevented_count"] += 1
            if item.get("task_completion") is True and regression_result is True:
                entry["fully_verified_task_count"] += 1
    for entry in language_breakdown.values():
        denominator = entry["baseline_valid_task_count"]
        measured_regressions = entry["regression_prevention_measured_count"]
        entry["task_completion_rate"] = (
            round(entry["task_completion_count"] / denominator, 4) if denominator else None
        )
        entry["patch_correctness_rate"] = (
            round(entry["patch_correct_count"] / denominator, 4) if denominator else None
        )
        entry["fully_verified_task_rate"] = (
            round(entry["fully_verified_task_count"] / denominator, 4) if denominator else None
        )
        entry["regression_prevention_rate"] = (
            round(entry["regression_prevented_count"] / measured_regressions, 4)
            if measured_regressions else None
        )
    context_utilizations = [item["utilization"] for item in context_measurements]
    context_evidence = {
        "measured_task_count": len(context_measurements),
        "context_exceeded_task_count": context_exceeded_count,
        "near_limit_task_count": sum(value >= 0.90 for value in context_utilizations),
        "mean_utilization": round(sum(context_utilizations) / len(context_utilizations), 4) if context_utilizations else None,
        "maximum_utilization": round(max(context_utilizations), 4) if context_utilizations else None,
        "measurements": context_measurements,
    }
    return {
        "task_count": count,
        "language_breakdown": language_breakdown,
        "context_evidence": context_evidence,
        "baseline_valid_task_count": valid_count,
        "invalid_fixture_count": count - valid_count,
        "completed_task_count": completed,
        "task_completion_rate": round(completed / valid_count, 4) if valid_count else None,
        "generation_attempted_task_count": len(generation_attempted),
        "generation_attempted_completed_task_count": generation_completed,
        "generation_attempted_task_completion_rate": generation_completion_rate,
        "fully_verified_task_count": full,
        "patch_correct_count": patch_count,
        "patch_correctness_rate": round(patch_count / valid_count, 4) if valid_count else None,
        "regression_measured_count": len(measured),
        "regression_prevented_count": regression_count,
        "regression_prevention_rate": round(regression_count / len(measured), 4) if measured else None,
        "generated_tokens": tokens,
        "aggregate_generation_tokens_per_second": round(tokens / generation_seconds, 2) if generation_seconds > 0 else None,
        "execution_time_ms": {"mean": round(sum(times) / len(times), 2) if times else None, "median": round(sorted(times)[len(times) // 2], 2) if times else None, "max": round(max(times), 2) if times else None},
    }


def language_summary_lines(language_breakdown: dict[str, dict[str, Any]]) -> list[str]:
    """Format the fully verified per-language counts emitted by summarize()."""
    return [
        "Language " + language + ": "
        + str(counts["fully_verified_task_count"])
        + "/" + str(counts["baseline_valid_task_count"])
        + " fully verified; " + str(counts["task_count"]) + " selected"
        for language, counts in sorted(language_breakdown.items())
    ]


def assess_loss_intelligence_divergence(summary: dict[str, Any], checkpoint: dict[str, Any], low_loss_threshold: float = LOW_TRAINING_LOSS_THRESHOLD) -> dict[str, Any]:
    train_loss = checkpoint.get("train_loss")
    baseline_valid = summary.get("baseline_valid_task_count", 0)
    generation_attempted = summary.get("generation_attempted_task_count", 0)
    enough = isinstance(generation_attempted, int) and generation_attempted >= 3
    low = isinstance(train_loss, (int, float)) and not isinstance(train_loss, bool) and math.isfinite(float(train_loss)) and train_loss <= low_loss_threshold
    completion_rate = summary.get("generation_attempted_task_completion_rate")
    detected = bool(enough and low and completion_rate == 0.0)
    return {
        "flag": "OVERFITTING_LOSS_INTELLIGENCE_DIVERGENCE" if detected else None,
        "detected": detected, "low_training_loss_threshold": low_loss_threshold,
        "observed_training_loss": train_loss, "baseline_valid_task_count": baseline_valid,
        "generation_attempted_task_count": generation_attempted,
        "generation_attempted_task_completion_rate": completion_rate,
        "held_out_task_completion_rate": summary.get("task_completion_rate"),
        "minimum_generation_attempts_required": 3,
        "interpretation": "Low recorded training loss coincides with zero target completions across at least three tasks whose prompts fit the checkpoint context and completed model generation. Context-rejected tasks remain in the overall end-to-end score but are excluded from this divergence heuristic. This is a warning, not proof of causal overfitting." if detected else "No divergence flag was triggered; this does not establish general coding capability.",
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute():
        requested = REPOSITORY_ROOT / requested
    resolved = requested.parent.resolve() / requested.name
    if not _is_within(REPOSITORY_ROOT.resolve(), resolved):
        raise BenchmarkError("Benchmark reports must be written inside the SYNTRA repository.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink():
        raise BenchmarkError("Benchmark report output must not be a symbolic link.")
    encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > 10 * 1024 * 1024:
        raise BenchmarkError("Benchmark report exceeds 10 MiB.")
    temporary = resolved.with_name(resolved.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if resolved.is_symlink():
            raise BenchmarkError("Benchmark report output became a symbolic link.")
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()
    return resolved


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    try:
        protocol_identity = evaluation_protocol_identity(REPOSITORY_ROOT, ENGINE_ROOT)
    except (OSError, ValueError) as exc:
        raise BenchmarkError("Cannot establish the raw-model evaluation protocol identity: " + str(exc)) from None
    suite = load_suite()
    try:
        loaded_protocol_identity = evaluation_protocol_identity(REPOSITORY_ROOT, ENGINE_ROOT)
    except (OSError, ValueError) as exc:
        raise BenchmarkError("Cannot revalidate the loaded raw-model evaluation protocol: " + str(exc)) from None
    if loaded_protocol_identity != protocol_identity:
        raise BenchmarkError("Raw-model evaluation inputs changed while the suite was being loaded.")
    requested = set(args.task or [])
    unknown = requested - {task["id"] for task in suite["tasks"]}
    if unknown:
        raise BenchmarkError("Unknown task ID(s): " + ", ".join(sorted(unknown)))
    selected = [task for task in suite["tasks"] if not requested or task["id"] in requested]
    if args.isolation == "host" and not args.allow_host_execution:
        raise BenchmarkError("Host execution requires the explicit --allow-host-execution acknowledgement.")
    from syntra_ai import runtime
    checkpoint = (Path(args.checkpoint).expanduser() if args.checkpoint else runtime.checkpoint_path()).resolve(strict=True)
    checkpoint_evidence = load_checkpoint_evidence(checkpoint)
    image_overrides = {
        "python": args.python_image,
        "javascript": args.node_image,
        "typescript": args.node_image,
        "go": args.go_image,
        "rust": args.rust_image,
    }
    selected_languages = sorted({task["language"] for task in selected})
    if "typescript" in selected_languages and (
        not TYPESCRIPT_ROOT.is_dir() or not TYPESCRIPT_TEST_RUNNER.is_file()
    ):
        raise BenchmarkError("TypeScript tasks require the repository's installed TypeScript package and benchmark harness before offline execution.")
    isolations = {}
    for language in selected_languages:
        environment = resolve_environment(language, image_overrides)
        isolations[language] = prepare_isolation(args.isolation, environment["image"])
    initial_identity = runtime._checkpoint_identity(checkpoint)
    if initial_identity is None:
        raise BenchmarkError("Native runtime could not establish checkpoint identity.")
    run_id, started_at, results = "raw-model-" + uuid.uuid4().hex, _now(), []
    for task in selected:
        print("[" + task["id"] + "] Evaluating the actual local SYNTRA checkpoint...")
        try:
            result = run_task(task, runtime=runtime, checkpoint=checkpoint, device=args.device, max_new_tokens=args.max_new_tokens, isolation=isolations[task["language"]], run_id=run_id)
        except (BenchmarkError, OSError, RuntimeError, ValueError) as exc:
            result = {
                "id": task["id"], "language": task["language"], "category": task.get("category", "general"), "title": task.get("title", task["id"]),
                "environment_image": isolations[task["language"]].get("image"), "status": "error", "baseline": {}, "patch_correctness": False, "task_completion": False,
                "regression_prevention": None, "baseline_validated": False, "changed_paths": [], "error": str(exc)[:1500],
                "execution_speed": {"total_ms": 0, "generation_ms": None, "patch_apply_ms": None, "syntax_check_ms": None, "target_test_ms": None, "regression_test_ms": None, "prompt_tokens": 0, "context_window_tokens": None, "context_utilization": None, "context_tokens_over_limit": 0, "context_status": "not-measured", "generated_tokens": 0, "output_tokens_per_second": None},
            }
        results.append(result)
        print("[" + task["id"] + "] " + result["status"] + " | patch=" + str(result["patch_correctness"]) + " | target=" + str(result["task_completion"]) + " | regression=" + str(result["regression_prevention"]))
    if runtime._checkpoint_identity(checkpoint) != initial_identity:
        raise BenchmarkError("Checkpoint identity changed during the benchmark; the run is invalid.")
    try:
        final_protocol_identity = evaluation_protocol_identity(REPOSITORY_ROOT, ENGINE_ROOT)
    except (OSError, ValueError) as exc:
        raise BenchmarkError("Cannot revalidate the raw-model evaluation protocol: " + str(exc)) from None
    if final_protocol_identity != protocol_identity:
        raise BenchmarkError("Raw-model evaluation inputs changed during the benchmark; the run is invalid.")
    summary = summarize(results)
    divergence = assess_loss_intelligence_divergence(summary, checkpoint_evidence, args.low_loss_threshold)
    report = {
        "schema_version": 1, "benchmark_track": "raw-native-model", "suite_version": suite["version"],
        "evaluation_protocol": protocol_identity,
        "run_id": run_id, "started_at": started_at, "completed_at": _now(),
        "model": {
            "engine": "syntra-native-pytorch", "checkpoint": checkpoint_evidence["path"],
            "checkpoint_sha256": checkpoint_evidence["sha256"], "checkpoint_identity": initial_identity,
            "device_requested": args.device, "trained_steps": checkpoint_evidence["step"],
            "training_mode": checkpoint_evidence["training_mode"],
            "instruction_dataset_sha256": checkpoint_evidence["instruction_dataset_sha256"],
            "instruction_dataset_bytes": checkpoint_evidence["instruction_dataset_bytes"],
        },
        "training_loss": {
            "train_loss": checkpoint_evidence["train_loss"], "validation_loss": checkpoint_evidence["validation_loss"],
            "best_validation_loss": checkpoint_evidence["best_validation_loss"],
            "history_available": checkpoint_evidence["history_available"], "history_point_count": len(checkpoint_evidence["history"]),
            "history": checkpoint_evidence["history"], "history_error": checkpoint_evidence["history_error"],
            "metadata_error": checkpoint_evidence["metadata_error"],
        },
        "loss_alignment": {
            "benchmark_checkpoint_step": checkpoint_evidence["step"], "benchmark_train_loss": checkpoint_evidence["train_loss"],
            "benchmark_validation_loss": checkpoint_evidence["validation_loss"], "task_completion_rate": summary["task_completion_rate"],
            "divergence": divergence,
        },
        "isolation": {
            "mode": args.isolation,
            "environments": isolations,
            "warning": "Docker execution is networkless and source/evaluation mounts are read-only, but the boundary still trusts the Docker daemon and host kernel. Host mode is not sandboxed.",
        },
        "summary": summary, "tasks": results,
        "measurement_notes": [
            "This track calls syntra_ai.runtime.stream_generate directly and does not invoke the repository-aware agent.",
            "The existing 11-task local/benchmark.mjs suite remains the separate end-to-end agent benchmark.",
            "Hidden target and regression tests are not copied into the model workspace or included in the model prompt.",
            "Task completion means held-out target-test success; regression prevention is measured independently.",
            "Patch correctness uses SYNTRA's existing validatePatch/writeCandidateChanges implementation and requires the exact expected multi-file changed-path set.",
            "Python uses AST parsing; JavaScript uses Node syntax checks and hidden ESM tests; TypeScript uses the repository-locked TypeScript compiler; Go uses go test; Rust uses offline cargo check before hidden tests.",
            "Every evaluator container uses a pre-existing local image, --pull=never, --network none, read-only workspace/evaluation mounts, and bounded writable build tmpfs.",
            "Prompt token count and context-window utilization are recorded per task; over-context prompts are rejected before generation with their measured overflow retained.",
            "Context-rejected tasks remain in end-to-end completion scoring but are excluded from the loss/intelligence divergence heuristic, which uses within-context generation attempts only.",
            "Per-language and aggregate reports separate context-window rejections from completed model-generation attempts.",
            "Generation throughput and timings are local measurements affected by hardware and system load.",
            "Docker mode uses an already-present image, disables networking, and mounts workspace and tests read-only; images are never pulled.",
            "Host execution is not sandboxed and requires explicit command-line acknowledgement.",
            "The loss/intelligence divergence flag is heuristic, not causal proof of overfitting.",
            "This controlled fixture suite is not a statistically representative estimate of performance across all unfamiliar repositories.",
        ],
    }
    output = write_report(Path(args.output), report)
    print("")
    print(f"Raw native benchmark: {summary['completed_task_count']}/{summary['baseline_valid_task_count']} baseline-valid target tasks completed (selected: {summary['task_count']}; invalid fixtures: {summary['invalid_fixture_count']})")
    if summary["patch_correctness_rate"] is None:
        print("Patch correctness: unavailable (no baseline-valid fixtures)")
    else:
        print(f"Patch correctness: {summary['patch_correctness_rate']:.1%}")
    if summary["regression_prevention_rate"] is not None:
        print(f"Regression prevention: {summary['regression_prevention_rate']:.1%}")
    else:
        print("Regression prevention: unavailable (no completed regression measurements)")
    for line in language_summary_lines(summary["language_breakdown"]):
        print(line)
    print("Generated tokens: " + str(summary["generated_tokens"]))
    print("Loss/intelligence divergence: " + ("DETECTED" if divergence["detected"] else "not detected"))
    print("Report: " + str(output))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SYNTRA's native PyTorch model directly on held-out coding fixtures.")
    parser.add_argument("--list", action="store_true", help="List tasks without loading PyTorch or a checkpoint.")
    parser.add_argument("--task", action="append", default=[], help="Select a task ID; may be repeated.")
    parser.add_argument("--checkpoint", help="Optional explicit native checkpoint path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--python-image", "--docker-image", dest="python_image", help="Local Python image tag (legacy --docker-image alias).")
    parser.add_argument("--node-image", default=None, help="Local Node.js image tag for TypeScript tasks.")
    parser.add_argument("--go-image", default=None, help="Local Go image tag for Go tasks.")
    parser.add_argument("--rust-image", default=None, help="Local Rust image tag for Rust tasks.")
    parser.add_argument("--isolation", choices=("docker", "host"), default="docker")
    parser.add_argument("--allow-host-execution", action="store_true", help="Acknowledge that host execution is not sandboxed.")
    parser.add_argument("--low-loss-threshold", type=float, default=LOW_TRAINING_LOSS_THRESHOLD)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("--max-new-tokens must be between 1 and 4096.")
    if not math.isfinite(args.low_loss_threshold) or args.low_loss_threshold < 0:
        parser.error("--low-loss-threshold must be finite and non-negative.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        suite = load_suite()
        if args.list:
            print("Raw native model benchmark suite " + suite["version"])
            for task in suite["tasks"]:
                print("  " + task["id"] + " [" + task["language"] + "] — " + task["title"])
            return 0
        run_suite(args)
        return 0
    except (BenchmarkError, OSError, RuntimeError, ValueError) as exc:
        print("SYNTRA raw-model benchmark: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

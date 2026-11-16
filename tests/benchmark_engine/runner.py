"""Held-out raw-native-model benchmark. Hidden evaluation code is not model-visible."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform as host_platform
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


def _cpu_model() -> str | None:
    """Read a local CPU model identifier when the current OS exposes one."""
    candidates: list[str] = []
    try:
        processor = host_platform.processor().strip()
    except (OSError, RuntimeError, TypeError, ValueError):
        processor = ""
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                processor_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if isinstance(processor_name, str) and processor_name.strip():
                candidates.insert(0, processor_name.strip())
        except (ImportError, OSError, TypeError, ValueError):
            pass
    elif sys.platform == "darwin":
        try:
            output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if output.returncode == 0 and output.stdout.strip():
                candidates.insert(0, output.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    if processor:
        candidates.append(processor)
    identifier = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if identifier:
        candidates.append(identifier)
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as processor_key:
                registered_name, _ = winreg.QueryValueEx(processor_key, "ProcessorNameString")
            if isinstance(registered_name, str) and registered_name.strip():
                candidates.insert(0, registered_name.strip())
        except (ImportError, OSError, TypeError, ValueError):
            pass
    elif sys.platform == "darwin":
        executable = shutil.which("sysctl")
        if executable:
            result = run_command(
                [executable, "-n", "machdep.cpu.brand_string"],
                cwd=REPOSITORY_ROOT,
                env=dict(os.environ),
                timeout_seconds=3,
                output_limit=1024,
            )
            value = result["stdout"].strip()
            if result["returncode"] == 0 and not result["timed_out"] and not result["output_limited"] and value:
                candidates.insert(0, value)
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.is_file() and cpuinfo.stat().st_size <= 1024 * 1024:
                for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                    key, separator, value = line.partition(":")
                    if separator and key.strip().casefold() in {"model name", "hardware"} and value.strip():
                        candidates.insert(0, value.strip())
                        break
        except OSError:
            pass
    for value in candidates:
        cleaned = " ".join(value.split())
        if cleaned and len(cleaned) <= 256:
            return cleaned
    return None


def _physical_memory_bytes() -> int | None:
    """Return installed physical memory only when the OS provides an exact value."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                result = int(status.ullTotalPhys)
                return result if result > 0 else None
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        return None

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * page_size
        return total if pages > 0 and page_size > 0 and total > 0 else None
    except (AttributeError, OSError, TypeError, ValueError, OverflowError):
        return None


def collect_hardware_identity(requested_device: str, resolved_device: Any, torch_module: Any) -> dict[str, Any]:
    """Capture actual host, PyTorch, and selected-device identity without estimates."""
    device_type = getattr(resolved_device, "type", None)
    device_index = getattr(resolved_device, "index", None)
    if device_type not in {"cpu", "cuda", "mps"}:
        raise BenchmarkError("The native runtime resolved an unsupported benchmark device.")

    cpu_model = _cpu_model()
    logical_cores = os.cpu_count()
    if type(logical_cores) is not int or logical_cores < 1:
        logical_cores = None
    total_memory = _physical_memory_bytes()
    try:
        torch_version = str(torch_module.__version__)
    except (AttributeError, TypeError):
        torch_version = None

    cuda_runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
    hip_runtime = getattr(getattr(torch_module, "version", None), "hip", None)
    accelerator_vendor = "cpu" if device_type == "cpu" else "apple_mps"
    accelerator_name = cpu_model if device_type == "cpu" else None
    accelerator_memory = None
    compute_capability = None
    cuda_device_count = None
    driver_version = None
    limitations: list[str] = []

    if device_type == "cuda":
        accelerator_vendor = "rocm" if hip_runtime else "nvidia_cuda"
        try:
            cuda_device_count = int(torch_module.cuda.device_count())
            if device_index is None:
                device_index = int(torch_module.cuda.current_device())
            if type(device_index) is not int or not 0 <= device_index < cuda_device_count:
                raise ValueError("Resolved CUDA device index is outside the available device list.")
            properties = torch_module.cuda.get_device_properties(device_index)
            accelerator_name = str(torch_module.cuda.get_device_name(device_index)).strip() or None
            memory_value = getattr(properties, "total_memory", None)
            accelerator_memory = int(memory_value) if isinstance(memory_value, (int, float)) and not isinstance(memory_value, bool) and memory_value > 0 else None
            major, minor = getattr(properties, "major", None), getattr(properties, "minor", None)
            if type(major) is int and type(minor) is int and major >= 0 and minor >= 0:
                compute_capability = str(major) + "." + str(minor)
            driver_lookup_type = "cuda-rocm" if hip_runtime else "cuda"
            driver_version = _discover_accelerator_driver_version(driver_lookup_type)
            if driver_version is None:
                limitations.append("The installed accelerator driver version could not be measured through local driver utilities.")
            limitations.append("CUDA device properties are reported by the selected PyTorch runtime; MIG partitioning may affect available memory.")
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
            limitations.append("CUDA hardware details could not be read: " + str(exc)[:240])
    elif device_type == "mps":
        accelerator_vendor = "apple_mps"
        limitations.append("PyTorch does not expose a stable Apple GPU model/total-memory identity through this benchmark collector.")
    elif not cpu_model:
        limitations.append("The operating system did not expose a CPU model identifier.")

    if total_memory is None:
        limitations.append("Installed physical memory could not be measured through the current operating-system API.")
    if logical_cores is None:
        limitations.append("Logical CPU count is unavailable.")

    resolved_name = (
        "cuda:" + str(device_index) if device_type == "cuda" and type(device_index) is int
        else device_type
    )
    device_record = {
        "requested": requested_device,
        "resolved": resolved_name,
        "type": device_type,
        "index": device_index if type(device_index) is int else None,
        "vendor": accelerator_vendor,
        "name": accelerator_name,
        "total_memory_bytes": accelerator_memory,
        "compute_capability": compute_capability,
        "available_device_count": cuda_device_count,
        "cuda_runtime_version": cuda_runtime,
        "rocm_runtime_version": hip_runtime,
        "driver_version": driver_version,
    }
    host_record = {
        "system": host_platform.system() or None,
        "release": host_platform.release() or None,
        "machine": host_platform.machine() or None,
        "cpu_model": cpu_model,
        "logical_cores": logical_cores,
        "physical_memory_bytes": total_memory,
    }
    python_record = {
        "implementation": host_platform.python_implementation(),
        "version": sys.version.split()[0],
    }
    identity_complete = bool(
        host_record["system"]
        and host_record["release"]
        and host_record["machine"]
        and host_record["cpu_model"]
        and logical_cores
        and total_memory
        and torch_version
        and device_record["name"]
        and (
            device_type == "cpu"
            or (
                device_type == "cuda"
                and accelerator_memory
                and (cuda_runtime or hip_runtime)
                and driver_version
            )
        )
    )
    fingerprint_device_record = {
        key: value for key, value in device_record.items() if key != "requested"
    }
    fingerprint_payload = {
        "schema_version": 1,
        "host": host_record,
        "python": python_record,
        "pytorch": {
            "version": torch_version,
            "cuda_runtime_version": cuda_runtime,
            "rocm_runtime_version": hip_runtime,
        },
        "device": fingerprint_device_record,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "identity_complete": identity_complete,
        "fingerprint_sha256": fingerprint,
        **fingerprint_payload,
        "device": device_record,
        "limitations": limitations,
    }


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


def _discover_accelerator_driver_version(device_type: str) -> str | None:
    """Read a host-wide accelerator driver version from local OS tools."""
    if device_type == "cuda":
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None
        result = run_command(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPOSITORY_ROOT,
            env=dict(os.environ),
            timeout_seconds=5,
            output_limit=16 * 1024,
        )
        if result["returncode"] != 0 or result["timed_out"] or result["output_limited"]:
            return None
        versions = {
            line.strip()
            for line in result["stdout"].splitlines()
            if re.fullmatch(r"[0-9]+(?:\.[0-9A-Za-z_-]+){0,3}", line.strip())
        }
        # NVIDIA's installed kernel driver is host-wide. If all visible GPUs
        # report the same value, it is independent of CUDA_VISIBLE_DEVICES
        # logical-to-physical ordinal remapping. Conflicting output is ambiguous.
        return next(iter(versions)) if len(versions) == 1 else None

    if device_type == "cuda-rocm":
        executable = shutil.which("rocm-smi")
        if executable:
            result = run_command(
                [executable, "--showdriverversion"],
                cwd=REPOSITORY_ROOT,
                env=dict(os.environ),
                timeout_seconds=5,
                output_limit=16 * 1024,
            )
            if result["returncode"] == 0 and not result["timed_out"] and not result["output_limited"]:
                match = re.search(
                    r"(?im)^\s*(?:GPU\s+)?Driver version\s*:\s*([0-9][A-Za-z0-9._+-]{0,63})\s*$",
                    result["stdout"],
                )
                if match:
                    return match.group(1)
        sysfs_version = Path("/sys/module/amdgpu/version")
        try:
            if not sysfs_version.is_symlink():
                info = sysfs_version.lstat()
                if stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 256:
                    value = sysfs_version.read_text(encoding="utf-8").strip()
                    if re.fullmatch(r"[0-9][A-Za-z0-9._+-]{0,63}", value):
                        return value
        except OSError:
            pass
    return None


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


def _report_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(normalized) and normalized >= 0


def _report_rate(value: Any, numerator: int, denominator: int) -> bool:
    expected = round(numerator / denominator, 4) if denominator else None
    if expected is None:
        return value is None
    if not _report_number(value):
        return False
    return float(value) == expected


def validate_raw_benchmark_report(report: Any) -> dict[str, Any]:
    """Validate report provenance and recompute its task-denominator invariants."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise BenchmarkError("Raw benchmark report must use schema_version 1.")
    if report.get("benchmark_track") != "raw-native-model":
        raise BenchmarkError("The selected report is not from the raw-native-model benchmark track.")
    suite_version = report.get("suite_version")
    if not isinstance(suite_version, str) or re.fullmatch(r"\d+\.\d+\.\d+", suite_version) is None:
        raise BenchmarkError("Raw benchmark report contains an invalid suite version.")
    if not isinstance(report.get("run_id"), str) or not report["run_id"].strip():
        raise BenchmarkError("Raw benchmark report is missing its run ID.")
    protocol = report.get("evaluation_protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema_version") != 1
        or not isinstance(protocol.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", protocol["sha256"]) is None
        or type(protocol.get("file_count")) is not int
        or protocol["file_count"] < 1
        or type(protocol.get("total_bytes")) is not int
        or protocol["total_bytes"] < 1
    ):
        raise BenchmarkError("Raw benchmark report is missing complete evaluation-protocol identity.")

    model = report.get("model")
    if (
        not isinstance(model, dict)
        or model.get("engine") != "syntra-native-pytorch"
        or not isinstance(model.get("checkpoint_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", model["checkpoint_sha256"]) is None
    ):
        raise BenchmarkError("Raw benchmark report is missing native checkpoint identity.")

    tasks = report.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 100:
        raise BenchmarkError("Raw benchmark report must contain between 1 and 100 task results.")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise BenchmarkError("Raw benchmark task result must be an object.")
        task_id = task.get("id")
        if not isinstance(task_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", task_id) is None or task_id in seen:
            raise BenchmarkError("Raw benchmark report contains an invalid or duplicate task ID.")
        seen.add(task_id)
        if not isinstance(task.get("language"), str) or not task["language"]:
            raise BenchmarkError("Raw benchmark task " + task_id + " is missing its language.")
        for key in ("baseline_validated", "patch_correctness", "task_completion"):
            if type(task.get(key)) is not bool:
                raise BenchmarkError("Raw benchmark task " + task_id + " has an invalid " + key + " value.")
        if task.get("regression_prevention") is not None and type(task["regression_prevention"]) is not bool:
            raise BenchmarkError("Raw benchmark task " + task_id + " has an invalid regression-prevention measurement.")
        speed = task.get("execution_speed")
        if not isinstance(speed, dict) or not _report_number(speed.get("total_ms")):
            raise BenchmarkError("Raw benchmark task " + task_id + " is missing valid total execution time.")
        if speed.get("generation_ms") is not None and not _report_number(speed["generation_ms"]):
            raise BenchmarkError("Raw benchmark task " + task_id + " has an invalid generation-time measurement.")
        generated_tokens = speed.get("generated_tokens")
        if type(generated_tokens) is not int or generated_tokens < 0:
            raise BenchmarkError("Raw benchmark task " + task_id + " has an invalid generated-token count.")
        throughput = speed.get("output_tokens_per_second")
        if throughput is not None and not _report_number(throughput):
            raise BenchmarkError("Raw benchmark task " + task_id + " has an invalid generation-throughput measurement.")
        generation_ms = speed.get("generation_ms")
        if generation_ms is None:
            if generated_tokens != 0 or throughput is not None:
                raise BenchmarkError("Raw benchmark task " + task_id + " reports generated tokens or throughput without a generation-time measurement.")
        elif generation_ms == 0:
            if throughput is not None:
                raise BenchmarkError("Raw benchmark task " + task_id + " reports throughput for a zero-duration generation.")
        else:
            expected_throughput = round(generated_tokens / (float(generation_ms) / 1000), 2)
            if not _report_number(throughput) or float(throughput) != expected_throughput:
                raise BenchmarkError("Raw benchmark task " + task_id + " has generation throughput inconsistent with its measured tokens/time.")

    valid = [task for task in tasks if task["baseline_validated"] is True]
    valid_count = len(valid)
    completed_count = sum(task["task_completion"] is True for task in valid)
    patch_count = sum(task["patch_correctness"] is True for task in valid)
    regression_measured = [task for task in valid if task["regression_prevention"] is not None]
    regression_count = sum(task["regression_prevention"] is True for task in regression_measured)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise BenchmarkError("Raw benchmark report is missing its measured summary.")
    expected_counts = {
        "task_count": len(tasks),
        "baseline_valid_task_count": valid_count,
        "invalid_fixture_count": len(tasks) - valid_count,
        "completed_task_count": completed_count,
        "patch_correct_count": patch_count,
        "regression_measured_count": len(regression_measured),
        "regression_prevented_count": regression_count,
    }
    for key, expected in expected_counts.items():
        if type(summary.get(key)) is not int or summary[key] != expected:
            raise BenchmarkError("Raw benchmark summary has inconsistent " + key + " evidence.")
    expected_rates = {
        "task_completion_rate": (completed_count, valid_count),
        "patch_correctness_rate": (patch_count, valid_count),
        "regression_prevention_rate": (regression_count, len(regression_measured)),
    }
    for key, (numerator, denominator) in expected_rates.items():
        if not _report_rate(summary.get(key), numerator, denominator):
            raise BenchmarkError("Raw benchmark summary has an inconsistent " + key + " denominator or rate.")

    hardware = report.get("hardware")
    if hardware is not None:
        if (
            not isinstance(hardware, dict)
            or hardware.get("schema_version") != 1
            or type(hardware.get("identity_complete")) is not bool
            or not isinstance(hardware.get("fingerprint_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", hardware["fingerprint_sha256"]) is None
        ):
            raise BenchmarkError("Raw benchmark report contains malformed hardware identity.")
        host = hardware.get("host")
        python_info = hardware.get("python")
        pytorch = hardware.get("pytorch")
        device = hardware.get("device")
        if not all(isinstance(value, dict) for value in (host, python_info, pytorch, device)):
            raise BenchmarkError("Raw benchmark hardware identity is missing host, Python, PyTorch, or device fields.")
        def nonempty_text(value: Any) -> bool:
            return isinstance(value, str) and bool(value.strip()) and len(value) <= 512

        def positive_integer(value: Any) -> bool:
            return type(value) is int and value > 0

        if device.get("type") not in {"cpu", "cuda", "mps"} or not nonempty_text(device.get("resolved")):
            raise BenchmarkError("Raw benchmark hardware identity contains an invalid resolved device.")
        if model.get("device_resolved") != device.get("resolved"):
            raise BenchmarkError("Raw benchmark model and hardware records disagree about the resolved device.")
        if (
            not isinstance(hardware.get("limitations"), list)
            or any(not isinstance(item, str) for item in hardware["limitations"])
        ):
            raise BenchmarkError("Raw benchmark hardware limitations must be a list of strings.")
        optional_text_fields = (
            (host, "system"), (host, "release"), (host, "machine"), (host, "cpu_model"),
            (python_info, "implementation"), (python_info, "version"),
            (pytorch, "version"), (pytorch, "cuda_runtime_version"), (pytorch, "rocm_runtime_version"),
            (device, "requested"), (device, "resolved"), (device, "vendor"), (device, "name"),
            (device, "compute_capability"), (device, "driver_version"),
        )
        for source, field in optional_text_fields:
            value = source.get(field)
            if value is not None and not nonempty_text(value):
                raise BenchmarkError("Raw benchmark hardware field " + field + " must be non-empty text or null.")
        for source, field in ((host, "logical_cores"), (host, "physical_memory_bytes"),
                              (device, "index"), (device, "total_memory_bytes"),
                              (device, "available_device_count")):
            value = source.get(field)
            if value is not None and (
                type(value) is not int or value < (0 if field == "index" else 1)
            ):
                raise BenchmarkError("Raw benchmark hardware field " + field + " has an invalid numeric value.")
        if device.get("type") == "cpu" and (
            device.get("resolved") != "cpu" or device.get("vendor") != "cpu"
        ):
            raise BenchmarkError("CPU hardware identity must resolve to the CPU backend.")
        if device.get("type") == "mps" and (
            device.get("resolved") != "mps" or device.get("vendor") != "apple_mps"
        ):
            raise BenchmarkError("MPS hardware identity must resolve to the Apple MPS backend.")
        if device.get("type") == "cuda" and (
            re.fullmatch(r"cuda:[0-9]+", device["resolved"]) is None
            or device.get("vendor") not in {"nvidia_cuda", "rocm"}
        ):
            raise BenchmarkError("CUDA/ROCm hardware identity must include its selected device index and backend.")
        if hardware["identity_complete"]:
            common_complete = all((
                nonempty_text(host.get("system")),
                nonempty_text(host.get("release")),
                nonempty_text(host.get("machine")),
                nonempty_text(host.get("cpu_model")),
                positive_integer(host.get("logical_cores")),
                positive_integer(host.get("physical_memory_bytes")),
                nonempty_text(python_info.get("implementation")),
                nonempty_text(python_info.get("version")),
                nonempty_text(pytorch.get("version")),
                nonempty_text(device.get("name")),
            ))
            if not common_complete:
                raise BenchmarkError("Hardware identity is marked complete but required host/runtime/device measurements are missing.")
            if device["type"] == "cpu":
                if device.get("name") != host.get("cpu_model"):
                    raise BenchmarkError("Complete CPU identity must match the measured host CPU model.")
            elif device["type"] == "cuda":
                runtime_version = pytorch.get("rocm_runtime_version") if device.get("vendor") == "rocm" else pytorch.get("cuda_runtime_version")
                device_count = device.get("available_device_count")
                device_index = device.get("index")
                if not all((
                    type(device_index) is int and device_index >= 0,
                    positive_integer(device_count) and device_index < device_count,
                    positive_integer(device.get("total_memory_bytes")),
                    nonempty_text(device.get("driver_version")),
                    nonempty_text(runtime_version),
                    device.get("cuda_runtime_version") == pytorch.get("cuda_runtime_version"),
                    device.get("rocm_runtime_version") == pytorch.get("rocm_runtime_version"),
                    device.get("resolved") == "cuda:" + str(device_index),
                    (
                        device.get("vendor") != "nvidia_cuda"
                        or nonempty_text(device.get("compute_capability"))
                    ),
                )):
                    raise BenchmarkError("CUDA/ROCm identity is marked complete without consistent selected-device, memory, runtime, or driver evidence.")
            elif device["type"] == "mps":
                raise BenchmarkError("MPS identity cannot be marked complete without a measured Apple GPU identity and memory specification.")
        fingerprint_device = {key: value for key, value in device.items() if key != "requested"}
        fingerprint_payload = {
            "schema_version": 1,
            "host": host,
            "python": python_info,
            "pytorch": pytorch,
            "device": fingerprint_device,
        }
        expected_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if expected_fingerprint != hardware["fingerprint_sha256"]:
            raise BenchmarkError("Raw benchmark hardware fingerprint does not match its recorded fields.")
    return report


def _comparison_metric(
    key: str,
    baseline: float | None,
    candidate: float | None,
    *,
    higher_is_better: bool,
    comparable: bool = True,
) -> dict[str, Any]:
    if not comparable:
        direction = "incomparable"
    elif baseline is None or candidate is None:
        direction = "unavailable"
    else:
        delta = candidate - baseline
        direction = (
            "unchanged" if delta == 0
            else "improved" if ((delta > 0) == higher_is_better)
            else "regressed"
        )
    delta = candidate - baseline if baseline is not None and candidate is not None else None
    relative = (
        round((delta / baseline) * 100, 2)
        if comparable and delta is not None and baseline not in (None, 0)
        else None
    )
    return {
        "key": key,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "relative_change_percent": relative,
        "direction": direction,
    }


def compare_raw_benchmark_reports(baseline_input: Any, candidate_input: Any) -> dict[str, Any]:
    """Compare same-protocol raw-model reports without hiding denominator or hardware changes."""
    baseline = validate_raw_benchmark_report(baseline_input)
    candidate = validate_raw_benchmark_report(candidate_input)
    if baseline["suite_version"] != candidate["suite_version"]:
        raise BenchmarkError("Cannot compare raw-model reports from different suite versions.")
    if baseline["evaluation_protocol"]["sha256"] != candidate["evaluation_protocol"]["sha256"]:
        raise BenchmarkError("Cannot compare raw-model reports from different evaluation protocols; rerun both against the same protocol identity.")

    baseline_tasks = {task["id"]: task for task in baseline["tasks"]}
    candidate_tasks = {task["id"]: task for task in candidate["tasks"]}
    if set(baseline_tasks) != set(candidate_tasks):
        raise BenchmarkError("Cannot compare different raw-model task selections.")
    baseline_valid = {task_id for task_id, task in baseline_tasks.items() if task["baseline_validated"]}
    candidate_valid = {task_id for task_id, task in candidate_tasks.items() if task["baseline_validated"]}
    if baseline_valid != candidate_valid:
        raise BenchmarkError("Cannot compare reports whose hidden-baseline-valid task sets differ.")

    baseline_hardware = baseline.get("hardware") or {}
    candidate_hardware = candidate.get("hardware") or {}
    hardware_comparable = bool(
        baseline_hardware.get("identity_complete") is True
        and candidate_hardware.get("identity_complete") is True
        and baseline_hardware.get("fingerprint_sha256") == candidate_hardware.get("fingerprint_sha256")
    )
    baseline_device = baseline_hardware.get("device", {})
    candidate_device = candidate_hardware.get("device", {})
    if baseline_device.get("resolved") != candidate_device.get("resolved"):
        hardware_comparable = False

    ordered_valid = sorted(baseline_valid)
    count = len(ordered_valid)
    baseline_completion = sum(baseline_tasks[task_id]["task_completion"] for task_id in ordered_valid)
    candidate_completion = sum(candidate_tasks[task_id]["task_completion"] for task_id in ordered_valid)
    baseline_patch = sum(baseline_tasks[task_id]["patch_correctness"] for task_id in ordered_valid)
    candidate_patch = sum(candidate_tasks[task_id]["patch_correctness"] for task_id in ordered_valid)

    baseline_regression_ids = {
        task_id for task_id in ordered_valid
        if baseline_tasks[task_id]["regression_prevention"] is not None
    }
    candidate_regression_ids = {
        task_id for task_id in ordered_valid
        if candidate_tasks[task_id]["regression_prevention"] is not None
    }
    regression_comparable = baseline_regression_ids == candidate_regression_ids
    baseline_regression_count = sum(
        baseline_tasks[task_id]["regression_prevention"] is True
        for task_id in sorted(baseline_regression_ids)
    )
    candidate_regression_count = sum(
        candidate_tasks[task_id]["regression_prevention"] is True
        for task_id in sorted(candidate_regression_ids)
    )

    baseline_generation_ids = {
        task_id for task_id in ordered_valid
        if baseline_tasks[task_id]["execution_speed"]["generation_ms"] is not None
    }
    candidate_generation_ids = {
        task_id for task_id in ordered_valid
        if candidate_tasks[task_id]["execution_speed"]["generation_ms"] is not None
    }
    generation_denominators_match = baseline_generation_ids == candidate_generation_ids
    baseline_generation_ms = sum(
        float(baseline_tasks[task_id]["execution_speed"]["generation_ms"])
        for task_id in sorted(baseline_generation_ids)
    )
    candidate_generation_ms = sum(
        float(candidate_tasks[task_id]["execution_speed"]["generation_ms"])
        for task_id in sorted(candidate_generation_ids)
    )
    baseline_generated_tokens = sum(
        baseline_tasks[task_id]["execution_speed"]["generated_tokens"]
        for task_id in sorted(baseline_generation_ids)
    )
    candidate_generated_tokens = sum(
        candidate_tasks[task_id]["execution_speed"]["generated_tokens"]
        for task_id in sorted(candidate_generation_ids)
    )
    baseline_throughput = (
        baseline_generated_tokens / (baseline_generation_ms / 1000)
        if baseline_generation_ms > 0 else None
    )
    candidate_throughput = (
        candidate_generated_tokens / (candidate_generation_ms / 1000)
        if candidate_generation_ms > 0 else None
    )

    baseline_total_ms = sum(
        float(baseline_tasks[task_id]["execution_speed"]["total_ms"])
        for task_id in ordered_valid
    ) / count if count else None
    candidate_total_ms = sum(
        float(candidate_tasks[task_id]["execution_speed"]["total_ms"])
        for task_id in ordered_valid
    ) / count if count else None

    metrics = [
        _comparison_metric(
            "task_completion_rate",
            baseline_completion / count if count else None,
            candidate_completion / count if count else None,
            higher_is_better=True,
        ),
        _comparison_metric(
            "patch_correctness_rate",
            baseline_patch / count if count else None,
            candidate_patch / count if count else None,
            higher_is_better=True,
        ),
        _comparison_metric(
            "regression_prevention_rate",
            baseline_regression_count / len(baseline_regression_ids) if baseline_regression_ids else None,
            candidate_regression_count / len(candidate_regression_ids) if candidate_regression_ids else None,
            higher_is_better=True,
            comparable=regression_comparable,
        ),
        _comparison_metric(
            "mean_total_task_ms",
            baseline_total_ms,
            candidate_total_ms,
            higher_is_better=False,
            comparable=hardware_comparable,
        ),
        _comparison_metric(
            "generation_tokens_per_second",
            baseline_throughput,
            candidate_throughput,
            higher_is_better=True,
            comparable=hardware_comparable and generation_denominators_match,
        ),
    ]
    quality_directions = [
        metric["direction"] for metric in metrics[:3]
        if metric["direction"] not in {"unavailable", "incomparable"}
    ]
    improved = sum(value == "improved" for value in quality_directions)
    regressed = sum(value == "regressed" for value in quality_directions)
    quality_status = (
        "unavailable" if not quality_directions
        else "mixed" if improved and regressed
        else "improved" if improved
        else "regressed" if regressed
        else "unchanged"
    )

    warnings = []
    if count == 0:
        warnings.append("No task has a valid hidden baseline in both reports; task-quality comparison is unavailable.")
    if not hardware_comparable:
        warnings.append("Hardware/device identity is missing, incomplete, or different. Latency and throughput are reported but marked incomparable.")
    if not regression_comparable:
        warnings.append("Regression-prevention measurements cover different task subsets; regression rates are incomparable.")
    if not generation_denominators_match:
        warnings.append("Generated-token throughput covers different task subsets; throughput is incomparable.")
    if baseline["model"]["checkpoint_sha256"] == candidate["model"]["checkpoint_sha256"]:
        warnings.append("Both reports use the same checkpoint SHA-256; this is a repeatability comparison, not a checkpoint-change comparison.")
    if quality_status in {"improved", "regressed"} and not hardware_comparable:
        warnings.append("Task-outcome differences are measured observations, not a controlled causal quality claim, because execution environments differ or are incomplete.")

    tasks = []
    for task_id in sorted(baseline_tasks):
        before = baseline_tasks[task_id]
        after = candidate_tasks[task_id]
        if task_id not in baseline_valid:
            task_status = "excluded-invalid-baseline"
        else:
            changes = []
            for field in ("task_completion", "patch_correctness"):
                if before[field] != after[field]:
                    changes.append("improved" if after[field] else "regressed")
            if before["regression_prevention"] is not None and after["regression_prevention"] is not None:
                if before["regression_prevention"] != after["regression_prevention"]:
                    changes.append("improved" if after["regression_prevention"] else "regressed")
            task_status = (
                "mixed" if "improved" in changes and "regressed" in changes
                else changes[0] if changes
                else "unchanged"
            )
        tasks.append({
            "id": task_id,
            "language": before["language"],
            "baseline_validated": task_id in baseline_valid,
            "baseline_task_completion": before["task_completion"],
            "candidate_task_completion": after["task_completion"],
            "baseline_patch_correctness": before["patch_correctness"],
            "candidate_patch_correctness": after["patch_correctness"],
            "baseline_regression_prevention": before["regression_prevention"],
            "candidate_regression_prevention": after["regression_prevention"],
            "status": task_status,
        })

    return {
        "schema_version": 1,
        "comparison_track": "raw-native-model",
        "suite_version": baseline["suite_version"],
        "evaluation_protocol_sha256": baseline["evaluation_protocol"]["sha256"],
        "compared_at": _now(),
        "baseline": {
            "run_id": baseline["run_id"],
            "checkpoint_sha256": baseline["model"]["checkpoint_sha256"],
            "device_resolved": baseline.get("model", {}).get("device_resolved"),
            "hardware_fingerprint_sha256": baseline_hardware.get("fingerprint_sha256"),
            "baseline_valid_task_count": count,
        },
        "candidate": {
            "run_id": candidate["run_id"],
            "checkpoint_sha256": candidate["model"]["checkpoint_sha256"],
            "device_resolved": candidate.get("model", {}).get("device_resolved"),
            "hardware_fingerprint_sha256": candidate_hardware.get("fingerprint_sha256"),
            "baseline_valid_task_count": count,
        },
        "environment": {
            "hardware_comparable": hardware_comparable,
            "generation_task_denominators_match": generation_denominators_match,
            "regression_task_denominators_match": regression_comparable,
        },
        "quality_status": quality_status,
        "metrics": metrics,
        "tasks": tasks,
        "warnings": warnings,
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
    from syntra_ai.hardware_telemetry import (
        device_memory_sample,
        process_peak_rss_bytes,
        reset_device_peak_stats,
    )
    checkpoint = (Path(args.checkpoint).expanduser() if args.checkpoint else runtime.checkpoint_path()).resolve(strict=True)
    checkpoint_evidence = load_checkpoint_evidence(checkpoint)
    try:
        resolved_device = runtime.choose_device(args.device)
        hardware_identity = collect_hardware_identity(args.device, resolved_device, runtime.torch)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkError("Cannot establish the actual native inference device/hardware identity: " + str(exc)) from None
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
    device_peak_reset_succeeded = reset_device_peak_stats(resolved_device, runtime.torch)
    run_id, started_at, results = "raw-model-" + uuid.uuid4().hex, _now(), []
    for task in selected:
        print("[" + task["id"] + "] Evaluating the actual local SYNTRA checkpoint...")
        try:
            result = run_task(task, runtime=runtime, checkpoint=checkpoint, device=hardware_identity["device"]["resolved"], max_new_tokens=args.max_new_tokens, isolation=isolations[task["language"]], run_id=run_id)
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
    process_peak = process_peak_rss_bytes()
    process_peak_method = (
        "windows_peak_working_set_bytes" if os.name == "nt"
        else "posix_getrusage_peak_rss_bytes"
    ) if process_peak is not None else "unavailable"
    if resolved_device.type == "cuda" and not device_peak_reset_succeeded:
        device_memory_bytes, device_memory_method = None, "cuda_peak_reset_failed"
    else:
        device_memory_bytes, device_memory_method = device_memory_sample(resolved_device, runtime.torch)
    if device_memory_method == "cuda_peak_allocated":
        device_memory_scope = "suite_peak_since_reset_before_first_task"
    elif device_memory_method in {"mps_driver_allocated_current", "mps_allocator_current"}:
        device_memory_scope = "post_suite_current_sample_not_peak"
    elif device_memory_method == "cpu_not_applicable":
        device_memory_scope = "not_applicable"
    else:
        device_memory_scope = "unavailable"
    memory_evidence = {
        "process_peak_rss_bytes": process_peak,
        "process_peak_rss_method": process_peak_method,
        "process_peak_rss_scope": "process_lifetime_including_imports_and_benchmark_preflight",
        "device_allocated_memory_bytes": device_memory_bytes,
        "device_memory_method": device_memory_method,
        "device_memory_scope": device_memory_scope,
    }
    report = {
        "schema_version": 1, "benchmark_track": "raw-native-model", "suite_version": suite["version"],
        "evaluation_protocol": protocol_identity,
        "run_id": run_id, "started_at": started_at, "completed_at": _now(),
        "model": {
            "engine": "syntra-native-pytorch", "checkpoint": checkpoint_evidence["path"],
            "checkpoint_sha256": checkpoint_evidence["sha256"], "checkpoint_identity": initial_identity,
            "device_requested": args.device, "device_resolved": hardware_identity["device"]["resolved"],
            "trained_steps": checkpoint_evidence["step"],
            "training_mode": checkpoint_evidence["training_mode"],
            "instruction_dataset_sha256": checkpoint_evidence["instruction_dataset_sha256"],
            "instruction_dataset_bytes": checkpoint_evidence["instruction_dataset_bytes"],
        },
        "hardware": hardware_identity,
        "memory": memory_evidence,
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
            "Host CPU model, logical-core count, installed physical memory, operating-system identity, PyTorch build, resolved inference device and exposed accelerator properties are recorded as measured; unavailable properties remain null and are listed as limitations.",
            "Hardware identity is incomplete when an essential host/device property or (for CUDA/ROCm) the driver version is unavailable; do not make controlled performance comparisons from incomplete identities.",
            "Process peak RSS is an OS high-water measurement for the complete benchmark process lifetime, including imports and preflight; it is not a per-task measurement.",
            "CUDA memory is PyTorch allocator peak allocated bytes since reset immediately before the task loop. MPS memory, when exposed, is a post-suite current-allocation sample and is not reported as a peak.",
            "CPU device-memory usage is not applicable; unavailable RSS/device measurements remain null with their method and scope recorded.",
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

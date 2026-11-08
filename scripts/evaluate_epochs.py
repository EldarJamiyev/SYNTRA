"""Run SYNTRA Phase 4 local pretraining, curriculum SFT, and background benchmarks.

All training, checkpointing, curriculum generation, and evaluation run locally.
The benchmark uses pre-provisioned offline Docker images by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "training" / "phase4_milestone4.json"


class OrchestrationError(RuntimeError):
    """A local training, dataset, snapshot, or evaluation integrity failure."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _inside_root(path: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = ROOT / raw
    lexical = Path(os.path.abspath(raw))
    root = ROOT.resolve()
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        raise OrchestrationError("All orchestration outputs and checkpoints must stay inside the SYNTRA repository.") from None

    # Check every existing component before resolving: resolving first would
    # erase the evidence that an in-repository symlink redirected the write.
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise OrchestrationError("Symbolic-link path components are not allowed for orchestration outputs or checkpoints.")
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise OrchestrationError("All orchestration outputs and checkpoints must stay inside the SYNTRA repository.") from None
    return resolved


def _strict_json(value: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("Invalid JSON constant: " + value)

    return json.loads(value, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def _read_json(path: Path, limit: int = 10 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError("Expected a regular, non-symlink JSON file: " + str(path))
    if path.stat().st_size > limit:
        raise OrchestrationError("JSON file exceeds the bounded read limit: " + str(path))
    try:
        result = _strict_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OrchestrationError("Cannot read valid JSON from " + str(path) + ": " + str(exc)) from None
    if not isinstance(result, dict):
        raise OrchestrationError("Expected a JSON object in " + str(path))
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = _inside_root(path)
    if path.is_symlink():
        raise OrchestrationError("Refusing to replace a symbolic-link output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hash_regular_file(path: Path) -> str:
    if path.is_symlink():
        raise OrchestrationError("Checkpoint artifact must not be a symbolic link: " + str(path))
    try:
        before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise OrchestrationError("Cannot inspect checkpoint artifact " + str(path) + ": " + str(exc)) from None
    if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
        raise OrchestrationError("Checkpoint artifact must be a non-empty regular file: " + str(path))
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise OrchestrationError("Checkpoint artifact changed while opening: " + str(path))
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise OrchestrationError("Cannot hash checkpoint artifact " + str(path) + ": " + str(exc)) from None
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise OrchestrationError("Checkpoint artifact changed while being hashed: " + str(path))
    return digest.hexdigest()


def _copy_stable_read_only(source: Path, destination: Path) -> str:
    destination = _inside_root(destination)
    digest_before = _hash_regular_file(source)
    source_stat = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        current = source.stat()
        if (
            source_stat.st_size != current.st_size
            or source_stat.st_mtime_ns != current.st_mtime_ns
            or source_stat.st_ctime_ns != current.st_ctime_ns
        ):
            raise OrchestrationError("Checkpoint changed while its evaluation snapshot was being copied.")
        copied_hash = _hash_regular_file(temporary)
        if copied_hash != digest_before:
            raise OrchestrationError("Evaluation snapshot checksum does not match its source checkpoint.")
        os.utime(temporary, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary, destination)
        return copied_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _format_metric(value: Any, digits: int = 6) -> str:
    number = _number(value)
    return "unavailable" if number is None else f"{number:.{digits}g}"


def load_profile(path: Path) -> dict[str, Any]:
    profile = _read_json(path, limit=256 * 1024)
    if profile.get("schema_version") != 1:
        raise OrchestrationError("Training profile must use schema_version 1.")
    for section in ("pretraining", "curriculum", "sft", "evaluation"):
        if not isinstance(profile.get(section), dict):
            raise OrchestrationError("Training profile section is missing or malformed: " + section)
    return profile


def _resolve_profile_path(value: str, *, default_base: Path = ROOT) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = default_base / candidate
    return _inside_root(candidate)


def _validate_positive_int(section: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = section.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise OrchestrationError(f"Profile value {key} must be an integer between {minimum} and {maximum}.")
    return value


def _validate_profile_values(profile: dict[str, Any]) -> None:
    pre = profile["pretraining"]
    _validate_positive_int(pre, "max_steps", 1, 1_000_000)
    _validate_positive_int(pre, "batch_size", 1, 256)
    _validate_positive_int(pre, "block_size", 16, 16_384)
    _validate_positive_int(pre, "layers", 1, 24)
    _validate_positive_int(pre, "heads", 1, 16)
    _validate_positive_int(pre, "width", 32, 1024)
    _validate_positive_int(pre, "vocab_size", 258, 8192)
    _validate_positive_int(pre, "warmup_steps", 1, 1_000_000)
    _validate_positive_int(pre, "decay_steps", 2, 10_000_000)
    _validate_positive_int(pre, "evaluation_batches", 1, 1000)
    _validate_positive_int(pre, "tokenizer_bytes", 1024, 10 * 1024 * 1024)
    _validate_positive_int(pre, "max_dataset_bytes", 1024, 256 * 1024 * 1024)
    if pre["decay_steps"] <= pre["warmup_steps"]:
        raise OrchestrationError("pretraining.decay_steps must exceed pretraining.warmup_steps.")
    base_lr = _number(pre.get("learning_rate"))
    min_lr = _number(pre.get("minimum_learning_rate"))
    weight_decay = _number(pre.get("weight_decay"))
    dropout = _number(pre.get("dropout"))
    if base_lr is None or not 1e-6 <= base_lr <= 1e-2:
        raise OrchestrationError("pretraining.learning_rate is outside the supported range.")
    if min_lr is None or not 1e-7 <= min_lr <= base_lr:
        raise OrchestrationError("pretraining.minimum_learning_rate must be positive and no greater than learning_rate.")
    if weight_decay is None or not 0 <= weight_decay <= 1:
        raise OrchestrationError("pretraining.weight_decay must be between 0 and 1.")
    if dropout is None or not 0 <= dropout < 0.5:
        raise OrchestrationError("pretraining.dropout must be between 0 and 0.5.")
    curriculum = profile["curriculum"]
    _validate_positive_int(curriculum, "max_examples", 4, 10_000)
    _validate_positive_int(curriculum, "max_tokens", 1024, 3_000_000)
    sft = profile["sft"]
    _validate_positive_int(sft, "steps", 1, 1_000_000)
    _validate_positive_int(sft, "batch_size", 1, 128)
    sft_lr = _number(sft.get("learning_rate"))
    if sft_lr is None or not 1e-6 <= sft_lr <= 1e-2:
        raise OrchestrationError("sft.learning_rate is outside the supported range.")
    evaluation = profile["evaluation"]
    _validate_positive_int(evaluation, "every_steps", 1, 100_000)
    _validate_positive_int(evaluation, "max_parallel", 1, 1)
    _validate_positive_int(evaluation, "max_new_tokens", 1, 4096)
    if evaluation.get("device") not in {"auto", "cpu", "cuda", "mps"}:
        raise OrchestrationError("evaluation.device must be auto, cpu, cuda, or mps.")
    if evaluation.get("isolation") not in {"docker", "host"}:
        raise OrchestrationError("evaluation.isolation must be docker or host.")


@dataclass
class EvaluationJob:
    stage: str
    step: int
    checkpoint_sha256: str
    checkpoint_path: Path
    report_path: Path
    log_path: Path
    process: subprocess.Popen
    log_handle: Any


class CheckpointSnapshots:
    """Create immutable, checksum-verified copies of trainer checkpoints."""

    def __init__(self, checkpoint: Path, snapshot_root: Path, run_id: str):
        self.checkpoint = checkpoint
        self.snapshot_root = _inside_root(snapshot_root) / run_id

    def create(self, stage: str, step: int) -> tuple[Path, str]:
        if not self.checkpoint.is_file() or self.checkpoint.is_symlink():
            raise OrchestrationError("The trainer did not produce a regular checkpoint at the expected path.")
        checkpoint_hash = _hash_regular_file(self.checkpoint)
        directory = self.snapshot_root / f"{stage}-step-{step:09d}-{checkpoint_hash[:12]}"
        if directory.exists():
            raise OrchestrationError("Refusing to reuse an existing checkpoint snapshot directory.")
        directory.mkdir(parents=True, exist_ok=False)
        assets = [self.checkpoint]
        sidecar = self.checkpoint.with_suffix(self.checkpoint.suffix + ".json")
        if sidecar.is_file() and not sidecar.is_symlink():
            assets.append(sidecar)
        try:
            from syntra_ai.training import inference_checkpoint_path
            companion = inference_checkpoint_path(self.checkpoint)
        except (ImportError, OSError, ValueError):
            companion = None
        if companion is not None and companion.is_file() and not companion.is_symlink():
            assets.append(companion)

        copied: dict[str, str] = {}
        try:
            for source in assets:
                target = directory / source.name
                copied[source.name] = _copy_stable_read_only(source, target)
            if copied.get(self.checkpoint.name) != checkpoint_hash:
                raise OrchestrationError("Staged checkpoint SHA-256 does not match the live trainer checkpoint.")
            metadata = {
                "schema_version": 1,
                "stage": stage,
                "global_step": step,
                "created_at": _now(),
                "checkpoint_file": self.checkpoint.name,
                "checkpoint_sha256": checkpoint_hash,
                "artifacts": copied,
                "snapshot_is_read_only": True,
            }
            _write_json_atomic(directory / "snapshot.json", metadata)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return directory / self.checkpoint.name, checkpoint_hash


class BackgroundEvaluator:
    """Run real raw-checkpoint benchmarks concurrently with local training."""

    def __init__(
        self,
        run_root: Path,
        run_id: str,
        *,
        device: str,
        isolation: str,
        allow_host_execution: bool,
        max_parallel: int = 1,
        max_new_tokens: int = 512,
    ):
        self.run_root = _inside_root(run_root)
        self.run_id = run_id
        self.device = device
        self.isolation = isolation
        self.allow_host_execution = allow_host_execution
        self.max_parallel = max_parallel
        self.max_new_tokens = max_new_tokens
        self.jobs: list[EvaluationJob] = []
        self.results: list[dict[str, Any]] = []
        self.active: list[EvaluationJob] = []
        self.reports_dir = self.run_root / "evaluations"
        self.logs_dir = self.run_root / "logs"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _npm_command(self, args: list[str]) -> list[str]:
        if os.name == "nt":
            executable = shutil.which("npm.cmd") or shutil.which("npm")
            if not executable:
                raise OrchestrationError("npm is required for npm run benchmark:model.")
            command_line = subprocess.list2cmdline([executable, *args])
            return ["cmd.exe", "/d", "/s", "/c", command_line]
        executable = shutil.which("npm")
        if not executable:
            raise OrchestrationError("npm is required for npm run benchmark:model.")
        return [executable, *args]

    def _wait_one(self) -> None:
        job = self.active.pop(0)
        return_code = job.process.wait()
        job.log_handle.flush()
        job.log_handle.close()
        record: dict[str, Any] = {
            "stage": job.stage,
            "step": job.step,
            "checkpoint_sha256": job.checkpoint_sha256,
            "checkpoint": str(job.checkpoint_path),
            "report": str(job.report_path),
            "log": str(job.log_path),
            "return_code": return_code,
            "status": "failed",
            "train_loss": None,
            "validation_loss": None,
            "task_completion_rate": None,
            "completed_task_count": None,
            "baseline_valid_task_count": None,
            "suite_version": None,
            "regression_prevention_rate": None,
            "error": None,
        }
        if return_code != 0:
            record["error"] = self._log_tail(job.log_path) or f"benchmark process exited with code {return_code}"
        else:
            try:
                report = _read_json(job.report_path)
                model = report.get("model")
                summary = report.get("summary")
                if report.get("benchmark_track") != "raw-native-model" or report.get("schema_version") != 1:
                    raise OrchestrationError("Benchmark output is not a supported raw-native-model report.")
                if not isinstance(model, dict) or model.get("checkpoint_sha256") != job.checkpoint_sha256:
                    raise OrchestrationError("Benchmark report checkpoint SHA-256 does not match the staged snapshot.")
                if not isinstance(summary, dict):
                    raise OrchestrationError("Benchmark report is missing its measured summary.")
                rate = summary.get("task_completion_rate")
                if rate is not None and (_number(rate) is None or not 0.0 <= float(rate) <= 1.0):
                    raise OrchestrationError("Benchmark report contains an invalid task completion rate.")
                training_loss = report.get("training_loss")
                if not isinstance(training_loss, dict):
                    training_loss = {}
                record.update({
                    "status": "completed",
                    "suite_version": report.get("suite_version"),
                    "train_loss": _number(training_loss.get("train_loss")),
                    "validation_loss": _number(training_loss.get("validation_loss")),
                    "task_completion_rate": None if rate is None else float(rate),
                    "completed_task_count": summary.get("completed_task_count"),
                    "baseline_valid_task_count": summary.get("baseline_valid_task_count"),
                    "regression_prevention_rate": _number(summary.get("regression_prevention_rate")),
                })
            except (OrchestrationError, OSError, ValueError, TypeError) as exc:
                record["error"] = str(exc)
        self.results.append(record)
        print(
            f"[evaluation] {job.stage} step={job.step} status={record['status']} "
            f"completion={_format_metric(record['task_completion_rate'])}",
            flush=True,
        )

    @staticmethod
    def _log_tail(path: Path, maximum_chars: int = 4000) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-maximum_chars:].strip()
        except OSError:
            return ""

    def launch(self, stage: str, step: int, checkpoint: Path, checkpoint_hash: str) -> None:
        while len(self.active) >= self.max_parallel:
            self._wait_one()
        name = f"{stage}-step-{step:09d}-{checkpoint_hash[:12]}"
        report_path = self.reports_dir / (name + ".json")
        log_path = self.logs_dir / (name + ".log")
        if report_path.exists() or log_path.exists():
            raise OrchestrationError("Evaluation output already exists; refusing to overwrite a previous run.")
        command_args = [
            "run", "benchmark:model", "--",
            "--checkpoint", str(checkpoint),
            "--device", self.device,
            "--max-new-tokens", str(self.max_new_tokens),
            "--output", str(report_path),
            "--isolation", self.isolation,
        ]
        if self.isolation == "host":
            if not self.allow_host_execution:
                raise OrchestrationError("Host benchmark execution requires explicit --allow-host-execution.")
            command_args.append("--allow-host-execution")
        command = self._npm_command(command_args)
        handle = log_path.open("xb")
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=dict(os.environ),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            handle.close()
            log_path.unlink(missing_ok=True)
            raise OrchestrationError("Could not start npm run benchmark:model: " + str(exc)) from None
        job = EvaluationJob(
            stage=stage, step=step, checkpoint_sha256=checkpoint_hash,
            checkpoint_path=checkpoint, report_path=report_path, log_path=log_path,
            process=process, log_handle=handle,
        )
        self.jobs.append(job)
        self.active.append(job)
        print(f"[evaluation] started {stage} step={step}; report={report_path}", flush=True)

    def wait_all(self) -> list[dict[str, Any]]:
        while self.active:
            self._wait_one()
        return self.results


def _preflight_benchmark(isolation: str, allow_host_execution: bool) -> dict[str, Any]:
    if isolation == "host" and not allow_host_execution:
        raise OrchestrationError("Host execution is unsandboxed and requires --allow-host-execution.")
    try:
        from tests.benchmark_engine import runner as benchmark_runner
        suite = benchmark_runner.load_suite()
        environments = {}
        for language in sorted({task["language"] for task in suite["tasks"]}):
            environment = benchmark_runner.resolve_environment(language)
            environments[language] = benchmark_runner.prepare_isolation(isolation, environment["image"])
        return {
            "suite_version": suite["version"],
            "task_count": len(suite["tasks"]),
            "languages": environments,
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise OrchestrationError(
            "Benchmark preflight failed before training began. Confirm Docker and all required local images, "
            "or explicitly select host mode with --allow-host-execution. " + str(exc)
        ) from None


def _should_snapshot(stage: str, progress: dict[str, Any], interval: int, seen: set[tuple[str, int]]) -> bool:
    step = progress.get("step")
    if type(step) is not int or step < 1:
        return False
    key = (stage, step)
    if key in seen:
        return False
    message = progress.get("message")
    saved_checkpoint = isinstance(message, str) and (
        "Checkpoint saved" in message or "checkpoint saved" in message
        or "training-step block saved" in message
    )
    completed = progress.get("status") == "completed"
    return (saved_checkpoint and step % interval == 0) or completed


def _training_callback(
    stage: str,
    snapshots: CheckpointSnapshots,
    evaluator: BackgroundEvaluator | None,
    interval: int,
    seen: set[tuple[str, int]],
):
    def callback(progress: dict[str, Any]) -> None:
        if not _should_snapshot(stage, progress, interval, seen):
            return
        step = progress["step"]
        seen.add((stage, step))
        snapshot_path, checkpoint_hash = snapshots.create(stage, step)
        print(
            f"[checkpoint] stage={stage} step={step} sha256={checkpoint_hash} "
            f"train_loss={_format_metric(progress.get('train_loss'))} "
            f"validation_loss={_format_metric(progress.get('validation_loss'))}",
            flush=True,
        )
        if evaluator is not None:
            evaluator.launch(stage, step, snapshot_path, checkpoint_hash)

    return callback


def _ascii_plot(records: list[dict[str, Any]], field: str, title: str, *, height: int = 7) -> str:
    values = [
        (record["step"], _number(record.get(field)))
        for record in records
        if record.get("status") == "completed" and _number(record.get(field)) is not None
    ]
    if not values:
        return f"{title}\nNo verified measurements are available."
    numbers = [value for _, value in values]
    low, high = min(numbers), max(numbers)
    width = max(1, len(values))
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for index, (_, value) in enumerate(values):
        if high == low:
            row = height // 2
        else:
            row = round((high - value) / (high - low) * (height - 1))
        grid[row][index] = "*"
        if index:
            previous = numbers[index - 1]
            if high == low:
                previous_row = height // 2
            else:
                previous_row = round((high - previous) / (high - low) * (height - 1))
            start, end = sorted((previous_row, row))
            for bridge_row in range(start + 1, end):
                if grid[bridge_row][index - 1] == " ":
                    grid[bridge_row][index - 1] = "|"
    lines = [title]
    for row, cells in enumerate(grid):
        label_value = high if height == 1 else high - ((high - low) * row / (height - 1))
        lines.append(f"{label_value:>10.4g} | " + " ".join(cells))
    lines.append("            +-" + "-" * (2 * width - 1))
    lines.append("              " + " ".join(str(step) for step, _ in values))
    lines.append("              x-axis: checkpoint global step")
    return "\n".join(lines)


def write_summary(run_root: Path, run_id: str, records: list[dict[str, Any]], training: dict[str, Any]) -> tuple[Path, Path]:
    run_root = _inside_root(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.md"
    csv_path = run_root / "summary.csv"
    ordered = sorted(records, key=lambda item: (item.get("step", 0), item.get("stage", "")))
    lines = [
        "# SYNTRA Phase 4 Milestone 4 — Local training and evaluation",
        "",
        f"- Run: {run_id}",
        f"- Generated: {_now()}",
        "- Model-quality measurements come only from checkpoint-matched raw native benchmark JSON reports.",
        "- Missing, failed, or unverified measurements are reported as unavailable, never as zero.",
        "",
        "## Checkpoint comparison",
        "",
        "| Stage | Global step | Benchmark | Train loss | Validation loss | Target completion | Verified tasks | Regression prevention |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in ordered:
        tasks = "unavailable"
        if isinstance(record.get("completed_task_count"), int) and isinstance(record.get("baseline_valid_task_count"), int):
            tasks = f"{record['completed_task_count']}/{record['baseline_valid_task_count']}"
        completion = "unavailable"
        if _number(record.get("task_completion_rate")) is not None:
            completion = f"{100 * float(record['task_completion_rate']):.1f}%"
        lines.append(
            "| {stage} | {step} | {status} | {train} | {val} | {completion} | {tasks} | {regression} |".format(
                stage=record.get("stage", "unknown"),
                step=record.get("step", "—"),
                status=record.get("status", "unknown"),
                train=_format_metric(record.get("train_loss")),
                val=_format_metric(record.get("validation_loss")),
                completion=completion,
                tasks=tasks,
                regression=_format_metric(record.get("regression_prevention_rate")),
            )
        )
    lines.extend([
        "",
        "## Loss and task-completion trend",
        "",
        "Each point is one separately staged checkpoint evaluated by the raw native-model suite. Only values present in a checkpoint-matched JSON report are plotted.",
        "",
        _ascii_plot(ordered, "train_loss", "Training loss (lower is better)"),
        "",
        _ascii_plot(ordered, "validation_loss", "Validation loss (lower is better)"),
        "",
        _ascii_plot(ordered, "task_completion_rate", "Held-out target completion rate (higher is better)"),
        "",
        "## Training execution",
        "",
        f"- Pretraining result: {training.get('pretraining_status', 'not run')}",
        f"- SFT result: {training.get('sft_status', 'not run')}",
        f"- Curriculum dataset: {training.get('curriculum_dataset', 'not generated')}",
        f"- Checkpoint: {training.get('checkpoint', 'not available')}",
        f"- Benchmark reports: {len([item for item in ordered if item.get('status') == 'completed'])} verified, "
        f"{len([item for item in ordered if item.get('status') != 'completed'])} failed or unverified",
        "",
        "Training loss is not treated as a substitute for coding-task success. This controlled suite is not proof of general coding ability.",
        "",
    ])
    temporary = summary_path.with_name(summary_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, summary_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    columns = [
        "stage", "step", "status", "checkpoint_sha256", "train_loss", "validation_loss",
        "task_completion_rate", "completed_task_count", "baseline_valid_task_count",
        "regression_prevention_rate", "suite_version", "report", "log", "error",
    ]
    temporary_csv = csv_path.with_name(csv_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in ordered:
                writer.writerow({key: record.get(key) for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, csv_path)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()
    return summary_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local SYNTRA pretraining, AST-derived curriculum fine-tuning, and checkpoint benchmarks."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Local JSON profile; no external configuration is fetched.")
    parser.add_argument("--corpus-root", help="Approved local source directory. Defaults to the SYNTRA repository.")
    parser.add_argument("--checkpoint", help="Local staging checkpoint path; defaults to the profile.")
    parser.add_argument("--curriculum-output", help="Approved local JSONL destination; defaults to the profile.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Training device; auto selects an available supported backend.")
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-pretrain-steps", type=int)
    parser.add_argument("--sft-steps", type=int)
    parser.add_argument("--evaluation-every", type=int)
    parser.add_argument("--isolation", choices=("docker", "host"))
    parser.add_argument("--allow-host-execution", action="store_true", help="Acknowledge that host-mode generated-code evaluation is not sandboxed.")
    parser.add_argument("--resume-pretraining", action="store_true", help="Resume only when source fingerprint and saved training settings match.")
    parser.add_argument("--overwrite-checkpoint", action="store_true", help="Back up an existing checkpoint before starting a fresh pretraining run.")
    parser.add_argument("--skip-pretraining", action="store_true", help="Use an existing local checkpoint and run curriculum fine-tuning only.")
    parser.add_argument("--skip-sft", action="store_true", help="Run repository source pretraining only.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Do not run model evaluations; training still writes real checkpoints and loss metrics.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile_path = _resolve_profile_path(args.config)
        profile = load_profile(profile_path)
        _validate_profile_values(profile)
        pre = profile["pretraining"]
        cur = profile["curriculum"]
        sft = profile["sft"]
        evaluation = profile["evaluation"]

        corpus_value = args.corpus_root or profile.get("corpus_root", ".")
        corpus_candidate = Path(corpus_value).expanduser()
        if not corpus_candidate.is_absolute():
            corpus_candidate = ROOT / corpus_candidate
        from syntra_ai.safe_paths import resolve_corpus_root
        corpus_root = resolve_corpus_root(corpus_candidate)

        checkpoint_value = args.checkpoint or profile.get("checkpoint")
        if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
            raise OrchestrationError("A checkpoint path is required in the profile or command line.")
        checkpoint = _resolve_profile_path(checkpoint_value)
        default_production_checkpoint = (ROOT / "data" / "checkpoints" / "syntra.pt").resolve()
        if checkpoint == default_production_checkpoint:
            raise OrchestrationError("Use a separate staging checkpoint; this workflow will not overwrite the default production checkpoint.")
        if checkpoint.is_symlink():
            raise OrchestrationError("The selected checkpoint path must not be a symbolic link.")

        curriculum_value = args.curriculum_output or cur.get("dataset")
        if not isinstance(curriculum_value, str) or not curriculum_value.strip():
            raise OrchestrationError("A local curriculum JSONL destination is required.")
        curriculum_path = _resolve_profile_path(curriculum_value)

        max_pretrain_steps = pre["max_steps"] if args.max_pretrain_steps is None else args.max_pretrain_steps
        sft_steps = sft["steps"] if args.sft_steps is None else args.sft_steps
        evaluation_every = evaluation["every_steps"] if args.evaluation_every is None else args.evaluation_every
        if type(max_pretrain_steps) is not int or not 1 <= max_pretrain_steps <= 1_000_000:
            raise OrchestrationError("--max-pretrain-steps must be between 1 and 1,000,000.")
        if type(sft_steps) is not int or not 1 <= sft_steps <= 1_000_000:
            raise OrchestrationError("--sft-steps must be between 1 and 1,000,000.")
        if type(evaluation_every) is not int or not 1 <= evaluation_every <= 100_000:
            raise OrchestrationError("--evaluation-every must be between 1 and 100,000.")
        training_device = args.device or profile.get("device", "auto")
        evaluation_device = args.evaluation_device or evaluation["device"]
        isolation = args.isolation or evaluation["isolation"]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_root = _inside_root(ROOT / profile.get("run_directory", "data/benchmarks/phase4-milestone4") / run_id)
        run_root.mkdir(parents=True, exist_ok=False)
        snapshots = CheckpointSnapshots(checkpoint, run_root / "snapshots", run_id)

        benchmark_preflight: dict[str, Any] | None = None
        evaluator: BackgroundEvaluator | None = None
        if not args.skip_benchmark:
            if args.skip_pretraining and not checkpoint.is_file():
                raise OrchestrationError("--skip-pretraining requires an existing native checkpoint.")
            benchmark_preflight = _preflight_benchmark(isolation, args.allow_host_execution)
            evaluator = BackgroundEvaluator(
                run_root, run_id, device=evaluation_device,
                isolation=isolation, allow_host_execution=args.allow_host_execution,
                max_parallel=evaluation["max_parallel"], max_new_tokens=evaluation["max_new_tokens"],
            )
        elif isolation == "host" and not args.allow_host_execution:
            raise OrchestrationError("Host execution is unsandboxed and requires --allow-host-execution.")

        if args.resume_pretraining and args.skip_pretraining:
            raise OrchestrationError("--resume-pretraining and --skip-pretraining cannot be used together.")
        if args.resume_pretraining and not checkpoint.is_file():
            raise OrchestrationError("--resume-pretraining requires an existing checkpoint.")
        if not args.skip_pretraining and checkpoint.exists() and not args.resume_pretraining and not args.overwrite_checkpoint:
            raise OrchestrationError("Checkpoint already exists. Use --resume-pretraining for an exact resume or --overwrite-checkpoint to preserve a backup and start fresh.")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": _now(),
            "profile": str(profile_path),
            "corpus_root": str(corpus_root),
            "checkpoint": str(checkpoint),
            "curriculum_dataset": str(curriculum_path),
            "training_device_requested": training_device,
            "evaluation_device": evaluation_device,
            "isolation": isolation,
            "benchmark_preflight": benchmark_preflight,
            "settings": profile,
        }
        _write_json_atomic(run_root / "run_manifest.json", manifest)
        seen: set[tuple[str, int]] = set()
        training_results: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "curriculum_dataset": str(curriculum_path),
            "pretraining_status": "skipped",
            "sft_status": "skipped",
        }

        if not args.skip_sft:
            from syntra_ai.curriculum import synthesize_repository_curriculum
            curriculum_result = synthesize_repository_curriculum(
                root=corpus_root,
                output_path=curriculum_path,
                max_examples=cur["max_examples"],
            )
            if not isinstance(curriculum_result, dict) or not curriculum_result.get("dataset_path"):
                raise OrchestrationError("Curriculum synthesis did not return a verified output dataset.")
            if Path(curriculum_result["dataset_path"]).resolve() != curriculum_path.resolve():
                raise OrchestrationError("Curriculum synthesis output path differs from the requested local dataset path.")
            manifest["curriculum_result"] = curriculum_result
            _write_json_atomic(run_root / "run_manifest.json", manifest)

        if not args.skip_pretraining:
            if args.overwrite_checkpoint and checkpoint.exists() and not args.resume_pretraining:
                backup_dir = run_root / "pretraining-backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                _copy_stable_read_only(checkpoint, backup_dir / checkpoint.name)
                old_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
                if old_sidecar.is_file() and not old_sidecar.is_symlink():
                    _copy_stable_read_only(old_sidecar, backup_dir / old_sidecar.name)
                try:
                    from syntra_ai.training import inference_checkpoint_path
                    old_companion = inference_checkpoint_path(checkpoint)
                except (ImportError, OSError, ValueError):
                    old_companion = None
                if old_companion is not None and old_companion.is_file() and not old_companion.is_symlink():
                    _copy_stable_read_only(old_companion, backup_dir / old_companion.name)

            from brain.model import TransformerConfig, select_device
            from brain.train import train as pretrain
            config = TransformerConfig(
                vocab_size=pre["vocab_size"],
                block_size=pre["block_size"],
                n_layer=pre["layers"],
                n_head=pre["heads"],
                n_embd=pre["width"],
                dropout=pre["dropout"],
            )
            config.validate()
            selected_device = select_device(training_device)
            print(f"[pretraining] selected device: {selected_device}", flush=True)
            pretraining_result = pretrain(
                [str(corpus_root)],
                checkpoint_path=checkpoint,
                config=config,
                epochs=1,
                max_steps=max_pretrain_steps,
                batch_size=pre["batch_size"],
                learning_rate=pre["learning_rate"],
                minimum_learning_rate=pre["minimum_learning_rate"],
                warmup_steps=pre["warmup_steps"],
                decay_steps=pre["decay_steps"],
                weight_decay=pre["weight_decay"],
                tokenizer_vocab_size=pre["vocab_size"],
                tokenizer_bytes=pre["tokenizer_bytes"],
                max_dataset_bytes=pre["max_dataset_bytes"],
                evaluation_batches=pre["evaluation_batches"],
                save_every_steps=evaluation_every,
                device_name=training_device,
                seed=pre["seed"],
                resume=args.resume_pretraining,
                callback=_training_callback("pretraining", snapshots, evaluator, evaluation_every, seen),
            )
            training_results["pretraining_status"] = pretraining_result.get("status", "unknown")
            training_results["pretraining_result"] = pretraining_result
            print(
                f"[pretraining] status={pretraining_result.get('status')} "
                f"step={pretraining_result.get('step')} "
                f"train_loss={_format_metric(pretraining_result.get('train_loss'))} "
                f"validation_loss={_format_metric(pretraining_result.get('validation_loss'))}",
                flush=True,
            )

        if not args.skip_sft:
            if not checkpoint.is_file():
                raise OrchestrationError("Pretraining checkpoint is missing; instruction fine-tuning cannot start.")
            from syntra_ai.instruction import audit_instruction_dataset, finetune
            from syntra_ai.runtime import _read_checkpoint
            _, _, checkpoint_config, checkpoint_tokenizer = _read_checkpoint(checkpoint)
            audit = audit_instruction_dataset(
                curriculum_path,
                seq_len=checkpoint_config.seq_len,
                tokenizer=checkpoint_tokenizer,
                max_examples=cur["max_examples"],
                max_tokens=cur["max_tokens"],
            )
            training_results["curriculum_audit"] = audit
            if audit.get("valid") is not True or audit.get("split_available") is not True:
                raise OrchestrationError(
                    "The synthesized curriculum failed checkpoint-tokenizer audit or does not contain an independent validation split: "
                    + str(audit.get("error") or audit.get("warnings") or "split unavailable")
                )
            print(
                f"[sft] audited {audit.get('example_count')} source records and "
                f"{audit.get('context_window_count')} checkpoint-tokenized contexts.",
                flush=True,
            )
            sft_result = finetune(
                dataset_path=curriculum_path,
                steps=sft_steps,
                batch_size=sft["batch_size"],
                learning_rate=sft["learning_rate"],
                device_name=training_device,
                checkpoint=checkpoint,
                callback=_training_callback("sft", snapshots, evaluator, evaluation_every, seen),
            )
            training_results["sft_status"] = sft_result.get("status", "unknown")
            training_results["sft_result"] = sft_result
            print(
                f"[sft] status={sft_result.get('status')} step={sft_result.get('step')} "
                f"train_loss={_format_metric(sft_result.get('train_loss'))} "
                f"validation_loss={_format_metric(sft_result.get('validation_loss'))}",
                flush=True,
            )

        evaluation_results = evaluator.wait_all() if evaluator is not None else []
        training_results["evaluation_count"] = len(evaluation_results)
        training_results["evaluation_failures"] = sum(item.get("status") != "completed" for item in evaluation_results)
        training_results["finished_at"] = _now()
        _write_json_atomic(run_root / "evaluation_results.json", {"schema_version": 1, "run_id": run_id, "results": evaluation_results})
        summary_path, csv_path = write_summary(run_root, run_id, evaluation_results, training_results)
        print(f"[summary] {summary_path}", flush=True)
        print(f"[summary] {csv_path}", flush=True)
        if training_results["evaluation_failures"]:
            print("[result] Training completed, but one or more checkpoint evaluations failed or were unverified.", file=sys.stderr)
            return 1
        return 0
    except (OrchestrationError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print("SYNTRA Phase 4 orchestration failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

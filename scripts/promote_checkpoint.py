"""Verify raw benchmark provenance before promoting a local SYNTRA checkpoint.

Promotion is a local, atomic file operation. No network, package, model-provider,
or remote inference call is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARK_ROOT = ROOT / "data" / "benchmarks"
ARCHIVE_ROOT = ROOT / "data" / "checkpoints" / "archive"
ARCHIVE_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$")


class PromotionError(RuntimeError):
    """A report, checkpoint, archive, or promotion integrity failure."""


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _safe_repo_path(path: str | Path, *, must_exist: bool = False) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = ROOT / raw
    lexical = Path(os.path.abspath(raw))
    root = ROOT.resolve()
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        raise PromotionError("Promotion paths must remain inside the SYNTRA repository.") from None
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise PromotionError("Symbolic-link path components are not allowed in promotion paths.")
    resolved = lexical.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PromotionError("Promotion path resolves outside the SYNTRA repository.") from None
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    if path.is_symlink():
        raise PromotionError("Refusing to hash a symbolic-link artifact: " + str(path))
    try:
        before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise PromotionError("Cannot inspect artifact " + str(path) + ": " + str(exc)) from None
    if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
        raise PromotionError("Artifact must be a non-empty regular file: " + str(path))
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise PromotionError("Artifact changed while opening: " + str(path))
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise PromotionError("Cannot hash artifact " + str(path) + ": " + str(exc)) from None
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise PromotionError("Artifact changed while being hashed: " + str(path))
    return digest.hexdigest()


def _read_json(path: Path, maximum_bytes: int = 10 * 1024 * 1024) -> dict[str, Any]:
    path = _safe_repo_path(path, must_exist=True)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise PromotionError("Expected a bounded regular JSON file: " + str(path))

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError("Invalid JSON constant: " + item)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PromotionError("Cannot read valid JSON from " + str(path) + ": " + str(exc)) from None
    if not isinstance(value, dict):
        raise PromotionError("Expected a JSON object in " + str(path))
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = _safe_repo_path(path)
    if path.is_symlink():
        raise PromotionError("Refusing to replace symbolic-link JSON output.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    source = _safe_repo_path(source, must_exist=True)
    destination = _safe_repo_path(destination)
    if _sha256(source) != expected_sha256:
        raise PromotionError("Source checkpoint SHA-256 changed after verification.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            source_stat = os.fstat(input_handle.fileno())
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            current_source = os.fstat(input_handle.fileno())
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ctime_ns) != (
            current_source.st_dev, current_source.st_ino, current_source.st_size,
            current_source.st_mtime_ns, current_source.st_ctime_ns
        ):
            raise PromotionError("Source checkpoint changed during promotion copy.")
        if _sha256(temporary) != expected_sha256:
            raise PromotionError("Copied checkpoint SHA-256 does not match its verified source.")
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PromotionError("Benchmark report is missing " + label + ".")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise PromotionError("Benchmark report contains an invalid " + label + ".") from None


def verify_benchmark_candidate(report_path: str | Path) -> dict[str, Any]:
    """Verify a completed raw benchmark and its immutable snapshot manifest."""
    report_file = _safe_repo_path(report_path, must_exist=True)
    if not _is_within(report_file, BENCHMARK_ROOT.resolve()):
        raise PromotionError("The selected benchmark report must be stored under data/benchmarks/.")
    report = _read_json(report_file)
    if report.get("schema_version") != 1 or report.get("benchmark_track") != "raw-native-model":
        raise PromotionError("Only schema-v1 raw-native-model benchmark reports can authorize promotion.")
    if not isinstance(report.get("run_id"), str) or not report["run_id"].strip():
        raise PromotionError("Benchmark report is missing its run ID.")
    _parse_timestamp(report.get("started_at"), "started_at")
    _parse_timestamp(report.get("completed_at"), "completed_at")

    model = report.get("model")
    summary = report.get("summary")
    tasks = report.get("tasks")
    if not isinstance(model, dict) or not isinstance(summary, dict) or not isinstance(tasks, list):
        raise PromotionError("Benchmark report is missing model, summary, or task evidence.")
    checkpoint_value = model.get("checkpoint")
    expected_hash = model.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise PromotionError("Benchmark report does not identify its checkpoint file.")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise PromotionError("Benchmark report does not contain a valid full SHA-256 checkpoint identity.")
    checkpoint = _safe_repo_path(checkpoint_value, must_exist=True)
    if not _is_within(checkpoint, BENCHMARK_ROOT.resolve()):
        raise PromotionError("Only immutable checkpoint snapshots under data/benchmarks/ can be promoted.")
    if _sha256(checkpoint) != expected_hash:
        raise PromotionError("Staged checkpoint SHA-256 does not match the completed benchmark report.")

    task_count = summary.get("task_count")
    valid_count = summary.get("baseline_valid_task_count")
    invalid_count = summary.get("invalid_fixture_count")
    completed_count = summary.get("completed_task_count")
    rate = summary.get("task_completion_rate")
    if type(task_count) is not int or task_count < 1 or task_count != len(tasks):
        raise PromotionError("Benchmark task count does not match the measured task evidence.")
    if type(valid_count) is not int or valid_count != task_count:
        raise PromotionError("Benchmark has invalid or unmeasured hidden baseline fixtures.")
    if type(invalid_count) is not int or invalid_count != 0:
        raise PromotionError("Benchmark contains invalid hidden baseline fixtures.")
    if type(completed_count) is not int or not 0 <= completed_count <= valid_count:
        raise PromotionError("Benchmark completion count is invalid.")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)) or not 0 <= float(rate) <= 1:
        raise PromotionError("Benchmark task-completion rate is unavailable or invalid.")
    if abs(float(rate) - (completed_count / valid_count)) > 0.00011:
        raise PromotionError("Benchmark completion rate is inconsistent with measured task counts.")
    task_ids = [item.get("id") for item in tasks if isinstance(item, dict)]
    if len(task_ids) != task_count or len(set(task_ids)) != task_count or any(not isinstance(item, str) or not item for item in task_ids):
        raise PromotionError("Benchmark task evidence is incomplete or contains duplicate task IDs.")
    if any(item.get("baseline_validated") is not True for item in tasks if isinstance(item, dict)):
        raise PromotionError("Every benchmark task must pass its hidden baseline validation before promotion.")
    measured_completed = sum(
        1 for item in tasks
        if isinstance(item, dict) and item.get("task_completion") is True
    )
    if measured_completed != completed_count:
        raise PromotionError("Benchmark completion count does not match per-task completion evidence.")
    fully_verified_count = summary.get("fully_verified_task_count")
    if type(fully_verified_count) is not int or not 0 <= fully_verified_count <= valid_count:
        raise PromotionError("Benchmark fully-verified task count is invalid.")
    measured_fully_verified = sum(
        1 for item in tasks
        if isinstance(item, dict)
        and item.get("task_completion") is True
        and item.get("regression_prevention") is True
    )
    if measured_fully_verified != fully_verified_count:
        raise PromotionError("Fully-verified task count does not match target and regression-test evidence.")
    suite = _read_json(ROOT / "tests" / "benchmark_engine" / "suite.v1.json", maximum_bytes=4 * 1024 * 1024)
    suite_tasks = suite.get("tasks")
    expected_task_ids = [
        item.get("id") for item in suite_tasks if isinstance(item, dict)
    ] if isinstance(suite_tasks, list) else []
    if (
        report.get("suite_version") != suite.get("version")
        or len(expected_task_ids) != task_count
        or set(task_ids) != set(expected_task_ids)
    ):
        raise PromotionError("Promotion requires a complete benchmark from the currently checked-in 13-task polyglot suite, not a partial or stale suite run.")

    snapshot_path = _safe_repo_path(checkpoint.parent / "snapshot.json", must_exist=True)
    snapshot = _read_json(snapshot_path, maximum_bytes=1024 * 1024)
    artifacts = snapshot.get("artifacts")
    if (
        snapshot.get("schema_version") != 1
        or snapshot.get("snapshot_is_read_only") is not True
        or snapshot.get("checkpoint_file") != checkpoint.name
        or snapshot.get("checkpoint_sha256") != expected_hash
        or not isinstance(artifacts, dict)
        or artifacts.get(checkpoint.name) != expected_hash
    ):
        raise PromotionError("Checkpoint snapshot manifest does not authenticate this staging artifact.")
    if snapshot.get("stage") not in {"pretraining", "sft", "baseline"}:
        raise PromotionError("Checkpoint snapshot stage is not recognized.")
    snapshot_step = snapshot.get("global_step")
    if type(snapshot_step) is not int or snapshot_step < 0:
        raise PromotionError("Checkpoint snapshot does not contain a valid global step.")
    if model.get("trained_steps") != snapshot_step:
        raise PromotionError("Benchmark report trained_steps does not match the checkpoint snapshot global step.")
    expected_snapshot_directory = f"{snapshot['stage']}-step-{snapshot_step:09d}-{expected_hash[:12]}"
    if checkpoint.parent.name != expected_snapshot_directory:
        raise PromotionError("Checkpoint snapshot directory name does not match its stage, step, and digest.")

    report_relative = report_file.relative_to(BENCHMARK_ROOT.resolve())
    checkpoint_relative = checkpoint.relative_to(BENCHMARK_ROOT.resolve())
    if (
        len(report_relative.parts) != 4
        or report_relative.parts[2] != "evaluations"
        or len(checkpoint_relative.parts) != 6
        or checkpoint_relative.parts[2] != "snapshots"
        or checkpoint_relative.parts[3] != report_relative.parts[1]
        or checkpoint_relative.parts[1] != report_relative.parts[1]
        or checkpoint.parents[2].name != "snapshots"
        or checkpoint.parents[3] != report_file.parent.parent
    ):
        raise PromotionError("Benchmark report and staged checkpoint are not from the same structured evaluation run.")
    isolation = report.get("isolation")
    if not isinstance(isolation, dict) or isolation.get("mode") != "docker":
        raise PromotionError("Checkpoint promotion requires a completed network-disabled Docker benchmark, not host execution.")
    return {
        "report": report_file,
        "checkpoint": checkpoint,
        "checkpoint_sha256": expected_hash,
        "run_id": report["run_id"],
        "suite_version": report.get("suite_version"),
        "stage": snapshot["stage"],
        "step": snapshot.get("global_step"),
        "task_completion_rate": float(rate),
        "completed_task_count": completed_count,
        "task_completion_rate": float(rate),
        "fully_verified_task_count": fully_verified_count,
        "fully_verified_task_rate": fully_verified_count / valid_count,
        "task_count": task_count,
        "fully_verified_by_id": {
            item["id"]: item["task_completion"] is True and item["regression_prevention"] is True
            for item in tasks
        },
        "snapshot": snapshot_path,
    }


def verify_paired_benchmark(candidate: dict[str, Any], baseline_report_path: str | Path) -> dict[str, Any]:
    """Require a like-for-like SFT gain with no per-task regression."""
    baseline = verify_benchmark_candidate(baseline_report_path)
    if candidate.get("stage") != "sft":
        raise PromotionError("Paired improvement gating is only valid for an SFT candidate checkpoint.")
    if baseline.get("stage") not in {"pretraining", "baseline"}:
        raise PromotionError("The comparison report must identify a pretraining or baseline checkpoint.")
    if candidate.get("suite_version") != baseline.get("suite_version"):
        raise PromotionError("Candidate and baseline benchmark suite versions differ.")
    if candidate["report"].parent != baseline["report"].parent:
        raise PromotionError("Candidate and pretraining baseline reports must belong to the same orchestrated training run.")
    candidate_tasks = candidate.get("fully_verified_by_id")
    baseline_tasks = baseline.get("fully_verified_by_id")
    if not isinstance(candidate_tasks, dict) or not isinstance(baseline_tasks, dict):
        raise PromotionError("Candidate and baseline reports must contain complete per-task results.")
    if set(candidate_tasks) != set(baseline_tasks):
        raise PromotionError("Candidate and baseline must measure the exact same task IDs.")
    regressions = sorted(
        task_id for task_id, baseline_passed in baseline_tasks.items()
        if baseline_passed and not candidate_tasks[task_id]
    )
    if regressions:
        raise PromotionError("SFT candidate regresses on baseline-passing tasks: " + ", ".join(regressions))
    improvements = sorted(
        task_id for task_id, candidate_passed in candidate_tasks.items()
        if candidate_passed and not baseline_tasks[task_id]
    )
    if not improvements or candidate["fully_verified_task_rate"] <= baseline["fully_verified_task_rate"]:
        raise PromotionError("SFT candidate must improve fully-verified task completion and pass at least one additional fully-verified task.")
    return {
        "baseline_report": str(baseline["report"]),
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "baseline_completed_task_count": baseline["completed_task_count"],
        "baseline_task_completion_rate": baseline["task_completion_rate"],
        "candidate_completed_task_count": candidate["completed_task_count"],
        "candidate_task_completion_rate": candidate["task_completion_rate"],
        "baseline_fully_verified_task_count": baseline["fully_verified_task_count"],
        "candidate_fully_verified_task_count": candidate["fully_verified_task_count"],
        "baseline_fully_verified_task_rate": baseline["fully_verified_task_rate"],
        "candidate_fully_verified_task_rate": candidate["fully_verified_task_rate"],
        "improved_task_ids": improvements,
        "regressed_task_ids": regressions,
        "fully_verified_task_delta": candidate["fully_verified_task_rate"] - baseline["fully_verified_task_rate"],
    }


def _active_artifact_paths(active: Path) -> list[tuple[str, Path]]:
    from syntra_ai.training import inference_checkpoint_path
    return [
        ("checkpoint", active),
        ("metadata", active.with_suffix(active.suffix + ".json")),
        ("inference_companion", inference_checkpoint_path(active)),
    ]


def _archive_active(active: Path) -> str | None:
    artifacts = []
    for kind, path in _active_artifact_paths(active):
        if path.is_symlink():
            raise PromotionError("Active checkpoint artifact must not be a symbolic link: " + str(path))
        if path.exists():
            artifacts.append((kind, path))
    if not artifacts:
        return None
    if not any(kind == "checkpoint" for kind, _path in artifacts):
        raise PromotionError("Orphaned active metadata or inference companion exists without its checkpoint; inspect it before promoting.")
    for _kind, path in artifacts:
        _safe_repo_path(path, must_exist=True)
        if not path.is_file():
            raise PromotionError("Active checkpoint artifact must be a regular file: " + str(path))
    archive_id = _now_id()
    archive_dir = _safe_repo_path(ARCHIVE_ROOT / archive_id)
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: dict[str, dict[str, str]] = {}
    try:
        for kind, source in artifacts:
            digest = _sha256(source)
            target = archive_dir / source.name
            _atomic_copy_verified(source, target, digest)
            if _sha256(target) != digest:
                raise PromotionError("Previous active checkpoint archive failed SHA-256 verification.")
            entries[kind] = {"file": target.name, "sha256": digest}
        _write_json_atomic(archive_dir / "archive.json", {
            "schema_version": 1,
            "archive_id": archive_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "active_checkpoint": str(active),
            "artifacts": entries,
        })
    except Exception:
        shutil.rmtree(archive_dir, ignore_errors=True)
        raise
    return archive_id


def _restore_archive(active: Path, archive_id: str, *, archive_current: bool = True) -> dict[str, Any]:
    if ARCHIVE_ID_RE.fullmatch(archive_id) is None:
        raise PromotionError("Rollback requires a valid archive ID returned by brain:promote.")
    active = _safe_repo_path(active)
    archive_dir = _safe_repo_path(ARCHIVE_ROOT / archive_id, must_exist=True)
    manifest = _read_json(archive_dir / "archive.json", maximum_bytes=1024 * 1024)
    if manifest.get("schema_version") != 1 or manifest.get("archive_id") != archive_id:
        raise PromotionError("Rollback archive manifest is invalid.")
    archived_active = manifest.get("active_checkpoint")
    if not isinstance(archived_active, str) or _safe_repo_path(archived_active) != active:
        raise PromotionError("Rollback archive was created for a different active checkpoint path.")
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict) or "checkpoint" not in entries:
        raise PromotionError("Rollback archive has no previous active checkpoint.")

    # Verify the complete archive before touching the current active files.
    verified_sources: dict[str, tuple[Path, str]] = {}
    for kind in ("checkpoint", "metadata", "inference_companion"):
        entry = entries.get(kind)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise PromotionError("Rollback archive contains a malformed artifact entry.")
        filename = entry.get("file")
        digest = entry.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename or not isinstance(digest, str):
            raise PromotionError("Rollback archive contains an unsafe artifact entry.")
        source = _safe_repo_path(archive_dir / filename, must_exist=True)
        if _sha256(source) != digest:
            raise PromotionError("Rollback archive SHA-256 verification failed for " + filename)
        verified_sources[kind] = (source, digest)
    if "checkpoint" not in verified_sources:
        raise PromotionError("Rollback archive is missing the verified checkpoint payload.")

    current_archive = _archive_active(active) if archive_current else None
    destinations = dict(_active_artifact_paths(active))
    try:
        # The inference companion embeds source_checkpoint_mtime_ns. Verify its
        # archived bytes above, but rebuild it against the restored checkpoint's
        # new filesystem timestamp rather than restoring a necessarily stale binding.
        for kind in ("checkpoint", "metadata"):
            destination = destinations[kind]
            verified = verified_sources.get(kind)
            if verified is None:
                if destination.exists() or destination.is_symlink():
                    if destination.is_dir() and not destination.is_symlink():
                        raise PromotionError("Refusing to remove a directory at an active checkpoint artifact path.")
                    destination.unlink()
                continue
            source, digest = verified
            _atomic_copy_verified(source, destination, digest)
        companion_destination = destinations["inference_companion"]
        if companion_destination.exists() or companion_destination.is_symlink():
            companion_destination.unlink()
        from syntra_ai import runtime
        from syntra_ai.training import _save_inference_checkpoint
        _target, payload, _config, _tokenizer = runtime._read_checkpoint(active)
        _save_inference_checkpoint(active, payload)
        runtime._get_resident_bundle(path=active, device_name="cpu")
    except Exception as exc:
        if current_archive:
            try:
                _restore_archive(active, current_archive, archive_current=False)
            except Exception as restore_error:
                raise PromotionError(
                    "Rollback failed and automatic restoration of the previous active files also failed; "
                    + str(exc) + "; recovery error: " + str(restore_error)
                    + ". The safety archive is " + current_archive
                ) from None
        if isinstance(exc, PromotionError):
            raise
        raise PromotionError("Rollback failed; previous active files were restored where possible: " + str(exc)) from None
    return {
        "rollback_archive_id": archive_id,
        "archive_created_for_replaced_checkpoint": current_archive,
        "active_checkpoint": str(active),
    }


def _validate_runtime_checkpoint(path: Path) -> None:
    try:
        from syntra_ai import runtime
        runtime._get_resident_bundle(path=path, device_name="cpu")
    except Exception as exc:
        raise PromotionError("SYNTRA runtime rejected the staged checkpoint: " + str(exc)) from None


def promote(
    report_path: str | Path,
    active_checkpoint: str | Path | None = None,
    *,
    dry_run: bool = False,
    confirm_sha256: str | None = None,
    baseline_report_path: str | Path | None = None,
    allow_initial_baseline: bool = False,
) -> dict[str, Any]:
    candidate = verify_benchmark_candidate(report_path)
    if not dry_run and confirm_sha256 != candidate["checkpoint_sha256"]:
        raise PromotionError(
            "Promotion requires --confirm-sha256 with the exact verified checkpoint digest: "
            + candidate["checkpoint_sha256"]
        )
    from syntra_ai import runtime
    active = _safe_repo_path(active_checkpoint or runtime.checkpoint_path())
    comparison: dict[str, Any] | None = None
    if candidate["stage"] == "sft":
        if allow_initial_baseline:
            raise PromotionError("--allow-initial-baseline cannot be used for an SFT checkpoint.")
        if baseline_report_path is None:
            raise PromotionError("SFT promotion requires --baseline-report with the matching pretraining benchmark.")
        comparison = verify_paired_benchmark(candidate, baseline_report_path)
    else:
        if candidate["stage"] not in {"pretraining", "baseline"}:
            raise PromotionError("Only pretraining baselines or SFT checkpoints can be promoted.")
        if not allow_initial_baseline:
            raise PromotionError("Initial pretraining promotion requires explicit --allow-initial-baseline acknowledgement.")
        if baseline_report_path is not None:
            raise PromotionError("--baseline-report is only valid when promoting an SFT candidate.")
        if active.exists() or any(path.exists() for _kind, path in _active_artifact_paths(active)[1:]):
            raise PromotionError("An initial baseline can only be promoted when no active checkpoint or companion artifacts exist.")
        if candidate["fully_verified_task_count"] < 1:
            raise PromotionError("An initial baseline must fully verify at least one task, including its regression tests.")
    if active == candidate["checkpoint"]:
        raise PromotionError("The evaluated staging checkpoint is already the active checkpoint.")
    if active.is_symlink():
        raise PromotionError("Active checkpoint destination must not be a symbolic link.")
    _validate_runtime_checkpoint(candidate["checkpoint"])
    if dry_run:
        return {**candidate, "active_checkpoint": active, "dry_run": True, "archive_id": None, "comparison": comparison}

    metadata_source = candidate["checkpoint"].with_suffix(candidate["checkpoint"].suffix + ".json")
    expected_metadata_hash = None
    snapshot = _read_json(candidate["snapshot"], maximum_bytes=1024 * 1024)
    artifact_hashes = snapshot["artifacts"]
    if metadata_source.is_symlink():
        raise PromotionError("Staged checkpoint metadata must not be a symbolic link.")
    if metadata_source.is_file():
        expected_metadata_hash = artifact_hashes.get(metadata_source.name)
        if expected_metadata_hash != _sha256(metadata_source):
            raise PromotionError("Staged checkpoint metadata is not covered by its snapshot manifest.")
    archive_id = _archive_active(active)

    companion = _active_artifact_paths(active)[2][1]
    try:
        _atomic_copy_verified(candidate["checkpoint"], active, candidate["checkpoint_sha256"])
        active_metadata = _active_artifact_paths(active)[1][1]
        if expected_metadata_hash:
            _atomic_copy_verified(metadata_source, active_metadata, expected_metadata_hash)
        elif active_metadata.exists():
            active_metadata.unlink()
        if companion.exists():
            companion.unlink()
        # Rebuild a correctly bound companion against the new active checkpoint mtime.
        from syntra_ai import runtime
        from syntra_ai.training import _save_inference_checkpoint
        _target, payload, _config, _tokenizer = runtime._read_checkpoint(active)
        _save_inference_checkpoint(active, payload)
        _validate_runtime_checkpoint(active)
    except Exception as exc:
        if archive_id:
            _restore_archive(active, archive_id, archive_current=False)
        else:
            for _kind, path in _active_artifact_paths(active):
                if path.exists() and not path.is_symlink():
                    path.unlink()
        if isinstance(exc, PromotionError):
            raise
        raise PromotionError("Promotion failed; previous active checkpoint was restored where one existed: " + str(exc)) from None

    return {
        "promoted": True,
        "active_checkpoint": str(active),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "benchmark_report": str(candidate["report"]),
        "benchmark_run_id": candidate["run_id"],
        "task_completion_rate": candidate["task_completion_rate"],
        "fully_verified_task_count": candidate["fully_verified_task_count"],
        "comparison": comparison,
        "completed_task_count": candidate["completed_task_count"],
        "task_count": candidate["task_count"],
        "archive_id": archive_id,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and promote a benchmarked SYNTRA checkpoint, or roll back an archive.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", help="Completed raw-native-model benchmark JSON stored under data/benchmarks/.")
    group.add_argument("--rollback", metavar="ARCHIVE_ID", help="Restore a previous active checkpoint from data/checkpoints/archive/.")
    parser.add_argument("--active-checkpoint", help="Optional active checkpoint override; it must remain inside the repository.")
    parser.add_argument("--dry-run", action="store_true", help="Validate benchmark provenance and runtime loadability without changing active files.")
    parser.add_argument("--confirm-sha256", help="Required for promotion: explicitly acknowledge the exact verified staging checkpoint SHA-256 printed by --dry-run.")
    parser.add_argument("--baseline-report", help="Required for SFT promotion: completed pretraining benchmark JSON from the same task suite.")
    parser.add_argument("--allow-initial-baseline", action="store_true", help="Explicitly allow the first pretraining checkpoint only when no active checkpoint exists and at least one task passes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from syntra_ai import runtime
        active = _safe_repo_path(args.active_checkpoint or runtime.checkpoint_path())
        if args.rollback:
            result = _restore_archive(active, args.rollback)
        else:
            result = promote(
                args.report,
                active_checkpoint=active,
                dry_run=args.dry_run,
                confirm_sha256=args.confirm_sha256,
                baseline_report_path=args.baseline_report,
                allow_initial_baseline=args.allow_initial_baseline,
            )
        serializable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in result.items()
        }
        print(json.dumps(serializable, ensure_ascii=False, indent=2, default=str))
        return 0
    except (PromotionError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print("SYNTRA checkpoint promotion failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

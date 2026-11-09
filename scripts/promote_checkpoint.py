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

    report_relative = report_file.relative_to(BENCHMARK_ROOT.resolve())
    checkpoint_relative = checkpoint.relative_to(BENCHMARK_ROOT.resolve())
    if report_relative.parts[:3] != checkpoint_relative.parts[:3]:
        raise PromotionError("Benchmark report and staged checkpoint are not from the same evaluation run.")
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
        "task_count": task_count,
        "snapshot": snapshot_path,
    }


def _active_artifact_paths(active: Path) -> list[tuple[str, Path]]:
    from syntra_ai.training import inference_checkpoint_path
    return [
        ("checkpoint", active),
        ("metadata", active.with_suffix(active.suffix + ".json")),
        ("inference_companion", inference_checkpoint_path(active)),
    ]


def _archive_active(active: Path) -> str | None:
    artifacts = [(kind, path) for kind, path in _active_artifact_paths(active) if path.exists()]
    if not artifacts:
        return None
    for _kind, path in artifacts:
        _safe_repo_path(path, must_exist=True)
        if path.is_symlink() or not path.is_file():
            raise PromotionError("Active checkpoint artifact must be a regular file: " + str(path))
    archive_id = _now_id()
    archive_dir = _safe_repo_path(ARCHIVE_ROOT / archive_id)
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: dict[str, dict[str, str]] = {}
    try:
        for kind, source in artifacts:
            digest = _sha256(source)
            target = archive_dir / source.name
            shutil.copy2(source, target)
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
    archive_dir = _safe_repo_path(ARCHIVE_ROOT / archive_id, must_exist=True)
    manifest = _read_json(archive_dir / "archive.json", maximum_bytes=1024 * 1024)
    if manifest.get("schema_version") != 1 or manifest.get("archive_id") != archive_id:
        raise PromotionError("Rollback archive manifest is invalid.")
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict) or "checkpoint" not in entries:
        raise PromotionError("Rollback archive has no previous active checkpoint.")
    current_archive = _archive_active(active) if archive_current else None
    try:
        for kind in ("checkpoint", "metadata", "inference_companion"):
            entry = entries.get(kind)
            destination = dict(_active_artifact_paths(active)).get(kind)
            if not isinstance(entry, dict) or destination is None:
                if destination is not None and destination.exists():
                    destination.unlink()
                continue
            filename = entry.get("file")
            digest = entry.get("sha256")
            if not isinstance(filename, str) or Path(filename).name != filename or not isinstance(digest, str):
                raise PromotionError("Rollback archive contains an unsafe artifact entry.")
            source = _safe_repo_path(archive_dir / filename, must_exist=True)
            if _sha256(source) != digest:
                raise PromotionError("Rollback archive SHA-256 verification failed for " + filename)
            _atomic_copy_verified(source, destination, digest)
        from syntra_ai import runtime
        runtime._get_resident_bundle(path=active, device_name="cpu")
    except Exception:
        if current_archive:
            # Keep the just-created archive; the operator can restore it manually.
            pass
        raise
    return {"rollback_archive_id": archive_id, "archive_created_for_replaced_checkpoint": current_archive, "active_checkpoint": str(active)}


def _validate_runtime_checkpoint(path: Path) -> None:
    try:
        from syntra_ai import runtime
        runtime._get_resident_bundle(path=path, device_name="cpu")
    except Exception as exc:
        raise PromotionError("SYNTRA runtime rejected the staged checkpoint: " + str(exc)) from None


def promote(report_path: str | Path, active_checkpoint: str | Path | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    candidate = verify_benchmark_candidate(report_path)
    from syntra_ai import runtime
    active = _safe_repo_path(active_checkpoint or runtime.checkpoint_path())
    if active == candidate["checkpoint"]:
        raise PromotionError("The evaluated staging checkpoint is already the active checkpoint.")
    if active.is_symlink():
        raise PromotionError("Active checkpoint destination must not be a symbolic link.")
    _validate_runtime_checkpoint(candidate["checkpoint"])
    if dry_run:
        return {**candidate, "active_checkpoint": active, "dry_run": True, "archive_id": None}

    archive_id = _archive_active(active)
    metadata_source = candidate["checkpoint"].with_suffix(candidate["checkpoint"].suffix + ".json")
    expected_metadata_hash = None
    snapshot = _read_json(candidate["snapshot"], maximum_bytes=1024 * 1024)
    artifact_hashes = snapshot["artifacts"]
    if metadata_source.is_file() and not metadata_source.is_symlink():
        expected_metadata_hash = artifact_hashes.get(metadata_source.name)
        if expected_metadata_hash != _sha256(metadata_source):
            raise PromotionError("Staged checkpoint metadata is not covered by its snapshot manifest.")

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from syntra_ai import runtime
        active = _safe_repo_path(args.active_checkpoint or runtime.checkpoint_path())
        if args.rollback:
            result = _restore_archive(active, args.rollback)
        else:
            result = promote(args.report, active_checkpoint=active, dry_run=args.dry_run)
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

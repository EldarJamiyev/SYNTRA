import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_epochs.py"
SPEC = importlib.util.spec_from_file_location("syntra_evaluate_epochs_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_epochs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_epochs
SPEC.loader.exec_module(evaluate_epochs)


class Phase4Milestone4ProfileTests(unittest.TestCase):
    def test_checked_in_local_profile_has_the_requested_optimizer_schedule(self):
        profile = evaluate_epochs.load_profile(evaluate_epochs.DEFAULT_CONFIG)
        evaluate_epochs._validate_profile_values(profile)
        pre = profile["pretraining"]
        self.assertEqual(pre["learning_rate"], 3e-4)
        self.assertEqual(pre["minimum_learning_rate"], 1e-5)
        self.assertEqual(pre["warmup_steps"], 100)
        self.assertEqual(pre["decay_steps"], 5000)
        self.assertEqual(pre["gradient_clip_norm"], 1.0)
        self.assertEqual(profile["evaluation"]["every_steps"], 500)
        self.assertEqual(profile["evaluation"]["isolation"], "docker")
        self.assertEqual(profile["checkpoint"], "data/checkpoints/staging/syntra-brain.pt")

    def test_phase5_checked_in_profiles_cover_smoke_cpu_and_gpu_targets(self):
        smoke = evaluate_epochs.load_profile(ROOT / "configs" / "training" / "phase4_smoke.json")
        cpu = evaluate_epochs.load_profile(ROOT / "configs" / "training" / "local_cpu_fast.json")
        gpu = evaluate_epochs.load_profile(ROOT / "configs" / "training" / "local_gpu_scale.json")
        for profile in (smoke, cpu, gpu):
            evaluate_epochs._validate_profile_values(profile)
            self.assertGreaterEqual(profile["pretraining"]["gradient_accumulation_steps"], 1)
            self.assertGreaterEqual(profile["sft"]["gradient_accumulation_steps"], 1)
            self.assertEqual(profile["evaluation"]["isolation"], "docker")
        self.assertEqual(smoke["pretraining"]["max_steps"], 50)
        self.assertEqual(smoke["sft"]["steps"], 20)
        self.assertEqual(smoke["evaluation"]["every_steps"], 50)
        self.assertEqual(cpu["pretraining"]["batch_size"], 1)
        self.assertGreaterEqual(cpu["pretraining"]["gradient_accumulation_steps"], 4)
        self.assertEqual(gpu["pretraining"]["layers"], 8)
        self.assertEqual(gpu["pretraining"]["heads"], 8)
        self.assertEqual(gpu["pretraining"]["width"], 256)
        self.assertEqual(gpu["pretraining"]["block_size"], 1024)

    def test_smoke_profile_selects_one_task_for_each_local_benchmark_runtime(self):
        profile = evaluate_epochs.load_profile(ROOT / "configs" / "training" / "phase4_smoke.json")
        task_ids = profile["evaluation"]["task_ids"]
        runner = evaluate_epochs._load_benchmark_runner()
        suite = runner.load_suite()
        by_id = {task["id"]: task for task in suite["tasks"]}
        selected = [by_id[item] for item in task_ids]
        self.assertEqual(len(selected), 5)
        self.assertEqual({task["language"] for task in selected}, {"python", "javascript", "typescript", "go", "rust"})
        preflight_languages = {task["language"] for task in suite["tasks"]}
        self.assertEqual(preflight_languages, {"python", "javascript", "typescript", "go", "rust"})

    def test_profile_rejects_duplicate_or_unknown_task_ids(self):
        profile = evaluate_epochs.load_profile(ROOT / "configs" / "training" / "phase4_smoke.json")
        duplicate = json.loads(json.dumps(profile))
        duplicate["evaluation"]["task_ids"].append(duplicate["evaluation"]["task_ids"][0])
        with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "unique list"):
            evaluate_epochs._validate_profile_values(duplicate)

        unknown = ["does-not-exist"]
        with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "unknown raw benchmark task ID"):
            evaluate_epochs._preflight_benchmark("docker", False, unknown)

    def test_profile_rejects_invalid_decay_and_parallel_benchmarking(self):
        profile = evaluate_epochs.load_profile(evaluate_epochs.DEFAULT_CONFIG)
        invalid_decay = json.loads(json.dumps(profile))
        invalid_decay["pretraining"]["decay_steps"] = 100
        with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "must exceed"):
            evaluate_epochs._validate_profile_values(invalid_decay)

        invalid_parallel = json.loads(json.dumps(profile))
        invalid_parallel["evaluation"]["max_parallel"] = 2
        with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "max_parallel"):
            evaluate_epochs._validate_profile_values(invalid_parallel)

    def test_windows_npm_benchmark_command_preserves_paths_with_spaces(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(prefix=".syntra-windows-command-", dir=ROOT) as temp:
            evaluator = evaluate_epochs.BackgroundEvaluator(
                Path(temp), "windows-command-test", device="cpu", isolation="docker",
                allow_host_execution=False,
            )
            executable = r"C:\Program Files\nodejs\npm.cmd"
            with patch.object(evaluate_epochs.os, "name", "nt"), patch.object(
                evaluate_epochs.shutil, "which", return_value=executable
            ):
                command = evaluator._npm_command([
                    "run", "benchmark:model", "--", "--checkpoint",
                    r"C:\SYNTRA Workspace\data\checkpoints\snapshot.pt",
                ])
            self.assertEqual(command[:4], ["cmd.exe", "/d", "/s", "/c"])
            self.assertTrue(command[4].startswith('"'))
            self.assertTrue(command[4].endswith('"'))
            self.assertIn('"C:\\Program Files\\nodejs\\npm.cmd"', command[4])
            self.assertIn('"C:\\SYNTRA Workspace\\data\\checkpoints\\snapshot.pt"', command[4])

    def test_repository_benchmark_runner_loads_without_importing_a_site_packages_tests_package(self):
        runner = evaluate_epochs._load_benchmark_runner()
        suite = runner.load_suite()
        self.assertEqual(suite["version"], "1.3.0")
        self.assertEqual(len(suite["tasks"]), 13)

    def test_profile_paths_cannot_escape_the_repository(self):
        with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "inside the SYNTRA repository"):
            evaluate_epochs._resolve_profile_path("../outside.json")

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            evaluate_epochs._strict_json('{"step":1,"step":2}')

    def test_in_repository_symlink_components_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-output-symlink-", dir=ROOT) as temp:
            root = Path(temp)
            actual = root / "real-directory"
            actual.mkdir()
            link = root / "redirected-directory"
            try:
                link.symlink_to(actual, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("This filesystem does not support directory symlinks.")
            with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "Symbolic-link path components"):
                evaluate_epochs._inside_root(link / "report.json")


class Phase4CheckpointEvaluationTests(unittest.TestCase):
    def test_snapshot_copies_checkpoint_sidecar_and_inference_companion_with_hashes(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-evaluate-snapshot-", dir=ROOT) as temp:
            root = Path(temp)
            checkpoint = root / "staging" / "syntra-brain.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"native checkpoint artifact bytes")
            sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
            sidecar.write_text('{"step":500,"train_loss":1.25}', encoding="utf-8")
            companion = root / "staging" / "syntra-brain.inference.pt"
            companion.write_bytes(b"native inference companion bytes")

            expected_checkpoint_hash = evaluate_epochs._hash_regular_file(checkpoint)
            snapshots = evaluate_epochs.CheckpointSnapshots(
                checkpoint,
                root / "snapshots",
                "unit-test-run",
            )
            staged, staged_hash = snapshots.create("pretraining", 500)
            staged_directory = staged.parent
            metadata = json.loads((staged_directory / "snapshot.json").read_text(encoding="utf-8"))

            self.assertEqual(staged_hash, expected_checkpoint_hash)
            self.assertEqual(evaluate_epochs._hash_regular_file(staged), expected_checkpoint_hash)
            self.assertIn(sidecar.name, metadata["artifacts"])
            self.assertIn(companion.name, metadata["artifacts"])
            self.assertEqual(
                evaluate_epochs._hash_regular_file(staged_directory / companion.name),
                evaluate_epochs._hash_regular_file(companion),
            )
            self.assertTrue(metadata["snapshot_is_read_only"])
            self.assertFalse(os.stat(staged).st_mode & stat.S_IWUSR)
            self.assertEqual(metadata["global_step"], 500)

    def test_fresh_pretraining_refuses_an_unhandled_stale_inference_companion(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-stale-companion-", dir=ROOT) as temp:
            root = Path(temp)
            checkpoint = root / "staging" / "syntra-brain.pt"
            checkpoint.parent.mkdir()
            companion = root / "staging" / "syntra-brain.inference.pt"
            companion.write_bytes(b"old inference weights")
            with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "stale inference companion"):
                evaluate_epochs._prepare_inference_companion_for_fresh_pretraining(
                    checkpoint,
                    root / "backup",
                    resume=False,
                    overwrite=False,
                )
            self.assertTrue(companion.is_file())

    def test_fresh_pretraining_backs_up_and_removes_stale_inference_companion(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-stale-companion-backup-", dir=ROOT) as temp:
            root = Path(temp)
            checkpoint = root / "staging" / "syntra-brain.pt"
            checkpoint.parent.mkdir()
            companion = root / "staging" / "syntra-brain.inference.pt"
            companion.write_bytes(b"old inference weights")
            backup = evaluate_epochs._prepare_inference_companion_for_fresh_pretraining(
                checkpoint,
                root / "backup",
                resume=False,
                overwrite=True,
            )
            self.assertIsNotNone(backup)
            self.assertFalse(companion.exists())
            self.assertEqual(backup.read_bytes(), b"old inference weights")

    def test_pretraining_resume_preserves_existing_inference_companion(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-resume-companion-", dir=ROOT) as temp:
            root = Path(temp)
            checkpoint = root / "staging" / "syntra-brain.pt"
            checkpoint.parent.mkdir()
            companion = root / "staging" / "syntra-brain.inference.pt"
            companion.write_bytes(b"resumable inference weights")
            returned = evaluate_epochs._prepare_inference_companion_for_fresh_pretraining(
                checkpoint,
                root / "backup",
                resume=True,
                overwrite=False,
            )
            self.assertEqual(returned, companion)
            self.assertTrue(companion.is_file())

    def test_checkpoint_snapshot_rejects_symlinks(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-evaluate-symlink-", dir=ROOT) as temp:
            root = Path(temp)
            actual = root / "real.pt"
            actual.write_bytes(b"checkpoint")
            link = root / "linked.pt"
            try:
                link.symlink_to(actual)
            except (OSError, NotImplementedError):
                self.skipTest("This filesystem does not support symlinks.")
            snapshots = evaluate_epochs.CheckpointSnapshots(link, root / "snapshots", "symlink-test")
            with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "regular checkpoint"):
                snapshots.create("pretraining", 1)

    def test_skip_benchmark_callback_does_not_duplicate_model_checkpoints(self):
        from unittest.mock import Mock

        snapshots = Mock()
        seen = set()
        callback = evaluate_epochs._training_callback(
            "pretraining", snapshots, None, 500, seen
        )
        callback({
            "status": "completed",
            "step": 5000,
            "message": "Requested training-step block saved.",
        })
        snapshots.create.assert_not_called()
        self.assertEqual(seen, set())

    def test_snapshot_schedule_only_uses_saved_interval_or_stage_completion(self):
        seen = set()
        interval_progress = {
            "status": "running",
            "step": 500,
            "message": "Checkpoint saved; training and held-out validation metrics updated.",
        }
        self.assertTrue(evaluate_epochs._should_snapshot("pretraining", interval_progress, 500, seen))
        seen.add(("pretraining", 500))
        self.assertFalse(evaluate_epochs._should_snapshot("pretraining", interval_progress, 500, seen))
        self.assertFalse(evaluate_epochs._should_snapshot(
            "pretraining",
            {"status": "running", "step": 501, "message": "Loss updated."},
            500,
            seen,
        ))
        self.assertTrue(evaluate_epochs._should_snapshot(
            "sft",
            {"status": "completed", "step": 725, "message": "Fine-tuning finished."},
            500,
            seen,
        ))

    def test_completed_benchmark_requires_the_exact_staged_checkpoint_hash(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory(prefix=".syntra-eval-report-identity-", dir=ROOT) as temp:
            root = Path(temp)
            run_root = root / "run"
            run_root.mkdir()
            evaluator = evaluate_epochs.BackgroundEvaluator(
                run_root, "identity-test", device="cpu", isolation="docker",
                allow_host_execution=False,
            )
            checkpoint_hash = "a" * 64
            report_path = evaluator.reports_dir / "matched.json"
            log_path = evaluator.logs_dir / "matched.log"
            report_path.write_text(json.dumps({
                "schema_version": 1,
                "benchmark_track": "raw-native-model",
                "suite_version": "1.3.0",
                "model": {"checkpoint_sha256": checkpoint_hash},
                "training_loss": {"train_loss": 1.25, "validation_loss": 1.5},
                "summary": {
                    "task_completion_rate": 0.25,
                    "completed_task_count": 3,
                    "baseline_valid_task_count": 12,
                    "regression_prevention_rate": 0.8,
                },
            }), encoding="utf-8")
            log_path.write_text("benchmark completed", encoding="utf-8")
            process = Mock()
            process.wait.return_value = 0
            handle = log_path.open("a", encoding="utf-8")
            evaluator.active.append(evaluate_epochs.EvaluationJob(
                stage="pretraining", step=500, checkpoint_sha256=checkpoint_hash,
                checkpoint_path=root / "snapshot.pt", report_path=report_path,
                log_path=log_path, process=process, log_handle=handle,
            ))
            evaluator._wait_one()
            self.assertEqual(evaluator.results[0]["status"], "completed")
            self.assertEqual(evaluator.results[0]["task_completion_rate"], 0.25)
            self.assertEqual(evaluator.results[0]["completed_task_count"], 3)
            self.assertEqual(evaluator.results[0]["train_loss"], 1.25)

    def test_benchmark_report_from_a_different_checkpoint_is_rejected(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory(prefix=".syntra-eval-report-mismatch-", dir=ROOT) as temp:
            root = Path(temp)
            evaluator = evaluate_epochs.BackgroundEvaluator(
                root / "run", "identity-mismatch", device="cpu", isolation="docker",
                allow_host_execution=False,
            )
            report_path = evaluator.reports_dir / "mismatch.json"
            log_path = evaluator.logs_dir / "mismatch.log"
            report_path.write_text(json.dumps({
                "schema_version": 1,
                "benchmark_track": "raw-native-model",
                "suite_version": "1.3.0",
                "model": {"checkpoint_sha256": "b" * 64},
                "training_loss": {"train_loss": 0.1, "validation_loss": 0.2},
                "summary": {"task_completion_rate": 1.0, "completed_task_count": 12, "baseline_valid_task_count": 12},
            }), encoding="utf-8")
            log_path.write_text("", encoding="utf-8")
            process = Mock()
            process.wait.return_value = 0
            handle = log_path.open("a", encoding="utf-8")
            evaluator.active.append(evaluate_epochs.EvaluationJob(
                stage="pretraining", step=500, checkpoint_sha256="a" * 64,
                checkpoint_path=root / "snapshot.pt", report_path=report_path,
                log_path=log_path, process=process, log_handle=handle,
            ))
            evaluator._wait_one()
            self.assertEqual(evaluator.results[0]["status"], "failed")
            self.assertIsNone(evaluator.results[0]["task_completion_rate"])
            self.assertIn("does not match", evaluator.results[0]["error"])

    def test_background_evaluator_passes_selected_tasks_to_real_benchmark_command(self):
        from unittest.mock import Mock, patch

        with tempfile.TemporaryDirectory(prefix=".syntra-selected-tasks-", dir=ROOT) as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"snapshot")
            evaluator = evaluate_epochs.BackgroundEvaluator(
                root / "run", "selected-tasks", device="cpu", isolation="docker",
                allow_host_execution=False,
                task_ids=["python-clamp", "rust-safe-slice"],
            )
            process = Mock()
            with patch.object(evaluator, "_npm_command", side_effect=lambda args: ["npm", *args]) as command, patch.object(
                evaluate_epochs.subprocess, "Popen", return_value=process
            ):
                evaluator.launch("pretraining", 50, checkpoint, "a" * 64)
            args = command.call_args.args[0]
            self.assertEqual(
                [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--task"],
                ["python-clamp", "rust-safe-slice"],
            )
            self.assertFalse(any(value in {"--network", "pull"} for value in args))
            for job in evaluator.active:
                job.log_handle.close()

    def test_checkpoint_callback_attaches_measured_training_telemetry(self):
        from unittest.mock import Mock

        snapshots = Mock()
        snapshots.create.return_value = (ROOT / "data" / "benchmarks" / "snapshot.pt", "a" * 64)
        evaluator = Mock()
        callback = evaluate_epochs._training_callback(
            "pretraining", snapshots, evaluator, 500, set()
        )
        callback({
            "status": "completed",
            "step": 500,
            "message": "Requested training-step block saved.",
            "tokens_per_second": 123.5,
            "step_latency_ms": 8.2,
            "mean_step_latency_ms": 9.1,
            "peak_rss_bytes": 1000,
            "peak_device_memory_bytes": 2000,
            "device_memory_metric": "cuda_peak_allocated",
            "gradient_accumulation_steps": 2,
        })
        kwargs = evaluator.launch.call_args.kwargs
        self.assertEqual(kwargs["telemetry"]["tokens_per_second"], 123.5)
        self.assertEqual(kwargs["telemetry"]["mean_step_latency_ms"], 9.1)
        self.assertEqual(kwargs["telemetry"]["peak_device_memory_bytes"], 2000)
        self.assertEqual(kwargs["telemetry"]["gradient_accumulation_steps"], 2)

    def test_markdown_and_csv_summaries_include_real_hardware_telemetry(self):
        import csv

        with tempfile.TemporaryDirectory(prefix=".syntra-phase5-summary-", dir=ROOT) as temp:
            run_root = Path(temp)
            records = [{
                "stage": "pretraining", "step": 500, "status": "completed",
                "checkpoint_sha256": "a" * 64, "train_loss": 1.2, "validation_loss": 1.3,
                "task_completion_rate": 0.25, "completed_task_count": 3,
                "baseline_valid_task_count": 12, "regression_prevention_rate": 0.5,
                "tokens_per_second": 123.5, "supervised_tokens_processed": 6000,
                "step_latency_ms": 8.2, "mean_step_latency_ms": 9.1,
                "peak_rss_bytes": 512 * 1024 * 1024,
                "peak_rss_scope": "process_lifetime_high_water",
                "peak_device_memory_bytes": 1024 * 1024 * 1024,
                "device_memory_metric": "cuda_peak_allocated",
                "estimated_training_memory_bytes": 2 * 1024 * 1024 * 1024,
                "device": "cuda", "gradient_accumulation_steps": 2,
                "throughput_basis": "supervised_target_tokens",
                "throughput_scope": "stage_wall_clock_including_validation_and_checkpoint_io",
                "suite_version": "1.3.0", "report": "report.json", "log": "run.log",
            }]
            markdown, csv_path = evaluate_epochs.write_summary(
                run_root, "phase5-telemetry-test", records, {"pretraining_status": "completed"}
            )
            summary = markdown.read_text(encoding="utf-8")
            self.assertIn("Tokens/s", summary)
            self.assertIn("Mean step ms", summary)
            self.assertIn("Peak RSS MiB", summary)
            self.assertIn("Peak device MiB", summary)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["tokens_per_second"], "123.5")
            self.assertEqual(row["step_latency_ms"], "8.2")
            self.assertEqual(row["peak_rss_bytes"], str(512 * 1024 * 1024))
            self.assertEqual(row["peak_device_memory_bytes"], str(1024 * 1024 * 1024))
            self.assertEqual(row["device_memory_metric"], "cuda_peak_allocated")
            self.assertEqual(row["peak_rss_scope"], "process_lifetime_high_water")
            self.assertEqual(row["throughput_basis"], "supervised_target_tokens")
            self.assertEqual(row["throughput_scope"], "stage_wall_clock_including_validation_and_checkpoint_io")

    def test_ascii_trends_do_not_turn_missing_measurements_into_zero(self):
        empty = evaluate_epochs._ascii_plot([], "task_completion_rate", "Completion")
        self.assertIn("No verified measurements", empty)
        actual = evaluate_epochs._ascii_plot(
            [
                {"step": 500, "status": "completed", "task_completion_rate": 0.25},
                {"step": 1000, "status": "completed", "task_completion_rate": 0.5},
                {"step": 1500, "status": "failed", "task_completion_rate": None},
            ],
            "task_completion_rate",
            "Completion",
        )
        self.assertIn("0.25", actual)
        self.assertIn("0.5", actual)
        self.assertNotIn("1500", actual)


if __name__ == "__main__":
    unittest.main()

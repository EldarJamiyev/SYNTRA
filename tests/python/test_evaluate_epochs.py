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
        self.assertEqual(profile["evaluation"]["every_steps"], 500)
        self.assertEqual(profile["evaluation"]["isolation"], "docker")
        self.assertEqual(profile["checkpoint"], "data/checkpoints/staging/syntra-brain.pt")

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

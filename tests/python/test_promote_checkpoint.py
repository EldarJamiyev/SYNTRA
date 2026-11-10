import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "data" / "benchmarks"
CHECKPOINT_ROOT = ROOT / "data" / "checkpoints"
SCRIPT = ROOT / "scripts" / "promote_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("syntra_promote_checkpoint_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
promoter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = promoter
SPEC.loader.exec_module(promoter)


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run checkpoint-promotion tests.")
class BenchmarkVerifiedPromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    def _write_checkpoint(self, path: Path, seed: int) -> str:
        from syntra_ai.config import ModelConfig
        from syntra_ai.model import SyntraTransformer
        from syntra_ai.tokenizer import ByteBPETokenizer

        torch.manual_seed(seed)
        config = ModelConfig(vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        payload = {
            "format": "syntra-transformer-v1",
            "config": config.to_dict(),
            "tokenizer": ByteBPETokenizer.empty().to_dict(),
            "model_state": model.state_dict(),
            "step": seed,
            "created_by": "SYNTRA Phase 5 promotion unit test",
            "train_loss": 1.25,
            "validation_loss": 1.5,
            "training_mode": "causal-source-pretraining",
            "training_history": [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_active_metadata_and_companion(self, checkpoint: Path) -> tuple[str, str]:
        from syntra_ai import runtime
        from syntra_ai.training import _save_inference_checkpoint

        _path, payload, _config, _tokenizer = runtime._read_checkpoint(checkpoint)
        metadata = {
            "checkpoint": str(checkpoint),
            "step": payload["step"],
            "train_loss": payload["train_loss"],
            "validation_loss": payload["validation_loss"],
            "training_mode": payload["training_mode"],
        }
        metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        companion = _save_inference_checkpoint(checkpoint, payload)
        return hashlib.sha256(metadata_path.read_bytes()).hexdigest(), hashlib.sha256(companion.read_bytes()).hexdigest()

    def _candidate(self, run_root: Path, checkpoint: Path, digest: str, *, stage: str = "sft", step: int = 20) -> Path:
        snapshot_dir = run_root / "snapshots" / run_root.name / f"{stage}-step-{step:09d}-{digest[:12]}"
        snapshot_dir.mkdir(parents=True)
        staged = snapshot_dir / "syntra-brain.pt"
        shutil.copy2(checkpoint, staged)
        snapshot = {
            "schema_version": 1,
            "stage": stage,
            "global_step": step,
            "created_at": "2026-11-09T10:00:00Z",
            "checkpoint_file": staged.name,
            "checkpoint_sha256": digest,
            "artifacts": {staged.name: digest},
            "snapshot_is_read_only": True,
        }
        (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        return staged

    def _report(
        self,
        run_root: Path,
        staged: Path,
        digest: str,
        *,
        stage: str = "sft",
        step: int = 20,
        outcomes: dict[str, bool] | None = None,
        regression_outcomes: dict[str, bool] | None = None,
        run_id: str = "raw-model-phase9-test",
    ) -> Path:
        suite = json.loads((ROOT / "tests" / "benchmark_engine" / "suite.v1.json").read_text(encoding="utf-8"))
        task_ids = [task["id"] for task in suite["tasks"]]
        measured = outcomes if outcomes is not None else {task_id: True for task_id in task_ids}
        regression = regression_outcomes if regression_outcomes is not None else measured
        completed = sum(1 for task_id in task_ids if measured.get(task_id) is True)
        fully_verified = sum(
            1 for task_id in task_ids
            if measured.get(task_id) is True and regression.get(task_id) is True
        )
        report_dir = run_root / "evaluations"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{stage}-step-{step:09d}.json"
        report = {
            "schema_version": 1,
            "benchmark_track": "raw-native-model",
            "suite_version": suite["version"],
            "run_id": run_id,
            "started_at": "2026-11-09T10:01:00Z",
            "completed_at": "2026-11-09T10:02:00Z",
            "model": {
                "engine": "syntra-native-pytorch",
                "checkpoint": str(staged),
                "checkpoint_sha256": digest,
                "trained_steps": step,
            },
            "isolation": {"mode": "docker", "environments": {}},
            "summary": {
                "task_count": len(task_ids),
                "baseline_valid_task_count": len(task_ids),
                "invalid_fixture_count": 0,
                "completed_task_count": completed,
                "fully_verified_task_count": fully_verified,
                "task_completion_rate": completed / len(task_ids),
            },
            "tasks": [
                {
                    "id": task_id,
                    "baseline_validated": True,
                    "task_completion": measured.get(task_id) is True,
                    "regression_prevention": regression.get(task_id) is True,
                }
                for task_id in task_ids
            ],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path

    def _baseline_report(self, run_root: Path, *, outcomes: dict[str, bool] | None = None) -> Path:
        source = run_root.parent / "baseline-source.pt"
        digest = self._write_checkpoint(source, 9)
        staged = self._candidate(run_root, source, digest, stage="pretraining", step=10)
        return self._report(run_root, staged, digest, stage="pretraining", step=10, outcomes=outcomes, run_id="raw-model-pretraining-invocation")

    def test_report_verification_binds_report_to_snapshot_and_full_checkpoint_hash(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-verify-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            checkpoint = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(checkpoint, 7)
            staged = self._candidate(run_root, checkpoint, digest)
            report = self._report(run_root, staged, digest)
            verified = promoter.verify_benchmark_candidate(report)
            self.assertEqual(verified["checkpoint_sha256"], digest)
            self.assertEqual(verified["checkpoint"], staged)
            self.assertEqual(verified["stage"], "sft")
            self.assertEqual(verified["task_completion_rate"], 1.0)

    def test_report_hash_mismatch_is_rejected_before_runtime_loading(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-hash-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            checkpoint = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(checkpoint, 8)
            staged = self._candidate(run_root, checkpoint, digest)
            report = self._report(run_root, staged, digest)
            data = json.loads(report.read_text(encoding="utf-8"))
            data["model"]["checkpoint_sha256"] = "0" * 64
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(promoter.PromotionError, "does not match"):
                promoter.verify_benchmark_candidate(report)

    def test_promotion_archives_previous_active_and_rollback_restores_it(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-roundtrip-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            candidate_source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(candidate_source, 11)
            staged = self._candidate(run_root, candidate_source, digest)
            report = self._report(run_root, staged, digest)
            baseline_report = self._baseline_report(run_root, outcomes={})

            active_root = CHECKPOINT_ROOT / (".phase5-active-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            old_digest = self._write_checkpoint(active, 3)
            old_metadata_digest, old_companion_digest = self._write_active_metadata_and_companion(active)
            result = promoter.promote(report, active_checkpoint=active, confirm_sha256=digest, baseline_report_path=baseline_report)

            self.assertTrue(result["promoted"])
            self.assertEqual(result["checkpoint_sha256"], digest)
            self.assertGreater(result["comparison"]["fully_verified_task_delta"], 0)
            self.assertEqual(result["comparison"]["regressed_task_ids"], [])
            self.assertEqual(promoter._sha256(active), digest)
            self.assertTrue(result["archive_id"])
            self.assertTrue(active.with_name("syntra.inference.pt").is_file())
            archive_manifest = json.loads(
                (promoter.ARCHIVE_ROOT / result["archive_id"] / "archive.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(archive_manifest["artifacts"]),
                {"checkpoint", "metadata", "inference_companion"},
            )
            for artifact in archive_manifest["artifacts"].values():
                archived_file = promoter.ARCHIVE_ROOT / result["archive_id"] / artifact["file"]
                self.assertEqual(promoter._sha256(archived_file), artifact["sha256"])

            rollback = promoter._restore_archive(active, result["archive_id"])
            self.assertEqual(promoter._sha256(active), old_digest)
            self.assertEqual(promoter._sha256(active.with_suffix(active.suffix + ".json")), old_metadata_digest)
            self.assertTrue(rollback["archive_created_for_replaced_checkpoint"])
            self.assertTrue(active.with_name("syntra.inference.pt").is_file())
            from syntra_ai import runtime
            _path, restored_payload, _config, _tokenizer = runtime._read_checkpoint(active)
            self.assertEqual(restored_payload["step"], 3)

            shutil.rmtree(active_root, ignore_errors=True)

    def test_promotion_requires_exact_digest_acknowledgement(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-ack-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 21)
            staged = self._candidate(run_root, source, digest)
            report = self._report(run_root, staged, digest)
            active_root = CHECKPOINT_ROOT / (".phase5-ack-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            self._write_checkpoint(active, 22)
            with self.assertRaisesRegex(promoter.PromotionError, "requires --confirm-sha256"):
                promoter.promote(report, active_checkpoint=active)
            with self.assertRaisesRegex(promoter.PromotionError, "requires --confirm-sha256"):
                promoter.promote(report, active_checkpoint=active, confirm_sha256="0" * 64)
            self.assertNotEqual(promoter._sha256(active), digest)
            shutil.rmtree(active_root, ignore_errors=True)

    def test_dry_run_validates_loadability_without_changing_active_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-dry-run-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            candidate_source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(candidate_source, 14)
            staged = self._candidate(run_root, candidate_source, digest)
            report = self._report(run_root, staged, digest)
            baseline_report = self._baseline_report(run_root, outcomes={})
            active_root = CHECKPOINT_ROOT / (".phase5-dry-run-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            old_digest = self._write_checkpoint(active, 5)
            result = promoter.promote(report, active_checkpoint=active, dry_run=True, baseline_report_path=baseline_report)
            self.assertTrue(result["dry_run"])
            self.assertEqual(promoter._sha256(active), old_digest)
            shutil.rmtree(active_root, ignore_errors=True)

    def test_sft_promotion_rejects_missing_baseline_and_non_improving_candidate(self):
        with tempfile.TemporaryDirectory(prefix=".phase9-no-gain-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 31)
            staged = self._candidate(run_root, source, digest)
            candidate_report = self._report(run_root, staged, digest, outcomes={})
            active_root = CHECKPOINT_ROOT / (".phase9-no-gain-active-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            self._write_checkpoint(active, 32)
            with self.assertRaisesRegex(promoter.PromotionError, "requires --baseline-report"):
                promoter.promote(candidate_report, active_checkpoint=active, confirm_sha256=digest)
            baseline_report = self._baseline_report(run_root, outcomes={})
            candidate = promoter.verify_benchmark_candidate(candidate_report)
            with self.assertRaisesRegex(promoter.PromotionError, "must improve fully-verified"):
                promoter.verify_paired_benchmark(candidate, baseline_report)
            self.assertNotEqual(promoter._sha256(active), digest)
            shutil.rmtree(active_root, ignore_errors=True)

    def test_promotion_rejects_target_passes_without_regression_test_success(self):
        with tempfile.TemporaryDirectory(prefix=".phase9-regression-check-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            suite = json.loads((ROOT / "tests" / "benchmark_engine" / "suite.v1.json").read_text(encoding="utf-8"))
            task_ids = [task["id"] for task in suite["tasks"]]
            baseline_outcomes = {task_id: index != 0 for index, task_id in enumerate(task_ids)}
            baseline_report = self._baseline_report(run_root, outcomes=baseline_outcomes)
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 35)
            staged = self._candidate(run_root, source, digest)
            suite = json.loads((ROOT / "tests" / "benchmark_engine" / "suite.v1.json").read_text(encoding="utf-8"))
            task_ids = [task["id"] for task in suite["tasks"]]
            target_passes = {task_id: True for task_id in task_ids}
            regression_passes = {task_id: index != 0 for index, task_id in enumerate(task_ids)}
            candidate_report = self._report(
                run_root,
                staged,
                digest,
                outcomes=target_passes,
                regression_outcomes=regression_passes,
            )
            candidate = promoter.verify_benchmark_candidate(candidate_report)
            self.assertEqual(candidate["completed_task_count"], len(task_ids))
            self.assertEqual(candidate["fully_verified_task_count"], len(task_ids) - 1)
            with self.assertRaisesRegex(promoter.PromotionError, "must improve fully-verified"):
                promoter.verify_paired_benchmark(candidate, baseline_report)

    def test_sft_comparison_rejects_baseline_from_another_training_run(self):
        with tempfile.TemporaryDirectory(prefix=".phase9-run-identity-", dir=BENCHMARK_ROOT) as temp:
            baseline_run_root = Path(temp) / "baseline-run"
            baseline_report = self._baseline_report(baseline_run_root, outcomes={})
            candidate_run_root = Path(temp) / "candidate-run"
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 34)
            staged = self._candidate(candidate_run_root, source, digest)
            candidate_report = self._report(candidate_run_root, staged, digest)
            candidate = promoter.verify_benchmark_candidate(candidate_report)
            with self.assertRaisesRegex(promoter.PromotionError, "same orchestrated training run"):
                promoter.verify_paired_benchmark(candidate, baseline_report)

    def test_sft_promotion_rejects_per_task_regression_even_if_other_tasks_improve(self):
        with tempfile.TemporaryDirectory(prefix=".phase9-regression-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            suite = json.loads((ROOT / "tests" / "benchmark_engine" / "suite.v1.json").read_text(encoding="utf-8"))
            task_ids = [task["id"] for task in suite["tasks"]]
            baseline_outcomes = {task_id: index < 2 for index, task_id in enumerate(task_ids)}
            candidate_outcomes = {task_id: index != 0 for index, task_id in enumerate(task_ids)}
            baseline_report = self._baseline_report(run_root, outcomes=baseline_outcomes)
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 33)
            staged = self._candidate(run_root, source, digest)
            candidate_report = self._report(run_root, staged, digest, outcomes=candidate_outcomes)
            candidate = promoter.verify_benchmark_candidate(candidate_report)
            with self.assertRaisesRegex(promoter.PromotionError, "regresses on baseline-passing tasks"):
                promoter.verify_paired_benchmark(candidate, baseline_report)

    def test_initial_baseline_requires_explicit_ack_and_empty_active_destination(self):
        with tempfile.TemporaryDirectory(prefix=".phase9-initial-baseline-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            suite = json.loads((ROOT / "tests" / "benchmark_engine" / "suite.v1.json").read_text(encoding="utf-8"))
            first_task = suite["tasks"][0]["id"]
            outcomes = {task["id"]: task["id"] == first_task for task in suite["tasks"]}
            source = Path(temp) / "initial-source.pt"
            digest = self._write_checkpoint(source, 41)
            staged = self._candidate(run_root, source, digest, stage="pretraining", step=10)
            report = self._report(run_root, staged, digest, stage="pretraining", step=10, outcomes=outcomes)
            active_root = CHECKPOINT_ROOT / (".phase9-initial-active-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            with self.assertRaisesRegex(promoter.PromotionError, "requires explicit --allow-initial-baseline"):
                promoter.promote(report, active_checkpoint=active, dry_run=True)
            result = promoter.promote(report, active_checkpoint=active, dry_run=True, allow_initial_baseline=True)
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["fully_verified_task_count"], 1)
            self.assertFalse(active.exists())

            active_root.mkdir(parents=True)
            active.write_bytes(b"existing active checkpoint")
            with self.assertRaisesRegex(promoter.PromotionError, "only be promoted when no active checkpoint"):
                promoter.promote(report, active_checkpoint=active, dry_run=True, allow_initial_baseline=True)
            shutil.rmtree(active_root, ignore_errors=True)

    def test_smoke_or_partial_suite_report_cannot_promote_a_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-partial-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 19)
            staged = self._candidate(run_root, source, digest)
            report = self._report(run_root, staged, digest)
            data = json.loads(report.read_text(encoding="utf-8"))
            data["tasks"] = data["tasks"][:5]
            data["summary"]["task_count"] = 5
            data["summary"]["baseline_valid_task_count"] = 5
            data["summary"]["completed_task_count"] = 5
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(promoter.PromotionError, "currently checked-in 13-task"):
                promoter.verify_benchmark_candidate(report)

    def test_report_trained_step_must_match_snapshot_directory_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-step-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 24)
            staged = self._candidate(run_root, source, digest)
            report = self._report(run_root, staged, digest)
            data = json.loads(report.read_text(encoding="utf-8"))
            data["model"]["trained_steps"] = 21
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(promoter.PromotionError, "trained_steps"):
                promoter.verify_benchmark_candidate(report)

    def test_host_benchmark_report_cannot_authorize_promotion(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-host-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(source, 20)
            staged = self._candidate(run_root, source, digest)
            report = self._report(run_root, staged, digest)
            data = json.loads(report.read_text(encoding="utf-8"))
            data["isolation"]["mode"] = "host"
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(promoter.PromotionError, "network-disabled Docker"):
                promoter.verify_benchmark_candidate(report)

    def test_unrecognized_archive_id_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-archive-id-", dir=CHECKPOINT_ROOT) as temp:
            active = Path(temp) / "syntra.pt"
            with self.assertRaisesRegex(promoter.PromotionError, "valid archive ID"):
                promoter._restore_archive(active, "../latest")


if __name__ == "__main__":
    unittest.main()

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

    def _candidate(self, run_root: Path, checkpoint: Path, digest: str) -> Path:
        snapshot_dir = run_root / "snapshots" / "test-run" / f"sft-step-000000020-{digest[:12]}"
        snapshot_dir.mkdir(parents=True)
        staged = snapshot_dir / "syntra-brain.pt"
        shutil.copy2(checkpoint, staged)
        snapshot = {
            "schema_version": 1,
            "stage": "sft",
            "global_step": 20,
            "created_at": "2026-11-09T10:00:00Z",
            "checkpoint_file": staged.name,
            "checkpoint_sha256": digest,
            "artifacts": {staged.name: digest},
            "snapshot_is_read_only": True,
        }
        (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        return staged

    def _report(self, run_root: Path, staged: Path, digest: str) -> Path:
        report_dir = run_root / "evaluations"
        report_dir.mkdir(parents=True)
        report_path = report_dir / "sft-step-000000020.json"
        report = {
            "schema_version": 1,
            "benchmark_track": "raw-native-model",
            "suite_version": "1.3.0",
            "run_id": "raw-model-phase5-test",
            "started_at": "2026-11-09T10:01:00Z",
            "completed_at": "2026-11-09T10:02:00Z",
            "model": {
                "engine": "syntra-native-pytorch",
                "checkpoint": str(staged),
                "checkpoint_sha256": digest,
                "trained_steps": 20,
            },
            "summary": {
                "task_count": 1,
                "baseline_valid_task_count": 1,
                "invalid_fixture_count": 0,
                "completed_task_count": 1,
                "task_completion_rate": 1.0,
            },
            "tasks": [{"id": "python-test", "baseline_validated": True, "task_completion": True}],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path

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

            active_root = CHECKPOINT_ROOT / (".phase5-active-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            old_digest = self._write_checkpoint(active, 3)
            result = promoter.promote(report, active_checkpoint=active)

            self.assertTrue(result["promoted"])
            self.assertEqual(result["checkpoint_sha256"], digest)
            self.assertEqual(promoter._sha256(active), digest)
            self.assertTrue(result["archive_id"])
            self.assertTrue(active.with_name("syntra.inference.pt").is_file())

            rollback = promoter._restore_archive(active, result["archive_id"])
            self.assertEqual(promoter._sha256(active), old_digest)
            self.assertTrue(rollback["archive_created_for_replaced_checkpoint"])
            self.assertTrue(active.with_name("syntra.inference.pt").is_file())
            from syntra_ai import runtime
            _path, restored_payload, _config, _tokenizer = runtime._read_checkpoint(active)
            self.assertEqual(restored_payload["step"], 3)

            shutil.rmtree(active_root, ignore_errors=True)

    def test_dry_run_validates_loadability_without_changing_active_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-dry-run-", dir=BENCHMARK_ROOT) as temp:
            run_root = Path(temp) / "run"
            candidate_source = Path(temp) / "candidate-source.pt"
            digest = self._write_checkpoint(candidate_source, 14)
            staged = self._candidate(run_root, candidate_source, digest)
            report = self._report(run_root, staged, digest)
            active_root = CHECKPOINT_ROOT / (".phase5-dry-run-" + uuid.uuid4().hex)
            active = active_root / "syntra.pt"
            old_digest = self._write_checkpoint(active, 5)
            result = promoter.promote(report, active_checkpoint=active, dry_run=True)
            self.assertTrue(result["dry_run"])
            self.assertEqual(promoter._sha256(active), old_digest)
            shutil.rmtree(active_root, ignore_errors=True)

    def test_unrecognized_archive_id_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".phase5-promotion-archive-id-", dir=CHECKPOINT_ROOT) as temp:
            active = Path(temp) / "syntra.pt"
            with self.assertRaisesRegex(promoter.PromotionError, "valid archive ID"):
                promoter._restore_archive(active, "../latest")


if __name__ == "__main__":
    unittest.main()

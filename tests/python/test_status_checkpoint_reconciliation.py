from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to test checkpoint reconciliation.")
class SftStatusCheckpointReconciliationTests(unittest.TestCase):
    @staticmethod
    def _payload(stage_step: int) -> dict:
        return {
            "format": "syntra-transformer-v1",
            "config": {
                "vocab_size": 300,
                "seq_len": 32,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 32,
            },
            "tokenizer": {"vocab_size": 300},
            "model_state": {"embedding.weight": torch.ones(300, 32)},
            "optimizer_state": {"state": {}, "param_groups": []},
            "finetuning_rng_state": {"cpu": torch.get_rng_state()},
            "step": 10 + stage_step,
            "train_loss": 1.25 - stage_step * 0.01,
            "validation_loss": 1.5 - stage_step * 0.01,
            "training_mode": "supervised-instruction-finetuning",
            "finetuning_resume_schema": "syntra-sft-resume-v1",
            "finetuning_settings": {
                "scheduler": "stage-cosine-warmup-v2",
                "steps": 4,
                "base_learning_rate": 1e-4,
                "batch_size": 1,
            },
            "finetuning_optimizer_scope": "resumable-stage",
            "finetuning_stage_base_step": 10,
            "finetuning_stage_steps": stage_step,
            "finetuning_stage_target_steps": 4,
            "finetuning_validation_stage_step": 1,
            "finetuning_remaining_steps": 4 - stage_step,
            "finetuning_resume_supported": True,
            "finetuning_learning_rate": 5e-5,
            "finetuning_gradient_clip_norm": 1.0,
            "finetuning_gradient_norm_before_clip": 0.4,
            "instruction_examples": 6,
            "instruction_context_windows": 6,
            "instruction_dataset_quality": {"dataset_sha256": "a" * 64},
            "instruction_dataset_sha256": "a" * 64,
            "instruction_dataset_bytes": 1024,
            "training_history": [
                {
                    "step": 11,
                    "phase": "supervised-instruction-finetuning",
                    "train_loss": 1.24,
                    "validation_loss": 1.49,
                }
            ],
        }

    @staticmethod
    def _metadata(identity: dict, stage_step: int) -> dict:
        return {
            "checkpoint_identity": identity,
            "step": 10 + stage_step,
            "config": {"vocab_size": 300, "seq_len": 32},
            "train_loss": 1.25 - stage_step * 0.01,
            "validation_loss": 1.5 - stage_step * 0.01,
            "training_mode": "supervised-instruction-finetuning",
            "finetuning_resume_schema": "syntra-sft-resume-v1",
            "finetuning_settings": {"scheduler": "stage-cosine-warmup-v2", "steps": 4},
            "finetuning_stage_base_step": 10,
            "finetuning_stage_steps": stage_step,
            "finetuning_stage_target_steps": 4,
            "finetuning_validation_stage_step": 1,
            "finetuning_remaining_steps": 4 - stage_step,
            "finetuning_resume_supported": True,
            "training_history": [
                {
                    "step": 11,
                    "phase": "supervised-instruction-finetuning",
                    "train_loss": 1.24,
                    "validation_loss": 1.49,
                }
            ],
        }

    def test_status_repairs_stale_sft_json_from_newer_full_checkpoint(self):
        from syntra_ai.runtime import _file_stat_identity_record, status
        from syntra_ai.training import _atomic_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-status-reconcile-") as temp:
            checkpoint = Path(temp) / "model.pt"
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")

            _atomic_checkpoint(checkpoint, self._payload(stage_step=1))
            old_identity = _file_stat_identity_record(checkpoint)
            metadata_path.write_text(
                json.dumps(self._metadata(old_identity, stage_step=1)),
                encoding="utf-8",
            )

            # Simulate a process interruption after the full recovery checkpoint
            # was atomically committed but before the JSON status write occurred.
            _atomic_checkpoint(checkpoint, self._payload(stage_step=2))
            current_identity = _file_stat_identity_record(checkpoint)
            self.assertNotEqual(current_identity, old_identity)

            with patch.dict(os.environ, {"SYNTRA_CHECKPOINT": str(checkpoint)}):
                recovered = status()
                self.assertEqual(recovered["trained_steps"], 12)
                self.assertEqual(recovered["finetuning_stage_steps"], 2)
                self.assertEqual(recovered["finetuning_remaining_steps"], 2)
                self.assertTrue(recovered["checkpoint_metadata_reconciled"])
                self.assertFalse(recovered["checkpoint_metadata_stale"])
                self.assertTrue(recovered["checkpoint_metadata_identity_verified"])
                self.assertIsNone(recovered["checkpoint_metadata_warning"])

                repaired_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(repaired_metadata["checkpoint_identity"], current_identity)
                self.assertEqual(repaired_metadata["tokenizer_data"], self._payload(stage_step=2)["tokenizer"])
                self.assertEqual(repaired_metadata["finetuning_stage_steps"], 2)
                self.assertEqual(repaired_metadata["finetuning_remaining_steps"], 2)

                second_read = status()

            self.assertEqual(second_read["trained_steps"], 12)
            self.assertFalse(second_read["checkpoint_metadata_reconciled"])
            self.assertFalse(second_read["checkpoint_metadata_stale"])

    def test_status_warns_when_stale_metadata_cannot_be_reconciled(self):
        from syntra_ai.runtime import _file_stat_identity_record, status
        from syntra_ai.training import _atomic_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-status-reconcile-failure-") as temp:
            checkpoint = Path(temp) / "model.pt"
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
            _atomic_checkpoint(checkpoint, self._payload(stage_step=1))
            old_identity = _file_stat_identity_record(checkpoint)
            metadata_path.write_text(
                json.dumps(self._metadata(old_identity, stage_step=1)),
                encoding="utf-8",
            )

            checkpoint.write_bytes(b"not a valid PyTorch checkpoint")
            with patch.dict(os.environ, {"SYNTRA_CHECKPOINT": str(checkpoint)}):
                result = status()

            self.assertTrue(result["checkpoint_metadata_stale"])
            self.assertFalse(result["checkpoint_metadata_reconciled"])
            self.assertFalse(result["checkpoint_metadata_identity_verified"])
            self.assertIn("could not be reconciled", result["checkpoint_metadata_warning"])
            self.assertEqual(result["trained_steps"], 11)



@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to test source checkpoint reconciliation.")
class SourceTrainingStatusReconciliationTests(unittest.TestCase):
    @staticmethod
    def _payload(step: int) -> dict:
        return {
            "format": "syntra-transformer-v1",
            "config": {
                "vocab_size": 300,
                "seq_len": 32,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 32,
            },
            "tokenizer": {"vocab_size": 300},
            "model_state": {"embedding.weight": torch.ones(300, 32)},
            "optimizer_state": {"state": {}, "param_groups": []},
            "rng_state": {"cpu": torch.get_rng_state()},
            "step": step,
            "train_loss": 1.2 - step * 0.01,
            "validation_loss": 1.4 - step * 0.01,
            "learning_rate": 2.5e-4,
            "training_scheduler": "global-cosine-warmup-v2",
            "minimum_learning_rate": 1e-5,
            "warmup_steps": 100,
            "decay_steps": 10_000,
            "gradient_clip_norm": 1.0,
            "gradient_norm_before_clip": 0.5,
            "training_mode": "causal-source-pretraining",
            "training_settings": {
                "batch_size": 1,
                "learning_rate": 3e-4,
                "seed": 31,
                "max_tokens": 4096,
                "scheduler": "global-cosine-warmup-v2",
                "minimum_learning_rate": 1e-5,
                "warmup_steps": 100,
                "decay_steps": 10_000,
                "gradient_clip_norm": 1.0,
            },
            "training_data_sha256": "c" * 64,
            "validation_strategy": "sha256-grouped-document-holdout-v1",
            "corpus_files": 6,
            "corpus_tokens": 1024,
            "training_documents": 5,
            "validation_documents": 1,
            "duplicate_documents_excluded": 0,
            "token_budget_documents_excluded": 0,
            "training_history": [
                {
                    "step": step,
                    "phase": "causal-source-pretraining",
                    "train_loss": 1.2 - step * 0.01,
                    "validation_loss": 1.4 - step * 0.01,
                }
            ],
        }

    def test_status_recovers_stale_source_pretraining_progress(self):
        from syntra_ai.runtime import _file_stat_identity_record, status
        from syntra_ai.training import _atomic_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-source-status-reconcile-") as temp:
            checkpoint = Path(temp) / "source-model.pt"
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
            _atomic_checkpoint(checkpoint, self._payload(step=10))
            previous_identity = _file_stat_identity_record(checkpoint)
            metadata_path.write_text(
                json.dumps({
                    "checkpoint_identity": previous_identity,
                    "step": 10,
                    "training_mode": "causal-source-pretraining",
                    "training_history": self._payload(step=10)["training_history"],
                }),
                encoding="utf-8",
            )

            # Simulate a crash after replacing the full checkpoint but before
            # updating the JSON metadata or the inference companion.
            _atomic_checkpoint(checkpoint, self._payload(step=11))
            current_identity = _file_stat_identity_record(checkpoint)

            with patch.dict(os.environ, {"SYNTRA_CHECKPOINT": str(checkpoint)}):
                recovered = status()
                repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
                second_read = status()

            self.assertEqual(recovered["trained_steps"], 11)
            self.assertEqual(recovered["training_mode"], "causal-source-pretraining")
            self.assertEqual(recovered["training_scheduler"], "global-cosine-warmup-v2")
            self.assertEqual(recovered["validation_strategy"], "sha256-grouped-document-holdout-v1")
            self.assertEqual(recovered["training_documents"], 5)
            self.assertTrue(recovered["checkpoint_metadata_reconciled"])
            self.assertTrue(recovered["checkpoint_metadata_identity_verified"])
            self.assertEqual(repaired["checkpoint_identity"], current_identity)
            self.assertEqual(repaired["training_data_sha256"], "c" * 64)
            self.assertEqual(repaired["training_settings"]["batch_size"], 1)
            self.assertEqual(second_read["trained_steps"], 11)
            self.assertFalse(second_read["checkpoint_metadata_reconciled"])
            self.assertFalse(second_read["checkpoint_metadata_stale"])

    def test_status_can_upgrade_legacy_source_metadata_without_identity(self):
        from syntra_ai.runtime import status
        from syntra_ai.training import _atomic_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-source-legacy-status-") as temp:
            checkpoint = Path(temp) / "source-model.pt"
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
            _atomic_checkpoint(checkpoint, self._payload(step=7))
            metadata_path.write_text(
                json.dumps({
                    "step": 6,
                    "training_mode": "causal-source-pretraining",
                    "training_history": [],
                }),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"SYNTRA_CHECKPOINT": str(checkpoint)}):
                upgraded = status()
                persisted = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(upgraded["trained_steps"], 7)
            self.assertTrue(upgraded["checkpoint_metadata_reconciled"])
            self.assertTrue(upgraded["checkpoint_metadata_identity_verified"])
            self.assertIn("checkpoint_identity", persisted)
            self.assertEqual(persisted["training_data_sha256"], "c" * 64)


if __name__ == "__main__":
    unittest.main()

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


class NativeTrainingSchedulerTests(unittest.TestCase):
    def test_global_cosine_scheduler_warmup_decay_floor_and_resume_boundaries(self):
        from syntra_ai.training import _scheduled_learning_rate

        base = 3e-4
        floor = 1e-5
        warmup_steps = 100
        decay_steps = 10_000

        first = _scheduled_learning_rate(
            base, 1, floor, warmup_steps, decay_steps
        )
        warmup_end = _scheduled_learning_rate(
            base, warmup_steps, floor, warmup_steps, decay_steps
        )
        first_decay = _scheduled_learning_rate(
            base, warmup_steps + 1, floor, warmup_steps, decay_steps
        )
        midpoint = _scheduled_learning_rate(
            base, (warmup_steps + decay_steps) // 2, floor, warmup_steps, decay_steps
        )
        decay_end = _scheduled_learning_rate(
            base, decay_steps, floor, warmup_steps, decay_steps
        )
        after_end = _scheduled_learning_rate(
            base, decay_steps + 500, floor, warmup_steps, decay_steps
        )

        self.assertGreater(first, floor)
        self.assertLess(first, base)
        self.assertAlmostEqual(warmup_end, base)
        self.assertLess(first_decay, warmup_end)
        self.assertLess(midpoint, first_decay)
        self.assertAlmostEqual(decay_end, floor)
        self.assertAlmostEqual(after_end, floor)
        uninterrupted = [
            _scheduled_learning_rate(base, step, floor, warmup_steps, decay_steps)
            for step in range(1, 201)
        ]
        split_before_restart = [
            _scheduled_learning_rate(base, step, floor, warmup_steps, decay_steps)
            for step in range(1, 101)
        ]
        split_after_restart = [
            _scheduled_learning_rate(base, step, floor, warmup_steps, decay_steps)
            for step in range(101, 201)
        ]
        self.assertEqual(
            uninterrupted,
            split_before_restart + split_after_restart,
            "segmenting training at a process boundary must not restart the global schedule",
        )

    def test_cosine_scheduler_rejects_invalid_or_incompatible_settings(self):
        from syntra_ai.training import _scheduled_learning_rate

        invalid = (
            (0, 1e-5, 100, 10_000),
            (3e-4, 4e-4, 100, 10_000),
            (3e-4, 1e-5, 0, 10_000),
            (3e-4, 1e-5, 100, 100),
        )
        for base, floor, warmup, decay in invalid:
            with self.subTest(base=base, floor=floor, warmup=warmup, decay=decay):
                with self.assertRaises(ValueError):
                    _scheduled_learning_rate(base, 1, floor, warmup, decay)

        with self.assertRaisesRegex(ValueError, "global_step"):
            _scheduled_learning_rate(3e-4, 0)

    def test_pre_cosine_schedule_preserves_old_application_behavior(self):
        from syntra_ai.training import _legacy_scheduled_learning_rate

        base = 3e-4
        self.assertAlmostEqual(_legacy_scheduled_learning_rate(base, 1), base / 100)
        self.assertAlmostEqual(_legacy_scheduled_learning_rate(base, 100), base)
        self.assertAlmostEqual(_legacy_scheduled_learning_rate(base, 101), base)


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run native training tests.")
class NativeTrainingSchedulerCheckpointTests(unittest.TestCase):
    @staticmethod
    def _source_documents(root: Path) -> None:
        for index in range(6):
            source = (
                f"def handler_{index}(value):\n"
                f"    return value + {index}\n"
                f"# independent source document {index}\n"
            ) * 5
            (root / f"source-{index}.py").write_text(source, encoding="utf-8")

    @staticmethod
    def _config():
        from syntra_ai.config import ModelConfig

        return ModelConfig(
            vocab_size=300,
            seq_len=16,
            n_layer=1,
            n_head=1,
            n_embd=32,
            dropout=0.0,
        )

    def test_checkpoint_persists_cosine_schedule_and_global_step_resume(self):
        from syntra_ai.checkpoint_io import load_torch_checkpoint_snapshot
        from syntra_ai.runtime import status
        from syntra_ai.training import (
            SOURCE_GRADIENT_CLIP_NORM,
            SOURCE_MINIMUM_LEARNING_RATE,
            SOURCE_SCHEDULER_DECAY_STEPS,
            SOURCE_SCHEDULER_WARMUP_STEPS,
            SOURCE_TRAINING_SCHEDULER,
            _scheduled_learning_rate,
            train,
        )

        with tempfile.TemporaryDirectory(prefix="syntra-cosine-checkpoint-") as temp:
            root = Path(temp) / "corpus"
            root.mkdir()
            self._source_documents(root)
            checkpoint = Path(temp) / "model.pt"
            environment = {
                "SYNTRA_ALLOWED_CORPUS_ROOTS": str(root),
                "SYNTRA_CHECKPOINT": str(checkpoint),
            }
            with patch.dict(os.environ, environment):
                first = train(
                    roots=[str(root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=self._config(),
                    learning_rate=3e-4,
                    max_tokens=4096,
                    device_name="cpu",
                    seed=31,
                )
                saved = load_torch_checkpoint_snapshot(checkpoint)
                model_status = status()

                self.assertEqual(first["status"], "completed")
                self.assertEqual(first["training_scheduler"], SOURCE_TRAINING_SCHEDULER)
                self.assertEqual(first["learning_rate"], _scheduled_learning_rate(3e-4, 1))
                self.assertGreaterEqual(first["gradient_norm_before_clip"], 0.0)
                self.assertEqual(first["gradient_clip_norm"], SOURCE_GRADIENT_CLIP_NORM)
                self.assertEqual(saved["training_settings"]["scheduler"], SOURCE_TRAINING_SCHEDULER)
                self.assertEqual(saved["training_settings"]["minimum_learning_rate"], SOURCE_MINIMUM_LEARNING_RATE)
                self.assertEqual(saved["training_settings"]["warmup_steps"], SOURCE_SCHEDULER_WARMUP_STEPS)
                self.assertEqual(saved["training_settings"]["decay_steps"], SOURCE_SCHEDULER_DECAY_STEPS)
                self.assertEqual(saved["training_settings"]["gradient_clip_norm"], SOURCE_GRADIENT_CLIP_NORM)
                self.assertEqual(saved["learning_rate"], first["learning_rate"])
                self.assertEqual(saved["gradient_norm_before_clip"], first["gradient_norm_before_clip"])
                self.assertEqual(model_status["training_scheduler"], SOURCE_TRAINING_SCHEDULER)
                self.assertEqual(model_status["learning_rate"], first["learning_rate"])
                self.assertEqual(model_status["gradient_clip_norm"], SOURCE_GRADIENT_CLIP_NORM)
                self.assertTrue(model_status["checkpoint_metadata_identity_verified"])
                metadata = json.loads(
                    checkpoint.with_suffix(checkpoint.suffix + ".json").read_text(encoding="utf-8")
                )
                from syntra_ai.runtime import _file_stat_identity_record
                self.assertEqual(metadata["checkpoint_identity"], _file_stat_identity_record(checkpoint))

                resumed = train(
                    roots=[str(root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=self._config(),
                    learning_rate=3e-4,
                    max_tokens=4096,
                    device_name="cpu",
                    seed=31,
                    resume=True,
                )

            self.assertEqual(resumed["step"], 2)
            self.assertEqual(resumed["training_scheduler"], SOURCE_TRAINING_SCHEDULER)
            self.assertAlmostEqual(
                resumed["learning_rate"],
                _scheduled_learning_rate(3e-4, 2),
            )

    def test_checkpoint_without_scheduler_metadata_resumes_legacy_linear_warmup(self):
        from syntra_ai.checkpoint_io import load_torch_checkpoint_snapshot
        from syntra_ai.training import (
            LEGACY_SOURCE_TRAINING_SCHEDULER,
            train,
        )

        with tempfile.TemporaryDirectory(prefix="syntra-legacy-scheduler-") as temp:
            root = Path(temp) / "corpus"
            root.mkdir()
            self._source_documents(root)
            checkpoint = Path(temp) / "model.pt"
            environment = {
                "SYNTRA_ALLOWED_CORPUS_ROOTS": str(root),
                "SYNTRA_CHECKPOINT": str(checkpoint),
            }
            with patch.dict(os.environ, environment):
                train(
                    roots=[str(root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=self._config(),
                    learning_rate=3e-4,
                    max_tokens=4096,
                    device_name="cpu",
                    seed=37,
                )
                saved = load_torch_checkpoint_snapshot(checkpoint)
                settings = saved["training_settings"]
                for key in (
                    "scheduler",
                    "minimum_learning_rate",
                    "warmup_steps",
                    "decay_steps",
                    "gradient_clip_norm",
                ):
                    settings.pop(key, None)
                torch.save(saved, checkpoint)

                resumed = train(
                    roots=[str(root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=self._config(),
                    learning_rate=3e-4,
                    max_tokens=4096,
                    device_name="cpu",
                    seed=37,
                    resume=True,
                )
                resumed_checkpoint = load_torch_checkpoint_snapshot(checkpoint)

        self.assertEqual(resumed["step"], 2)
        self.assertEqual(resumed["training_scheduler"], LEGACY_SOURCE_TRAINING_SCHEDULER)
        self.assertAlmostEqual(resumed["learning_rate"], 3e-4 * 2 / 100)
        self.assertEqual(
            resumed_checkpoint["training_settings"]["scheduler"],
            LEGACY_SOURCE_TRAINING_SCHEDULER,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run SFT resume tests.")
class FineTuningResumeValidationTests(unittest.TestCase):
    def _checkpoint_state(self):
        from syntra_ai.instruction import (
            SFT_RESUME_SCHEMA,
            _finetuning_scheduler_settings,
        )

        settings = _finetuning_scheduler_settings(
            steps=5,
            learning_rate=1e-4,
            batch_size=2,
            gradient_accumulation_steps=3,
        )
        dataset_quality = {
            "dataset_sha256": "a" * 64,
            "dataset_bytes": 4096,
        }
        saved = {
            "training_mode": "supervised-instruction-finetuning",
            "finetuning_resume_schema": SFT_RESUME_SCHEMA,
            "finetuning_resume_supported": True,
            "finetuning_settings": settings,
            "instruction_dataset_sha256": dataset_quality["dataset_sha256"],
            "instruction_dataset_bytes": dataset_quality["dataset_bytes"],
            "optimizer_state": {"state": {}, "param_groups": []},
            "finetuning_rng_state": {"cpu": torch.get_rng_state()},
            "finetuning_device_type": "cpu",
            "finetuning_cuda_device_count": 0,
            "finetuning_stage_base_step": 10,
            "finetuning_stage_steps": 2,
            "step": 12,
        }
        return saved, settings, dataset_quality

    def test_valid_checkpoint_restores_the_saved_stage_position(self):
        from syntra_ai.instruction import _validate_finetuning_resume_state

        saved, settings, dataset_quality = self._checkpoint_state()
        self.assertEqual(
            _validate_finetuning_resume_state(
                saved,
                settings,
                dataset_quality,
                torch.device("cpu"),
            ),
            (10, 2),
        )

    def test_resume_rejects_changed_dataset_or_optimizer_settings(self):
        from syntra_ai.instruction import _validate_finetuning_resume_state

        saved, settings, dataset_quality = self._checkpoint_state()
        changed_dataset = dict(dataset_quality, dataset_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "instruction dataset differs"):
            _validate_finetuning_resume_state(
                saved, settings, changed_dataset, torch.device("cpu")
            )

        changed_settings = dict(settings, batch_size=4)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            _validate_finetuning_resume_state(
                saved, changed_settings, dataset_quality, torch.device("cpu")
            )

    def test_resume_rejects_missing_optimizer_rng_or_legacy_resume_schema(self):
        from syntra_ai.instruction import _validate_finetuning_resume_state

        saved, settings, dataset_quality = self._checkpoint_state()
        for field in ("optimizer_state", "finetuning_rng_state", "finetuning_resume_schema"):
            broken = dict(saved)
            broken.pop(field)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    _validate_finetuning_resume_state(
                        broken, settings, dataset_quality, torch.device("cpu")
                    )

    def test_resume_rejects_device_or_global_step_mismatch(self):
        from syntra_ai.instruction import _validate_finetuning_resume_state

        saved, settings, dataset_quality = self._checkpoint_state()
        wrong_device = dict(saved, finetuning_device_type="cuda")
        with self.assertRaisesRegex(ValueError, "device type"):
            _validate_finetuning_resume_state(
                wrong_device, settings, dataset_quality, torch.device("cpu")
            )

        wrong_step = dict(saved, step=99)
        with self.assertRaisesRegex(ValueError, "global optimizer-step"):
            _validate_finetuning_resume_state(
                wrong_step, settings, dataset_quality, torch.device("cpu")
            )

    def test_completed_stage_cannot_be_resumed(self):
        from syntra_ai.instruction import _validate_finetuning_resume_state

        saved, settings, dataset_quality = self._checkpoint_state()
        complete = dict(
            saved,
            finetuning_stage_steps=settings["steps"],
            step=saved["finetuning_stage_base_step"] + settings["steps"],
        )
        with self.assertRaisesRegex(ValueError, "already complete"):
            _validate_finetuning_resume_state(
                complete, settings, dataset_quality, torch.device("cpu")
            )


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run native SFT integration tests.")
class FineTuningResumeIntegrationTests(unittest.TestCase):
    @staticmethod
    def _copy_training_checkpoint(source: Path, destination: Path) -> Path:
        from syntra_ai.training import inference_checkpoint_path

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        shutil.copy2(inference_checkpoint_path(source), inference_checkpoint_path(destination))
        shutil.copy2(source.with_suffix(source.suffix + ".json"), destination.with_suffix(destination.suffix + ".json"))
        return destination

    @staticmethod
    def _assert_tree_equal(test: unittest.TestCase, left, right, path: str = "root") -> None:
        if torch.is_tensor(left):
            test.assertTrue(torch.is_tensor(right), path)
            test.assertEqual(left.dtype, right.dtype, path)
            test.assertEqual(left.shape, right.shape, path)
            test.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            test.assertIsInstance(right, dict, path)
            test.assertEqual(set(left), set(right), path)
            for key in left:
                FineTuningResumeIntegrationTests._assert_tree_equal(
                    test, left[key], right[key], f"{path}.{key}"
                )
        elif isinstance(left, (list, tuple)):
            test.assertIsInstance(right, type(left), path)
            test.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                FineTuningResumeIntegrationTests._assert_tree_equal(
                    test, left_item, right_item, f"{path}[{index}]"
                )
        else:
            test.assertEqual(left, right, path)

    @staticmethod
    def _write_instruction_dataset(path: Path) -> None:
        records = [
            {"prompt": "Write add function", "completion": "def add(a, b): return a + b"},
            {"prompt": "Write maximum function", "completion": "def maximum(a, b): return max(a, b)"},
            {"prompt": "Write minimum function", "completion": "def minimum(a, b): return min(a, b)"},
            {"prompt": "Write normalize function", "completion": "def normalize(value): return value / 100"},
            {"prompt": "Write clamp function", "completion": "def clamp(x, low, high): return min(max(x, low), high)"},
            {"prompt": "Write square function", "completion": "def square(value): return value * value"},
        ]
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_interrupted_stage_resumes_to_the_same_weights_and_optimizer_as_uninterrupted(self):
        from syntra_ai.config import ModelConfig
        from syntra_ai.instruction import finetune
        from syntra_ai.training import train

        class StopAfterFirstCommittedUpdate(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix="syntra-sft-resume-") as temp:
            root = Path(temp)
            source_root = root / "source"
            source_root.mkdir()
            (source_root / "math.py").write_text(
                ("def add(a, b): return a + b\n" "def subtract(a, b): return a - b\n") * 40,
                encoding="utf-8",
            )
            (source_root / "strings.py").write_text(
                ("def trim(value): return value.strip()\n" "def upper(value): return value.upper()\n") * 40,
                encoding="utf-8",
            )
            dataset = root / "instructions.jsonl"
            self._write_instruction_dataset(dataset)
            bootstrap = root / "bootstrap" / "model.pt"
            environment = {
                "SYNTRA_ALLOWED_CORPUS_ROOTS": str(root),
                "SYNTRA_CHECKPOINT": str(bootstrap),
            }

            with patch.dict(os.environ, environment):
                train(
                    roots=[str(source_root)],
                    checkpoint=bootstrap,
                    steps=1,
                    batch_size=1,
                    config=ModelConfig(
                        vocab_size=300,
                        seq_len=32,
                        n_layer=1,
                        n_head=1,
                        n_embd=32,
                        dropout=0.0,
                    ),
                    learning_rate=3e-4,
                    max_tokens=20_000,
                    device_name="cpu",
                    seed=41,
                )

            continuous_checkpoint = self._copy_training_checkpoint(
                bootstrap, root / "continuous" / "model.pt"
            )
            interrupted_checkpoint = self._copy_training_checkpoint(
                bootstrap, root / "interrupted" / "model.pt"
            )

            torch.manual_seed(123456)
            initial_rng_state = torch.get_rng_state().clone()
            settings = {
                "steps": 3,
                "batch_size": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1e-4,
                "device_name": "cpu",
            }

            with patch.dict(os.environ, environment):
                torch.set_rng_state(initial_rng_state)
                uninterrupted = finetune(
                    dataset_path=dataset,
                    checkpoint=continuous_checkpoint,
                    **settings,
                )

                torch.set_rng_state(initial_rng_state)

                def interrupt_after_saved_step(progress: dict) -> None:
                    if (
                        progress.get("status") == "running"
                        and progress.get("finetuning_stage_steps") == 1
                        and progress.get("finetuning_remaining_steps") == 2
                    ):
                        raise StopAfterFirstCommittedUpdate()

                with self.assertRaises(StopAfterFirstCommittedUpdate):
                    finetune(
                        dataset_path=dataset,
                        checkpoint=interrupted_checkpoint,
                        callback=interrupt_after_saved_step,
                        **settings,
                    )

                partial = torch.load(
                    interrupted_checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(partial["finetuning_stage_steps"], 1)
                self.assertIn("optimizer_state", partial)
                self.assertIn("finetuning_rng_state", partial)
                self.assertEqual(partial["step"], partial["finetuning_stage_base_step"] + 1)

                # The caller's current process RNG is deliberately changed.
                # Resume must restore the checkpoint's actual CPU RNG state.
                torch.manual_seed(999999)
                resumed = finetune(
                    dataset_path=dataset,
                    checkpoint=interrupted_checkpoint,
                    resume=True,
                    **settings,
                )

            self.assertEqual(uninterrupted["status"], "completed")
            self.assertEqual(resumed["status"], "completed")
            self.assertTrue(resumed["finetuning_resume"])
            self.assertEqual(resumed["finetuning_stage_steps"], 3)
            self.assertEqual(resumed["finetuning_remaining_steps"], 0)

            continuous_saved = torch.load(
                continuous_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            resumed_saved = torch.load(
                interrupted_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            self._assert_tree_equal(
                self,
                continuous_saved["model_state"],
                resumed_saved["model_state"],
                "model_state",
            )
            self._assert_tree_equal(
                self,
                continuous_saved["optimizer_state"],
                resumed_saved["optimizer_state"],
                "optimizer_state",
            )
            self.assertEqual(
                continuous_saved["finetuning_rng_state"]["cpu"].tolist(),
                resumed_saved["finetuning_rng_state"]["cpu"].tolist(),
            )
            self.assertEqual(
                continuous_saved["training_history"],
                resumed_saved["training_history"],
            )

    def test_resume_rejects_dataset_changes_and_completed_stages(self):
        from syntra_ai.config import ModelConfig
        from syntra_ai.instruction import finetune
        from syntra_ai.training import train

        class StopAfterFirstCommittedUpdate(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix="syntra-sft-resume-reject-") as temp:
            root = Path(temp)
            source_root = root / "source"
            source_root.mkdir()
            (source_root / "one.py").write_text(
                ("def one(value): return value + 1\n") * 60,
                encoding="utf-8",
            )
            (source_root / "two.py").write_text(
                ("def two(value): return value * 2\n") * 60,
                encoding="utf-8",
            )
            dataset = root / "instructions.jsonl"
            self._write_instruction_dataset(dataset)
            checkpoint = root / "checkpoint" / "model.pt"
            environment = {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}
            with patch.dict(os.environ, environment):
                train(
                    roots=[str(source_root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=ModelConfig(
                        vocab_size=300,
                        seq_len=32,
                        n_layer=1,
                        n_head=1,
                        n_embd=32,
                        dropout=0.0,
                    ),
                    max_tokens=20_000,
                    device_name="cpu",
                    seed=53,
                )

                def interrupt(progress: dict) -> None:
                    if progress.get("finetuning_stage_steps") == 1 and progress.get("status") == "running":
                        raise StopAfterFirstCommittedUpdate()

                with self.assertRaises(StopAfterFirstCommittedUpdate):
                    finetune(
                        dataset,
                        checkpoint=checkpoint,
                        steps=3,
                        batch_size=1,
                        learning_rate=1e-4,
                        device_name="cpu",
                        callback=interrupt,
                    )

                original = dataset.read_text(encoding="utf-8")
                dataset.write_text(original + json.dumps({
                    "prompt": "Write another function",
                    "completion": "def another(): return 1",
                }) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "instruction dataset differs"):
                    finetune(
                        dataset,
                        checkpoint=checkpoint,
                        steps=3,
                        batch_size=1,
                        learning_rate=1e-4,
                        device_name="cpu",
                        resume=True,
                    )
                dataset.write_text(original, encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "batch_size"):
                    finetune(
                        dataset,
                        checkpoint=checkpoint,
                        steps=3,
                        batch_size=2,
                        learning_rate=1e-4,
                        device_name="cpu",
                        resume=True,
                    )

                finetune(
                    dataset,
                    checkpoint=checkpoint,
                    steps=3,
                    batch_size=1,
                    learning_rate=1e-4,
                    device_name="cpu",
                    resume=True,
                )
                with self.assertRaisesRegex(ValueError, "already complete"):
                    finetune(
                        dataset,
                        checkpoint=checkpoint,
                        steps=3,
                        batch_size=1,
                        learning_rate=1e-4,
                        device_name="cpu",
                        resume=True,
                    )


if __name__ == "__main__":
    unittest.main()

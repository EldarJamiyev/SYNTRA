import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from benchmark_engine import runner
from syntra_ai.training_history import (
    MAX_TRAINING_HISTORY_POINTS,
    append_training_history,
    validate_training_history,
)


class TrainingHistoryTests(unittest.TestCase):
    def test_history_rejects_non_finite_and_non_monotonic_values(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_training_history([
                {"step": 2, "phase": "pretrain", "train_loss": 1.0, "validation_loss": 1.2},
                {"step": 2, "phase": "pretrain", "train_loss": 0.9, "validation_loss": 1.1},
            ])
        with self.assertRaisesRegex(ValueError, "invalid training loss"):
            validate_training_history([
                {"step": 1, "phase": "pretrain", "train_loss": float("nan"), "validation_loss": 1.2},
            ])

    def test_history_appends_measured_points_and_keeps_a_bounded_window(self):
        history = []
        for step in range(1, MAX_TRAINING_HISTORY_POINTS + 9):
            history = append_training_history(
                history,
                step=step,
                phase="causal-source-pretraining",
                train_loss=2.0 / step,
                validation_loss=2.5 / step,
            )
        self.assertEqual(len(history), MAX_TRAINING_HISTORY_POINTS)
        self.assertEqual(history[0]["step"], 9)
        self.assertEqual(history[-1]["step"], MAX_TRAINING_HISTORY_POINTS + 8)
        with self.assertRaisesRegex(ValueError, "follow the saved history"):
            append_training_history(
                history,
                step=history[-1]["step"],
                phase="causal-source-pretraining",
                train_loss=0.1,
                validation_loss=0.2,
            )


class RawModelBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = runner.load_suite()

    def test_suite_has_five_unique_held_out_tasks(self):
        self.assertEqual(self.suite["version"], "1.0.0")
        self.assertEqual(len(self.suite["tasks"]), 5)
        self.assertEqual(len({task["id"] for task in self.suite["tasks"]}), 5)
        self.assertTrue(all(task["hidden_evaluation"].startswith("hidden/") for task in self.suite["tasks"]))

    def test_suite_rejects_hidden_paths_that_escape_the_evaluation_directory(self):
        invalid = copy.deepcopy(self.suite)
        invalid["tasks"][0]["hidden_evaluation"] = "hidden/../fixtures"
        with self.assertRaisesRegex(runner.BenchmarkError, "unsafe hidden_evaluation"):
            runner.validate_suite(invalid)

    def test_hidden_tests_are_not_copied_into_the_model_prompt_workspace(self):
        task = self.suite["tasks"][0]
        with tempfile.TemporaryDirectory(prefix="syntra-benchmark-visibility-") as temp:
            workspace = runner._copy_fixture_workspace(task, Path(temp))
            source_files = runner.collect_sources(workspace)
            self.assertEqual([path for path, _ in source_files], ["src/slug.py"])
            self.assertFalse(any("test_target" in path or "test_regression" in path for path, _ in source_files))
            hidden = (runner.ENGINE_ROOT / task["hidden_evaluation"]).resolve()
            self.assertTrue((hidden / task["target_test"]).is_file())
            self.assertFalse((workspace / task["target_test"]).exists())

    def test_every_fixture_has_a_failing_target_and_passing_regression_baseline(self):
        isolation = {"mode": "host-opt-in", "image": None, "image_id": None}
        for task in self.suite["tasks"]:
            with self.subTest(task=task["id"]), tempfile.TemporaryDirectory(prefix="syntra-benchmark-baseline-") as temp:
                workspace = runner._copy_fixture_workspace(task, Path(temp))
                hidden = (runner.ENGINE_ROOT / task["hidden_evaluation"]).resolve()
                target = runner.run_hidden_test(
                    isolation=isolation, workspace=workspace, hidden_dir=hidden,
                    test_name=task["target_test"], run_id="baseline-test", phase="target",
                )
                regression = runner.run_hidden_test(
                    isolation=isolation, workspace=workspace, hidden_dir=hidden,
                    test_name=task["regression_test"], run_id="baseline-test", phase="regression",
                )
                self.assertTrue(runner._expected_baseline_failure(target), target["stdout"] + target["stderr"])
                self.assertEqual(regression["returncode"], 0, regression["stdout"] + regression["stderr"])

    def test_patch_parser_rejects_extra_text_and_duplicate_json_keys(self):
        valid = {
            "format": "SYNTRA_AGENT_PATCH_V1",
            "summary": "Implement an exact source fix.",
            "edits": [{
                "path": "src/module.py", "find": "return value",
                "replace": "return value + 1", "reason": "Correct the result.",
            }],
        }
        parsed = runner.parse_patch_output(json.dumps(valid))
        self.assertEqual(parsed["format"], "SYNTRA_AGENT_PATCH_V1")
        with self.assertRaisesRegex(runner.BenchmarkError, "strict JSON"):
            runner.parse_patch_output(json.dumps(valid) + " trailing text")
        with self.assertRaisesRegex(runner.BenchmarkError, "strict JSON"):
            runner.parse_patch_output('{"format":"SYNTRA_AGENT_PATCH_V1","format":"OTHER","summary":"x","edits":[]}')

    def test_candidate_patch_uses_the_existing_agent_patch_validator(self):
        with tempfile.TemporaryDirectory(prefix="syntra-benchmark-patch-") as temp:
            workspace = Path(temp)
            source = workspace / "src" / "module.py"
            source.parent.mkdir()
            source.write_text("def value():\n    return 1\n", encoding="utf-8")
            patch = {
                "format": "SYNTRA_AGENT_PATCH_V1",
                "summary": "Correct the returned constant.",
                "edits": [{
                    "path": "src/module.py", "find": "return 1",
                    "replace": "return 2", "reason": "Return the requested value.",
                }],
            }
            changed = runner.apply_patch(workspace, patch, ["src/module.py"])
            self.assertEqual(changed, ["src/module.py"])
            self.assertEqual(source.read_text(encoding="utf-8"), "def value():\n    return 2\n")

            unsafe = copy.deepcopy(patch)
            unsafe["edits"][0]["path"] = "../outside.py"
            with self.assertRaises(runner.BenchmarkError):
                runner.apply_patch(workspace, unsafe, ["../outside.py"])
            self.assertEqual(source.read_text(encoding="utf-8"), "def value():\n    return 2\n")

    def test_checkpoint_loss_history_is_read_from_the_actual_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-evidence-") as temp:
            checkpoint = Path(temp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint-bytes-used-for-identity")
            sidecar = checkpoint.with_suffix(".pt.json")
            sidecar.write_text(json.dumps({
                "step": 100,
                "train_loss": 0.42,
                "validation_loss": 0.76,
                "best_validation_loss": 0.71,
                "training_mode": "supervised-instruction-finetuning",
                "training_history": [
                    {"step": 50, "phase": "causal-source-pretraining", "train_loss": 1.2, "validation_loss": 1.4},
                    {"step": 100, "phase": "supervised-instruction-finetuning", "train_loss": 0.42, "validation_loss": 0.76},
                ],
            }), encoding="utf-8")
            evidence = runner.load_checkpoint_evidence(checkpoint)
            self.assertEqual(evidence["step"], 100)
            self.assertAlmostEqual(evidence["train_loss"], 0.42)
            self.assertEqual([point["phase"] for point in evidence["history"]], [
                "causal-source-pretraining", "supervised-instruction-finetuning",
            ])
            self.assertTrue(evidence["history_available"])

    def test_loss_intelligence_divergence_requires_low_loss_and_at_least_three_zero_success_tasks(self):
        checkpoint = {"train_loss": 0.4}
        summary = {"task_count": 5, "task_completion_rate": 0.0}
        warning = runner.assess_loss_intelligence_divergence(summary, checkpoint)
        self.assertTrue(warning["detected"])
        self.assertEqual(warning["flag"], "OVERFITTING_LOSS_INTELLIGENCE_DIVERGENCE")
        self.assertIn("not proof", warning["interpretation"])

        too_small = runner.assess_loss_intelligence_divergence(
            {"task_count": 2, "task_completion_rate": 0.0}, checkpoint,
        )
        self.assertFalse(too_small["detected"])
        high_loss = runner.assess_loss_intelligence_divergence(
            {"task_count": 5, "task_completion_rate": 0.0}, {"train_loss": 1.5},
        )
        self.assertFalse(high_loss["detected"])

    def test_summary_keeps_target_completion_and_regression_prevention_separate(self):
        results = [
            {"task_completion": True, "patch_correctness": True, "regression_prevention": False, "execution_speed": {"total_ms": 100, "generation_ms": 50, "generated_tokens": 10}},
            {"task_completion": False, "patch_correctness": False, "regression_prevention": None, "execution_speed": {"total_ms": 200, "generation_ms": None, "generated_tokens": 0}},
        ]
        summary = runner.summarize(results)
        self.assertEqual(summary["task_completion_rate"], 0.5)
        self.assertEqual(summary["patch_correctness_rate"], 0.5)
        self.assertEqual(summary["regression_prevention_rate"], 0.0)
        self.assertEqual(summary["fully_verified_task_count"], 0)


if __name__ == "__main__":
    unittest.main()

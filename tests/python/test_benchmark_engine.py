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
        with self.assertRaisesRegex(ValueError, "invalid training loss"):
            validate_training_history([
                {"step": 1, "phase": "pretrain", "train_loss": 10 ** 10000, "validation_loss": 1.2},
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

    def test_suite_has_thirteen_unique_held_out_tasks_across_five_languages(self):
        self.assertEqual(self.suite["version"], "1.3.0")
        self.assertEqual(len(self.suite["tasks"]), 13)
        self.assertEqual(len({task["id"] for task in self.suite["tasks"]}), 13)
        self.assertEqual({task["language"] for task in self.suite["tasks"]}, {"python", "javascript", "typescript", "go", "rust"})
        self.assertEqual(
            {language: sum(task["language"] == language for task in self.suite["tasks"]) for language in {"python", "javascript", "typescript", "go", "rust"}},
            {"python": 9, "javascript": 1, "typescript": 1, "go": 1, "rust": 1},
        )
        self.assertTrue(all(task["hidden_evaluation"].startswith("hidden/") for task in self.suite["tasks"]))

    def test_environment_matrix_uses_pinned_offline_images_and_language_test_extensions(self):
        expected = {
            "python": ("python:3.12-slim", ".py"),
            "javascript": ("node:20-slim", ".mjs"),
            "typescript": ("node:20-slim", ".ts"),
            "go": ("golang:1.22-alpine", ".go"),
            "rust": ("rust:1.78-slim", ".rs"),
        }
        for language, (image, extension) in expected.items():
            environment = runner.resolve_environment(language)
            self.assertEqual(environment["image"], image)
            self.assertEqual(runner.expected_test_filename(language, "target"), "test_target" + extension)
            self.assertEqual(runner.expected_test_filename(language, "regression"), "test_regression" + extension)
        self.assertEqual(runner.resolve_environment("go", {"go": "golang:1.22-alpine"})["image"], "golang:1.22-alpine")
        self.assertEqual(runner.resolve_environment("javascript", {"javascript": "node:20-slim"})["image"], "node:20-slim")
        with self.assertRaises(ValueError):
            runner.resolve_environment("unknown")

    def test_suite_rejects_a_test_extension_that_does_not_match_its_language(self):
        invalid = copy.deepcopy(self.suite)
        invalid["tasks"][8]["target_test"] = "test_target.py"
        with self.assertRaisesRegex(runner.BenchmarkError, "filename for typescript"):
            runner.validate_suite(invalid)

    def test_suite_rejects_hidden_paths_that_escape_the_evaluation_directory(self):
        invalid = copy.deepcopy(self.suite)
        invalid["tasks"][0]["hidden_evaluation"] = "hidden/../fixtures"
        with self.assertRaisesRegex(runner.BenchmarkError, "unsafe hidden_evaluation"):
            runner.validate_suite(invalid)

    def test_suite_rejects_noncanonical_and_duplicate_changed_paths(self):
        noncanonical = copy.deepcopy(self.suite)
        noncanonical["tasks"][0]["expected_changed_paths"] = ["src//slug.py"]
        with self.assertRaisesRegex(runner.BenchmarkError, "unsafe changed path"):
            runner.validate_suite(noncanonical)
        duplicate = copy.deepcopy(self.suite)
        duplicate["tasks"][0]["expected_changed_paths"] = ["src/slug.py", "src/slug.py"]
        with self.assertRaisesRegex(runner.BenchmarkError, "duplicate changed paths"):
            runner.validate_suite(duplicate)

    def test_suite_rejects_a_task_count_that_does_not_match_the_manifest(self):
        invalid = copy.deepcopy(self.suite)
        invalid["task_count"] -= 1
        with self.assertRaisesRegex(runner.BenchmarkError, "task_count"):
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

    def test_collect_sources_includes_multi_file_and_polyglot_context(self):
        expected = {
            "typescript-object-mapper": {"src/mapper.ts", "src/types.ts", "package.json"},
            "go-channel-sync": {"go.mod", "syncutil/collector.go"},
            "rust-safe-slice": {"Cargo.toml", "Cargo.lock", "src/lib.rs"},
            "python-multi-file-stream-state": {"src/limits.py", "src/state.py"},
            "javascript-event-normalizer": {"package.json", "src/schema.mjs", "src/normalize-event.mjs"},
        }
        by_id = {task["id"]: task for task in self.suite["tasks"]}
        for task_id, wanted in expected.items():
            with self.subTest(task=task_id), tempfile.TemporaryDirectory(prefix="syntra-polyglot-context-") as temp:
                workspace = runner._copy_fixture_workspace(by_id[task_id], Path(temp))
                paths = {path for path, _ in runner.collect_sources(workspace)}
                self.assertTrue(wanted.issubset(paths), (task_id, paths))

    def test_build_prompt_exposes_all_sibling_modules_as_connected_context(self):
        task = next(task for task in self.suite["tasks"] if task["id"] == "python-multi-file-stream-state")
        with tempfile.TemporaryDirectory(prefix="syntra-multifile-prompt-") as temp:
            workspace = runner._copy_fixture_workspace(task, Path(temp))
            prompt = runner.build_prompt(task, runner.collect_sources(workspace))
            self.assertIn("FILE: src/limits.py", prompt)
            self.assertIn("FILE: src/state.py", prompt)
            self.assertIn("single JSON response", prompt)
            self.assertNotIn("test_target.py", prompt)

    def test_javascript_runtime_checks_and_executes_hidden_modules_against_workspace(self):
        isolation = {"mode": "host-opt-in", "image": None, "image_id": None}
        with tempfile.TemporaryDirectory(prefix="syntra-javascript-runtime-") as temp:
            workspace = Path(temp)
            source = workspace / "src" / "value.mjs"
            source.parent.mkdir()
            source.write_text("export function readValue() { return 41; }", encoding="utf-8")
            syntax = runner.syntax_check(
                workspace, ["src/value.mjs"], language="javascript",
                isolation=isolation, run_id="javascript-smoke",
            )
            self.assertTrue(syntax["passed"], syntax["error"])
            hidden = (runner.ENGINE_ROOT / "hidden" / "javascript-runtime").resolve()
            target = runner.run_hidden_test(
                isolation=isolation, workspace=workspace, hidden_dir=hidden,
                test_name="test_target.mjs", run_id="javascript-smoke", phase="target",
                language="javascript",
            )
            regression = runner.run_hidden_test(
                isolation=isolation, workspace=workspace, hidden_dir=hidden,
                test_name="test_regression.mjs", run_id="javascript-smoke", phase="regression",
                language="javascript",
            )
            self.assertEqual(target["returncode"], 0, target["stdout"] + target["stderr"])
            self.assertEqual(regression["returncode"], 0, regression["stdout"] + regression["stderr"])

    def test_every_fixture_has_a_failing_target_and_passing_regression_baseline(self):
        isolation = {"mode": "host-opt-in", "image": None, "image_id": None}
        for task in self.suite["tasks"]:
            with self.subTest(task=task["id"]), tempfile.TemporaryDirectory(prefix="syntra-benchmark-baseline-") as temp:
                workspace = runner._copy_fixture_workspace(task, Path(temp))
                hidden = (runner.ENGINE_ROOT / task["hidden_evaluation"]).resolve()
                target = runner.run_hidden_test(
                    isolation=isolation, workspace=workspace, hidden_dir=hidden,
                    test_name=task["target_test"], run_id="baseline-test", phase="target", language=task["language"],
                )
                regression = runner.run_hidden_test(
                    isolation=isolation, workspace=workspace, hidden_dir=hidden,
                    test_name=task["regression_test"], run_id="baseline-test", phase="regression", language=task["language"],
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
        multi_file = copy.deepcopy(valid)
        multi_file["edits"].append({
            "path": "src/sibling.py", "find": "return sibling",
            "replace": "return sibling + 1", "reason": "Update the dependent sibling.",
        })
        self.assertEqual(len(runner.parse_patch_output(json.dumps(multi_file))["edits"]), 2)
        repeated_path = copy.deepcopy(multi_file)
        repeated_path["edits"][1]["path"] = "src/module.py"
        with self.assertRaisesRegex(runner.BenchmarkError, "each path only once"):
            runner.parse_patch_output(json.dumps(repeated_path))
        unsafe_path = copy.deepcopy(valid)
        unsafe_path["edits"][0]["path"] = "../outside.py"
        with self.assertRaisesRegex(runner.BenchmarkError, "safe relative"):
            runner.parse_patch_output(json.dumps(unsafe_path))
        for path in ("src//module.py", "src/./module.py", "C:/outside.py"):
            ambiguous = copy.deepcopy(valid)
            ambiguous["edits"][0]["path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(runner.BenchmarkError, "safe relative"):
                runner.parse_patch_output(json.dumps(ambiguous))
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

            sibling = workspace / "src" / "sibling.py"
            sibling.write_text("def sibling():\n    return 10\n", encoding="utf-8")
            multi_patch = {
                "format": "SYNTRA_AGENT_PATCH_V1",
                "summary": "Update two linked workspace modules.",
                "edits": [
                    {"path": "src/module.py", "find": "return 2", "replace": "return 3", "reason": "Update the primary result."},
                    {"path": "src/sibling.py", "find": "return 10", "replace": "return 11", "reason": "Keep the sibling module contract aligned."},
                ],
            }
            multi_changed = runner.apply_patch(workspace, multi_patch, ["src/module.py", "src/sibling.py"])
            self.assertEqual(multi_changed, ["src/module.py", "src/sibling.py"])
            self.assertEqual(source.read_text(encoding="utf-8"), "def value():\n    return 3\n")
            self.assertEqual(sibling.read_text(encoding="utf-8"), "def sibling():\n    return 11\n")

            unsafe = copy.deepcopy(patch)
            unsafe["edits"][0]["path"] = "../outside.py"
            with self.assertRaises(runner.BenchmarkError):
                runner.apply_patch(workspace, unsafe, ["../outside.py"])
            self.assertEqual(source.read_text(encoding="utf-8"), "def value():\n    return 3\n")

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

    def test_generation_rejects_context_overflow_before_streaming_the_model(self):
        class TokenizerEvidenceOnly:
            stream_called = False

            def tokenize_text(self, prompt, path=None):
                return {
                    "token_count": 9,
                    "context_window_tokens": 8,
                    "tokens_over_context": 1,
                }

            def stream_generate(self, *args, **kwargs):
                self.stream_called = True
                raise AssertionError("Generation must not start when the prompt exceeds context.")

        runtime = TokenizerEvidenceOnly()
        with self.assertRaises(runner.ContextWindowExceeded) as captured:
            runner._generate_patch(runtime, "oversized prompt", Path("unused-checkpoint"), "cpu", 16)
        self.assertFalse(runtime.stream_called)
        self.assertEqual(captured.exception.prompt_tokens, 9)
        self.assertEqual(captured.exception.context_window_tokens, 8)
        self.assertEqual(captured.exception.tokens_over_context, 1)
        self.assertEqual(captured.exception.context_utilization, 1.125)

    def test_run_task_records_context_overflow_without_starting_generation(self):
        class ContextLimitedRuntime:
            stream_called = False

            def tokenize_text(self, prompt, path=None):
                return {
                    "token_count": 120,
                    "context_window_tokens": 100,
                    "tokens_over_context": 21,
                }

            def stream_generate(self, *args, **kwargs):
                self.stream_called = True
                raise AssertionError("Model generation must not start for an over-context prompt.")

        runtime = ContextLimitedRuntime()
        task = next(task for task in self.suite["tasks"] if task["id"] == "python-clamp")
        result = runner.run_task(
            task, runtime=runtime, checkpoint=Path("unused-checkpoint"),
            device="cpu", max_new_tokens=16,
            isolation={"mode": "host-opt-in", "image": None, "image_id": None},
            run_id="context-evidence-test",
        )
        self.assertFalse(runtime.stream_called)
        self.assertTrue(result["baseline_validated"])
        self.assertEqual(result["status"], "context-exceeded")
        self.assertEqual(result["execution_speed"]["prompt_tokens"], 120)
        self.assertEqual(result["execution_speed"]["context_window_tokens"], 100)
        self.assertEqual(result["execution_speed"]["context_tokens_over_limit"], 21)
        self.assertEqual(result["execution_speed"]["context_utilization"], 1.2)
        self.assertEqual(result["execution_speed"]["context_status"], "exceeded")

    def test_loss_intelligence_divergence_requires_low_loss_and_at_least_three_zero_success_tasks(self):
        checkpoint = {"train_loss": 0.4}
        summary = {
            "task_count": 5,
            "baseline_valid_task_count": 5,
            "task_completion_rate": 0.0,
            "generation_attempted_task_count": 5,
            "generation_attempted_task_completion_rate": 0.0,
        }
        warning = runner.assess_loss_intelligence_divergence(summary, checkpoint)
        self.assertTrue(warning["detected"])
        self.assertEqual(warning["flag"], "OVERFITTING_LOSS_INTELLIGENCE_DIVERGENCE")
        self.assertIn("not proof", warning["interpretation"])

        too_small = runner.assess_loss_intelligence_divergence(
            {
                "baseline_valid_task_count": 2,
                "generation_attempted_task_count": 2,
                "generation_attempted_task_completion_rate": 0.0,
                "task_completion_rate": 0.0,
            },
            checkpoint,
        )
        self.assertFalse(too_small["detected"])
        context_only = runner.assess_loss_intelligence_divergence(
            {
                "baseline_valid_task_count": 5,
                "generation_attempted_task_count": 0,
                "generation_attempted_task_completion_rate": None,
                "task_completion_rate": 0.0,
            },
            checkpoint,
        )
        self.assertFalse(context_only["detected"], "Context-limit rejections are not evidence of failed code generation.")
        high_loss = runner.assess_loss_intelligence_divergence(
            {
                "baseline_valid_task_count": 5,
                "generation_attempted_task_count": 5,
                "generation_attempted_task_completion_rate": 0.0,
                "task_completion_rate": 0.0,
            },
            {"train_loss": 1.5},
        )
        self.assertFalse(high_loss["detected"])

    def test_summary_keeps_target_completion_and_regression_prevention_separate(self):
        results = [
            {"baseline_validated": True, "task_completion": True, "patch_correctness": True, "regression_prevention": False, "execution_speed": {"total_ms": 100, "generation_ms": 50, "generated_tokens": 10, "prompt_tokens": 90, "context_window_tokens": 100, "context_utilization": 0.9, "context_status": "within-window"}},
            {"baseline_validated": True, "task_completion": False, "patch_correctness": False, "regression_prevention": None, "execution_speed": {"total_ms": 200, "generation_ms": None, "generated_tokens": 0, "prompt_tokens": 110, "context_window_tokens": 100, "context_utilization": 1.1, "context_tokens_over_limit": 11, "context_status": "exceeded"}},
            {"baseline_validated": False, "task_completion": False, "patch_correctness": False, "regression_prevention": None, "execution_speed": {"total_ms": 5, "generation_ms": None, "generated_tokens": 0}},
        ]
        summary = runner.summarize(results)
        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(summary["baseline_valid_task_count"], 2)
        self.assertEqual(summary["invalid_fixture_count"], 1)
        self.assertEqual(summary["task_completion_rate"], 0.5)
        self.assertEqual(summary["patch_correctness_rate"], 0.5)
        self.assertEqual(summary["regression_prevention_rate"], 0.0)
        self.assertEqual(summary["fully_verified_task_count"], 0)
        self.assertEqual(summary["context_evidence"]["measured_task_count"], 2)
        self.assertEqual(summary["context_evidence"]["context_exceeded_task_count"], 1)
        self.assertEqual(summary["context_evidence"]["near_limit_task_count"], 2)
        self.assertEqual(summary["context_evidence"]["maximum_utilization"], 1.1)
        self.assertEqual(summary["generation_attempted_task_count"], 1)
        self.assertEqual(summary["generation_attempted_completed_task_count"], 1)
        self.assertEqual(summary["generation_attempted_task_completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

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


class BenchmarkHardwareIdentityTests(unittest.TestCase):
    def _mock_torch(self, *, cuda_runtime=None, hip_runtime=None):
        version = type("TorchVersion", (), {
            "cuda": cuda_runtime,
            "hip": hip_runtime,
        })()
        return type("TorchModule", (), {
            "__version__": "2.7.1+test",
            "version": version,
        })()

    def test_cpu_identity_records_measured_host_and_selected_runtime(self):
        device = type("Device", (), {"type": "cpu", "index": None})()
        torch_module = self._mock_torch()
        with (
            mock.patch.object(runner, "_cpu_model", return_value="Measured Test CPU"),
            mock.patch.object(runner, "_physical_memory_bytes", return_value=16 * 1024 ** 3),
            mock.patch.object(runner.os, "cpu_count", return_value=12),
            mock.patch.object(runner.host_platform, "system", return_value="TestOS"),
            mock.patch.object(runner.host_platform, "release", return_value="1.2"),
            mock.patch.object(runner.host_platform, "machine", return_value="x86_64"),
        ):
            identity = runner.collect_hardware_identity("auto", device, torch_module)
            repeated = runner.collect_hardware_identity("cpu", device, torch_module)

        self.assertTrue(identity["identity_complete"])
        self.assertEqual(identity["host"]["cpu_model"], "Measured Test CPU")
        self.assertEqual(identity["host"]["logical_cores"], 12)
        self.assertEqual(identity["host"]["physical_memory_bytes"], 16 * 1024 ** 3)
        self.assertEqual(identity["device"]["resolved"], "cpu")
        self.assertEqual(identity["device"]["name"], "Measured Test CPU")
        self.assertEqual(identity["pytorch"]["version"], "2.7.1+test")
        self.assertEqual(identity["fingerprint_sha256"], repeated["fingerprint_sha256"])
        self.assertEqual(identity["device"]["requested"], "auto")
        self.assertEqual(repeated["device"]["requested"], "cpu")

    def test_cuda_identity_records_selected_gpu_and_measured_driver_evidence(self):
        device = type("Device", (), {"type": "cuda", "index": None})()
        torch_module = self._mock_torch(cuda_runtime="12.4")
        torch_module.cuda = mock.Mock()
        torch_module.cuda.device_count.return_value = 2
        torch_module.cuda.current_device.return_value = 0
        torch_module.cuda.get_device_name.return_value = "Measured Test Accelerator"
        torch_module.cuda.get_device_properties.return_value = type(
            "CudaProperties",
            (),
            {"total_memory": 24 * 1024 ** 3, "major": 9, "minor": 0},
        )()
        with (
            mock.patch.object(runner, "_cpu_model", return_value="Measured Host CPU"),
            mock.patch.object(runner, "_physical_memory_bytes", return_value=64 * 1024 ** 3),
            mock.patch.object(runner.os, "cpu_count", return_value=32),
            mock.patch.object(runner, "_discover_accelerator_driver_version", return_value="550.90"),
        ):
            identity = runner.collect_hardware_identity("auto", device, torch_module)

        self.assertEqual(identity["device"]["resolved"], "cuda:0")
        self.assertEqual(identity["device"]["index"], 0)
        self.assertEqual(identity["device"]["vendor"], "nvidia_cuda")
        self.assertEqual(identity["device"]["name"], "Measured Test Accelerator")
        self.assertEqual(identity["device"]["total_memory_bytes"], 24 * 1024 ** 3)
        self.assertEqual(identity["device"]["compute_capability"], "9.0")
        self.assertEqual(identity["device"]["cuda_runtime_version"], "12.4")
        self.assertEqual(identity["device"]["driver_version"], "550.90")
        self.assertTrue(identity["identity_complete"])

    def test_cuda_identity_is_incomplete_when_selected_driver_cannot_be_measured(self):
        device = type("Device", (), {"type": "cuda", "index": 0})()
        torch_module = self._mock_torch(cuda_runtime="12.4")
        torch_module.cuda = mock.Mock()
        torch_module.cuda.device_count.return_value = 1
        torch_module.cuda.get_device_name.return_value = "Measured Test Accelerator"
        torch_module.cuda.get_device_properties.return_value = type(
            "CudaProperties",
            (),
            {"total_memory": 24 * 1024 ** 3, "major": 9, "minor": 0},
        )()
        with (
            mock.patch.object(runner, "_cpu_model", return_value="Measured Host CPU"),
            mock.patch.object(runner, "_physical_memory_bytes", return_value=64 * 1024 ** 3),
            mock.patch.object(runner.os, "cpu_count", return_value=32),
            mock.patch.object(runner, "_discover_accelerator_driver_version", return_value=None),
        ):
            identity = runner.collect_hardware_identity("cuda", device, torch_module)

        self.assertIsNone(identity["device"]["driver_version"])
        self.assertFalse(identity["identity_complete"])
        self.assertTrue(any("driver version" in item.lower() for item in identity["limitations"]))

    def test_nvidia_driver_probe_requires_a_consistent_host_driver_identity(self):
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(
                runner,
                "run_command",
                return_value={
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "stdout": "550.90\n550.90\n",
                    "stderr": "",
                    "elapsed_ms": 1,
                },
            ) as run_command,
        ):
            self.assertEqual(runner._discover_accelerator_driver_version("cuda"), "550.90")
        command = run_command.call_args.args[0]
        self.assertIn("--query-gpu=driver_version", command)

    def test_nvidia_driver_probe_rejects_conflicting_device_driver_values(self):
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(
                runner,
                "run_command",
                return_value={
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "stdout": "550.90\n555.10\n",
                    "stderr": "",
                    "elapsed_ms": 1,
                },
            ),
        ):
            self.assertIsNone(runner._discover_accelerator_driver_version("cuda"))

    def test_driver_probe_fails_closed_on_missing_or_ambiguous_tool_output(self):
        with mock.patch.object(runner.shutil, "which", return_value=None):
            self.assertIsNone(runner._discover_accelerator_driver_version("cuda"))
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(
                runner,
                "run_command",
                return_value={
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "stdout": "driver version unknown\n",
                    "stderr": "",
                    "elapsed_ms": 1,
                },
            ),
        ):
            self.assertIsNone(runner._discover_accelerator_driver_version("cuda"))

    def test_rocm_driver_probe_reads_the_reported_installed_driver_version(self):
        with (
            mock.patch.object(runner.shutil, "which", return_value="/opt/rocm/bin/rocm-smi"),
            mock.patch.object(
                runner,
                "run_command",
                return_value={
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "stdout": "ROCm System Management Interface\nDriver version: 6.2.4\n",
                    "stderr": "",
                    "elapsed_ms": 1,
                },
            ) as run_command,
        ):
            self.assertEqual(runner._discover_accelerator_driver_version("cuda-rocm"), "6.2.4")
        self.assertEqual(run_command.call_args.args[0][-1], "--showdriverversion")

    def test_mps_identity_does_not_invent_gpu_model_or_total_memory(self):
        device = type("Device", (), {"type": "mps", "index": None})()
        torch_module = self._mock_torch()
        with (
            mock.patch.object(runner, "_cpu_model", return_value="Measured Apple Host CPU"),
            mock.patch.object(runner, "_physical_memory_bytes", return_value=32 * 1024 ** 3),
            mock.patch.object(runner.os, "cpu_count", return_value=10),
        ):
            identity = runner.collect_hardware_identity("mps", device, torch_module)

        self.assertEqual(identity["device"]["resolved"], "mps")
        self.assertEqual(identity["device"]["vendor"], "apple_mps")
        self.assertIsNone(identity["device"]["name"])
        self.assertIsNone(identity["device"]["total_memory_bytes"])
        self.assertFalse(identity["identity_complete"])
        self.assertTrue(any("apple gpu" in item.lower() for item in identity["limitations"]))

    def test_missing_host_measurements_remain_null_and_identity_incomplete(self):
        device = type("Device", (), {"type": "cpu", "index": None})()
        torch_module = self._mock_torch()
        with (
            mock.patch.object(runner, "_cpu_model", return_value=None),
            mock.patch.object(runner, "_physical_memory_bytes", return_value=None),
            mock.patch.object(runner.os, "cpu_count", return_value=None),
        ):
            identity = runner.collect_hardware_identity("cpu", device, torch_module)

        self.assertIsNone(identity["host"]["cpu_model"])
        self.assertIsNone(identity["host"]["physical_memory_bytes"])
        self.assertIsNone(identity["host"]["logical_cores"])
        self.assertFalse(identity["identity_complete"])
        self.assertTrue(identity["limitations"])


class RawModelReportComparisonTests(unittest.TestCase):
    @staticmethod
    def _hardware(*, cpu_model="Test CPU", logical_cores=8, torch_version="2.7.1+cpu", complete=True):
        host = {
            "system": "TestOS",
            "release": "1.0",
            "machine": "x86_64",
            "cpu_model": cpu_model,
            "logical_cores": logical_cores,
            "physical_memory_bytes": 16 * 1024 ** 3,
        }
        python_info = {"implementation": "CPython", "version": "3.12.8"}
        pytorch = {
            "version": torch_version,
            "cuda_runtime_version": None,
            "rocm_runtime_version": None,
        }
        device = {
            "requested": "auto",
            "resolved": "cpu",
            "type": "cpu",
            "index": None,
            "vendor": "cpu",
            "name": cpu_model,
            "total_memory_bytes": None,
            "compute_capability": None,
            "available_device_count": None,
            "cuda_runtime_version": None,
            "rocm_runtime_version": None,
            "driver_version": None,
        }
        fingerprint_payload = {
            "schema_version": 1,
            "host": host,
            "python": python_info,
            "pytorch": pytorch,
            "device": {key: value for key, value in device.items() if key != "requested"},
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 1,
            "identity_complete": complete,
            "fingerprint_sha256": fingerprint,
            **fingerprint_payload,
            "limitations": [] if complete else ["Hardware identity is incomplete in this fixture."],
            "device": device,
        }

    @classmethod
    def _report(
        cls,
        *,
        run_id,
        checkpoint,
        task_completion,
        patch_correctness,
        regression_prevention,
        total_ms,
        generation_ms,
        generated_tokens,
        hardware=None,
        protocol="c" * 64,
        baseline_validated=True,
    ):
        output_rate = (
            round(generated_tokens / (generation_ms / 1000), 2)
            if generation_ms is not None and generation_ms > 0
            else None
        )
        task = {
            "id": "python-clamp",
            "language": "python",
            "status": "completed" if task_completion else "target-failed",
            "baseline_validated": baseline_validated,
            "patch_correctness": patch_correctness,
            "task_completion": task_completion,
            "regression_prevention": regression_prevention,
            "execution_speed": {
                "total_ms": total_ms,
                "generation_ms": generation_ms,
                "generated_tokens": generated_tokens,
                "output_tokens_per_second": output_rate,
            },
        }
        valid = int(baseline_validated)
        completed = int(baseline_validated and task_completion)
        patch_count = int(baseline_validated and patch_correctness)
        regression_measured = int(baseline_validated and regression_prevention is not None)
        regression_count = int(baseline_validated and regression_prevention is True)
        report = {
            "schema_version": 1,
            "benchmark_track": "raw-native-model",
            "suite_version": "1.4.0",
            "evaluation_protocol": {
                "schema_version": 1,
                "sha256": protocol,
                "file_count": 100,
                "total_bytes": 1_000_000,
            },
            "run_id": run_id,
            "model": {
                "engine": "syntra-native-pytorch",
                "name": run_id,
                "checkpoint_sha256": checkpoint,
                "device_resolved": "cpu" if hardware else None,
            },
            "hardware": hardware,
            "summary": {
                "task_count": 1,
                "baseline_valid_task_count": valid,
                "invalid_fixture_count": 1 - valid,
                "completed_task_count": completed,
                "task_completion_rate": round(completed / valid, 4) if valid else None,
                "patch_correct_count": patch_count,
                "patch_correctness_rate": round(patch_count / valid, 4) if valid else None,
                "regression_measured_count": regression_measured,
                "regression_prevented_count": regression_count,
                "regression_prevention_rate": round(regression_count / regression_measured, 4) if regression_measured else None,
            },
            "tasks": [task],
        }
        return report

    @classmethod
    def _with_tasks(cls, report, records):
        report = copy.deepcopy(report)
        tasks = []
        for record in records:
            task = copy.deepcopy(report["tasks"][0])
            task.update({
                "id": record["id"],
                "language": record["language"],
                "baseline_validated": record["baseline_validated"],
                "task_completion": record["task_completion"],
                "patch_correctness": record["patch_correctness"],
                "regression_prevention": record["regression_prevention"],
                "status": "completed" if record["task_completion"] else "target-failed",
            })
            generation_ms = record["generation_ms"]
            generated_tokens = record["generated_tokens"]
            task["execution_speed"].update({
                "total_ms": record["total_ms"],
                "generation_ms": generation_ms,
                "generated_tokens": generated_tokens,
                "output_tokens_per_second": (
                    round(generated_tokens / (generation_ms / 1000), 2)
                    if generation_ms is not None and generation_ms > 0
                    else None
                ),
            })
            tasks.append(task)
        valid = [task for task in tasks if task["baseline_validated"]]
        regression = [task for task in valid if task["regression_prevention"] is not None]
        completed_count = sum(task["task_completion"] for task in valid)
        patch_count = sum(task["patch_correctness"] for task in valid)
        regression_count = sum(task["regression_prevention"] is True for task in regression)
        denominator = len(valid)
        report["tasks"] = tasks
        report["summary"].update({
            "task_count": len(tasks),
            "baseline_valid_task_count": denominator,
            "invalid_fixture_count": len(tasks) - denominator,
            "completed_task_count": completed_count,
            "task_completion_rate": round(completed_count / denominator, 4) if denominator else None,
            "patch_correct_count": patch_count,
            "patch_correctness_rate": round(patch_count / denominator, 4) if denominator else None,
            "regression_measured_count": len(regression),
            "regression_prevented_count": regression_count,
            "regression_prevention_rate": round(regression_count / len(regression), 4) if regression else None,
        })
        return report

    def test_language_matrix_uses_per_language_valid_denominators_and_marks_small_samples(self):
        hardware = self._hardware()
        base = self._report(
            run_id="language-baseline",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        baseline = self._with_tasks(base, [
            {
                "id": "python-clamp",
                "language": "python",
                "baseline_validated": True,
                "task_completion": True,
                "patch_correctness": True,
                "regression_prevention": True,
                "total_ms": 100,
                "generation_ms": 50,
                "generated_tokens": 10,
            },
            {
                "id": "javascript-event-normalizer",
                "language": "javascript",
                "baseline_validated": True,
                "task_completion": False,
                "patch_correctness": False,
                "regression_prevention": None,
                "total_ms": 200,
                "generation_ms": 100,
                "generated_tokens": 20,
            },
            {
                "id": "typescript-object-mapper",
                "language": "typescript",
                "baseline_validated": False,
                "task_completion": False,
                "patch_correctness": False,
                "regression_prevention": None,
                "total_ms": 10,
                "generation_ms": None,
                "generated_tokens": 0,
            },
        ])
        candidate = self._with_tasks({
            **base,
            "run_id": "language-candidate",
            "model": {**base["model"], "name": "language-candidate", "checkpoint_sha256": "b" * 64},
        }, [
            {
                "id": "python-clamp",
                "language": "python",
                "baseline_validated": True,
                "task_completion": True,
                "patch_correctness": True,
                "regression_prevention": True,
                "total_ms": 80,
                "generation_ms": 40,
                "generated_tokens": 10,
            },
            {
                "id": "javascript-event-normalizer",
                "language": "javascript",
                "baseline_validated": True,
                "task_completion": True,
                "patch_correctness": True,
                "regression_prevention": False,
                "total_ms": 150,
                "generation_ms": 80,
                "generated_tokens": 40,
            },
            {
                "id": "typescript-object-mapper",
                "language": "typescript",
                "baseline_validated": False,
                "task_completion": False,
                "patch_correctness": False,
                "regression_prevention": None,
                "total_ms": 10,
                "generation_ms": None,
                "generated_tokens": 0,
            },
        ])

        comparison = runner.compare_raw_benchmark_reports(baseline, candidate)
        matrix = comparison["language_breakdown"]
        self.assertEqual(set(matrix), {"python", "javascript", "typescript"})

        python = matrix["python"]
        self.assertEqual(python["task_count"], 1)
        self.assertEqual(python["baseline_valid_task_count"], 1)
        self.assertEqual(python["baseline"]["task_completion_rate"], 1.0)
        self.assertEqual(python["candidate"]["task_completion_rate"], 1.0)
        self.assertEqual(python["quality_status"], "unchanged")
        self.assertTrue(python["sample_size_warning"])

        javascript = matrix["javascript"]
        self.assertEqual(javascript["baseline_valid_task_ids"], ["javascript-event-normalizer"])
        self.assertEqual(javascript["baseline"]["task_completion_rate"], 0.0)
        self.assertEqual(javascript["candidate"]["task_completion_rate"], 1.0)
        self.assertEqual(javascript["quality_status"], "improved")
        self.assertFalse(javascript["environment"]["regression_task_denominators_match"])
        regression_metric = next(
            metric for metric in javascript["metrics"]
            if metric["key"] == "regression_prevention_rate"
        )
        self.assertEqual(regression_metric["direction"], "incomparable")
        self.assertTrue(javascript["environment"]["hardware_comparable"])

        typescript = matrix["typescript"]
        self.assertEqual(typescript["task_count"], 1)
        self.assertEqual(typescript["baseline_valid_task_count"], 0)
        self.assertEqual(typescript["invalid_fixture_count"], 1)
        self.assertEqual(typescript["quality_status"], "unavailable")
        self.assertTrue(typescript["sample_size_warning"])
        self.assertIsNone(typescript["baseline"]["task_completion_rate"])
        self.assertIn("not evidence of general language capability", typescript["sample_size_note"])

        self.assertEqual(comparison["languages_with_small_samples"], ["javascript", "python", "typescript"])
        self.assertIn("invalid hidden baselines are excluded", comparison["language_matrix_note"])

    def test_language_matrix_blocks_performance_when_hardware_is_incomplete(self):
        baseline = self._report(
            run_id="language-incomplete-baseline",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=self._hardware(complete=False),
        )
        candidate = copy.deepcopy(baseline)
        candidate["run_id"] = "language-incomplete-candidate"
        candidate["model"] = {
            **candidate["model"],
            "name": "language-incomplete-candidate",
            "checkpoint_sha256": "b" * 64,
        }
        comparison = runner.compare_raw_benchmark_reports(baseline, candidate)
        language = comparison["language_breakdown"]["python"]
        self.assertEqual(language["quality_status"], "unchanged")
        self.assertFalse(language["environment"]["hardware_comparable"])
        self.assertEqual(
            next(metric for metric in language["metrics"] if metric["key"] == "mean_total_task_ms")["direction"],
            "incomparable",
        )
        self.assertEqual(
            next(metric for metric in language["metrics"] if metric["key"] == "generation_tokens_per_second")["direction"],
            "incomparable",
        )

    def test_report_validation_recomputes_denominators_and_hardware_fingerprint(self):
        hardware = self._hardware()
        valid = self._report(
            run_id="baseline",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        self.assertIs(runner.validate_raw_benchmark_report(valid), valid)

        inconsistent = copy.deepcopy(valid)
        inconsistent["summary"]["completed_task_count"] = 0
        with self.assertRaisesRegex(runner.BenchmarkError, "completed_task_count"):
            runner.validate_raw_benchmark_report(inconsistent)

        altered_hardware = copy.deepcopy(valid)
        altered_hardware["hardware"]["host"]["cpu_model"] = "Different CPU"
        altered_hardware["hardware"]["device"]["name"] = "Different CPU"
        with self.assertRaisesRegex(runner.BenchmarkError, "fingerprint"):
            runner.validate_raw_benchmark_report(altered_hardware)

        altered_throughput = copy.deepcopy(valid)
        altered_throughput["tasks"][0]["execution_speed"]["output_tokens_per_second"] = 999
        with self.assertRaisesRegex(runner.BenchmarkError, "inconsistent with its measured tokens/time"):
            runner.validate_raw_benchmark_report(altered_throughput)

    def test_report_validator_rejects_complete_hardware_with_missing_release_or_driver(self):
        hardware = self._hardware()
        hardware["host"]["release"] = None
        report = self._report(
            run_id="missing-release",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        payload = {
            "schema_version": 1,
            "host": hardware["host"],
            "python": hardware["python"],
            "pytorch": hardware["pytorch"],
            "device": {key: value for key, value in hardware["device"].items() if key != "requested"},
        }
        hardware["fingerprint_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(runner.BenchmarkError, "marked complete"):
            runner.validate_raw_benchmark_report(report)

        cuda_hardware = copy.deepcopy(self._hardware())
        cuda_hardware["device"].update({
            "resolved": "cuda:0",
            "type": "cuda",
            "index": 0,
            "vendor": "nvidia_cuda",
            "name": "Measured GPU",
            "total_memory_bytes": 8 * 1024 ** 3,
            "compute_capability": "8.6",
            "available_device_count": 1,
            "cuda_runtime_version": "12.4",
            "driver_version": "550.90",
        })
        cuda_hardware["pytorch"]["cuda_runtime_version"] = "12.4"
        cuda_payload = {
            "schema_version": 1,
            "host": cuda_hardware["host"],
            "python": cuda_hardware["python"],
            "pytorch": cuda_hardware["pytorch"],
            "device": {key: value for key, value in cuda_hardware["device"].items() if key != "requested"},
        }
        cuda_hardware["fingerprint_sha256"] = hashlib.sha256(
            json.dumps(cuda_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        cuda_report = self._report(
            run_id="cuda-zero",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=cuda_hardware,
        )
        cuda_report["model"]["device_resolved"] = "cuda:0"
        self.assertIs(runner.validate_raw_benchmark_report(cuda_report), cuda_report)

    def test_same_protocol_and_complete_same_hardware_reports_compare_task_quality_and_performance(self):
        hardware = self._hardware()
        baseline = self._report(
            run_id="baseline",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=False,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        candidate = self._report(
            run_id="candidate",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=80,
            generation_ms=40,
            generated_tokens=20,
            hardware=copy.deepcopy(hardware),
        )
        result = runner.compare_raw_benchmark_reports(baseline, candidate)
        self.assertEqual(result["quality_status"], "improved")
        self.assertTrue(result["environment"]["hardware_comparable"])
        self.assertEqual(
            result["metrics"][0]["direction"],
            "improved",
        )
        self.assertEqual(
            next(metric for metric in result["metrics"] if metric["key"] == "mean_total_task_ms")["direction"],
            "improved",
        )
        self.assertEqual(
            next(metric for metric in result["metrics"] if metric["key"] == "generation_tokens_per_second")["direction"],
            "improved",
        )
        self.assertEqual(result["tasks"][0]["status"], "improved")

    def test_different_or_incomplete_hardware_keeps_quality_but_blocks_performance_claims(self):
        baseline = self._report(
            run_id="baseline",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=None,
            total_ms=200,
            generation_ms=100,
            generated_tokens=10,
            hardware=self._hardware(),
        )
        candidate = self._report(
            run_id="candidate",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=None,
            total_ms=100,
            generation_ms=50,
            generated_tokens=20,
            hardware=self._hardware(cpu_model="Different CPU"),
        )
        result = runner.compare_raw_benchmark_reports(baseline, candidate)
        self.assertEqual(result["quality_status"], "improved")
        self.assertFalse(result["environment"]["hardware_comparable"])
        self.assertEqual(
            next(metric for metric in result["metrics"] if metric["key"] == "mean_total_task_ms")["direction"],
            "incomparable",
        )
        self.assertEqual(
            next(metric for metric in result["metrics"] if metric["key"] == "generation_tokens_per_second")["direction"],
            "incomparable",
        )
        self.assertTrue(any("incomparable" in warning.lower() for warning in result["warnings"]))

    def test_comparison_rejects_protocol_task_selection_and_baseline_denominator_changes(self):
        baseline = self._report(
            run_id="baseline",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=None,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=self._hardware(),
        )
        candidate = self._report(
            run_id="candidate",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=None,
            total_ms=80,
            generation_ms=40,
            generated_tokens=20,
            hardware=self._hardware(),
        )
        with self.assertRaisesRegex(runner.BenchmarkError, "different evaluation protocols"):
            runner.compare_raw_benchmark_reports(baseline, {
                **candidate,
                "evaluation_protocol": {**candidate["evaluation_protocol"], "sha256": "d" * 64},
            })
        with self.assertRaisesRegex(runner.BenchmarkError, "different raw-model task selections"):
            runner.compare_raw_benchmark_reports(baseline, {
                **candidate,
                "tasks": [{**candidate["tasks"][0], "id": "different-task"}],
                "summary": {**candidate["summary"]},
            })
        with self.assertRaisesRegex(runner.BenchmarkError, "different languages to the same task ID"):
            relabeled = copy.deepcopy(candidate)
            relabeled["tasks"][0]["language"] = "typescript"
            runner.compare_raw_benchmark_reports(baseline, relabeled)

        with self.assertRaisesRegex(runner.BenchmarkError, "baseline-valid task sets differ"):
            runner.compare_raw_benchmark_reports(baseline, {
                **candidate,
                "tasks": [{**candidate["tasks"][0], "baseline_validated": False, "task_completion": False, "patch_correctness": False}],
                "summary": {
                    **candidate["summary"],
                    "baseline_valid_task_count": 0,
                    "invalid_fixture_count": 1,
                    "completed_task_count": 0,
                    "task_completion_rate": None,
                    "patch_correct_count": 0,
                    "patch_correctness_rate": None,
                    "regression_measured_count": 0,
                    "regression_prevented_count": 0,
                    "regression_prevention_rate": None,
                },
            })

    def test_unavailable_hardware_and_unequal_measurement_denominators_remain_incomparable(self):
        baseline = self._report(
            run_id="baseline",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=None,
        )
        candidate = self._report(
            run_id="candidate",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=None,
            total_ms=80,
            generation_ms=None,
            generated_tokens=0,
            hardware=None,
        )
        result = runner.compare_raw_benchmark_reports(baseline, candidate)
        regression = next(metric for metric in result["metrics"] if metric["key"] == "regression_prevention_rate")
        throughput = next(metric for metric in result["metrics"] if metric["key"] == "generation_tokens_per_second")
        latency = next(metric for metric in result["metrics"] if metric["key"] == "mean_total_task_ms")
        self.assertEqual(regression["direction"], "incomparable")
        self.assertEqual(throughput["direction"], "incomparable")
        self.assertEqual(latency["direction"], "incomparable")


    def test_raw_benchmark_compare_cli_compares_content_addressed_reports(self):
        hardware = self._hardware()
        baseline = self._report(
            run_id="baseline-cli",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=False,
            total_ms=120,
            generation_ms=60,
            generated_tokens=12,
            hardware=hardware,
        )
        candidate = self._report(
            run_id="candidate-cli",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=15,
            hardware=copy.deepcopy(hardware),
        )
        with tempfile.TemporaryDirectory(prefix="syntra-raw-benchmark-compare-") as temp:
            baseline_path = Path(temp) / "baseline.json"
            candidate_path = Path(temp) / "candidate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "compare_raw_benchmarks.py"),
                    "--baseline", str(baseline_path),
                    "--candidate", str(candidate_path),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Quality outcome: improved", result.stdout)
        self.assertIn("Hardware-controlled performance: yes", result.stdout)
        self.assertIn("task_completion_rate", result.stdout)

    def test_raw_benchmark_compare_cli_rejects_protocol_mismatch(self):
        baseline = self._report(
            run_id="baseline-protocol",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=self._hardware(),
        )
        candidate = self._report(
            run_id="candidate-protocol",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=90,
            generation_ms=45,
            generated_tokens=12,
            hardware=self._hardware(),
        )
        candidate["evaluation_protocol"]["sha256"] = "d" * 64
        with tempfile.TemporaryDirectory(prefix="syntra-raw-benchmark-invalid-") as temp:
            baseline_path = Path(temp) / "baseline.json"
            candidate_path = Path(temp) / "candidate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "compare_raw_benchmarks.py"),
                    "--baseline", str(baseline_path),
                    "--candidate", str(candidate_path),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("different evaluation protocols", result.stderr)


class RawBenchmarkComparisonCliTests(unittest.TestCase):
    def test_npm_script_exposes_the_raw_benchmark_comparison_cli(self):
        package = json.loads((REPOSITORY_ROOT / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(
            package.get("scripts", {}).get("benchmark:model:compare"),
            "python scripts/compare_raw_benchmarks.py",
        )
        self.assertTrue((REPOSITORY_ROOT / "scripts" / "compare_raw_benchmarks.py").is_file())

    @staticmethod
    def _invoke(baseline: Path, candidate: Path, output: Path | None = None):
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "compare_raw_benchmarks.py"),
            "--baseline", str(baseline),
            "--candidate", str(candidate),
        ]
        if output is not None:
            command.extend(["--output", str(output)])
        return subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_comparison_cli_compares_real_report_files_on_matching_hardware(self):
        hardware = RawModelReportComparisonTests._hardware()
        baseline = RawModelReportComparisonTests._report(
            run_id="cli-baseline",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=False,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        candidate = RawModelReportComparisonTests._report(
            run_id="cli-candidate",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=True,
            total_ms=80,
            generation_ms=40,
            generated_tokens=20,
            hardware=copy.deepcopy(hardware),
        )
        with tempfile.TemporaryDirectory(prefix="syntra-raw-cli-") as temp:
            baseline_path = Path(temp) / "baseline.json"
            candidate_path = Path(temp) / "candidate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = self._invoke(baseline_path, candidate_path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Quality outcome: improved", result.stdout)
        self.assertIn("Hardware-controlled performance: yes", result.stdout)
        self.assertIn("generation_tokens_per_second", result.stdout)

    def test_comparison_cli_marks_different_hardware_performance_incomparable(self):
        baseline = RawModelReportComparisonTests._report(
            run_id="cli-baseline-host-a",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=None,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=RawModelReportComparisonTests._hardware(),
        )
        candidate = RawModelReportComparisonTests._report(
            run_id="cli-candidate-host-b",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=None,
            total_ms=80,
            generation_ms=40,
            generated_tokens=20,
            hardware=RawModelReportComparisonTests._hardware(cpu_model="Different Measured CPU"),
        )
        with tempfile.TemporaryDirectory(prefix="syntra-raw-cli-different-") as temp:
            baseline_path = Path(temp) / "baseline.json"
            candidate_path = Path(temp) / "candidate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = self._invoke(baseline_path, candidate_path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Quality outcome: improved", result.stdout)
        self.assertIn("Hardware-controlled performance: no", result.stdout)
        self.assertIn("incomparable", result.stdout)

    def test_comparison_cli_rejects_invalid_json_and_symbolic_link_reports(self):
        with tempfile.TemporaryDirectory(prefix="syntra-raw-cli-invalid-") as temp:
            invalid_path = Path(temp) / "invalid.json"
            valid_path = Path(temp) / "valid.json"
            invalid_path.write_text('{"schema_version": 1,}', encoding="utf-8")
            valid_path.write_text("{}", encoding="utf-8")
            invalid = self._invoke(invalid_path, valid_path)
            self.assertEqual(invalid.returncode, 1)
            self.assertIn("invalid JSON", invalid.stderr)

            if os.name != "nt":
                link_path = Path(temp) / "linked.json"
                link_path.symlink_to(valid_path)
                linked = self._invoke(link_path, valid_path)
                self.assertEqual(linked.returncode, 1)
                self.assertIn("non-symlink", linked.stderr)

    def test_comparison_cli_output_is_atomic_and_never_overwrites_an_existing_file(self):
        hardware = RawModelReportComparisonTests._hardware()
        baseline = RawModelReportComparisonTests._report(
            run_id="cli-output-baseline",
            checkpoint="a" * 64,
            task_completion=False,
            patch_correctness=False,
            regression_prevention=None,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        candidate = RawModelReportComparisonTests._report(
            run_id="cli-output-candidate",
            checkpoint="b" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=None,
            total_ms=80,
            generation_ms=40,
            generated_tokens=20,
            hardware=copy.deepcopy(hardware),
        )
        with tempfile.TemporaryDirectory(prefix="syntra-raw-cli-output-", dir=REPOSITORY_ROOT) as temp:
            root = Path(temp)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_path = root / "comparisons" / "comparison.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            first = self._invoke(baseline_path, candidate_path, output_path)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output_path.read_bytes()
            self.assertTrue(original.endswith(b"\n"))
            parsed = json.loads(original)
            self.assertEqual(parsed["comparison_track"], "raw-native-model")
            self.assertIn("Per-language fixture matrix", first.stdout)
            self.assertIn("python", first.stdout)
            self.assertIn("Sample limitation", first.stdout)

            second = self._invoke(baseline_path, candidate_path, output_path)
            self.assertEqual(second.returncode, 1)
            self.assertIn("already exists", second.stderr)
            self.assertEqual(output_path.read_bytes(), original)

    def test_comparison_cli_rejects_output_paths_outside_repository_before_creating_directories(self):
        hardware = RawModelReportComparisonTests._hardware()
        baseline = RawModelReportComparisonTests._report(
            run_id="outside-output-baseline",
            checkpoint="a" * 64,
            task_completion=True,
            patch_correctness=True,
            regression_prevention=None,
            total_ms=100,
            generation_ms=50,
            generated_tokens=10,
            hardware=hardware,
        )
        candidate = copy.deepcopy(baseline)
        candidate["run_id"] = "outside-output-candidate"
        with tempfile.TemporaryDirectory(prefix="syntra-raw-cli-outside-") as temp:
            root = Path(temp)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            forbidden_directory = root / "must-not-be-created"
            output_path = forbidden_directory / "comparison.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = self._invoke(baseline_path, candidate_path, output_path)
            self.assertEqual(result.returncode, 1)
            self.assertIn("inside the SYNTRA repository", result.stderr)
            self.assertFalse(forbidden_directory.exists())

    def test_comparison_cli_help_does_not_require_report_paths(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "compare_raw_benchmarks.py"),
                "--help",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("raw SYNTRA PyTorch benchmark reports", result.stdout)
        self.assertIn("--baseline", result.stdout)
        self.assertIn("--candidate", result.stdout)


class RawModelBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = runner.load_suite()

    def test_suite_has_fourteen_unique_held_out_tasks_across_five_languages(self):
        self.assertEqual(self.suite["version"], "1.4.0")
        self.assertEqual(len(self.suite["tasks"]), 14)
        self.assertEqual(len({task["id"] for task in self.suite["tasks"]}), 14)
        self.assertEqual({task["language"] for task in self.suite["tasks"]}, {"python", "javascript", "typescript", "go", "rust"})
        self.assertEqual(
            {language: sum(task["language"] == language for task in self.suite["tasks"]) for language in {"python", "javascript", "typescript", "go", "rust"}},
            {"python": 9, "javascript": 2, "typescript": 1, "go": 1, "rust": 1},
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
            self.assertEqual(runner.expected_test_filename(language, "baseline-target"), "test_target" + extension)
            self.assertEqual(runner.expected_test_filename(language, "baseline-regression"), "test_regression" + extension)
        self.assertEqual(runner.resolve_environment("go", {"go": "golang:1.22-alpine"})["image"], "golang:1.22-alpine")
        self.assertEqual(runner.resolve_environment("javascript", {"javascript": "node:20-slim"})["image"], "node:20-slim")
        with self.assertRaises(ValueError):
            runner.resolve_environment("unknown")

    def test_docker_preflight_records_a_full_immutable_image_id(self):
        image_id = "sha256:" + ("a" * 64)
        with mock.patch.object(runner.shutil, "which", return_value="/usr/bin/docker"), mock.patch.object(
            runner,
            "run_command",
            return_value={"returncode": 0, "stdout": image_id + "\n", "stderr": "", "timed_out": False, "output_limited": False, "elapsed_ms": 1},
        ):
            isolation = runner.prepare_isolation("docker", "node:20-slim")
        self.assertEqual(isolation["image"], "node:20-slim")
        self.assertEqual(isolation["image_id"], image_id)
        self.assertEqual(isolation["mode"], "docker-no-network")

    def test_docker_preflight_rejects_malformed_image_identity(self):
        for image_id in ("sha256:short", "node:20-slim", "sha256:" + ("g" * 64)):
            with self.subTest(image_id=image_id), mock.patch.object(
                runner.shutil, "which", return_value="/usr/bin/docker"
            ), mock.patch.object(
                runner,
                "run_command",
                return_value={"returncode": 0, "stdout": image_id + "\n", "stderr": "", "timed_out": False, "output_limited": False, "elapsed_ms": 1},
            ):
                with self.assertRaisesRegex(runner.BenchmarkError, "valid immutable image ID"):
                    runner.prepare_isolation("docker", "node:20-slim")

    def test_docker_execution_uses_the_preflight_image_id_not_its_mutable_tag(self):
        image_id = "sha256:" + ("b" * 64)
        with tempfile.TemporaryDirectory(prefix="syntra-docker-image-id-") as temp:
            workspace = Path(temp)
            isolation = {"mode": "docker-no-network", "image": "node:20-slim", "image_id": image_id}
            outcome = {"returncode": 0, "timed_out": False, "output_limited": False, "stdout": "v20\n", "stderr": "", "elapsed_ms": 3}
            with mock.patch.object(runner.shutil, "which", return_value="/usr/bin/docker"), mock.patch.object(
                runner, "run_command", return_value=outcome
            ) as command_runner:
                result = runner.run_environment_command(
                    language="javascript",
                    isolation=isolation,
                    workspace=workspace,
                    argv=["node", "--version"],
                    run_id="raw-model-image-id-test",
                    phase="preflight-identity",
                )
        command = command_runner.call_args.args[0]
        self.assertIn("--pull=never", command)
        self.assertIn(image_id, command)
        self.assertNotIn("node:20-slim", command)
        self.assertEqual(result, outcome)

    def test_docker_execution_rejects_missing_or_mutable_image_identity(self):
        with tempfile.TemporaryDirectory(prefix="syntra-docker-image-guard-") as temp:
            for image_id in (None, "node:20-slim", "sha256:short"):
                isolation = {"mode": "docker-no-network", "image": "node:20-slim", "image_id": image_id}
                with self.subTest(image_id=image_id), mock.patch.object(
                    runner.shutil, "which", return_value="/usr/bin/docker"
                ), mock.patch.object(runner, "run_command") as command_runner:
                    with self.assertRaisesRegex(runner.BenchmarkError, "immutable image ID"):
                        runner.run_environment_command(
                            language="javascript",
                            isolation=isolation,
                            workspace=Path(temp),
                            argv=["node", "--version"],
                            run_id="raw-model-image-id-reject",
                            phase="preflight-identity",
                        )
                    command_runner.assert_not_called()

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
            "javascript-bounded-map": {"src/map-concurrent.mjs"},
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
                "instruction_dataset_sha256": "c" * 64,
                "instruction_dataset_bytes": 4096,
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
            self.assertEqual(evidence["instruction_dataset_sha256"], "c" * 64)
            self.assertEqual(evidence["instruction_dataset_bytes"], 4096)

    def test_checkpoint_evidence_rejects_malformed_sft_dataset_provenance(self):
        with tempfile.TemporaryDirectory(prefix="syntra-invalid-sft-provenance-") as temp:
            checkpoint = Path(temp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint-bytes-used-for-identity")
            sidecar = checkpoint.with_suffix(".pt.json")
            sidecar.write_text(json.dumps({
                "training_mode": "supervised-instruction-finetuning",
                "instruction_dataset_sha256": "not-a-sha256",
                "instruction_dataset_bytes": 10,
            }), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkError, "malformed SFT dataset SHA-256"):
                runner.load_checkpoint_evidence(checkpoint)

    def test_checkpoint_evidence_rejects_partial_sft_dataset_provenance(self):
        with tempfile.TemporaryDirectory(prefix="syntra-partial-sft-provenance-") as temp:
            checkpoint = Path(temp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint-bytes-used-for-identity")
            sidecar = checkpoint.with_suffix(".pt.json")
            sidecar.write_text(json.dumps({
                "training_mode": "supervised-instruction-finetuning",
                "instruction_dataset_sha256": "c" * 64,
            }), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkError, "incomplete SFT dataset provenance"):
                runner.load_checkpoint_evidence(checkpoint)

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
        self.assertEqual(captured.exception.tokens_over_context, 2)
        self.assertEqual(captured.exception.context_utilization, 1.125)

    def test_generation_rejects_malformed_tokenizer_evidence(self):
        class MalformedTokenizer:
            def tokenize_text(self, prompt, path=None):
                return ["not", "a", "tokenizer", "result"]

            def stream_generate(self, *args, **kwargs):
                raise AssertionError("Generation must not run without validated tokenizer evidence.")

        with self.assertRaisesRegex(runner.BenchmarkError, "malformed context evidence"):
            runner._generate_patch(MalformedTokenizer(), "prompt", Path("unused"), "cpu", 16)

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
            {"id": "js", "language": "javascript", "baseline_validated": True, "task_completion": True, "patch_correctness": True, "regression_prevention": False, "execution_speed": {"total_ms": 100, "generation_ms": 50, "generated_tokens": 10, "prompt_tokens": 90, "context_window_tokens": 100, "context_utilization": 0.9, "context_status": "within-window"}},
            {"id": "ts", "language": "typescript", "baseline_validated": True, "task_completion": False, "patch_correctness": False, "regression_prevention": None, "execution_speed": {"total_ms": 200, "generation_ms": None, "generated_tokens": 0, "prompt_tokens": 110, "context_window_tokens": 100, "context_utilization": 1.1, "context_tokens_over_limit": 11, "context_status": "exceeded"}},
            {"id": "py", "language": "python", "baseline_validated": False, "task_completion": False, "patch_correctness": False, "regression_prevention": None, "execution_speed": {"total_ms": 5, "generation_ms": None, "generated_tokens": 0}},
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
        self.assertEqual(summary["language_breakdown"]["javascript"]["task_completion_count"], 1)
        self.assertEqual(summary["language_breakdown"]["javascript"]["patch_correct_count"], 1)
        self.assertEqual(summary["language_breakdown"]["javascript"]["regression_prevention_measured_count"], 1)
        self.assertEqual(summary["language_breakdown"]["javascript"]["regression_prevention_rate"], 0.0)
        self.assertEqual(summary["language_breakdown"]["typescript"]["context_exceeded_task_count"], 1)
        self.assertIsNone(summary["language_breakdown"]["typescript"]["regression_prevention_rate"])
        self.assertEqual(summary["language_breakdown"]["python"]["baseline_valid_task_count"], 0)

        self.assertEqual(
            runner.language_summary_lines(summary["language_breakdown"]),
            [
                "Language javascript: 0/1 fully verified; 1 selected",
                "Language python: 0/0 fully verified; 1 selected",
                "Language typescript: 0/1 fully verified; 1 selected",
            ],
            "CLI language summaries must use fully_verified_task_count, not target-only completion counts",
        )


if __name__ == "__main__":
    unittest.main()

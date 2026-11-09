import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_epochs
import run_local_baseline
from syntra_ai.model import EOS_ID
from syntra_ai.tokenizer import ByteBPETokenizer
from syntra_ai.instruction import load_instruction_examples


class Phase8SftPipelineTests(unittest.TestCase):
    def _multi_file_record(self, prompt: str, summary: str = "paired patch"):
        payload = {
            "format": "SYNTRA_AGENT_PATCH_V1",
            "summary": summary,
            "edits": [
                {
                    "path": "src/implementation.py",
                    "find": "value == None",
                    "replace": "value is None",
                    "reason": "Use identity for the Python None singleton.",
                },
                {
                    "path": "src/consumer.py",
                    "find": "item == None",
                    "replace": "item is None",
                    "reason": "Use identity for the Python None singleton.",
                },
            ],
        }
        return {
            "prompt": prompt,
            "completion": "<thought>\n1. Follow the import.\n2. Verify both anchors.\n</thought>\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "kind": "multi_file_reasoning",
            "source_path": "src/implementation.py",
            "split_group": "multi-file:phase8-test-pair",
        }

    def test_training_profiles_select_all_five_phase7_languages(self):
        paths = (
            "phase4_milestone4.json",
            "phase4_smoke.json",
            "local_cpu_fast.json",
            "local_gpu_scale.json",
        )
        expected = {"python", "javascript", "typescript", "go", "rust"}
        for name in paths:
            with self.subTest(profile=name):
                profile = evaluate_epochs.load_profile(ROOT / "configs" / "training" / name)
                evaluate_epochs._validate_profile_values(profile)
                self.assertEqual(profile["curriculum"]["mode"], "all")
                self.assertEqual(set(profile["curriculum"]["languages"]), expected)
                self.assertEqual(len(profile["curriculum"]["languages"]), 5)

    def test_epoch_runner_invokes_all_mode_and_merges_both_phase7_outputs(self):
        languages = ["python", "javascript", "typescript", "go", "rust"]
        with tempfile.TemporaryDirectory(prefix=".syntra-phase8-curriculum-", dir=ROOT) as temp:
            root = Path(temp)
            repository_output = root / "repository.jsonl"
            reasoning_output = root / "reasoning.jsonl"
            final_output = root / "combined.jsonl"
            repository_records = [
                {
                    "prompt": "Explain the local function and its callers.",
                    "completion": "It normalizes values before validation.",
                    "kind": "module_comprehension",
                    "source_path": "src/module.py",
                    "split_group": "repository:module.py",
                },
                {
                    "prompt": "Repair the verified Python sentinel comparison.",
                    "completion": json.dumps({
                        "format": "SYNTRA_AGENT_PATCH_V1",
                        "summary": "Use identity for the sentinel.",
                        "edits": [{
                            "path": "src/module.py",
                            "find": "value == None",
                            "replace": "value is None",
                            "reason": "None is a singleton sentinel.",
                        }],
                    }),
                    "kind": "single_file_repair",
                    "source_path": "src/module.py",
                    "split_group": "repository:module.py",
                },
            ]
            reasoning_record = self._multi_file_record(
                "Task: inspect the implementation and consumer.\n"
                "FILE: src/implementation.py\n"
                "def run(value): return value == None\n"
                "FILE: src/consumer.py\n"
                "def use(item): return item == None\n"
            )

            exclusions = [root / "previous-runs", root / "generated-curricula"]
            def fake_subprocess(command, **kwargs):
                args = list(command)
                self.assertIn("run", args)
                self.assertIn("ai:curriculum", args)
                self.assertEqual(args[args.index("--mode") + 1], "all")
                exclude_start = args.index("--exclude") + 1
                self.assertEqual(
                    args[exclude_start:exclude_start + len(exclusions)],
                    [str(path) for path in exclusions],
                )
                language_start = args.index("--languages") + 1
                self.assertEqual(args[language_start:language_start + 5], languages)
                self.assertEqual(args[args.index("--max-examples") + 1], "20")
                repo_path = Path(args[args.index("--output") + 1])
                reasoning_path = Path(args[args.index("--multifile-output") + 1])
                repo_path.write_text(
                    "".join(json.dumps(item) + "\n" for item in repository_records),
                    encoding="utf-8",
                )
                reasoning_path.write_text(json.dumps(reasoning_record) + "\n", encoding="utf-8")
                output = {
                    "repository": {"dataset_path": str(repo_path), "example_count": 2},
                    "multifile": {"dataset_path": str(reasoning_path), "example_count": 1},
                }
                return subprocess.CompletedProcess(
                    command, 0, stdout=json.dumps(output), stderr=""
                )

            with patch.object(evaluate_epochs, "_npm_command", side_effect=lambda args: ["npm", *args]), patch.object(
                evaluate_epochs.subprocess, "run", side_effect=fake_subprocess
            ) as command:
                generated = evaluate_epochs._run_phase7_curriculum(
                    root, repository_output, reasoning_output,
                    max_examples=20, languages=languages, exclude_paths=exclusions,
                )
            self.assertEqual(generated["mode"], "all")
            self.assertEqual(generated["languages"], languages)
            self.assertEqual(command.call_count, 1)

            merged = evaluate_epochs._merge_phase7_curricula(
                repository_output, reasoning_output, final_output,
                max_examples_per_generator=20,
            )
            self.assertEqual(merged["repository_records"], 2)
            self.assertEqual(merged["multi_file_reasoning_records"], 1)
            self.assertEqual(merged["combined_records"], 3)
            rows = [
                json.loads(line)
                for line in final_output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            multi_file = next(item for item in rows if item["kind"] == "multi_file_reasoning")
            self.assertIn("<thought>", multi_file["completion"])
            self.assertIn("SYNTRA_AGENT_PATCH_V1", multi_file["completion"])
            self.assertEqual(multi_file["split_group"], "multi-file:phase8-test-pair")

    def test_curriculum_cli_passes_exclusions_to_both_generators(self):
        from syntra_ai import cli

        repository_result = {"dataset_path": "repository.jsonl", "example_count": 2}
        reasoning_result = {"dataset_path": "reasoning.jsonl", "example_count": 1}
        with patch.object(
            cli, "synthesize_repository_curriculum", return_value=repository_result
        ) as repository, patch.object(
            cli, "synthesize_multifile_reasoning_curriculum", return_value=reasoning_result
        ) as reasoning:
            result = cli.main([
                "curriculum",
                "--mode", "all",
                "--languages", "python", "javascript", "typescript", "go", "rust",
                "--max-examples", "12",
                "--exclude", "data/benchmarks", "data/curricula", "data/checkpoints",
            ])
        self.assertEqual(result, 0)
        exclusions = ["data/benchmarks", "data/curricula", "data/checkpoints"]
        self.assertEqual(repository.call_args.kwargs["exclude_paths"], exclusions)
        self.assertEqual(reasoning.call_args.kwargs["exclude_paths"], exclusions)

    def test_multifile_context_filter_drops_only_overlong_prompts_and_keeps_full_completion_eos(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-phase8-context-", dir=ROOT) as temp:
            root = Path(temp)
            dataset = root / "dataset.jsonl"
            tokenizer = ByteBPETokenizer.empty()
            oversized = self._multi_file_record("x" * 512, summary="must be dropped")
            fitting = self._multi_file_record(
                "p" * 511,
                summary="retained " + ("reasoning " * 120),
            )
            repository = {
                "prompt": "A repository-level instruction with an unchanged prompt policy.",
                "completion": "A short verified repository completion.",
                "kind": "module_comprehension",
                "source_path": "src/module.py",
            }
            dataset.write_text(
                "\n".join(json.dumps(item) for item in (oversized, fitting, repository)) + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = evaluate_epochs._filter_multifile_prompts_to_checkpoint_context(
                    dataset, tokenizer, seq_len=512
                )
                rows = [
                    json.loads(line)
                    for line in dataset.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(result["multi_file_candidates"], 2)
                self.assertEqual(result["multi_file_records_kept"], 1)
                self.assertEqual(result["multi_file_records_dropped_prompt_overflow"], 1)
                self.assertEqual(result["multi_file_prompt_token_limit"], 511)
                self.assertEqual(len(rows), 2)
                kept = next(item for item in rows if item.get("kind") == "multi_file_reasoning")
                self.assertEqual(kept["prompt"], "p" * 511)

                # The full completion is split over actual checkpoint-sized
                # windows. Prompt targets remain -100; all completion tokens
                # and the terminal EOS remain supervised exactly once.
                encoded = load_instruction_examples(
                    dataset, seq_len=512, tokenizer=tokenizer,
                    max_examples=10, max_tokens=100_000,
                )
                supervised = [
                    token for item in encoded for token in item["labels"]
                    if token != -100
                ]
                expected = (
                    tokenizer.encode(fitting["completion"]) + [EOS_ID]
                    + tokenizer.encode(repository["completion"]) + [EOS_ID]
                )
                self.assertEqual(supervised, expected)
                self.assertGreater(len(encoded), 2)

    def test_multifile_merger_rejects_unsafe_patch_paths(self):
        with tempfile.TemporaryDirectory(prefix=".syntra-phase8-invalid-patch-", dir=ROOT) as temp:
            root = Path(temp)
            repository = root / "repository.jsonl"
            reasoning = root / "reasoning.jsonl"
            output = root / "combined.jsonl"
            repository.write_text(
                json.dumps({"prompt": "repository task", "completion": "repository answer"}) + "\n",
                encoding="utf-8",
            )
            record = self._multi_file_record(
                "Task: patch files.\n"
                "FILE: ../outside.py\n"
                "FILE: src/consumer.py\n"
            )
            payload = json.loads(record["completion"].split("</thought>\n", 1)[1])
            payload["edits"][0]["path"] = "../outside.py"
            record["completion"] = "<thought>\nFind the related modules.\n</thought>\n" + json.dumps(payload)
            reasoning.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(evaluate_epochs.OrchestrationError, "unsafe or ungrounded"):
                evaluate_epochs._merge_phase7_curricula(
                    repository, reasoning, output, max_examples_per_generator=10
                )
            self.assertFalse(output.exists())


class LocalBaselinePreflightTests(unittest.TestCase):
    def test_cpu_device_preflight_runs_a_real_pytorch_tensor_operation(self):
        report = run_local_baseline.probe_torch_devices("cpu")
        self.assertTrue(report["python"]["ready"])
        self.assertTrue(report["pytorch"]["ready"])
        self.assertTrue(report["devices"]["cpu"]["available"])
        self.assertTrue(report["devices"]["cpu"]["ready"])
        self.assertTrue(report["selected_device_ready"])
        self.assertEqual(report["selected_device"], "cpu")

    def test_docker_image_inspection_requires_exact_tag_and_sha256_digests(self):
        image = "python:3.12-slim"
        valid = [{
            "Id": "sha256:" + "a" * 64,
            "RepoTags": [image],
            "RepoDigests": ["docker.io/library/python@sha256:" + "b" * 64],
            "Os": "linux",
            "Architecture": "amd64",
        }]
        checked = run_local_baseline.validate_docker_image_inspect(image, valid)
        self.assertTrue(checked["ready"])
        self.assertTrue(checked["id_sha256_valid"])
        self.assertTrue(checked["repository_digest_valid"])
        self.assertTrue(checked["tag_matches"])

        no_digest = [dict(valid[0], RepoDigests=[])]
        rejected = run_local_baseline.validate_docker_image_inspect(image, no_digest)
        self.assertFalse(rejected["ready"])
        self.assertIn("repository sha256 digest", rejected["error"])

        wrong_tag = [dict(valid[0], RepoTags=["python:latest"])]
        rejected_tag = run_local_baseline.validate_docker_image_inspect(image, wrong_tag)
        self.assertFalse(rejected_tag["ready"])
        self.assertIn("requested tag", rejected_tag["error"])

    def test_docker_preflight_uses_only_daemon_info_and_local_image_inspect(self):
        commands = []

        def fake_runner(command, **kwargs):
            commands.append(list(command))
            if command[1:3] == ["info", "--format"]:
                return subprocess.CompletedProcess(command, 0, stdout="27.3.1\n", stderr="")
            if command[1:3] == ["image", "inspect"]:
                image = command[3]
                name = image.split(":", 1)[0]
                result = [{
                    "Id": "sha256:" + "a" * 64,
                    "RepoTags": [image],
                    "RepoDigests": [f"docker.io/library/{name}@sha256:" + "b" * 64],
                    "Os": "linux",
                    "Architecture": "amd64",
                }]
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps(result), stderr="")
            self.fail("Preflight invoked an unsupported Docker operation: " + " ".join(command))

        with patch.object(run_local_baseline.shutil, "which", return_value="/usr/bin/docker"):
            result = run_local_baseline.inspect_local_docker_images(fake_runner)
        self.assertTrue(result["engine_ready"])
        self.assertTrue(result["images_ready"])
        self.assertEqual(
            {item["image"] for item in result["images"]},
            set(run_local_baseline.REQUIRED_DOCKER_IMAGES),
        )
        self.assertEqual(len(commands), 1 + len(run_local_baseline.REQUIRED_DOCKER_IMAGES))
        self.assertTrue(all("pull" not in command and "build" not in command and "run" not in command for command in commands))
        self.assertTrue(all(command[1:3] in (["info", "--format"], ["image", "inspect"]) for command in commands))

    def test_preflight_reports_no_go_when_a_required_image_has_no_repository_digest(self):
        with patch.object(run_local_baseline, "_load_profile", return_value={
            "profile": "unit-test",
            "device": "cpu",
        }), patch.object(run_local_baseline, "probe_torch_devices", return_value={
            "python": {"ready": True},
            "pytorch": {"ready": True},
            "devices": {"cpu": {"available": True, "ready": True}},
            "cpu_required_ready": True,
            "selected_device_ready": True,
            "selected_device": "cpu",
        }), patch.object(run_local_baseline, "inspect_local_docker_images", return_value={
            "engine_ready": True,
            "images_ready": False,
            "images": [],
        }):
            report = run_local_baseline.build_preflight_report(device="cpu")
        self.assertEqual(report["status"], "NO-GO")
        self.assertFalse(report["checks"]["local_docker_images_with_sha256"])


if __name__ == "__main__":
    unittest.main()

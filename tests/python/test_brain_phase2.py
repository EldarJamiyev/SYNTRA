import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from brain.tokenizer import BytePairTokenizer


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run brain training tests.")
class TransformerCacheTests(unittest.TestCase):
    def test_incremental_kv_cache_logits_match_full_context(self):
        from brain.model import LocalTransformer, TransformerConfig

        config = TransformerConfig(
            vocab_size=258,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        model = LocalTransformer(config).eval()
        tokens = torch.tensor([[256, 100, 101, 102, 103, 104, 105]], dtype=torch.long)
        with torch.inference_mode():
            full_logits, _ = model(tokens)
            _, _, cache = model(tokens[:, :3], use_cache=True)
            for index in range(3, tokens.shape[1]):
                step_logits, _, cache = model(
                    tokens[:, index:index + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                torch.testing.assert_close(
                    step_logits[:, -1, :],
                    full_logits[:, index, :],
                    rtol=1e-4,
                    atol=1e-4,
                )
        self.assertEqual(cache[0][0].shape[-2], tokens.shape[1])

    def test_model_rejects_invalid_cache_dimensions_and_context_overflow(self):
        from brain.model import LocalTransformer, TransformerConfig

        config = TransformerConfig(
            vocab_size=258,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        model = LocalTransformer(config).eval()
        tokens = torch.ones((1, 4), dtype=torch.long)
        with torch.inference_mode():
            _, _, cache = model(tokens, use_cache=True)
            with self.assertRaisesRegex(ValueError, "exceeds block_size"):
                model(torch.ones((1, 13), dtype=torch.long), past_key_values=cache)
            malformed = ((torch.zeros((1, 1, 4, 16)), torch.zeros((1, 1, 4, 16))),)
            with self.assertRaisesRegex(ValueError, "batch/head dimensions"):
                model(tokens[:, :1], past_key_values=malformed)


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run brain training tests.")
class LocalTrainingPipelineTests(unittest.TestCase):
    def _config(self):
        from brain.model import TransformerConfig

        return TransformerConfig(
            vocab_size=270,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )

    def _dataset(self, path: Path):
        source = (
            "def add(left, right):\n"
            "    return left + right\n\n"
            "class Calculator:\n"
            "    def multiply(self, left, right):\n"
            "        return left * right\n\n"
        ) * 250
        path.write_text(source, encoding="utf-8")
        return source

    def test_instruction_context_blocks_mask_prompt_and_supervise_response(self):
        from brain.tokenizer import BytePairTokenizer
        from brain.train import _instruction_context_blocks

        records = [{
            "prompt": "Write a function that returns the larger integer.",
            "completion": "def larger(a, b): return a if a >= b else b",
        }]
        tokenizer = BytePairTokenizer.empty()
        inputs, targets = _instruction_context_blocks(records, tokenizer, block_size=128)

        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(targets), 1)
        prompt_prefix = "Instruction:\n" + records[0]["prompt"] + "\nResponse:\n"
        prompt_token_count = len(tokenizer.encode(prompt_prefix))
        self.assertTrue(all(token == -100 for token in targets[0][:prompt_token_count]))
        self.assertNotEqual(targets[0][prompt_token_count], -100)
        self.assertIn(257, targets[0], "EOS must remain supervised after the response.")

    def test_structured_jsonl_reader_and_plain_text_fallback(self):
        from brain.train import _read_instruction_jsonl_datasets, _read_local_datasets

        with tempfile.TemporaryDirectory(prefix="syntra-brain-jsonl-loader-") as temp:
            root = Path(temp)
            structured = root / "instructions.jsonl"
            structured.write_text(
                json.dumps({"Instruction": "Explain addition.", "Response": "Add the two values."}) + "\n" +
                json.dumps({"prompt": "Explain multiplication.", "completion": "Multiply the two values."}),
                encoding="utf-8",
            )
            plain = root / "legacy.jsonl"
            plain.write_text("legacy source text\nwith a second line\n" * 100, encoding="utf-8")
            long_plain = root / "legacy-long-line.jsonl"
            long_plain.write_text("long raw text fallback " + ("x" * (1024 * 1024 + 1)), encoding="utf-8")

            (
                records,
                files,
                byte_count,
                fingerprint,
                scan_bytes,
                scan_files,
                fallback_files,
            ) = _read_instruction_jsonl_datasets([root], 2 * 1024 * 1024)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["prompt"], "Explain addition.")
            self.assertEqual(records[1]["completion"], "Multiply the two values.")
            self.assertEqual(len(files), 1)
            self.assertGreater(byte_count, 0)
            self.assertEqual(len(fingerprint), 64)
            self.assertEqual(scan_files, 3)
            self.assertEqual(
                set(fallback_files),
                {str(plain.resolve()), str(long_plain.resolve())},
            )

            train_text, validation_text, raw_files, _, _ = _read_local_datasets(
                [root],
                2 * 1024 * 1024 - scan_bytes,
                exclude_paths=set(files),
                preloaded_files=fallback_files,
                already_scanned_files=scan_files,
            )
            self.assertIn("legacy source text", train_text + validation_text)
            self.assertIn("long raw text fallback", train_text + validation_text)
            self.assertEqual(len(raw_files), 2)

    def test_jsonl_instruction_schema_fails_closed_after_detection(self):
        from brain.train import _read_instruction_jsonl_datasets

        with tempfile.TemporaryDirectory(prefix="syntra-brain-jsonl-schema-") as temp:
            root = Path(temp)
            malformed_json = root / "malformed.jsonl"
            malformed_json.write_text(
                json.dumps({"Instruction": "Explain alpha.", "Response": "Alpha is valid."})
                + "\n"
                + '{"Instruction": "Explain beta.",',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"Invalid instruction JSON.*malformed.jsonl:2"):
                _read_instruction_jsonl_datasets([malformed_json], 1024 * 1024)

            malformed_fields = root / "missing-response.jsonl"
            malformed_fields.write_text(
                json.dumps({"Instruction": "Explain alpha.", "Response": "Alpha is valid."})
                + "\n"
                + json.dumps({"Instruction": "Explain beta."}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"Incomplete instruction schema.*missing-response.jsonl:2"):
                _read_instruction_jsonl_datasets([malformed_fields], 1024 * 1024)

            invalid_group = root / "invalid-group.jsonl"
            invalid_group.write_text(
                json.dumps({
                    "Instruction": "Explain alpha.",
                    "Response": "Alpha is valid.",
                    "split_group": ["not", "a", "string"],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"split_group.*invalid-group.jsonl:1"):
                _read_instruction_jsonl_datasets([invalid_group], 1024 * 1024)

            generic_jsonl = root / "generic.jsonl"
            generic_jsonl.write_text(
                json.dumps({"event": "startup", "status": "ready"}) + "\n",
                encoding="utf-8",
            )
            (
                records,
                structured_paths,
                _instruction_bytes,
                _instruction_hash,
                _scan_bytes,
                _scan_files,
                fallback_files,
            ) = _read_instruction_jsonl_datasets([generic_jsonl], 1024 * 1024)
            self.assertEqual(records, [])
            self.assertEqual(structured_paths, [])
            self.assertIn(str(generic_jsonl.resolve()), fallback_files)

    def test_multiple_selected_roots_keep_colliding_jsonl_paths_distinct(self):
        from brain.train import _read_instruction_jsonl_datasets, _read_local_datasets

        with tempfile.TemporaryDirectory(prefix="syntra-multi-root-jsonl-") as temp:
            root = Path(temp)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first_source = first_root / "records.jsonl"
            second_source = second_root / "records.jsonl"
            first_source.write_text("raw fallback from first root\n", encoding="utf-8")
            second_source.write_text("raw fallback from second root\n", encoding="utf-8")

            (
                records,
                structured_files,
                instruction_bytes,
                _instruction_hash,
                scan_bytes,
                scan_files,
                fallback_files,
            ) = _read_instruction_jsonl_datasets(
                [first_root, second_root],
                1024 * 1024,
            )
            self.assertEqual(records, [])
            self.assertEqual(structured_files, [])
            self.assertEqual(instruction_bytes, 0)
            self.assertEqual(scan_files, 2)
            self.assertEqual(
                {item[0] for item in fallback_files.values()},
                {"records.jsonl", "selected-root-2/records.jsonl"},
            )

            train_text, validation_text, raw_files, raw_bytes, _raw_hash = _read_local_datasets(
                [first_root, second_root],
                1024 * 1024 - scan_bytes,
                preloaded_files=fallback_files,
                already_scanned_files=scan_files,
            )
            combined = train_text + validation_text
            self.assertIn("first root", combined)
            self.assertIn("second root", combined)
            self.assertIn("# SYNTRA_SOURCE_FILE: records.jsonl", combined)
            self.assertIn("# SYNTRA_SOURCE_FILE: selected-root-2/records.jsonl", combined)
            self.assertEqual(set(raw_files), {str(first_source.resolve()), str(second_source.resolve())})
            self.assertEqual(raw_bytes, first_source.stat().st_size + second_source.stat().st_size)

    def test_excluded_curriculum_jsonl_is_omitted_from_both_raw_and_structured_pretraining(self):
        from brain.train import _read_instruction_jsonl_datasets, _read_local_datasets, train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-curriculum-exclusion-") as temp:
            root = Path(temp)
            source = root / "source.py"
            docs = root / "README.md"
            curriculum = root / "workspace-instructions.jsonl"
            source.write_text("def source_function(value):\\n    return value + 3\\n" * 100, encoding="utf-8")
            docs.write_text("# Repository corpus\\n" + ("source documentation and design notes\\n" * 500), encoding="utf-8")
            curriculum.write_text(
                "\\n".join(json.dumps({
                    "prompt": f"Implement repository function number {index}.",
                    "completion": f"def generated_{index}(value): return value + {index}",
                    "split_group": f"module:{index}",
                }) for index in range(6)),
                encoding="utf-8",
            )
            benchmark_artifacts = root / "data" / "benchmarks"
            benchmark_artifacts.mkdir(parents=True)
            (benchmark_artifacts / "old-run.json").write_text(
                json.dumps({"model": {"checkpoint_sha256": "not-source-code"}}),
                encoding="utf-8",
            )
            excluded = {str(curriculum.resolve()), str(benchmark_artifacts.resolve())}

            records, names, _size, _digest, _scan, _count, fallbacks = _read_instruction_jsonl_datasets(
                [root], 4 * 1024 * 1024, exclude_paths=excluded
            )
            self.assertEqual(records, [])
            self.assertNotIn(str(curriculum.resolve()), names)
            self.assertNotIn(str(curriculum.resolve()), fallbacks)

            train_text, validation_text, raw_names, _raw_size, _raw_digest = _read_local_datasets(
                [root], 4 * 1024 * 1024, exclude_paths=excluded
            )
            self.assertNotIn("Implement repository function number", train_text + validation_text)
            self.assertNotIn(str(curriculum.resolve()), raw_names)

            checkpoint = root / "checkpoint.pt"
            result = train(
                [root],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                exclude_dataset_paths=[curriculum, benchmark_artifacts],
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["instruction_examples"], 0)
            self.assertEqual(result["dataset_files"], 2)
            from brain.checkpoint import load_checkpoint
            saved = load_checkpoint(checkpoint)
            self.assertEqual(set(saved["dataset_files"]), {str(source.resolve()), str(docs.resolve())})

    def test_structured_jsonl_training_saves_masked_training_format(self):
        from brain.checkpoint import load_checkpoint
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-jsonl-train-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            records = [
                {"Instruction": "Write a function that adds two integers.", "Response": "def add(a, b): return a + b"},
                {"Instruction": "Write a function that multiplies two integers.", "Response": "def multiply(a, b): return a * b"},
                {"Instruction": "Write a function that clamps an integer.", "Response": "def clamp(x, low, high): return min(max(x, low), high)"},
                {"Instruction": "Write a function that returns the larger integer.", "Response": "def larger(a, b): return max(a, b)"},
            ]
            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            checkpoint = root / "checkpoints" / "instruction-model.pt"

            result = train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=2,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=100,
                device_name="cpu",
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["training_data_format"], "instruction-jsonl-masked-v1")
            self.assertEqual(result["instruction_examples"], 4)
            self.assertGreater(result["training_tokens"], 0)
            self.assertGreater(result["validation_tokens"], 0)
            self.assertEqual(
                result["supervised_tokens_processed"],
                result["training_tokens"],
                "one full epoch must process every supervised target exactly once",
            )
            self.assertEqual(result["throughput_basis"], "supervised_target_tokens")

            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["training_data_format"], "instruction-jsonl-masked-v1")
            self.assertEqual(saved["instruction_example_count"], 4)
            self.assertGreater(saved["training_context_blocks"], 0)

            resumed = train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=2,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=100,
                device_name="cpu",
                resume=True,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["training_data_format"], "instruction-jsonl-masked-v1")
            self.assertGreater(resumed["step"], result["step"])
            resumed_saved = load_checkpoint(checkpoint)
            self.assertEqual(resumed_saved["training_data_format"], "instruction-jsonl-masked-v1")
            self.assertEqual(resumed_saved["instruction_example_count"], 4)

    def test_tokenizer_is_fitted_only_on_instruction_training_records(self):
        from brain.checkpoint import load_checkpoint
        from brain.tokenizer import BytePairTokenizer
        from brain.train import _split_instruction_records, train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-tokenizer-holdout-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            records = [
                {
                    "prompt": f"Write routine number {index}.",
                    "completion": f"def routine_{index}(value): return value + {index}",
                    "split_group": f"module:{index}",
                }
                for index in range(4)
            ]
            _training, validation = _split_instruction_records(records)
            validation_prompts = {item["prompt"] for item in validation}
            validation_marker = "VALIDATION_ONLY_TOKEN_SEQUENCE_ZEBRA_QUARTZ_"
            for item in records:
                if item["prompt"] in validation_prompts:
                    item["completion"] = (
                        "def validation_only(value): return "
                        + validation_marker * 30
                    )

            training_records, validation_records = _split_instruction_records(records)
            self.assertEqual(len(training_records), 3)
            self.assertEqual(len(validation_records), 1)
            training_text = "\n\n".join(
                "Instruction:\n" + item["prompt"] + "\nResponse:\n" + item["completion"]
                for item in training_records
            )
            full_text = "\n\n".join(
                "Instruction:\n" + item["prompt"] + "\nResponse:\n" + item["completion"]
                for item in records
            )
            expected_training_tokenizer = BytePairTokenizer.train_from_text(
                training_text,
                vocab_size=self._config().vocab_size,
                max_training_bytes=1024,
            )
            full_dataset_tokenizer = BytePairTokenizer.train_from_text(
                full_text,
                vocab_size=self._config().vocab_size,
                max_training_bytes=1024,
            )
            self.assertNotEqual(
                expected_training_tokenizer,
                full_dataset_tokenizer,
                "the held-out completion must materially affect a tokenizer fitted on all records",
            )

            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            checkpoint = root / "training-only-tokenizer.pt"
            train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=2,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=100,
                device_name="cpu",
                seed=1337,
            )
            saved = load_checkpoint(checkpoint)
            actual_tokenizer = BytePairTokenizer.from_dict(saved["tokenizer"])
            self.assertEqual(
                actual_tokenizer,
                expected_training_tokenizer,
                "the persisted BPE merges must be learned from training prompts/responses only",
            )

    def test_mixed_raw_text_and_instruction_jsonl_training(self):
        from brain.checkpoint import load_checkpoint
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-mixed-training-") as temp:
            root = Path(temp)
            instructions = root / "instructions.jsonl"
            records = [
                {"Instruction": "Explain addition.", "Response": "Addition combines two values."},
                {"Instruction": "Explain multiplication.", "Response": "Multiplication scales a value."},
                {"Instruction": "Explain subtraction.", "Response": "Subtraction finds a difference."},
                {"Instruction": "Explain division.", "Response": "Division splits a value into parts."},
            ]
            instructions.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            raw = root / "legacy.txt"
            raw.write_text(("Legacy raw corpus remains part of causal language-model training.\n" * 120), encoding="utf-8")
            checkpoint = root / "checkpoints" / "mixed-model.pt"

            result = train(
                [root],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=64,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=100,
                device_name="cpu",
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["training_data_format"], "mixed-raw-and-instruction-jsonl-v1")
            self.assertEqual(result["instruction_examples"], 4)
            saved = load_checkpoint(checkpoint)
            self.assertIn(str(raw.resolve()), saved["dataset_files"])
            self.assertIn(str(instructions.resolve()), saved["dataset_files"])
            self.assertGreater(result["training_context_blocks"], 0)
            self.assertGreater(result["validation_context_blocks"], 0)

    def test_structured_and_raw_jsonl_scans_share_file_and_byte_budgets_without_rereading(self):
        import importlib
        train_module = importlib.import_module("brain.train")

        with tempfile.TemporaryDirectory(prefix="syntra-brain-shared-dataset-budget-") as temp:
            root = Path(temp)
            structured = root / "instructions.jsonl"
            structured_records = [
                {"Instruction": "Explain alpha.", "Response": "Alpha is the first operation."},
                {"Instruction": "Explain beta.", "Response": "Beta is the second operation."},
            ]
            structured.write_text(
                "\n".join(json.dumps(item) for item in structured_records),
                encoding="utf-8",
            )
            raw_jsonl = root / "legacy.jsonl"
            raw_text = "Legacy raw language-model source remains supported.\n" * 10
            raw_jsonl.write_text(raw_text, encoding="utf-8")
            source = root / "module.py"
            source_text = "def transform(value): return value + 1\n" * 5
            source.write_text(source_text, encoding="utf-8")

            real_reader = train_module._read_bounded_regular_file
            with patch.object(train_module, "_read_bounded_regular_file", wraps=real_reader) as read_file:
                (
                    records,
                    structured_paths,
                    instruction_bytes,
                    _instruction_hash,
                    scan_bytes,
                    scan_files,
                    fallback_files,
                ) = train_module._read_instruction_jsonl_datasets([root], 1024 * 1024)
                self.assertEqual(len(records), 2)
                self.assertEqual(scan_files, 2)
                self.assertEqual(scan_bytes, structured.stat().st_size + raw_jsonl.stat().st_size)
                self.assertEqual(set(fallback_files), {str(raw_jsonl.resolve())})

                train_text, validation_text, raw_paths, raw_bytes, _raw_hash = (
                    train_module._read_local_datasets(
                        [root],
                        1024 * 1024 - scan_bytes,
                        exclude_paths=set(structured_paths),
                        allow_empty=False,
                        require_holdout=False,
                        max_files=20,
                        preloaded_files=fallback_files,
                        already_scanned_files=scan_files,
                    )
                )
                self.assertIn(raw_text, train_text + validation_text)
                self.assertIn(source_text, train_text + validation_text)
                self.assertIn(str(raw_jsonl.resolve()), raw_paths)
                self.assertIn(str(source.resolve()), raw_paths)
                self.assertEqual(raw_bytes, len(raw_text.encode("utf-8")) + len(source_text.encode("utf-8")))
                self.assertLessEqual(instruction_bytes + raw_bytes, 1024 * 1024)
                self.assertEqual(
                    read_file.call_count,
                    3,
                    "both JSONL candidates are read once and the raw fallback is reused from memory",
                )

            with self.assertRaisesRegex(ValueError, "2-file limit"):
                train_module._read_local_datasets(
                    [root],
                    1024 * 1024 - scan_bytes,
                    exclude_paths=set(structured_paths),
                    allow_empty=False,
                    require_holdout=False,
                    max_files=scan_files,
                    preloaded_files=fallback_files,
                    already_scanned_files=scan_files,
                )

    def test_long_instruction_response_is_packed_without_missing_or_duplicate_targets(self):
        from brain.tokenizer import BytePairTokenizer, EOS_ID
        from brain.train import _instruction_context_blocks

        response = "def transform(value): return value + 1\n" * 24
        record = {"prompt": "Implement the transform function.", "completion": response}
        tokenizer = BytePairTokenizer.empty()
        inputs, targets = _instruction_context_blocks([record], tokenizer, block_size=64)

        self.assertGreater(len(inputs), 1)
        prompt = "Instruction:\n" + record["prompt"] + "\nResponse:\n"
        supervised = [
            token
            for row in targets
            for token in row
            if token != -100
        ]
        self.assertEqual(supervised, tokenizer.encode(response) + [EOS_ID])
        self.assertTrue(all(len(row) == 64 for row in inputs))
    def test_instruction_record_limit_is_enforced_during_jsonl_parsing(self):
        import importlib
        train_module = importlib.import_module("brain.train")

        with tempfile.TemporaryDirectory(prefix="syntra-brain-record-cap-") as temp:
            root = Path(temp)
            dataset = root / "records.jsonl"
            dataset.write_text(
                "\n".join(json.dumps({
                    "Instruction": f"Explain unique operation number {index}.",
                    "Response": f"Operation number {index} is local and deterministic.",
                }) for index in range(3)),
                encoding="utf-8",
            )
            with patch.object(train_module, "MAX_INSTRUCTION_RECORDS", 2):
                with self.assertRaisesRegex(ValueError, "2-record safety limit"):
                    train_module._read_instruction_jsonl_datasets([dataset], 1024 * 1024)

    def test_checkpoint_rng_restore_is_exact_and_rejects_malformed_state(self):
        import torch
        from brain.train import _restore_rng_state, _rng_state

        device = torch.device("cpu")
        torch.manual_seed(12345)
        saved_state = _rng_state(device)
        expected = torch.rand(16)

        torch.manual_seed(98765)
        _restore_rng_state(saved_state, device)
        actual = torch.rand(16)
        self.assertTrue(torch.equal(actual, expected))

        malformed = [
            None,
            {},
            {"cpu": torch.tensor([1, 2, 3], dtype=torch.int64)},
            {"cpu": torch.empty(0, dtype=torch.uint8)},
            {"cpu": "not-a-tensor"},
        ]
        for state in malformed:
            with self.subTest(state=type(state).__name__):
                with self.assertRaisesRegex(ValueError, "random-generator state"):
                    _restore_rng_state(state, device)

    def test_resume_order_validation_rejects_corrupt_permutations_and_cursors(self):
        import torch
        from brain.train import _validate_resume_order

        order, next_batch = _validate_resume_order(
            torch.tensor([2, 0, 1], dtype=torch.long),
            1,
            chunk_count=3,
            steps_per_epoch=2,
        )
        self.assertTrue(torch.equal(order, torch.tensor([2, 0, 1])))
        self.assertEqual(next_batch, 1)

        invalid_orders = [
            torch.tensor([0, 0, 1], dtype=torch.long),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0, 1, 3], dtype=torch.long),
        ]
        for invalid in invalid_orders:
            with self.subTest(order=invalid.tolist()):
                with self.assertRaisesRegex(ValueError, "epoch order"):
                    _validate_resume_order(invalid, 0, chunk_count=3, steps_per_epoch=2)

        for invalid_cursor in (-1, 3, 1.5, True):
            with self.subTest(cursor=invalid_cursor):
                with self.assertRaisesRegex(ValueError, "resume position"):
                    _validate_resume_order(
                        torch.tensor([0, 1, 2], dtype=torch.long),
                        invalid_cursor,
                        chunk_count=3,
                        steps_per_epoch=2,
                    )

        with self.assertRaisesRegex(ValueError, "without a saved epoch permutation"):
            _validate_resume_order(None, 1, chunk_count=3, steps_per_epoch=2)

    def test_global_learning_rate_schedule_is_stable_across_resume_boundaries(self):
        from brain.train import _scheduled_learning_rate

        base = 3e-4
        schedule = [_scheduled_learning_rate(base, step) for step in range(1, 250)]
        first_run = [_scheduled_learning_rate(base, step) for step in range(1, 101)]
        resumed_run = [_scheduled_learning_rate(base, step) for step in range(101, 250)]
        self.assertEqual(schedule, first_run + resumed_run)
        self.assertGreater(schedule[0], 0)
        self.assertGreater(schedule[99], schedule[100])
        self.assertGreaterEqual(min(schedule), 1e-5)
        self.assertAlmostEqual(_scheduled_learning_rate(base, 100), base, places=12)
        self.assertAlmostEqual(_scheduled_learning_rate(base, 10_000), 1e-5, places=12)
        self.assertEqual(_scheduled_learning_rate(base, 10_001), 1e-5)
        self.assertAlmostEqual(
            _scheduled_learning_rate(base, 500, minimum_learning_rate=1e-5, warmup_steps=100, decay_steps=500),
            1e-5,
            places=12,
        )

    def test_gradient_accumulation_counts_optimizer_steps_and_emits_hardware_telemetry(self):
        from brain.checkpoint import load_checkpoint
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-accumulation-") as temp:
            root = Path(temp)
            dataset = root / "dataset.txt"
            self._dataset(dataset)
            checkpoint = root / "checkpoints" / "accumulated.pt"
            result = train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                max_steps=2,
                batch_size=1,
                gradient_accumulation_steps=2,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=1,
                device_name="cpu",
                seed=321,
            )
            saved = load_checkpoint(checkpoint)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["step"], 2, "steps count optimizer updates, not micro-batches")
            self.assertEqual(result["gradient_accumulation_steps"], 2)
            self.assertEqual(saved["training_settings"]["gradient_accumulation_steps"], 2)
            self.assertEqual(saved["step"], 2)
            self.assertGreater(result["tokens_per_second"], 0)
            self.assertGreater(result["step_latency_ms"], 0)
            self.assertGreater(result["mean_step_latency_ms"], 0)
            self.assertIsInstance(result["peak_rss_bytes"], int)
            self.assertGreater(result["peak_rss_bytes"], 0)
            self.assertIsNone(result["peak_device_memory_bytes"])
            self.assertEqual(result["device_memory_metric"], "cpu_not_applicable")

    def test_exact_max_step_budget_saves_resumable_mid_epoch_checkpoint_and_clips_gradients(self):
        from brain.checkpoint import load_checkpoint
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-step-budget-") as temp:
            root = Path(temp)
            dataset = root / "dataset.txt"
            self._dataset(dataset)
            checkpoint = root / "checkpoints" / "step-budget.pt"
            original_clip = torch.nn.utils.clip_grad_norm_
            with patch(
                "torch.nn.utils.clip_grad_norm_",
                wraps=original_clip,
            ) as clip:
                first = train(
                    [dataset],
                    checkpoint_path=checkpoint,
                    config=self._config(),
                    epochs=1,
                    max_steps=3,
                    batch_size=128,
                    tokenizer_bytes=1024,
                    evaluation_batches=1,
                    save_every_steps=2,
                    device_name="cpu",
                    seed=123,
                )
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["step"], 3)
            self.assertEqual(clip.call_count, 3)
            self.assertTrue(all(call.kwargs["max_norm"] == 1.0 for call in clip.call_args_list))
            self.assertTrue(all(call.kwargs["error_if_nonfinite"] is True for call in clip.call_args_list))
            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["step"], 3)
            self.assertEqual(saved["training_settings"]["scheduler"], "global-cosine-warmup-v2")
            self.assertIsInstance(saved["epoch_order"], torch.Tensor)
            self.assertGreaterEqual(saved["next_batch_index"], 0)

            resumed = train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                max_steps=2,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                seed=123,
                resume=True,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["step"], 5)
            self.assertGreaterEqual(resumed["gradient_norm_before_clip"], 0.0)
            self.assertGreaterEqual(resumed["learning_rate"], 1e-5)

            boundary_checkpoint = root / "checkpoints" / "epoch-boundary.pt"
            boundary = train(
                [dataset],
                checkpoint_path=boundary_checkpoint,
                config=self._config(),
                epochs=1,
                max_steps=first["steps_per_epoch"],
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                seed=123,
            )
            boundary_saved = load_checkpoint(boundary_checkpoint)
            self.assertEqual(boundary["epoch"], 1)
            self.assertEqual(boundary_saved["epoch"], 1)
            self.assertIsNone(boundary_saved["epoch_order"])
            self.assertEqual(boundary_saved["next_batch_index"], 0)
            boundary_resumed = train(
                [dataset],
                checkpoint_path=boundary_checkpoint,
                config=self._config(),
                epochs=1,
                max_steps=1,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                seed=123,
                resume=True,
            )
            self.assertEqual(boundary_resumed["step"], boundary["step"] + 1)

    def test_source_related_instruction_examples_never_cross_validation_split(self):
        from brain.train import _split_instruction_records

        records = [
            {"prompt": "Implement alpha.", "completion": "alpha()", "split_group": "symbol:alpha"},
            {"prompt": "Explain alpha.", "completion": "alpha is a function.", "split_group": "symbol:alpha"},
            {"prompt": "Implement beta.", "completion": "beta()", "split_group": "symbol:beta"},
            {"prompt": "Explain beta.", "completion": "beta is a function.", "split_group": "symbol:beta"},
            {"prompt": "Repeated generic prompt.", "completion": "first completion.", "split_group": "module:gamma"},
            {"prompt": "Repeated generic prompt.", "completion": "second completion.", "split_group": "module:delta"},
            {"prompt": "Explain gamma.", "completion": "gamma module.", "split_group": "module:gamma"},
        ]
        training, validation = _split_instruction_records(records)

        memberships = {}
        for split_name, collection in (("train", training), ("validation", validation)):
            for record in collection:
                group = record["split_group"]
                memberships.setdefault(group, set()).add(split_name)
        self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))

        repeated_prompt_memberships = set()
        for split_name, collection in (("train", training), ("validation", validation)):
            if any(item["prompt"] == "Repeated generic prompt." for item in collection):
                repeated_prompt_memberships.add(split_name)
        self.assertEqual(len(repeated_prompt_memberships), 1)

    def test_train_checkpoint_resume_and_generation_pipeline(self):
        from brain.checkpoint import load_checkpoint, read_checkpoint_metadata
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-train-") as temp:
            root = Path(temp)
            dataset = root / "dataset.txt"
            self._dataset(dataset)
            checkpoint = root / "checkpoints" / "model.pt"

            first = train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                seed=123,
            )
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["epoch"], 1)
            self.assertGreater(first["estimated_training_memory_bytes"], 0)
            self.assertIsNotNone(first["train_loss"])
            self.assertIsNotNone(first["validation_loss"])
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(checkpoint.with_suffix(".pt.json").is_file())

            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["training_settings"]["scheduler"], "global-cosine-warmup-v2")
            self.assertEqual(saved["training_settings"]["minimum_learning_rate"], 1e-5)
            self.assertEqual(saved["training_settings"]["warmup_steps"], 100)
            self.assertEqual(saved["epoch"], 1)
            uninterrupted_checkpoint = root / "checkpoints" / "uninterrupted.pt"
            uninterrupted = train(
                [dataset],
                checkpoint_path=uninterrupted_checkpoint,
                config=self._config(),
                epochs=2,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                seed=123,
            )
            self.assertEqual(uninterrupted["status"], "completed")

            self.assertGreater(saved["step"], 0)
            self.assertEqual(saved["format"], "syntra-brain-checkpoint")
            self.assertEqual(
                read_checkpoint_metadata(checkpoint)["step"],
                saved["step"],
            )

            resumed = train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                save_every_steps=2,
                device_name="cpu",
                seed=123,
                resume=True,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["epoch"], 2)
            self.assertGreater(resumed["step"], first["step"])
            resumed_saved = load_checkpoint(checkpoint)
            uninterrupted_saved = load_checkpoint(uninterrupted_checkpoint)
            self.assertEqual(resumed_saved["step"], uninterrupted_saved["step"])
            for parameter_name, expected_value in uninterrupted_saved["model_state_dict"].items():
                torch.testing.assert_close(
                    resumed_saved["model_state_dict"][parameter_name],
                    expected_value,
                    rtol=0,
                    atol=0,
                    msg=f"resumed model diverged from uninterrupted training at {parameter_name}",
                )

            from brain.generate import generate, load_for_inference

            model, tokenizer, device, payload = load_for_inference(checkpoint, "cpu")
            self.assertEqual(device.type, "cpu")
            self.assertEqual(payload["epoch"], 2)
            result = generate(
                model,
                tokenizer,
                "def add(left, right):",
                max_new_tokens=6,
                temperature=0,
                top_k=1,
                repetition_penalty=1.0,
                seed=99,
            )
            self.assertLessEqual(result.tokens_generated, 6)
            self.assertGreaterEqual(result.tokens_per_second, 0)
            self.assertIn(result.finish_reason, {"length", "eos", "cancelled"})

    def test_resume_rejects_batch_and_optimizer_settings_that_change_cursor_semantics(self):
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-resume-settings-") as temp:
            root = Path(temp)
            dataset = root / "dataset.txt"
            self._dataset(dataset)
            checkpoint = root / "model.pt"
            train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=128,
                learning_rate=3e-4,
                weight_decay=0.1,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                device_name="cpu",
                seed=123,
            )

            incompatible_settings = [
                ({"batch_size": 64}, "batch_size"),
                ({"learning_rate": 1e-4}, "learning_rate"),
                ({"weight_decay": 0.2}, "weight_decay"),
                ({"seed": 456}, "seed"),
                ({"evaluation_batches": 2}, "evaluation_batches"),
                ({"save_every_steps": 3}, "save_every_steps"),
            ]
            for overrides, expected_setting in incompatible_settings:
                with self.subTest(setting=expected_setting):
                    with self.assertRaisesRegex(ValueError, expected_setting):
                        train(
                            [dataset],
                            checkpoint_path=checkpoint,
                            config=self._config(),
                            epochs=1,
                            batch_size=overrides.get("batch_size", 128),
                            learning_rate=overrides.get("learning_rate", 3e-4),
                            weight_decay=overrides.get("weight_decay", 0.1),
                            tokenizer_bytes=1024,
                            evaluation_batches=overrides.get("evaluation_batches", 1),
                            save_every_steps=overrides.get("save_every_steps", 100),
                            device_name="cpu",
                            seed=overrides.get("seed", 123),
                            resume=True,
                        )

    def test_resume_rejects_a_changed_dataset(self):
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-dataset-change-") as temp:
            root = Path(temp)
            dataset = root / "dataset.txt"
            self._dataset(dataset)
            checkpoint = root / "model.pt"
            train(
                [dataset],
                checkpoint_path=checkpoint,
                config=self._config(),
                epochs=1,
                batch_size=128,
                tokenizer_bytes=1024,
                evaluation_batches=1,
                device_name="cpu",
            )
            dataset.write_text("different content\n" * 1000, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the checkpoint"):
                train(
                    [dataset],
                    checkpoint_path=checkpoint,
                    config=self._config(),
                    epochs=1,
                    batch_size=128,
                    tokenizer_bytes=1024,
                    evaluation_batches=1,
                    device_name="cpu",
                    resume=True,
                )

    def test_directory_dataset_discovers_source_and_excludes_generated_and_sensitive_files(self):
        from brain.train import _read_local_datasets

        with tempfile.TemporaryDirectory(prefix="syntra-brain-project-scan-") as temp:
            root = Path(temp) / "project"
            (root / "src").mkdir(parents=True)
            (root / "node_modules" / "package").mkdir(parents=True)
            (root / "dist").mkdir()
            (root / "src" / "main.py").write_text("def main():\\n    return 42\\n", encoding="utf-8")
            (root / "README.md").write_text("# Local project docs\\n", encoding="utf-8")
            (root / ".env").write_text("API_KEY=do-not-train\\n", encoding="utf-8")
            (root / "secrets.py").write_text("PRIVATE_TOKEN = 'do-not-train'\\n", encoding="utf-8")
            (root / "secrets").mkdir()
            (root / "secrets" / "module.py").write_text("PRIVATE_TOKEN = 'do-not-train'\\n", encoding="utf-8")
            (root / "node_modules" / "package" / "ignored.py").write_text("generated dependency\\n", encoding="utf-8")
            (root / "dist" / "ignored.js").write_text("generated build output\\n", encoding="utf-8")
            for sensitive_name in (
                ".npmrc", ".pypirc", ".netrc", ".git-credentials",
                ".dockerconfigjson", "service-account.json", "serviceaccount.json",
            ):
                (root / sensitive_name).write_text("excluded-config-marker", encoding="utf-8")
            for directory, filename in (
                (".ssh", "config"),
                (".aws", "credentials"),
                (".azure", "profile.json"),
                (".kube", "config"),
                (".docker", "config.json"),
                (".gnupg", "gpg.conf"),
                (".terraform", "credentials.tfrc.json"),
            ):
                secret_dir = root / directory
                secret_dir.mkdir()
                (secret_dir / filename).write_text("excluded-directory-marker", encoding="utf-8")

            train_text, validation_text, names, byte_count, fingerprint = _read_local_datasets(
                [root], 1024 * 1024
            )
            combined = train_text + validation_text
            self.assertIn("src/main.py", combined)
            self.assertIn("README.md", combined)
            self.assertNotIn("API_KEY", combined)
            self.assertNotIn("PRIVATE_TOKEN", combined)
            self.assertNotIn("generated dependency", combined)
            self.assertNotIn("generated build output", combined)
            self.assertNotIn("excluded-config-marker", combined)
            self.assertNotIn("excluded-directory-marker", combined)
            self.assertEqual(len(names), 2)
            self.assertGreater(byte_count, 0)
            self.assertEqual(len(fingerprint), 64)

            with self.assertRaisesRegex(ValueError, "sensitive file"):
                _read_local_datasets([root / ".env"], 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "sensitive or generated directory root"):
                _read_local_datasets([root / ".ssh"], 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "sensitive or generated directory root"):
                _read_local_datasets([root / ".kube"], 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "sensitive directory"):
                _read_local_datasets([root / "secrets" / "module.py"], 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "generated or excluded directory"):
                _read_local_datasets([root / "node_modules" / "package" / "ignored.py"], 1024 * 1024)

    def test_raw_repository_validation_split_keeps_each_source_file_wholly_separate(self):
        from brain.train import _read_local_datasets

        with tempfile.TemporaryDirectory(prefix="syntra-source-file-holdout-") as temp:
            root = Path(temp)
            expected_names = set()
            for index in range(7):
                relative = f"src/module_{index}.py"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    f'"""Unique source module {index}."""\n'
                    f"def operation_{index}(value):\n"
                    f"    return value + {index}\n",
                    encoding="utf-8",
                )
                expected_names.add(relative)

            first_train, first_validation, names, _byte_count, fingerprint = (
                _read_local_datasets([root], 1024 * 1024)
            )
            second_train, second_validation, _names, _bytes, second_fingerprint = (
                _read_local_datasets([root], 1024 * 1024)
            )

            def source_names(value: str) -> set[str]:
                return {
                    line.removeprefix("# SYNTRA_SOURCE_FILE: ").strip()
                    for line in value.splitlines()
                    if line.startswith("# SYNTRA_SOURCE_FILE: ")
                }

            training_sources = source_names(first_train)
            validation_sources = source_names(first_validation)
            self.assertTrue(training_sources)
            self.assertTrue(validation_sources)
            self.assertEqual(training_sources.intersection(validation_sources), set())
            self.assertEqual(training_sources.union(validation_sources), expected_names)
            self.assertEqual(len(names), len(expected_names))
            self.assertEqual(len(fingerprint), 64)
            self.assertEqual(first_train, second_train)
            self.assertEqual(first_validation, second_validation)
            self.assertEqual(fingerprint, second_fingerprint)

    def test_training_rejects_invalid_utf8_and_missing_files(self):
        from brain.train import train

        with tempfile.TemporaryDirectory(prefix="syntra-brain-invalid-data-") as temp:
            root = Path(temp)
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"valid text\xffbroken")
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                train([invalid], checkpoint_path=root / "model.pt")
            with self.assertRaisesRegex(ValueError, "Cannot read local dataset"):
                train([root / "missing.txt"], checkpoint_path=root / "model.pt")

    def test_checkpoint_rejects_unknown_format(self):
        from brain.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-brain-bad-checkpoint-") as temp:
            path = Path(temp) / "invalid.pt"
            torch.save({"format": "not-syntra"}, path)
            with self.assertRaisesRegex(ValueError, "not a supported"):
                load_checkpoint(path)


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run generation tests.")
class StreamingGenerationTests(unittest.TestCase):
    def test_streaming_yields_text_fragments_and_reports_token_speed(self):
        from brain.generate import generate, stream_generate
        from brain.model import EOS_ID, LocalTransformer, TransformerConfig

        config = TransformerConfig(
            vocab_size=258,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        model = LocalTransformer(config).eval()
        # Keep EOS out of the greedy winner so the test exercises all requested steps.
        with torch.no_grad():
            model.token_embedding.weight[EOS_ID].fill_(-1000.0)
        tokenizer = BytePairTokenizer.empty()
        events = []
        result = generate(
            model,
            tokenizer,
            "local test",
            max_new_tokens=6,
            temperature=0,
            top_k=1,
            repetition_penalty=1.0,
            seed=7,
            on_token=lambda text, count, speed: events.append((text, count, speed)),
        )
        self.assertEqual(result.tokens_generated, 6)
        self.assertEqual(result.finish_reason, "length")
        self.assertTrue(events)
        self.assertEqual(events[-1][1], 6)
        self.assertGreaterEqual(events[-1][2], 0)

        fragments = list(stream_generate(
            model,
            tokenizer,
            "local test",
            max_new_tokens=6,
            temperature=0,
            top_k=1,
            repetition_penalty=1.0,
            seed=7,
        ))
        self.assertEqual("".join(fragments), result.text)

    def test_generation_handles_a_full_context_window_and_validates_sampling(self):
        from brain.generate import stream_generate
        from brain.model import EOS_ID, LocalTransformer, TransformerConfig

        config = TransformerConfig(
            vocab_size=258,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        model = LocalTransformer(config).eval()
        with torch.no_grad():
            model.token_embedding.weight[EOS_ID].fill_(-1000.0)
        tokenizer = BytePairTokenizer.empty()
        output = "".join(stream_generate(
            model,
            tokenizer,
            "0123456789abcdef0123456789",
            max_new_tokens=4,
            temperature=0,
            top_k=1,
            repetition_penalty=1.0,
        ))
        self.assertIsInstance(output, str)
        with self.assertRaisesRegex(ValueError, "top_p"):
            list(stream_generate(
                model,
                tokenizer,
                "prompt",
                top_p=0,
            ))


if __name__ == "__main__":
    unittest.main()

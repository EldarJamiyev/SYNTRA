import json
import tempfile
import unittest
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

            records, files, byte_count, fingerprint = _read_instruction_jsonl_datasets([root], 1024 * 1024)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["prompt"], "Explain addition.")
            self.assertEqual(records[1]["completion"], "Multiply the two values.")
            self.assertEqual(len(files), 1)
            self.assertGreater(byte_count, 0)
            self.assertEqual(len(fingerprint), 64)

            train_text, validation_text, raw_files, _, _ = _read_local_datasets(
                [root], 1024 * 1024, exclude_paths=set(files)
            )
            self.assertIn("legacy source text", train_text + validation_text)
            self.assertEqual(len(raw_files), 1)

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

            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["training_data_format"], "instruction-jsonl-masked-v1")
            self.assertEqual(saved["instruction_example_count"], 4)
            self.assertGreater(saved["training_context_blocks"], 0)

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
            instructions.write_text("\\n".join(json.dumps(item) for item in records), encoding="utf-8")
            raw = root / "legacy.txt"
            raw.write_text(("Legacy raw corpus remains part of causal language-model training.\\n" * 120), encoding="utf-8")
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
            self.assertIsNotNone(first["train_loss"])
            self.assertIsNotNone(first["validation_loss"])
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(checkpoint.with_suffix(".pt.json").is_file())

            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["epoch"], 1)
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
            (root / "node_modules" / "package" / "ignored.py").write_text("generated dependency\\n", encoding="utf-8")
            (root / "dist" / "ignored.js").write_text("generated build output\\n", encoding="utf-8")

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
            self.assertEqual(len(names), 2)
            self.assertGreater(byte_count, 0)
            self.assertEqual(len(fingerprint), 64)

            with self.assertRaisesRegex(ValueError, "sensitive file"):
                _read_local_datasets([root / ".env"], 1024 * 1024)

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

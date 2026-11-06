import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from syntra_ai.config import ModelConfig
from syntra_ai.tokenizer import ByteBPETokenizer


TORCH_SKIP_REASON = "Install requirements.txt to run neural model tests."


class TransformerTests(unittest.TestCase):
    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_random_initialized_transformer_forward_and_loss(self):
        from syntra_ai.model import SyntraTransformer
        config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        x = torch.randint(0, config.vocab_size - 1, (2, 24))
        y = torch.randint(0, config.vocab_size, (2, 24))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 24, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_cached_incremental_logits_match_full_forward(self):
        from syntra_ai.model import BOS_ID, SyntraTransformer

        config = ModelConfig(vocab_size=258, seq_len=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        model.eval()
        tokens = torch.tensor([[BOS_ID, 100, 101, 102, 103, 104, 105]], dtype=torch.long)
        with torch.no_grad():
            full_logits, _ = model(tokens)
            prefix_length = 3
            _, _, cache = model(tokens[:, :prefix_length], use_cache=True)
            for position in range(prefix_length, tokens.shape[1]):
                incremental_logits, _, cache = model(
                    tokens[:, position:position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                torch.testing.assert_close(
                    incremental_logits[:, -1, :],
                    full_logits[:, position, :],
                    rtol=1e-4,
                    atol=1e-4,
                )
        self.assertEqual(len(cache), config.n_layer)
        self.assertEqual(cache[0][0].shape[-2], tokens.shape[1])
        self.assertEqual(cache[0][1].shape[-2], tokens.shape[1])

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_cached_generation_recomputes_context_at_window_saturation(self):
        from syntra_ai.model import BOS_ID, EOS_ID, SyntraTransformer

        torch.manual_seed(2026)
        config = ModelConfig(vocab_size=258, seq_len=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        with torch.no_grad():
            model.token_embedding.weight[EOS_ID].zero_()
        prompt = torch.tensor([[BOS_ID, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]], dtype=torch.long)
        original_forward = model.forward
        input_lengths = []

        def record_forward(tokens, *args, **kwargs):
            input_lengths.append(tokens.shape[1])
            return original_forward(tokens, *args, **kwargs)

        model.forward = record_forward
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=1)
        self.assertGreaterEqual(generated.shape[1], prompt.shape[1] + 1)
        self.assertLessEqual(generated.shape[1], prompt.shape[1] + 10)
        self.assertEqual(generated[0, :prompt.shape[1]].tolist(), prompt[0].tolist())
        self.assertIn(1, input_lengths, "incremental generation should pass one new token into the cache")
        self.assertIn(config.seq_len, input_lengths, "generation should rebuild the rolling window at context saturation")

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_model_generation_uses_actual_neural_forward(self):
        from syntra_ai.model import BOS_ID, SyntraTransformer
        config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        prompt = torch.tensor([[BOS_ID, 100, 101, 102]], dtype=torch.long)
        generated = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=4)
        self.assertGreaterEqual(generated.shape[1], prompt.shape[1] + 1)
        self.assertLessEqual(generated.shape[1], prompt.shape[1] + 5)
        self.assertEqual(generated[0, :4].tolist(), prompt[0].tolist())

    def test_byte_bpe_tokenizer_roundtrips_and_compresses_repeated_source(self):
        source = ("def calculate_total(item, item_count):\\n    return item * item_count\\n") * 100
        tokenizer = ByteBPETokenizer.train([source], vocab_size=400, max_training_bytes=20_000)
        encoded = tokenizer.encode(source)
        self.assertEqual(tokenizer.decode(encoded), source)
        self.assertGreater(tokenizer.vocab_size, 258)
        self.assertLess(len(encoded), len(source.encode("utf-8")))
        self.assertEqual(ByteBPETokenizer.from_dict(tokenizer.to_dict()), tokenizer)

    def test_config_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            ModelConfig(n_embd=190, n_head=6).validate()
        with self.assertRaises(ValueError):
            ModelConfig(seq_len=16385).validate()
        for flag in ("bias", "use_rope", "use_rmsnorm", "use_swiglu"):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, flag + " must be a boolean"):
                    ModelConfig(**{flag: "false"}).validate()

    def test_resume_architecture_comparison_covers_every_model_field(self):
        from dataclasses import replace
        from syntra_ai.training import _same_architecture

        base = ModelConfig(vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        self.assertTrue(_same_architecture(base, base))
        changes = (
            ("vocab_size", 259),
            ("bias", False),
            ("use_rope", True),
            ("use_rmsnorm", True),
            ("use_swiglu", True),
        )
        for field, value in changes:
            with self.subTest(field=field):
                changed = replace(base, **{field: value})
                self.assertFalse(_same_architecture(changed, base))

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_validation_loss_helpers_restore_the_callers_model_mode(self):
        from syntra_ai.instruction import _instruction_loss
        from syntra_ai.model import BOS_ID, SyntraTransformer
        from syntra_ai.training import _estimate_loss

        config = ModelConfig(vocab_size=258, seq_len=16, n_layer=1, n_head=2, n_embd=32, dropout=0.2)
        model = SyntraTransformer(config)
        data = torch.randint(0, config.vocab_size, (96,), dtype=torch.long)

        model.train()
        train_loss = _estimate_loss(model, data, batch_size=1, seq_len=config.seq_len, device=torch.device("cpu"), batches=1)
        self.assertTrue(torch.isfinite(torch.tensor(train_loss)).item())
        self.assertTrue(model.training, "training-time validation must restore training mode")

        model.eval()
        eval_loss = _estimate_loss(model, data, batch_size=1, seq_len=config.seq_len, device=torch.device("cpu"), batches=1)
        self.assertTrue(torch.isfinite(torch.tensor(eval_loss)).item())
        self.assertFalse(model.training, "standalone evaluation must not enable dropout on the resident model")

        examples = [{
            "input_ids": [BOS_ID, 65, 66, 67],
            "labels": [-100, 65, 66, 67],
        }]
        model.train()
        instruction_train_loss = _instruction_loss(model, examples, torch.device("cpu"), batch_size=1, batches=1)
        self.assertTrue(torch.isfinite(torch.tensor(instruction_train_loss)).item())
        self.assertTrue(model.training, "instruction validation must restore training mode")

        model.eval()
        instruction_eval_loss = _instruction_loss(model, examples, torch.device("cpu"), batch_size=1, batches=1)
        self.assertTrue(torch.isfinite(torch.tensor(instruction_eval_loss)).item())
        self.assertFalse(model.training, "instruction evaluation must preserve inference mode")

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_inference_loader_falls_back_when_companion_is_stale(self):
        from syntra_ai.model import SyntraTransformer
        from syntra_ai.runtime import _read_checkpoint
        from syntra_ai.training import inference_checkpoint_path

        with tempfile.TemporaryDirectory(prefix="syntra-stale-inference-") as temp:
            root = Path(temp)
            checkpoint = root / "fixture.pt"
            config = ModelConfig(vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
            tokenizer = ByteBPETokenizer.empty()
            model = SyntraTransformer(config)
            full_payload = {
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": {"sentinel": torch.tensor([1])},
                "step": 3,
                "training_mode": "causal-source-pretraining",
            }
            torch.save(full_payload, checkpoint)

            companion = inference_checkpoint_path(checkpoint)
            light_payload = {key: full_payload[key] for key in (
                "format", "config", "tokenizer", "model_state", "step", "training_mode"
            )}
            light_payload["source_checkpoint_mtime_ns"] = -1
            torch.save(light_payload, companion)
            source_mtime = checkpoint.stat().st_mtime_ns
            future_mtime = source_mtime + 2_000_000_000
            os.utime(companion, ns=(future_mtime, future_mtime))

            target, saved, loaded_config, loaded_tokenizer = _read_checkpoint(checkpoint)
            self.assertEqual(target, checkpoint.resolve())
            self.assertEqual(loaded_config, config)
            self.assertEqual(loaded_tokenizer, tokenizer)
            self.assertIn("optimizer_state", saved,
                          "a stale inference companion must fall back to the full checkpoint")

            companion.write_bytes(b"not a valid torch checkpoint")
            corrupt_mtime = checkpoint.stat().st_mtime_ns + 4_000_000_000
            os.utime(companion, ns=(corrupt_mtime, corrupt_mtime))
            _, recovered, _, _ = _read_checkpoint(checkpoint)
            self.assertEqual(recovered["step"], 3)
            self.assertIn("optimizer_state", recovered,
                          "a corrupted inference companion must fall back to the full checkpoint")

    def test_checkpoint_identity_changes_when_native_checkpoint_revision_changes(self):
        from syntra_ai.runtime import _checkpoint_identity
        from syntra_ai.training import inference_checkpoint_path

        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-identity-") as temp:
            checkpoint = Path(temp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint-one")
            first = _checkpoint_identity(checkpoint)
            self.assertRegex(first, r"^[a-f0-9]{64}$")

            companion = inference_checkpoint_path(checkpoint)
            companion.write_bytes(b"inference-one")
            future = checkpoint.stat().st_mtime_ns + 2_000_000_000
            os.utime(companion, ns=(future, future))
            second = _checkpoint_identity(checkpoint)
            self.assertNotEqual(first, second)

            companion.write_bytes(b"inference-two-with-different-weights")
            newer = future + 2_000_000_000
            os.utime(companion, ns=(newer, newer))
            third = _checkpoint_identity(checkpoint)
            self.assertNotEqual(second, third)

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_embed_texts_uses_the_explicit_checkpoint_identity(self):
        from syntra_ai.model import SyntraTransformer
        from syntra_ai.runtime import _checkpoint_identity, embed_texts

        with tempfile.TemporaryDirectory(prefix="syntra-explicit-embedding-checkpoint-") as temp:
            root = Path(temp)
            checkpoint = root / "selected-model.pt"
            unavailable_default = root / "default-model-does-not-exist.pt"
            config = ModelConfig(
                vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0
            )
            tokenizer = ByteBPETokenizer.empty()
            model = SyntraTransformer(config)
            torch.save({
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": model.state_dict(),
                "step": 7,
                "training_mode": "causal-source-pretraining",
            }, checkpoint)

            with patch.dict(os.environ, {"SYNTRA_CHECKPOINT": str(unavailable_default)}):
                result = embed_texts(
                    ["explicit checkpoint identity regression"],
                    path=checkpoint,
                    device_name="cpu",
                )

            self.assertEqual(result["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(result["checkpoint_identity"], _checkpoint_identity(checkpoint))
            self.assertEqual(result["trained_steps"], 7)
            self.assertEqual(len(result["embeddings"]), 1)
            self.assertEqual(len(result["embeddings"][0]), config.n_embd)

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_status_does_not_report_a_stale_resident_checkpoint(self):
        from syntra_ai.model import SyntraTransformer
        from syntra_ai.runtime import _get_resident_bundle, clear_resident_cache, status

        with tempfile.TemporaryDirectory(prefix="syntra-resident-checkpoint-status-") as temp:
            checkpoint = Path(temp) / "selected-model.pt"
            config = ModelConfig(
                vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0
            )
            tokenizer = ByteBPETokenizer.empty()
            initial_model = SyntraTransformer(config)
            payload = {
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": initial_model.state_dict(),
                "step": 1,
                "training_mode": "causal-source-pretraining",
            }
            torch.save(payload, checkpoint)

            with patch.dict(os.environ, {"SYNTRA_CHECKPOINT": str(checkpoint)}):
                clear_resident_cache()
                _get_resident_bundle(path=checkpoint, device_name="cpu")
                self.assertTrue(status()["resident_in_memory"])

                replacement_model = SyntraTransformer(config)
                with torch.no_grad():
                    replacement_model.token_embedding.weight[0, 0].add_(0.5)
                payload["model_state"] = replacement_model.state_dict()
                payload["step"] = 2
                previous_mtime = checkpoint.stat().st_mtime_ns
                torch.save(payload, checkpoint)
                newer_mtime = max(checkpoint.stat().st_mtime_ns, previous_mtime + 2_000_000_000)
                os.utime(checkpoint, ns=(newer_mtime, newer_mtime))

                self.assertFalse(
                    status()["resident_in_memory"],
                    "status must not report a cached model as resident after its checkpoint changes",
                )
                _get_resident_bundle(path=checkpoint, device_name="cpu")
                self.assertTrue(status()["resident_in_memory"])
                clear_resident_cache()

    def test_training_memory_estimate_rejects_unsafe_context_settings(self):
        from syntra_ai.resources import estimate_training_memory_bytes, validate_training_memory_budget

        small = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        estimate = estimate_training_memory_bytes(small, batch_size=1)
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, 1024 ** 3)
        self.assertEqual(
            validate_training_memory_budget(small, batch_size=1, device="cpu", available_bytes=2 * 1024 ** 3),
            estimate,
        )

        large = ModelConfig(seq_len=8192, n_layer=12, n_head=12, n_embd=768, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "safety budget"):
            validate_training_memory_budget(large, batch_size=16, device="cpu", available_bytes=8 * 1024 ** 3)

    def test_training_memory_preflight_does_not_assume_memory_when_unavailable(self):
        from syntra_ai.resources import estimate_training_memory_bytes, validate_training_memory_budget

        config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        expected = estimate_training_memory_bytes(config, batch_size=1)
        self.assertEqual(
            validate_training_memory_budget(config, batch_size=1, device="cpu", available_bytes=None),
            expected,
        )

    def test_corpus_loader_rejects_unbounded_resource_requests(self):
        from syntra_ai.data import MAX_CORPUS_FILE_BYTES, MAX_CORPUS_FILES, MAX_CORPUS_TOKENS, load_corpus

        with tempfile.TemporaryDirectory(prefix="syntra-data-limits-") as temp:
            root = Path(temp)
            (root / "safe.py").write_text("def safe(): return 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_tokens must be between"):
                load_corpus([root], max_tokens=MAX_CORPUS_TOKENS + 1)
            with self.assertRaisesRegex(ValueError, "max_files must be between"):
                load_corpus([root], max_files=MAX_CORPUS_FILES + 1)
            with self.assertRaisesRegex(ValueError, "max_file_bytes must be between"):
                load_corpus([root], max_file_bytes=MAX_CORPUS_FILE_BYTES + 1)

    def test_corpus_loader_rejects_unapproved_and_home_shorthand_roots(self):
        from syntra_ai.data import load_corpus

        with tempfile.TemporaryDirectory(prefix="syntra-unapproved-corpus-") as temp:
            root = Path(temp)
            (root / "safe.py").write_text("def safe(): return 1\n", encoding="utf-8")

            with patch.dict(os.environ):
                os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
                with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                    load_corpus([root], max_tokens=1024)
                with self.assertRaisesRegex(ValueError, "must be absolute"):
                    load_corpus([Path("~") / "unapproved"], max_tokens=1024)

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                corpus = load_corpus([root], max_tokens=1024)
                self.assertTrue(any(path.endswith("safe.py") for path in corpus.files))

    def test_corpus_loader_skips_generated_checkpoint_and_run_directories(self):
        from syntra_ai.data import load_corpus

        with tempfile.TemporaryDirectory(prefix="syntra-generated-data-") as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "data" / "checkpoints").mkdir(parents=True)
            (root / "wandb" / "run-1").mkdir(parents=True)
            (root / "src" / "safe.py").write_text("def safe(): return 1\n", encoding="utf-8")
            (root / "data" / "checkpoints" / "syntra.pt.json").write_text('{"checkpoint":"metadata"}', encoding="utf-8")
            (root / "wandb" / "run-1" / "config.yaml").write_text("api_key: should-not-train", encoding="utf-8")

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                corpus = load_corpus([root], max_tokens=1024)
            joined_paths = "\\n".join(corpus.files).lower()
            self.assertTrue(any(path.endswith("safe.py") for path in corpus.files))
            self.assertNotIn("checkpoints", joined_paths)
            self.assertNotIn("wandb", joined_paths)

    def test_instruction_dataset_loader_bounds_examples_records_and_token_memory(self):
        from syntra_ai.instruction import (
            MAX_INSTRUCTION_EXAMPLES,
            MAX_INSTRUCTION_TOKENS,
            load_instruction_examples,
        )

        tokenizer = ByteBPETokenizer.empty()
        with tempfile.TemporaryDirectory(prefix="syntra-instruction-limits-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            rows = [
                {"prompt": "what is add?", "completion": "def add(a, b): return a+b"},
                {"prompt": "what is max?", "completion": "def max2(a, b): return max(a, b)"},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            overflow_dataset = root / "token-overflow.jsonl"
            long_rows = [
                {"prompt": "p" * 6000, "completion": "c" * 6000},
                {"prompt": "p" * 6000, "completion": "c" * 6000},
            ]
            overflow_dataset.write_text("\n".join(json.dumps(row) for row in long_rows), encoding="utf-8")
            huge_field = root / "huge-field.jsonl"
            huge_field.write_text(json.dumps({
                "prompt": "x" * (512 * 1024 + 1),
                "completion": "ok"
            }), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                with self.assertRaisesRegex(ValueError, "max_examples must be between"):
                    load_instruction_examples(dataset, seq_len=32, tokenizer=tokenizer, max_examples=MAX_INSTRUCTION_EXAMPLES + 1)
                with self.assertRaisesRegex(ValueError, "max_tokens must be between"):
                    load_instruction_examples(dataset, seq_len=32, tokenizer=tokenizer, max_tokens=MAX_INSTRUCTION_TOKENS + 1)
                with self.assertRaisesRegex(ValueError, "safety cap"):
                    load_instruction_examples(overflow_dataset, seq_len=16384, tokenizer=tokenizer, max_tokens=1024)
                with self.assertRaisesRegex(ValueError, "512 KiB field limit"):
                    load_instruction_examples(huge_field, seq_len=32, tokenizer=tokenizer)

    def test_instruction_split_is_deterministic_and_keeps_repeated_prompts_together(self):
        from syntra_ai.instruction import load_instruction_examples, split_instruction_examples

        rows = [
            {"prompt": "Describe addition", "completion": "Add two numeric values."},
            {"prompt": "Explain maximum", "completion": "Return the larger value."},
            {"prompt": "  describe   ADDITION  ", "completion": "Add two numeric values."},
            {"prompt": "Explain minimum", "completion": "Return the smaller value."},
        ]
        with tempfile.TemporaryDirectory(prefix="syntra-instruction-split-") as temp:
            dataset = Path(temp) / "instructions.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(Path(temp))}):
                examples = load_instruction_examples(dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty())
            training, validation, quality = split_instruction_examples(examples)
            training_groups = {item["prompt_group_id"] for item in training}
            validation_groups = {item["prompt_group_id"] for item in validation}

            self.assertTrue(training)
            self.assertTrue(validation)
            self.assertFalse(training_groups & validation_groups, "normalized prompts must not cross train/validation")
            self.assertEqual(quality["unique_prompt_count"], 3)
            self.assertEqual(quality["repeated_prompt_example_count"], 1)
            self.assertEqual(quality["exact_duplicate_example_count"], 1)
            self.assertEqual(quality["prompt_leakage_groups"], 0)
            self.assertTrue(any("Low prompt diversity" in warning for warning in quality["warnings"]))
            self.assertTrue(any("exact prompt/completion duplicate" in warning for warning in quality["warnings"]))

            _, repeated_validation, repeated_quality = split_instruction_examples(examples)
            self.assertEqual(
                {item["prompt_group_id"] for item in repeated_validation},
                validation_groups,
                "the same dataset and seed must produce the same held-out prompt groups",
            )
            self.assertEqual(repeated_quality, quality)

    def test_instruction_split_rejects_datasets_with_only_one_unique_prompt(self):
        from syntra_ai.instruction import load_instruction_examples, split_instruction_examples

        rows = [
            {"prompt": "Explain addition", "completion": "Add values."},
            {"prompt": "  EXPLAIN   ADDITION ", "completion": "Sum values."},
        ]
        with tempfile.TemporaryDirectory(prefix="syntra-instruction-one-prompt-") as temp:
            dataset = Path(temp) / "instructions.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(Path(temp))}):
                examples = load_instruction_examples(dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty())
            with self.assertRaisesRegex(ValueError, "two distinct normalized prompts"):
                split_instruction_examples(examples)

    def test_instruction_dataset_audit_reports_quality_and_invalid_validation_splits(self):
        from syntra_ai.instruction import audit_instruction_dataset

        with tempfile.TemporaryDirectory(prefix="syntra-instruction-audit-") as temp:
            root = Path(temp)
            valid_dataset = root / "valid.jsonl"
            rows = [
                {"prompt": "Describe addition", "completion": "Add two numeric values."},
                {"prompt": "Explain maximum", "completion": "Return the larger value."},
                {"prompt": "  describe   ADDITION  ", "completion": "Add two numeric values."},
                {"prompt": "Explain minimum", "completion": "Return the smaller value."},
            ]
            valid_dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                report = audit_instruction_dataset(valid_dataset, seq_len=64)
            self.assertTrue(report["valid"])
            self.assertTrue(report["split_available"])
            self.assertEqual(report["example_count"], 4)
            self.assertEqual(report["unique_prompt_count"], 3)
            self.assertEqual(report["exact_duplicate_example_count"], 1)
            self.assertEqual(report["tokenizer_basis"], "byte-fallback")
            self.assertTrue(report["warnings"])

            one_prompt_dataset = root / "one-prompt.jsonl"
            one_prompt_rows = [
                {"prompt": "Explain addition", "completion": "Add values."},
                {"prompt": "  EXPLAIN   ADDITION ", "completion": "Sum values."},
            ]
            one_prompt_dataset.write_text("\n".join(json.dumps(row) for row in one_prompt_rows), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                one_prompt_report = audit_instruction_dataset(one_prompt_dataset, seq_len=64)
            self.assertTrue(one_prompt_report["valid"])
            self.assertFalse(one_prompt_report["split_available"])
            self.assertEqual(one_prompt_report["unique_prompt_count"], 1)
            self.assertIn("two distinct normalized prompts", one_prompt_report["error"])

            invalid_dataset = root / "invalid.jsonl"
            invalid_dataset.write_text("{not json}\n", encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                invalid_report = audit_instruction_dataset(invalid_dataset, seq_len=64)
            self.assertFalse(invalid_report["valid"])
            self.assertIn("Invalid JSON", invalid_report["error"])

    def test_instruction_audit_cli_returns_json_and_nonzero_for_unsafe_split(self):
        import contextlib
        import io
        from syntra_ai.cli import main

        with tempfile.TemporaryDirectory(prefix="syntra-instruction-cli-") as temp:
            dataset = Path(temp) / "instructions.jsonl"
            rows = [
                {"prompt": "Explain addition", "completion": "Add values."},
                {"prompt": "Explain maximum", "completion": "Return larger value."},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(Path(temp))}):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = main(["audit-instructions", "--dataset", str(dataset), "--seq-len", "64"])
                report = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(report["split_available"])

            one_prompt = Path(temp) / "one-prompt.jsonl"
            one_prompt.write_text(
                json.dumps({"prompt": "same prompt", "completion": "first"}) + "\n" +
                json.dumps({"prompt": " SAME   PROMPT ", "completion": "second"}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(Path(temp))}):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = main(["audit-instructions", "--dataset", str(one_prompt), "--seq-len", "64"])
                report = json.loads(output.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertFalse(report["split_available"])

    def test_checkpoint_file_budget_rejects_oversized_loads(self):
        from syntra_ai.resources import validate_checkpoint_load_budget

        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-budget-") as temp:
            checkpoint = Path(temp) / "fixture.pt"
            checkpoint.write_bytes(b"x" * 1000)
            self.assertEqual(validate_checkpoint_load_budget(checkpoint, available_bytes=5000), 1000)
            with self.assertRaisesRegex(ValueError, "checkpoint-load budget"):
                validate_checkpoint_load_budget(checkpoint, available_bytes=1000)

    def test_inference_memory_budget_scales_with_context_and_rejects_oversized_settings(self):
        from syntra_ai.resources import (
            estimate_inference_memory_bytes,
            validate_inference_memory_budget,
        )

        small = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        short_estimate = estimate_inference_memory_bytes(small, batch_size=1, sequence_length=8)
        long_estimate = estimate_inference_memory_bytes(small, batch_size=1, sequence_length=32)
        self.assertGreater(long_estimate, short_estimate)
        self.assertEqual(
            validate_inference_memory_budget(small, 1, 32, "cpu", available_bytes=2 * 1024 ** 3),
            long_estimate,
        )
        large = ModelConfig(seq_len=8192, n_layer=12, n_head=12, n_embd=768, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "safety budget"):
            validate_inference_memory_budget(large, 1, 8192, "cpu", available_bytes=8 * 1024 ** 3)

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_instruction_finetuning_rejects_oversized_config_before_model_allocation(self):
        from syntra_ai.instruction import finetune

        with tempfile.TemporaryDirectory(prefix="syntra-finetune-memory-") as temp:
            root = Path(temp)
            checkpoint = root / "oversized.pt"
            config = ModelConfig(seq_len=8192, n_layer=12, n_head=12, n_embd=768, dropout=0.0)
            torch.save({
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": ByteBPETokenizer.empty().to_dict(),
                "model_state": {},
                "step": 0,
            }, checkpoint)

            with self.assertRaisesRegex(ValueError, "safety budget"):
                finetune(
                    dataset_path=root / "unused.jsonl",
                    steps=1,
                    batch_size=16,
                    device_name="cpu",
                    checkpoint=checkpoint,
                )

    def test_data_loader_skips_credentials_directories(self):
        from syntra_ai.data import load_corpus

        with tempfile.TemporaryDirectory(prefix="syntra-data-safety-") as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "secrets").mkdir()
            (root / "credentials").mkdir()
            (root / "src" / "safe.py").write_text("def safe(): return 1\n", encoding="utf-8")
            (root / "secrets" / "private.txt").write_text("API_TOKEN=should-not-train-on-this", encoding="utf-8")
            (root / "credentials" / "credentials.json").write_text('{"password":"do-not-train"}', encoding="utf-8")
            (root / ".env").write_text("API_KEY=do-not-train", encoding="utf-8")

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                corpus = load_corpus([root], max_tokens=1024)
            joined_paths = "\\n".join(corpus.files).lower()
            self.assertTrue(any(path.endswith("safe.py") for path in corpus.files))
            self.assertNotIn("secrets", joined_paths)
            self.assertNotIn("credentials", joined_paths)
            self.assertNotIn(".env", joined_paths)
            self.assertNotIn(b"API_TOKEN", bytes(token for token in corpus.token_ids if token < 256))

    def test_unified_brain_checkpoint_normalization_and_modern_config(self):
        from syntra_ai.runtime import _normalize_checkpoint_payload

        config = ModelConfig.from_dict({
            "vocab_size": 300,
            "block_size": 64,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "dropout": 0.0,
            "use_rope": True,
            "use_rmsnorm": True,
            "use_swiglu": True,
        })
        self.assertEqual(config.seq_len, 64)
        self.assertEqual(config.block_size, 64)
        self.assertTrue(config.use_rope)
        self.assertTrue(config.use_rmsnorm)
        self.assertTrue(config.use_swiglu)

        brain_payload = {
            "format": "syntra-brain-checkpoint",
            "version": 1,
            "step": 12,
            "config": {"vocab_size": 258, "block_size": 32, "n_layer": 1, "n_head": 2, "n_embd": 32, "dropout": 0.0},
            "tokenizer": {"format": "syntra-local-byte-bpe-v1", "merges": []},
            "model_state_dict": {
                "token_embedding.weight": "w_tok",
                "blocks.0.norm_attention.weight": "w_ln1",
                "blocks.0.attention.qkv.weight": "w_qkv",
                "blocks.0.attention.projection.weight": "w_proj",
                "blocks.0.norm_feed_forward.weight": "w_ln2",
                "blocks.0.feed_forward.layers.0.weight": "w_mlp0",
                "blocks.0.feed_forward.layers.2.weight": "w_mlp2",
                "language_model_head.weight": "w_head",
            },
        }
        normalized = _normalize_checkpoint_payload(brain_payload)
        self.assertEqual(normalized["format"], "syntra-transformer-v1")
        self.assertEqual(normalized["source_format"], "syntra-brain-checkpoint")
        self.assertEqual(normalized["config"]["seq_len"], 32)
        self.assertEqual(normalized["tokenizer"]["format"], "syntra-byte-bpe-v1")
        self.assertIn("blocks.0.ln_1.weight", normalized["model_state"])
        self.assertIn("blocks.0.attn.qkv.weight", normalized["model_state"])
        self.assertIn("blocks.0.attn.proj.weight", normalized["model_state"])
        self.assertIn("blocks.0.ln_2.weight", normalized["model_state"])
        self.assertIn("blocks.0.mlp.0.weight", normalized["model_state"])
        self.assertIn("blocks.0.mlp.2.weight", normalized["model_state"])
        self.assertIn("lm_head.weight", normalized["model_state"])

    def test_synthesize_repository_curriculum_creates_audited_jsonl(self):
        from syntra_ai.instruction import synthesize_repository_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-curriculum-") as temp:
            root = Path(temp)
            (root / "math_utils.py").write_text(
                "def clamp_value(value, minimum, maximum):\n"
                "    if value < minimum:\n"
                "        return minimum\n"
                "    if value > maximum:\n"
                "        return maximum\n"
                "    return value\n\n"
                "def normalize_scores(items):\n"
                "    total = sum(items) or 1.0\n"
                "    return [round(item / total, 4) for item in items]\n",
                encoding="utf-8",
            )
            out_file = root / "curriculum.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_repository_curriculum(root=root, output_path=out_file, max_examples=20)
            self.assertTrue(out_file.is_file())
            self.assertGreaterEqual(result["example_count"], 4)
            self.assertTrue(result["audit"]["valid"])
            self.assertTrue(result["audit"]["split_available"])
            self.assertEqual(result["audit"]["prompt_leakage_groups"], 0)

    def test_ast_curriculum_emits_grounded_module_symbol_and_patch_records(self):
        from syntra_ai.curriculum import synthesize_repository_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-ast-curriculum-") as temp:
            root = Path(temp)
            source = root / "sentinels.py"
            source.write_text(
                '"""Utilities for sentinel checks and bounded arithmetic."""\n'
                "from typing import Optional\n"
                "__all__ = ['is_missing', 'clamp_value']\n\n"
                "def is_missing(value):\n"
                "    return value == None\n\n"
                "def clamp_value(value, lower, upper):\n"
                "    return min(max(value, lower), upper)\n",
                encoding="utf-8",
            )
            output = root / "curriculum.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_repository_curriculum(
                    root=root,
                    output_path=output,
                    max_examples=40,
                )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            kinds = {record["kind"] for record in records}
            self.assertIn("module_comprehension", kinds)
            self.assertIn("symbol_implementation", kinds)
            self.assertIn("anchor_patch_repair", kinds)
            module_record = next(item for item in records if item["kind"] == "module_comprehension")
            self.assertIn("Utilities for sentinel checks", module_record["completion"])
            self.assertIn("from typing import Optional", module_record["completion"])
            implementation = next(
                item for item in records
                if item["kind"] == "symbol_implementation" and "is_missing" in item["prompt"]
            )
            self.assertIn("return value == None", implementation["completion"])
            patch_record = next(item for item in records if item["kind"] == "anchor_patch_repair")
            patch_payload = json.loads(patch_record["completion"])
            self.assertEqual(patch_payload["edits"][0]["path"], "sentinels.py")
            self.assertEqual(patch_payload["edits"][0]["find"], "value == None")
            self.assertEqual(patch_payload["edits"][0]["replace"], "value is None")
            self.assertTrue(result["audit"]["valid"])
            self.assertGreaterEqual(result["example_count"], 4)

    def test_module_exports_exclude_class_methods_and_include_module_values(self):
        from syntra_ai.curriculum import _module_metadata

        source = (
            '"""Module documentation."""\n'
            "DEFAULT_LIMIT = 128\n\n"
            "class Calculator:\n"
            "    def add(self, left, right):\n"
            "        return left + right\n\n"
            "def normalize(value):\n"
            "    return value.strip()\n"
        )
        metadata = _module_metadata("calculator.py", source)
        self.assertIn("Calculator", metadata["exports"])
        self.assertIn("normalize", metadata["exports"])
        self.assertIn("DEFAULT_LIMIT", metadata["exports"])
        self.assertNotIn("Calculator.add", metadata["exports"])
        self.assertIn("Calculator.add", metadata["declarations"])

    def test_curriculum_skips_truncated_symbol_implementations(self):
        from syntra_ai.curriculum import synthesize_repository_curriculum, _module_metadata

        with tempfile.TemporaryDirectory(prefix="syntra-curriculum-large-symbol-") as temp:
            root = Path(temp)
            source = root / "large.py"
            long_literal = "x" * 12_500
            source.write_text(
                "def enormous_payload():\\n"
                f"    payload = {long_literal!r}\\n"
                "    return len(payload)\\n\\n"
                "def useful_symbol(value):\\n"
                "    return value + 1\\n",
                encoding="utf-8",
            )
            metadata = _module_metadata("large.py", source.read_text(encoding="utf-8"))
            enormous = next(item for item in metadata["symbols"] if item["name"] == "enormous_payload")
            self.assertTrue(enormous["truncated"])
            self.assertEqual(enormous["body"], "")

            output = root / "curriculum.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                synthesize_repository_curriculum(
                    root=root,
                    output_path=output,
                    max_examples=40,
                )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            implementation_prompts = [
                item["prompt"] for item in records
                if item["kind"] == "symbol_implementation"
            ]
            self.assertTrue(any("useful_symbol" in prompt for prompt in implementation_prompts))
            self.assertFalse(any("enormous_payload" in prompt for prompt in implementation_prompts))

    def test_curriculum_scans_all_configured_approved_roots(self):
        from syntra_ai import curriculum, safe_paths

        with tempfile.TemporaryDirectory(prefix="syntra-curriculum-roots-") as temp:
            root = Path(temp)
            project = root / "project"
            external = root / "external"
            project.mkdir()
            external.mkdir()
            (project / "inside.py").write_text(
                '"""Project-root module."""\n'
                "def inside(value):\n    return value + 1\n",
                encoding="utf-8",
            )
            (external / "outside.py").write_text(
                '"""Approved external module."""\n'
                "def outside(value):\n    return value * 2\n",
                encoding="utf-8",
            )

            with patch.object(curriculum, "PROJECT_ROOT", project):
                with patch.object(safe_paths, "PROJECT_ROOT", project):
                    with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(external)}):
                        result = curriculum.synthesize_repository_curriculum(max_examples=20)

            self.assertEqual(result["workspace_roots"], [str(project), str(external)])
            self.assertGreaterEqual(result["source_files"], 2)
            self.assertTrue(Path(result["dataset_path"]).is_file())
            records = [
                json.loads(line)
                for line in Path(result["dataset_path"]).read_text(encoding="utf-8").splitlines()
            ]
            sources = {item["source_path"] for item in records}
            self.assertIn("inside.py", sources)
            self.assertIn("outside.py", sources)

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_real_training_checkpoint_and_supervised_prompt_masking(self):
        from syntra_ai.instruction import finetune, load_instruction_examples
        from syntra_ai.runtime import load_model
        from syntra_ai.training import train
        from syntra_ai.model import BOS_ID

        with tempfile.TemporaryDirectory(prefix="syntra-model-test-") as temp:
            root = Path(temp)
            source = root / "sample.py"
            source.write_text(
                ("def add(a, b):\n    return a + b\n\n"
                 "def clamp(value, low, high):\n    return min(max(value, low), high)\n") * 80,
                encoding="utf-8",
            )
            checkpoint = root / "checkpoints" / "model.pt"
            config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = train([str(root)], checkpoint, steps=1, batch_size=1, config=config,
                               max_tokens=10000, device_name="cpu")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(checkpoint.with_suffix(".pt.json").is_file())
            from syntra_ai.training import inference_checkpoint_path, select_inference_checkpoint
            inference_checkpoint = inference_checkpoint_path(checkpoint)
            self.assertTrue(inference_checkpoint.is_file())
            self.assertEqual(select_inference_checkpoint(checkpoint), inference_checkpoint)
            full_saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            inference_saved = torch.load(inference_checkpoint, map_location="cpu", weights_only=True)
            self.assertIn("optimizer_state", full_saved)
            self.assertNotIn("optimizer_state", inference_saved)
            self.assertLess(inference_checkpoint.stat().st_size, checkpoint.stat().st_size)

            dataset = root / "instructions.jsonl"
            records = [
                {"prompt": "Write add function", "completion": "def add(a,b): return a+b"},
                {"prompt": "Write max function", "completion": "def max2(a,b): return max(a,b)"},
                {"prompt": "Write min function", "completion": "def min2(a,b): return min(a,b)"},
            ]
            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            tokenizer = ByteBPETokenizer.from_dict(checkpoint_state["tokenizer"])
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                examples = load_instruction_examples(dataset, seq_len=32, tokenizer=tokenizer)
            first = examples[0]
            prompt_tokens = first["prompt_tokens"]
            self.assertTrue(all(token == -100 for token in first["labels"][:prompt_tokens]))
            self.assertTrue(any(token != -100 for token in first["labels"][prompt_tokens:]))

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                tuned = finetune(dataset, steps=1, batch_size=1, learning_rate=1e-4,
                                 device_name="cpu", checkpoint=checkpoint)
            self.assertEqual(tuned["status"], "completed")
            inference_saved = torch.load(inference_checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(inference_saved["step"], tuned["step"])
            self.assertNotIn("optimizer_state", inference_saved)
            from syntra_ai.runtime import evaluate
            previous_allowed_roots = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(root)
            try:
                fresh_evaluation = evaluate(str(root), batches=2, device_name="cpu", max_tokens=10000, checkpoint=checkpoint)
            finally:
                if previous_allowed_roots is None:
                    os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
                else:
                    os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = previous_allowed_roots
            self.assertTrue(fresh_evaluation["evaluation_performed"])
            self.assertTrue(torch.isfinite(torch.tensor(fresh_evaluation["validation_loss"])).item())
            model, saved, device = load_model(checkpoint, "cpu")
            self.assertFalse(model.training, "fresh evaluation must leave the resident inference model in eval mode")
            self.assertEqual(saved["training_mode"], "supervised-instruction-finetuning")
            self.assertEqual(saved["step"], 2)
            self.assertNotIn("optimizer_state", saved)
            context = torch.tensor([[BOS_ID, *b"def add"]], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=2, temperature=1.0, top_k=2)
            self.assertGreaterEqual(generated.shape[1], context.shape[1] + 1)
            self.assertLessEqual(generated.shape[1], context.shape[1] + 2)


if __name__ == "__main__":
    unittest.main()

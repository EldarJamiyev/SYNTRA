import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from syntra_ai.config import ModelConfig
from syntra_ai.tokenizer import ByteBPETokenizer


@unittest.skipIf(torch is None, "Install requirements.txt to run neural model tests.")
class TransformerTests(unittest.TestCase):
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

            corpus = load_corpus([root], max_tokens=1024)
            joined_paths = "\\n".join(corpus.files).lower()
            self.assertTrue(any(path.endswith("safe.py") for path in corpus.files))
            self.assertNotIn("checkpoints", joined_paths)
            self.assertNotIn("wandb", joined_paths)

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

            corpus = load_corpus([root], max_tokens=1024)
            joined_paths = "\\n".join(corpus.files).lower()
            self.assertTrue(any(path.endswith("safe.py") for path in corpus.files))
            self.assertNotIn("secrets", joined_paths)
            self.assertNotIn("credentials", joined_paths)
            self.assertNotIn(".env", joined_paths)
            self.assertNotIn(b"API_TOKEN", bytes(token for token in corpus.token_ids if token < 256))

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
            result = train([str(root)], checkpoint, steps=1, batch_size=1, config=config,
                           max_tokens=10000, device_name="cpu")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(checkpoint.with_suffix(".pt.json").is_file())

            dataset = root / "instructions.jsonl"
            records = [
                {"prompt": "Write add function", "completion": "def add(a,b): return a+b"},
                {"prompt": "Write max function", "completion": "def max2(a,b): return max(a,b)"},
                {"prompt": "Write min function", "completion": "def min2(a,b): return min(a,b)"},
            ]
            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            tokenizer = ByteBPETokenizer.from_dict(checkpoint_state["tokenizer"])
            examples = load_instruction_examples(dataset, seq_len=32, tokenizer=tokenizer)
            first = examples[0]
            prompt_tokens = first["prompt_tokens"]
            self.assertTrue(all(token == -100 for token in first["labels"][:prompt_tokens]))
            self.assertTrue(any(token != -100 for token in first["labels"][prompt_tokens:]))

            tuned = finetune(dataset, steps=1, batch_size=1, learning_rate=1e-4,
                             device_name="cpu", checkpoint=checkpoint)
            self.assertEqual(tuned["status"], "completed")
            from syntra_ai.runtime import evaluate
            fresh_evaluation = evaluate(str(root), batches=2, device_name="cpu", max_tokens=10000, checkpoint=checkpoint)
            self.assertTrue(fresh_evaluation["evaluation_performed"])
            self.assertTrue(torch.isfinite(torch.tensor(fresh_evaluation["validation_loss"])).item())
            model, saved, device = load_model(checkpoint, "cpu")
            self.assertEqual(saved["training_mode"], "supervised-instruction-finetuning")
            self.assertEqual(saved["step"], 2)
            context = torch.tensor([[BOS_ID, *b"def add"]], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=2, temperature=1.0, top_k=2)
            self.assertGreaterEqual(generated.shape[1], context.shape[1] + 1)
            self.assertLessEqual(generated.shape[1], context.shape[1] + 2)


if __name__ == "__main__":
    unittest.main()

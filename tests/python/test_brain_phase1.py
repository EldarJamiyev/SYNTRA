import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from brain.tokenizer import BOS_ID, EOS_ID, BytePairTokenizer


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run brain tests.")
class LocalTransformerTests(unittest.TestCase):
    def setUp(self):
        from brain.model import LocalTransformer, TransformerConfig

        self.config = TransformerConfig(
            vocab_size=300,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        self.model = LocalTransformer(self.config)

    def test_model_initializes_locally_and_runs_forward_backward(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 12), dtype=torch.long)
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, self.config.vocab_size))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(self.model.token_embedding.weight.grad)
        self.assertEqual(
            self.model.language_model_head.weight.data_ptr(),
            self.model.token_embedding.weight.data_ptr(),
        )
        self.assertGreater(self.model.parameter_count, 0)

    def test_attention_is_causal(self):
        self.model.eval()
        first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        changed_future = torch.tensor([[1, 2, 99, 100]], dtype=torch.long)
        with torch.no_grad():
            first_logits, _ = self.model(first)
            changed_logits, _ = self.model(changed_future)
        torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])

    def test_model_rejects_invalid_shapes_dtypes_and_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(torch.zeros(3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "torch.long"):
            self.model(torch.zeros((1, 4), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "block_size"):
            self.model(torch.zeros((1, 33), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "outside the configured vocabulary"):
            self.model(torch.tensor([[0, 300]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.model(torch.tensor([[1, 2]], dtype=torch.long), torch.tensor([[1]], dtype=torch.long))

    def test_config_roundtrip_and_validation(self):
        from brain.model import TransformerConfig

        restored = TransformerConfig.from_dict(self.config.to_dict())
        self.assertEqual(restored, self.config)
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerConfig(n_embd=32, n_head=3).validate()
        with self.assertRaisesRegex(ValueError, "block_size"):
            TransformerConfig(block_size=0).validate()

    def test_device_selection(self):
        from brain.model import select_device

        selected = select_device("auto")
        self.assertIn(selected.type, {"cpu", "cuda", "mps"})
        self.assertEqual(select_device("cpu").type, "cpu")
        with self.assertRaisesRegex(ValueError, "Device must"):
            select_device("ollama")


class LocalBytePairTokenizerTests(unittest.TestCase):
    def test_trains_only_from_local_file_and_roundtrips_unicode(self):
        text = (
            "def calculate_total(item, count):\n"
            "    return item * count  # local corpus\n"
            "Привет, мир! Καλημέρα κόσμε. こんにちは。\n"
        ) * 30
        with tempfile.TemporaryDirectory(prefix="syntra-brain-tokenizer-") as temp:
            dataset = Path(temp) / "dataset.txt"
            dataset.write_text(text, encoding="utf-8")
            tokenizer = BytePairTokenizer.train_from_file(
                dataset,
                vocab_size=400,
                max_training_bytes=20_000,
            )
            encoded = tokenizer.encode(text)
            self.assertEqual(tokenizer.decode(encoded), text)
            self.assertGreater(tokenizer.vocab_size, 258)
            self.assertLessEqual(tokenizer.vocab_size, 400)

    def test_bos_eos_and_unseen_text_are_lossless(self):
        tokenizer = BytePairTokenizer.empty()
        text = "A🧠e\u0301中文"
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(encoded[0], BOS_ID)
        self.assertEqual(encoded[-1], EOS_ID)
        self.assertEqual(tokenizer.decode(encoded), text)
        with self.assertRaisesRegex(ValueError, "Special tokens"):
            tokenizer.decode(encoded, skip_special_tokens=False)

    def test_tokenizer_save_load_roundtrip(self):
        text = "repeat repeat repeat\n" * 100
        with tempfile.TemporaryDirectory(prefix="syntra-brain-tokenizer-save-") as temp:
            root = Path(temp)
            dataset = root / "dataset.txt"
            dataset.write_text(text, encoding="utf-8")
            tokenizer = BytePairTokenizer.train_from_file(dataset, vocab_size=320)
            saved = tokenizer.save(root / "tokenizer" / "vocab.json")
            loaded = BytePairTokenizer.load(saved)
            self.assertEqual(loaded, tokenizer)
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_training_rejects_missing_empty_and_invalid_utf8_files(self):
        with tempfile.TemporaryDirectory(prefix="syntra-brain-tokenizer-invalid-") as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Cannot read local tokenizer dataset"):
                BytePairTokenizer.train_from_file(root / "missing.txt")
            empty = root / "empty.txt"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                BytePairTokenizer.train_from_file(empty)
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"valid-prefix\xffinvalid")
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                BytePairTokenizer.train_from_file(invalid)

    def test_tokenizer_rejects_malformed_merges_and_ids(self):
        with self.assertRaisesRegex(ValueError, "Malformed tokenizer merge"):
            BytePairTokenizer.from_dict({
                "format": "syntra-local-byte-bpe-v1",
                "merges": [[1]],
            })
        with self.assertRaisesRegex(ValueError, "outside this tokenizer"):
            BytePairTokenizer.empty().decode([999])

    def test_training_limits_are_enforced(self):
        with tempfile.TemporaryDirectory(prefix="syntra-brain-tokenizer-limits-") as temp:
            dataset = Path(temp) / "dataset.txt"
            dataset.write_text("local corpus " * 200, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_training_bytes"):
                BytePairTokenizer.train_from_file(dataset, max_training_bytes=1)
            with self.assertRaisesRegex(ValueError, "vocab_size"):
                BytePairTokenizer.train_from_file(dataset, vocab_size=9000)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from syntra_ai.config import ModelConfig
from syntra_ai.tokenizer import ByteBPETokenizer


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run native embedding boundary tests.")
class NativeEmbeddingContextBoundaryTests(unittest.TestCase):
    @staticmethod
    def _checkpoint(path: Path, *, seq_len: int = 32) -> tuple[Path, ModelConfig, ByteBPETokenizer]:
        from syntra_ai.model import SyntraTransformer

        config = ModelConfig(
            vocab_size=258,
            seq_len=seq_len,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        tokenizer = ByteBPETokenizer.empty()
        model = SyntraTransformer(config)
        torch.save(
            {
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state": model.state_dict(),
                "step": 7,
                "training_mode": "causal-source-pretraining",
            },
            path,
        )
        return path, config, tokenizer

    def test_batched_token_count_uses_the_selected_checkpoint_tokenizer(self):
        from syntra_ai.runtime import tokenize_texts

        with tempfile.TemporaryDirectory(prefix="syntra-embedding-token-count-") as temp:
            checkpoint, config, tokenizer = self._checkpoint(Path(temp) / "model.pt")
            inputs = ["a" * 31, "café", "network gateway resolution"]
            result = tokenize_texts(inputs, path=checkpoint, max_tokens=256)

            self.assertEqual(result["token_counts"], [len(tokenizer.encode(item)) for item in inputs])
            self.assertEqual(result["max_input_tokens"], config.seq_len - 1)
            self.assertEqual(result["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(len(result["checkpoint_identity"]), 64)

    def test_embedding_rejects_over_context_text_and_accepts_the_exact_boundary(self):
        from syntra_ai.runtime import embed_texts

        with tempfile.TemporaryDirectory(prefix="syntra-embedding-no-truncation-") as temp:
            checkpoint, config, _ = self._checkpoint(Path(temp) / "model.pt")
            limit = config.seq_len - 1
            with self.assertRaisesRegex(
                ValueError,
                rf"Text at index 0 requires {limit + 1} tokenizer tokens.*input was not truncated",
            ):
                embed_texts(["x" * (limit + 1)], path=checkpoint, device_name="cpu")

            result = embed_texts(["x" * limit], path=checkpoint, device_name="cpu")
            self.assertEqual(result["input_token_counts"], [limit])
            self.assertEqual(result["max_input_tokens"], limit)
            self.assertEqual(len(result["embeddings"]), 1)
            self.assertEqual(len(result["embeddings"][0]), config.n_embd)

    def test_requested_embedding_limit_is_enforced_even_when_model_context_is_larger(self):
        from syntra_ai.runtime import embed_texts, tokenize_texts

        with tempfile.TemporaryDirectory(prefix="syntra-embedding-requested-limit-") as temp:
            checkpoint, _, tokenizer = self._checkpoint(Path(temp) / "model.pt", seq_len=64)
            inputs = ["a" * 12, "b" * 13]
            counts = tokenize_texts(inputs, path=checkpoint, max_tokens=12)
            self.assertEqual(counts["max_input_tokens"], 12)
            self.assertEqual(counts["token_counts"], [12, 13])

            with self.assertRaisesRegex(
                ValueError,
                "requires 13 tokenizer tokens, exceeding the embedding limit of 12",
            ):
                embed_texts(["b" * 13], path=checkpoint, device_name="cpu", max_tokens=12)

            result = embed_texts(["a" * 12], path=checkpoint, device_name="cpu", max_tokens=12)
            self.assertEqual(result["input_token_counts"], [12])
            self.assertEqual(result["max_input_tokens"], 12)
            self.assertEqual(len(tokenizer.encode("a" * 12)), 12)

    def test_low_level_model_embedding_rejects_oversized_tensors_instead_of_slicing(self):
        from syntra_ai.model import SyntraTransformer

        config = ModelConfig(
            vocab_size=258,
            seq_len=32,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        model = SyntraTransformer(config)
        oversized = torch.ones((1, config.seq_len + 1), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "input was not truncated"):
            model.embed_tokens(oversized)

    def test_invalid_requested_token_limits_are_rejected(self):
        from syntra_ai.runtime import embed_texts, tokenize_texts

        with tempfile.TemporaryDirectory(prefix="syntra-embedding-invalid-limit-") as temp:
            checkpoint, _, _ = self._checkpoint(Path(temp) / "model.pt")
            for invalid in (0, -1, True, 1.5, "32", 4097):
                with self.subTest(max_tokens=invalid):
                    with self.assertRaisesRegex(ValueError, "max_length must be an integer"):
                        tokenize_texts(["short"], path=checkpoint, max_tokens=invalid)
                    with self.assertRaisesRegex(ValueError, "max_length must be an integer"):
                        embed_texts(["short"], path=checkpoint, device_name="cpu", max_tokens=invalid)


if __name__ == "__main__":
    unittest.main()

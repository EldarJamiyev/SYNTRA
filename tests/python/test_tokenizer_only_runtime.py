from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install pinned PyTorch requirements to test native tokenizer loading.")
class TokenizerOnlyRuntimeTests(unittest.TestCase):
    def setUp(self):
        from syntra_ai.runtime import clear_resident_cache
        clear_resident_cache()

    @staticmethod
    def _tokenizer():
        from syntra_ai.tokenizer import ByteBPETokenizer
        return ByteBPETokenizer(((ord("f"), ord("u")),))

    def _write_checkpoint(self, path: Path, *, step: int, sidecar: bool):
        from syntra_ai.config import ModelConfig
        from syntra_ai.runtime import _file_stat_identity_record
        from syntra_ai.training import _atomic_checkpoint, _atomic_json

        tokenizer = self._tokenizer()
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            seq_len=32,
            n_layer=1,
            n_head=1,
            n_embd=32,
            dropout=0.0,
        )
        payload = {
            "format": "syntra-transformer-v1",
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state": {},
            "step": step,
        }
        _atomic_checkpoint(path, payload)
        if sidecar:
            _atomic_json(
                path.with_suffix(path.suffix + ".json"),
                {
                    "checkpoint_identity": _file_stat_identity_record(path),
                    "config": config.to_dict(),
                    "tokenizer_data": tokenizer.to_dict(),
                    "step": step,
                },
            )
        return tokenizer, config

    def test_identity_bound_metadata_tokenizes_without_deserializing_weights_or_loading_model(self):
        from syntra_ai.runtime import (
            _resident_cache_matches_checkpoint,
            tokenize_text,
        )

        with tempfile.TemporaryDirectory(prefix="syntra-tokenizer-metadata-") as temp:
            checkpoint = Path(temp) / "model.pt"
            tokenizer, config = self._write_checkpoint(checkpoint, step=7, sidecar=True)
            source = "function add(a, b) { return a + b; }"

            with (
                patch("syntra_ai.runtime._read_checkpoint", side_effect=AssertionError("weights must not be deserialized")),
                patch("syntra_ai.runtime._get_resident_bundle", side_effect=AssertionError("a resident model must not be loaded")),
            ):
                result = tokenize_text(source, path=checkpoint)

            self.assertEqual(result["token_count"], len(tokenizer.encode(source)))
            self.assertEqual(result["context_window_tokens"], config.seq_len)
            self.assertEqual(result["vocabulary_size"], tokenizer.vocab_size)
            self.assertEqual(result["trained_steps"], 7)
            self.assertFalse(_resident_cache_matches_checkpoint(checkpoint))

    def test_stale_metadata_falls_back_to_checkpoint_and_invalidates_tokenizer_cache(self):
        from syntra_ai.runtime import _read_checkpoint, tokenize_text
        from syntra_ai.training import _atomic_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-tokenizer-stale-metadata-") as temp:
            checkpoint = Path(temp) / "model.pt"
            tokenizer, _config = self._write_checkpoint(checkpoint, step=3, sidecar=True)
            source = "function add(a, b) { return a + b; }"

            with patch("syntra_ai.runtime._read_checkpoint", wraps=_read_checkpoint) as loader:
                first = tokenize_text(source, path=checkpoint)
                self.assertEqual(first["trained_steps"], 3)
                self.assertEqual(loader.call_count, 0, "matching metadata should avoid reading tensor weights")

                # Atomically advance the checkpoint but deliberately retain its old
                # metadata. The identity mismatch must force the checkpoint fallback.
                next_config = _config = __import__("syntra_ai.config", fromlist=["ModelConfig"]).ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    seq_len=48,
                    n_layer=1,
                    n_head=1,
                    n_embd=32,
                    dropout=0.0,
                )
                _atomic_checkpoint(
                    checkpoint,
                    {
                        "format": "syntra-transformer-v1",
                        "config": next_config.to_dict(),
                        "tokenizer": tokenizer.to_dict(),
                        "model_state": {},
                        "step": 9,
                    },
                )
                second = tokenize_text(source, path=checkpoint)
                third = tokenize_text(source, path=checkpoint)

            self.assertEqual(second["trained_steps"], 9)
            self.assertEqual(second["context_window_tokens"], 48)
            self.assertEqual(third, second)
            self.assertEqual(loader.call_count, 1, "stale sidecar fallback must be cached once")

    def test_concurrent_legacy_tokenization_shares_one_checkpoint_deserialization(self):
        from syntra_ai.runtime import _read_checkpoint, tokenize_text

        with tempfile.TemporaryDirectory(prefix="syntra-tokenizer-single-flight-") as temp:
            checkpoint = Path(temp) / "legacy-model.pt"
            self._write_checkpoint(checkpoint, step=11, sidecar=False)
            source = "same local tokenizer result"

            def slow_load(*args, **kwargs):
                time.sleep(0.05)
                return _read_checkpoint(*args, **kwargs)

            start = threading.Barrier(4)

            def run_request():
                start.wait(timeout=3)
                return tokenize_text(source, path=checkpoint)

            with patch("syntra_ai.runtime._read_checkpoint", side_effect=slow_load) as loader:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _index: run_request(), range(4)))

            self.assertTrue(all(item == results[0] for item in results))
            self.assertEqual(results[0]["trained_steps"], 11)
            self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install requirements.txt to test native inference.")
class StreamingInferenceGradModeTests(unittest.TestCase):
    def test_initial_and_incremental_forward_passes_disable_autograd(self):
        from syntra_ai.config import ModelConfig
        from syntra_ai.runtime import stream_generate
        from syntra_ai.tokenizer import ByteBPETokenizer, EOS_ID

        class RecordingModel:
            def __init__(self):
                self.grad_enabled: list[bool] = []
                self.calls = 0

            def parameters(self):
                return ()

            def __call__(self, tokens, *, use_cache, past_key_values=None):
                self.grad_enabled.append(torch.is_grad_enabled())
                batch_size, sequence_length = tokens.shape
                logits = torch.full((batch_size, sequence_length, 258), -20.0)
                next_id = 97 if self.calls == 0 else EOS_ID
                logits[:, :, next_id] = 20.0
                self.calls += 1

                cached_length = sequence_length
                if past_key_values is not None:
                    cached_length += past_key_values[0][0].shape[-2]
                cache = torch.zeros((batch_size, 1, cached_length, 1))
                return logits, None, ((cache, cache),)

        model = RecordingModel()
        config = ModelConfig(
            vocab_size=258,
            seq_len=32,
            n_layer=1,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        tokenizer = ByteBPETokenizer.empty()
        bundle = (
            Path("stream-test-checkpoint.pt"),
            model,
            {"step": 0},
            config,
            tokenizer,
            torch.device("cpu"),
        )

        with patch("syntra_ai.runtime._get_resident_bundle", return_value=bundle):
            with torch.enable_grad():
                events = list(stream_generate(
                    "x",
                    max_new_tokens=2,
                    temperature=0.0,
                    top_k=1,
                    device_name="cpu",
                ))

        self.assertEqual(model.calls, 2, "test must exercise both initial and incremental decoding")
        self.assertEqual(model.grad_enabled, [False, False])
        self.assertTrue(any(event.get("done") is True for event in events))
        self.assertTrue(any(event.get("delta") == "a" for event in events))


if __name__ == "__main__":
    unittest.main()

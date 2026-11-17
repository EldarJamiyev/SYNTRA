from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class NativeGenerationParameterPreflightTests(unittest.TestCase):
    def test_non_stream_generation_rejects_invalid_options_before_checkpoint_load(self):
        from syntra_ai.runtime import generate

        invalid_cases = [
            ({"max_new_tokens": True}, "max_new_tokens must be an integer"),
            ({"max_new_tokens": 1.5}, "max_new_tokens must be an integer"),
            ({"max_new_tokens": 4097}, "max_new_tokens must be an integer"),
            ({"temperature": False}, "temperature must be a finite number"),
            ({"temperature": float("nan")}, "temperature must be a finite number"),
            ({"temperature": 10**1000}, "temperature must be a finite number"),
            ({"top_k": True}, "top_k must be a positive integer"),
            ({"top_k": 0}, "top_k must be a positive integer"),
            ({"top_k": "40"}, "top_k must be a positive integer"),
            ({"top_p": False}, "top_p must be a finite number"),
            ({"top_p": 0.0}, "top_p must be a finite number"),
            ({"top_p": float("inf")}, "top_p must be a finite number"),
            ({"repetition_penalty": True}, "repetition_penalty must be a finite number"),
            ({"repetition_penalty": 0.9}, "repetition_penalty must be a finite number"),
            ({"repetition_penalty": 2.1}, "repetition_penalty must be a finite number"),
            ({"device_name": 1}, "device must be auto"),
            ({"device_name": "cuda:999999"}, "Requested device is unavailable"),
        ]

        with tempfile.TemporaryDirectory(prefix="syntra-generation-preflight-") as temp:
            missing_checkpoint = Path(temp) / "checkpoint-does-not-exist.pt"
            for options, expected in invalid_cases:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(ValueError, expected):
                        generate("valid prompt", path=missing_checkpoint, **options)

    def test_stream_generation_rejects_invalid_options_before_checkpoint_load(self):
        from syntra_ai.runtime import stream_generate

        invalid_cases = [
            ({"max_new_tokens": True}, "max_new_tokens must be an integer"),
            ({"temperature": "0.7"}, "temperature must be a finite number"),
            ({"top_k": 1.5}, "top_k must be a positive integer"),
            ({"top_p": 1.1}, "top_p must be a finite number"),
            ({"repetition_penalty": float("nan")}, "repetition_penalty must be a finite number"),
            ({"device_name": "invalid-device"}, "Requested device is unavailable"),
        ]

        with tempfile.TemporaryDirectory(prefix="syntra-stream-preflight-") as temp:
            missing_checkpoint = Path(temp) / "checkpoint-does-not-exist.pt"
            for options, expected in invalid_cases:
                with self.subTest(options=options):
                    stream = stream_generate("valid prompt", path=missing_checkpoint, **options)
                    with self.assertRaisesRegex(ValueError, expected):
                        next(stream)


if __name__ == "__main__":
    unittest.main()

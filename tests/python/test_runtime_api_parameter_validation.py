from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class NativeEmbeddingEvaluationParameterTests(unittest.TestCase):
    def test_embedding_device_is_validated_before_checkpoint_identity_lookup(self):
        from syntra_ai.runtime import embed_texts

        with tempfile.TemporaryDirectory(prefix="syntra-embed-preflight-") as temp:
            missing = Path(temp) / "missing-checkpoint.pt"
            for invalid_device in (123, "", "invalid-device"):
                with self.subTest(device=invalid_device):
                    with self.assertRaisesRegex(ValueError, "device must be auto|Requested device is unavailable"):
                        embed_texts(["valid text"], path=missing, device_name=invalid_device)

    def test_evaluation_counts_and_device_are_validated_before_root_or_checkpoint_access(self):
        from syntra_ai.runtime import evaluate

        with tempfile.TemporaryDirectory(prefix="syntra-evaluate-preflight-") as temp:
            missing_root = Path(temp) / "missing-root"
            missing_checkpoint = Path(temp) / "missing-checkpoint.pt"
            invalid_cases = [
                ({"batches": True}, "batches must be an integer"),
                ({"batches": 1.5}, "batches must be an integer"),
                ({"batches": "16"}, "batches must be an integer"),
                ({"max_tokens": True}, "max_tokens must be an integer"),
                ({"max_tokens": 1024.5}, "max_tokens must be an integer"),
                ({"max_tokens": "2000000"}, "max_tokens must be an integer"),
                ({"device_name": 123}, "device must be auto"),
                ({"device_name": "invalid-device"}, "Requested device is unavailable"),
            ]
            for overrides, expected in invalid_cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(ValueError, expected):
                        evaluate(
                            root=missing_root,
                            checkpoint=missing_checkpoint,
                            **overrides,
                        )


if __name__ == "__main__":
    unittest.main()

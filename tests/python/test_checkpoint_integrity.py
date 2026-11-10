from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run checkpoint tests.")
class CheckpointSidecarIntegrityTests(unittest.TestCase):
    @staticmethod
    def _payload(step: int = 1) -> dict:
        from brain.model import TransformerConfig
        from brain.tokenizer import BytePairTokenizer

        config = TransformerConfig(
            vocab_size=258,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
        tokenizer = BytePairTokenizer.empty()
        return {
            "format": "syntra-brain-checkpoint",
            "version": 1,
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "step": step,
            "epoch": step,
            "dataset_sha256": "a" * 64,
        }

    def test_sidecar_records_exact_checkpoint_digest_and_size(self):
        from brain.checkpoint import read_checkpoint_metadata, save_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-integrity-") as temp:
            checkpoint = Path(temp) / "model.pt"
            save_checkpoint(checkpoint, self._payload())

            metadata = read_checkpoint_metadata(checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()
            self.assertEqual(
                metadata["checkpoint_sha256"],
                hashlib.sha256(checkpoint_bytes).hexdigest(),
            )
            self.assertEqual(metadata["checkpoint_bytes"], len(checkpoint_bytes))
            self.assertEqual(metadata["checkpoint"], str(checkpoint.resolve()))

    def test_sidecar_is_rejected_when_checkpoint_bytes_change(self):
        from brain.checkpoint import read_checkpoint_metadata, save_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-tamper-") as temp:
            checkpoint = Path(temp) / "model.pt"
            save_checkpoint(checkpoint, self._payload())
            self.assertTrue(read_checkpoint_metadata(checkpoint))

            with checkpoint.open("ab") as handle:
                handle.write(b"unexpected trailing bytes")

            self.assertEqual(read_checkpoint_metadata(checkpoint), {})

    def test_failed_torch_serialization_removes_temporary_checkpoint(self):
        from brain.checkpoint import save_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-write-failure-") as temp:
            checkpoint = Path(temp) / "model.pt"
            with patch("brain.checkpoint.torch.save", side_effect=RuntimeError("serialization failed")):
                with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                    save_checkpoint(checkpoint, self._payload())

            self.assertFalse(checkpoint.exists())
            self.assertEqual(list(Path(temp).glob(".model.pt.*.tmp")), [])

    def test_failed_sidecar_update_does_not_expose_stale_metadata(self):
        from brain.checkpoint import read_checkpoint_metadata, save_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-checkpoint-sidecar-failure-") as temp:
            checkpoint = Path(temp) / "model.pt"
            save_checkpoint(checkpoint, self._payload(step=1))
            self.assertEqual(read_checkpoint_metadata(checkpoint)["step"], 1)

            with patch("brain.checkpoint._atomic_write_json", side_effect=OSError("sidecar write failed")):
                with self.assertRaisesRegex(OSError, "sidecar write failed"):
                    save_checkpoint(checkpoint, self._payload(step=2))

            # The checkpoint replacement succeeded, but the previous sidecar is now
            # stale and must not be returned as if it described the new checkpoint.
            self.assertEqual(read_checkpoint_metadata(checkpoint), {})


if __name__ == "__main__":
    unittest.main()

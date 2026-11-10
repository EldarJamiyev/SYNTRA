from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from syntra_ai.checkpoint_io import load_torch_checkpoint_snapshot


TORCH_SKIP_REASON = "Install requirements.txt to run native checkpoint snapshot tests."


@unittest.skipIf(torch is None, TORCH_SKIP_REASON)
class NativeCheckpointSnapshotTests(unittest.TestCase):
    def _write_checkpoint(self, path: Path, marker: int = 1) -> None:
        torch.save({
            "format": "syntra-transformer-v1",
            "marker": marker,
            "weights": torch.tensor([marker, marker + 1], dtype=torch.int64),
        }, path)

    def test_loads_real_checkpoint_bytes_from_a_verified_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="syntra-native-checkpoint-snapshot-") as temp:
            checkpoint = Path(temp) / "model.pt"
            self._write_checkpoint(checkpoint, marker=17)

            payload = load_torch_checkpoint_snapshot(checkpoint)

            self.assertEqual(payload["format"], "syntra-transformer-v1")
            self.assertEqual(payload["marker"], 17)
            torch.testing.assert_close(payload["weights"], torch.tensor([17, 18]))

    def test_rejects_final_component_symlink(self):
        with tempfile.TemporaryDirectory(prefix="syntra-native-checkpoint-symlink-") as temp:
            root = Path(temp)
            target = root / "target.pt"
            link = root / "linked.pt"
            self._write_checkpoint(target)
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("File symlinks are unavailable on this runner.")

            with self.assertRaisesRegex(ValueError, "non-empty regular file"):
                load_torch_checkpoint_snapshot(link)

    def test_rejects_path_replacement_after_open_before_deserialization(self):
        with tempfile.TemporaryDirectory(prefix="syntra-native-checkpoint-path-race-") as temp:
            root = Path(temp)
            checkpoint = root / "model.pt"
            replacement = root / "replacement.pt"
            self._write_checkpoint(checkpoint, marker=1)
            self._write_checkpoint(replacement, marker=99)

            real_stat = os.stat
            source_checks = 0

            def replace_before_final_path_check(path, *args, **kwargs):
                nonlocal source_checks
                if Path(path) == checkpoint and kwargs.get("follow_symlinks") is False:
                    source_checks += 1
                    if source_checks == 2:
                        os.replace(replacement, checkpoint)
                return real_stat(path, *args, **kwargs)

            with patch("syntra_ai.checkpoint_io.os.stat", side_effect=replace_before_final_path_check):
                with self.assertRaisesRegex(ValueError, "changed while it was being read"):
                    load_torch_checkpoint_snapshot(checkpoint)

            self.assertEqual(source_checks, 2)
            self.assertEqual(
                load_torch_checkpoint_snapshot(checkpoint)["marker"],
                99,
                "The replacement is real, but it must not be confused with the bytes read from the old descriptor.",
            )

    def test_enforces_the_byte_limit_during_snapshot_read(self):
        with tempfile.TemporaryDirectory(prefix="syntra-native-checkpoint-budget-") as temp:
            checkpoint = Path(temp) / "model.pt"
            self._write_checkpoint(checkpoint)

            with self.assertRaisesRegex(ValueError, "byte limit"):
                load_torch_checkpoint_snapshot(checkpoint, max_bytes=checkpoint.stat().st_size - 1)

    def test_enforces_available_ram_budget_inside_the_verified_snapshot_read(self):
        with tempfile.TemporaryDirectory(prefix="syntra-native-checkpoint-memory-budget-") as temp:
            checkpoint = Path(temp) / "model.pt"
            self._write_checkpoint(checkpoint)
            memory_budget = checkpoint.stat().st_size - 1

            with patch(
                "syntra_ai.checkpoint_io.checkpoint_load_budget_bytes",
                return_value=memory_budget,
            ):
                with self.assertRaisesRegex(ValueError, "40% checkpoint-load budget"):
                    load_torch_checkpoint_snapshot(checkpoint)


class NativeCheckpointIntegrationTests(unittest.TestCase):
    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_runtime_checkpoint_loader_uses_verified_snapshot(self):
        from syntra_ai.config import ModelConfig
        from syntra_ai.runtime import _read_checkpoint
        from syntra_ai.tokenizer import ByteBPETokenizer

        with tempfile.TemporaryDirectory(prefix="syntra-runtime-snapshot-") as temp:
            checkpoint = Path(temp) / "model.pt"
            config = ModelConfig(vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
            torch.save({
                "format": "syntra-transformer-v1",
                "config": config.to_dict(),
                "tokenizer": ByteBPETokenizer.empty().to_dict(),
                "model_state": {},
            }, checkpoint)

            with patch(
                "syntra_ai.runtime.load_torch_checkpoint_snapshot",
                wraps=load_torch_checkpoint_snapshot,
            ) as loader:
                _target, payload, loaded_config, _tokenizer = _read_checkpoint(checkpoint)

            loader.assert_called_once()
            self.assertEqual(payload["format"], "syntra-transformer-v1")
            self.assertEqual(loaded_config, config)

    @unittest.skipIf(torch is None, TORCH_SKIP_REASON)
    def test_training_resume_uses_verified_snapshot(self):
        from syntra_ai.training import train

        with tempfile.TemporaryDirectory(prefix="syntra-training-resume-snapshot-") as temp:
            checkpoint = Path(temp) / "model.pt"
            torch.save({"format": "unsupported-checkpoint"}, checkpoint)

            with patch(
                "syntra_ai.training.load_torch_checkpoint_snapshot",
                wraps=load_torch_checkpoint_snapshot,
            ) as loader:
                with self.assertRaisesRegex(ValueError, "supported SYNTRA Transformer checkpoint"):
                    train(
                        roots=[],
                        checkpoint=checkpoint,
                        steps=1,
                        batch_size=1,
                        max_tokens=1024,
                        resume=True,
                    )

            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()

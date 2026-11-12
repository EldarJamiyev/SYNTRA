from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from syntra_ai.instruction import load_instruction_examples


class InstructionDatasetPathIdentityTests(unittest.TestCase):
    def setUp(self):
        self.previous_roots = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")

    def tearDown(self):
        if self.previous_roots is None:
            os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
        else:
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = self.previous_roots

    @staticmethod
    def _dataset_text() -> str:
        return (
            json.dumps({"prompt": "Explain addition", "completion": "Add two values."}) + "\\n"
            + json.dumps({"prompt": "Explain maximum", "completion": "Return the larger value."}) + "\\n"
        )

    def test_reader_rejects_file_replacement_after_open(self):
        with tempfile.TemporaryDirectory(prefix="syntra-instruction-path-race-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            replacement = root / "replacement.jsonl"
            dataset.write_text(self._dataset_text(), encoding="utf-8")
            replacement.write_text(
                json.dumps({"prompt": "Different task", "completion": "Different result."}) + "\\n",
                encoding="utf-8",
            )
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(root)
            import syntra_ai.instruction as instruction

            real_stat = os.stat
            calls = 0

            def replace_before_final_path_check(path, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    os.replace(replacement, dataset)
                return real_stat(path, *args, **kwargs)

            with patch.object(instruction.os, "stat", side_effect=replace_before_final_path_check):
                with self.assertRaisesRegex(ValueError, "changed while it was being read"):
                    load_instruction_examples(dataset, seq_len=64)

            self.assertIn("Different task", dataset.read_text(encoding="utf-8"))

    def test_reader_rejects_parent_redirect_to_same_inode_outside_approved_root(self):
        with tempfile.TemporaryDirectory(prefix="syntra-instruction-parent-race-") as temp:
            workspace = Path(temp)
            approved = workspace / "approved"
            parent = approved / "datasets"
            outside = workspace / "outside"
            held = workspace / "held-datasets"
            parent.mkdir(parents=True)
            outside.mkdir()
            dataset = parent / "instructions.jsonl"
            dataset.write_text(self._dataset_text(), encoding="utf-8")
            try:
                os.link(dataset, outside / dataset.name)
            except OSError:
                self.skipTest("Hard-link creation is unavailable on this runner.")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(approved)
            import syntra_ai.instruction as instruction

            real_fstat = os.fstat
            calls = 0

            def redirect_parent_after_descriptor_read(descriptor):
                nonlocal calls
                calls += 1
                result = real_fstat(descriptor)
                if calls == 2:
                    os.replace(parent, held)
                    try:
                        parent.symlink_to(outside, target_is_directory=True)
                    except (OSError, NotImplementedError):
                        self.skipTest("Directory symlinks are unavailable on this runner.")
                return result

            with patch.object(instruction.os, "fstat", side_effect=redirect_parent_after_descriptor_read):
                with self.assertRaisesRegex(ValueError, "changed while it was being read"):
                    load_instruction_examples(dataset, seq_len=64)

    def test_reader_accepts_a_stable_approved_jsonl_file(self):
        with tempfile.TemporaryDirectory(prefix="syntra-instruction-stable-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            dataset.write_text(self._dataset_text(), encoding="utf-8")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(root)
            examples = load_instruction_examples(dataset, seq_len=64)
            self.assertGreaterEqual(len(examples), 2)


if __name__ == "__main__":
    unittest.main()
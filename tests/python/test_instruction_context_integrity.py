"""Regression tests for lossless instruction prompt conditioning."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from syntra_ai.instruction import load_instruction_examples
from syntra_ai.tokenizer import ByteBPETokenizer


class InstructionContextIntegrityTests(unittest.TestCase):
    def _load(self, root: Path, prompt: str, completion: str, seq_len: int = 16):
        dataset = root / "instructions.jsonl"
        dataset.write_text(
            json.dumps({"prompt": prompt, "completion": completion}) + "\\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
            return load_instruction_examples(
                dataset,
                seq_len=seq_len,
                tokenizer=ByteBPETokenizer.empty(),
            )

    def test_prompt_that_fits_checkpoint_context_is_preserved_in_full(self):
        with tempfile.TemporaryDirectory(prefix="syntra-prompt-context-fit-") as temp:
            root = Path(temp)
            prompt = "p" * 15
            examples = self._load(root, prompt, "ok")
            supervised = sum(
                token != -100
                for example in examples
                for token in example["labels"]
            )
            self.assertEqual(supervised, len(ByteBPETokenizer.empty().encode("ok")) + 1)
            self.assertEqual(examples[0]["input_ids"][1:], ByteBPETokenizer.empty().encode(prompt))

    def test_overlong_prompt_is_rejected_instead_of_silently_truncated(self):
        with tempfile.TemporaryDirectory(prefix="syntra-prompt-context-overflow-") as temp:
            root = Path(temp)
            with self.assertRaisesRegex(
                ValueError,
                r"Prompt on line 1 requires 16 tokens.*prompt was not truncated",
            ):
                self._load(root, "p" * 16, "ok")

    def test_long_completion_remains_fully_supervised_across_windows(self):
        with tempfile.TemporaryDirectory(prefix="syntra-completion-windows-") as temp:
            root = Path(temp)
            completion = "answer-" * 20
            examples = self._load(root, "task", completion)
            supervised = sum(
                token != -100
                for example in examples
                for token in example["labels"]
            )
            expected = len(ByteBPETokenizer.empty().encode(completion)) + 1
            self.assertEqual(supervised, expected)
            self.assertGreater(len(examples), 1)


if __name__ == "__main__":
    unittest.main()

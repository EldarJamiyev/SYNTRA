from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from syntra_ai.instruction import audit_instruction_dataset, load_instruction_examples
from syntra_ai.tokenizer import ByteBPETokenizer


class InstructionJsonlStrictnessTests(unittest.TestCase):
    def _write(self, root: Path, line: str) -> Path:
        dataset = root / "instructions.jsonl"
        dataset.write_text(line + "\n", encoding="utf-8")
        return dataset

    def _load(self, root: Path, line: str):
        dataset = self._write(root, line)
        with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
            return load_instruction_examples(
                dataset,
                seq_len=128,
                tokenizer=ByteBPETokenizer.empty(),
            )

    def test_duplicate_required_member_names_are_rejected_instead_of_last_value_winning(self):
        with tempfile.TemporaryDirectory(prefix="syntra-jsonl-duplicate-fields-") as temp:
            root = Path(temp)
            line = '{"prompt":"safe instruction","completion":"first answer","completion":"different answer"}'
            with self.assertRaisesRegex(ValueError, r"Invalid JSON on line 1: duplicate JSON object key 'completion'"):
                self._load(root, line)

    def test_duplicate_member_names_are_rejected_inside_unknown_nested_metadata(self):
        with tempfile.TemporaryDirectory(prefix="syntra-jsonl-duplicate-nested-") as temp:
            root = Path(temp)
            line = (
                '{"prompt":"safe instruction","completion":"answer",'
                '"metadata":{"review":{"status":"approved","status":"rejected"}}}'
            )
            with self.assertRaisesRegex(ValueError, r"Invalid JSON on line 1: duplicate JSON object key 'status'"):
                self._load(root, line)

    def test_non_standard_constants_and_overflowed_json_numbers_are_rejected(self):
        records = (
            '{"prompt":"task","completion":"answer","quality":NaN}',
            '{"prompt":"task","completion":"answer","quality":Infinity}',
            '{"prompt":"task","completion":"answer","quality":-Infinity}',
            '{"prompt":"task","completion":"answer","quality":1e999}',
        )
        with tempfile.TemporaryDirectory(prefix="syntra-jsonl-non-finite-") as temp:
            root = Path(temp)
            for line in records:
                with self.subTest(line=line), self.assertRaisesRegex(ValueError, r"Invalid JSON on line 1:"):
                    self._load(root, line)

    def test_excessive_json_nesting_returns_a_bounded_dataset_error(self):
        with tempfile.TemporaryDirectory(prefix="syntra-jsonl-deep-nesting-") as temp:
            root = Path(temp)
            nested = "[" * 1200 + "0" + "]" * 1200
            line = '{"prompt":"task","completion":"answer","metadata":' + nested + "}"
            with self.assertRaisesRegex(ValueError, r"Invalid JSON on line 1:"):
                self._load(root, line)

    def test_valid_nested_metadata_and_optional_training_groups_remain_supported(self):
        with tempfile.TemporaryDirectory(prefix="syntra-jsonl-valid-nested-") as temp:
            root = Path(temp)
            line = json.dumps({
                "prompt": "Explain the selected operation.",
                "completion": "Perform the operation safely.",
                "split_group": "release-17",
                "source_path": "src/operations/selected.py",
                "metadata": {
                    "review": {"status": "approved", "reviewers": ["one", "two"]},
                    "quality": {"score": 0.75},
                },
            })
            examples = self._load(root, line)
            self.assertTrue(examples)
            self.assertTrue(all(example["split_group_id"] for example in examples))
            self.assertTrue(all(example["source_path_group_id"] for example in examples))

    def test_dataset_audit_reports_strict_json_errors_as_invalid_input(self):
        with tempfile.TemporaryDirectory(prefix="syntra-jsonl-audit-strict-") as temp:
            root = Path(temp)
            dataset = self._write(
                root,
                '{"prompt":"task","prompt":"overwritten","completion":"answer"}',
            )
            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = audit_instruction_dataset(
                    dataset,
                    seq_len=128,
                    tokenizer=ByteBPETokenizer.empty(),
                )
            self.assertFalse(result["valid"])
            self.assertFalse(result["split_available"])
            self.assertIn("duplicate JSON object key", result["error"])


if __name__ == "__main__":
    unittest.main()

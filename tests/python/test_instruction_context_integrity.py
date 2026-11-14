"""Regression tests for lossless instruction prompt conditioning."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from syntra_ai.instruction import audit_instruction_dataset, load_instruction_examples, split_instruction_examples
from syntra_ai.tokenizer import ByteBPETokenizer


class InstructionContextIntegrityTests(unittest.TestCase):
    def _load(self, root: Path, prompt: str, completion: str, seq_len: int = 16):
        dataset = root / "instructions.jsonl"
        dataset.write_text(
            json.dumps({"prompt": prompt, "completion": completion}) + "\n",
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

    def test_dataset_fingerprint_matches_exact_jsonl_bytes_in_loader_and_audit(self):
        with tempfile.TemporaryDirectory(prefix="syntra-dataset-provenance-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            content = (
                json.dumps({"prompt": "task one", "completion": "answer one"}) + "\n"
            ).encode("utf-8")
            dataset.write_bytes(content)
            expected_sha256 = hashlib.sha256(content).hexdigest()
            tokenizer = ByteBPETokenizer.empty()

            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                examples = load_instruction_examples(dataset, seq_len=32, tokenizer=tokenizer)
                audit = audit_instruction_dataset(dataset, seq_len=32, tokenizer=tokenizer)

            self.assertTrue(examples)
            self.assertTrue(all(item["dataset_sha256"] == expected_sha256 for item in examples))
            self.assertTrue(all(item["dataset_bytes"] == len(content) for item in examples))
            self.assertEqual(audit["dataset_sha256"], expected_sha256)
            self.assertEqual(audit["dataset_bytes"], len(content))

            updated = content + (
                json.dumps({"prompt": "task two", "completion": "answer two"}) + "\n"
            ).encode("utf-8")
            dataset.write_bytes(updated)
            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                changed = load_instruction_examples(dataset, seq_len=32, tokenizer=tokenizer)
            self.assertNotEqual(changed[0]["dataset_sha256"], expected_sha256)
            self.assertEqual(changed[0]["dataset_sha256"], hashlib.sha256(updated).hexdigest())

    def test_split_rejects_context_windows_with_mixed_dataset_identities(self):
        with tempfile.TemporaryDirectory(prefix="syntra-mixed-dataset-identity-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            rows = [
                {"prompt": "Explain one.", "completion": "The first operation."},
                {"prompt": "Explain two.", "completion": "The second operation."},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                examples = load_instruction_examples(
                    dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty()
                )
            examples[-1]["dataset_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "one immutable dataset identity"):
                split_instruction_examples(examples)

    def test_source_path_related_records_never_cross_validation_split(self):
        with tempfile.TemporaryDirectory(prefix="syntra-source-path-split-") as temp:
            root = Path(temp)
            rows = [
                {"prompt": "Implement alpha.", "completion": "alpha()", "source_path": "./src/alpha.py"},
                {"prompt": "Explain alpha.", "completion": "alpha is a function.", "source_path": "src//alpha.py"},
                {"prompt": "Repair alpha.", "completion": "alpha is repaired.", "source_path": "src\\alpha.py"},
                {"prompt": "Implement beta.", "completion": "beta()", "source_path": "src/beta.py"},
                {"prompt": "Implement gamma.", "completion": "gamma()", "source_path": "src/gamma.py"},
            ]
            dataset = root / "instructions.jsonl"
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                examples = load_instruction_examples(
                    dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty()
                )

            training, validation, quality = split_instruction_examples(examples)
            memberships = {}
            for split_name, collection in (("train", training), ("validation", validation)):
                for item in collection:
                    memberships.setdefault(item["source_path_group_id"], set()).add(split_name)
            self.assertEqual(len(memberships), 3)
            self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))
            self.assertEqual(len(examples), 5)
            self.assertEqual(examples[0]["source_path_group_id"], examples[1]["source_path_group_id"])
            self.assertEqual(examples[1]["source_path_group_id"], examples[2]["source_path_group_id"])
            self.assertNotEqual(examples[0]["source_path_group_id"], examples[3]["source_path_group_id"])
            self.assertEqual(quality["source_path_group_count"], 3)
            self.assertEqual(
                quality["split_strategy"],
                "deterministic-sha256-prompt-source-path-and-source-groups-v4",
            )

    def test_source_path_aliases_normalize_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="syntra-source-path-aliases-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            rows = [
                {"prompt": "One.", "completion": "First.", "source_path": "src\\module.py"},
                {"prompt": "Two.", "completion": "Second.", "source_path": "./src//module.py"},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                examples = load_instruction_examples(
                    dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty()
                )
            self.assertEqual(
                examples[0]["source_path_group_id"],
                examples[1]["source_path_group_id"],
            )
            for unsafe_path in ("/etc/passwd", "C:\\repo\\module.py", "../outside.py", "src/../outside.py", "."):
                dataset.write_text(
                    json.dumps({"prompt": "Unsafe.", "completion": "Reject.", "source_path": unsafe_path}) + "\n",
                    encoding="utf-8",
                )
                with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                    with self.subTest(source_path=unsafe_path), self.assertRaisesRegex(
                        ValueError, "source_path on line 1 is invalid"
                    ):
                        load_instruction_examples(
                            dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty()
                        )

    def test_source_path_metadata_must_be_a_bounded_string(self):
        with tempfile.TemporaryDirectory(prefix="syntra-source-path-schema-") as temp:
            root = Path(temp)
            dataset = root / "instructions.jsonl"
            dataset.write_text(
                json.dumps({"prompt": "Explain alpha.", "completion": "Alpha is valid.", "source_path": []}) + "\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                with self.assertRaisesRegex(ValueError, "source_path on line 1"):
                    load_instruction_examples(
                        dataset, seq_len=64, tokenizer=ByteBPETokenizer.empty()
                    )

    def test_validation_loss_evaluates_every_held_out_window_once_in_stable_order(self):
        from syntra_ai import instruction as instruction_module

        if instruction_module.torch is None:
            self.skipTest("Install the pinned PyTorch requirements to run SFT validation tests.")
        supervised_tokens = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        examples = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [1, 4], "labels": [-100, 4]},
            {"input_ids": [1, 5, 6, 7], "labels": [-100, 5, 6, 7]},
            {"input_ids": [1, 8], "labels": [-100, 8]},
            {"input_ids": [1, 9, 10], "labels": [-100, 9, 10]},
        ]

        class RecordingModel:
            def __init__(self):
                self.training = True
                self.seen_tokens = []

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = mode
                return self

            def __call__(self, inputs, targets):
                selected = targets[targets != -100]
                self.seen_tokens.extend(selected.detach().cpu().tolist())
                return None, selected.float().mean()

        model = RecordingModel()
        with patch.object(
            instruction_module,
            "_sample_instruction_examples",
            side_effect=AssertionError("Validation must not sample examples randomly."),
        ):
            first_loss = instruction_module._instruction_loss(
                model, examples, device="cpu", batch_size=2
            )
            self.assertEqual(model.seen_tokens, supervised_tokens)
            self.assertTrue(model.training)

            model.seen_tokens.clear()
            second_loss = instruction_module._instruction_loss(
                model, examples, device="cpu", batch_size=2
            )

        self.assertEqual(model.seen_tokens, supervised_tokens)
        self.assertAlmostEqual(first_loss, sum(supervised_tokens) / len(supervised_tokens), places=6)
        self.assertAlmostEqual(second_loss, first_loss, places=7)
        self.assertTrue(model.training)

    def test_validation_loss_rejects_an_empty_holdout(self):
        from syntra_ai import instruction as instruction_module

        if instruction_module.torch is None:
            self.skipTest("Install the pinned PyTorch requirements to run SFT validation tests.")

        class Model:
            training = True

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

        with self.assertRaisesRegex(ValueError, "at least one held-out instruction example"):
            instruction_module._instruction_loss(Model(), [], device="cpu", batch_size=2)

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

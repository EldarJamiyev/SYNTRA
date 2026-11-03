import os
import tempfile
import unittest
from pathlib import Path

from syntra_ai.safe_paths import resolve_corpus_root, resolve_instruction_dataset


class ApprovedCorpusPathTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")
        os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
        else:
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = self.previous

    def test_request_paths_do_not_expand_home_shorthand(self):
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            resolve_corpus_root(str(Path("~") / "not-a-corpus"))

    def test_repository_root_is_the_only_default_external_data_boundary(self):
        from syntra_ai.safe_paths import PROJECT_ROOT

        self.assertEqual(resolve_corpus_root(str(PROJECT_ROOT)), PROJECT_ROOT.resolve())
        with tempfile.TemporaryDirectory(prefix="syntra-unapproved-") as temp:
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_corpus_root(temp)

    def test_configured_root_accepts_descendants_but_rejects_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory(prefix="syntra-approved-") as temp:
            parent = Path(temp)
            approved = parent / "approved"
            outside = parent / "outside"
            corpus = approved / "corpus"
            corpus.mkdir(parents=True)
            outside.mkdir()
            (outside / "secret.py").write_text("PRIVATE = True\n", encoding="utf-8")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(approved)

            self.assertEqual(resolve_corpus_root(str(corpus)), corpus.resolve())
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_corpus_root(str(approved / ".." / "outside"))

            escape = approved / "escape"
            try:
                escape.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks are unavailable on this runner.")
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_corpus_root(str(escape))

    def test_finetuning_dataset_must_be_jsonl_and_inside_an_approved_root(self):
        with tempfile.TemporaryDirectory(prefix="syntra-datasets-") as temp:
            root = Path(temp)
            approved = root / "approved"
            outside = root / "outside"
            approved.mkdir()
            outside.mkdir()
            valid = approved / "training.jsonl"
            valid.write_text('{"prompt":"hello","completion":"world"}\n', encoding="utf-8")
            wrong_extension = approved / "training.txt"
            wrong_extension.write_text("not jsonl\n", encoding="utf-8")
            external = outside / "outside.jsonl"
            external.write_text("{}\n", encoding="utf-8")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(approved)

            self.assertEqual(resolve_instruction_dataset(str(valid)), valid.resolve())
            with self.assertRaisesRegex(ValueError, "must use the .jsonl"):
                resolve_instruction_dataset(str(wrong_extension))
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_instruction_dataset(str(external))
            with self.assertRaisesRegex(ValueError, "existing regular file"):
                resolve_instruction_dataset(str(approved))


if __name__ == "__main__":
    unittest.main()

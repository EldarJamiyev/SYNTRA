import os
import tempfile
import unittest
from pathlib import Path

from syntra_ai.safe_paths import (
    resolve_corpus_root, resolve_instruction_dataset, resolve_curriculum_output_path,
)


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

    def test_configured_roots_must_be_absolute_without_home_expansion(self):
        os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(Path("~") / "not-an-approved-root")
        from syntra_ai.safe_paths import PROJECT_ROOT

        with self.assertRaisesRegex(ValueError, "entries must be absolute"):
            resolve_corpus_root(str(PROJECT_ROOT))

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

            prefix_escape = parent / "approved-neighbor"
            prefix_escape.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_corpus_root(str(prefix_escape))

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
            escaped_link = approved / "external-link.jsonl"
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(approved)

            try:
                escaped_link.symlink_to(external)
            except (OSError, NotImplementedError):
                escaped_link = None

            self.assertEqual(resolve_instruction_dataset(str(valid)), valid.resolve())
            if escaped_link is not None:
                with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                    resolve_instruction_dataset(str(escaped_link))
            with self.assertRaisesRegex(ValueError, "must use the .jsonl"):
                resolve_instruction_dataset(str(wrong_extension))
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_instruction_dataset(str(external))
            with self.assertRaisesRegex(ValueError, "existing regular file"):
                resolve_instruction_dataset(str(approved))



    def test_curriculum_output_must_stay_inside_workspace_or_project(self):
        with tempfile.TemporaryDirectory(prefix="syntra-output-path-") as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(workspace)

            nested = workspace / "generated" / "nested"
            nested_output = nested / "curriculum.jsonl"
            self.assertEqual(
                resolve_curriculum_output_path(nested_output, workspace),
                nested_output,
            )
            self.assertTrue(nested.is_dir())
            with self.assertRaisesRegex(ValueError, "inside the active workspace"):
                resolve_curriculum_output_path(outside / "escape.jsonl", workspace)
            with self.assertRaisesRegex(ValueError, "simple .jsonl filename"):
                resolve_curriculum_output_path(workspace / ".." / "outside" / "bad name.jsonl", workspace)

            link = workspace / "linked.jsonl"
            try:
                link.symlink_to(outside / "target.jsonl")
            except (OSError, NotImplementedError):
                self.skipTest("File symlinks are unavailable on this runner.")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                resolve_curriculum_output_path(link, workspace)

    def test_instruction_dataset_loader_boundary_rejects_unapproved_paths(self):
        with tempfile.TemporaryDirectory(prefix="syntra-direct-dataset-") as temp:
            root = Path(temp)
            dataset = root / "outside.jsonl"
            dataset.write_text('{"prompt":"hello","completion":"world"}\\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                resolve_instruction_dataset(dataset)

if __name__ == "__main__":
    unittest.main()

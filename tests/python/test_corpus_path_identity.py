from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from syntra_ai.data import _read_safe_file, load_corpus


class SourceCorpusPathIdentityTests(unittest.TestCase):
    def _read_with_replacement(self, replacement_kind: str):
        with tempfile.TemporaryDirectory(prefix="syntra-corpus-path-race-") as temp:
            root = Path(temp)
            source = root / "a.py"
            excluded_target_dir = root / ".excluded"
            excluded_target_dir.mkdir()
            replacement = excluded_target_dir / "replacement.py"
            stable = root / "b.py"
            source.write_text("ORIGINAL_CORPUS_BYTES = True\n", encoding="utf-8")
            replacement.write_text("REPLACEMENT_CORPUS_BYTES = True\n", encoding="utf-8")
            stable_content = b"STABLE_CORPUS_BYTES = True\n"
            stable.write_bytes(stable_content)

            if replacement_kind == "symlink":
                link_target = excluded_target_dir / "outside.py"
                link_target.write_text("LINK_TARGET_BYTES = True\n", encoding="utf-8")
                link_source = excluded_target_dir / "replacement-link.py"
                try:
                    link_source.symlink_to(link_target)
                except (OSError, NotImplementedError):
                    self.skipTest("File symlinks are unavailable on this runner.")
                replacement.unlink()
                replacement = link_source

            real_fstat = os.fstat
            calls = 0

            def replace_after_descriptor_read(descriptor):
                nonlocal calls
                calls += 1
                result = real_fstat(descriptor)
                # The first two fstat calls belong to a.py: descriptor-open
                # identity and post-read stability. Replace the pathname after
                # the post-read descriptor check but before its final lstat.
                if calls == 2:
                    os.replace(replacement, source)
                return result

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                with patch("syntra_ai.data.os.fstat", side_effect=replace_after_descriptor_read):
                    corpus = load_corpus(
                        [str(root)],
                        max_files=10,
                        max_file_bytes=1024,
                        max_tokens=4096,
                        include_tokens=False,
                    )

            self.assertEqual(corpus.files, [str(stable.resolve())])
            self.assertEqual(corpus.documents, [stable_content.decode("utf-8")])
            self.assertEqual(corpus.skipped_files, 1)
            expected_digest = hashlib.sha256()
            path_bytes = os.path.normcase(str(stable.resolve())).encode("utf-8")
            expected_digest.update(len(path_bytes).to_bytes(8, "big"))
            expected_digest.update(path_bytes)
            expected_digest.update(len(stable_content).to_bytes(8, "big"))
            expected_digest.update(stable_content)
            self.assertEqual(
                corpus.source_sha256,
                expected_digest.hexdigest(),
                "the digest must contain only bytes accepted under their final path identity",
            )

    def test_regular_file_replacement_after_open_is_excluded_from_corpus(self):
        self._read_with_replacement("regular")

    def test_symlink_substitution_after_open_is_excluded_from_corpus(self):
        self._read_with_replacement("symlink")

    def test_unchanged_source_paths_remain_included_and_hashed(self):
        with tempfile.TemporaryDirectory(prefix="syntra-corpus-path-stable-") as temp:
            root = Path(temp)
            source = root / "stable.py"
            content = b"STABLE_SOURCE = True\n"
            source.write_bytes(content)

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                corpus = load_corpus(
                    [str(root)],
                    max_files=10,
                    max_file_bytes=1024,
                    max_tokens=4096,
                    include_tokens=False,
                )

            self.assertEqual(corpus.files, [str(source.resolve())])
            self.assertEqual(corpus.documents, [content.decode("utf-8")])
            self.assertEqual(corpus.skipped_files, 0)
            expected = hashlib.sha256()
            path_bytes = os.path.normcase(str(source.resolve())).encode("utf-8")
            expected.update(len(path_bytes).to_bytes(8, "big"))
            expected.update(path_bytes)
            expected.update(len(content).to_bytes(8, "big"))
            expected.update(content)
            self.assertEqual(corpus.source_sha256, expected.hexdigest())


    def test_parent_directory_symlink_swap_cannot_escape_approved_corpus_root(self):
        with tempfile.TemporaryDirectory(prefix="syntra-corpus-parent-race-") as temp:
            base = Path(temp)
            root = base / "approved"
            source_dir = root / "src"
            outside_dir = base / "outside" / "src"
            root.mkdir()
            source_dir.mkdir()
            outside_dir.mkdir(parents=True)
            source = source_dir / "module.py"
            source.write_text("PATH_IDENTITY = True\\n", encoding="utf-8")
            outside_alias = outside_dir / "module.py"
            try:
                os.link(source, outside_alias)
            except OSError:
                self.skipTest("Hard links are unavailable on this runner.")

            held_directory = root / ".held-src"
            real_fstat = os.fstat
            calls = 0

            def swap_parent_after_descriptor_read(descriptor):
                nonlocal calls
                calls += 1
                result = real_fstat(descriptor)
                if calls == 2:
                    os.replace(source_dir, held_directory)
                    try:
                        source_dir.symlink_to(outside_dir, target_is_directory=True)
                    except (OSError, NotImplementedError):
                        self.skipTest("Directory symlinks are unavailable on this runner.")
                return result

            with patch("syntra_ai.data.os.fstat", side_effect=swap_parent_after_descriptor_read):
                accepted = _read_safe_file(str(source), str(root), 1024)

            self.assertTrue(source_dir.is_symlink())
            self.assertTrue(outside_alias.is_file())
            self.assertIsNone(
                accepted,
                "a parent-directory symlink swap must not keep an escaped file under its stale in-root path",
            )


if __name__ == "__main__":
    unittest.main()

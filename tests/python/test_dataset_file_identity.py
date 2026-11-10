from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brain.train import _read_bounded_regular_file


class DatasetPathIdentityTests(unittest.TestCase):
    def test_reader_rejects_path_replacement_after_open(self):
        with tempfile.TemporaryDirectory(prefix="syntra-dataset-path-race-") as temp:
            root = Path(temp)
            source = root / "dataset.txt"
            replacement = root / "replacement.txt"
            source.write_text("selected corpus\n", encoding="utf-8")
            replacement.write_text("different corpus\n", encoding="utf-8")
            real_stat = os.stat
            stat_calls = 0

            def replace_before_final_path_check(path, *args, **kwargs):
                nonlocal stat_calls
                stat_calls += 1
                if stat_calls == 2:
                    os.replace(replacement, source)
                return real_stat(path, *args, **kwargs)

            with patch("brain.train.os.stat", side_effect=replace_before_final_path_check):
                with self.assertRaisesRegex(ValueError, "changed while it was being read"):
                    _read_bounded_regular_file(source, 1024, label="training dataset")

            self.assertEqual(source.read_text(encoding="utf-8"), "different corpus\n")

    def test_reader_accepts_unchanged_regular_file(self):
        with tempfile.TemporaryDirectory(prefix="syntra-dataset-stable-") as temp:
            source = Path(temp) / "dataset.txt"
            source.write_text("stable corpus\n", encoding="utf-8")
            self.assertEqual(
                _read_bounded_regular_file(source, 1024, label="training dataset"),
                b"stable corpus\n",
            )


if __name__ == "__main__":
    unittest.main()

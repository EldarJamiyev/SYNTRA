import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from path_utils import normalize_relative_path


class ExistingBehavior(unittest.TestCase):
    def test_simple_relative_path_is_preserved(self):
        self.assertEqual(normalize_relative_path("src/module.py"), "src/module.py")

    def test_unicode_filename_is_preserved(self):
        self.assertEqual(normalize_relative_path("docs/əlavə.md"), "docs/əlavə.md")


if __name__ == "__main__":
    unittest.main()

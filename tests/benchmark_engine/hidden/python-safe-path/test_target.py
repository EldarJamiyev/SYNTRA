import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from path_utils import normalize_relative_path


class TargetBehavior(unittest.TestCase):
    def test_collapses_separators_and_dot_segments(self):
        self.assertEqual(normalize_relative_path("src//./utils\\parser.py"), "src/utils/parser.py")

    def test_rejects_parent_traversal(self):
        self.assertIsNone(normalize_relative_path("../secrets/token.json"))
        self.assertIsNone(normalize_relative_path("src/../../private"))

    def test_rejects_absolute_and_drive_prefixed_paths(self):
        self.assertIsNone(normalize_relative_path("/etc/passwd"))
        self.assertIsNone(normalize_relative_path("C:\\Users\\Eldar\\file.txt"))
        self.assertIsNone(normalize_relative_path("\\\\server\\share\\file.txt"))


if __name__ == "__main__":
    unittest.main()

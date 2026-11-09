import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from slug import slugify


class ExistingBehavior(unittest.TestCase):
    def test_none_remains_an_empty_string(self):
        self.assertEqual(slugify(None), "")

    def test_simple_valid_slug_is_preserved(self):
        self.assertEqual(slugify(" already-valid "), "already-valid")


if __name__ == "__main__":
    unittest.main()

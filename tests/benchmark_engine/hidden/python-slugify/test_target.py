import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from slug import slugify


class TargetBehavior(unittest.TestCase):
    def test_collapses_runs_of_non_ascii_alphanumeric_characters(self):
        self.assertEqual(slugify("Hello,   World!"), "hello-world")

    def test_treats_non_ascii_letters_as_separators(self):
        self.assertEqual(slugify("Crème brûlée"), "cr-me-br-l-e")

    def test_trims_leading_and_trailing_separators(self):
        self.assertEqual(slugify("  SYNTRA / Local AI  "), "syntra-local-ai")


if __name__ == "__main__":
    unittest.main()

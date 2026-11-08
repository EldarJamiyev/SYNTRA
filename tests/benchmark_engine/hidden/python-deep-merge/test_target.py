import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from config import merge_config


class TargetBehavior(unittest.TestCase):
    def test_recursively_merges_nested_dictionaries(self):
        base = {"service": {"host": "localhost", "port": 8080}}
        override = {"service": {"port": 9090}}
        self.assertEqual(merge_config(base, override), {"service": {"host": "localhost", "port": 9090}})

    def test_preserves_unoverridden_nested_levels(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 3}}}
        self.assertEqual(merge_config(base, override), {"a": {"b": {"c": 3, "d": 2}}})


if __name__ == "__main__":
    unittest.main()

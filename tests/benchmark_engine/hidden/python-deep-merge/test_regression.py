import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from config import merge_config


class ExistingBehavior(unittest.TestCase):
    def test_lists_are_replaced_instead_of_merged(self):
        self.assertEqual(merge_config({"items": [1, 2]}, {"items": [3]}), {"items": [3]})

    def test_inputs_are_not_mutated(self):
        base = {"feature": {"enabled": True}}
        override = {"feature": {"name": "native"}}
        before_base = {"feature": {"enabled": True}}
        before_override = {"feature": {"name": "native"}}
        merge_config(base, override)
        self.assertEqual(base, before_base)
        self.assertEqual(override, before_override)


if __name__ == "__main__":
    unittest.main()

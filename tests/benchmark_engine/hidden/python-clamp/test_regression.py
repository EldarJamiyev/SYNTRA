import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from math_utils import clamp


class ExistingBehavior(unittest.TestCase):
    def test_values_inside_the_range_are_unchanged(self):
        self.assertEqual(clamp(4.5, 0, 10), 4.5)

    def test_value_at_midpoint_is_unchanged(self):
        self.assertEqual(clamp(5, 0, 10), 5)


if __name__ == "__main__":
    unittest.main()

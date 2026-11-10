import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from math_utils import clamp


class TargetBehavior(unittest.TestCase):
    def test_values_below_minimum_are_clamped(self):
        self.assertEqual(clamp(-5, 0, 10), 0)

    def test_values_above_maximum_are_clamped(self):
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_boundary_values_are_inclusive(self):
        self.assertEqual(clamp(0, 0, 10), 0)
        self.assertEqual(clamp(10, 0, 10), 10)


if __name__ == "__main__":
    unittest.main()

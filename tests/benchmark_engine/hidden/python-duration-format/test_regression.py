import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from duration import format_duration


class ExistingBehavior(unittest.TestCase):
    def test_zero_is_rendered_explicitly(self):
        self.assertEqual(format_duration(0), "0s")

    def test_single_seconds_unit_is_compact(self):
        self.assertEqual(format_duration(7), "7s")


if __name__ == "__main__":
    unittest.main()

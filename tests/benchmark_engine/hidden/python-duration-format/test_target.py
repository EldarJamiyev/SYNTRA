import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from duration import format_duration


class TargetBehavior(unittest.TestCase):
    def test_formats_hours_minutes_and_seconds(self):
        self.assertEqual(format_duration(3663), "1h 1m 3s")

    def test_omits_zero_units_between_nonzero_units(self):
        self.assertEqual(format_duration(3605), "1h 5s")

    def test_formats_multiple_minutes(self):
        self.assertEqual(format_duration(125), "2m 5s")


if __name__ == "__main__":
    unittest.main()

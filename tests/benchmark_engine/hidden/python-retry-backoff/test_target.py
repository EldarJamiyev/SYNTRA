import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from backoff import calculate_backoff


class TargetBehavior(unittest.TestCase):
    def test_delay_doubles_for_each_attempt(self):
        self.assertEqual(calculate_backoff(2, 0.5, 30), 1.0)
        self.assertEqual(calculate_backoff(3, 0.5, 30), 2.0)

    def test_delay_is_capped_at_maximum(self):
        self.assertEqual(calculate_backoff(10, 1.0, 12.0), 12.0)

    def test_very_large_attempt_does_not_overflow(self):
        self.assertEqual(calculate_backoff(100000, 1.0, 8.0), 8.0)


if __name__ == "__main__":
    unittest.main()

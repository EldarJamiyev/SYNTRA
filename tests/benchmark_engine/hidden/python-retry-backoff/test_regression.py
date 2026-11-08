import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from backoff import calculate_backoff


class ExistingBehavior(unittest.TestCase):
    def test_first_attempt_uses_the_base_delay(self):
        self.assertEqual(calculate_backoff(1, 0.25, 10.0), 0.25)

    def test_non_positive_attempt_returns_zero(self):
        self.assertEqual(calculate_backoff(0, 1.0, 10.0), 0.0)


if __name__ == "__main__":
    unittest.main()

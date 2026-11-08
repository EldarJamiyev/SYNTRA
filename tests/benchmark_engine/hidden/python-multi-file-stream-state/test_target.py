import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from state import parse_stream_state


class MultiFileStreamStateTests(unittest.TestCase):
    def test_clamps_numeric_samples_through_the_shared_limits_module(self):
        result = parse_stream_state('{"state":"running","samples":[-8,3.5,20]}', 0, 10)
        self.assertEqual(result, {"state": "running", "samples": [0, 3.5, 10]})

    def test_rejects_malformed_or_incomplete_state_payloads(self):
        self.assertIsNone(parse_stream_state("{broken", 0, 10))
        self.assertIsNone(parse_stream_state('{"samples":[1]}', 0, 10))
        self.assertIsNone(parse_stream_state('{"state":"running","samples":"1,2"}', 0, 10))

    def test_rejects_invalid_bounds_without_returning_unclamped_data(self):
        self.assertIsNone(parse_stream_state('{"state":"running","samples":[1]}', 10, 0))


if __name__ == "__main__":
    unittest.main()

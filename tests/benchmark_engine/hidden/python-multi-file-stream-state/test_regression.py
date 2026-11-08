import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from state import parse_stream_state


class StreamStateRegressionTests(unittest.TestCase):
    def test_valid_in_range_samples_and_empty_streams_are_preserved(self):
        self.assertEqual(
            parse_stream_state('{"state":"idle","samples":[2,5,8]}', 0, 10),
            {"state": "idle", "samples": [2, 5, 8]},
        )
        self.assertEqual(
            parse_stream_state('{"state":"idle","samples":[]}', 0, 10),
            {"state": "idle", "samples": []},
        )


if __name__ == "__main__":
    unittest.main()

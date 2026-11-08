import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from chunks import chunk_array


class TargetBehavior(unittest.TestCase):
    def test_splits_values_into_consecutive_fixed_size_chunks(self):
        self.assertEqual(chunk_array([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_exact_multiple_has_no_empty_trailing_chunk(self):
        self.assertEqual(chunk_array([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_size_one_returns_single_item_chunks(self):
        self.assertEqual(chunk_array(["a", "b"], 1), [["a"], ["b"]])


if __name__ == "__main__":
    unittest.main()

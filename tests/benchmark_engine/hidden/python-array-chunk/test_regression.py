import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from chunks import chunk_array


class ExistingBehavior(unittest.TestCase):
    def test_empty_input_remains_empty(self):
        self.assertEqual(chunk_array([], 3), [])

    def test_input_sequence_is_not_mutated(self):
        values = [1, 2, 3]
        snapshot = list(values)
        chunk_array(values, 2)
        self.assertEqual(values, snapshot)


if __name__ == "__main__":
    unittest.main()

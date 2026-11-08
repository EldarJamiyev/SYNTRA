import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from headers import lookup_header


class ExistingBehavior(unittest.TestCase):
    def test_exact_case_match_remains_compatible(self):
        self.assertEqual(lookup_header({"Accept": "text/plain"}, "Accept"), "text/plain")

    def test_does_not_mutate_the_input_mapping(self):
        headers = {"Content-Type": "application/json"}
        snapshot = dict(headers)
        lookup_header(headers, "content-type")
        self.assertEqual(headers, snapshot)


if __name__ == "__main__":
    unittest.main()

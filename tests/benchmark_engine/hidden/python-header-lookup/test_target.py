import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.environ["SYNTRA_BENCHMARK_WORKSPACE"], "src"))
from headers import lookup_header


class TargetBehavior(unittest.TestCase):
    def test_matches_header_names_without_case_sensitivity(self):
        self.assertEqual(lookup_header({"Content-Type": "application/json"}, "content-type"), "application/json")

    def test_uses_casefold_and_first_matching_mapping_entry(self):
        headers = {"X-TRACE": "first", "x-trace": "second"}
        self.assertEqual(lookup_header(headers, "X-Trace"), "first")

    def test_returns_default_when_header_is_missing(self):
        self.assertEqual(lookup_header({"Accept": "application/json"}, "authorization", "missing"), "missing")


if __name__ == "__main__":
    unittest.main()

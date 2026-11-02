import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from duration import format_duration


class FormatDurationTests(unittest.TestCase):
    def test_formats_minutes_and_seconds(self):
        self.assertEqual(format_duration(0), "00:00")
        self.assertEqual(format_duration(9), "00:09")
        self.assertEqual(format_duration(61), "01:01")
        self.assertEqual(format_duration(3599), "59:59")

    def test_formats_hours_without_padding_the_hour(self):
        self.assertEqual(format_duration(3600), "1:00:00")
        self.assertEqual(format_duration(3661), "1:01:01")
        self.assertEqual(format_duration(36000), "10:00:00")

    def test_rejects_negative_seconds(self):
        with self.assertRaises(ValueError):
            format_duration(-1)

    def test_rejects_non_integer_and_boolean_values(self):
        for value in (1.5, "60", None, True, False):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    format_duration(value)


if __name__ == "__main__":
    unittest.main()

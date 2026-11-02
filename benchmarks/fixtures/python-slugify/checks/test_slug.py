import unittest
from src.slug import slugify

class SlugifyTests(unittest.TestCase):
    def test_normalizes_spaces_and_punctuation(self):
        self.assertEqual(slugify("Hello, SYNTRA Engineering!"), "hello-syntra-engineering")

    def test_collapses_separators(self):
        self.assertEqual(slugify("  Multi   Step---Agent  "), "multi-step-agent")

if __name__ == "__main__":
    unittest.main()

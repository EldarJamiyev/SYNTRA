from __future__ import annotations

import unittest

from syntra_ai.tokenizer import ByteBPETokenizer, FIRST_MERGE_ID


class TokenizerTrainingContractTests(unittest.TestCase):
    def test_vocab_size_requires_a_strict_integer_within_supported_range(self):
        for value in (258.0, True, "258", 257, 8193):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "vocab_size"):
                    ByteBPETokenizer.train([], vocab_size=value)

    def test_training_byte_budget_requires_a_strict_integer_and_minimum(self):
        for value in (1024.0, True, "1024", 1023, 0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_training_bytes"):
                    ByteBPETokenizer.train([], max_training_bytes=value)

    def test_minimum_pair_frequency_requires_a_positive_strict_integer(self):
        for value in (2.0, True, "2", 0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "min_pair_frequency"):
                    ByteBPETokenizer.train([], min_pair_frequency=value)

    def test_documents_must_be_an_iterable_of_text_not_a_scalar_or_invalid_item(self):
        invalid_sources = ("one document", b"one document", None, [None], [1])
        for documents in invalid_sources:
            with self.subTest(documents=documents):
                with self.assertRaisesRegex(ValueError, "documents|document"):
                    ByteBPETokenizer.train(documents)

    def test_unencodable_unicode_document_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            ByteBPETokenizer.train([chr(0xD800)])

    def test_bpe_training_does_not_learn_pairs_across_document_boundaries(self):
        tokenizer = ByteBPETokenizer.train(
            ["aaa", "bbb"],
            vocab_size=261,
            max_training_bytes=1024,
            min_pair_frequency=1,
        )
        # The first two merges are deterministic within-document repeats. The old
        # concatenated sample learned (a, merged-bb), which crosses the boundary.
        self.assertEqual(tokenizer.merges[:2], ((ord("a"), ord("a")), (ord("b"), ord("b"))))
        self.assertIn((FIRST_MERGE_ID, ord("a")), tokenizer.merges)
        self.assertNotIn((ord("a"), FIRST_MERGE_ID + 1), tokenizer.merges)

    def test_empty_iterable_is_valid_and_learns_no_merges(self):
        tokenizer = ByteBPETokenizer.train(iter(()), vocab_size=258)
        self.assertEqual(tokenizer.merges, ())
        self.assertEqual(tokenizer.vocab_size, 258)


if __name__ == "__main__":
    unittest.main()

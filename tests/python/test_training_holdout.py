from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def _documents() -> list[str]:
    return [
        (
            f"def handler_{index}(value):\n"
            f"    return value + {index}\n"
            f"# independent source document {index}\n"
        ) * 5
        for index in range(6)
    ]


class DocumentPartitionTests(unittest.TestCase):
    def test_exact_copies_are_deduplicated_and_never_cross_splits(self):
        from syntra_ai.tokenizer import ByteBPETokenizer
        from syntra_ai.training import (
            _encode_document_partition,
            _partition_source_documents,
        )

        documents = _documents()
        partition = _partition_source_documents(
            documents + [documents[2], documents[2]],
            max_tokens=4096,
        )
        split = _encode_document_partition(
            partition,
            ByteBPETokenizer.empty(),
            max_tokens=4096,
            seq_len=16,
        )

        training_hashes = set(split.training_document_hashes)
        validation_hashes = set(split.validation_document_hashes)
        self.assertTrue(training_hashes)
        self.assertTrue(validation_hashes)
        self.assertFalse(training_hashes.intersection(validation_hashes))
        self.assertEqual(len(training_hashes) + len(validation_hashes), len(documents))
        self.assertEqual(split.duplicate_documents_excluded, 2)
        self.assertLessEqual(
            len(split.training_tokens) + len(split.validation_tokens),
            4096,
        )
        self.assertEqual(
            split.training_tokens.count(256),
            len(split.training_document_hashes),
        )
        self.assertEqual(
            split.validation_tokens.count(256),
            len(split.validation_document_hashes),
        )
        self.assertEqual(
            split.training_tokens.count(257),
            len(split.training_document_hashes),
        )
        self.assertEqual(
            split.validation_tokens.count(257),
            len(split.validation_document_hashes),
        )

    def test_partition_and_encoded_streams_are_stable_when_input_order_changes(self):
        from syntra_ai.tokenizer import ByteBPETokenizer
        from syntra_ai.training import (
            _encode_document_partition,
            _partition_source_documents,
        )

        documents = _documents()
        first = _partition_source_documents(documents, max_tokens=4096)
        second = _partition_source_documents(list(reversed(documents)), max_tokens=4096)
        tokenizer_first = ByteBPETokenizer.train(first.training_documents, vocab_size=300)
        tokenizer_second = ByteBPETokenizer.train(second.training_documents, vocab_size=300)
        split_first = _encode_document_partition(
            first, tokenizer_first, max_tokens=4096, seq_len=16
        )
        split_second = _encode_document_partition(
            second, tokenizer_second, max_tokens=4096, seq_len=16
        )

        self.assertEqual(first.training_document_hashes, second.training_document_hashes)
        self.assertEqual(first.validation_document_hashes, second.validation_document_hashes)
        self.assertEqual(tokenizer_first.to_dict(), tokenizer_second.to_dict())
        self.assertEqual(split_first.training_tokens, split_second.training_tokens)
        self.assertEqual(split_first.validation_tokens, split_second.validation_tokens)

    def test_exact_duplicate_only_corpus_fails_closed(self):
        from syntra_ai.training import _partition_source_documents

        document = "one independent source file\n" * 10
        with self.assertRaisesRegex(ValueError, "at least two independent source documents"):
            _partition_source_documents([document, document], max_tokens=1024)

    def test_context_must_fit_inside_both_independent_partitions(self):
        from syntra_ai.tokenizer import ByteBPETokenizer
        from syntra_ai.training import (
            _encode_document_partition,
            _partition_source_documents,
        )

        documents = [
            "first short document\n" * 2,
            "second short document\n" * 2,
        ]
        partition = _partition_source_documents(documents, max_tokens=1024)
        with self.assertRaisesRegex(ValueError, "not contain enough independent tokens"):
            _encode_document_partition(
                partition,
                ByteBPETokenizer.empty(),
                max_tokens=1024,
                seq_len=128,
            )

    def test_corpus_loader_never_truncates_a_document_to_fill_the_remaining_budget(self):
        from syntra_ai.data import load_corpus

        with tempfile.TemporaryDirectory(prefix="syntra-document-budget-") as temp:
            root = Path(temp) / "corpus"
            root.mkdir()
            (root / "a.txt").write_text("a" * 40, encoding="utf-8")
            (root / "b.txt").write_text("b" * 1000, encoding="utf-8")
            (root / "c.txt").write_text("c" * 40, encoding="utf-8")
            with patch.dict(
                os.environ,
                {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)},
            ):
                corpus = load_corpus(
                    [root],
                    max_tokens=1024,
                    include_tokens=False,
                )

        self.assertEqual(corpus.documents, ["a" * 40, "c" * 40])
        self.assertEqual(corpus.token_count, 84)
        self.assertEqual(corpus.skipped_files, 1)
        self.assertEqual(corpus.token_ids, [])

    def test_exact_duplicate_files_do_not_consume_the_corpus_token_budget(self):
        from syntra_ai.data import load_corpus

        repeated = "same independent document\n" * 25
        additional = "another independent document\n" * 20
        with tempfile.TemporaryDirectory(prefix="syntra-document-dedup-budget-") as temp:
            root = Path(temp) / "corpus"
            root.mkdir()
            (root / "a.txt").write_text(repeated, encoding="utf-8")
            (root / "b.txt").write_text(repeated, encoding="utf-8")
            (root / "c.txt").write_text(additional, encoding="utf-8")
            token_budget = len(repeated.encode("utf-8")) + len(additional.encode("utf-8")) + 4
            with patch.dict(
                os.environ,
                {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)},
            ):
                corpus = load_corpus(
                    [root],
                    max_tokens=token_budget,
                    include_tokens=False,
                )

        self.assertEqual(corpus.documents, [repeated, repeated, additional])
        self.assertEqual(corpus.token_count, token_budget)
        self.assertEqual(len(corpus.files), 3)
        from syntra_ai.training import _partition_source_documents
        partition = _partition_source_documents(corpus.documents, max_tokens=token_budget)
        self.assertEqual(partition.duplicate_documents_excluded, 1)
        self.assertEqual(
            len(partition.training_documents) + len(partition.validation_documents),
            2,
        )

    def test_directory_traversal_order_is_stable_under_a_bounded_file_count(self):
        from syntra_ai.data import load_corpus

        with tempfile.TemporaryDirectory(prefix="syntra-document-order-") as temp:
            root = Path(temp) / "corpus"
            (root / "z-directory").mkdir(parents=True)
            (root / "a-directory").mkdir(parents=True)
            (root / "z-directory" / "module.py").write_text("z" * 80, encoding="utf-8")
            (root / "a-directory" / "module.py").write_text("a" * 80, encoding="utf-8")
            with patch.dict(
                os.environ,
                {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)},
            ):
                corpus = load_corpus(
                    [root],
                    max_files=1,
                    max_tokens=1024,
                    include_tokens=False,
                )

        self.assertEqual(corpus.documents, ["a" * 80])


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run native training tests.")
class NativeTrainingHoldoutIntegrationTests(unittest.TestCase):
    def test_training_fits_tokenizer_only_on_training_documents_and_records_holdout(self):
        from syntra_ai.checkpoint_io import load_torch_checkpoint_snapshot
        from syntra_ai.config import ModelConfig
        from syntra_ai.runtime import status
        from syntra_ai.tokenizer import ByteBPETokenizer
        from syntra_ai.training import _partition_source_documents, train

        documents = _documents()
        with tempfile.TemporaryDirectory(prefix="syntra-native-holdout-training-") as temp:
            root = Path(temp) / "corpus"
            root.mkdir()
            for index, document in enumerate(documents):
                (root / f"source-{index}.py").write_text(document, encoding="utf-8")
            (root / "source-duplicate.py").write_text(documents[1], encoding="utf-8")
            checkpoint = Path(temp) / "native-checkpoint.pt"

            partition = _partition_source_documents(
                documents + [documents[1]],
                max_tokens=4096,
            )
            expected_tokenizer = ByteBPETokenizer.train(
                partition.training_documents,
                vocab_size=300,
            )
            config = ModelConfig(
                vocab_size=300,
                seq_len=16,
                n_layer=1,
                n_head=1,
                n_embd=32,
                dropout=0.0,
            )

            with patch.dict(
                os.environ,
                {
                    "SYNTRA_ALLOWED_CORPUS_ROOTS": str(root),
                    "SYNTRA_CHECKPOINT": str(checkpoint),
                },
            ):
                result = train(
                    roots=[str(root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=config,
                    max_tokens=4096,
                    device_name="cpu",
                    seed=17,
                )
                saved = load_torch_checkpoint_snapshot(checkpoint)
                current_status = status()
                metadata = json.loads(
                    checkpoint.with_suffix(checkpoint.suffix + ".json").read_text(encoding="utf-8")
                )

            self.assertEqual(saved["tokenizer"], expected_tokenizer.to_dict())
            self.assertEqual(saved["training_mode"], "causal-source-pretraining")
            self.assertEqual(saved["step"], 1)
            self.assertEqual(saved["validation_strategy"], "sha256-grouped-document-holdout-v1")
            self.assertEqual(saved["training_documents"], len(partition.training_documents))
            self.assertEqual(saved["validation_documents"], len(partition.validation_documents))
            self.assertEqual(saved["duplicate_documents_excluded"], 1)
            self.assertEqual(
                saved["training_documents"] + saved["validation_documents"],
                len(documents),
            )
            self.assertEqual(metadata["validation_strategy"], saved["validation_strategy"])
            self.assertEqual(metadata["duplicate_documents_excluded"], 1)
            self.assertEqual(current_status["validation_strategy"], saved["validation_strategy"])
            self.assertEqual(current_status["training_documents"], saved["training_documents"])
            self.assertEqual(current_status["validation_documents"], saved["validation_documents"])
            self.assertEqual(result["validation_strategy"], saved["validation_strategy"])
            self.assertTrue(torch.isfinite(torch.tensor(saved["validation_loss"])).item())

    def test_old_token_offset_checkpoint_fails_resume_with_an_explicit_migration_error(self):
        from syntra_ai.checkpoint_io import load_torch_checkpoint_snapshot
        from syntra_ai.config import ModelConfig
        from syntra_ai.training import train

        documents = _documents()
        with tempfile.TemporaryDirectory(prefix="syntra-old-holdout-resume-") as temp:
            root = Path(temp) / "corpus"
            root.mkdir()
            for index, document in enumerate(documents):
                (root / f"source-{index}.py").write_text(document, encoding="utf-8")
            checkpoint = Path(temp) / "native-checkpoint.pt"
            config = ModelConfig(
                vocab_size=300,
                seq_len=16,
                n_layer=1,
                n_head=1,
                n_embd=32,
                dropout=0.0,
            )
            env = {
                "SYNTRA_ALLOWED_CORPUS_ROOTS": str(root),
                "SYNTRA_CHECKPOINT": str(checkpoint),
            }
            with patch.dict(os.environ, env):
                train(
                    roots=[str(root)],
                    checkpoint=checkpoint,
                    steps=1,
                    batch_size=1,
                    config=config,
                    max_tokens=4096,
                    device_name="cpu",
                    seed=23,
                )
                saved = load_torch_checkpoint_snapshot(checkpoint)
                from syntra_ai.data import load_corpus
                corpus = load_corpus(
                    [str(root)],
                    max_tokens=4096,
                    include_tokens=False,
                )
                legacy_fingerprint_payload = {
                    "schema": "syntra-causal-source-corpus-v1",
                    "source_sha256": corpus.source_sha256,
                    "max_tokens": 4096,
                    "config": saved["config"],
                    "tokenizer": saved["tokenizer"],
                }
                legacy_fingerprint = sha256(
                    json.dumps(
                        legacy_fingerprint_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                saved["training_data_sha256"] = legacy_fingerprint
                torch.save(saved, checkpoint)
                with self.assertRaisesRegex(ValueError, "token-offset validation checkpoints are not compatible"):
                    train(
                        roots=[str(root)],
                        checkpoint=checkpoint,
                        steps=1,
                        batch_size=1,
                        config=config,
                        max_tokens=4096,
                        device_name="cpu",
                        seed=23,
                        resume=True,
                    )


if __name__ == "__main__":
    unittest.main()

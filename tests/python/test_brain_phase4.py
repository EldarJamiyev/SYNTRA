import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from brain.corpus import build_instruction_corpus, normalize_source_text
from brain.train import _pack_instruction_records, _split_instruction_records
from brain.tokenizer import BytePairTokenizer


class PhaseFourCorpusTests(unittest.TestCase):
    def test_normalization_removes_bom_line_endings_html_comments_and_markdown_boilerplate(self):
        source = (
            "\ufeff# Guide\r\n"
            "<!-- generated content -->\r\n"
            "## Table of Contents\r\n"
            "- [Install](#install)\r\n"
            "![build](https://shields.io/badge/build-passing)\r\n"
            "## Install\r\n"
            "Run the installer.\r\n"
        )
        normalized = normalize_source_text(source, ".md")
        self.assertTrue(normalized.startswith("# Guide\n"))
        self.assertIn("## Install\nRun the installer.", normalized)
        self.assertNotIn("generated content", normalized)
        self.assertNotIn("Table of Contents", normalized)
        self.assertNotIn("shields.io", normalized)
        self.assertNotIn("\r", normalized)

    def test_corpus_scanner_collects_only_supported_clean_files(self):
        with tempfile.TemporaryDirectory(prefix="syntra-phase4-corpus-") as temp:
            root = Path(temp) / "approved"
            (root / "src").mkdir(parents=True)
            (root / "node_modules" / "dep").mkdir(parents=True)
            (root / "dist").mkdir()
            (root / ".hidden").mkdir()
            (root / "secrets").mkdir()
            (root / "src" / "clamp.py").write_text(
                "def clamp(value, minimum, maximum):\n"
                "    return max(minimum, min(value, maximum))\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "# Project guide\n\nUse the clamp helper from the source module.\n",
                encoding="utf-8",
            )
            (root / "package.json").write_text('{"name":"example","private":true}\n', encoding="utf-8")
            (root / ".env").write_text("TOKEN=never-ingest\n", encoding="utf-8")
            (root / "node_modules" / "dep" / "index.js").write_text("dependency noise " * 30, encoding="utf-8")
            (root / "dist" / "compiled.js").write_text("compiled noise " * 30, encoding="utf-8")
            (root / ".hidden" / "private.py").write_text("hidden noise " * 30, encoding="utf-8")
            (root / "secrets" / "credentials.json").write_text('{"token":"not-for-training"}', encoding="utf-8")
            old = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(root)
            try:
                corpus = build_instruction_corpus([root])
            finally:
                if old is None:
                    os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
                else:
                    os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = old

            self.assertEqual(len(corpus.files), 3)
            self.assertGreaterEqual(len(corpus.records), 4)
            combined = "\n".join(item["prompt"] + item["completion"] for item in corpus.records)
            self.assertIn("clamp", combined)
            self.assertIn("Project guide", combined)
            self.assertNotIn("never-ingest", combined)
            self.assertNotIn("dependency noise", combined)
            self.assertNotIn("compiled noise", combined)
            self.assertNotIn("hidden noise", combined)
            self.assertNotIn("not-for-training", combined)
            self.assertEqual(len(corpus.sha256), 64)

    def test_corpus_rejects_unapproved_external_root(self):
        with tempfile.TemporaryDirectory(prefix="syntra-phase4-unapproved-") as temp:
            root = Path(temp) / "outside"
            root.mkdir()
            old = os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
            try:
                with self.assertRaisesRegex(ValueError, "outside"):
                    build_instruction_corpus([root])
            finally:
                if old is not None:
                    os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = old

    def test_total_byte_limit_is_enforced_not_silently_skipped(self):
        with tempfile.TemporaryDirectory(prefix="syntra-phase4-byte-limit-") as temp:
            root = Path(temp) / "approved"
            root.mkdir()
            (root / "one.py").write_text("def one():\n    return '" + ("x" * 1500) + "'\n", encoding="utf-8")
            old = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(root)
            try:
                with self.assertRaisesRegex(ValueError, "total source limit"):
                    build_instruction_corpus([root], max_total_bytes=1024, max_file_bytes=4096, chunk_chars=200)
            finally:
                if old is None:
                    os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
                else:
                    os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = old


@unittest.skipIf(torch is None, "Install the pinned PyTorch requirements to run instruction-training tests.")
class PhaseFourInstructionTrainingTests(unittest.TestCase):
    def test_source_grouped_split_and_prompt_masking(self):
        records = [
            {"source": "a.py", "prompt": "instruction A", "completion": "answer A"},
            {"source": "a.py", "prompt": "instruction A2", "completion": "answer A2"},
            {"source": "b.py", "prompt": "instruction B", "completion": "answer B"},
        ]
        train, validation = _split_instruction_records(records)
        self.assertEqual({item["source"] for item in train}.isdisjoint(
            {item["source"] for item in validation}
        ), True)
        tokenizer = BytePairTokenizer.train_from_text(
            "\n".join(item["prompt"] + item["completion"] for item in records),
            vocab_size=270,
            max_training_bytes=4096,
        )
        inputs, targets, supervised = _pack_instruction_records(records, tokenizer, 64)
        self.assertEqual(tuple(inputs.shape), tuple(targets.shape))
        self.assertEqual(inputs.shape[1], 64)
        self.assertGreater(supervised, 0)
        self.assertTrue(torch.all(targets[:, :1] == -100).item())
        self.assertTrue(torch.any(targets != -100).item())
        self.assertTrue(torch.all(inputs >= 0).item())

    def test_train_instruction_mode_with_gradient_accumulation_and_checkpoint(self):
        from brain.model import TransformerConfig
        from brain.train import train
        from brain.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory(prefix="syntra-phase4-train-") as temp:
            root = Path(temp) / "approved"
            root.mkdir()
            source = root / "clamp.py"
            source.write_text(
                "def clamp(value, minimum, maximum):\n"
                "    return max(minimum, min(value, maximum))\n\n"
                "class Range:\n"
                "    def contains(self, value):\n"
                "        return 0 <= value <= 10\n",
                encoding="utf-8",
            )
            old = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")
            os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = str(root)
            try:
                checkpoint = Path(temp) / "model.pt"
                result = train(
                    [root],
                    checkpoint_path=checkpoint,
                    config=TransformerConfig(vocab_size=270, block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0),
                    epochs=1,
                    batch_size=1,
                    tokenizer_vocab_size=270,
                    tokenizer_bytes=4096,
                    max_dataset_bytes=1024 * 1024,
                    evaluation_batches=1,
                    save_every_steps=1,
                    device_name="cpu",
                    seed=7,
                    gradient_accumulation_steps=2,
                    warmup_steps=0,
                    instruction_mode=True,
                    run_benchmarks=False,
                )
                saved = load_checkpoint(checkpoint)
            finally:
                if old is None:
                    os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
                else:
                    os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = old

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["instruction_mode"])
        self.assertGreater(result["training_tokens"], 0)
        self.assertGreater(result["validation_tokens"], 0)
        self.assertEqual(result["gradient_accumulation_steps"], 2)
        self.assertTrue(saved["instruction_mode"])
        self.assertEqual(saved["epoch"], 1)


if __name__ == "__main__":
    unittest.main()

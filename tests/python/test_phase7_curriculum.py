import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PolyglotCurriculumTests(unittest.TestCase):
    def test_all_four_non_python_languages_receive_quality_and_fingerprints(self):
        from syntra_ai.curriculum import _module_metadata

        samples = {
            "module.mjs": (
                "/** Normalize a user-supplied label. */\n"
                "export function normalizeLabel(value) {\n"
                "  const normalized = value.trim();\n"
                "  if (normalized.length > 1) {\n"
                "    return normalized.toLowerCase();\n"
                "  }\n"
                "  return normalized;\n"
                "}\n"
            ),
            "contract.ts": (
                "/** Account data shared by API handlers. */\n"
                "export interface Account {\n"
                "  id: string;\n"
                "  email: string;\n"
                "  active: boolean;\n"
                "}\n"
                "export function displayEmail(account: Account): string {\n"
                "  return account.email.trim();\n"
                "}\n"
            ),
            "clamp.go": (
                "package clamp\n"
                "// Clamp bounds an integer to the requested interval.\n"
                "func Clamp(value, minimum, maximum int) int {\n"
                "  if value < minimum {\n"
                "    return minimum\n"
                "  }\n"
                "  if value > maximum {\n"
                "    return maximum\n"
                "  }\n"
                "  return value\n"
                "}\n"
            ),
            "slice.rs": (
                "/// Return a clamped value without changing the input.\n"
                "pub fn clamp(value: i32, minimum: i32, maximum: i32) -> i32 {\n"
                "    if value < minimum { return minimum; }\n"
                "    if value > maximum { return maximum; }\n"
                "    value\n"
                "}\n"
            ),
        }
        for path, source in samples.items():
            with self.subTest(path=path):
                metadata = _module_metadata(path, source)
                scored = [
                    item for item in metadata["symbols"]
                    if isinstance(item.get("quality"), dict)
                ]
                self.assertTrue(scored, metadata)
                self.assertTrue(all(item.get("structural_fingerprint") for item in scored))
                self.assertTrue(all(item["quality_score"] == item["quality"]["score"] for item in scored))
                self.assertTrue(any(item["quality_score"] >= 1 for item in scored))

    def test_polyglot_quality_filters_trivial_accessors_and_empty_blocks(self):
        from syntra_ai.curriculum import _module_metadata

        source = (
            "export function getName(record) {\n"
            "  return record.name;\n"
            "}\n"
            "export function setName(record, value) {\n"
            "  record.name = value;\n"
            "}\n"
            "export function notImplemented(value) {\n"
            "  return value;\n"
            "}\n"
        )
        metadata = _module_metadata("accessors.mjs", source)
        by_name = {item["name"]: item for item in metadata["symbols"]}
        self.assertTrue(by_name["getName"]["quality"]["boilerplate"])
        self.assertTrue(by_name["setName"]["quality"]["boilerplate"])
        self.assertTrue(by_name["notImplemented"]["quality"]["boilerplate"])

    def test_polyglot_fingerprint_normalizes_identifiers_but_keeps_behavior(self):
        from syntra_ai.polyglot import _polyglot_fingerprint

        first = (
            "function normalize(input) {\n"
            "  const local = input.trim();\n"
            "  if (local.length > 1) return local.toLowerCase();\n"
            "  return local;\n"
            "}\n"
        )
        equivalent = (
            "function format(value) {\n"
            "  const result = value.trim();\n"
            "  if (result.length > 1) return result.toLowerCase();\n"
            "  return result;\n"
            "}\n"
        )
        changed_operator = equivalent.replace("length > 1", "length >= 1")
        a = _polyglot_fingerprint("javascript", "normalize", "normalize(input)", first)
        b = _polyglot_fingerprint("javascript", "format", "format(value)", equivalent)
        c = _polyglot_fingerprint("javascript", "format", "format(value)", changed_operator)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

        # Whitespace must not change token adjacency: API calls and member names
        # remain semantic identifiers even when source formatting adds spaces.
        spaced = (
            "function format ( value ) {\n"
            "  const result = value . trim ( ) ;\n"
            "  if ( result . length > 1 ) return result . toLowerCase ( ) ;\n"
            "  return result ;\n"
            "}\n"
        )
        spaced_fingerprint = _polyglot_fingerprint(
            "javascript", "format", "format ( value )", spaced
        )
        self.assertEqual(b, spaced_fingerprint)

        go_first = "func (cache *Cache) Get(value int) int"
        go_second = "func (store *Store) Lookup(count int) int"
        go_body_first = "func Get(value int) int { if value > 0 { return cache.base + value }; return 0 }"
        go_body_second = "func Lookup(count int) int { if count > 0 { return store.base + count }; return 0 }"
        self.assertEqual(
            _polyglot_fingerprint("go", "Get", go_first, go_body_first),
            _polyglot_fingerprint("go", "Lookup", go_second, go_body_second),
        )

    def test_repository_curriculum_deduplicates_polyglot_declarations_deterministically(self):
        from syntra_ai.curriculum import synthesize_repository_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-polyglot-dedup-") as temp:
            root = Path(temp)
            (root / "alpha.mjs").write_text(
                "/** Normalize a user label. */\n"
                "export function normalize(input) {\n"
                "  const local = input.trim();\n"
                "  if (local.length > 1) return local.toLowerCase();\n"
                "  return local;\n"
                "}\n",
                encoding="utf-8",
            )
            (root / "beta.mjs").write_text(
                "/** Format an incoming value. */\n"
                "export function format(value) {\n"
                "  const result = value.trim();\n"
                "  if (result.length > 1) return result.toLowerCase();\n"
                "  return result;\n"
                "}\n",
                encoding="utf-8",
            )
            first_output = root / "first.jsonl"
            second_output = root / "second.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                first = synthesize_repository_curriculum(
                    root=root, output_path=first_output, max_examples=20,
                    languages=("javascript",),
                )
                second = synthesize_repository_curriculum(
                    root=root, output_path=second_output, max_examples=20,
                    languages=("javascript",),
                )
            self.assertEqual(first["quality_statistics"]["near_duplicate_symbols_skipped"], 1)
            self.assertEqual(
                first_output.read_text(encoding="utf-8"),
                second_output.read_text(encoding="utf-8"),
            )
            self.assertEqual(first["examples_by_type"], second["examples_by_type"])
            self.assertEqual(
                first["quality_statistics"]["by_language"]["javascript"]["quality_scored"],
                2,
            )

    def test_language_repairs_are_exact_unique_and_comment_safe(self):
        from syntra_ai.polyglot import safe_repair_examples

        javascript = (
            "// typeof ignored == 'string'\n"
            "export function isText(value) {\n"
            "  return typeof value == \"string\";\n"
            "}\n"
            "const note = \"typeof other == 'boolean'\";\n"
        )
        js_repairs = safe_repair_examples("guards.mjs", javascript)
        self.assertEqual(len(js_repairs), 1)
        self.assertEqual(js_repairs[0]["before"], 'typeof value == "string"')
        self.assertEqual(js_repairs[0]["after"], 'typeof value === "string"')

        typescript = (
            "export function isNumeric(value: unknown): boolean {\n"
            "  return typeof value != \"number\";\n"
            "}\n"
        )
        ts_repairs = safe_repair_examples("guards.ts", typescript)
        self.assertEqual(ts_repairs[0]["after"], 'typeof value !== "number"')

        go = (
            "package lookup\n"
            "func Load(name string) (*Record, error) {\n"
            "  if err != nil {\n"
            "    return nil\n"
            "  }\n"
            "  return record, nil\n"
            "}\n"
        )
        go_repairs = safe_repair_examples("store.go", go)
        self.assertEqual(len(go_repairs), 1)
        self.assertEqual(go_repairs[0]["after"], "return nil, err")

        rust = (
            "pub fn item_at(values: &[String], index: usize) -> Option<&String> {\n"
            "    Some(&values[index])\n"
            "}\n"
        )
        rust_repairs = safe_repair_examples("slice.rs", rust)
        self.assertEqual(len(rust_repairs), 1)
        self.assertEqual(rust_repairs[0]["after"], "values.get(index)")

        already_checked = rust.replace(
            "    Some(&values[index])",
            "    if index >= values.len() { return None; }\n    Some(&values[index])",
        )
        self.assertEqual(safe_repair_examples("slice.rs", already_checked), [])

    def test_non_python_go_receiver_methods_are_bounded_as_members(self):
        from syntra_ai.curriculum import _extract_logical_symbols

        source = (
            "package cache\n"
            "func (cache *Cache) Get(key string) string {\n"
            "  return cache.items[key]\n"
            "}\n"
        )
        symbols = _extract_logical_symbols(source)
        method = next(item for item in symbols if item["name"] == "Get")
        self.assertTrue(method["is_member"])
        self.assertFalse(method["truncated"])
        self.assertIn("cache.items[key]", method["body"])

    def test_multifile_reasoning_curriculum_is_deterministic_and_source_grounded(self):
        from syntra_ai.instruction import load_instruction_examples
        from syntra_ai.model import EOS_ID
        from syntra_ai.reasoning_curriculum import synthesize_multifile_reasoning_curriculum
        from syntra_ai.tokenizer import ByteBPETokenizer

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-multifile-") as temp:
            root = Path(temp)
            source_dir = root / "src"
            source_dir.mkdir()
            implementation = (
                "/** Check whether a value is text. */\n"
                "export function isText(value) {\n"
                "  return typeof value == \"string\";\n"
                "}\n"
            )
            consumer = (
                "import { isText } from \"./guards.mjs\";\n"
                "/** Validate a value before accepting it. */\n"
                "export function validatePayload(value) {\n"
                "  if (typeof value != \"number\" && isText(value)) {\n"
                "    return true;\n"
                "  }\n"
                "  return false;\n"
                "}\n"
            )
            (source_dir / "guards.mjs").write_text(implementation, encoding="utf-8")
            (source_dir / "consumer.mjs").write_text(consumer, encoding="utf-8")
            first_output = root / "data" / "curricula" / "reasoning-one.jsonl"
            second_output = root / "data" / "curricula" / "reasoning-two.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                first = synthesize_multifile_reasoning_curriculum(
                    root=root, output_path=first_output, max_examples=10,
                    languages=("javascript",),
                )
                second = synthesize_multifile_reasoning_curriculum(
                    root=root, output_path=second_output, max_examples=10,
                    languages=("javascript",),
                )
                self.assertEqual(first["example_count"], 1, first)
                self.assertTrue(first_output.is_file())
                self.assertEqual(
                    first_output.read_text(encoding="utf-8"),
                    second_output.read_text(encoding="utf-8"),
                )
                record = json.loads(first_output.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(record["kind"], "multi_file_reasoning")
                self.assertIn("src/guards.mjs", record["prompt"])
                self.assertIn("src/consumer.mjs", record["prompt"])
                self.assertIn("<thought>", record["completion"])
                self.assertIn("</thought>", record["completion"])
                payload_text = record["completion"].split("</thought>\n", 1)[1]
                payload = json.loads(payload_text)
                self.assertEqual(payload["format"], "SYNTRA_AGENT_PATCH_V1")
                self.assertEqual(
                    {edit["path"] for edit in payload["edits"]},
                    {"src/guards.mjs", "src/consumer.mjs"},
                )
                self.assertEqual(len(payload["edits"]), 2)

                tokenizer = ByteBPETokenizer.empty()
                encoded = load_instruction_examples(
                    first_output, seq_len=1024, tokenizer=tokenizer
                )
                supervised = [
                    token
                    for item in encoded
                    for token in item["labels"]
                    if token != -100
                ]
                self.assertEqual(
                    supervised,
                    tokenizer.encode(record["completion"]) + [EOS_ID],
                )
                self.assertEqual(
                    encoded[0]["prompt_tokens"],
                    len(tokenizer.encode(record["prompt"])),
                )
                self.assertGreater(encoded[0]["completion_tokens"], 0)

    def test_relative_import_resolution_is_language_specific(self):
        from syntra_ai.reasoning_curriculum import _related_targets

        paths = {
            "src/guards.js",
            "src/guards.ts",
            "src/guards.mjs",
            "src/guards.mts",
        }
        source = 'import { normalize } from "./guards";\n'
        self.assertEqual(
            _related_targets("src/consumer.ts", source, "typescript", paths),
            ["src/guards.ts"],
        )
        self.assertEqual(
            _related_targets("src/consumer.mjs", source, "javascript", paths),
            ["src/guards.js"],
        )

    def test_python_multifile_curriculum_follows_relative_imports(self):
        from syntra_ai.reasoning_curriculum import synthesize_multifile_reasoning_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-python-pair-") as temp:
            root = Path(temp)
            package = root / "src" / "pkg"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "normalize.py").write_text(
                '"""Normalize values before use."""\n'
                "def normalize_value(value):\n"
                "    cleaned = value.strip()\n"
                "    if cleaned:\n"
                "        return cleaned\n"
                "    return value\n"
                "\n"
                "def is_missing(value):\n"
                "    return value == None\n",
                encoding="utf-8",
            )
            (package / "consumer.py").write_text(
                '"""Validate a normalized value."""\n'
                "from .normalize import normalize_value\n"
                "\n"
                "def validate_value(value):\n"
                "    normalized = normalize_value(value)\n"
                "    return normalized == None\n",
                encoding="utf-8",
            )
            output = root / "reasoning.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_multifile_reasoning_curriculum(
                    root=root,
                    output_path=output,
                    max_examples=8,
                    languages=("python",),
                )

            self.assertEqual(result["example_count"], 1, result)
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["kind"], "multi_file_reasoning")
            self.assertIn("src/pkg/normalize.py", record["prompt"])
            self.assertIn("src/pkg/consumer.py", record["prompt"])
            completion = record["completion"]
            self.assertIn("<thought>", completion)
            self.assertIn("</thought>", completion)
            payload = json.loads(completion.split("</thought>\n", 1)[1])
            self.assertEqual(payload["format"], "SYNTRA_AGENT_PATCH_V1")
            self.assertEqual(
                {edit["path"] for edit in payload["edits"]},
                {"src/pkg/normalize.py", "src/pkg/consumer.py"},
            )
            self.assertEqual(
                [edit["replace"] for edit in payload["edits"]],
                ["value is None", "normalized is None"],
            )

    def test_go_multifile_reasoning_pairs_same_package_test_and_implementation(self):
        from syntra_ai.reasoning_curriculum import synthesize_multifile_reasoning_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-go-pair-") as temp:
            root = Path(temp)
            source_dir = root / "store"
            source_dir.mkdir()
            (source_dir / "store.go").write_text(
                "package store\n"
                "type Record struct { Name string }\n"
                "func LoadRecord(key string) (*Record, error) {\n"
                "  record, err := readRecord(key)\n"
                "  if err != nil { return nil }\n"
                "  return record, nil\n"
                "}\n",
                encoding="utf-8",
            )
            (source_dir / "store_test.go").write_text(
                "package store\n"
                "import \"testing\"\n"
                "func loadRecordFixture(key string, err error) (*Record, error) {\n"
                "  if err != nil { return nil }\n"
                "  return nil, nil\n"
                "}\n"
                "func TestLoadRecord(t *testing.T) {\n"
                "  _, _ = LoadRecord(\"fixture\")\n"
                "}\n",
                encoding="utf-8",
            )
            output = root / "reasoning.jsonl"
            from syntra_ai.polyglot import safe_repair_examples
            from syntra_ai.curriculum import _module_metadata
            from syntra_ai.reasoning_curriculum import _related_targets, _referenced_export

            implementation_source = (source_dir / "store.go").read_text(encoding="utf-8")
            consumer_source = (source_dir / "store_test.go").read_text(encoding="utf-8")
            self.assertEqual(len(safe_repair_examples("store/store.go", implementation_source)), 1)
            self.assertEqual(len(safe_repair_examples("store/store_test.go", consumer_source)), 1)
            exports = _module_metadata("store/store.go", implementation_source)["exports"]
            self.assertIn("LoadRecord", exports, exports)
            self.assertEqual(
                _related_targets(
                    "store/store_test.go", consumer_source, "go",
                    {"store/store.go", "store/store_test.go"},
                ),
                ["store/store.go"],
            )
            self.assertEqual(_referenced_export(exports, consumer_source, "go"), "LoadRecord")

            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_multifile_reasoning_curriculum(
                    root=root, output_path=output, max_examples=8, languages=("go",)
                )
            self.assertEqual(result["example_count"], 1, result)
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            payload = json.loads(record["completion"].split("</thought>\n", 1)[1])
            self.assertEqual(payload["format"], "SYNTRA_AGENT_PATCH_V1")
            self.assertEqual(
                {edit["path"] for edit in payload["edits"]},
                {"store/store.go", "store/store_test.go"},
            )
            self.assertEqual(
                {edit["replace"] for edit in payload["edits"]},
                {"return nil, err"},
            )

    def test_rust_multifile_reasoning_follows_local_mod_and_use(self):
        from syntra_ai.reasoning_curriculum import synthesize_multifile_reasoning_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-rust-pair-") as temp:
            root = Path(temp)
            source_dir = root / "src"
            source_dir.mkdir()
            (source_dir / "store.rs").write_text(
                "pub fn item_at(values: &[String], index: usize) -> Option<&String> {\n"
                "    Some(&values[index])\n"
                "}\n",
                encoding="utf-8",
            )
            (source_dir / "lib.rs").write_text(
                "pub mod store;\n"
                "use crate::store::item_at;\n"
                "pub fn lookup(values: &[String], index: usize) -> Option<&String> {\n"
                "    Some(&values[index])\n"
                "}\n"
                "pub fn lookup_from_module(values: &[String], index: usize) -> Option<&String> {\n"
                "    item_at(values, index)\n"
                "}\n",
                encoding="utf-8",
            )
            output = root / "reasoning.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_multifile_reasoning_curriculum(
                    root=root, output_path=output, max_examples=8, languages=("rust",)
                )
            self.assertEqual(result["example_count"], 1, result)
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            payload = json.loads(record["completion"].split("</thought>\n", 1)[1])
            self.assertEqual(payload["format"], "SYNTRA_AGENT_PATCH_V1")
            self.assertEqual(
                {edit["path"] for edit in payload["edits"]},
                {"src/store.rs", "src/lib.rs"},
            )
            self.assertEqual(
                {edit["replace"] for edit in payload["edits"]},
                {"values.get(index)"},
            )

    def test_multifile_generator_does_not_emit_unrelated_or_unverified_file_pairs(self):
        from syntra_ai.reasoning_curriculum import synthesize_multifile_reasoning_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-no-pair-") as temp:
            root = Path(temp)
            (root / "a.mjs").write_text(
                "export function isText(value) { return typeof value == \"string\"; }\n",
                encoding="utf-8",
            )
            (root / "b.mjs").write_text(
                "export function isNumber(value) { return typeof value == \"number\"; }\n",
                encoding="utf-8",
            )
            output = root / "reasoning.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_multifile_reasoning_curriculum(
                    root=root, output_path=output, languages=("javascript",)
                )
            self.assertEqual(result["example_count"], 0)
            self.assertFalse(output.exists())

    def test_repository_curriculum_excludes_hidden_benchmark_targets(self):
        from syntra_ai.curriculum import synthesize_repository_curriculum

        with tempfile.TemporaryDirectory(prefix="syntra-phase7-hidden-exclusion-") as temp:
            root = Path(temp)
            (root / "public.py").write_text(
                '"""A local public utility module."""\n'
                "def normalize(value):\n"
                "    cleaned = value.strip().lower()\n"
                "    if cleaned:\n"
                "        return cleaned\n"
                "    return \"\"\n\n"
                "def count_items(items):\n"
                "    total = len(items)\n"
                "    if total > 0:\n"
                "        return total\n"
                "    return 0\n",
                encoding="utf-8",
            )
            hidden = root / "tests" / "benchmark_engine" / "hidden"
            hidden.mkdir(parents=True)
            (hidden / "target.mjs").write_text(
                "export function isText(value) { return typeof value == \"string\"; }\n",
                encoding="utf-8",
            )
            output = root / "curriculum.jsonl"
            with patch.dict(os.environ, {"SYNTRA_ALLOWED_CORPUS_ROOTS": str(root)}):
                result = synthesize_repository_curriculum(
                    root=root, output_path=output, max_examples=12
                )
            self.assertEqual(result["patch_candidate_count"], 0)
            self.assertFalse(any("benchmark_engine/hidden" in line for line in output.read_text(encoding="utf-8").splitlines()))

    def test_cli_exposes_all_languages_and_curriculum_modes(self):
        from syntra_ai.cli import _parser

        args = _parser().parse_args([
            "curriculum", "--mode", "multifile", "--languages",
            "javascript", "typescript", "go", "rust", "--max-examples", "12",
        ])
        self.assertEqual(args.mode, "multifile")
        self.assertEqual(args.languages, ["javascript", "typescript", "go", "rust"])
        self.assertEqual(args.max_examples, 12)


if __name__ == "__main__":
    unittest.main()

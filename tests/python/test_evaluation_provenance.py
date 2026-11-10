import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "benchmark_engine"))
from evaluation_provenance import EvaluationProvenanceError, evaluation_protocol_identity


class EvaluationProtocolIdentityTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[Path, Path]:
        engine = root / "tests" / "benchmark_engine"
        files = {
            "tests/benchmark_engine/suite.v1.json": '{"version":"1.0.0"}',
            "tests/benchmark_engine/runner.py": "runner",
            "tests/benchmark_engine/environments.py": "environment",
            "tests/benchmark_engine/apply_patch_bridge.mjs": "bridge",
            "tests/benchmark_engine/typescript_test_runner.mjs": "typescript",
            "package-lock.json": '{"lockfileVersion":3}',
            "local/agent.mjs": "agent",
            "local/workspace.mjs": "workspace",
            "local/tool-runtime.mjs": "tool-runtime",
            "tests/benchmark_engine/fixtures/task/src.py": "source",
            "tests/benchmark_engine/hidden/task/test.py": "assert True",
        }
        for relative, value in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        (engine / "fixtures").mkdir(exist_ok=True)
        (engine / "hidden").mkdir(exist_ok=True)
        return root, engine

    def test_hidden_evaluation_content_changes_protocol_identity(self):
        with tempfile.TemporaryDirectory(prefix="syntra-protocol-identity-") as temp:
            repository, engine = self._workspace(Path(temp))
            original = evaluation_protocol_identity(repository, engine)
            hidden_test = engine / "hidden" / "task" / "test.py"
            hidden_test.write_text("assert False", encoding="utf-8")
            changed = evaluation_protocol_identity(repository, engine)
            self.assertNotEqual(original["sha256"], changed["sha256"])
            self.assertEqual(original["file_count"], changed["file_count"])

    def test_fixture_content_changes_protocol_identity(self):
        with tempfile.TemporaryDirectory(prefix="syntra-protocol-fixture-") as temp:
            repository, engine = self._workspace(Path(temp))
            original = evaluation_protocol_identity(repository, engine)
            fixture = engine / "fixtures" / "task" / "src.py"
            fixture.write_text("source = 'changed'", encoding="utf-8")
            changed = evaluation_protocol_identity(repository, engine)
            self.assertNotEqual(original["sha256"], changed["sha256"])

    def test_symbolic_links_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="syntra-protocol-symlink-") as temp:
            repository, engine = self._workspace(Path(temp))
            outside = repository / "outside.py"
            outside.write_text("assert True", encoding="utf-8")
            link = engine / "hidden" / "task" / "linked_test.py"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symbolic-link creation is unavailable on this runner.")
            with self.assertRaisesRegex(EvaluationProvenanceError, "symbolic links"):
                evaluation_protocol_identity(repository, engine)


if __name__ == "__main__":
    unittest.main()

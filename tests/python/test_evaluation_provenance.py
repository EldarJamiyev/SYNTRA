import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "benchmark_engine"))
import evaluation_provenance as provenance
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
            "tests/benchmark_engine/evaluation_provenance.py": "provenance",
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


    def test_path_replacement_after_open_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="syntra-protocol-path-race-") as temp:
            root = Path(temp)
            target = root / "protocol.py"
            replacement = root / "replacement.py"
            target.write_text("original = True", encoding="utf-8")
            replacement.write_text("replacement = True", encoding="utf-8")
            real_fstat = os.fstat
            calls = 0

            def replace_after_descriptor_read(descriptor):
                nonlocal calls
                calls += 1
                result = real_fstat(descriptor)
                if calls == 2:
                    os.replace(replacement, target)
                return result

            with patch.object(provenance.os, "fstat", side_effect=replace_after_descriptor_read):
                with self.assertRaisesRegex(EvaluationProvenanceError, "path identity was replaced"):
                    provenance._hash_regular_file(target, repository_root=root)

    def test_parent_directory_redirect_to_same_inode_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="syntra-protocol-parent-swap-") as temp:
            workspace = Path(temp)
            repository = workspace / "repository"
            parent = repository / "protocol"
            outside = workspace / "outside"
            held = workspace / "held-protocol"
            parent.mkdir(parents=True)
            outside.mkdir()
            target = parent / "source.py"
            target.write_bytes(b"same inode, different canonical path")
            try:
                os.link(target, outside / target.name)
            except OSError:
                self.skipTest("Hard-link creation is unavailable on this runner.")

            real_fstat = os.fstat
            calls = 0

            def redirect_parent_after_descriptor_read(descriptor):
                nonlocal calls
                calls += 1
                result = real_fstat(descriptor)
                if calls == 2:
                    os.replace(parent, held)
                    parent.symlink_to(outside, target_is_directory=True)
                return result

            with patch.object(provenance.os, "fstat", side_effect=redirect_parent_after_descriptor_read):
                with self.assertRaisesRegex(EvaluationProvenanceError, "path identity was replaced/redirected"):
                    provenance._hash_regular_file(target, repository_root=repository)

    def test_stable_regular_file_hash_is_bound_to_approved_root(self):
        with tempfile.TemporaryDirectory(prefix="syntra-protocol-stable-") as temp:
            root = Path(temp)
            target = root / "protocol.py"
            content = b"stable protocol content"
            target.write_bytes(content)
            digest, size = provenance._hash_regular_file(target, repository_root=root)
            import hashlib
            self.assertEqual(digest, hashlib.sha256(content).hexdigest())
            self.assertEqual(size, len(content))

if __name__ == "__main__":
    unittest.main()

"""Tests for lock freshness and validation of the native Python environment."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import setup_syntra_env


class PythonEnvironmentLockTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.python = self.root / "python"
        self.python.write_text("test executable", encoding="utf-8")
        self.stamp = self.root / ".syntra-requirements-lock.sha256"
        self.fingerprint = "a" * 64
        self.stamp.write_text(self.fingerprint + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_matching_lock_and_healthy_cpu_environment_is_current(self):
        runner = Mock(
            side_effect=[
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=0),
            ]
        )
        self.assertTrue(
            setup_syntra_env.environment_is_current(
                self.python, self.stamp, self.fingerprint, run=runner
            )
        )
        self.assertEqual(runner.call_count, 2)
        self.assertIn("torch.version.cuda is None", runner.call_args_list[0].args[0][-1])
        self.assertEqual(runner.call_args_list[1].args[0][-3:], ["-m", "pip", "check"])

    def test_lock_change_invalidates_existing_environment_without_running_pip(self):
        runner = Mock()
        self.assertFalse(
            setup_syntra_env.environment_is_current(
                self.python, self.stamp, "b" * 64, run=runner
            )
        )
        runner.assert_not_called()

    def test_cuda_build_is_rejected_as_stale(self):
        runner = Mock(return_value=SimpleNamespace(returncode=1))
        self.assertFalse(
            setup_syntra_env.environment_is_current(
                self.python, self.stamp, self.fingerprint, run=runner
            )
        )
        self.assertEqual(runner.call_count, 1)

    def test_broken_dependency_graph_is_rejected(self):
        runner = Mock(
            side_effect=[
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=1),
            ]
        )
        self.assertFalse(
            setup_syntra_env.environment_is_current(
                self.python, self.stamp, self.fingerprint, run=runner
            )
        )
        self.assertEqual(runner.call_count, 2)

    def test_missing_environment_is_never_current(self):
        runner = Mock()
        self.assertFalse(
            setup_syntra_env.environment_is_current(
                self.root / "missing-python",
                self.stamp,
                self.fingerprint,
                run=runner,
            )
        )
        runner.assert_not_called()

    def test_lock_fingerprint_changes_when_lock_contents_change(self):
        lock = self.root / "requirements.lock.txt"
        lock.write_text("torch==2.10.0+cpu\n", encoding="utf-8")
        first = setup_syntra_env.lock_fingerprint(lock)
        lock.write_text("torch==2.10.1+cpu\n", encoding="utf-8")
        self.assertNotEqual(first, setup_syntra_env.lock_fingerprint(lock))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from syntra_ai.compute_lock import (
    ComputeBusyError,
    configure_compute_inference_capacity,
    exclusive_training_compute,
    inference_compute_guard,
    inference_compute_slot,
)


class NativeComputeLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="syntra-compute-lock-test-")
        self.addCleanup(self.temp.cleanup)
        self.lock_dir = Path(self.temp.name) / "locks"
        self.env_patch = patch.dict(os.environ, {
            "SYNTRA_COMPUTE_LOCK_DIR": str(self.lock_dir),
            "SYNTRA_MAX_CONCURRENT_INFERENCE": "1",
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        configure_compute_inference_capacity(1)

    def test_training_and_inference_exclude_each_other_in_one_process(self):
        with inference_compute_slot():
            with self.assertRaises(ComputeBusyError):
                with exclusive_training_compute():
                    self.fail("training must not enter while inference owns a slot")

        with exclusive_training_compute():
            with self.assertRaises(ComputeBusyError):
                with inference_compute_slot():
                    self.fail("inference must not enter while training owns all slots")

        with inference_compute_slot():
            pass

    def test_symlink_lock_directory_is_rejected_without_leaking_local_admission(self):
        target = Path(self.temp.name) / "private-locks"
        target.mkdir(mode=0o700)
        link = Path(self.temp.name) / "linked-locks"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Directory symlinks are unavailable on this runner.")

        with patch.dict(os.environ, {"SYNTRA_COMPUTE_LOCK_DIR": str(link)}):
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                with inference_compute_slot():
                    self.fail("a symlinked lock directory must fail closed")

        with inference_compute_slot():
            pass

    def test_configured_inference_slots_are_bounded(self):
        configure_compute_inference_capacity(2)
        with patch.dict(os.environ, {"SYNTRA_MAX_CONCURRENT_INFERENCE": "2"}):
            with inference_compute_slot():
                with inference_compute_slot():
                    with self.assertRaises(ComputeBusyError):
                        with inference_compute_slot():
                            self.fail("a third request must not exceed two configured slots")
        for invalid in (True, 0, 5, 1.5, "2"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                    configure_compute_inference_capacity(invalid)
        configure_compute_inference_capacity(1)

    def test_streaming_decorator_holds_lease_until_iteration_finishes(self):
        @inference_compute_guard
        def stream():
            yield "first"
            yield "last"

        iterator = stream()
        self.assertEqual(next(iterator), "first")
        with self.assertRaises(ComputeBusyError):
            with exclusive_training_compute():
                self.fail("training must remain excluded during streamed generation")
        self.assertEqual(list(iterator), ["last"])
        with exclusive_training_compute():
            pass

    def _run_child_holding(self, lock_kind: str):
        ready = Path(self.temp.name) / f"{lock_kind}.ready"
        if ready.exists():
            ready.unlink()
        context = "exclusive_training_compute" if lock_kind == "training" else "inference_compute_slot"
        source = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from syntra_ai.compute_lock import " + context + "\n"
            "with " + context + "():\n"
            "    Path(sys.argv[1]).write_text('acquired', encoding='utf-8')\n"
            "    time.sleep(30)\n"
        )
        env = dict(os.environ)
        env["SYNTRA_COMPUTE_LOCK_DIR"] = str(self.lock_dir)
        env["SYNTRA_MAX_CONCURRENT_INFERENCE"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-c", source, str(ready)],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ready.exists():
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            self.fail(
                f"child failed to acquire {lock_kind} lease; "
                f"exit={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
            )
        return process

    def test_training_in_another_process_blocks_inference_and_process_exit_releases_locks(self):
        process = self._run_child_holding("training")
        try:
            with self.assertRaises(ComputeBusyError):
                with inference_compute_slot():
                    self.fail("inference must see the child process's training lease")
        finally:
            process.terminate()
            process.wait(timeout=5)
        with inference_compute_slot():
            pass

    def test_inference_in_another_process_blocks_training_and_process_exit_releases_locks(self):
        process = self._run_child_holding("inference")
        try:
            with self.assertRaises(ComputeBusyError):
                with exclusive_training_compute():
                    self.fail("training must see the child process's inference lease")
        finally:
            process.terminate()
            process.wait(timeout=5)
        with exclusive_training_compute():
            pass


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from threading import Thread

from syntra_ai.compute_lock import (
    ComputeBusyError,
    acquire_compute_lease,
    compute_operation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def probe_compute_from_child(mode: str) -> subprocess.CompletedProcess[str]:
    script = (
        "from syntra_ai.compute_lock import ComputeBusyError, acquire_compute_lease\n"
        "import sys\n"
        "try:\n"
        "    with acquire_compute_lease(sys.argv[1]):\n"
        "        print('ACQUIRED')\n"
        "except ComputeBusyError:\n"
        "    print('BUSY')\n"
    )
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", script, mode],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


class NativeComputeLeaseTests(unittest.TestCase):
    def test_same_process_inference_shares_lease_but_training_is_exclusive(self):
        first = acquire_compute_lease("inference")
        second = acquire_compute_lease("inference")
        try:
            with self.assertRaisesRegex(ComputeBusyError, "inference is active"):
                acquire_compute_lease("training")
        finally:
            first.release()
        # The second in-flight inference must continue owning the OS reservation.
        with self.assertRaises(ComputeBusyError):
            acquire_compute_lease("training")
        second.release()

        with acquire_compute_lease("training"):
            with self.assertRaisesRegex(ComputeBusyError, "training is active"):
                acquire_compute_lease("inference")

    def test_os_lock_excludes_other_process_until_owner_releases(self):
        with acquire_compute_lease("training"):
            blocked = probe_compute_from_child("inference")
            self.assertEqual(blocked.returncode, 0, blocked.stderr)
            self.assertEqual(blocked.stdout.strip(), "BUSY")

        admitted = probe_compute_from_child("inference")
        self.assertEqual(admitted.returncode, 0, admitted.stderr)
        self.assertEqual(admitted.stdout.strip(), "ACQUIRED")

    def test_inference_owner_excludes_training_in_another_process(self):
        with acquire_compute_lease("inference"):
            blocked = probe_compute_from_child("training")
            self.assertEqual(blocked.returncode, 0, blocked.stderr)
            self.assertEqual(blocked.stdout.strip(), "BUSY")

    def test_lease_is_released_when_operation_raises(self):
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with acquire_compute_lease("inference"):
                raise RuntimeError("intentional failure")

        with acquire_compute_lease("training"):
            pass

    def test_generator_operation_holds_and_releases_lease_with_iteration(self):
        @compute_operation("inference")
        def events():
            yield "first"
            yield "last"

        stream = events()
        self.assertEqual(next(stream), "first")
        with self.assertRaises(ComputeBusyError):
            acquire_compute_lease("training")

        stream.close()
        with acquire_compute_lease("training"):
            pass

    def test_reserved_training_lease_can_be_transferred_to_worker_thread(self):
        @compute_operation("training")
        def run_training_step():
            return "completed"

        lease = acquire_compute_lease("training")
        result: list[str] = []
        errors: list[BaseException] = []

        def worker():
            try:
                with lease:
                    result.append(run_training_step())
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=worker, name="syntra-compute-lease-test")
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "worker did not release its compute lease")
        self.assertEqual(errors, [])
        self.assertEqual(result, ["completed"])
        with acquire_compute_lease("inference"):
            pass

    def test_lock_directory_and_file_are_private_on_posix(self):
        if os.name == "nt":
            self.skipTest("POSIX permission bits are not authoritative on Windows.")

        with acquire_compute_lease("inference"):
            import syntra_ai.compute_lock as compute_lock

            directory = compute_lock._lock_directory()
            lock_file = directory / "native-compute.lock"
            self.assertEqual(directory.stat().st_mode & 0o077, 0)
            self.assertEqual(lock_file.stat().st_mode & 0o077, 0)
            self.assertFalse(lock_file.is_symlink())


if __name__ == "__main__":
    unittest.main()

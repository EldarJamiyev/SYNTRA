from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
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

    def test_api_returns_retryable_503_before_starting_worker_when_other_process_owns_compute(self):
        import json
        from http.client import HTTPConnection
        from http.server import ThreadingHTTPServer
        from queue import Queue
        from unittest.mock import patch

        from syntra_ai import server as native_server

        child_script = (
            "from syntra_ai.compute_lock import acquire_compute_lease\n"
            "import time\n"
            "with acquire_compute_lease('training'):\n"
            "    print('READY', flush=True)\n"
            "    time.sleep(30)\n"
        )
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        child = subprocess.Popen(
            [sys.executable, "-c", child_script],
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready_line: Queue[str] = Queue()

        def read_child_ready():
            assert child.stdout is not None
            ready_line.put(child.stdout.readline())

        reader = Thread(target=read_child_ready, daemon=True)
        api_server = None
        api_thread = None
        try:
            reader.start()
            self.assertEqual(ready_line.get(timeout=15).strip(), "READY")
            from syntra_ai.server import Handler

            api_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            api_thread = Thread(target=api_server.serve_forever, daemon=True)
            api_thread.start()

            with patch.object(
                native_server,
                "_validated_training_request",
                return_value={"steps": 1, "resume": False},
            ):
                with patch.object(native_server, "_worker", None):
                    with patch.object(native_server, "_job", {"status": "idle"}):
                        with patch.object(native_server, "Thread") as worker:
                            connection = HTTPConnection(
                                "127.0.0.1", api_server.server_port, timeout=5
                            )
                            connection.request(
                                "POST",
                                "/model/train",
                                body=json.dumps({"root": "validated-by-test"}),
                                headers={"Content-Type": "application/json"},
                            )
                            response = connection.getresponse()
                            payload = json.loads(response.read())
                            connection.close()

                            self.assertEqual(response.status, 503)
                            self.assertEqual(response.getheader("Retry-After"), "1")
                            self.assertIn("another SYNTRA process", payload["error"])
                            worker.assert_not_called()
        finally:
            if api_server is not None:
                api_server.shutdown()
                api_server.server_close()
            if api_thread is not None:
                api_thread.join(timeout=3)
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
            if child.stdout is not None:
                child.stdout.close()
            if child.stderr is not None:
                child.stderr.close()

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

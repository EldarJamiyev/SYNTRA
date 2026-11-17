from __future__ import annotations

import json
import os
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from threading import Thread
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

    def test_api_training_endpoints_return_503_when_another_process_owns_compute(self):
        from syntra_ai import server as native_server

        process = self._run_child_holding("inference")
        http_server = ThreadingHTTPServer(("127.0.0.1", 0), native_server.Handler)
        server_thread = Thread(target=http_server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with patch.object(
                native_server,
                "_validated_training_request",
                return_value={"steps": 1, "resume": False},
            ), patch.object(
                native_server,
                "_validated_finetuning_request",
                return_value={"steps": 1, "resume": False},
            ), patch.object(native_server, "_worker", None), patch.object(native_server, "Thread") as worker:
                connection = HTTPConnection("127.0.0.1", http_server.server_port, timeout=3)
                requests = (
                    ("/model/train", {"root": "validated-by-test"}),
                    ("/model/finetune", {"dataset": "validated-by-test"}),
                )
                for route, body in requests:
                    connection.request(
                        "POST",
                        route,
                        body=json.dumps(body),
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    payload = json.loads(response.read())
                    self.assertEqual(response.status, 503)
                    self.assertEqual(response.getheader("Retry-After"), "1")
                    self.assertIn("another SYNTRA process", payload["error"])
                worker.assert_not_called()
                connection.request("GET", "/health")
                health_response = connection.getresponse()
                health = json.loads(health_response.read())
                self.assertEqual(health_response.status, 200)
                self.assertEqual(health["status"], "ok")
                connection.close()
        finally:
            http_server.shutdown()
            http_server.server_close()
            server_thread.join(timeout=5)
            process.terminate()
            process.wait(timeout=5)

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

    def test_cli_reports_busy_compute_with_nonzero_exit(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from unittest.mock import patch
        from syntra_ai import cli

        error_output = StringIO()
        with patch.object(cli, "generate", side_effect=ComputeBusyError("another process is training")):
            with redirect_stderr(error_output):
                exit_code = cli.main(["generate", "--prompt", "test prompt"])
        self.assertEqual(exit_code, 3)
        self.assertIn("SYNTRA compute admission rejected", error_output.getvalue())
        self.assertIn("another process is training", error_output.getvalue())

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

import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install requirements.txt to run local API smoke tests.")
class LocalApiSmokeTests(unittest.TestCase):
    def test_health_status_missing_checkpoint_and_origin_guard(self):
        from syntra_ai.server import Handler

        with tempfile.TemporaryDirectory(prefix="syntra-api-test-") as temp:
            missing_checkpoint = str(Path(temp) / "not-trained.pt")
            old_checkpoint = os.environ.get("SYNTRA_CHECKPOINT")
            os.environ["SYNTRA_CHECKPOINT"] = missing_checkpoint
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)

                connection.request("GET", "/health")
                response = connection.getresponse()
                health = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(health["service"], "SYNTRA native Transformer")
                self.assertFalse(health["pretrained_weights"])

                connection.request("GET", "/model/status")
                response = connection.getresponse()
                model_status = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertFalse(model_status["checkpoint_exists"])
                self.assertFalse(model_status["external_ai_api"])

                strict_json_cases = [
                    ('{"text":"first","text":"second"}', "duplicate JSON object key"),
                    ('{"text":"valid","metadata":{"mode":"safe","mode":"unsafe"}}', "duplicate JSON object key"),
                    ('{"text":"valid","metadata":NaN}', "non-standard JSON constant"),
                    ('{"text":"valid","metadata":Infinity}', "non-standard JSON constant"),
                    ('{"text":"valid","metadata":1e9999}', "non-finite JSON number"),
                    (
                        '{"text":"valid","metadata":' + "[" * 64 + "0" + "]" * 64 + "}",
                        "JSON nesting exceeds the 64-level request body limit",
                    ),
                ]
                for raw_body, expected_error in strict_json_cases:
                    with self.subTest(raw_body=raw_body[:100]):
                        connection.request(
                            "POST",
                            "/model/tokenize",
                            body=raw_body,
                            headers={"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        rejected = json.loads(response.read())
                        self.assertEqual(response.status, 400)
                        self.assertIn(expected_error, rejected["error"])

                invalid_generation_cases = [
                    ({"max_new_tokens": True}, False, "max_new_tokens must be an integer"),
                    ({"max_new_tokens": 1.5}, True, "max_new_tokens must be an integer"),
                    ({"temperature": "0.7"}, False, "temperature must be a finite number"),
                    ({"top_k": False}, True, "top_k must be a positive integer"),
                    ({"top_p": 0.0}, True, "top_p must be a finite number"),
                    ({"repetition_penalty": 0.5}, False, "repetition_penalty must be a finite number"),
                    ({"device": 123}, False, "device must be auto"),
                    ({"stream": "true"}, False, "stream must be a boolean"),
                ]
                for overrides, streaming, expected_error in invalid_generation_cases:
                    with self.subTest(overrides=overrides, streaming=streaming):
                        request_body = {"prompt": "valid prompt", **overrides}
                        if streaming and "stream" not in request_body:
                            request_body["stream"] = True
                        connection.request(
                            "POST",
                            "/model/generate",
                            body=json.dumps(request_body),
                            headers={"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        rejected = json.loads(response.read())
                        self.assertEqual(response.status, 400)
                        self.assertIn(expected_error, rejected["error"])

                other_endpoint_invalid_cases = [
                    ("/model/embed", {"input": "valid text", "device": 123}, "device must be auto"),
                    ("/model/evaluate", {"root": temp, "batches": True}, "batches must be an integer"),
                    ("/model/evaluate", {"root": temp, "max_tokens": 10.5}, "max_tokens must be an integer"),
                    ("/model/evaluate", {"root": temp, "device": 123}, "device must be auto"),
                    ("/model/curriculum", {"root": temp, "max_examples": 4.5}, "max_examples must be between"),
                    ("/model/curriculum", {"root": temp, "max_examples": True}, "max_examples must be between"),
                ]
                for route, request_body, expected_error in other_endpoint_invalid_cases:
                    with self.subTest(route=route, request_body=request_body):
                        connection.request(
                            "POST",
                            route,
                            body=json.dumps(request_body),
                            headers={"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        rejected = json.loads(response.read())
                        self.assertEqual(response.status, 400)
                        self.assertIn(expected_error, rejected["error"])

                connection.request(
                    "POST",
                    "/model/generate",
                    body=json.dumps({"prompt": "def add(a, b):"}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                body = json.loads(response.read())
                self.assertEqual(response.status, 409)
                self.assertIn("No trained SYNTRA checkpoint", body["error"])

                from syntra_ai.config import ModelConfig
                from syntra_ai.model import SyntraTransformer

                from syntra_ai.tokenizer import ByteBPETokenizer
                config = ModelConfig(vocab_size=258, seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
                tokenizer = ByteBPETokenizer.empty()
                model = SyntraTransformer(config)
                torch.save({
                    "format": "syntra-transformer-v1",
                    "config": config.to_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "model_state": model.state_dict(),
                    "step": 0,
                    "train_loss": None,
                    "validation_loss": None,
                }, missing_checkpoint)
                connection.request(
                    "POST",
                    "/model/tokenize",
                    body=json.dumps({"text": "function add(a, b) { return a + b; }"}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                tokenized = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(tokenized["token_count"], len("function add(a, b) { return a + b; }".encode("utf-8")))
                self.assertEqual(tokenized["context_window_tokens"], 32)

                connection.request(
                    "POST",
                    "/model/generate",
                    body=json.dumps({
                        "prompt": "def add(",
                        "max_new_tokens": 2,
                        "temperature": 1.0,
                        "top_k": 2,
                        "device": "cpu",
                    }),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                generated = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertTrue(generated["is_model_output"])
                # The native generator may stop on EOS immediately. EOS is not
                # included in generated_tokens, so zero decoded completion
                # tokens is a valid real model outcome rather than a fake result.
                self.assertIsInstance(generated["text"], str)
                self.assertIs(type(generated["generated_tokens"]), int)
                self.assertGreaterEqual(generated["generated_tokens"], 0)
                self.assertLessEqual(generated["generated_tokens"], 2)
                self.assertFalse(generated["prompt_truncated"])

                # Both ordinary and streaming APIs must reject overflow before
                # inference instead of silently discarding the prompt prefix.
                oversized_prompt = "x" * tokenized["context_window_tokens"]
                for streaming in (False, True):
                    connection.request(
                        "POST",
                        "/model/generate",
                        body=json.dumps({
                            "prompt": oversized_prompt,
                            "max_new_tokens": 2,
                            "device": "cpu",
                            "stream": streaming,
                        }),
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    overflow = json.loads(response.read())
                    self.assertEqual(response.status, 422)
                    self.assertEqual(overflow["code"], "context_window_exceeded")
                    self.assertEqual(overflow["prompt_tokens"], tokenized["context_window_tokens"])
                    self.assertEqual(overflow["context_window_tokens"], tokenized["context_window_tokens"])
                    self.assertEqual(overflow["usable_context_tokens"], tokenized["context_window_tokens"] - 1)
                    self.assertEqual(overflow["tokens_over_context"], 1)

                connection.request(
                    "POST",
                    "/model/train",
                    body=json.dumps({"root": temp, "steps": True}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                invalid_training = json.loads(response.read())
                self.assertEqual(response.status, 400)
                self.assertIn("steps must be an integer", invalid_training["error"])

                connection.request("GET", "/model/train/status")
                response = connection.getresponse()
                training_status = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(training_status["status"], "idle")

                connection.request(
                    "POST",
                    "/model/train",
                    body=json.dumps({"root": temp}),
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://untrusted.example",
                    },
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 403)
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
                if old_checkpoint is None:
                    os.environ.pop("SYNTRA_CHECKPOINT", None)
                else:
                    os.environ["SYNTRA_CHECKPOINT"] = old_checkpoint


class NativeApiPathBoundaryTests(unittest.TestCase):
    def test_training_and_finetuning_reject_unapproved_absolute_paths(self):
        from syntra_ai.server import start_finetuning, start_training

        previous = os.environ.get("SYNTRA_ALLOWED_CORPUS_ROOTS")
        os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
        try:
            with tempfile.TemporaryDirectory(prefix="syntra-unapproved-api-") as temp:
                dataset = Path(temp) / "instructions.jsonl"
                dataset.write_text('{}\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                    start_training({"root": temp})
                with self.assertRaisesRegex(ValueError, "outside the SYNTRA project"):
                    start_finetuning({"dataset": str(dataset)})
        finally:
            if previous is None:
                os.environ.pop("SYNTRA_ALLOWED_CORPUS_ROOTS", None)
            else:
                os.environ["SYNTRA_ALLOWED_CORPUS_ROOTS"] = previous



class MockComputeLease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class TrainingApiRequestValidationTests(unittest.TestCase):
    def test_invalid_source_training_parameters_never_create_a_worker(self):
        from unittest.mock import patch
        from syntra_ai import server

        invalid_requests = [
            ({"steps": True}, "steps must be an integer"),
            ({"steps": "2"}, "steps must be an integer"),
            ({"batch_size": 129}, "batch_size must be an integer"),
            ({"learning_rate": float("nan")}, "learning_rate must be a finite"),
            ({"learning_rate": 10**1000}, "learning_rate must be a finite"),
            ({"max_tokens": 10}, "max_tokens must be an integer"),
            ({"seed": 1.5}, "seed must be a signed/unsigned"),
            ({"resume": "false"}, "resume must be a boolean"),
            ({"device": "meta"}, "device must be auto"),
            ({"steps": 1, "unexpected": True}, "unsupported field"),
            ({"config": {"seq_len": 32, "seq_lne": 32}}, "unsupported field"),
            ({"config": {"block_size": 32, "seq_len": 32}}, "either block_size or seq_len"),
            ({"config": {"seq_len": 16385}}, "seq_len must be between"),
        ]

        with tempfile.TemporaryDirectory(prefix="syntra-training-request-") as temp:
            for overrides, expected in invalid_requests:
                with self.subTest(overrides=overrides):
                    with patch.object(server, "resolve_corpus_root", return_value=Path(temp)):
                        with patch.object(server, "_worker", None):
                            with patch.object(server, "Thread") as worker:
                                with self.assertRaisesRegex(ValueError, expected):
                                    server.start_training({"root": temp, **overrides})
                                worker.assert_not_called()

    def test_source_training_request_is_normalized_before_worker_creation(self):
        from unittest.mock import patch
        from syntra_ai import server

        with tempfile.TemporaryDirectory(prefix="syntra-training-request-valid-") as temp:
            config = {
                "vocab_size": 258,
                "seq_len": 32,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 32,
                "dropout": 0.0,
            }
            lease = MockComputeLease()
            with patch.object(server, "resolve_corpus_root", return_value=Path(temp)):
                with patch.object(server, "_worker", None):
                    with patch.object(server, "_job", {"status": "idle"}):
                        with patch.object(server, "acquire_compute_lease", return_value=lease):
                            with patch.object(server, "Thread") as worker:
                                started = server.start_training({
                                    "root": temp,
                                    "steps": 2,
                                    "batch_size": 1,
                                    "config": config,
                                    "learning_rate": 1e-4,
                                    "max_tokens": 1024,
                                    "device": "cpu",
                                    "seed": 17,
                                    "resume": False,
                                })
                                worker_args = worker.call_args.kwargs["args"]
                                worker_body = worker_args[0]
                                self.assertIs(worker_args[1], lease)

            lease.release()
            self.assertEqual(started["status"], "starting")
            self.assertEqual(worker_body["root"], str(Path(temp)))
            self.assertEqual(worker_body["steps"], 2)
            self.assertEqual(worker_body["batch_size"], 1)
            self.assertEqual(worker_body["config"]["seq_len"], 32)
            self.assertEqual(worker_body["learning_rate"], 1e-4)
            self.assertEqual(worker_body["max_tokens"], 1024)
            self.assertEqual(worker_body["device"], "cpu")
            self.assertEqual(worker_body["seed"], 17)
            self.assertIs(worker_body["resume"], False)

    def test_source_training_resume_requires_an_existing_checkpoint(self):
        from unittest.mock import patch
        from syntra_ai import server

        with tempfile.TemporaryDirectory(prefix="syntra-training-resume-api-") as temp:
            checkpoint = Path(temp) / "missing-checkpoint.pt"
            with patch.object(server, "resolve_corpus_root", return_value=Path(temp)):
                with patch.object(server, "_default_checkpoint", return_value=str(checkpoint)):
                    with patch.object(server, "_worker", None):
                        with patch.object(server, "Thread") as worker:
                            with self.assertRaisesRegex(FileNotFoundError, "checkpoint file does not exist"):
                                server.start_training({"root": temp, "resume": True})
                            worker.assert_not_called()


class FineTuningApiResumeContractTests(unittest.TestCase):
    def test_resume_is_boolean_and_worker_receives_resolved_dataset_path(self):
        from unittest.mock import patch
        from syntra_ai import server

        with tempfile.TemporaryDirectory(prefix="syntra-finetune-api-") as temp:
            resolved_dataset = Path(temp) / "canonical-instructions.jsonl"
            resolved_dataset.write_text('{"prompt":"p","completion":"c"}\\n', encoding="utf-8")
            checkpoint = Path(temp) / "native-checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")

            with patch.object(server, "_default_checkpoint", return_value=str(checkpoint)):
                with patch.object(server, "resolve_instruction_dataset", return_value=resolved_dataset):
                    with patch.object(server, "_job", {"status": "idle"}):
                        with patch.object(server, "_worker", None):
                            with self.assertRaisesRegex(ValueError, "resume must be a boolean"):
                                server.start_finetuning({
                                    "dataset": str(resolved_dataset),
                                    "resume": "true",
                                })

                        with patch.object(server, "_worker", None):
                            lease = MockComputeLease()
                            with patch.object(server, "acquire_compute_lease", return_value=lease):
                                with patch.object(server, "Thread") as worker:
                                    started = server.start_finetuning({
                                        "dataset": "untrusted-input-path.jsonl",
                                        "resume": True,
                                        "gradient_accumulation_steps": 2,
                                        "device": "cpu",
                                    })
                                    worker_args = worker.call_args.kwargs["args"]
                                    worker_body = worker_args[0]
                                    self.assertIs(worker_args[1], lease)

            lease.release()
            self.assertEqual(started["status"], "starting")
            self.assertEqual(worker_body["dataset"], str(resolved_dataset))
            self.assertTrue(worker_body["resume"])
            self.assertEqual(worker_body["gradient_accumulation_steps"], 2)

class FineTuningApiValidationTests(unittest.TestCase):
    def test_invalid_finetuning_parameters_never_create_a_worker(self):
        from unittest.mock import patch
        from syntra_ai import server

        invalid_requests = [
            ({"steps": True}, "steps must be an integer"),
            ({"batch_size": 0}, "batch_size must be an integer"),
            ({"gradient_accumulation_steps": 129}, "gradient_accumulation_steps must be an integer"),
            ({"learning_rate": float("inf")}, "learning_rate must be a finite"),
            ({"resume": "true"}, "resume must be a boolean"),
            ({"device": "meta"}, "device must be auto"),
            ({"unrecognized": 1}, "unsupported field"),
        ]

        with tempfile.TemporaryDirectory(prefix="syntra-finetuning-request-") as temp:
            dataset = Path(temp) / "instructions.jsonl"
            dataset.write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
            for overrides, expected in invalid_requests:
                with self.subTest(overrides=overrides):
                    with patch.object(server, "resolve_instruction_dataset", return_value=dataset):
                        with patch.object(server, "_worker", None):
                            with patch.object(server, "Thread") as worker:
                                with self.assertRaisesRegex(ValueError, expected):
                                    server.start_finetuning({"dataset": str(dataset), **overrides})
                                worker.assert_not_called()


class FineTuningCliResumeContractTests(unittest.TestCase):
    def test_finetune_cli_exposes_explicit_resume_and_accumulation_settings(self):
        from syntra_ai.cli import _parser

        args = _parser().parse_args([
            "finetune",
            "--dataset", "instructions.jsonl",
            "--steps", "100",
            "--batch-size", "2",
            "--gradient-accumulation-steps", "3",
            "--resume",
        ])
        self.assertEqual(args.command, "finetune")
        self.assertTrue(args.resume)
        self.assertEqual(args.gradient_accumulation_steps, 3)
        self.assertEqual(args.steps, 100)


class InferenceAdmissionControlTests(unittest.TestCase):
    def _start_server(self):
        from syntra_ai.server import Handler
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_busy_inference_slot_returns_retryable_503_without_running_endpoint(self):
        from threading import BoundedSemaphore
        from unittest.mock import patch

        with patch("syntra_ai.server._INFERENCE_SLOTS", BoundedSemaphore(1)) as slot:
            self.assertTrue(slot.acquire(blocking=False))
            server, thread = self._start_server()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                connection.request(
                    "POST",
                    "/model/embed",
                    body=json.dumps({"input": ["bounded test"]}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 503)
                self.assertEqual(response.getheader("Retry-After"), "1")
                self.assertIn("inference capacity is busy", payload["error"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                slot.release()

    def test_inference_slot_is_released_when_endpoint_validation_fails(self):
        from threading import BoundedSemaphore
        from unittest.mock import patch

        slot = BoundedSemaphore(1)
        with patch("syntra_ai.server._INFERENCE_SLOTS", slot):
            with patch("syntra_ai.server.embed_texts", side_effect=ValueError("invalid embedding input")):
                server, thread = self._start_server()
                try:
                    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                    connection.request(
                        "POST",
                        "/model/embed",
                        body=json.dumps({"input": ["bounded test"]}),
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    payload = json.loads(response.read())
                    self.assertEqual(response.status, 400)
                    self.assertIn("invalid embedding input", payload["error"])
                    self.assertTrue(slot.acquire(blocking=False), "request failures must release their slot")
                    slot.release()
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_streaming_holds_inference_slot_until_all_events_are_written(self):
        from threading import BoundedSemaphore
        from unittest.mock import patch

        slot = BoundedSemaphore(1)
        observed = []

        def stream_events(**_kwargs):
            observed.append(slot.acquire(blocking=False))
            if observed[-1]:
                slot.release()
            yield {"delta": "x", "done": False}
            yield {"delta": "", "text": "x", "done": True}

        with patch("syntra_ai.server._INFERENCE_SLOTS", slot):
            with patch("syntra_ai.server.stream_generate", side_effect=stream_events):
                server, thread = self._start_server()
                try:
                    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                    connection.request(
                        "POST",
                        "/model/generate",
                        body=json.dumps({"prompt": "x", "stream": True}),
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    stream = response.read().decode("utf-8")
                    self.assertEqual(response.status, 200)
                    self.assertEqual(len([line for line in stream.splitlines() if line.strip()]), 2)
                    self.assertEqual(observed, [False], "the slot must stay occupied while streaming")
                    self.assertTrue(slot.acquire(blocking=False), "stream completion must release its slot")
                    slot.release()
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_training_and_finetuning_are_rejected_while_inference_is_active(self):
        from unittest.mock import patch
        from syntra_ai import server

        operations = (
            ("start_training", "_validated_training_request"),
            ("start_finetuning", "_validated_finetuning_request"),
        )
        for starter_name, validator_name in operations:
            with self.subTest(operation=starter_name):
                with server._inference_slot() as admitted:
                    self.assertTrue(admitted)
                    with patch.object(server, validator_name, return_value={"steps": 1, "resume": False}):
                        with patch.object(server, "_worker", None):
                            with patch.object(server, "Thread") as worker:
                                with self.assertRaises(server.ComputeBusyError):
                                    getattr(server, starter_name)({})
                                worker.assert_not_called()

    def test_training_http_request_returns_retryable_503_during_inference(self):
        from unittest.mock import patch
        from syntra_ai import server as native_server

        with native_server._inference_slot() as admitted:
            self.assertTrue(admitted)
            with patch.object(
                native_server,
                "_validated_training_request",
                return_value={"steps": 1, "resume": False},
            ):
                with patch.object(native_server, "Thread") as worker:
                    http_server, thread = self._start_server()
                    try:
                        connection = HTTPConnection("127.0.0.1", http_server.server_port, timeout=3)
                        connection.request(
                            "POST",
                            "/model/train",
                            body=json.dumps({"root": "validated-by-test"}),
                            headers={"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        payload = json.loads(response.read())
                        self.assertEqual(response.status, 503)
                        self.assertEqual(response.getheader("Retry-After"), "1")
                        self.assertIn("active inference", payload["error"])
                        worker.assert_not_called()

                        connection.request("GET", "/health")
                        health_response = connection.getresponse()
                        health = json.loads(health_response.read())
                        self.assertEqual(health_response.status, 200)
                        self.assertEqual(health["status"], "ok")
                    finally:
                        http_server.shutdown()
                        http_server.server_close()
                        thread.join(timeout=2)

    def test_inference_is_rejected_while_training_worker_is_alive(self):
        from threading import BoundedSemaphore
        from unittest.mock import patch
        class RunningWorker:
            @staticmethod
            def is_alive():
                return True

        slot = BoundedSemaphore(1)
        with patch("syntra_ai.server._INFERENCE_SLOTS", slot):
            with patch("syntra_ai.server._worker", RunningWorker()):
                with patch("syntra_ai.server.embed_texts") as embed:
                    server, thread = self._start_server()
                    try:
                        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                        connection.request(
                            "POST",
                            "/model/embed",
                            body=json.dumps({"input": ["training exclusion"]}),
                            headers={"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        payload = json.loads(response.read())
                        self.assertEqual(response.status, 503)
                        self.assertEqual(response.getheader("Retry-After"), "1")
                        self.assertIn("inference capacity is busy", payload["error"])
                        embed.assert_not_called()
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)

    def test_failed_worker_start_does_not_leave_training_gate_stuck(self):
        from unittest.mock import Mock, patch
        from syntra_ai import server

        worker = Mock()
        worker.start.side_effect = RuntimeError("thread start failed")
        with patch.object(server, "_validated_training_request", return_value={"steps": 1, "resume": False}):
            with patch.object(server, "_worker", None):
                with patch.object(server, "_job", {"status": "idle"}):
                    with patch.object(server, "Thread", return_value=worker):
                        with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                            server.start_training({})
                        self.assertIsNone(server._worker)
                        self.assertEqual(server._job["status"], "failed")

        from syntra_ai.compute_lock import acquire_compute_lease
        with acquire_compute_lease("inference"):
            pass

    def test_inference_capacity_configuration_is_strictly_bounded(self):
        from syntra_ai.server import _configure_inference_slots

        self.assertEqual(_configure_inference_slots("1"), 1)
        self.assertEqual(_configure_inference_slots("4"), 4)
        for invalid in ("0", "5", "1.5", "auto", ""):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                    _configure_inference_slots(invalid)
        _configure_inference_slots("1")


if __name__ == "__main__":
    unittest.main()

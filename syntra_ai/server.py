"""Loopback-only HTTP API for SYNTRA's trained local model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import json
import os
import time

from .config import ModelConfig
from .runtime import (
    ContextWindowExceededError, embed_texts, evaluate, generate, status, stream_generate, tokenize_text,
)
from .training import train
from .instruction import finetune, synthesize_repository_curriculum
from .safe_paths import resolve_corpus_root, resolve_instruction_dataset

HOST = "127.0.0.1"
PORT = int(os.environ.get("SYNTRA_AI_PORT", "3002"))
MAX_BODY_BYTES = 1_000_000
ALLOWED_ORIGINS = {
    "http://127.0.0.1:3000", "http://localhost:3000",
    "http://127.0.0.1:5173", "http://localhost:5173",
}
_job_lock = Lock()
_job: dict[str, Any] = {
    "status": "idle", "step": 0, "max_steps": 0, "train_loss": None,
    "validation_loss": None, "message": "No training job has run in this process."
}
_worker: Thread | None = None


def _set_job(value: dict) -> None:
    global _job
    with _job_lock:
        _job = dict(value)


def get_job() -> dict:
    with _job_lock:
        return dict(_job)


def _default_checkpoint() -> str:
    configured = os.environ.get("SYNTRA_CHECKPOINT")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str((Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "syntra.pt").resolve())


def _run_training(body: dict) -> None:
    try:
        architecture = body.get("config") or {}
        config = ModelConfig.from_dict(architecture) if architecture else ModelConfig()
        result = train(
            roots=[body["root"]],
            checkpoint=_default_checkpoint(),
            steps=int(body.get("steps", 1000)),
            batch_size=int(body.get("batch_size", 8)),
            config=config,
            learning_rate=float(body.get("learning_rate", 3e-4)),
            max_tokens=int(body.get("max_tokens", 5_000_000)),
            device_name=str(body.get("device", "auto")),
            seed=int(body.get("seed", 1337)),
            resume=body.get("resume") is True,
            callback=_set_job,
        )
        _set_job(result)
    except Exception as exc:
        current = get_job()
        current.update({
            "status": "failed",
            "finished_at": time.time(),
            "message": str(exc),
        })
        _set_job(current)


def start_training(body: dict) -> dict:
    global _worker
    normalized = resolve_corpus_root(body.get("root"))
    body = dict(body)
    body["root"] = str(normalized)
    with _job_lock:
        if _worker is not None and _worker.is_alive():
            raise ValueError("SYNTRA training is already running.")
        _job.update({
            "status": "starting", "step": 0, "max_steps": int(body.get("steps", 1000)),
            "train_loss": None, "validation_loss": None, "message": "Training worker is starting."
        })
        _worker = Thread(target=_run_training, args=(dict(body),), daemon=True, name="syntra-training")
        _worker.start()
        return dict(_job)


def _run_finetuning(body: dict) -> None:
    try:
        result = finetune(
            dataset_path=body["dataset"],
            checkpoint=_default_checkpoint(),
            steps=int(body.get("steps", 1000)),
            batch_size=int(body.get("batch_size", 4)),
            learning_rate=float(body.get("learning_rate", 1e-4)),
            device_name=str(body.get("device", "auto")),
            callback=_set_job,
        )
        _set_job(result)
    except Exception as exc:
        current = get_job()
        current.update({"status": "failed", "finished_at": time.time(), "message": str(exc)})
        _set_job(current)


def start_finetuning(body: dict) -> dict:
    global _worker
    dataset_path = resolve_instruction_dataset(body.get("dataset"))
    with _job_lock:
        if _worker is not None and _worker.is_alive():
            raise ValueError("SYNTRA training is already running.")
        _job.update({
            "status": "starting", "step": 0, "max_steps": int(body.get("steps", 1000)),
            "train_loss": None, "validation_loss": None,
            "message": "Supervised instruction-tuning worker is starting."
        })
        _worker = Thread(
            target=_run_finetuning, args=({"dataset": str(dataset_path), **body},),
            daemon=True, name="syntra-instruction-finetuning"
        )
        _worker.start()
        return dict(_job)


class Handler(BaseHTTPRequestHandler):
    server_version = "SYNTRA-Local-AI/0.2"

    def log_message(self, format: str, *args: object) -> None:
        print("[SYNTRA AI] " + (format % args))

    def _send(self, status_code: int, value: dict) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("Request body exceeds the local AI API size limit.")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object.")
        return value

    def do_GET(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin not in ALLOWED_ORIGINS:
            self._send(403, {"error": "Origin is not allowed by the loopback AI service."})
            return
        if self.path == "/health":
            self._send(200, {
                "status": "ok", "service": "SYNTRA native Transformer",
                "host": HOST, "port": PORT, "framework": "PyTorch",
                "pretrained_weights": False, "external_ai_api": False
            })
        elif self.path == "/model/status":
            current = status()
            current["training"] = get_job()
            self._send(200, current)
        elif self.path == "/model/train/status":
            self._send(200, get_job())
        else:
            self._send(404, {"error": "Route not found."})

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_POST(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin not in ALLOWED_ORIGINS:
            self._send(403, {"error": "Origin is not allowed by the loopback AI service."})
            return
        if not origin and self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._send(403, {"error": "Cross-site requests are not accepted by the loopback AI service."})
            return
        try:
            body = self._body()
            if self.path == "/model/train":
                result = start_training(body)
                self._send(202, result)
            elif self.path == "/model/finetune":
                result = start_finetuning(body)
                self._send(202, result)
            elif self.path == "/model/tokenize":
                result = tokenize_text(body.get("text", ""))
                self._send(200, result)
            elif self.path == "/model/embed":
                texts = body.get("input") if "input" in body else body.get("texts", [])
                if isinstance(texts, str):
                    texts = [texts]
                result = embed_texts(
                    texts=texts,
                    device_name=str(body.get("device", "auto")),
                )
                self._send(200, result)
            elif self.path == "/model/curriculum":
                result = synthesize_repository_curriculum(
                    root=body.get("root"),
                    output_path=body.get("output") or body.get("dataset"),
                    max_examples=int(body.get("max_examples", 240)),
                )
                self._send(200, result)
            elif self.path == "/model/evaluate":
                result = evaluate(
                    root=body.get("root", ""),
                    batches=int(body.get("batches", 16)),
                    device_name=str(body.get("device", "auto")),
                    max_tokens=int(body.get("max_tokens", 2_000_000)),
                )
                self._send(200, result)
            elif self.path == "/model/generate":
                if body.get("stream") is True:
                    stream = stream_generate(
                        prompt=body.get("prompt", ""),
                        max_new_tokens=int(body.get("max_new_tokens", 256)),
                        temperature=float(body.get("temperature", 0.7)),
                        top_k=int(body.get("top_k", 40)),
                        top_p=float(body.get("top_p", 1.0)),
                        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
                        device_name=str(body.get("device", "auto")),
                    )
                    first_event = next(stream, None)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    if first_event is not None:
                        self.wfile.write((json.dumps(first_event, ensure_ascii=False) + "\n").encode("utf-8"))
                        self.wfile.flush()
                    for event in stream:
                        self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                        self.wfile.flush()
                else:
                    result = generate(
                        prompt=body.get("prompt", ""),
                        max_new_tokens=int(body.get("max_new_tokens", 256)),
                        temperature=float(body.get("temperature", 0.7)),
                        top_k=int(body.get("top_k", 40)),
                        top_p=float(body.get("top_p", 1.0)),
                        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
                        device_name=str(body.get("device", "auto")),
                    )
                    self._send(200, result)
            else:
                self._send(404, {"error": "Route not found."})
        except FileNotFoundError as exc:
            self._send(409, {"error": str(exc)})
        except ContextWindowExceededError as exc:
            self._send(422, {
                "error": str(exc),
                "code": "context_window_exceeded",
                "prompt_tokens": exc.prompt_tokens,
                "context_window_tokens": exc.context_window_tokens,
                "usable_context_tokens": exc.usable_context_tokens,
                "tokens_over_context": exc.tokens_over_context,
            })
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": type(exc).__name__ + ": " + str(exc)})


def main() -> None:
    if not 1024 <= PORT <= 65535:
        raise SystemExit("SYNTRA_AI_PORT must be between 1024 and 65535.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SYNTRA native AI API listening on http://{HOST}:{PORT}")
    print("Loopback-only. No hosted model API or pretrained weights are used.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SYNTRA AI service.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Loopback-only HTTP API for SYNTRA's trained local model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import json
import math
import os
import re
import time

from .config import ModelConfig
from .data import MAX_CORPUS_TOKENS
from .runtime import (
    ContextWindowExceededError, embed_texts, evaluate, generate, status, stream_generate, tokenize_text,
)
from .training import choose_device, train, torch as torch_runtime
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



TRAINING_REQUEST_FIELDS = {
    "root", "steps", "batch_size", "config", "learning_rate",
    "max_tokens", "device", "seed", "resume",
}
FINETUNING_REQUEST_FIELDS = {
    "dataset", "steps", "batch_size", "gradient_accumulation_steps",
    "learning_rate", "device", "resume",
}
MODEL_CONFIG_FIELDS = set(ModelConfig.__dataclass_fields__) | {"block_size"}


def _validate_request_fields(body: dict, allowed: set[str], operation: str) -> None:
    if not isinstance(body, dict):
        raise ValueError(operation + " request must be a JSON object.")
    unknown = sorted(str(key) for key in body if key not in allowed)
    if unknown:
        raise ValueError(
            operation + " request contains unsupported field(s): " + ", ".join(unknown) + "."
        )


def _request_integer(
    body: dict,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = body.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            name + " must be an integer between " + f"{minimum:,} and {maximum:,}."
        )
    return value


def _request_learning_rate(body: dict, default: float) -> float:
    value = body.get("learning_rate", default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 1e-6 <= value <= 1e-2
    ):
        raise ValueError("learning_rate must be a finite number between 1e-6 and 1e-2.")
    return float(value)


def _request_device(value: Any) -> tuple[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps.")
    normalized = value.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"} and not re.fullmatch(r"cuda:[0-9]+", normalized):
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps.")
    if torch_runtime is None:
        raise ValueError("PyTorch is unavailable; native training cannot start.")
    try:
        selected = choose_device(normalized)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Requested device is unavailable: " + str(exc)) from None
    if selected.type == "cuda" and selected.index is not None:
        if selected.index >= torch_runtime.cuda.device_count():
            raise ValueError(
                "Requested CUDA device index " + str(selected.index)
                + " is outside the available device count."
            )
    return normalized, selected


def _validated_training_request(body: dict) -> dict:
    _validate_request_fields(body, TRAINING_REQUEST_FIELDS, "Training")
    raw_root = body.get("root")
    if not isinstance(raw_root, str) or not raw_root.strip() or len(raw_root) > 4096 or "\x00" in raw_root:
        raise ValueError("root must be a non-empty absolute directory path.")
    resume = body.get("resume", False)
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean.")

    raw_config = body.get("config", {})
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("config must be a model-configuration object.")
    unknown_config = sorted(str(key) for key in raw_config if key not in MODEL_CONFIG_FIELDS)
    if unknown_config:
        raise ValueError("config contains unsupported field(s): " + ", ".join(unknown_config) + ".")
    if "block_size" in raw_config and "seq_len" in raw_config:
        raise ValueError("config must use either block_size or seq_len, not both.")
    config = ModelConfig.from_dict(raw_config) if raw_config else ModelConfig()
    config.validate()

    steps = _request_integer(body, "steps", 1000, 1, 1_000_000)
    batch_size = _request_integer(body, "batch_size", 8, 1, 128)
    max_tokens = _request_integer(body, "max_tokens", 5_000_000, 1_024, MAX_CORPUS_TOKENS)
    learning_rate = _request_learning_rate(body, 3e-4)
    seed = body.get("seed", 1337)
    if type(seed) is not int or not -(1 << 63) <= seed <= (1 << 64) - 1:
        raise ValueError("seed must be a signed/unsigned 64-bit integer accepted by PyTorch.")
    normalized_root = resolve_corpus_root(raw_root)
    device_name, _selected_device = _request_device(body.get("device", "auto"))

    checkpoint = Path(_default_checkpoint())
    if resume and not checkpoint.is_file():
        raise FileNotFoundError(
            "Cannot resume source pretraining because the checkpoint file does not exist. "
            "Start a new run with resume=False."
        )
    return {
        "root": str(normalized_root),
        "steps": steps,
        "batch_size": batch_size,
        "config": config.to_dict(),
        "learning_rate": learning_rate,
        "max_tokens": max_tokens,
        "device": device_name,
        "seed": seed,
        "resume": resume,
    }


def _validated_finetuning_request(body: dict) -> dict:
    _validate_request_fields(body, FINETUNING_REQUEST_FIELDS, "Fine-tuning")
    raw_dataset = body.get("dataset")
    if not isinstance(raw_dataset, str) or not raw_dataset.strip() or len(raw_dataset) > 4096 or "\x00" in raw_dataset:
        raise ValueError("dataset must be a non-empty absolute JSONL file path.")
    resume = body.get("resume", False)
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean.")
    steps = _request_integer(body, "steps", 1000, 1, 1_000_000)
    batch_size = _request_integer(body, "batch_size", 4, 1, 128)
    accumulation = _request_integer(body, "gradient_accumulation_steps", 1, 1, 128)
    learning_rate = _request_learning_rate(body, 1e-4)
    dataset_path = resolve_instruction_dataset(raw_dataset)
    device_name, selected_device = _request_device(body.get("device", "auto"))
    if resume and selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("Exact SFT resume currently supports CPU and CUDA devices only.")
    if not Path(_default_checkpoint()).is_file():
        raise FileNotFoundError("Train a SYNTRA checkpoint from local source code before instruction fine-tuning.")
    return {
        "dataset": str(dataset_path),
        "steps": steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": learning_rate,
        "device": device_name,
        "resume": resume,
    }


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
            steps=body["steps"],
            batch_size=body["batch_size"],
            config=config,
            learning_rate=body["learning_rate"],
            max_tokens=body["max_tokens"],
            device_name=body["device"],
            seed=body["seed"],
            resume=body["resume"],
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
    body = _validated_training_request(body)
    with _job_lock:
        if _worker is not None and _worker.is_alive():
            raise ValueError("SYNTRA training is already running.")
        _job.update({
            "status": "starting", "step": 0, "max_steps": body["steps"],
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
            steps=body["steps"],
            batch_size=body["batch_size"],
            gradient_accumulation_steps=body["gradient_accumulation_steps"],
            learning_rate=body["learning_rate"],
            device_name=body["device"],
            callback=_set_job,
            resume=body["resume"],
        )
        _set_job(result)
    except Exception as exc:
        current = get_job()
        current.update({"status": "failed", "finished_at": time.time(), "message": str(exc)})
        _set_job(current)


def start_finetuning(body: dict) -> dict:
    global _worker
    worker_body = _validated_finetuning_request(body)
    with _job_lock:
        if _worker is not None and _worker.is_alive():
            raise ValueError("SYNTRA training is already running.")
        _job.update({
            "status": "starting", "step": 0, "max_steps": worker_body["steps"],
            "train_loss": None, "validation_loss": None,
            "message": (
                "Resuming the saved supervised fine-tuning stage."
                if worker_body["resume"] else "Supervised instruction-tuning worker is starting."
            )
        })
        _worker = Thread(
            target=_run_finetuning, args=(worker_body,),
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

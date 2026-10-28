"""Local-only FastAPI service for SYNTRA Studio.

The default server address is loopback. No model is downloaded; the API can
prepare a local corpus, launch training in a background thread, and generate
text from a checkpoint built by SYNTRA itself.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from syntra.config import SyntraConfig
from syntra.data import prepare_corpus, read_token_stream
from syntra.evaluation import load_checkpoint
from syntra.generation import generate_text
from syntra.training import train as train_model

app = FastAPI(
    title="SYNTRA Local Runtime",
    version="0.1.0",
    description="Local training and inference API for SYNTRA, a model trained from random initialization.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_runtime_lock = threading.Lock()
_runtime: dict[str, Any] = {}
_training_lock = threading.Lock()
_training_state: dict[str, Any] = {
    "status": "idle",
    "step": 0,
    "max_steps": 0,
    "train_loss": None,
    "validation_loss": None,
    "message": "Training has not been started.",
    "started_at": None,
    "finished_at": None,
}


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: int = Field(default=40, ge=1, le=260)


def checkpoint_path() -> Path:
    return Path(os.environ.get("SYNTRA_CHECKPOINT", "checkpoints/latest.pt"))


def get_runtime() -> dict[str, Any]:
    path = checkpoint_path().resolve()
    with _runtime_lock:
        if _runtime.get("path") == str(path) and _runtime.get("model") is not None:
            return _runtime
        model, config, device, payload = load_checkpoint(path)
        _runtime.clear()
        _runtime.update({
            "path": str(path),
            "model": model,
            "config": config,
            "device": device,
            "step": int(payload.get("step", 0)),
        })
        return _runtime


def _progress_update(metrics: dict[str, float | int]) -> None:
    with _training_lock:
        _training_state.update(
            status="running",
            step=int(metrics["step"]),
            train_loss=float(metrics["train_loss"]),
            validation_loss=float(metrics["validation_loss"]),
            message="Training is running; metrics update at each evaluation step.",
        )


def _training_worker() -> None:
    try:
        config = SyntraConfig.load("config/default.json")
        train_model(config, progress_callback=_progress_update)
        with _training_lock:
            _training_state.update(
                status="completed",
                step=config.max_steps,
                max_steps=config.max_steps,
                message="Training completed. Latest checkpoint was saved locally.",
                finished_at=time.time(),
            )
        with _runtime_lock:
            _runtime.clear()
    except Exception as exc:
        with _training_lock:
            _training_state.update(
                status="failed",
                message=type(exc).__name__ + ": " + str(exc),
                finished_at=time.time(),
            )


def _training_snapshot() -> dict[str, Any]:
    with _training_lock:
        return dict(_training_state)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "SYNTRA Local Runtime"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    path = checkpoint_path().resolve()
    loaded = _runtime.get("model") is not None and _runtime.get("path") == str(path)
    return {
        "name": "SYNTRA",
        "mode": "local-only",
        "checkpoint": str(path),
        "checkpoint_exists": path.is_file(),
        "model_loaded": loaded,
        "step": _runtime.get("step") if loaded else None,
        "device": str(_runtime.get("device")) if loaded else None,
        "pretrained_weights": False,
        "external_ai_api": False,
    }


@app.get("/api/train/status")
def training_status() -> dict[str, Any]:
    return _training_snapshot()


@app.post("/api/data/prepare")
def prepare_data() -> dict[str, int | str]:
    if _training_snapshot()["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot re-prepare data while training is running.")
    try:
        return prepare_corpus("data/raw", "data/processed/tokens.bin")
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train/start")
def start_training() -> dict[str, Any]:
    config_path = Path("config/default.json")
    try:
        config = SyntraConfig.load(config_path)
        read_token_stream(config.data_file)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _training_lock:
        if _training_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Training is already running.")
        _training_state.update({
            "status": "running",
            "step": 0,
            "max_steps": config.max_steps,
            "train_loss": None,
            "validation_loss": None,
            "message": "Training process started.",
            "started_at": time.time(),
            "finished_at": None,
        })
    thread = threading.Thread(target=_training_worker, daemon=True, name="syntra-training")
    thread.start()
    return _training_snapshot()


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    if not checkpoint_path().is_file():
        raise HTTPException(
            status_code=503,
            detail="No trained checkpoint found. Train SYNTRA first, then retry.",
        )
    try:
        runtime = get_runtime()
        output = generate_text(
            runtime["model"],
            request.prompt,
            runtime["config"],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=min(request.top_k, runtime["config"].vocab_size),
            device=runtime["device"],
        )
        return {
            "output": output,
            "step": runtime["step"],
            "device": str(runtime["device"]),
            "warning": "Output quality depends on the amount and quality of SYNTRA training.",
        }
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

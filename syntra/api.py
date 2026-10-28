"""Minimal local FastAPI service for SYNTRA Studio.

The default server address is loopback only. No model is downloaded and all
generation is performed by a local checkpoint created by SYNTRA training.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from syntra.evaluation import load_checkpoint
from syntra.generation import generate_text

app = FastAPI(
    title="SYNTRA Local Runtime",
    version="0.1.0",
    description="Local inference API for a SYNTRA model trained from random initialization.",
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

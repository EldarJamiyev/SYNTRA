"""Cross-process admission for SYNTRA's local native model compute.

Inference callers lease one of four OS-backed slots. Training/fine-tuning acquire
all four slots, excluding every inference process for this OS user. The lock
files persist, but operating-system locks are released automatically on exit.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from inspect import isgeneratorfunction, unwrap
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator
import os
import re
import stat

from filelock import FileLock, Timeout as FileLockTimeout

MAX_CONCURRENT_INFERENCE = 4
_DEFAULT_INFERENCE_CAPACITY = 1
_capacity_lock = Lock()
_inference_capacity = _DEFAULT_INFERENCE_CAPACITY
_local_slots = tuple(Lock() for _ in range(MAX_CONCURRENT_INFERENCE))


class ComputeBusyError(RuntimeError):
    """Raised when another native SYNTRA operation owns the local compute budget."""


def configure_compute_inference_capacity(value: int) -> int:
    """Set this process's inference lease count; training always reserves all slots."""
    if type(value) is not int or not 1 <= value <= MAX_CONCURRENT_INFERENCE:
        raise ValueError("Native inference capacity must be an integer between 1 and 4.")
    global _inference_capacity
    with _capacity_lock:
        _inference_capacity = value
    return value


def _configured_inference_capacity() -> int:
    raw = os.environ.get("SYNTRA_MAX_CONCURRENT_INFERENCE")
    if raw is None:
        with _capacity_lock:
            return _inference_capacity
    if not isinstance(raw, str) or not re.fullmatch(r"[1-4]", raw):
        raise ValueError("SYNTRA_MAX_CONCURRENT_INFERENCE must be an integer between 1 and 4.")
    return int(raw)


def _lock_root() -> Path:
    """Resolve a private lock directory shared by this user's server and CLI processes."""
    configured = os.environ.get("SYNTRA_COMPUTE_LOCK_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".syntra" / "runtime-locks"
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = root.lstat()
    except OSError as exc:
        raise RuntimeError("Cannot initialize the native compute lock directory: " + str(exc)) from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("The native compute lock path must be a real directory, not a symlink.")
    if os.name != "nt":
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(
                "The native compute lock directory must be owned by the current user "
                "and inaccessible to group/other users."
            )
    return root


def _lock_path(slot: int) -> Path:
    return _lock_root() / f"inference-slot-{slot}.lock"


@contextmanager
def inference_compute_slot() -> Iterator[None]:
    """Acquire one non-queued inference lease shared with every SYNTRA process."""
    capacity = _configured_inference_capacity()
    for slot in range(capacity):
        local = _local_slots[slot]
        if not local.acquire(blocking=False):
            continue
        try:
            file_lock = FileLock(str(_lock_path(slot)))
            file_lock.acquire(timeout=0)
        except FileLockTimeout:
            local.release()
            continue
        except Exception:
            local.release()
            raise
        try:
            yield
        finally:
            try:
                file_lock.release()
            finally:
                local.release()
        return
    raise ComputeBusyError(
        "Native compute capacity is busy in another request or SYNTRA process; retry after it finishes."
    )


@contextmanager
def exclusive_training_compute() -> Iterator[None]:
    """Acquire every inference slot so training cannot overlap native inference."""
    local_acquired: list[Lock] = []
    file_locks: list[FileLock] = []
    try:
        for local in _local_slots:
            if not local.acquire(blocking=False):
                raise ComputeBusyError(
                    "Native compute is busy with inference; retry training after active requests finish."
                )
            local_acquired.append(local)
        for slot in range(MAX_CONCURRENT_INFERENCE):
            file_lock = FileLock(str(_lock_path(slot)))
            try:
                file_lock.acquire(timeout=0)
            except FileLockTimeout:
                raise ComputeBusyError(
                    "Native compute is busy in another SYNTRA process; retry training after it finishes."
                ) from None
            file_locks.append(file_lock)
        yield
    finally:
        release_error: Exception | None = None
        for file_lock in reversed(file_locks):
            try:
                file_lock.release()
            except Exception as exc:
                if release_error is None:
                    release_error = exc
        for local in reversed(local_acquired):
            local.release()
        if release_error is not None:
            raise release_error


def inference_compute_guard(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate sync or streaming inference and hold its lease for full execution."""
    if isgeneratorfunction(unwrap(function)):
        @wraps(function)
        def guarded_generator(*args: Any, **kwargs: Any) -> Iterator[Any]:
            with inference_compute_slot():
                yield from function(*args, **kwargs)
        return guarded_generator

    @wraps(function)
    def guarded_call(*args: Any, **kwargs: Any) -> Any:
        with inference_compute_slot():
            return function(*args, **kwargs)
    return guarded_call


def exclusive_compute_guard(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate the entire source-training or SFT operation with exclusive admission."""
    @wraps(function)
    def guarded_call(*args: Any, **kwargs: Any) -> Any:
        with exclusive_training_compute():
            return function(*args, **kwargs)
    return guarded_call

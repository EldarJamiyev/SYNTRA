"""Cross-process exclusion for native SYNTRA model compute."""

from __future__ import annotations

from contextvars import ContextVar, Token
from functools import wraps
from inspect import isgeneratorfunction
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator, Literal, TypeVar
import errno
import getpass
import hashlib
import os
import stat
import tempfile


ComputeMode = Literal["inference", "training"]
T = TypeVar("T")

_ACTIVE_LEASE: ContextVar[tuple[ComputeMode, int, int] | None] = ContextVar(
    "syntra_active_compute_lease",
    default=None,
)
_STATE_LOCK = RLock()
_STATE_PID = os.getpid()
_STATE_FD: int | None = None
_STATE_MODE: ComputeMode | None = None
_STATE_COUNT = 0
_STATE_GENERATION = 0


class ComputeBusyError(RuntimeError):
    """Raised when another native operation owns the local compute reservation."""


class _LockAlreadyHeld(Exception):
    """Internal signal for an OS-level lock currently owned by another process."""


def _lock_directory() -> Path:
    """Create a private, per-user directory for the persistent OS lock file."""
    username = os.environ.get("USERNAME") or os.environ.get("USER") or getpass.getuser()
    user_key = hashlib.sha256(username.encode("utf-8", "surrogatepass")).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()).resolve() / ("syntra-native-compute-" + user_key)
    directory.mkdir(mode=0o700, parents=False, exist_ok=True)

    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise OSError("SYNTRA compute lock directory must be a real directory.")
    if os.name != "nt":
        os.chmod(directory, 0o700)
    return directory


def _open_lock_file() -> int:
    """Open a no-follow, single-link lock file in the private per-user directory."""
    lock_path = _lock_directory() / "native-compute.lock"
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or opened.st_nlink != 1
            or current.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise OSError("SYNTRA compute lock path is not a stable regular file.")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        if opened.st_size == 0:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _lock_file(descriptor: int) -> None:
    """Acquire one non-blocking exclusive byte/file lock on Windows or POSIX."""
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise _LockAlreadyHeld from None
            raise
        return

    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise _LockAlreadyHeld from None
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise _LockAlreadyHeld from None
        raise


def _unlock_file(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _after_fork_in_child() -> None:
    """Discard inherited process-local bookkeeping without releasing the parent's lock."""
    global _STATE_LOCK, _STATE_PID, _STATE_FD, _STATE_MODE, _STATE_COUNT, _STATE_GENERATION
    inherited = _STATE_FD
    if inherited is not None:
        try:
            os.close(inherited)
        except OSError:
            pass
    _STATE_LOCK = RLock()
    _STATE_PID = os.getpid()
    _STATE_FD = None
    _STATE_MODE = None
    _STATE_COUNT = 0
    _STATE_GENERATION = 0
    _ACTIVE_LEASE.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class ComputeLease:
    """A reserved compute operation that may be transferred to a worker thread."""

    def __init__(self, mode: ComputeMode, generation: int, pid: int):
        self.mode = mode
        self.generation = generation
        self.pid = pid
        self._released = False
        self._context_token: Token[tuple[ComputeMode, int, int] | None] | None = None

    def __enter__(self) -> "ComputeLease":
        if self._released:
            raise RuntimeError("A released SYNTRA compute lease cannot be entered.")
        if self._context_token is not None:
            raise RuntimeError("A SYNTRA compute lease cannot be entered more than once.")
        self._context_token = _ACTIVE_LEASE.set((self.mode, self.generation, self.pid))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._context_token is not None:
            _ACTIVE_LEASE.reset(self._context_token)
            self._context_token = None
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        _release_lease(self.mode, self.generation, self.pid)


def _ensure_process_state() -> None:
    """Reset inherited lock descriptors if code is called in a forked child."""
    global _STATE_PID, _STATE_FD, _STATE_MODE, _STATE_COUNT, _STATE_GENERATION
    current_pid = os.getpid()
    if current_pid == _STATE_PID:
        return
    inherited = _STATE_FD
    if inherited is not None:
        try:
            os.close(inherited)
        except OSError:
            pass
    _STATE_PID = current_pid
    _STATE_FD = None
    _STATE_MODE = None
    _STATE_COUNT = 0
    _STATE_GENERATION = 0


def acquire_compute_lease(mode: ComputeMode) -> ComputeLease:
    """Reserve native compute or fail immediately when another operation owns it.

    Multiple inference calls in this process share one OS reservation, preserving
    SYNTRA_MAX_CONCURRENT_INFERENCE. Training is exclusive against all inference.
    Other processes are excluded by a kernel-managed file lock, not a stale PID file.
    """
    if mode not in ("inference", "training"):
        raise ValueError("Compute mode must be inference or training.")

    global _STATE_FD, _STATE_MODE, _STATE_COUNT, _STATE_GENERATION
    with _STATE_LOCK:
        _ensure_process_state()

        if _STATE_MODE == "training":
            raise ComputeBusyError(
                "Native training is active in this SYNTRA process; retry after it finishes."
            )
        if mode == "training" and _STATE_MODE == "inference":
            raise ComputeBusyError(
                "Native inference is active in this SYNTRA process; retry training after it finishes."
            )
        if mode == "inference" and _STATE_MODE == "inference":
            _STATE_COUNT += 1
            return ComputeLease(mode, _STATE_GENERATION, _STATE_PID)

        descriptor = _open_lock_file()
        try:
            _lock_file(descriptor)
        except _LockAlreadyHeld:
            os.close(descriptor)
            raise ComputeBusyError(
                "Native compute is active in another SYNTRA process; retry after that process finishes."
            ) from None
        except Exception:
            os.close(descriptor)
            raise

        _STATE_FD = descriptor
        _STATE_MODE = mode
        _STATE_COUNT = 1
        _STATE_GENERATION += 1
        return ComputeLease(mode, _STATE_GENERATION, _STATE_PID)


def _release_lease(mode: ComputeMode, generation: int, pid: int) -> None:
    global _STATE_FD, _STATE_MODE, _STATE_COUNT
    with _STATE_LOCK:
        if pid != os.getpid() or pid != _STATE_PID or generation != _STATE_GENERATION:
            return
        if _STATE_MODE != mode or _STATE_COUNT < 1:
            raise RuntimeError("SYNTRA compute lease state is inconsistent.")
        _STATE_COUNT -= 1
        if _STATE_COUNT:
            return

        descriptor = _STATE_FD
        _STATE_FD = None
        _STATE_MODE = None
        if descriptor is not None:
            try:
                _unlock_file(descriptor)
            finally:
                os.close(descriptor)


def compute_operation(mode: ComputeMode):
    """Guard a model operation; generator locks remain held until iteration ends."""
    if mode not in ("inference", "training"):
        raise ValueError("Compute mode must be inference or training.")

    def decorate(function: Callable[..., T]):
        if isgeneratorfunction(function):
            @wraps(function)
            def guarded_generator(*args, **kwargs) -> Iterator:
                active = _ACTIVE_LEASE.get()
                if active is not None:
                    if active[0] != mode:
                        raise ComputeBusyError(
                            "The requested native compute operation conflicts with the active operation."
                        )
                    yield from function(*args, **kwargs)
                    return
                with acquire_compute_lease(mode):
                    yield from function(*args, **kwargs)
            return guarded_generator

        @wraps(function)
        def guarded_call(*args, **kwargs):
            active = _ACTIVE_LEASE.get()
            if active is not None:
                if active[0] != mode:
                    raise ComputeBusyError(
                        "The requested native compute operation conflicts with the active operation."
                    )
                return function(*args, **kwargs)
            with acquire_compute_lease(mode):
                return function(*args, **kwargs)
        return guarded_call

    return decorate

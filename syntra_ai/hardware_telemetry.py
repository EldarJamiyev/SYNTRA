"""Portable process/device high-water telemetry for local PyTorch training.

Values are measurements from the current process and selected allocator. They are
not estimates and unavailable platform metrics remain None.
"""
from __future__ import annotations

import os
import sys
from typing import Any


def process_peak_rss_bytes() -> int | None:
    """Return the process peak resident working set in bytes when supported."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else None
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux and the other supported Unix CI targets report KiB.
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, AttributeError, OSError, ValueError, OverflowError):
        return None


def reset_device_peak_stats(device: Any, torch_module: Any) -> None:
    """Reset CUDA allocator peaks; other backends are sampled as observed high-water."""
    try:
        selected = torch_module.device(device)
        if selected.type == "cuda" and torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats(selected)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def device_memory_sample(device: Any, torch_module: Any) -> tuple[int | None, str]:
    """Return allocator peak/current bytes and the exact measurement method."""
    try:
        selected = torch_module.device(device)
        if selected.type == "cuda" and torch_module.cuda.is_available():
            return int(torch_module.cuda.max_memory_allocated(selected)), "cuda_peak_allocated"
        if selected.type == "mps":
            mps = getattr(torch_module, "mps", None)
            sample = getattr(mps, "driver_allocated_memory", None)
            if callable(sample):
                return int(sample()), "mps_observed_driver_high_water"
            sample = getattr(mps, "current_allocated_memory", None)
            if callable(sample):
                return int(sample()), "mps_observed_allocator_high_water"
            return None, "mps_unavailable"
        if selected.type == "cpu":
            return None, "cpu_not_applicable"
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        return None, "device_memory_unavailable"
    return None, "device_memory_unavailable"

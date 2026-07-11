#!/usr/bin/env python3
"""Memory helpers for batch factor materialize on ~30G RAM / no swap."""
from __future__ import annotations

import gc
from typing import Any


def read_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024**2)
    except OSError:
        pass
    return 8.0


def read_mem_total_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except OSError:
        pass
    return 30.0


def release_memory(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    try:
        from data_access import reset_store  # type: ignore

        reset_store()
    except ImportError:
        pass


def ensure_memory_floor(min_gb: float = 4.0) -> None:
    avail = read_mem_available_gb()
    if avail < min_gb:
        release_memory()
        avail = read_mem_available_gb()
    if avail < min_gb:
        raise MemoryError(
            f"available memory {avail:.1f}GB below floor {min_gb:.1f}GB; "
            "pause batch or free memory before continuing"
        )

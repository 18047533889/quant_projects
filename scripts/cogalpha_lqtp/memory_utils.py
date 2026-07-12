#!/usr/bin/env python3
"""Memory helpers for batch factor materialize on ~30G RAM / no swap."""
from __future__ import annotations

import gc
import time
from typing import Any

# Keep headroom so SSH / OS stay responsive (no swap on this host).
DEFAULT_MEM_RESERVE_GB = 10.0
LQTP_JOB_MEM_GB = 5.0
LOCAL_MATERIALIZE_MEM_GB = 10.0
LOCAL_REUSE_MEM_GB = 5.0
DEFAULT_MAX_TOTAL_PARALLEL = 4


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


def wait_for_memory(
    *,
    required_gb: float,
    reserve_gb: float = DEFAULT_MEM_RESERVE_GB,
    timeout_sec: float = 600.0,
    poll_sec: float = 5.0,
) -> None:
    """Block until MemAvailable >= required_gb + reserve_gb."""
    deadline = time.monotonic() + timeout_sec
    target = required_gb + reserve_gb
    while True:
        avail = read_mem_available_gb()
        if avail >= target:
            return
        if time.monotonic() >= deadline:
            raise MemoryError(
                f"timeout waiting for memory: avail={avail:.1f}G need>={target:.1f}G "
                f"(required={required_gb:.1f}G reserve={reserve_gb:.1f}G)"
            )
        release_memory()
        time.sleep(poll_sec)


def estimate_local_parallel_workers(
    *,
    mem_avail_gb: float,
    mem_reserve_gb: float = DEFAULT_MEM_RESERVE_GB,
    n_materialize: int,
    n_reuse: int,
    mem_per_materialize_gb: float = LOCAL_MATERIALIZE_MEM_GB,
    mem_per_reuse_gb: float = LOCAL_REUSE_MEM_GB,
    max_workers: int = 2,
) -> int:
    """How many local-engine factor jobs can run concurrently without OOM."""
    if n_materialize + n_reuse == 0:
        return 0
    headroom = max(0.0, mem_avail_gb - mem_reserve_gb)
    if headroom < mem_per_reuse_gb:
        return 1
    mat_slots = int(headroom / mem_per_materialize_gb) if n_materialize else 0
    reuse_slots = int(headroom / mem_per_reuse_gb) if n_reuse else 0
    if n_materialize and n_reuse:
        workers = max(1, min(max_workers, mat_slots + reuse_slots, n_materialize + n_reuse))
    elif n_materialize:
        workers = max(1, mat_slots)
    else:
        workers = max(1, reuse_slots)
    return max(1, min(max_workers, workers, n_materialize + n_reuse))


def estimate_lqtp_parallel_workers(
    *,
    pending_count: int,
    mem_avail_gb: float,
    mem_reserve_gb: float = DEFAULT_MEM_RESERVE_GB,
    mem_per_job_gb: float = LQTP_JOB_MEM_GB,
    default: int = 3,
    max_workers: int = 4,
) -> int:
    """RunFactor still pulls ~8M rows locally for backtest; cap by RAM."""
    if pending_count <= 0:
        return 0
    headroom = max(0.0, mem_avail_gb - mem_reserve_gb)
    by_mem = max(1, int(headroom / mem_per_job_gb))
    return max(1, min(max_workers, default, pending_count, by_mem))


def cap_combined_parallel(
    *,
    lqtp_workers: int,
    local_workers: int,
    has_lqtp: bool,
    has_local: bool,
    max_total: int = DEFAULT_MAX_TOTAL_PARALLEL,
) -> tuple[int, int]:
    """Ensure lqtp + local concurrent processes stay within a safe total."""
    if not has_lqtp:
        lqtp_workers = 0
    if not has_local:
        local_workers = 0
    total = lqtp_workers + local_workers
    if total <= max_total:
        return lqtp_workers, local_workers
    if has_lqtp and has_local:
        # Prefer 1 local materialize slot + rest for lqtp, but keep both >0 when possible.
        local_workers = min(local_workers, max(1, max_total // 2))
        lqtp_workers = min(lqtp_workers, max_total - local_workers)
        if lqtp_workers < 1 and has_lqtp:
            lqtp_workers = 1
            local_workers = max(0, max_total - 1)
    elif has_lqtp:
        lqtp_workers = min(lqtp_workers, max_total)
    else:
        local_workers = min(local_workers, max_total)
    return lqtp_workers, local_workers

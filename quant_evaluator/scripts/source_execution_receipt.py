"""Validate actual whole-source schedules for zero-OOM benchmark receipts.

Request/API caps are admission limits, not the device's chosen working width.
The actual source width also cannot exceed the number of requested factors.
This verifies coverage and execution shape, not metric correctness or timing.
"""
from __future__ import annotations

from collections.abc import Mapping

from quant_evaluator.contracts._hashutil import stable_content_hex


def validate_source_execution_receipt(
    *, reads, factor_count: int, admitted_cap: int, metadata: Mapping,
) -> dict:
    if type(factor_count) is not int or factor_count < 1:
        raise ValueError("factor count must be a positive integer")
    if type(admitted_cap) is not int or admitted_cap < 1:
        raise ValueError("admitted source cap must be a positive integer")
    if not isinstance(metadata, Mapping):
        raise ValueError("execution metadata must be a mapping")
    backend = metadata.get("backend_used")
    if type(backend) is not str or backend not in ("cpu", "cuda"):
        raise ValueError("actual execution backend is missing or invalid")
    if backend == "cuda":
        width = metadata.get("factor_tile_size")
        oom = metadata.get("oom_retries")
        if type(oom) is not int or oom != 0:
            raise ValueError("CUDA qualification requires an integer zero-OOM receipt")
    else:
        # A final partial tile is not a larger actual working width. In
        # particular, a five-factor request admitted at cap 16 runs one
        # five-factor tile, not a sixteen-factor tile.
        width = min(admitted_cap, factor_count)
        oom = metadata.get("oom_retries", 0)
        if type(oom) is not int or oom != 0:
            raise ValueError("CPU qualification requires a zero-OOM receipt")
    if (type(width) is not int or not 1 <= width <= admitted_cap
            or width > factor_count):
        raise ValueError("actual execution width is outside source admission")
    expected = tuple((start, min(start + width, factor_count))
                     for start in range(0, factor_count, width))
    if type(reads) not in (list, tuple):
        raise ValueError("actual source schedule is missing")
    if (any(type(item) is not tuple or len(item) != 2
            or any(type(v) is not int for v in item) for item in reads)
            or tuple(reads) != expected):
        raise ValueError("source API did not read exact ordered tile coverage")
    count = metadata.get("factor_tiles_processed")
    if type(count) is not int or count != len(expected):
        raise ValueError("source execution count does not cover the whole request")
    # Current zero-OOM GPU executor reads each tile at its compute width.
    # Future split-source implementations need separate compute-range receipts.
    digest = stable_content_hex(tag="SourceExecutionSchedule.v1", fields={
        "backend": backend, "factor_count": factor_count,
        "source_width": width, "source_ranges": expected,
        "compute_ranges": expected, "oom_retries": oom,
    })
    return {
        "actual_source_tile_size": width,
        "actual_gpu_factor_tile_size": width if backend == "cuda" else None,
        "execution_schedule_sha256": digest,
        "execution_schedule_scope": "zero_oom_source_equals_compute_v1",
        "compute_tile_ranges": expected,
    }


__all__ = ("validate_source_execution_receipt",)

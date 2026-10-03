"""Bounded in-memory A/B harness for the opt-in FE native rank candidate.

The caller must supply an already-loaded ndarray tile. This module performs no
data loading, receipt publication, registry bootstrap, or production routing.
Its timings are candidate measurements only; they do not establish a fastest
implementation or real-data provenance.
"""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
import platform
import sys
from math import prod
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from factor_preprocess.adapters.fe_rank import (
    cs_rank_fe_native as _fe_cs_rank,
    execution_identity as _fe_execution_identity,
)
from factor_preprocess.transforms.cross_sectional import cs_rank as _fp_cs_rank

_DEFAULT_MAX_RESULT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_CHUNK_CELLS = 1_000_000
_HASH_BUFFER_ELEMENTS = 65_536
_MAX_SOURCE_BYTES = 1024 * 1024
_ABBA_ORDER = ("FP", "FE", "FE", "FP")


def _positive_integral(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _module_source_identity(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    with source_path.open("rb") as stream:
        content = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(content) > _MAX_SOURCE_BYTES:
        raise ValueError(f"benchmark source exceeds the {_MAX_SOURCE_BYTES}-byte read bound")
    return {
        "path": str(source_path),
        "bytes": len(content),
        "sha256": sha256(content).hexdigest(),
    }


def _function_source_path(function: Callable) -> str:
    path = inspect.getsourcefile(function)
    if path is None:
        code = getattr(function, "__code__", None)
        path = getattr(code, "co_filename", None)
    if not path:
        raise ValueError(f"cannot identify source file for {function!r}")
    return path


def _iter_flat_blocks(values: np.ndarray):
    iterator = np.nditer(
        values,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=_HASH_BUFFER_ELEMENTS,
    )
    for block in iterator:
        yield block


def _input_fingerprint(values: np.ndarray) -> dict[str, Any]:
    content_hash = sha256()
    nan_mask_hash = sha256()
    metadata = {
        "shape": [int(size) for size in values.shape],
        "dtype": values.dtype.str,
        "strides": [int(stride) for stride in values.strides],
    }
    content_hash.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    nan_count = 0
    for block in _iter_flat_blocks(values):
        content_hash.update(block.tobytes(order="C"))
        if values.dtype.kind == "f":
            nan_mask = np.isnan(block)
            nan_count += int(np.count_nonzero(nan_mask))
        else:
            nan_mask = np.zeros(block.shape, dtype=np.bool_)
        nan_mask_hash.update(nan_mask.view(np.uint8).tobytes(order="C"))
    return {
        **metadata,
        "content_sha256": content_hash.hexdigest(),
        "nan_mask_sha256": nan_mask_hash.hexdigest(),
        "nan_count": nan_count,
        "nbytes": int(values.nbytes),
    }


def _new_output_hashers(shape: tuple[int, ...]):
    prefix = json.dumps(
        {"shape": [int(size) for size in shape], "dtype": np.dtype(np.float64).str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    value_hash = sha256(prefix)
    nan_mask_hash = sha256(prefix)
    return value_hash, nan_mask_hash, 0, 0


def _feed_output_hashers(hashers, output: np.ndarray):
    value_hash, nan_mask_hash, nan_count, cell_count = hashers
    for block in _iter_flat_blocks(output):
        nan_mask = np.isnan(block)
        nan_count += int(np.count_nonzero(nan_mask))
        cell_count += int(block.size)
        canonical_values = block.copy()
        canonical_values[nan_mask] = 0.0
        value_hash.update(canonical_values.tobytes(order="C"))
        nan_mask_hash.update(nan_mask.view(np.uint8).tobytes(order="C"))
    return value_hash, nan_mask_hash, nan_count, cell_count


def _finish_output_hashers(hashers) -> dict[str, Any]:
    value_hash, nan_mask_hash, nan_count, cell_count = hashers
    return {
        "shape": None,
        "dtype": np.dtype(np.float64).str,
        "values_sha256": value_hash.hexdigest(),
        "nan_mask_sha256": nan_mask_hash.hexdigest(),
        "nan_count": nan_count,
        "cell_count": cell_count,
    }


def _output_fingerprint(output: np.ndarray) -> dict[str, Any]:
    hashers = _new_output_hashers(output.shape)
    result = _finish_output_hashers(_feed_output_hashers(hashers, output))
    result["shape"] = [int(size) for size in output.shape]
    return result


def _runtime_fingerprint() -> dict[str, Any]:
    import polars as pl
    import scipy
    import threadpoolctl

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "polars_thread_pool_size": int(pl.thread_pool_size()),
        "scipy": scipy.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "threadpools": [
            {
                "internal_api": pool.get("internal_api"),
                "prefix": pool.get("prefix"),
                "version": pool.get("version"),
                "num_threads": pool.get("num_threads"),
            }
            for pool in threadpoolctl.threadpool_info()
        ],
    }


def _source_fingerprints() -> dict[str, Any]:
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    fe_identity = _fe_execution_identity()
    return {
        "benchmark_harness": _module_source_identity(_function_source_path(_source_fingerprints)),
        "fp_transform": _module_source_identity(_function_source_path(_fp_cs_rank)),
        "fe_adapter": fe_identity["modules"]["adapter"],
        "fe_rank_helper": fe_identity["modules"]["fe_helper"],
        "fe_helper_symbol": getattr(polars_cs_rank_expr, "__qualname__", ""),
    }


def _measurement_context(values: np.ndarray) -> dict[str, Any]:
    return {
        "input": _input_fingerprint(values),
        "runtime": _runtime_fingerprint(),
        "sources": _source_fingerprints(),
    }


def _context_digest(context: dict[str, Any]) -> str:
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _outer_group_count(values: np.ndarray) -> int:
    group_shape = tuple(int(size) for size in values.shape[:-1])
    return prod(group_shape) if group_shape else 1


def _iter_group_tiles(values: np.ndarray, groups_per_tile: int):
    width = int(values.shape[-1])
    group_shape = tuple(int(size) for size in values.shape[:-1])
    group_count = _outer_group_count(values)
    for start in range(0, group_count, groups_per_tile):
        stop = min(start + groups_per_tile, group_count)
        if group_shape:
            coordinates = np.unravel_index(np.arange(start, stop, dtype=np.int64), group_shape)
            index = tuple(coordinates) + (slice(None),)
            tile = values[index]
        else:
            tile = values.reshape(1, width)
        yield tile


def _validate_inputs(
    values: Any,
    *,
    max_chunk_cells: Any,
    max_result_bytes: Any,
) -> tuple[np.ndarray, int, int, int]:
    max_chunk_cells = _positive_integral(max_chunk_cells, "max_chunk_cells")
    max_result_bytes = _positive_integral(max_result_bytes, "max_result_bytes")
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be an externally loaded NumPy ndarray")
    if values.ndim == 0:
        raise ValueError("benchmark input must have at least one axis")
    if values.size == 0 or values.shape[-1] == 0:
        raise ValueError("benchmark input must be non-empty")
    if values.dtype.kind not in "biuf":
        raise TypeError("benchmark input must use a real numeric dtype")
    if values.dtype.kind == "f" and values.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("floating input wider than Float64 is unsupported")
    width = int(values.shape[-1])
    if width > max_chunk_cells:
        raise ValueError(
            f"one complete cross-section has {width} cells, exceeding "
            f"max_chunk_cells={max_chunk_cells}"
        )
    full_result_bytes = int(values.size) * np.dtype(np.float64).itemsize
    if full_result_bytes > max_result_bytes:
        raise MemoryError(
            f"one full result allocation requires {full_result_bytes} bytes, "
            f"exceeding max_result_bytes={max_result_bytes}"
        )
    min_pair_bytes = 2 * width * np.dtype(np.float64).itemsize
    if min_pair_bytes > max_result_bytes:
        raise MemoryError(
            f"oracle comparison needs at least {min_pair_bytes} simultaneous result bytes, "
            f"exceeding max_result_bytes={max_result_bytes}"
        )
    groups_per_tile = max_result_bytes // min_pair_bytes
    return values, max_chunk_cells, max_result_bytes, groups_per_tile


def _oracle_check(
    values: np.ndarray,
    *,
    max_chunk_cells: int,
    max_result_bytes: int,
    groups_per_tile: int,
) -> dict[str, Any]:
    width = int(values.shape[-1])
    fp_hashers = _new_output_hashers(values.shape)
    fe_hashers = _new_output_hashers(values.shape)
    verified = 0
    for tile in _iter_group_tiles(values, groups_per_tile):
        tile_bytes = int(tile.size) * np.dtype(np.float64).itemsize
        fp_result = _fp_cs_rank(tile, method="average", pct=True, axis=-1)
        fe_result = _fe_cs_rank(
            tile,
            method="average",
            pct=True,
            axis=-1,
            max_chunk_cells=max_chunk_cells,
            max_result_bytes=tile_bytes,
        )
        if fp_result.shape != tile.shape or fe_result.shape != tile.shape:
            raise AssertionError("FP oracle mismatch: output shape differs from input tile")
        fp_nan = np.isnan(fp_result)
        fe_nan = np.isnan(fe_result)
        if not np.array_equal(fp_nan, fe_nan):
            raise AssertionError("FP oracle mismatch: output NaN masks differ")
        valid = ~fp_nan
        if not np.array_equal(fp_result[valid], fe_result[valid]):
            raise AssertionError("FP oracle mismatch: finite output values differ")
        fp_hashers = _feed_output_hashers(fp_hashers, fp_result)
        fe_hashers = _feed_output_hashers(fe_hashers, fe_result)
        verified += int(tile.size)
        del fp_result, fe_result
    fp_fingerprint = _finish_output_hashers(fp_hashers)
    fe_fingerprint = _finish_output_hashers(fe_hashers)
    fp_fingerprint["shape"] = [int(size) for size in values.shape]
    fe_fingerprint["shape"] = [int(size) for size in values.shape]
    if fp_fingerprint != fe_fingerprint:
        raise AssertionError("FP oracle mismatch: full output fingerprints differ")
    return {
        "status": "passed",
        "oracle": "actual_FP_cs_rank_parity_baseline",
        "verified_values": verified,
        "nan_count": fp_fingerprint["nan_count"],
        "result_fingerprint": fp_fingerprint,
    }


def benchmark_fe_native_rank(
    values: np.ndarray,
    *,
    max_chunk_cells: int = _DEFAULT_MAX_CHUNK_CELLS,
    max_result_bytes: int = _DEFAULT_MAX_RESULT_BYTES,
) -> dict[str, Any]:
    """Compare actual FP rank and the opt-in FE candidate on one loaded ndarray.

    The measured order is FP/FE/FE/FP. A bounded, untimed full-value parity pass
    first treats actual FP as a parity baseline and compares every output and
    NaN position in tiles. Timed calls each retain only one full result at a
    time; parity tiles retain at most two Float64 result buffers whose combined
    size is bounded by ``max_result_bytes``. The budget covers returned result
    arrays, not input residency or hidden library working memory.

    ``values`` must already be loaded by the caller. No path, COS access, data
    copy, receipt write, or performance verdict is accepted or performed.
    """
    values, max_chunk_cells, max_result_bytes, groups_per_tile = _validate_inputs(
        values,
        max_chunk_cells=max_chunk_cells,
        max_result_bytes=max_result_bytes,
    )
    # Initialize numeric backends before snapshotting runtime/thread-pool state.
    from scipy.stats import rankdata as _rankdata  # noqa: F401
    from factor_engine.backend.rank_spec import polars_cs_rank_expr as _rank_expr  # noqa: F401

    initial_context = _measurement_context(values)
    initial_context_digest = _context_digest(initial_context)

    oracle_check = _oracle_check(
        values,
        max_chunk_cells=max_chunk_cells,
        max_result_bytes=max_result_bytes,
        groups_per_tile=groups_per_tile,
    )
    after_oracle = _measurement_context(values)
    if after_oracle != initial_context:
        raise RuntimeError("input, runtime, or source fingerprint changed during oracle validation")

    runners = {
        "FP": lambda: _fp_cs_rank(values, method="average", pct=True, axis=-1),
        "FE": lambda: _fe_cs_rank(
            values,
            method="average",
            pct=True,
            axis=-1,
            max_chunk_cells=max_chunk_cells,
            max_result_bytes=max_result_bytes,
        ),
    }
    expected_digest = oracle_check["result_fingerprint"]
    runs = []
    outputs_by_backend = {"FP": [], "FE": []}
    for backend in _ABBA_ORDER:
        before_run = _measurement_context(values)
        if before_run != initial_context:
            raise RuntimeError(f"input, runtime, or source fingerprint changed before {backend} run")
        started = perf_counter()
        result = runners[backend]()
        elapsed = perf_counter() - started
        after_run = _measurement_context(values)
        if after_run != initial_context:
            raise RuntimeError(f"input, runtime, or source fingerprint changed during {backend} run")
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"{backend} timing must be finite and strictly positive")
        if not isinstance(result, np.ndarray) or result.shape != values.shape:
            raise AssertionError(f"{backend} rank result shape does not match input")
        if result.dtype != np.dtype(np.float64):
            raise AssertionError(f"{backend} rank result must use Float64")
        result_digest = _output_fingerprint(result)
        if result_digest != expected_digest:
            raise AssertionError(f"{backend} timed output differs from full FP oracle")
        outputs_by_backend[backend].append(result_digest)
        runs.append(
            {
                "backend": backend,
                "elapsed_seconds": elapsed,
                "result_digest": result_digest["values_sha256"],
                "nan_mask_digest": result_digest["nan_mask_sha256"],
                "context_fingerprint": initial_context_digest,
            }
        )
        del result

    final_context = _measurement_context(values)
    if final_context != initial_context:
        raise RuntimeError("input, runtime, or source fingerprint changed during timing")
    for backend, fingerprints in outputs_by_backend.items():
        if fingerprints[0] != fingerprints[1]:
            raise AssertionError(f"{backend} output changed between paired ABBA runs")

    return {
        "status": "candidate_measurement",
        "production_admitted": False,
        "performance_claim": None,
        "abba_order": list(_ABBA_ORDER),
        "runs": runs,
        "backend_elapsed_seconds": {
            backend: [run["elapsed_seconds"] for run in runs if run["backend"] == backend]
            for backend in ("FP", "FE")
        },
        "oracle_check": oracle_check,
        "output_fingerprints": {
            backend: fingerprints[0] for backend, fingerprints in outputs_by_backend.items()
        },
        "input_fingerprint": initial_context["input"],
        "runtime_fingerprint": initial_context["runtime"],
        "source_fingerprints": initial_context["sources"],
        "context_fingerprint": initial_context_digest,
        "result_allocation": {
            "max_live_result_bytes": max_result_bytes,
            "full_result_bytes_per_call": int(values.size) * np.dtype(np.float64).itemsize,
            "oracle_pair_limit_bytes": max_result_bytes,
        },
        "evidence_scope": (
            "caller-supplied in-memory ndarray only; no data provenance, full-registry admission, "
            "production readiness, or fastest-implementation conclusion"
            "; runtime/module fingerprints are not a transitive or loaded-code closure proof"
        ),
    }


__all__ = ["benchmark_fe_native_rank"]

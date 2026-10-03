"""Explicit opt-in bridge to FE's native-Polars cross-sectional rank helper.

This is a candidate executor only. It does not register an FP transform, change
default routing, or claim official FE operator admission.
"""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
from math import prod
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

_ADAPTER_VERSION = "grouped-long-rank-bridge-v1"
_FE_FUNCTION = "factor_engine.backend.rank_spec.polars_cs_rank_expr"
_MAX_IDENTITY_SOURCE_BYTES = 1024 * 1024
_DEFAULT_MAX_CHUNK_CELLS = 1_000_000
_DEFAULT_MAX_RESULT_BYTES = 256 * 1024 * 1024
_GROUP = "__fp_fe_group"
_ORDER = "__fp_fe_order"
_VALUE = "__fp_fe_value"
_RANKED = "__fp_fe_ranked"


def _validated_memory_bounds(max_chunk_cells: Any, max_result_bytes: Any):
    if isinstance(max_chunk_cells, (bool, np.bool_)) or not isinstance(
        max_chunk_cells, Integral
    ) or max_chunk_cells <= 0:
        raise ValueError("max_chunk_cells must be a positive integer")
    if max_result_bytes is not None and (
        isinstance(max_result_bytes, (bool, np.bool_))
        or not isinstance(max_result_bytes, Integral)
        or max_result_bytes <= 0
    ):
        raise ValueError("max_result_bytes must be None or a positive integer")
    return int(max_chunk_cells), (
        None if max_result_bytes is None else int(max_result_bytes)
    )


def _validate_candidate_parameters(method: Any, pct: Any):
    if not isinstance(method, str) or method != "average":
        raise ValueError("FE native rank candidate supports only method='average'")
    if type(pct) is not bool or not pct:
        raise ValueError("FE native rank candidate supports only pct=True")


def _normalize_last_axis(axis: Any, ndim: int) -> None:
    if ndim == 0:
        axis_error = getattr(np, "AxisError", None)
        if axis_error is None:
            axis_error = np.exceptions.AxisError
        raise axis_error(axis, ndim=0)
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
        raise TypeError("axis must be an integer selecting the last axis")
    value = int(axis)
    if value < -ndim or value >= ndim:
        axis_error = getattr(np, "AxisError", None)
        if axis_error is None:
            axis_error = np.exceptions.AxisError
        raise axis_error(value, ndim=ndim)
    if value % ndim != ndim - 1:
        raise ValueError("FE native rank candidate supports only the last axis")


def cs_rank_fe_native(
    values: np.ndarray,
    *,
    method: str = "average",
    pct: bool = True,
    axis: int = -1,
    max_chunk_cells: int = _DEFAULT_MAX_CHUNK_CELLS,
    max_result_bytes: int | None = _DEFAULT_MAX_RESULT_BYTES,
) -> np.ndarray:
    """Rank each last-axis cross-section with FE's real Polars expression.

    Supported FP semantics are exactly ``method='average'``, ``pct=True``,
    and a cross-section along the last axis. NaN and +/-Inf are excluded from
    both ranking and the denominator. Every complete cross-section is sent to
    FE in one bounded carrier; a single cross-section wider than
    ``max_chunk_cells`` is rejected. The Float64 result allocation is bounded
    by ``max_result_bytes`` (pass ``None`` only when the caller manages output
    memory). NumPy is used only for bounded transport, indexing, and output.
    No FP numerical fallback or wide-panel conversion is performed.
    """
    _validate_candidate_parameters(method, pct)
    max_chunk_cells, max_result_bytes = _validated_memory_bounds(
        max_chunk_cells, max_result_bytes
    )
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy ndarray")
    source = values
    if source.dtype.kind not in "biuf":
        raise TypeError("values must use a real numeric NumPy dtype; complex/object data are unsupported")
    if source.dtype.kind == "f" and source.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("floating input wider than Float64 is unsupported")
    if source.ndim == 0:
        _normalize_last_axis(axis, source.ndim)
    # Match the FP kernel's early empty-input return before checking axis bounds.
    if source.size == 0:
        return source.copy()
    _normalize_last_axis(axis, source.ndim)

    width = int(source.shape[-1])
    if width > max_chunk_cells:
        raise ValueError(
            f"one complete cross-section has {width} cells, exceeding "
            f"max_chunk_cells={max_chunk_cells}"
        )
    output_bytes = int(source.size) * np.dtype(np.float64).itemsize
    if max_result_bytes is not None and output_bytes > max_result_bytes:
        raise MemoryError(
            f"Float64 output requires {output_bytes} bytes, exceeding "
            f"max_result_bytes={max_result_bytes}"
        )

    group_shape = tuple(int(size) for size in source.shape[:-1])
    group_count = prod(group_shape) if group_shape else 1
    groups_per_chunk = max(1, max_chunk_cells // width)
    output = np.empty(source.shape, dtype=np.float64)

    # Polars owns all numerical ranking; NumPy only builds bounded carriers.
    import polars as pl
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    for start in range(0, group_count, groups_per_chunk):
        stop = min(start + groups_per_chunk, group_count)
        group_ids = np.arange(start, stop, dtype=np.int64)
        if group_shape:
            coordinates = np.unravel_index(group_ids, group_shape)
            index = tuple(coordinates) + (slice(None),)
            selected = source[index]
            grouped = selected.reshape(stop - start, width)
        else:
            index = (slice(None),)
            grouped = source.reshape(1, width)

        if grouped.dtype.kind == "b" or (
            grouped.dtype.kind == "f" and grouped.dtype.itemsize < np.dtype(np.float64).itemsize
        ):
            values_for_fe = grouped.astype(np.float64, copy=False)
        else:
            # Preserve integer ordering exactly, including distinct int64/uint64
            # values above Float64's exact-integer range.
            values_for_fe = grouped
        carrier = pl.DataFrame(
            {
                _GROUP: np.repeat(np.arange(stop - start, dtype=np.int64), width),
                _ORDER: np.tile(np.arange(width, dtype=np.int64), stop - start),
                _VALUE: values_for_fe.reshape(-1),
            }
        )
        ranked = carrier.with_columns(
            polars_cs_rank_expr(
                _VALUE,
                partition_cols=(_GROUP,),
                order_by=_ORDER,
                canon="rank",
            ).alias(_RANKED)
        )[_RANKED].to_numpy()
        output[index] = ranked.reshape(stop - start, width)

    return output


def _module_source_identity(path: str | Path, *, max_bytes: int = _MAX_IDENTITY_SOURCE_BYTES):
    source_path = Path(path).resolve()
    with source_path.open("rb") as stream:
        source = stream.read(max_bytes + 1)
    if len(source) > max_bytes:
        raise ValueError(f"identity source exceeds the {max_bytes}-byte read limit")
    return {
        "path": str(source_path),
        "bytes": len(source),
        "sha256": sha256(source).hexdigest(),
    }


def execution_identity() -> dict[str, Any]:
    """Describe this optional candidate without implying production admission."""
    import polars as pl
    from factor_engine.backend.rank_spec import polars_cs_rank_expr

    helper_path = inspect.getsourcefile(polars_cs_rank_expr)
    if helper_path is None:
        helper_path = polars_cs_rank_expr.__code__.co_filename
    payload = {
        "status": "opt_in_candidate",
        "adapter": "factor_preprocess.adapters.fe_rank.cs_rank_fe_native",
        "adapter_version": _ADAPTER_VERSION,
        "fe_function": _FE_FUNCTION,
        "canonical": "rank",
        "backend": "polars_native_expr",
        "semantics": {
            "method": "average",
            "pct": True,
            "formula": "(rank - 1) / (finite_n - 1)",
            "singleton": 0.5,
            "invalid": ["null", "NaN", "+Inf", "-Inf"],
            "axis": "last",
        },
        "identity_scope": (
            "bounded on-disk adapter and FE helper module source only; not a loaded-code "
            "closure; transitive imports and runtime dependencies are not fingerprinted"
        ),
        "modules": {
            "adapter": _module_source_identity(Path(__file__)),
            "fe_helper": _module_source_identity(helper_path),
        },
        "runtime_versions": {"numpy": np.__version__, "polars": pl.__version__},
        "default_memory_bounds": {
            "max_chunk_cells": _DEFAULT_MAX_CHUNK_CELLS,
            "max_result_bytes": _DEFAULT_MAX_RESULT_BYTES,
        },
        "production_admitted": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "digest": sha256(canonical.encode("utf-8")).hexdigest()}


__all__ = ["cs_rank_fe_native", "execution_identity"]

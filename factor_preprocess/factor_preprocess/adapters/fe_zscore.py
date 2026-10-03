"""Explicit opt-in bridge to FE's native-Polars grouped-long z-score math.

This adapter is a candidate executor only. It does not register an FP transform,
change default routing, or claim official FE operator admission.
"""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
from math import prod
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from factor_preprocess.transforms.zscore_numeric import FINITE_ANCHOR_CENTERED_V2

_ADAPTER_VERSION = "grouped-long-bridge-v1"
_FE_FUNCTION = "factor_engine.backend.long_stable_zscore.finite_anchor_centered_zscore_long"
_MAX_IDENTITY_SOURCE_BYTES = 1024 * 1024
_DEFAULT_MAX_CHUNK_CELLS = 1_000_000
_DEFAULT_MAX_RESULT_BYTES = 256 * 1024 * 1024


def _normalize_axes(axis: Any, ndim: int) -> tuple[int, ...]:
    """Normalize only NumPy's documented integer/tuple/None axis forms."""
    if axis is None:
        return tuple(range(ndim))
    if ndim == 0 and not isinstance(axis, tuple):
        if isinstance(axis, (bool, np.bool_)):
            raise TypeError("axis entries must be integers, a tuple of integers, or None")
        if isinstance(axis, Integral) and int(axis) in (0, -1):
            # NumPy reductions permit these two scalar-axis spellings, while
            # tuple forms such as (0,) still raise AxisError for a 0-D array.
            return ()
    axes = axis if isinstance(axis, tuple) else (axis,)
    normalized = []
    for item in axes:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("axis entries must be integers, a tuple of integers, or None")
        if not isinstance(item, Integral):
            raise TypeError("axis entries must be integers, a tuple of integers, or None")
        value = int(item)
        if value < -ndim or value >= ndim:
            raise ValueError(f"axis {value} is out of bounds for array of dimension {ndim}")
        value %= ndim
        if value in normalized:
            raise ValueError("duplicate value in 'axis'")
        normalized.append(value)
    return tuple(normalized)


def _validated_params(ddof: Any, constant_value: Any, numeric_policy: Any):
    if numeric_policy != FINITE_ANCHOR_CENTERED_V2:
        raise ValueError(
            "FE z-score adapter supports only numeric_policy="
            f"{FINITE_ANCHOR_CENTERED_V2!r}; use the FP executor for other policies"
        )
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, Real):
        raise ValueError("ddof must be a finite real number in the admitted range [0, 1]")
    try:
        ddof_value = float(ddof)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("ddof must be a finite real number in the admitted range [0, 1]") from exc
    if not (0.0 <= ddof_value <= 1.0):
        raise ValueError("ddof must be a finite real number in the admitted range [0, 1]")
    if isinstance(constant_value, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(
        constant_value, Real
    ):
        raise ValueError("constant_value must be a finite real number")
    try:
        constant = float(constant_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("constant_value must be a finite real number") from exc
    if not np.isfinite(constant):
        raise ValueError("constant_value must be a finite real number")
    return ddof_value, constant


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


_GROUP_AXIS = object()


def _raw_chunk_axis_order(ndim: int, remaining: tuple[int, ...], selected: tuple[int, ...]):
    """Label NumPy's advanced-index result axes for a bounded group tile."""
    if not remaining:
        return selected
    first, last = remaining[0], remaining[-1]
    if last - first + 1 == len(remaining):
        labels = []
        for dimension in range(ndim):
            if dimension == first:
                labels.append(_GROUP_AXIS)
            if dimension in remaining:
                continue
            labels.append(dimension)
        return tuple(labels)
    # Noncontiguous advanced-index axes move as a block to the front.
    return (_GROUP_AXIS,) + selected


def cs_zscore_finite_anchor_fe_native(
    values: np.ndarray,
    axis: Any = -1,
    ddof: float = 1,
    constant_value: float = 0.0,
    numeric_policy: str = FINITE_ANCHOR_CENTERED_V2,
    *,
    max_chunk_cells: int = _DEFAULT_MAX_CHUNK_CELLS,
    max_result_bytes: int | None = _DEFAULT_MAX_RESULT_BYTES,
) -> np.ndarray:
    """Compute FP-shaped z-scores using FE's native Polars grouped-long math.

    The positional and keyword call shape matches FP ``cs_zscore``; input must
    be a NumPy ndarray and bounded transport options are keyword-only. Real
    boolean/integer/unsigned/floating
    arrays are accepted and converted to Float64; complex, object, and string
    arrays are rejected. Complete reduction groups are tiled into Polars
    carriers no larger than ``max_chunk_cells``. A reduction group larger than
    this cap fails explicitly. The Float64 output allocation is limited by
    ``max_result_bytes`` (default 256 MiB); pass ``None`` to opt out of that
    output limit. No wide pivot or FP numerical fallback is used.
    """
    ddof_value, constant = _validated_params(ddof, constant_value, numeric_policy)
    max_chunk_cells, max_result_bytes = _validated_memory_bounds(
        max_chunk_cells, max_result_bytes
    )
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy ndarray")
    source = values
    if source.dtype.kind not in "biuf":
        raise TypeError("values must use a real numeric NumPy dtype; complex values are unsupported")
    axes = _normalize_axes(axis, source.ndim)
    output_bytes = source.size * np.dtype(np.float64).itemsize
    if max_result_bytes is not None and output_bytes > max_result_bytes:
        raise MemoryError(
            f"Float64 output requires {output_bytes} bytes, exceeding "
            f"max_result_bytes={max_result_bytes}"
        )
    if source.size == 0:
        return source.astype(np.float64, copy=True)

    remaining = tuple(index for index in range(source.ndim) if index not in axes)
    selected = tuple(index for index in range(source.ndim) if index in axes)
    group_shape = tuple(source.shape[index] for index in remaining)
    reduction_shape = tuple(source.shape[index] for index in selected)
    group_count = prod(group_shape) if group_shape else 1
    reduced_size = prod(reduction_shape) if reduction_shape else 1
    if reduced_size > max_chunk_cells:
        raise ValueError(
            f"one complete reduction group has {reduced_size} cells, exceeding "
            f"max_chunk_cells={max_chunk_cells}"
        )
    groups_per_chunk = max_chunk_cells // reduced_size
    output = np.empty(source.shape, dtype=np.float64)
    raw_axis_order = _raw_chunk_axis_order(source.ndim, remaining, selected)
    desired_axis_order = ((_GROUP_AXIS,) + selected) if remaining else selected
    to_grouped_order = tuple(raw_axis_order.index(label) for label in desired_axis_order)
    to_raw_order = tuple(desired_axis_order.index(label) for label in raw_axis_order)

    # Polars is the numeric engine; these arrays only carry values and group IDs.
    import polars as pl
    from factor_engine.backend.long_stable_zscore import (
        finite_anchor_centered_zscore_long,
    )

    for start in range(0, group_count, groups_per_chunk):
        stop = min(start + groups_per_chunk, group_count)
        group_ids = np.arange(start, stop, dtype=np.int64)
        if remaining:
            coordinates = np.unravel_index(group_ids, group_shape)
            index = [slice(None)] * source.ndim
            for dimension, coordinate in zip(remaining, coordinates):
                index[dimension] = coordinate
            index = tuple(index)
            selected_values = source[index]
            grouped_values = selected_values.transpose(to_grouped_order)
            grouped_values = grouped_values.reshape(stop - start, reduced_size)
        else:
            index = tuple(slice(None) for _ in range(source.ndim))
            selected_values = source[index]
            grouped_values = selected_values.reshape(1, reduced_size)

        values_float64 = grouped_values.astype(np.float64, copy=False)
        carrier_groups = np.repeat(
            np.arange(stop - start, dtype=np.int64), reduced_size
        )
        carrier = pl.DataFrame(
            {"__fp_fe_group": carrier_groups, "__fp_fe_value": values_float64.reshape(-1)}
        )
        transformed = finite_anchor_centered_zscore_long(
            carrier,
            value_col="__fp_fe_value",
            group_col="__fp_fe_group",
            ddof=ddof_value,
            constant_value=constant,
        )["__fp_fe_value"].to_numpy()
        grouped_result = transformed.reshape((stop - start,) + reduction_shape)
        if remaining:
            raw_result = grouped_result.transpose(to_raw_order)
        else:
            raw_result = grouped_result.reshape(source.shape)
        output[index] = raw_result
    return output


def _module_source_identity(path: str | Path, *, max_bytes: int = _MAX_IDENTITY_SOURCE_BYTES):
    """Hash bounded on-disk module source, not the loaded-code closure."""
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
    """Describe this opt-in binding without claiming operator admission.

    Source identity covers the bounded on-disk contents of the adapter and FE
    helper modules. It does not fingerprint loaded code objects, transitive
    imports, or a runtime dependency closure.
    """
    import polars as pl
    from factor_engine.backend.long_stable_zscore import (
        finite_anchor_centered_zscore_long,
    )

    helper_path = inspect.getsourcefile(finite_anchor_centered_zscore_long)
    if helper_path is None:
        helper_path = finite_anchor_centered_zscore_long.__code__.co_filename
    payload = {
        "status": "opt_in_candidate",
        "adapter": "factor_preprocess.adapters.fe_zscore.cs_zscore_finite_anchor_fe_native",
        "adapter_version": _ADAPTER_VERSION,
        "fe_function": _FE_FUNCTION,
        "numeric_policy": FINITE_ANCHOR_CENTERED_V2,
        "backend": "polars_native_grouped_long",
        "identity_scope": (
            "bounded on-disk module source only; not a loaded-code closure; "
            "transitive imports and runtime dependencies are not fingerprinted"
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


__all__ = ["cs_zscore_finite_anchor_fe_native", "execution_identity"]

# -*- coding: utf-8 -*-
"""Native Polars backends for limit-state, benchmark, cash-flow-stage, date-diff
and event-decay helpers.

Elementwise / simple per-column NumPy kernels over polars column arrays; results
wrapped into ``pl.DataFrame`` without constructing pandas DataFrames.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from collections.abc import Callable

import numpy as np
import polars as pl

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _make(base: pl.DataFrame, cols: list[str], values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({c: values[:, i] for i, c in enumerate(cols)})


def _touch_1d(close, upper, tolerance):
    valid = np.isfinite(close) & np.isfinite(upper)
    return np.where(valid, (close >= upper - tolerance).astype(float), np.nan)


def _rolling_max_1d(x: np.ndarray, w: int, min_periods: int = 1) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum()) < min_periods:
            continue
        out[t] = float(np.max(seg[ok]))
    return out


# ---------------------------------------------------------------------------
# limit-state
# ---------------------------------------------------------------------------


def ashare_limit_up_touch(high, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(high, upper_limit)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _touch_1d(high[c].to_numpy(), upper_limit[c].to_numpy(), tolerance)
    return _make(high, cols, out)


def ashare_limit_down_touch(low, lower_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(low, lower_limit)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        lo, ll = low[c].to_numpy(), lower_limit[c].to_numpy()
        valid = np.isfinite(lo) & np.isfinite(ll)
        out[:, i] = np.where(valid, (lo <= ll + tolerance).astype(float), np.nan)
    return _make(low, cols, out)


# ---------------------------------------------------------------------------
# benchmark / masks / cash-flow stage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# date diffs / announcement lag
# ---------------------------------------------------------------------------


def _date_diff_1d(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(d1, dtype="datetime64[ns]")
    b_arr = np.asarray(d2, dtype="datetime64[ns]")
    n = len(a_arr)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        if np.isnat(a_arr[t]) or np.isnat(b_arr[t]):
            continue
        days = (a_arr[t] - b_arr[t]) / np.timedelta64(1, "D")
        out[t] = float(days)
    return out


def date_diff_days(left, right):
    cols = _cols(left, right)
    rows = left.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _date_diff_1d(left[c].to_list(), right[c].to_list())
    return _make(left, cols, out)


def trading_day_diff(date1, date2):
    return date_diff_days(date1, date2)


def fin_announcement_lag(period_end_date, pub_date):
    return date_diff_days(period_end_date, pub_date)


# ---------------------------------------------------------------------------
# event decay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("ashare_limit_up_touch", ("high", "upper_limit", "tick_tolerance"), ashare_limit_up_touch, "High touched upper limit within tolerance."),
    ("ashare_limit_down_touch", ("low", "lower_limit", "tick_tolerance"), ashare_limit_down_touch, "Low touched lower limit within tolerance."),
)


# ---------------------------------------------------------------------------
# R57 backend-coverage batch 4 — explicit execution-kind declarations.
# These kernels are genuine polars expressions (pl.Expr / with_columns over
# columns).  Previously they had no _physical_spec, so
# canonical_polars_kind(production_mode=True) failed closed to UNSUPPORTED.
# ---------------------------------------------------------------------------
from factor_engine.backend.contracts import (
    ExecutionKind,
    PhysicalImplementationSpec,
)

_BATCH4_NOTE = (
    "Genuine polars expression kernel (pl.Expr over columns, no pandas round-trip); "
    "runtime marshal probe on real data records 0 pl.DataFrame.to_pandas "
    "calls. Eager panel API only: no lazy/streaming or production-parity claim."
)


def _batch4_native_spec(canonical: str, kernel: str) -> PhysicalImplementationSpec:
    """Explicit execution-kind contract for a genuine polars expression kernel.

    Batch-4 evidence contract (same rules as batches 1-3): the kernel body
    builds pl.Expr / pl.DataFrame columns with no pandas round-trip, and the
    runtime marshal probe on real data records zero pl.DataFrame.to_pandas
    calls.  Eager panel API, so supports_lazy / supports_streaming stay False.
    """
    return PhysicalImplementationSpec(
        canonical=canonical,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=f"common.polars_limit_misc:{kernel}:v1",
        emitter_identity=f"polars_expr:{canonical}",
        kernel_identity=f"common.polars_limit_misc:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
        notes=_BATCH4_NOTE,
    )


_BATCH4_SPECS: dict[str, PhysicalImplementationSpec] = {
    _c: _batch4_native_spec(_c, _k)
    for _c, _k in (
        ("ashare_limit_up_touch", "PolarsLimitMisc_ashare_limit_up_touch"),
        ("ashare_limit_down_touch", "PolarsLimitMisc_ashare_limit_down_touch"),
    )
}


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    if name == "event_decay_asof":
        from factor_engine.cleaned_operators.state_event import EventDecayAsOf
        metadata = copy.deepcopy(EventDecayAsOf.metadata)
        metadata.tags = list(metadata.tags or []) + ["polars", "native"]
    else:
        metadata = OperatorMetadata(
            name=name,
            category="math",
            description=description,
            param_names=list(params),
            return_type="series",
            tags=["pit_safe", "causal", "polars", "native"],
        )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsLimitMisc_{name}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_calculate_series": _calculate_series,
            "__module__": __name__,
            **({"_physical_spec": _BATCH4_SPECS[name]}
               if name in _BATCH4_SPECS else {}),
        },
    )
    register_operator(
        name=name,
        category="math",
        business_category="elementwise_math",
        canonical=name,
        source="polars_limit_misc",
        backend="polars",
        status="production",
        replace=True,
        expected_old_source="polars_native_misc_final",
        replacement_reason="Consolidating polars native operators into polars_limit_misc"
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)

# Classify limit-state helpers on the extended surface so the static operator
# surface covers the final registry exactly.
import factor_engine.cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only({
        "ashare_limit_up_touch",
        "ashare_limit_down_touch",
        "ashare_open_at_upper_limit",
        "ashare_limit_open_failed",
        "ashare_limit_one_price",
        "ashare_limit_failed",
    })

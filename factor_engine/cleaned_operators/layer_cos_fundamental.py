# -*- coding: utf-8 -*-
"""COS-aware PIT primitives and final semantic overrides."""
from __future__ import annotations

from typing import Any
import math
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base import ParamSpec, ParamRole
from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_bool, strict_enum

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.overhaul.base import (
    EPS, Spec, aligned_pd, frame_pd, pl, pl_base_with, pl_cols,
    positive_int, register_specs,
)
from factor_engine.cleaned_operators.fiscal_strict import (
    period_ordinal, pd_ttm_from_quarterly, pd_yoy_by_period,
)


def pd_true_range(high, low, close, **_):
    high, low, close = aligned_pd(high, low, close)
    h, l, c = (f.to_numpy(dtype=float) for f in (high, low, close))
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    out = h - l
    has_prev = np.isfinite(prev)
    out[has_prev] = np.maximum.reduce([out[has_prev], np.abs(h[has_prev] - prev[has_prev]), np.abs(l[has_prev] - prev[has_prev])])
    out[~(np.isfinite(h) & np.isfinite(l))] = np.nan
    return frame_pd(close, out)


def pl_true_range(high, low, close, **_):
    cols = [c for c in pl_cols(close) if c in high.columns and c in low.columns]
    return close.with_columns([
        # PARITY-B: pandas reference (pd_true_range) keeps the row finite whenever
        # high & low are finite, using the previous close ONLY where it is finite
        # (the max is over the terms that exist; the seed high-low stays).  A NaN
        # previous close therefore does NOT blank the row (the three-term max
        # skips it).  The old ``max_horizontal`` propagated the NaN previous close
        # and blanked rows after every gap — matching the oracle now.
        pl.when(
            high[c].cast(pl.Float64, strict=False).is_null()
            | ~high[c].cast(pl.Float64, strict=False).is_finite()
            | low[c].cast(pl.Float64, strict=False).is_null()
            | ~low[c].cast(pl.Float64, strict=False).is_finite()
        ).then(None)
        .otherwise(
            pl.max_horizontal(
                high[c].cast(pl.Float64, strict=False) - low[c].cast(pl.Float64, strict=False),
                pl.when(close[c].cast(pl.Float64, strict=False).shift(1).is_finite()).then((high[c].cast(pl.Float64, strict=False) - close[c].cast(pl.Float64, strict=False).shift(1)).abs()).otherwise(float("-inf")),
                pl.when(close[c].cast(pl.Float64, strict=False).shift(1).is_finite()).then((low[c].cast(pl.Float64, strict=False) - close[c].cast(pl.Float64, strict=False).shift(1)).abs()).otherwise(float("-inf")),
            )
        ).alias(c)
        for c in cols
    ])


def _datetime_days(values):
    parsed = pd.to_datetime(values.reshape(-1), errors="coerce", utc=True)
    raw = np.asarray(parsed.view("int64"), dtype=float).reshape(values.shape)
    raw[raw == float(np.iinfo(np.int64).min)] = np.nan
    return raw / 86_400_000_000_000.0


def pd_staleness(available_at, decision_time, **_):
    available_at, decision_time = aligned_pd(available_at, decision_time)
    out = _datetime_days(decision_time.to_numpy(dtype=object)) - _datetime_days(available_at.to_numpy(dtype=object))
    out[(~np.isfinite(out)) | (out < 0)] = np.nan
    return frame_pd(available_at, out)


def pl_staleness(available_at, decision_time, **_):
    cols = [c for c in pl_cols(available_at) if c in decision_time.columns]
    return available_at.with_columns([(((decision_time[c].cast(pl.Datetime, strict=False) - available_at[c].cast(pl.Datetime, strict=False)).dt.total_seconds() / 86400.0).cast(pl.Float64).alias(c)) for c in cols])


def _finite_float(value):
    return float(value) if np.isfinite(value) and abs(value) <= np.finfo(float).max else np.nan


def _strict_axes(*panels):
    if all(isinstance(v, pd.DataFrame) for v in panels):
        first = panels[0]
        if any(not v.index.is_unique or not v.columns.is_unique or
            not first.index.equals(v.index) or not first.columns.equals(v.columns) for v in panels):
            raise ValueError("Fiscal panels must have unique exactly aligned axes")
    else:
        from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
        verify_frames_share_identity("fiscal_period", *panels)


def _revision_array(values, periods, revisions, mode):
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        last: dict[int, tuple[Any, float]] = {}
        for row in range(values.shape[0]):
            ordinal = period_ordinal(periods[row, col])
            value, revision = values[row, col], revisions[row, col]
            if ordinal is None or not np.isfinite(value) or pd.isna(revision):
                continue
            previous = last.get(ordinal)
            if previous is not None and revision != previous[0]:
                if mode == "absolute":
                    change = np.longdouble(value) - np.longdouble(previous[1])
                    out[row, col] = float(change) if abs(change) <= np.finfo(float).max else np.nan
                elif previous[1] != 0:
                    change = np.longdouble(value) / np.longdouble(previous[1]) - 1
                    out[row, col] = float(change) if abs(change) <= np.finfo(float).max else np.nan
            last[ordinal] = (revision, float(value))
    return out


def pd_revision_delta(x, period_id, revision_id, mode="absolute", **_):
    _strict_axes(x, period_id, revision_id)
    x, period_id, revision_id = aligned_pd(x, period_id, revision_id)
    mode = strict_enum(mode, "mode", ("absolute", "ratio"))
    if mode not in {"absolute", "ratio"}:
        raise ValueError("mode must be absolute or ratio")
    return frame_pd(x, _revision_array(x.to_numpy(dtype=float), period_id.to_numpy(dtype=object), revision_id.to_numpy(dtype=object), mode))


def pl_revision_delta(x, period_id, revision_id, mode="absolute", **_):
    _strict_axes(x, period_id, revision_id)
    mode = strict_enum(mode, "mode", ("absolute", "ratio"))
    if mode not in {"absolute", "ratio"}:
        raise ValueError("mode must be absolute or ratio")
    cols = [c for c in pl_cols(x) if c in period_id.columns and c in revision_id.columns]
    values = np.column_stack([x[c].cast(pl.Float64, strict=False).to_numpy() for c in cols])
    periods = np.column_stack([np.asarray(period_id[c].to_list(), dtype=object) for c in cols])
    revisions = np.column_stack([np.asarray(revision_id[c].to_list(), dtype=object) for c in cols])
    out = _revision_array(values, periods, revisions, mode)
    return pl_base_with(x, {c: pl.Series(c, out[:, i]) for i, c in enumerate(cols)})


def _stability_column(values, periods, count, method, consecutive):
    out = np.full(values.shape, np.nan)
    known: dict[int, float] = {}
    for row, (value, raw_period) in enumerate(zip(values, periods)):
        ordinal = period_ordinal(raw_period)
        if ordinal is None:
            continue
        if np.isfinite(value):
            known[ordinal] = float(value)
        sample = [known[key] for key in [ordinal - lag for lag in range(count)] if key in known]
        if (consecutive and len(sample) != count) or len(sample) < 2:
            continue
        array = np.asarray(sample, dtype=np.longdouble)
        if method == "std":
            out[row] = _finite_float(np.std(array, ddof=1))
        elif method == "mad":
            out[row] = _finite_float(np.median(np.abs(array - np.median(array))))
        else:
            mean = np.mean(array)
            if mean != 0:
                out[row] = _finite_float(np.std(array, ddof=1) / abs(mean))
    return out


def pd_period_stability(x, period_id, periods=8, method="mad", require_consecutive=True, **_):
    _strict_axes(x, period_id)
    x, period_id = aligned_pd(x, period_id)
    count = strict_int(periods, "periods", minimum=2)
    method = strict_enum(method, "method", ("std", "mad", "cv"))
    require_consecutive = strict_bool(require_consecutive, "require_consecutive")
    if method not in {"std", "mad", "cv"}:
        raise ValueError("method must be std, mad, or cv")
    out = np.full(x.shape, np.nan)
    for col in range(x.shape[1]):
        out[:, col] = _stability_column(x.iloc[:, col].to_numpy(dtype=float), period_id.iloc[:, col].to_numpy(dtype=object), count, method, bool(require_consecutive))
    return frame_pd(x, out)


def pl_period_stability(x, period_id, periods=8, method="mad", require_consecutive=True, **_):
    _strict_axes(x, period_id)
    count = strict_int(periods, "periods", minimum=2)
    method = strict_enum(method, "method", ("std", "mad", "cv"))
    require_consecutive = strict_bool(require_consecutive, "require_consecutive")
    if method not in {"std", "mad", "cv"}:
        raise ValueError("method must be std, mad, or cv")
    cols = [c for c in pl_cols(x) if c in period_id.columns]
    return pl_base_with(x, {c: pl.Series(c, _stability_column(x[c].cast(pl.Float64, strict=False).to_numpy(), np.asarray(period_id[c].to_list(), dtype=object), count, method, bool(require_consecutive))) for c in cols})


def pd_safe_div(x, y, epsilon=EPS, **_):
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        x = pd.DataFrame(x, index=y.index, columns=y.columns, dtype=float)
    elif not isinstance(y, pd.DataFrame) and isinstance(x, pd.DataFrame):
        y = pd.DataFrame(y, index=x.index, columns=x.columns, dtype=float)
    elif not isinstance(x, pd.DataFrame) and not isinstance(y, pd.DataFrame):
        # 2026-08-29: scalar/scalar ``safe_div(0, 0)`` is the LQTP platform's
        # NaN literal branch (``where(cond, x, safe_div(0,0))`` ≡ NaN when
        # cond false).  Return the platform NaN scalar; the surrounding
        # ``where`` scalar-broadcast turns it into a constant NaN panel.
        try:
            if epsilon > 0 and float(y) != 0 and abs(float(y)) > epsilon:
                return float(x) / float(y)
        except (TypeError, ValueError):
            pass
        return float("nan")
    x, y = aligned_pd(x, y)
    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv) & (np.abs(yv) > epsilon)
    out = np.full(x.shape, np.nan)
    out[valid] = xv[valid] / yv[valid]
    return frame_pd(x, out)


def pl_safe_div(x, y, epsilon=EPS, **_):
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    x_panel, y_panel = isinstance(x, pl.DataFrame), isinstance(y, pl.DataFrame)
    if not x_panel and not y_panel:
        try:
            return float(x) / float(y) if np.isfinite(float(x)) and np.isfinite(float(y)) and abs(float(y)) > epsilon else float("nan")
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return float("nan")
    template = x if x_panel else y
    if x_panel and y_panel:
        _strict_axes(x, y)
    cols = [c for c in pl_cols(template) if (not x_panel or c in x.columns) and (not y_panel or c in y.columns)]
    expressions = []
    for c in cols:
        left = x[c].cast(pl.Float64, strict=False) if x_panel else pl.lit(float(x))
        right = y[c].cast(pl.Float64, strict=False) if y_panel else pl.lit(float(y))
        expressions.append(pl.when(left.is_finite() & right.is_finite() & (right.abs() > epsilon)).then(left / right).otherwise(None).alias(c))
    return template.with_columns(expressions)


def _register_period_contracts():
    from factor_engine.cleaned_operators.overhaul.base import PandasFunctionOperator, PolarsFunctionOperator
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    from factor_engine.runtime.execution_contract import declare_stateful
    entries = (
        ("revision_delta", ["x", "period_id", "revision_id", "mode"],
            ("x", "period_id", "revision_id"), pd_revision_delta, pl_revision_delta,
            {"mode": ParamSpec(dtype=str, choices=("absolute", "ratio"), default="absolute", param_role=ParamRole.POLICY)}),
        ("period_stability", ["x", "period_id", "periods", "method", "require_consecutive"],
            ("x", "period_id"), pd_period_stability, pl_period_stability,
            {"periods": ParamSpec(dtype=int, min=2, default=8, param_role=ParamRole.HORIZON),
             "method": ParamSpec(dtype=str, choices=("mad", "std", "cv"), default="mad", param_role=ParamRole.POLICY),
             "require_consecutive": ParamSpec(dtype=bool, default=True, searchable=False, param_role=ParamRole.SUPPORT_POLICY)}),
    )
    for name, params, panels, pd_fn, pl_fn, specs in entries:
        for backend, cls, fn in (("pandas_numpy", PandasFunctionOperator, pd_fn),
                                ("polars", PolarsFunctionOperator, pl_fn)):
            if backend == "polars" and pl is None:
                continue
            op = cls(name, "fundamental_period", params, fn.__name__, fn,
                param_specs=specs, panel_params=panels,
                scalar_params=tuple(k for k in params if k not in panels))
            op.metadata.window_semantics = "full_history"
            op.metadata.checkpointable = False
            op.metadata.available_at = "report_date"
            op.metadata.same_session_usable = False
            if backend == "polars":
                from factor_engine.cleaned_operators import fiscal_strict
                digest = hashlib.sha256(Path(__file__).read_bytes() +
                    Path(fiscal_strict.__file__).read_bytes()).hexdigest()
                op._physical_spec = PhysicalImplementationSpec(canonical=name, backend="polars",
                    execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
                    materializes_full_panel=True, supports_nan=True, supports_nulls=True,
                    supports_inf=True, implementation_source_hash=digest,
                    kernel_identity="layer_cos_fundamental." + name,
                    notes="CPU per-period state over NumPy arrays; scalar fiscal parsing; full replay, no GPU.")
            prior = (OperatorRegistry._catalog.get(name, {}).get("backend_meta") or {}).get(backend, {})
            OperatorRegistry.register(op, canonical=name, backend=backend,
                source="fiscal_period_exact", status="implemented", backend_explicit=True,
                replace=bool(prior), expected_old_source=prior.get("source") if prior else None,
                replacement_reason="Exact fiscal state and shared contracts")
        declare_stateful(name, state_model="recursive", chunking="required_full_history")


def register() -> None:
    register_specs({
        "true_range": Spec("price_volume", ["high", "low", "close"], "true range with explicit first-row high-low fallback", pd_true_range, pl_true_range),
        "fundamental_staleness": Spec("fundamental_period", ["available_at", "decision_time"], "calendar-day age of visible fundamental data", pd_staleness, pl_staleness),
        "safe_div_null": Spec(
            "elementwise", ["x", "y", "epsilon"],
            "finite safe division returning null for near-zero denominator",
            pd_safe_div, pl_safe_div,
            param_specs={"epsilon": ParamSpec(dtype=float, min=0.0, default=EPS, searchable=False, param_role=ParamRole.NUMERICAL)},
            scalar_params=("epsilon",), mixed_params=("x", "y"),
        ),
        "ttm_from_quarterly": Spec("fundamental_period", ["x", "period_id", "periods", "require_consecutive", "revision_policy"], "strict consecutive-period TTM", pd_ttm_from_quarterly),
        "yoy_by_period": Spec("fundamental_period", ["x", "period_id", "periods", "denominator", "require_consecutive", "revision_policy"], "strict fiscal-period growth", pd_yoy_by_period),
    })

    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    safe_div_polars = OperatorRegistry.get("safe_div_null", "polars")
    if safe_div_polars is not None:
        safe_div_polars._physical_spec = PhysicalImplementationSpec(
            canonical="safe_div_null", backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            supports_lazy=False, supports_streaming=False,
            materializes_full_panel=False, supports_nulls=True,
            supports_nan=True, supports_inf=True,
            notes="Direct eager Polars expressions with scalar/panel broadcasting.",
        )

    _register_period_contracts()


register()

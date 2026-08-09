# -*- coding: utf-8 -*-
"""Source-safe building blocks (audit section 19).

These operators do not add alpha surface on their own; they exist so other
factors can be constructed safely: validity/coverage/staleness gates, strict
bounded imputation, gap-limited forward fill, explicit log domains, and
unambiguous extreme-position semantics.

Every operator is causal (only data ``<= t`` is consumed) and preserves the
``timestamp x instrument`` panel shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamRole, ParamSpec
from cleaned_operators.overhaul.base import (
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry

EPS = 1e-12
SOURCE = "safe_ops"

# P0-58 ImputationPolicy: gap forward-fill is only a legitimate imputation for
# price/level lineages.  financial / return / event / revision values must never
# be forward-filled — a missing fundamental disclosure or a missing return is
# not "the same as yesterday's number".
FFILL_LINEAGE_POLICY: dict[str, bool] = {
    "price": True,
    "financial": False,
    "return": False,
    "event": False,
    "revision": False,
}


def _check_ffill_lineage(lineage: str | None) -> None:
    """P0-58 ImputationPolicy gate for ``ts_ffill_limited``.

    ``lineage`` names the value family of ``x``.  The policy forbids gap
    forward-fill for financial/return/event/revision lineages (a missing
    observation there must stay missing).  ``None`` keeps the legacy behaviour
    (forward-fill allowed) and is the caller's assertion that ``x`` is a
    price/level series.
    """
    if lineage is None:
        return
    if not isinstance(lineage, str):
        raise ValueError(
            f"ts_ffill_limited: lineage must be a string or None, got {lineage!r}"
        )
    allowed = FFILL_LINEAGE_POLICY.get(lineage)
    if allowed is False:
        raise ValueError(
            f"ts_ffill_limited: forward-fill is not permitted for value lineage "
            f"{lineage!r} (limited_ffill=False); only price/level series may be "
            "gap-forward-filled"
        )


def _register(
    name: str,
    category: str,
    params: list[str],
    description: str,
    pandas_fn,
    polars_fn=None,
    param_specs: dict | None = None,
) -> None:
    pandas_op = PandasFunctionOperator(name, category, params, description, pandas_fn)
    if param_specs:
        pandas_op.metadata.param_specs = dict(param_specs)
    OperatorRegistry.register(
        pandas_op,
        canonical=name,
        backend="pandas_numpy",
        source=SOURCE,
        status="implemented",
        backend_explicit=True,
    )
    if polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(name, category, params, description, polars_fn),
            canonical=name,
            backend="polars",
            source=SOURCE,
            status="implemented",
            backend_explicit=True,
        )
    # Classify as an extended production candidate so the static-surface gate
    # (layer_governance) covers it.  These operators are source-safe building
    # blocks by construction; they are NOT fail-closed by the S2 gate.
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})


# --------------------------------------------------------------------------
# Validity / coverage / staleness gates
# --------------------------------------------------------------------------

def _finite_mask(x: pd.DataFrame) -> pd.DataFrame:
    """``1.0`` for finite (np.isfinite), ``NaN`` for NaN / +Inf / -Inf.

    Master Spec Part C-14: "valid" is ``np.isfinite`` — a ``+Inf`` / ``-Inf``
    observation is as unusable as ``NaN`` and must never count toward a
    validity / coverage statistic.  ``rolling().count()`` counts non-NaN, so a
    raw mask would still count Inf; mapping non-finite to NaN makes the count
    agree with ``np.isfinite``.
    """
    arr = x.to_numpy(dtype=float)
    finite = np.isfinite(arr).astype(float)
    finite[~np.isfinite(arr)] = np.nan
    return pd.DataFrame(finite, index=x.index, columns=x.columns)


def pd_ts_valid_count(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    mp = int(min_periods)
    if mp < 1:
        raise ValueError("ts_valid_count: min_periods must be >= 1")
    return _finite_mask(x).rolling(w, min_periods=mp).count()


def pd_ts_coverage_ratio(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    mp = int(min_periods)
    if mp < 1:
        raise ValueError("ts_coverage_ratio: min_periods must be >= 1")
    count = _finite_mask(x).rolling(w, min_periods=mp).count()
    return count / float(w)


def pd_ts_staleness(x, window, **_):
    """Bars since the last finite value within the causal window.

    A value at bar ``t`` carrying the last finite observation from ``t-k``
    reports ``k``; a window with no finite value reports NaN.  This is the
    "how old is the data I am actually using" gate.
    """
    w = positive_int(window, "window")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        col_arr = arr[:, col]
        last_seen = np.full(w + 1, np.nan, dtype=float)  # ring of recent positions
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = col_arr[start : row + 1]
            finite_idx = np.flatnonzero(np.isfinite(seg))
            if finite_idx.size == 0:
                continue
            out[row, col] = float(row - (start + finite_idx[-1]))
    return frame_pd(x, out)


def _pl_finite_count_expr(col, w: int, mp: int):
    """Polars twin of :func:`_finite_mask` + pandas ``rolling().count()``.

    ``is_finite()`` → True for finite, null for NaN, False for ±Inf (so Inf is
    never counted); ``when/then/otherwise`` maps the finite cells to ``1`` and
    everything else (incl. null) to ``null``.

    Pandas ``rolling(w, min_periods=mp).count()`` gates on WINDOW SLOTS — the
    number of rows in the window (``min(r+1, w)``) — not on the number of valid
    values.  With ``w >= mp`` (enforced by the central validator) that gate
    reduces to ``row_index + 1 >= mp``, expressible directly with
    ``pl.int_range`` (a literal 1 series does NOT roll in Polars, so no slot
    column is used).  An all-null window (every row non-finite) must yield 0 —
    not null — exactly like ``pandas .count()``, so ``fill_null(0)`` precedes
    the slot gate.
    """
    mask = pl.when(pl.col(col).is_finite()).then(pl.lit(1)).otherwise(None).cast(pl.Int32)
    cnt = mask.rolling_sum(window_size=w, min_samples=1).fill_null(0)
    slot_gate = (pl.int_range(0, pl.len()) + 1) >= mp
    return pl.when(slot_gate).then(cnt).otherwise(None)


def _pl_ts_valid_count(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    mp = int(min_periods)
    if mp < 1:
        raise ValueError("ts_valid_count: min_periods must be >= 1")
    cols = pl_cols(x)
    return x.with_columns([
        _pl_finite_count_expr(c, w, mp).alias(c) for c in cols
    ])


def _pl_ts_coverage_ratio(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    mp = int(min_periods)
    if mp < 1:
        raise ValueError("ts_coverage_ratio: min_periods must be >= 1")
    cols = pl_cols(x)
    return x.with_columns([
        (_pl_finite_count_expr(c, w, mp) / w).alias(c) for c in cols
    ])


_VALIDITY_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=1, history_semantics="max_rows"),
    # min_periods <= window is enforced centrally by _validate_common_integer_relations
    "min_periods": ParamSpec(
        dtype=int, min=1,
        param_role=ParamRole.POLICY,  # support knob: never a search dimension
    ),
}

_register(
    "ts_valid_count",
    "time_series",
    ["x", "window", "min_periods"],
    "窗口内有限值个数（valid = np.isfinite，NaN 与 ±Inf 均不计）。",
    pd_ts_valid_count,
    _pl_ts_valid_count,
    param_specs=_VALIDITY_PARAM_SPECS,
)
_register(
    "ts_coverage_ratio",
    "time_series",
    ["x", "window", "min_periods"],
    "窗口内有限值占比（有限值个数 / window；valid = np.isfinite）。",
    pd_ts_coverage_ratio,
    _pl_ts_coverage_ratio,
    param_specs=_VALIDITY_PARAM_SPECS,
)
_register(
    "ts_staleness",
    "time_series",
    ["x", "window"],
    "距窗口内最近一个有限值的 bar 数。",
    pd_ts_staleness,
)


def pd_cs_valid_count(x, **_):
    arr = x.to_numpy(dtype=float)
    counts = np.sum(np.isfinite(arr), axis=1, keepdims=True)
    out = np.broadcast_to(counts, arr.shape).astype(float)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_cs_coverage_ratio(x, **_):
    arr = x.to_numpy(dtype=float)
    count = np.sum(np.isfinite(arr), axis=1, keepdims=True)
    total = x.shape[1]
    out = np.broadcast_to(count / float(total), arr.shape).astype(float)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


_register(
    "cs_valid_count",
    "cross_sectional",
    ["x"],
    "横截面每行有限值个数（广播到每列）。",
    pd_cs_valid_count,
)
_register(
    "cs_coverage_ratio",
    "cross_sectional",
    ["x"],
    "横截面每行有限值占比（广播到每列）。",
    pd_cs_coverage_ratio,
)


def pd_group_valid_count(x, group, **_):
    x, group = aligned_pd(x, group)
    arr = x.to_numpy(dtype=float)
    garr = group.to_numpy(dtype=object)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            g = garr[row, col]
            if g is None or pd.isna(g):
                continue
            mask = garr[row] == g
            out[row, col] = float(np.sum(np.isfinite(arr[row][mask])))
    return frame_pd(x, out)


_register(
    "group_valid_count",
    "group",
    ["x", "group"],
    "组内有限值个数。",
    pd_group_valid_count,
)


def pd_group_impute_median(x, group, min_group_size=3, **_):
    x, group = aligned_pd(x, group)
    arr = x.to_numpy(dtype=float)
    garr = group.to_numpy(dtype=object)
    out = arr.copy()
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            g = garr[row, col]
            if g is None or pd.isna(g):
                continue
            mask = garr[row] == g
            group_vals = arr[row][mask]
            finite = group_vals[np.isfinite(group_vals)]
            if finite.size < int(min_group_size):
                continue
            median = float(np.median(finite))
            out[row] = np.where(
                np.isfinite(out[row]),
                out[row],
                np.where(garr[row] == g, median, np.nan),
            )
    return frame_pd(x, out)


_register(
    "group_impute_median",
    "group",
    ["x", "group", "min_group_size"],
    "组内中位数填补缺失值（组内有限样本不足 min_group_size 时不填）。",
    pd_group_impute_median,
)


# --------------------------------------------------------------------------
# Strict missing-value handling
# --------------------------------------------------------------------------

def pd_ts_ffill_limited(x, max_gap, lineage: str | None = None, **_):
    limit = positive_int(max_gap, "max_gap")
    _check_ffill_lineage(lineage)
    return x.ffill(limit=limit)


def _pl_ts_ffill_limited(x, max_gap, lineage: str | None = None, **_):
    limit = positive_int(max_gap, "max_gap")
    _check_ffill_lineage(lineage)
    return x.with_columns([pl.col(c).forward_fill(limit=limit).alias(c) for c in pl_cols(x)])


def pd_log_positive_or_nan(x, **_):
    arr = x.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(arr > 0, np.log(arr), np.nan)
    return frame_pd(x, out)


def _pl_log_positive_or_nan(x, **_):
    cols = pl_cols(x)
    return x.with_columns([
        pl.when(pl.col(c) > 0).then(pl.col(c).log()).otherwise(None).alias(c)
        for c in cols
    ])


_register(
    "ts_ffill_limited",
    "data_cleaning",
    ["x", "max_gap", "lineage"],
    "前向填充，最多跨过 max_gap 根 bar（防长期停牌/断档污染）；lineage 策略禁止对 financial/return/event/revision 值前向填充。",
    pd_ts_ffill_limited,
    _pl_ts_ffill_limited,
)
_register(
    "log_positive_or_nan",
    "elementwise",
    ["x"],
    "对数正域映射：x>0 取 log(x)，否则 NaN（不把非法域夹到 epsilon）。",
    pd_log_positive_or_nan,
    _pl_log_positive_or_nan,
)


# --------------------------------------------------------------------------
# Unambiguous extreme position semantics
# --------------------------------------------------------------------------

def _pd_argext(x, window, pick, min_periods):
    w = positive_int(window, "window")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = arr[start : row + 1, col]
            valid = np.isfinite(seg)
            if valid.sum() < int(min_periods):
                continue
            target = np.nanmax(seg) if pick == "max" else np.nanmin(seg)
            hits = np.flatnonzero(valid & (seg == target))
            age = (seg.size - 1) - hits[-1]          # ties -> most recent
            out[row, col] = float(age)
    return frame_pd(x, out)


def pd_ts_argmax_age(x, window, min_periods=1, **_):
    return _pd_argext(x, window, "max", min_periods)


def pd_ts_argmin_age(x, window, min_periods=1, **_):
    return _pd_argext(x, window, "min", min_periods)


def _pd_argidx(x, window, pick, min_periods):
    """Position of the extreme measured from the window's OLDEST bar."""
    w = positive_int(window, "window")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = arr[start : row + 1, col]
            valid = np.isfinite(seg)
            if valid.sum() < int(min_periods):
                continue
            target = np.nanmax(seg) if pick == "max" else np.nanmin(seg)
            hits = np.flatnonzero(valid & (seg == target))
            out[row, col] = float(hits[-1])  # 0 = window oldest
    return frame_pd(x, out)


def pd_ts_argmax_index_from_oldest(x, window, min_periods=1, **_):
    return _pd_argidx(x, window, "max", min_periods)


def pd_ts_argmin_index_from_oldest(x, window, min_periods=1, **_):
    return _pd_argidx(x, window, "min", min_periods)


_register(
    "ts_argmax_age",
    "time_series_order",
    ["x", "window", "min_periods"],
    "距窗口内最近一次最大值的 bar 数（0=当前行即极值）。",
    pd_ts_argmax_age,
)
_register(
    "ts_argmin_age",
    "time_series_order",
    ["x", "window", "min_periods"],
    "距窗口内最近一次最小值的 bar 数（0=当前行即极值）。",
    pd_ts_argmin_age,
)
_register(
    "ts_argmax_index_from_oldest",
    "time_series_order",
    ["x", "window", "min_periods"],
    "窗口内最大值位置，0=窗口最旧 bar。",
    pd_ts_argmax_index_from_oldest,
)
_register(
    "ts_argmin_index_from_oldest",
    "time_series_order",
    ["x", "window", "min_periods"],
    "窗口内最小值位置，0=窗口最旧 bar。",
    pd_ts_argmin_index_from_oldest,
)


# --------------------------------------------------------------------------
# Cross-section imputation (explicit broadcasting)
# --------------------------------------------------------------------------

def _cs_impute_bounded(x, stat_fn, min_finite, label):
    """Row-impute missing cells with a cross-sectional statistic, gated by the
    number of finite values in the row (R11 #175-177).

    ``min_finite`` is the declared minimum finite observations a row must have
    BEFORE imputation is allowed: a row with fewer than ``min_finite`` finite
    values is left fully NaN (fail closed) instead of being imputed from a
    degenerate / empty cross-section.
    """
    mf = int(min_finite)
    if mf < 1:
        raise ValueError(f"{label}: min_finite must be >= 1")
    arr = x.to_numpy(dtype=float)
    finite_count = np.isfinite(arr).sum(axis=1)
    impute_rows = finite_count >= mf
    # Mean/median over the finite cells only; a row with 0 finite values has no
    # statistic at all and its NaN cells stay NaN (no-op) even when mf == 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.full(arr.shape[0], np.nan, dtype=float)
        for r in range(arr.shape[0]):
            vals = arr[r][np.isfinite(arr[r])]
            if vals.size:
                stat[r] = stat_fn(vals)
    out = arr.copy()
    mask = np.isnan(out) & impute_rows[:, None]
    out[mask] = np.broadcast_to(stat[:, None], arr.shape)[mask]
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_cs_impute_mean(x, min_finite=1, **_):
    return _cs_impute_bounded(x, lambda v: float(np.mean(v)), min_finite, "cs_impute_mean")


def pd_cs_impute_median(x, min_finite=1, **_):
    return _cs_impute_bounded(x, lambda v: float(np.median(v)), min_finite, "cs_impute_median")


_register(
    "cs_impute_mean",
    "cross_sectional",
    ["x", "min_finite"],
    "横截面均值填补缺失（逐行广播；行内有限样本不足 min_finite 时不填）。",
    pd_cs_impute_mean,
    param_specs={"min_finite": ParamSpec(dtype=int, min=1)},
)
_register(
    "cs_impute_median",
    "cross_sectional",
    ["x", "min_finite"],
    "横截面中位数填补缺失（逐行广播；行内有限样本不足 min_finite 时不填）。",
    pd_cs_impute_median,
    param_specs={"min_finite": ParamSpec(dtype=int, min=1)},
)


# --------------------------------------------------------------------------
# Semantic Closure declarations (Master Spec P0)
# --------------------------------------------------------------------------

def _declare_closure_contracts() -> None:
    """Register the MissingPolicy / WindowSemantics / axis-contract side-
    registry entries for the source-safe building blocks (Parts C-10 / D-16 /
    BM-259).  A validity/count operator counts only finite values and never
    connects across a gap, so it is WINDOW_VALID / MIN_SUPPORT_WINDOW."""
    from cleaned_operators.closure import (
        MissingPolicy,
        WindowSemantics,
        declare_missing_policy,
        declare_window_semantics,
    )

    declare_missing_policy("ts_valid_count", MissingPolicy.WINDOW_VALID)
    declare_missing_policy("ts_coverage_ratio", MissingPolicy.WINDOW_VALID)
    declare_missing_policy("ts_staleness", MissingPolicy.WINDOW_VALID)
    declare_missing_policy("cs_valid_count", MissingPolicy.PAIRWISE_VALID)
    declare_missing_policy("cs_coverage_ratio", MissingPolicy.PAIRWISE_VALID)
    declare_missing_policy("group_valid_count", MissingPolicy.WINDOW_VALID)
    declare_window_semantics("ts_valid_count", WindowSemantics.MIN_SUPPORT_WINDOW)
    declare_window_semantics("ts_coverage_ratio", WindowSemantics.MIN_SUPPORT_WINDOW)


_declare_closure_contracts()

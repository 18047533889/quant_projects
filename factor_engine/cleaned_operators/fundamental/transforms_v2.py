# -*- coding: utf-8 -*-
"""Generic PIT-safe fundamental transformation operators.

These operators are intentionally field-agnostic.  They operate on a daily
as-of panel ``x`` together with a daily-aligned ``period_id`` panel identifying
the report period currently visible at each decision timestamp.  Calculations
advance only when a new report period becomes visible; a later revision of an
already-visible period updates results only from the revision timestamp onward.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator

# Model-audit Phase 4 (search-space hygiene): explicit ParamSpec for the
# out-of-history fundamental scoring canonicals.  ``periods`` (how many prior
# visible reports form the reference sample) is the alpha horizon — HORIZON,
# searched.  ``period_id`` is the reporting-period identifier panel, not a
# search scalar (M-115/M-162/M-170).
_VS_PRIOR_HISTORY_PARAM_SPECS: dict[str, ParamSpec] = {
    "periods": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON, searchable=True),
}
from cleaned_operators.fiscal_strict import (
    period_ordinal,
    reject_ytd_growth,
    require_same_flow_grain,
)

_EPS = 1e-12

# Round-11 #51: expected reporting-flow grain per operator, surfaced as a
# ``flow_type:*`` metadata tag on the contract.  The runtime ``flow_type``
# parameter additionally fails closed on a mismatched declaration.
# Round-3 item 30: input-semantic declaration for the fundamental period-change
# family.  These operators measure period-over-period change / growth / spread —
# a value that is a Return/Rate (ROE, ROA, margin, revenue growth), NOT a raw
# Price/Volume.  ``> 0`` on a rate is a meaningful signal; ``> 0`` on a raw
# price/volume is almost always True.  Exposed as ``input_units`` metadata so a
# semantic-input-type audit can reject raw price/volume misuse at the binding
# layer, and as ``input_semantics:*`` tags for catalog readers.
_INPUT_UNITS: dict[str, dict[str, str]] = {
    "fin_lag": {"x": "same_as:output", "period_id": "fiscal_period"},
    "fin_diff": {"x": "same_as:output", "period_id": "fiscal_period"},
    "fin_pct_change": {"x": "rate", "period_id": "fiscal_period"},
    "fin_log_change": {"x": "rate", "period_id": "fiscal_period"},
    "fin_qoq": {"x": "rate", "period_id": "fiscal_period"},
    "fin_yoy": {"x": "rate", "period_id": "fiscal_period"},
    "fin_ttm": {"x": "flow", "period_id": "fiscal_period"},
    "fin_average_balance": {"x": "stock", "period_id": "fiscal_period"},
    "fin_growth": {"x": "rate", "period_id": "fiscal_period"},
    "fin_cagr": {"x": "rate", "period_id": "fiscal_period"},
    "fin_growth_acceleration": {"x": "rate", "period_id": "fiscal_period"},
    "fin_growth_change": {"x": "rate", "period_id": "fiscal_period"},
    "fin_growth_volatility": {"x": "rate", "period_id": "fiscal_period"},
    "fin_growth_stability": {"x": "rate", "period_id": "fiscal_period"},
    "fin_growth_persistence": {"x": "rate", "period_id": "fiscal_period"},
    "fin_positive_streak": {"x": "rate", "period_id": "fiscal_period"},
    "fin_negative_streak": {"x": "rate", "period_id": "fiscal_period"},
    "fin_sign_change_count": {"x": "rate", "period_id": "fiscal_period"},
}
_INPUT_SEMANTICS_RATE = frozenset({
    "fin_pct_change", "fin_log_change", "fin_qoq", "fin_yoy", "fin_growth",
    "fin_cagr", "fin_growth_acceleration", "fin_growth_change",
    "fin_growth_volatility", "fin_growth_stability", "fin_growth_persistence",
    "fin_positive_streak", "fin_negative_streak", "fin_sign_change_count",
})

_FLOW_TYPE_TAGS = {
    "fin_pct_change": "flow_type:SinglePeriodFlow",
    "fin_log_change": "flow_type:SinglePeriodFlow",
    "fin_qoq": "flow_type:SinglePeriodFlow",
    "fin_yoy": "flow_type:SinglePeriodFlow",
    "fin_ttm": "flow_type:SinglePeriodFlow",
    "fin_growth": "flow_type:SinglePeriodFlow",
    "fin_cagr": "flow_type:SinglePeriodFlow",
    "fin_growth_acceleration": "flow_type:SinglePeriodFlow",
    "fin_growth_change": "flow_type:SinglePeriodFlow",
    "fin_growth_volatility": "flow_type:SinglePeriodFlow",
    "fin_growth_stability": "flow_type:SinglePeriodFlow",
    "fin_growth_persistence": "flow_type:SinglePeriodFlow",
    "fin_accrual_ratio": "flow_type:SinglePeriodFlow",
    "fin_cash_earnings_gap": "flow_type:SinglePeriodFlow",
    "fin_cash_conversion": "flow_type:SinglePeriodFlow",
}


def _pos_int(value, name: str, minimum: int = 1) -> int:
    # NEW-005/006/088: THE single strict integer gate.  The old body was
    # ``int(value)`` — it silently truncated ``periods=3.7`` to ``3`` and
    # accepted ``True`` as ``1``.  It now delegates to
    # ``common.strict_params`` (which routes through ``base.strict_int_param``):
    # fractional / NaN / Inf / bool values are REJECTED, never coerced.  The
    # legacy ``parameter_contract_v2`` monkey-patch is removed; every module
    # (``expectation_v2`` / ``flow_semantics_v2`` / ``quality_v2``) imports this
    # same strict function object, so there is exactly one validator per process.
    from cleaned_operators.common.strict_params import strict_int

    return strict_int(value, name, minimum=minimum)


def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= _EPS:
        return np.nan
    return float(a / b)


def _period_key(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value) if isinstance(value, (str, np.datetime64)) else value
    except Exception:
        return value


def _default_require_parseable() -> bool:
    """Auto-detect the production/research split for unparseable period ids.

    Round-7 P1: production fundamental must fail closed — an unparseable report
    period has no fiscal-ordinal contract, so first-seen ordering (a research
    compatibility fallback) would silently corrupt lag/TTM/growth math.  When the
    production-policy module is unavailable we default to production (fail
    closed) so an unclassified run never silently falls back to first-seen order.
    """
    try:
        from runtime.production_policy import is_production_mode

        return bool(is_production_mode())
    except Exception:
        return True


def _period_insert(
    order: list[object], key: object, *, require_parseable: bool | None = None
) -> None:
    """Insert ``key`` into ``order`` keeping fiscal-ordinal sorted order.

    Report periods must advance by fiscal-quarter ordinal (``year*4 + quarter``),
    not by first-appearance order.  A late-disclosed revision or a back-filled
    older period (e.g. a restated 2025Q2 arriving after 2025Q3) must not reorder
    the sequence that lag / TTM / growth operators walk — that reordering corrupts
    every lag, trend and growth factor (audit §4.1).

    ``require_parseable`` (round-7 P1): when ``True`` an unparseable period id
    (``period_ordinal(key) is None``) is NOT placed — the row stays NaN (fail
    closed).  First-seen order is research-compat only; production passes
    ``require_parseable=True``.
    """
    if require_parseable is None:
        require_parseable = _default_require_parseable()
    target = period_ordinal(key)
    if target is None:
        if require_parseable:
            # Production fail-closed: refuse to place the period.  The row's
            # output remains NaN because every consumer resolves the current key
            # through ``order.index(current)`` / ordinal lookups, which cannot
            # find an unplaced key.
            return
        # Research compatibility: unparseable period ids keep first-seen order.
        order.append(key)
        return
    for position, existing in enumerate(order):
        existing_ord = period_ordinal(existing)
        if existing_ord is not None and existing_ord > target:
            order.insert(position, key)
            return
    order.append(key)


def _walk_periods(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    fn: Callable[[list[object], OrderedDict, object], float],
    *,
    require_parseable: bool | None = None,
) -> pd.DataFrame:
    # R23-292: strict panel alignment, never silent reindex.  A period-id panel
    # that is not on the exact same date/instrument grid as the value panel is a
    # caller bug — silently reindexing pairs values with the wrong fiscal period
    # (same day / permuted instrument order is the classic lookahead footgun).
    from cleaned_operators.alignment import align_panel_inputs

    x, period_id = align_panel_inputs(x, period_id, names=("x", "period_id"))
    if require_parseable is None:
        require_parseable = _default_require_parseable()
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        order: list[object] = []
        visible: OrderedDict[object, float] = OrderedDict()
        xv = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
        pv = period_id[col].to_numpy()
        arr = np.full(len(x), np.nan, dtype=float)
        for i, (value, raw_period) in enumerate(zip(xv, pv)):
            key = _period_key(raw_period)
            if key is not None and np.isfinite(value):
                if key not in visible:
                    _period_insert(order, key, require_parseable=require_parseable)
                visible[key] = float(value)
            if key is None or key not in visible:
                continue
            try:
                arr[i] = fn(order, visible, key)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                arr[i] = np.nan
        out[col] = arr
    return out


def _window_keys(
    order: list[object],
    visible: OrderedDict,
    current: object,
    count: int | None = None,
    require_consecutive: bool = False,
) -> list[object]:
    """Keys of the most recent ``count`` visible report periods ending at
    ``current`` (or the full visible history when ``count`` is None).

    ``require_consecutive=True`` fails closed on a skipped fiscal period: the
    selected window's ordinals must be contiguous (each adjacent pair differs by
    exactly one fiscal period) and ordinal-parseable.  Otherwise a missing
    report (e.g. a vendor gap at 2025Q2) would silently substitute a
    non-adjacent quarter — acceptable for "historical distribution" summaries
    but wrong for sums/averages whose math requires adjacent periods (review
    P0-03).  Operators that MUST treat fiscal time as real time (TTM, average
    balance, period trends, growth sequences, monotonicity, streaks, sign
    change counts) pass ``require_consecutive=True``; distribution / percentile
    / dispersion summaries intentionally use the last-N-*visible* window and
    document that missing reports are skipped.
    """
    try:
        pos = order.index(current)
    except ValueError:
        return []
    keys = order[: pos + 1]
    if count is not None:
        keys = keys[-int(count):]
    if require_consecutive and len(keys) > 1:
        ords = [period_ordinal(k) for k in keys]
        if any(o is None for o in ords):
            return []
        for prev_o, o in zip(ords, ords[1:]):
            if o != prev_o + 1:
                return []
    return list(keys)


def _values(
    order: list[object],
    visible: OrderedDict,
    current: object,
    count: int | None = None,
    require_consecutive: bool = False,
):
    """Values of the most recent ``count`` visible report periods ending at
    ``current`` (or the full visible history when ``count`` is None).

    ``require_consecutive=True`` fails closed on a skipped fiscal period (see
    ``_window_keys``).  Without it the window is the last-N-*visible* reports:
    missing report periods are skipped, which is valid for distribution /
    percentile / dispersion summaries but NOT for sums, averages, trends or
    other math that requires adjacent fiscal periods (review R4-23).
    """
    keys = _window_keys(order, visible, current, count, require_consecutive)
    vals = [float(visible[k]) for k in keys if k in visible and np.isfinite(visible[k])]
    return vals


def _streak_span(
    order: list[object],
    visible: OrderedDict,
    current: object,
    *,
    positive: bool,
    max_periods: int | None = None,
) -> float:
    """Count consecutive report-to-report value *changes* with the given sign.

    Each counted step must be between *adjacent* fiscal periods: a skipped
    report (e.g. Q2 missing between Q1 and Q3) ends the streak instead of being
    silently treated as a single step (review R4-23).  ``max_periods`` bounds
    how far back the streak may look.  Unparseable period ids (no fiscal
    ordinal) fall back to position-based appearance order — the only notion of
    "consecutive" available for them.
    """
    try:
        pos = order.index(current)
    except ValueError:
        return 0.0
    cur_ord = period_ordinal(current)
    if cur_ord is None:
        start = max(0, pos - max_periods + 1) if max_periods else 0
        vals = [float(visible[order[i]]) for i in range(start, pos + 1)]
        if len(vals) < 2:
            return 0.0
        count = 0
        for z in np.diff(vals)[::-1]:
            if (z > 0) if positive else (z < 0):
                count += 1
            else:
                break
        return float(count)
    start = max(0, pos - max_periods + 1) if max_periods else 0
    count = 0
    prev_key = current
    for key in reversed(order[start:pos]):
        key_ord = period_ordinal(key)
        if key_ord is None or cur_ord - key_ord != 1:
            break
        d = float(visible[prev_key]) - float(visible[key])
        if (d > 0) if positive else (d < 0):
            count += 1
        else:
            break
        prev_key = key
        cur_ord = key_ord
    return float(count)


def _value_streak_span(
    order: list[object],
    visible: OrderedDict,
    current: object,
    *,
    positive: bool,
    max_periods: int | None = None,
) -> float:
    """Count consecutive visible report periods whose *value* has the given sign.

    The walk breaks at a skipped fiscal period (a missing report's sign is
    unknown — it must not bridge two non-adjacent quarters, review R4-23).
    ``max_periods`` bounds how many periods (including the current one) may
    contribute.  Unparseable period ids fall back to appearance order.
    """
    try:
        pos = order.index(current)
    except ValueError:
        return 0.0
    cur_ord = period_ordinal(current)
    if cur_ord is None:
        start = max(0, pos - max_periods + 1) if max_periods else 0
        vals = [float(visible[order[i]]) for i in range(start, pos + 1)]
        streak = 0
        for v in vals[::-1]:
            if (v > 0) if positive else (v < 0):
                streak += 1
            else:
                break
        return float(streak)
    cur_val = float(visible[current])
    if (cur_val > 0) if positive else (cur_val < 0):
        streak = 1
    else:
        return 0.0
    prev_ord = cur_ord
    walked = 0
    for key in reversed(order[:pos]):
        if max_periods is not None and walked >= max_periods - 1:
            break
        key_ord = period_ordinal(key)
        if key_ord is None or prev_ord - key_ord != 1:
            break
        val = float(visible[key])
        if (val > 0) if positive else (val < 0):
            streak += 1
            walked += 1
            prev_ord = key_ord
        else:
            break
    return float(streak)


def _lag_value(order, visible, current, periods: int):
    """Exact ordinal lag: the value whose fiscal ordinal is ``current - periods``.

    Position-based lookup breaks when a quarter is missing (a skipped report period
    must yield NaN, not the value of a non-adjacent quarter) — audit §4.2.
    """
    periods = _pos_int(periods, "periods")
    target = period_ordinal(current)
    if target is None:
        return np.nan
    target -= periods
    for key in reversed(order):
        if period_ordinal(key) == target:
            return float(visible.get(key, np.nan))
    return np.nan


def _register(name: str, params: Iterable[str], fn, description: str, *, tags=(), param_specs: dict[str, ParamSpec] | None = None):
    metadata = OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=list(params),
        return_type="series",
        param_specs={k: v for k, v in (param_specs or {}).items() if k in params},
        tags=["fundamental", "period_aware", "pit_safe", "causal", "production_extension", *tags],
    )
    # Round-11 #51: expose the expected reporting-flow grain on the contract.
    flow_tag = _FLOW_TYPE_TAGS.get(name)
    if flow_tag is not None and flow_tag not in metadata.tags:
        metadata.tags = list(metadata.tags) + [flow_tag]
    # Round-3 item 30: declare the input semantics (Return/Rate vs raw price/volume).
    if name in _INPUT_SEMANTICS_RATE:
        metadata.tags = list(metadata.tags) + ["input_semantics:rate"]
    units = _INPUT_UNITS.get(name)
    if units is not None:
        metadata.input_units = dict(units)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"FundamentalV2_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental_transforms_v2",
        backend="pandas_numpy",
        status="production",
    )(cls)


def fin_lag(x, period_id, periods=1):
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: _lag_value(o, v, c, p))


def fin_diff(x, period_id, periods=1):
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, p))


def fin_pct_change(x, period_id, periods=1, flow_type=None):
    # Finding #52: growth over a cumulative-YTD input is not a period growth
    # rate; reject a caller-declared YTD grain instead of silently computing
    # Q2-YTD / Q1-YTD as "quarterly growth".
    reject_ytd_growth("fin_pct_change", flow_type)
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: _safe_div(float(v[c]), _lag_value(o, v, c, p)) - 1.0)


def fin_log_change(x, period_id, periods=1, flow_type=None):
    reject_ytd_growth("fin_log_change", flow_type)
    p = _pos_int(periods, "periods")
    def calc(o, v, c):
        old = _lag_value(o, v, c, p); cur = float(v[c])
        return float(np.log(cur / old)) if cur > 0 and old > 0 else np.nan
    return _walk_periods(x, period_id, calc)


def fin_qoq(x, period_id, flow_type=None):
    return fin_pct_change(x, period_id, 1, flow_type)


def fin_yoy(x, period_id, periods_per_year=4, flow_type=None):
    return fin_pct_change(x, period_id, _pos_int(periods_per_year, "periods_per_year"), flow_type)


def fin_ttm(x, period_id, periods_per_year=4, flow_type=None):
    reject_ytd_growth("fin_ttm", flow_type)
    n = _pos_int(periods_per_year, "periods_per_year")
    def calc(o, v, c):
        # TTM is a sum of *adjacent* fiscal periods: a skipped report must not
        # pull in a non-adjacent quarter (review P0-03).
        vals = _values(o, v, c, n, require_consecutive=True)
        return float(np.sum(vals)) if len(vals) == n else np.nan
    return _walk_periods(x, period_id, calc)


def fin_average_balance(x, period_id, periods=2):
    n = _pos_int(periods, "periods")
    def calc(o, v, c):
        # Average balance over a contiguous window of fiscal periods (review
        # P0-03): a skipped report period fails closed instead of substituting
        # a non-adjacent report.
        vals = _values(o, v, c, n, require_consecutive=True)
        return float(np.mean(vals)) if len(vals) == n else np.nan
    return _walk_periods(x, period_id, calc)


def fin_growth(x, period_id, periods=1, flow_type=None):
    return fin_pct_change(x, period_id, periods, flow_type)


def fin_cagr(x, period_id, periods=4, periods_per_year=4, flow_type=None):
    reject_ytd_growth("fin_cagr", flow_type)
    p = _pos_int(periods, "periods")
    ppy = _pos_int(periods_per_year, "periods_per_year")
    def calc(o, v, c):
        old = _lag_value(o, v, c, p); cur = float(v[c])
        if cur <= 0 or old <= 0:
            return np.nan
        return float((cur / old) ** (ppy / p) - 1.0)
    return _walk_periods(x, period_id, calc)


def fin_growth_acceleration(x, period_id, short_periods=1, long_periods=4, flow_type=None):
    s = _pos_int(short_periods, "short_periods")
    l = _pos_int(long_periods, "long_periods")
    if s >= l:
        raise ValueError("short_periods must be < long_periods")
    short = fin_pct_change(x, period_id, s, flow_type)
    long = fin_pct_change(x, period_id, l, flow_type)
    return short - long


def fin_growth_change(x, period_id, growth_periods=4, compare_periods=1, flow_type=None):
    g = fin_pct_change(x, period_id, _pos_int(growth_periods, "growth_periods"), flow_type)
    return g - fin_lag(g, period_id, _pos_int(compare_periods, "compare_periods"))


def _rolling_period_stat(x, period_id, periods: int, reducer, require_consecutive: bool = False):
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n, require_consecutive), dtype=float)
        return float(reducer(vals)) if len(vals) == n else np.nan
    return _walk_periods(x, period_id, calc)


def fin_std(x, period_id, periods=8):
    return _rolling_period_stat(x, period_id, periods, lambda a: np.std(a, ddof=1))


def fin_mean_abs_deviation(x, period_id, periods=8):
    """Mean absolute deviation: mean(|x - mean(x)|).

    This is the (non-robust) L1 dispersion of reporting-period values.  The
    legacy name ``fin_mad`` is retained as an alias, but the truly robust
    median-based MAD lives under ``fin_median_abs_deviation`` (review R4-28).
    """
    return _rolling_period_stat(x, period_id, periods, lambda a: np.mean(np.abs(a - np.mean(a))))


# ``fin_mad`` historically implemented mean absolute deviation, NOT the robust
# median(|x - median|) MAD.  Keep it as a compatibility alias of
# ``fin_mean_abs_deviation``; the real MAD is ``fin_median_abs_deviation``.
fin_mad = fin_mean_abs_deviation


def fin_median_abs_deviation(x, period_id, periods=8):
    """Robust median absolute deviation: median(|x - median(x)|) (review R4-28)."""
    return _rolling_period_stat(x, period_id, periods, lambda a: float(np.median(np.abs(a - np.median(a)))))


def fin_cv(x, period_id, periods=8):
    def reducer(a):
        mean = float(np.mean(a)); sd = float(np.std(a, ddof=1))
        return sd / abs(mean) if abs(mean) > _EPS else np.nan
    return _rolling_period_stat(x, period_id, periods, reducer)


def fin_stability(x, period_id, periods=8):
    cv = fin_cv(x, period_id, periods)
    return 1.0 / (1.0 + cv.abs())


def fin_range(x, period_id, periods=8):
    return _rolling_period_stat(x, period_id, periods, lambda a: np.max(a) - np.min(a))


def fin_zscore_history(x, period_id, periods=8):
    """INCLUSIVE z-score: the current report is part of the reference sample.

    This mechanically compresses extremity (the current observation is always
    inside the sample it is scored against).  Prefer
    ``fin_zscore_vs_prior_history`` when the current report should be scored
    out-of-history against the previous reports only (review R4-25).
    """
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        if len(vals) != n:
            return np.nan
        sd = float(np.std(vals, ddof=1))
        # Zero variance -> the z-score is undefined; 0 would masquerade as
        # "exactly at the historical mean".  Return NaN and let a dedicated
        # constant-flag operator surface the degenerate case.
        return (float(vals[-1]) - float(np.mean(vals))) / sd if sd > _EPS else np.nan
    return _walk_periods(x, period_id, calc)


def fin_zscore_vs_prior_history(x, period_id, periods=8):
    """Z-score of the current report against the PRIOR reports only.

    The reference sample is the previous ``periods - 1`` visible reports; the
    current report is an out-of-history observation, so its extremity is not
    mechanically compressed by inclusion in its own reference distribution
    (review R4-25).  Skips missing report periods (last-N-visible).
    """
    n = _pos_int(periods, "periods", 3)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        if len(vals) != n:
            return np.nan
        history = vals[:-1]
        current = float(vals[-1])
        sd = float(np.std(history, ddof=1))
        return (current - float(np.mean(history))) / sd if sd > _EPS else np.nan
    return _walk_periods(x, period_id, calc)


def fin_percentile_history(x, period_id, periods=8):
    """INCLUSIVE percentile: the current report is part of the reference sample.

    Prefer ``fin_percentile_vs_prior_history`` to score the current report
    against the previous reports only (review R4-25).
    """
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        if len(vals) != n:
            return np.nan
        return float((np.sum(vals < vals[-1]) + 0.5 * np.sum(vals == vals[-1])) / len(vals))
    return _walk_periods(x, period_id, calc)


def fin_percentile_vs_prior_history(x, period_id, periods=8):
    """Percentile of the current report against the PRIOR reports only.

    The reference sample is the previous ``periods - 1`` visible reports; the
    current report is an out-of-history observation (review R4-25).  Skips
    missing report periods (last-N-visible).
    """
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        if len(vals) != n:
            return np.nan
        history = vals[:-1]
        current = float(vals[-1])
        return float((np.sum(history < current) + 0.5 * np.sum(history == current)) / len(history))
    return _walk_periods(x, period_id, calc)


def _trend_stat(x, period_id, periods, which: str, require_consecutive: bool = True):
    n = _pos_int(periods, "periods", 3)
    def calc(o, v, c):
        keys = _window_keys(o, v, c, n, require_consecutive)
        if len(keys) != n:
            return np.nan
        y = np.asarray([float(v[k]) for k in keys], dtype=float)
        # R4-24: the trend time axis is the *fiscal ordinal* (period_id's fiscal
        # sequence), not arange(n): a skipped Q2 between Q1 and Q3 must not be
        # counted as a single step.  require_consecutive additionally fails the
        # window closed on any gap, so the ordinals are guaranteed contiguous.
        xx = np.asarray([float(period_ordinal(k)) for k in keys], dtype=float)
        if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(y)):
            return np.nan
        xb, yb = float(xx.mean()), float(y.mean())
        den = float(np.sum((xx-xb)**2))
        if den <= _EPS:
            return np.nan
        slope = float(np.sum((xx-xb)*(y-yb))/den)
        if which == "slope":
            return slope
        fitted = yb + slope*(xx-xb)
        resid = y-fitted
        ss_res = float(np.sum(resid**2)); ss_tot=float(np.sum((y-yb)**2))
        if which == "r2":
            # Constant series has no identifiable trend; a perfect "fit" of 1.0
            # would be meaningless, so return NaN instead.
            return 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan
        if n <= 2:
            return np.nan
        mse = ss_res/(n-2)
        se = float(np.sqrt(mse/den)) if mse >= 0 else np.nan
        return slope/se if np.isfinite(se) and se > _EPS else np.nan
    return _walk_periods(x, period_id, calc)


def fin_trend_slope(x, period_id, periods=8): return _trend_stat(x, period_id, periods, "slope")
def fin_trend_r2(x, period_id, periods=8): return _trend_stat(x, period_id, periods, "r2")
def fin_trend_tstat(x, period_id, periods=8): return _trend_stat(x, period_id, periods, "tstat")


def fin_trend_acceleration(x, period_id, short_periods=4, long_periods=8):
    s=_pos_int(short_periods,"short_periods",3); l=_pos_int(long_periods,"long_periods",3)
    if s >= l: raise ValueError("short_periods must be < long_periods")
    return fin_trend_slope(x,period_id,s)-fin_trend_slope(x,period_id,l)


def fin_monotonicity(x, period_id, periods=8):
    # Monotonicity is measured over *adjacent* fiscal periods: a skipped report
    # (missing Q2) must fail closed rather than being treated as a single step
    # (review R4-23).
    n=_pos_int(periods,"periods",2)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n,require_consecutive=True),dtype=float)
        if len(vals)!=n:return np.nan
        d=np.diff(vals)
        return float((np.sum(d>0)-np.sum(d<0))/max(1,len(d)))
    return _walk_periods(x,period_id,calc)


def _streak(x, period_id, positive: bool):
    # A streak counts report-to-report changes only between *adjacent* fiscal
    # periods; a skipped report ends the streak (review R4-23).
    return _walk_periods(
        x, period_id, lambda o, v, c: _streak_span(o, v, c, positive=positive)
    )


def fin_positive_streak(x,period_id): return _streak(x,period_id,True)
def fin_negative_streak(x,period_id): return _streak(x,period_id,False)


def fin_sign_change_count(x,period_id,periods=8):
    # Direction reversals are counted over *adjacent* fiscal periods (review
    # R4-23): a skipped report fails the window closed.
    n=_pos_int(periods,"periods",3)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n,require_consecutive=True),dtype=float)
        if len(vals)!=n:return np.nan
        signs=np.sign(np.diff(vals)); signs=signs[signs!=0]
        return float(np.sum(signs[1:]!=signs[:-1])) if len(signs)>1 else 0.0
    return _walk_periods(x,period_id,calc)


def fin_growth_volatility(x,period_id,growth_periods=1,window_periods=8,flow_type=None):
    # The growth sequence is measured over *adjacent* fiscal periods: a skipped
    # report fails the window closed instead of mixing multi-period changes
    # into the volatility (review R4-23).  A cumulative-YTD input is rejected
    # (finding #52).
    reject_ytd_growth("fin_growth_volatility", flow_type)
    g=fin_pct_change(x,period_id,_pos_int(growth_periods,"growth_periods"),flow_type)
    return _rolling_period_stat(
        g, period_id, _pos_int(window_periods, "window_periods", 2),
        lambda a: np.std(a, ddof=1), require_consecutive=True,
    )


def fin_growth_stability(x,period_id,growth_periods=1,window_periods=8,flow_type=None):
    vol=fin_growth_volatility(x,period_id,growth_periods,window_periods,flow_type)
    return 1.0/(1.0+vol.abs())


def fin_growth_persistence(x,period_id,growth_periods=1,window_periods=8,flow_type=None):
    reject_ytd_growth("fin_growth_persistence", flow_type)
    g=fin_pct_change(x,period_id,_pos_int(growth_periods,"growth_periods"),flow_type)
    n=_pos_int(window_periods,"window_periods",2)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n,require_consecutive=True),dtype=float)
        return float(np.mean(vals>0)) if len(vals)==n else np.nan
    return _walk_periods(g,period_id,calc)


def fin_ratio(numerator,denominator):
    den=denominator.replace(0,np.nan)
    return (numerator/den).replace([np.inf,-np.inf],np.nan)


def fin_common_size(x,base): return fin_ratio(x,base)


def fin_turnover(flow,balance,period_id,average_periods=2):
    avg=fin_average_balance(balance,period_id,_pos_int(average_periods,"average_periods"))
    return fin_ratio(flow,avg)


def fin_divergence(x,y,period_id,periods=4):
    p=_pos_int(periods,"periods")
    return fin_pct_change(x,period_id,p)-fin_pct_change(y,period_id,p)


def fin_cash_earnings_gap(earnings,cashflow,scale,flow_type=None):
    # Finding #51/#53: the two flows must be the SAME reporting grain.
    require_same_flow_grain("fin_cash_earnings_gap", flow_type, 2)
    return fin_ratio(earnings-cashflow,scale.abs())


def fin_accrual_ratio(earnings,cashflow,assets,flow_type=None):
    require_same_flow_grain("fin_accrual_ratio", flow_type, 2)
    return fin_ratio(earnings-cashflow,assets.abs())


def fin_cash_conversion(cashflow,earnings,flow_type=None):
    require_same_flow_grain("fin_cash_conversion", flow_type, 2)
    return fin_ratio(cashflow,earnings)


def fin_working_capital_change(working_capital,period_id,periods=1):
    return fin_diff(working_capital,period_id,_pos_int(periods,"periods"))


_SPECS = [
    ("fin_lag",["x","period_id","periods"],fin_lag,"Lag by visible reporting periods, never by trading days."),
    ("fin_diff",["x","period_id","periods"],fin_diff,"Difference versus a prior visible report period."),
    ("fin_pct_change",["x","period_id","periods","flow_type"],fin_pct_change,"Percent change versus a prior visible report period. flow_type declares the input reporting grain (CumulativeYTDFlow rejected, #52)."),
    ("fin_log_change",["x","period_id","periods","flow_type"],fin_log_change,"Log change versus a prior visible report period."),
    ("fin_qoq",["x","period_id","flow_type"],fin_qoq,"Quarter-over-quarter/one-report-period growth."),
    ("fin_yoy",["x","period_id","periods_per_year","flow_type"],fin_yoy,"Year-over-year growth with configurable periods per year."),
    ("fin_ttm",["x","period_id","periods_per_year","flow_type"],fin_ttm,"Rolling sum over *adjacent* fiscal report periods; a skipped report fails closed to NaN (review R4-23)."),
    ("fin_average_balance",["x","period_id","periods"],fin_average_balance,"Average balance over *adjacent* fiscal report periods; a skipped report fails closed to NaN (review R4-23)."),
    ("fin_growth",["x","period_id","periods","flow_type"],fin_growth,"Generic reporting-period growth. Cumulative-YTD input rejected (#52); convert via fin_quarter_from_cumulative first."),
    ("fin_cagr",["x","period_id","periods","periods_per_year","flow_type"],fin_cagr,"Reporting-period CAGR with configurable annualization."),
    ("fin_growth_acceleration",["x","period_id","short_periods","long_periods","flow_type"],fin_growth_acceleration,"Short-horizon minus long-horizon fundamental growth."),
    ("fin_growth_change",["x","period_id","growth_periods","compare_periods","flow_type"],fin_growth_change,"Change in a reporting-period growth rate."),
    ("fin_growth_volatility",["x","period_id","growth_periods","window_periods","flow_type"],fin_growth_volatility,"Volatility of period growth across *adjacent* fiscal periods; a skipped report fails closed to NaN (review R4-23)."),
    ("fin_growth_stability",["x","period_id","growth_periods","window_periods","flow_type"],fin_growth_stability,"Inverse growth-volatility stability score over adjacent fiscal periods (skips fail closed, review R4-23)."),
    ("fin_growth_persistence",["x","period_id","growth_periods","window_periods","flow_type"],fin_growth_persistence,"Fraction of recent *adjacent* fiscal reports with positive growth (review R4-23)."),
    ("fin_std",["x","period_id","periods"],fin_std,"Historical reporting-period standard deviation over the last-N-visible reports; missing report periods are skipped (review R4-23)."),
    ("fin_mean_abs_deviation",["x","period_id","periods"],fin_mean_abs_deviation,"Mean absolute deviation mean(|x-mean|) over the last-N-visible reports; missing report periods are skipped (review R4-28 naming)."),
    ("fin_median_abs_deviation",["x","period_id","periods"],fin_median_abs_deviation,"Robust median absolute deviation median(|x-median|) over the last-N-visible reports; missing report periods are skipped (review R4-28)."),
    ("fin_cv",["x","period_id","periods"],fin_cv,"Historical reporting-period coefficient of variation over the last-N-visible reports; missing report periods are skipped."),
    ("fin_stability",["x","period_id","periods"],fin_stability,"Generic bounded fundamental stability score over the last-N-visible reports; missing report periods are skipped."),
    ("fin_range",["x","period_id","periods"],fin_range,"Historical reporting-period range over the last-N-visible reports; missing report periods are skipped."),
    ("fin_zscore_history",["x","period_id","periods"],fin_zscore_history,"INCLUSIVE z-score: current report is inside the reference sample (review R4-25); last-N-visible, missing reports skipped."),
    ("fin_zscore_vs_prior_history",["x","period_id","periods"],fin_zscore_vs_prior_history,"Z-score of the current report vs the PRIOR reports only (out-of-history; review R4-25); last-N-visible, missing reports skipped."),
    ("fin_percentile_history",["x","period_id","periods"],fin_percentile_history,"INCLUSIVE percentile: current report is inside the reference sample (review R4-25); last-N-visible, missing reports skipped."),
    ("fin_percentile_vs_prior_history",["x","period_id","periods"],fin_percentile_vs_prior_history,"Percentile of the current report vs the PRIOR reports only (out-of-history; review R4-25); last-N-visible, missing reports skipped."),
    ("fin_trend_slope",["x","period_id","periods"],fin_trend_slope,"OLS slope across recent report periods on the fiscal-ordinal time axis; a skipped report fails closed (reviews R4-23/24)."),
    ("fin_trend_r2",["x","period_id","periods"],fin_trend_r2,"OLS trend R-squared across recent report periods on the fiscal-ordinal time axis; a skipped report fails closed (reviews R4-23/24)."),
    ("fin_trend_tstat",["x","period_id","periods"],fin_trend_tstat,"OLS trend slope t-statistic across recent report periods on the fiscal-ordinal time axis; a skipped report fails closed (reviews R4-23/24)."),
    ("fin_trend_acceleration",["x","period_id","short_periods","long_periods"],fin_trend_acceleration,"Difference between short and long reporting-period slopes (fiscal-ordinal axis, skips fail closed)."),
    ("fin_monotonicity",["x","period_id","periods"],fin_monotonicity,"Signed monotonicity of recent reporting-period changes over *adjacent* fiscal periods; a skipped report fails closed (review R4-23)."),
    ("fin_positive_streak",["x","period_id"],fin_positive_streak,"Current streak of positive report-to-report changes between adjacent fiscal periods; a skipped report ends the streak (review R4-23)."),
    ("fin_negative_streak",["x","period_id"],fin_negative_streak,"Current streak of negative report-to-report changes between adjacent fiscal periods; a skipped report ends the streak (review R4-23)."),
    ("fin_sign_change_count",["x","period_id","periods"],fin_sign_change_count,"Number of growth-direction reversals in recent reports over *adjacent* fiscal periods; a skipped report fails closed (review R4-23)."),
    ("fin_ratio",["numerator","denominator"],fin_ratio,"Generic finite ratio; domain ratios should be recipes over this primitive."),
    ("fin_common_size",["x","base"],fin_common_size,"Common-size accounting transform x/base."),
    ("fin_turnover",["flow","balance","period_id","average_periods"],fin_turnover,"Flow divided by average reporting-period balance."),
    ("fin_divergence",["x","y","period_id","periods"],fin_divergence,"Difference between two reporting-period growth rates."),
    ("fin_cash_earnings_gap",["earnings","cashflow","scale","flow_type"],fin_cash_earnings_gap,"Scaled earnings-minus-cash-flow gap; the two flows must share one grain (#53)."),
    ("fin_accrual_ratio",["earnings","cashflow","assets","flow_type"],fin_accrual_ratio,"Accrual proxy (earnings-cash flow)/assets; the two flows must share one grain (#53)."),
    ("fin_cash_conversion",["cashflow","earnings","flow_type"],fin_cash_conversion,"Cash flow divided by earnings; the two flows must share one grain (#53)."),
    ("fin_working_capital_change",["working_capital","period_id","periods"],fin_working_capital_change,"Reporting-period change in working capital."),
]

for _name,_params,_fn,_desc in _SPECS:
    _register(
        _name, _params, _fn, _desc,
        param_specs=(
            _VS_PRIOR_HISTORY_PARAM_SPECS
            if _name in ("fin_zscore_vs_prior_history", "fin_percentile_vs_prior_history")
            else None
        ),
    )

# R40 #146: ``fin_mad`` 是 ``fin_mean_abs_deviation`` 的 deprecated alias —— 两个
# 名字都注册 canonical 会让 mining 重复搜索同一语义空间。从 canonical 注册移除
# （保留上方 `fin_mad = fin_mean_abs_deviation` 函数别名），只留 registry alias，
# 使 ``resolve_canonical("fin_mad") -> "fin_mean_abs_deviation"``（公式解析不受影响）。
from cleaned_operators.registry import OperatorRegistry as _OperatorRegistry

_OperatorRegistry.register_compat_alias(
    "fin_mad",
    "fin_mean_abs_deviation",
    migration_reason="legacy name for mean absolute deviation; canonical is fin_mean_abs_deviation (R4-28)",
    deprecated_since="0.11.0",
    removal_version="1.0",
)

# New/renamed canonicals (review R4-25 / R4-28) must join the audited
# fundamental-v2 surface so production governance and the daily/extended
# partitions cover them (mirrors transforms_repairs_v2 / expectation_v2).
import cleaned_operators.operator_surface as _surface

_NEW_V2_CANONICALS = frozenset({
    "fin_mean_abs_deviation",
    "fin_median_abs_deviation",
    "fin_zscore_vs_prior_history",
    "fin_percentile_vs_prior_history",
})
_surface._FUNDAMENTAL_V2_CANONICALS = frozenset(
    set(_surface._FUNDAMENTAL_V2_CANONICALS) | _NEW_V2_CANONICALS
)
_surface.extend_extended_only(set(_NEW_V2_CANONICALS))

# Round-3 item 27: the per-period fundamental revision ledger (release +
# supersede/revision timestamps per period, PIT ``value_as_of`` lookup).
from cleaned_operators.fundamental import ledger as _fundamental_revision_ledger  # noqa: E402,F401


# ============================================================================
# Industry-grouped fiscal regression operators
# ============================================================================

def industry_fiscal_resid(
    y,
    period_id,
    industry,
    x1,
    x2=None,
    x3=None,
    x4=None,
    x5=None,
    periods=12,
    min_obs=None,
    add_intercept=True,
    require_consecutive=True,
    revision_policy="latest_available",
):
    """Industry-within fiscal-period regression residual.

    For each industry group independently, fits OLS regression over the most
    recent `periods` distinct fiscal report events and returns the current
    event's residual (y_t - predict(x_t)).

    Uses FiscalEventView to ensure PIT semantics: only report events visible
    at each decision timestamp contribute, and revisions update only from the
    revision timestamp forward.

    Parameters
    ----------
    y : pd.DataFrame
        Dependent variable panel (daily forward-filled).
    period_id : pd.DataFrame
        Fiscal period identifier panel (e.g., "2024Q3").
    industry : pd.DataFrame
        Industry classification panel (integer or string codes).
    x1, x2, x3, x4, x5 : pd.DataFrame or None
        Independent variable panels (variadic, up to 5 regressors).
    periods : int, default=12
        Rolling window length in distinct fiscal report events.
    min_obs : int or None
        Minimum observations required to fit; defaults to `periods`.
    add_intercept : bool, default=True
        Whether to include an intercept in the regression.
    require_consecutive : bool, default=True
        If True, fail closed when fiscal periods are not consecutive.
    revision_policy : str, default="latest_available"
        Either "latest_available" or "first_available" for revision handling.

    Returns
    -------
    pd.DataFrame
        Panel of residuals (y_t - y_hat_t) for the current fiscal event, or
        NaN when insufficient history, rank deficiency, or industry missing.

    Notes
    -----
    Each industry group maintains independent regression state. An instrument
    with missing `industry` returns NaN for all rows.
    """
    from cleaned_operators.fiscal_event_ops import FiscalEventView, _align

    # Collect all input panels
    inputs = [y, period_id, industry, x1]
    for x in (x2, x3, x4, x5):
        if x is not None:
            inputs.append(x)

    # Strict alignment check
    _align(*inputs)

    # Validate parameters
    periods = _pos_int(periods, "periods")
    min_obs = periods if min_obs is None else _pos_int(min_obs, "min_obs")
    if not isinstance(add_intercept, bool):
        add_intercept = bool(add_intercept)
    if not isinstance(require_consecutive, bool):
        require_consecutive = bool(require_consecutive)

    # Build FiscalEventViews for each panel
    y_view = FiscalEventView.from_panel(y, period_id, revision_policy=revision_policy)
    industry_view = FiscalEventView.from_panel(industry, period_id, revision_policy=revision_policy)

    x_views = [FiscalEventView.from_panel(x1, period_id, revision_policy=revision_policy)]
    for x in (x2, x3, x4, x5):
        if x is not None:
            x_views.append(FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy))

    out = np.full(y.shape, np.nan, dtype=float)

    # Process each column independently
    for col in range(y.shape[1]):
        # Build per-industry state: map industry_code -> list[(ordinal, y_val, x_vals)]
        industry_history: dict[object, list[tuple[int, float, list[float]]]] = {}

        for row in range(y.shape[0]):
            # Current visible histories (don't apply require_consecutive at view level)
            y_hist = dict(y_view.history(row, col, require_consecutive=False))
            ind_hist = dict(industry_view.history(row, col, require_consecutive=False))
            x_hists = [dict(view.history(row, col, require_consecutive=False)) for view in x_views]

            # Current ordinal
            current_ordinal = y_view.ordinals[row, col]
            if not np.isfinite(current_ordinal):
                continue
            current_ordinal = int(current_ordinal)

            # Update industry_history with all visible events up to current
            common_ordinals = set(y_hist) & set(ind_hist)
            for x_hist in x_hists:
                common_ordinals &= set(x_hist)

            for ordinal in sorted(common_ordinals):
                if ordinal > current_ordinal:
                    continue
                ind_code = ind_hist[ordinal]
                if not _finite(ind_code):
                    continue
                y_val = y_hist[ordinal]
                x_vals = [x_hist[ordinal] for x_hist in x_hists]
                if not _finite(y_val) or not all(_finite(xv) for xv in x_vals):
                    continue

                # Store or update this ordinal for this industry
                if ind_code not in industry_history:
                    industry_history[ind_code] = []

                # Replace if ordinal already exists (revision), otherwise append
                existing_idx = next((i for i, (o, _, _) in enumerate(industry_history[ind_code]) if o == ordinal), None)
                if existing_idx is not None:
                    industry_history[ind_code][existing_idx] = (ordinal, y_val, x_vals)
                else:
                    industry_history[ind_code].append((ordinal, y_val, x_vals))
                    industry_history[ind_code].sort(key=lambda item: item[0])

            # Now compute residual for the current row
            if current_ordinal not in ind_hist:
                continue
            current_industry = ind_hist[current_ordinal]
            if not _finite(current_industry) or current_industry not in industry_history:
                continue

            # Get the window for this industry
            ind_events = industry_history[current_industry]
            window = [evt for evt in ind_events if evt[0] <= current_ordinal][-periods:]

            if len(window) < min_obs:
                continue

            # Check consecutiveness if required
            if require_consecutive and len(window) > 1:
                ordinals = [evt[0] for evt in window]
                is_consecutive = all(ordinals[i] == ordinals[i-1] + 1 for i in range(1, len(ordinals)))
                if not is_consecutive:
                    continue

            # Extract y and X arrays
            y_array = np.array([evt[1] for evt in window], dtype=float)
            x_array = np.array([evt[2] for evt in window], dtype=float)  # shape: (n_obs, n_features)

            # Build design matrix
            if add_intercept:
                design = np.column_stack([np.ones(len(y_array)), x_array])
            else:
                design = x_array

            # Check rank
            residual_dof = len(y_array) - design.shape[1]
            if residual_dof <= 0 or np.linalg.matrix_rank(design) != design.shape[1]:
                continue

            # Fit OLS
            try:
                coefficients = np.linalg.lstsq(design, y_array, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            # Compute residual for current observation (last in window)
            current_y = y_array[-1]
            current_x_row = design[-1]
            y_hat = float(current_x_row @ coefficients)
            residual = current_y - y_hat

            if _finite(residual):
                out[row, col] = residual

    return pd.DataFrame(out, index=y.index, columns=y.columns)


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


# Register the operator
_register(
    "industry_fiscal_resid",
    ["y", "period_id", "industry", "x1", "x2", "x3", "x4", "x5", "periods", "min_obs", "add_intercept", "require_consecutive", "revision_policy"],
    industry_fiscal_resid,
    "Industry-grouped fiscal-period OLS regression residual with PIT semantics; each industry fits independently over distinct report events.",
)

_surface.extend_extended_only({"industry_fiscal_resid"})

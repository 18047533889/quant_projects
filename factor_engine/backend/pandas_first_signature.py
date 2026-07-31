# -*- coding: utf-8 -*-
"""Bounded production call validation for single-backend/Extended operators.

Production eligibility is backend-independent, but every public factor call must
also have a static, deterministic parameter contract. This validator runs on the
canonical Plan after aliases/macros have been normalized.
"""
from __future__ import annotations

import math
from typing import Any

from planner.logical_plan import PlanNode


_WINDOW_NAMES = (
    "window", "d", "n", "span", "period", "periods", "lookback",
    "max_lookback", "fast", "slow", "fast_period", "slow_period",
    "signal_span", "signal_window", "signal_period",
    "left_window", "right_window", "history_window",
)

_PANEL_NAMES = frozenset({
    "x", "y", "z", "a", "b", "w", "g", "left", "right",
    "numerator", "denominator", "ret", "returns", "benchmark_ret",
    "market_ret", "benchmark", "market", "open", "high", "low", "close",
    "price", "volume", "amount", "vwap", "turnover", "weight", "weights", "condition",
    "group", "industry", "sector", "fiscal_quarter", "period_id", "quarter",
    "revision_id", "decision_time", "available_time", "available_at",
    "exposure", "exposures", "control", "controls", "factor", "target",
    "mask", "event", "sort_col", "float_shares",
})

_MACD = frozenset({"MACD_line", "MACD_signal", "MACD_hist"})
_QUANTILE_Q = frozenset({
    "ts_quantile", "cs_quantile", "group_percentile", "ts_tail_mean",
    "tail_beta", "lqtp_historical_cvar",
})
_TOPK = frozenset({
    "ts_topk_sum", "ts_topk_mean", "ts_topk_std",
    "ts_bottomk_sum", "ts_bottomk_mean", "ts_bottomk_std",
})
_MIN_WINDOW_TWO = frozenset({
    "AROON", "AROON_up", "AROON_down", "CCI", "efficiency_ratio",
    "choppiness_index", "parkinson_vol", "garman_klass_vol",
    "rogers_satchell_vol", "overnight_volatility", "intraday_volatility",
    "range_volatility", "ulcer_index", "bollinger_pct_b", "bollinger_width",
})
_MIN_WINDOW_THREE = frozenset({"yang_zhang_vol"})
_STRUCTURE_WITH_HISTORY = frozenset({
    "ts_last_pivot_high", "ts_last_pivot_low", "ts_pivot_high_age", "ts_pivot_low_age",
    "ts_resistance_level", "ts_support_level", "ts_resistance_slope", "ts_support_slope",
    "ts_distance_to_resistance", "ts_distance_to_support", "ts_resistance_break",
    "ts_support_break",
})


def _literal(child: Any) -> Any | None:
    if getattr(child, "op", None) != "literal":
        return None
    return (getattr(child, "attrs", None) or {}).get("value")


def _is_panel_parameter(canonical: str, name: str) -> bool:
    if name in _PANEL_NAMES:
        return True
    if name == "signal":
        return canonical not in _MACD
    if name in {"value", "values", "fallback"}:
        return canonical not in {"fillna_const"}
    return False


def _call_values(node: PlanNode, canonical: str) -> tuple[dict[str, Any], list[str]]:
    from cleaned_operators.registry import OperatorRegistry

    catalog = OperatorRegistry._catalog.get(canonical, {})
    names = [str(x) for x in (catalog.get("param_names") or []) if str(x) != "..."]
    values: dict[str, Any] = {}
    dynamic: list[str] = []
    for index, name in enumerate(names):
        if _is_panel_parameter(canonical, name):
            continue
        if index >= len(node.inputs):
            continue
        child = node.inputs[index]
        if getattr(child, "op", None) == "literal":
            values[name] = (getattr(child, "attrs", None) or {}).get("value")
        else:
            dynamic.append(name)
    for name, value in dict(node.attrs or {}).items():
        if _is_panel_parameter(canonical, str(name)):
            continue
        values[str(name)] = value
    return values, sorted(set(dynamic))


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not _finite(value):
        return None
    number = float(value)
    if int(number) != number:
        return None
    return int(number)


def _first(values: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def _require_positive_int(canonical: str, values: dict[str, Any], name: str) -> str | None:
    if name not in values:
        return None
    value = _integer(values[name])
    if value is None or value <= 0:
        return f"{canonical}: {name} must be a positive integer"
    return None


def validate_pandas_first_call(canonical: str, node: PlanNode) -> tuple[bool, str]:
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS

    if canonical not in PANDAS_FIRST_PRODUCTION_CANONICALS:
        return True, ""

    values, dynamic = _call_values(node, canonical)
    if dynamic:
        return False, f"{canonical}: production tuning parameter(s) must be literal: {', '.join(dynamic)}"

    for name in _WINDOW_NAMES:
        error = _require_positive_int(canonical, values, name)
        if error:
            return False, error

    for name in ("min_periods", "min_obs", "run", "buckets", "top", "k", "points"):
        error = _require_positive_int(canonical, values, name)
        if error:
            return False, error

    window_raw = _first(values, ("window", "d", "span", "period"))
    window = _integer(window_raw) if window_raw is not None else None
    min_periods_raw = values.get("min_periods")
    if min_periods_raw is not None and window is not None:
        min_periods = _integer(min_periods_raw)
        if min_periods is not None and min_periods > window:
            return False, f"{canonical}: min_periods must be <= window"

    if canonical in _MIN_WINDOW_TWO and window is not None and window < 2:
        return False, f"{canonical}: window must be >= 2"
    if canonical in _MIN_WINDOW_THREE and window is not None and window < 3:
        return False, f"{canonical}: window must be >= 3"

    if canonical in _STRUCTURE_WITH_HISTORY:
        history = _integer(values.get("history_window")) if "history_window" in values else None
        points = _integer(values.get("points")) if "points" in values else None
        if points is not None and points < 2:
            return False, f"{canonical}: points must be >= 2"
        if history is not None and points is not None and points > history:
            return False, f"{canonical}: points must be <= history_window"

    if canonical in _QUANTILE_Q:
        for name in ("q", "quantile", "p", "fraction"):
            if name in values and (not _finite(values[name]) or not 0.0 <= float(values[name]) <= 1.0):
                return False, f"{canonical}: {name} must be in [0, 1]"

    if canonical in _TOPK and "k" in values and window is not None:
        k = _integer(values["k"])
        if k is not None and k > window:
            return False, f"{canonical}: k must be <= window"

    if canonical == "ts_moment" and "k" in values:
        k = _integer(values["k"])
        if k is None or not 1 <= k <= 8:
            return False, "ts_moment: production supports integer moment order 1..8"

    if canonical == "ts_sma_cn":
        n = _integer(values.get("n")) if "n" in values else None
        m = _integer(values.get("m")) if "m" in values else None
        if n is not None and m is not None and not 0 < m <= n:
            return False, "ts_sma_cn: require 0 < m <= n"

    if canonical == "digital_count":
        if "threshold" in values and (not _finite(values["threshold"]) or float(values["threshold"]) < 0):
            return False, "digital_count: threshold must be finite and non-negative"
        d = _integer(values.get("d")) if "d" in values else None
        run = _integer(values.get("run")) if "run" in values else None
        if d is not None and run is not None and run > d:
            return False, "digital_count: run must be <= d"

    if canonical == "hump_decay" and "hump" in values:
        if not _finite(values["hump"]) or float(values["hump"]) < 0:
            return False, "hump_decay: hump must be finite and non-negative"

    if canonical in _MACD:
        for name in ("fast", "slow", "signal", "fast_period", "slow_period", "signal_period"):
            error = _require_positive_int(canonical, values, name)
            if error:
                return False, error
        fast = _integer(_first(values, ("fast", "fast_period")))
        slow = _integer(_first(values, ("slow", "slow_period")))
        if fast is not None and slow is not None and fast >= slow:
            return False, f"{canonical}: fast period must be < slow period"

    if "std_dev" in values and (not _finite(values["std_dev"]) or float(values["std_dev"]) <= 0):
        return False, f"{canonical}: std_dev must be finite and > 0"

    if "ddof" in values:
        ddof = _integer(values["ddof"])
        if ddof not in {0, 1}:
            return False, f"{canonical}: ddof must be 0 or 1"

    if "decimals" in values:
        decimals = _integer(values["decimals"])
        if decimals is None or abs(decimals) > 12:
            return False, "round: production decimals must be an integer in [-12, 12]"

    if "add_intercept" in values and not isinstance(values["add_intercept"], bool):
        return False, f"{canonical}: add_intercept must be boolean"

    if canonical == "clip":
        lo, hi = values.get("lo"), values.get("hi")
        if lo is not None and hi is not None:
            if not (_finite(lo) and _finite(hi) and float(lo) <= float(hi)):
                return False, "clip: require finite lo <= hi"

    if canonical in {"group_winsorize", "winsorize"}:
        lo = values.get("lo", values.get("lower"))
        hi = values.get("hi", values.get("upper"))
        if lo is not None and hi is not None:
            if not (_finite(lo) and _finite(hi) and 0.0 <= float(lo) < float(hi) <= 1.0):
                return False, f"{canonical}: winsor bounds must satisfy 0 <= lower < upper <= 1"

    if "side" in values and str(values["side"]).lower() not in {"lower", "upper", "left", "right"}:
        return False, f"{canonical}: unsupported side={values['side']!r}"
    if "order" in values and str(values["order"]).lower() not in {"largest", "smallest", "asc", "desc"}:
        return False, f"{canonical}: unsupported order={values['order']!r}"

    for name, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not _finite(value):
            return False, f"{canonical}: {name} must be finite"

    return True, ""


def check_pandas_first_plan_signatures(plan: Any) -> list[str]:
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op:
            canonical = OperatorRegistry._aliases.get(op, op)
            if canonical in PANDAS_FIRST_PRODUCTION_CANONICALS:
                ok, message = validate_pandas_first_call(canonical, node)
                if not ok:
                    errors.append(message)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return errors

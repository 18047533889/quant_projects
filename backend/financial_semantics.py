# -*- coding: utf-8
"""金融算子公开定义：vwap / volatility / log_returns。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VwapKind = Literal["rolling_vwap"]
VolatilityKind = Literal["rolling_std_annualized"]
LogReturnKind = Literal["log_price_ratio"]


@dataclass(frozen=True)
class VwapSpec:
    """``vwap(price, volume, window)`` = 滚动 sum(price*volume)/sum(volume)。"""

    kind: VwapKind = "rolling_vwap"
    zero_volume_is_null: bool = True
    null_volume_is_null: bool = True
    negative_volume_is_null: bool = False
    reset_per_session: bool = False
    accumulate_in: Literal["float64", "decimal128"] = "float64"
    use_compensated_sum: bool = False


@dataclass(frozen=True)
class VolatilitySpec:
    """``volatility(x, window)`` = rolling_std(x, ddof=1) * sqrt(252)。

    输入 **x 为收益率序列**（调用方负责 log_returns / pct_change），不内部计算收益。
    """

    kind: VolatilityKind = "rolling_std_annualized"
    input_is_return: bool = True
    annualization_factor: float = 252.0**0.5
    ddof: int = 1
    min_periods_ratio: float = 0.5
    zero_vol_output: Literal["zero", "null"] = "zero"


@dataclass(frozen=True)
class LogReturnsSpec:
    """``log_returns(price)`` = log(price / lag(price, 1))。"""

    kind: LogReturnKind = "log_price_ratio"
    non_positive_is_null: bool = True
    null_propagates: bool = True


VWAP_SPEC = VwapSpec()
VOLATILITY_SPEC = VolatilitySpec()
LOG_RETURNS_SPEC = LogReturnsSpec()


def vwap_is_rolling() -> bool:
    return VWAP_SPEC.kind == "rolling_vwap"


def volatility_annualization_factor() -> float:
    return VOLATILITY_SPEC.annualization_factor


def log_returns_non_positive_is_null() -> bool:
    return LOG_RETURNS_SPEC.non_positive_is_null


def vwap_accumulate_float64() -> bool:
    return VWAP_SPEC.accumulate_in == "float64"


# ---------------------------------------------------------------------------
# P1-14: data-driven fiscal-event lookback.
#
# The analyzer sizes report-period operators (``fin_qoq``, ``fin_yoy``, ...) as
# ``rows_per_period * n_report_periods`` where ``rows_per_period`` is a fixed
# heuristic (``FACTOR_ENGINE_REPORT_PERIOD_LOOKBACK_ROWS``).  When a runtime
# financial source is active it can size the budget from its own fiscal-period
# calendar instead: N distinct fiscal events before ``decision_date`` map to a
# concrete number of pre-decision trading rows.
# ---------------------------------------------------------------------------
_ACTIVE_FINANCIAL_SOURCE = None


def set_active_financial_source(source) -> None:
    """Register the runtime financial source used to size report-period lookbacks.

    ``source`` should expose ``financial_period_calendar(decision_date, n_events)``
    returning the number of pre-``decision_date`` trading rows covering
    ``n_events`` distinct fiscal events (or ``None`` when it cannot answer).
    """
    global _ACTIVE_FINANCIAL_SOURCE
    _ACTIVE_FINANCIAL_SOURCE = source


def get_active_financial_source():
    """Return the registered runtime financial source (may be ``None``)."""
    return _ACTIVE_FINANCIAL_SOURCE


def fiscal_event_lookback(source, decision_date, n_events) -> int | None:
    """Return rows before ``decision_date`` covering ``n_events`` fiscal events.

    Consults ``source.financial_period_calendar(decision_date, n_events)`` when
    the source exposes it (the returned value is interpreted as a row count).
    Returns ``None`` when no calendar is available or the source answers ``None``,
    so the analyzer falls back to its env/heuristic default.
    """
    if source is None or n_events is None:
        return None
    try:
        n_events = int(n_events)
    except (TypeError, ValueError):
        return None
    if n_events <= 0:
        return None
    calendar_fn = getattr(source, "financial_period_calendar", None)
    if calendar_fn is None:
        return None
    try:
        rows = calendar_fn(decision_date, n_events)
    except Exception:
        return None
    if rows is None:
        return None
    try:
        rows = int(rows)
    except (TypeError, ValueError):
        return None
    return rows if rows > 0 else None

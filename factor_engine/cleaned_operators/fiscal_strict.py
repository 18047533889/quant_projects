# -*- coding: utf-8 -*-
"""Final strict PIT-safe fiscal-period operators.

The implementation is intentionally backend-symmetric.  Fiscal lags use exact
quarter ordinals, revisions are selected only from information visible at the
current row, and every semantic parameter is honoured by both Pandas and
Polars implementations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    Spec,
    finite_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    register_specs,
)

_REVISION_POLICIES = frozenset({"latest_available", "first_available"})


# ---------------------------------------------------------------------------
# FinancialFlowSemantics (round-11 findings #51/#52/#53)
#
# Income / cash-flow fields arrive at PIT panels in one of these grains:
# ``SinglePeriodFlow`` (one fiscal period), ``CumulativeYTDFlow`` (fiscal-YTD
# cumulative), ``TTMFlow`` (trailing twelve months), ``AnnualFlow`` (annual
# report) or ``Stock`` (balance-sheet point-in-time).  Growth / persistence /
# volatility math is only defined on the matching grain, and subtracting two
# flows of different grain silently mixes non-adjacent periods.  These helpers
# live in this leaf module so both ``fundamental/transforms_v2`` and
# ``fundamental/quality_v2`` can gate on them without a circular import.
# ---------------------------------------------------------------------------

class FinancialFlowSemantics:
    """Reporting-flow grain contract for income / cash-flow operators."""

    SINGLE_PERIOD = "SinglePeriodFlow"
    CUMULATIVE_YTD = "CumulativeYTDFlow"
    TTM = "TTMFlow"
    ANNUAL = "AnnualFlow"
    STOCK = "Stock"


FLOW_TYPE_SET = frozenset({
    FinancialFlowSemantics.SINGLE_PERIOD,
    FinancialFlowSemantics.CUMULATIVE_YTD,
    FinancialFlowSemantics.TTM,
    FinancialFlowSemantics.ANNUAL,
    FinancialFlowSemantics.STOCK,
})

GROWTH_FORBIDDEN_FLOW_TYPES = frozenset({FinancialFlowSemantics.CUMULATIVE_YTD})


def flow_types(flow_type: Any, n: int) -> tuple[str | None, ...]:
    """Normalise a ``flow_type`` declaration to one grain per flow input.

    ``None`` (undeclared) keeps legacy behaviour (no gate).  A string applies to
    every flow input; a tuple applies element-wise and must match the count.
    """
    if flow_type is None:
        return (None,) * n
    if isinstance(flow_type, str):
        types = (flow_type,) * n
    else:
        types = tuple(flow_type)
        if len(types) != n:
            raise ValueError(
                f"flow_type must be a string or a {n}-tuple matching the {n} "
                f"flow inputs; got {flow_type!r}"
            )
    for t in types:
        if t not in FLOW_TYPE_SET:
            raise ValueError(
                f"unknown flow_type {t!r}; expected one of {sorted(FLOW_TYPE_SET)}"
            )
    return types


def reject_ytd_growth(operator: str, flow_type: Any) -> None:
    """Reject growth/persistence on a ``CumulativeYTDFlow`` input (finding #52)."""
    if flow_type is None:
        return
    types = flow_types(flow_type, 1)
    if types[0] in GROWTH_FORBIDDEN_FLOW_TYPES:
        raise ValueError(
            f"{operator}: growth/period-change over a CumulativeYTDFlow input is "
            "NOT a period growth rate (Q2 YTD / Q1 YTD != quarterly growth). "
            "Convert with fin_quarter_from_cumulative (or fin_ttm_cumulative) "
            "before applying growth."
        )


def require_same_flow_grain(operator: str, flow_type: Any, n: int) -> None:
    """Reject combining flow inputs of different grain in one expression (#53)."""
    if flow_type is None:
        return
    types = flow_types(flow_type, n)
    base = types[0]
    for t in types[1:]:
        if t != base:
            raise ValueError(
                f"{operator}: mixing incompatible flow grains {types!r} — "
                "accrual/cash-gap subtraction requires the same period grain "
                "(e.g. a quarter accrual minus YTD depreciation is invalid)."
            )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if number < 1 or float(value) != float(number):
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if number < 0 or float(value) != float(number):
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _revision_policy(value: Any) -> str:
    policy = str(value).lower()
    if policy not in _REVISION_POLICIES:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return policy


def _strict_align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("fiscal operators require pandas DataFrame inputs")
    if not base.index.is_unique or not base.columns.is_unique:
        raise ValueError("fiscal operator input axes must be unique")
    for position, frame in enumerate(frames[1:], start=1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"fiscal input {position} must be a pandas DataFrame")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"fiscal input {position} axes must be unique")
        if not frame.index.equals(base.index):
            raise ValueError(f"fiscal input {position} index does not match the primary input")
        if not frame.columns.equals(base.columns):
            raise ValueError(f"fiscal input {position} columns do not match the primary input")
    return tuple(frames)


@dataclass(frozen=True)
class FiscalPeriod:
    """A fiscal reporting period carrying real (possibly non-calendar) metadata.

    ``fiscal_year`` is the fiscal year in which the period reports (e.g. 2025 for
    the fiscal year ending 2025-06-30), ``fiscal_quarter`` is 1-4 within that
    fiscal year, ``fiscal_year_end`` is the calendar month (1-12) in which the
    fiscal year ends, and ``timeframe`` is the report scope (quarterly / annual /
    trailing_twelve_months / ytd).

    ``period_ordinal`` accepts a ``FiscalPeriod`` and derives its monotonic
    ordinal from the *fiscal* year/quarter (not the calendar quarter), so US
    companies with non-December fiscal year-ends convert correctly (round-7 P0).
    """

    fiscal_year: int
    fiscal_quarter: int
    fiscal_year_end: int | None = None
    timeframe: str = "quarterly"

    def __post_init__(self) -> None:
        if not (1 <= int(self.fiscal_quarter) <= 4):
            raise ValueError("fiscal_quarter must be in 1..4")
        if self.fiscal_year_end is not None and not (1 <= int(self.fiscal_year_end) <= 12):
            raise ValueError("fiscal_year_end must be a calendar month in 1..12")

    @property
    def ordinal(self) -> int:
        return int(self.fiscal_year) * 4 + int(self.fiscal_quarter) - 1

    @staticmethod
    def from_timestamp(
        value: Any, fiscal_year_end: int | None
    ) -> "FiscalPeriod | None":
        """Fiscal period of a timestamp under a given fiscal year-end month.

        ``fiscal_year_end=None`` fails closed (returns ``None``) — a non-calendar
        fiscal year can never be inferred from a bare timestamp (round-7 P0).
        """
        if fiscal_year_end is None:
            return None
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        m = int(fiscal_year_end)
        if not (1 <= m <= 12):
            raise ValueError("fiscal_year_end must be a calendar month in 1..12")
        month = ts.month
        # The fiscal year begins the month AFTER the year-end month.  A date whose
        # calendar month exceeds the year-end month belongs to the NEXT fiscal year
        # (e.g. 2024-07 under a 2025-06 year-end is FY2025); otherwise it belongs
        # to the current calendar year's fiscal year.
        fiscal_year = ts.year + (1 if month > m else 0)
        fiscal_quarter = ((month - (m + 1)) % 12) // 3 + 1
        return FiscalPeriod(
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            fiscal_year_end=m,
        )


@dataclass(frozen=True)
class FiscalPeriodKey:
    """Fiscal identity ``(fiscal_year, fiscal_slot)`` for same-slot matching.

    ``fiscal_slot`` is the 1-based position of the report within its fiscal
    year (quarter 1-4, half-year 1-2, annual 1, ISO week 1-53 for a 53-week
    fiscal year).  Round-11 #180: a year-over-year lag is the SAME slot of the
    PRIOR fiscal year, not a fixed ``periods_per_year`` ordinal lag — a 53-week
    or non-calendar fiscal year has a different number of periods between the
    same slots.
    """

    fiscal_year: int
    fiscal_slot: int

    @classmethod
    def from_ordinal(cls, ordinal: int) -> "FiscalPeriodKey":
        # The engine's ordinal encoding is quarterly-scaled (year*4+quarter-1),
        # so the quarter-slot is ``ordinal % 4 + 1``.  Used only as a fallback
        # for period ids that expose an ordinal but no explicit slot label.
        return cls(int(ordinal) // 4, int(ordinal) % 4 + 1)


_FISCAL_PERIOD_KEY_RE = re.compile(
    r"^(?P<year>\d{4})\s*(?:Q(?P<q>[1-4])|H(?P<h>[1-2])|S(?P<s>[1-2])"
    r"|W(?P<w>[0-5]?\d)|(?:FY|A|ANNUAL|YTD))$"
)


def fiscal_period_key(value: Any) -> FiscalPeriodKey | None:
    """Parse a report-period identifier into ``FiscalPeriodKey`` or ``None``.

    Supports quarterly (``2024Q1`` / ``2024Q 1`` / compact ``20241``),
    semiannual (``2024H1`` / ``2024S1``), annual (``2024FY`` / ``2024A`` /
    ``2024``), 53-week fiscal years (``2024W53``) and ``FiscalPeriod`` objects.
    A bare timestamp is resolved with the *calendar* quarter; a non-calendar
    fiscal year cannot be inferred from a bare date and fails closed (``None``)
    unless the caller supplies a ``FiscalPeriod``.
    """
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, FiscalPeriod):
        return FiscalPeriodKey(value.fiscal_year, value.fiscal_quarter)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return FiscalPeriodKey(ts.year, (ts.month - 1) // 3 + 1)
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if 10_000 <= number <= 99_999:
            year, quarter = divmod(number, 10)
            if year >= 1000 and 1 <= quarter <= 4:
                return FiscalPeriodKey(year, quarter)
        return None
    text = str(value).strip().upper()
    match = re.fullmatch(r"(\d{4})\D*Q?([1-4])", text)
    if match:
        return FiscalPeriodKey(int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"(\d{4})\D*[HS]([1-2])", text)
    if match:
        return FiscalPeriodKey(int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"(\d{4})\D*W(\d{1,2})", text)
    if match:
        week = int(match.group(2))
        if 1 <= week <= 53:
            return FiscalPeriodKey(int(match.group(1)), week)
    match = re.fullmatch(r"(\d{4})(?:FY|A|ANNUAL|YTD)?", text)
    if match:
        return FiscalPeriodKey(int(match.group(1)), 1)
    return None


def period_ordinal(value: Any) -> int | None:
    """Parse an explicit fiscal-quarter identifier into a monotonic ordinal.

    Accepted explicit formats (round-7 P1): ``FiscalPeriod``, ``YYYYQn`` /
    ``YYYYQ n`` strings, ``YYYYn`` compact int (5 digits), ``YYYYMMDD`` (8-digit
    int or ISO string), and ``Timestamp``/``datetime``.  Any other scalar — in
    particular an arbitrary integer that is not one of the explicit encodings —
    is rejected (returns ``None``) instead of being silently accepted as a raw
    ordinal.  A bare Timestamp is interpreted with the *calendar* quarter; use a
    ``FiscalPeriod`` (or ``fiscal_ordinal(ts, fiscal_year_end)``) for companies
    with non-December fiscal year-ends (round-7 P0).
    """
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, FiscalPeriod):
        return value.ordinal
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else int(ts.year * 4 + (ts.month - 1) // 3)
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        number = int(value)
        # An 8-digit int is a YYYYMMDD date (e.g. 20240101), not a YYYYQ fiscal
        # encoding: do not silently map it through the year*10+quarter branch
        # (review §5.6).  Parse it as a calendar date quarter instead.
        if 10_000_000 <= number <= 99_999_999:
            try:
                ts = pd.Timestamp(f"{number // 10_000:04d}-{(number // 100) % 100:02d}-{number % 100:02d}")
            except Exception:
                ts = None
            if ts is not None and not pd.isna(ts):
                return int(ts.year * 4 + (ts.month - 1) // 3)
        # The only accepted integer encoding is the compact 5-digit ``YYYYn``
        # fiscal-quarter form (e.g. 20251 -> FY2025 Q1).  A 6+-digit arbitrary
        # integer (e.g. 123456) is NOT a valid fiscal ordinal and must fail closed
        # to None rather than pass through raw (round-7 P1).
        if 10_000 <= number <= 99_999:
            year, quarter = divmod(number, 10)
            if year >= 1000 and 1 <= quarter <= 4:
                return year * 4 + quarter - 1
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return period_ordinal(int(value))
    text = str(value).strip().upper()
    match = re.fullmatch(r"(\d{4})\D*Q?([1-4])", text)
    if match:
        return int(match.group(1)) * 4 + int(match.group(2)) - 1
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    return None if pd.isna(ts) else int(ts.year * 4 + (ts.month - 1) // 3)


def fiscal_ordinal(value: Any, fiscal_year_end: int | None = None) -> int | None:
    """Fiscal ordinal under a real year-end; ``None`` (fail closed) when unknown.

    Unlike ``period_ordinal`` (which uses the calendar quarter for bare
    timestamps), this resolves a timestamp with the company's actual fiscal
    year-end month.  When ``fiscal_year_end`` is unknown it returns ``None`` so
    YTD/quarter/TTM conversions fail closed instead of assuming the calendar year
    (round-7 P0).
    """
    if isinstance(value, FiscalPeriod):
        return value.ordinal
    fp = FiscalPeriod.from_timestamp(value, fiscal_year_end)
    return fp.ordinal if fp is not None else None


def ordinal_frame(period_id: pd.DataFrame) -> pd.DataFrame:
    return period_id.apply(lambda col: col.map(period_ordinal)).astype("Float64")


def _lag_array(values: np.ndarray, period_values: np.ndarray, lag: int, policy: str) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        visible: dict[int, float] = {}
        for row in range(values.shape[0]):
            ordinal = period_ordinal(period_values[row, col])
            if ordinal is None:
                continue
            value = values[row, col]
            if np.isfinite(value) and (policy == "latest_available" or ordinal not in visible):
                visible[int(ordinal)] = float(value)
            out[row, col] = visible.get(int(ordinal) - lag, np.nan)
    return out


def pd_period_lag(x, period_id, periods=1, revision_policy="latest_available", **_):
    x, period_id = _strict_align(x, period_id)
    lag = _nonnegative_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    return frame_pd(
        x,
        _lag_array(
            x.to_numpy(dtype=float),
            period_id.to_numpy(dtype=object),
            lag,
            policy,
        ),
    )


def _ordinal_lag(period_id, lag, policy):
    ordinal = ordinal_frame(period_id).astype(float)
    return pd_period_lag(ordinal, period_id, lag, policy)


def _period_validity(x, period_id, lag, policy, require_consecutive):
    previous = pd_period_lag(x, period_id, lag, policy)
    valid = finite_pd(x) & finite_pd(previous)
    if bool(require_consecutive):
        current_ordinal = ordinal_frame(period_id).astype(float)
        prior_ordinal = _ordinal_lag(period_id, lag, policy)
        valid &= current_ordinal.sub(prior_ordinal).eq(float(lag))
    return previous, valid


def pd_period_change(
    x,
    period_id,
    periods=1,
    mode="absolute",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    lag = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    previous, valid = _period_validity(x, period_id, lag, policy, require_consecutive)
    mode = str(mode).lower()
    if mode == "absolute":
        result = x - previous
    elif mode == "ratio":
        valid &= previous.abs().gt(EPS)
        result = np.where(previous.where(valid) - 1.0 != 0, x / previous.where(valid) - 1.0, np.nan)
    elif mode == "log":
        valid &= previous.abs().gt(EPS)
        ratio = np.where(previous.where(valid) != 0, x / previous.where(valid), np.nan)
        valid &= ratio.gt(0)
        result = np.log(ratio.where(valid))
    else:
        raise ValueError("mode must be 'absolute', 'ratio', or 'log'")
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def _pieces(x, period_id, count, policy):
    ordinal = ordinal_frame(period_id).astype(float)
    pieces = [x]
    strict = finite_pd(x)
    available = finite_pd(x)
    for lag in range(1, count):
        piece = pd_period_lag(x, period_id, lag, policy)
        prior_ordinal = _ordinal_lag(period_id, lag, policy)
        pieces.append(piece)
        available &= finite_pd(piece)
        strict &= finite_pd(piece) & ordinal.sub(prior_ordinal).eq(float(lag))
    return pieces, strict, available


def pd_period_average(
    x,
    period_id,
    periods=2,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    count = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    pieces, strict, available = _pieces(x, period_id, count, policy)
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return np.where(float(count)).where(strict if bool(require_consecutive) else available) != 0, (total / float(count)).where(strict if bool(require_consecutive) else available), np.nan)


def pd_period_cagr(
    x,
    period_id,
    periods=12,
    periods_per_year=4,
    sign_policy="strict",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    lag = _positive_int(periods, "periods")
    ppy = _positive_int(periods_per_year, "periods_per_year")
    policy = _revision_policy(revision_policy)
    previous, valid = _period_validity(x, period_id, lag, policy, require_consecutive)
    sign_policy = str(sign_policy).lower()
    if sign_policy == "strict":
        valid &= x.gt(0) & previous.gt(0)
        ratio = np.where(previous.where(valid) != 0, x / previous.where(valid), np.nan)
    elif sign_policy == "absolute":
        valid &= previous.abs().gt(EPS)
        ratio = np.where(previous.abs().where(valid) != 0, x.abs() / previous.abs().where(valid), np.nan)
    else:
        raise ValueError("sign_policy must be 'strict' or 'absolute'")
    result = np.where(float(lag)) - 1.0 != 0, ratio.pow(float(ppy) / float(lag)) - 1.0, np.nan)
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def pd_quarter_from_cumulative(
    x,
    period_id,
    fiscal_quarter=None,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    policy = _revision_policy(revision_policy)
    ordinal = ordinal_frame(period_id).astype(float)
    previous = pd_period_lag(x, period_id, 1, policy)
    previous_ordinal = _ordinal_lag(period_id, 1, policy)
    if fiscal_quarter is None:
        # Round-7 P0 fail-closed: without explicit fiscal metadata we cannot know
        # which visible report is the first period of its fiscal year.  Assuming
        # the calendar quarter (ordinal % 4) silently mis-converts companies with
        # non-December fiscal year-ends.  Emit an all-NaN quarter panel so the
        # cumulative->quarter conversion fails closed; callers with real fiscal
        # metadata must pass a fiscal_quarter panel (e.g. derived from a
        # FiscalPeriod period_id / fiscal_year_end).
        quarter = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    else:
        _, _, fiscal_quarter = _strict_align(x, period_id, fiscal_quarter)
        quarter = fiscal_quarter.astype(float)
    consecutive = ordinal.sub(previous_ordinal).eq(1.0)
    result = x.where(quarter.eq(1.0), x - previous.where(consecutive))
    valid = finite_pd(x) & finite_pd(quarter) & (quarter.eq(1.0) | consecutive)
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def pd_ttm_from_quarterly(
    x,
    period_id,
    periods=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    count = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    pieces, strict, available = _pieces(x, period_id, count, policy)
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return total.where(strict if bool(require_consecutive) else available).replace([np.inf, -np.inf], np.nan)


def pd_ttm_from_cumulative(
    x,
    period_id,
    fiscal_quarter=None,
    revision_policy="latest_available",
    **_,
):
    policy = _revision_policy(revision_policy)
    quarterly = pd_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter,
        revision_policy=policy,
    )
    return pd_ttm_from_quarterly(
        quarterly,
        period_id,
        periods=4,
        require_consecutive=True,
        revision_policy=policy,
    )


def pd_yoy_by_period(
    x,
    period_id,
    periods=4,
    denominator="signed",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    lag = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    previous, valid = _period_validity(x, period_id, lag, policy, require_consecutive)
    mode = str(denominator).lower()
    if mode == "signed":
        denom = previous
    elif mode == "absolute":
        denom = previous.abs()
    else:
        raise ValueError("denominator must be 'signed' or 'absolute'")
    valid &= denom.abs().gt(EPS)
    return np.where(denom.where(valid)).where(valid).replace([np.inf, -np.inf], np.nan) != 0, ((x - previous) / denom.where(valid)).where(valid).replace([np.inf, -np.inf], np.nan), np.nan)


if pl is not None:

    def _pl_utf8_of(frame, source_col: str, target_col: str | None = None):
        """Build a Utf8 expression for ``target_col`` (defaults to ``source_col``).

        Pandas object panels (report-period strings interleaved with leading
        NaN) reach Polars as ``pl.Object`` dtype, which cannot be ``.cast`` to
        Utf8.  Map Object columns element-wise; fast-cast String columns.
        """
        target_col = target_col or source_col
        col = pl.col(target_col)
        if frame.schema.get(source_col) == pl.Object:
            return col.map_elements(
                lambda v: None if v is None else str(v),
                return_dtype=pl.Utf8,
            )
        return col.cast(pl.Utf8, strict=False)

    def _pl_numeric_col(frame, source_col: str, target_col: str | None = None):
        """Expression usable for integer-encoded period ids.

        Object columns hold strings/dates (never encoded ints), so the numeric
        fallback is a null literal for them to avoid a hard ``cast`` failure.
        """
        target_col = target_col or source_col
        if frame.schema.get(source_col) == pl.Object:
            return pl.lit(None)
        return pl.col(target_col)

    def _pl_float_of(frame, source_col: str, target_col: str | None = None):
        """Float64 expression; Object columns (e.g. a quarter panel) map element-wise."""
        target_col = target_col or source_col
        col = pl.col(target_col)
        if frame.schema.get(source_col) == pl.Object:
            return col.map_elements(
                lambda v: None if v is None else float(v),
                return_dtype=pl.Float64,
            )
        return col.cast(pl.Float64, strict=False)

    def _pl_period_ordinal_expr(column: str, raw_utf8=None, raw_numeric=None):
        """Return the exact Polars equivalent of :func:`period_ordinal`.

        Polars' format-inferred ``str.to_date`` can fail the whole expression on
        mixed fiscal identifiers such as ``2023Q1`` and ISO dates.  Parse each
        supported representation with anchored expressions instead, and keep
        unknown values null rather than guessing.
        """
        raw = pl.col(column)
        text = (
            raw_utf8
            if raw_utf8 is not None
            else raw.cast(pl.Utf8, strict=False)
        ).str.strip_chars().str.to_uppercase()

        quarter_year = text.str.extract(
            r"^(\d{4})(?:\D*Q?)([1-4])$", 1
        ).cast(pl.Int64, strict=False)
        quarter_number = text.str.extract(
            r"^(\d{4})(?:\D*Q?)([1-4])$", 2
        ).cast(pl.Int64, strict=False)

        iso_year = text.str.extract(
            r"^(\d{4})[-/](\d{1,2})[-/]\d{1,2}(?:[ T].*)?$", 1
        ).cast(pl.Int64, strict=False)
        iso_month = text.str.extract(
            r"^(\d{4})[-/](\d{1,2})[-/]\d{1,2}(?:[ T].*)?$", 2
        ).cast(pl.Int64, strict=False)
        compact_year = text.str.extract(
            r"^(\d{4})(\d{2})\d{2}(?:[ T].*)?$", 1
        ).cast(pl.Int64, strict=False)
        compact_month = text.str.extract(
            r"^(\d{4})(\d{2})\d{2}(?:[ T].*)?$", 2
        ).cast(pl.Int64, strict=False)
        date_year = pl.coalesce([iso_year, compact_year])
        date_month = pl.coalesce([iso_month, compact_month])
        valid_date = date_year.is_not_null() & date_month.is_between(1, 12)

        raw_num = raw_numeric if raw_numeric is not None else raw
        numeric_int = raw_num.cast(pl.Int64, strict=False)
        numeric_year = numeric_int // 10
        numeric_quarter = numeric_int % 10
        # Round-7 P1 parity: only the compact 5-digit ``YYYYn`` encoding is a
        # valid integer fiscal period.  A 6+-digit arbitrary integer (e.g.
        # 123456) must NOT pass through as a raw ordinal.
        encoded_quarter = (
            numeric_int.is_not_null()
            & numeric_int.is_between(10_000, 99_999)
            & (numeric_year >= 1000)
            & numeric_quarter.is_between(1, 4)
        )

        return (
            pl.when(quarter_year.is_not_null() & quarter_number.is_not_null())
            .then(quarter_year * 4 + quarter_number - 1)
            .when(valid_date)
            .then(date_year * 4 + ((date_month - 1) // 3))
            .when(encoded_quarter)
            .then(numeric_year * 4 + numeric_quarter - 1)
            .otherwise(pl.lit(None))
            .cast(pl.Float64)
        )


    def _pl_ordinal_frame(period_id):
        return period_id.with_columns(
            [
                _pl_period_ordinal_expr(
                    col,
                    _pl_utf8_of(period_id, col),
                    _pl_numeric_col(period_id, col),
                ).alias(col)
                for col in pl_cols(period_id)
            ]
        )


    def pl_period_lag(
        x,
        period_id,
        periods=1,
        revision_policy="latest_available",
        **_,
    ):
        lag = _nonnegative_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        replacements = {}
        for col in [c for c in pl_cols(x) if c in period_id.columns]:
            temp = pl.DataFrame(
                {
                    "_row": pl.int_range(0, x.height, eager=True),
                    "_x": x[col].cast(pl.Float64, strict=False),
                    "_pid": period_id[col],
                }
            ).with_columns(
                _pl_period_ordinal_expr(
                    "_pid",
                    _pl_utf8_of(period_id, col, "_pid"),
                    _pl_numeric_col(period_id, col, "_pid"),
                ).alias("_ordinal")
            )
            targets = temp.select(
                "_row",
                "_pid",
                "_ordinal",
                (pl.col("_ordinal") - lag).alias("_target"),
            )
            history = temp.select(
                pl.col("_row").alias("_hrow"),
                pl.col("_ordinal").alias("_hordinal"),
                pl.col("_x").alias("_hx"),
            )
            candidates = (
                targets.select("_row", "_target")
                .join(history, left_on="_target", right_on="_hordinal", how="inner")
                .filter(
                    (pl.col("_hrow") <= pl.col("_row"))
                    & pl.col("_hx").is_not_null()
                    & pl.col("_hx").is_finite()
                )
                .sort(["_row", "_hrow"])
            )
            aggregate = (
                pl.col("_hx").first()
                if policy == "first_available"
                else pl.col("_hx").last()
            )
            selected = candidates.group_by("_row", maintain_order=True).agg(
                aggregate.alias("_out")
            )
            result = (
                targets.select("_row", "_pid", "_ordinal")
                .join(selected, on="_row", how="left", maintain_order="left")
                .select(
                    pl.when(pl.col("_ordinal").is_null())
                    .then(None)
                    .otherwise(pl.col("_out"))
                    .alias(col)
                )
            )[col]
            replacements[col] = result
        return pl_base_with(x, replacements)


    def _pl_valid(expr):
        return expr.is_not_null() & expr.is_finite()


    def _pl_binary_frames(x, y, fn):
        replacements = {}
        for col in [c for c in pl_cols(x) if c in y.columns]:
            temp = pl.DataFrame(
                {
                    "a": x[col].cast(pl.Float64, strict=False),
                    "b": y[col].cast(pl.Float64, strict=False),
                }
            )
            replacements[col] = temp.select(fn(pl.col("a"), pl.col("b")).alias(col))[col]
        return pl_base_with(x, replacements)


    def _pl_period_validity(x, period_id, lag, policy, require_consecutive):
        previous = pl_period_lag(x, period_id, lag, policy)
        if not bool(require_consecutive):
            return previous, None
        current_ordinal = _pl_ordinal_frame(period_id)
        prior_ordinal = pl_period_lag(current_ordinal, period_id, lag, policy)
        return previous, (current_ordinal, prior_ordinal)


    def pl_period_change(
        x,
        period_id,
        periods=1,
        mode="absolute",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        previous, ordinal_pair = _pl_period_validity(
            x, period_id, lag, policy, require_consecutive
        )
        mode = str(mode).lower()
        replacements = {}
        for col in [c for c in pl_cols(x) if c in previous.columns]:
            data = {
                "a": x[col].cast(pl.Float64, strict=False),
                "b": previous[col].cast(pl.Float64, strict=False),
            }
            if ordinal_pair is not None:
                data["o"] = ordinal_pair[0][col].cast(pl.Float64, strict=False)
                data["po"] = ordinal_pair[1][col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(data)
            a, b = pl.col("a"), pl.col("b")
            valid = _pl_valid(a) & _pl_valid(b)
            if ordinal_pair is not None:
                valid &= (pl.col("o") - pl.col("po") == float(lag))
            if mode == "absolute":
                expr = a - b
            elif mode == "ratio":
                valid &= b.abs() > EPS
                expr = np.where(b - 1.0 != 0, a / b - 1.0, np.nan)
            elif mode == "log":
                valid &= (b.abs() > EPS) & (a / b > 0)
                expr = np.where(b).log() != 0, (a / b).log(), np.nan)
            else:
                raise ValueError("mode must be 'absolute', 'ratio', or 'log'")
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def _pl_pieces(x, period_id, count, policy):
        ordinals = _pl_ordinal_frame(period_id)
        pieces = [x]
        prior_ordinals = [ordinals]
        for lag in range(1, count):
            pieces.append(pl_period_lag(x, period_id, lag, policy))
            prior_ordinals.append(pl_period_lag(ordinals, period_id, lag, policy))
        return pieces, prior_ordinals


    def pl_period_average(
        x,
        period_id,
        periods=2,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        count = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        pieces, ordinal_pieces = _pl_pieces(x, period_id, count, policy)
        replacements = {}
        for col in pl_cols(x):
            data = {f"v{i}": piece[col].cast(pl.Float64, strict=False) for i, piece in enumerate(pieces)}
            if bool(require_consecutive):
                data.update(
                    {f"o{i}": ordinal_pieces[i][col].cast(pl.Float64, strict=False) for i in range(count)}
                )
            temp = pl.DataFrame(data)
            valid = pl.all_horizontal(*[_pl_valid(pl.col(f"v{i}")) for i in range(count)])
            if bool(require_consecutive):
                valid &= pl.all_horizontal(
                    *[(pl.col("o0") - pl.col(f"o{i}") == float(i)) for i in range(1, count)]
                )
            total = pl.sum_horizontal(*[pl.col(f"v{i}") for i in range(count)])
            replacements[col] = temp.select(
                pl.when(valid).then(total / float(count)).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_period_cagr(
        x,
        period_id,
        periods=12,
        periods_per_year=4,
        sign_policy="strict",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = _positive_int(periods, "periods")
        ppy = _positive_int(periods_per_year, "periods_per_year")
        policy = _revision_policy(revision_policy)
        previous, ordinal_pair = _pl_period_validity(
            x, period_id, lag, policy, require_consecutive
        )
        sign_policy = str(sign_policy).lower()
        replacements = {}
        for col in [c for c in pl_cols(x) if c in previous.columns]:
            data = {
                "a": x[col].cast(pl.Float64, strict=False),
                "b": previous[col].cast(pl.Float64, strict=False),
            }
            if ordinal_pair is not None:
                data["o"] = ordinal_pair[0][col].cast(pl.Float64, strict=False)
                data["po"] = ordinal_pair[1][col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(data)
            a, b = pl.col("a"), pl.col("b")
            valid = _pl_valid(a) & _pl_valid(b)
            if ordinal_pair is not None:
                valid &= (pl.col("o") - pl.col("po") == float(lag))
            if sign_policy == "strict":
                valid &= (a > 0) & (b > 0)
                ratio = a / b if b != 0 else np.nan
            elif sign_policy == "absolute":
                valid &= b.abs() > EPS
                ratio = np.where(b.abs() != 0, a.abs() / b.abs(), np.nan)
            else:
                raise ValueError("sign_policy must be 'strict' or 'absolute'")
            expr = np.where(float(lag)) - 1.0 != 0, ratio.pow(float(ppy) / float(lag)) - 1.0, np.nan)
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=None,
        revision_policy="latest_available",
        **_,
    ):
        policy = _revision_policy(revision_policy)
        ordinal = _pl_ordinal_frame(period_id)
        previous = pl_period_lag(x, period_id, 1, policy)
        previous_ordinal = pl_period_lag(ordinal, period_id, 1, policy)
        quarter_frame = (
            # Round-7 P0 fail-closed: mirror the pandas side — a missing
            # fiscal_quarter panel must NOT fall back to the calendar quarter
            # (non-December fiscal year-ends would be silently mangled).
            ordinal.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(c) for c in pl_cols(ordinal)]
            )
            if fiscal_quarter is None
            else fiscal_quarter.with_columns(
                [_pl_float_of(fiscal_quarter, c).alias(c) for c in pl_cols(fiscal_quarter)]
            )
        )
        replacements = {}
        for col in [c for c in pl_cols(x) if c in quarter_frame.columns]:
            temp = pl.DataFrame(
                {
                    "x": x[col].cast(pl.Float64, strict=False),
                    "p": previous[col].cast(pl.Float64, strict=False),
                    "o": ordinal[col].cast(pl.Float64, strict=False),
                    "po": previous_ordinal[col].cast(pl.Float64, strict=False),
                    "q": quarter_frame[col].cast(pl.Float64, strict=False),
                }
            )
            consecutive = pl.col("o") - pl.col("po") == 1.0
            valid = _pl_valid(pl.col("x")) & _pl_valid(pl.col("q")) & (
                (pl.col("q") == 1.0) | consecutive
            )
            expr = (
                pl.when(pl.col("q") == 1.0)
                .then(pl.col("x"))
                .otherwise(pl.col("x") - pl.col("p"))
            )
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_ttm_from_quarterly(
        x,
        period_id,
        periods=4,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        count = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        pieces, ordinal_pieces = _pl_pieces(x, period_id, count, policy)
        replacements = {}
        for col in pl_cols(x):
            data = {f"v{i}": piece[col].cast(pl.Float64, strict=False) for i, piece in enumerate(pieces)}
            if bool(require_consecutive):
                data.update(
                    {f"o{i}": ordinal_pieces[i][col].cast(pl.Float64, strict=False) for i in range(count)}
                )
            temp = pl.DataFrame(data)
            valid = pl.all_horizontal(*[_pl_valid(pl.col(f"v{i}")) for i in range(count)])
            if bool(require_consecutive):
                valid &= pl.all_horizontal(
                    *[(pl.col("o0") - pl.col(f"o{i}") == float(i)) for i in range(1, count)]
                )
            total = pl.sum_horizontal(*[pl.col(f"v{i}") for i in range(count)])
            replacements[col] = temp.select(
                pl.when(valid).then(total).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_ttm_from_cumulative(
        x,
        period_id,
        fiscal_quarter=None,
        revision_policy="latest_available",
        **_,
    ):
        policy = _revision_policy(revision_policy)
        quarterly = pl_quarter_from_cumulative(
            x,
            period_id,
            fiscal_quarter,
            revision_policy=policy,
        )
        return pl_ttm_from_quarterly(
            quarterly,
            period_id,
            periods=4,
            require_consecutive=True,
            revision_policy=policy,
        )


    def pl_yoy_by_period(
        x,
        period_id,
        periods=4,
        denominator="signed",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        previous, ordinal_pair = _pl_period_validity(
            x, period_id, lag, policy, require_consecutive
        )
        denominator = str(denominator).lower()
        replacements = {}
        for col in [c for c in pl_cols(x) if c in previous.columns]:
            data = {
                "a": x[col].cast(pl.Float64, strict=False),
                "b": previous[col].cast(pl.Float64, strict=False),
            }
            if ordinal_pair is not None:
                data["o"] = ordinal_pair[0][col].cast(pl.Float64, strict=False)
                data["po"] = ordinal_pair[1][col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(data)
            a, b = pl.col("a"), pl.col("b")
            valid = _pl_valid(a) & _pl_valid(b)
            if ordinal_pair is not None:
                valid &= (pl.col("o") - pl.col("po") == float(lag))
            if denominator == "signed":
                denom = b
            elif denominator == "absolute":
                denom = b.abs()
            else:
                raise ValueError("denominator must be 'signed' or 'absolute'")
            valid &= denom.abs() > EPS
            expr = (a - b) / denom if denom != 0 else np.nan
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)

else:  # pragma: no cover - optional dependency
    pl_period_lag = None
    pl_period_change = None
    pl_period_average = None
    pl_period_cagr = None
    pl_quarter_from_cumulative = None
    pl_ttm_from_quarterly = None
    pl_ttm_from_cumulative = None
    pl_yoy_by_period = None


def register() -> None:
    register_specs(
        {
            "period_lag": Spec(
                "fundamental_period",
                ["x", "period_id", "periods", "revision_policy"],
                "exact fiscal ordinal lag with visible revision policy",
                pd_period_lag,
                pl_period_lag,
            ),
            "period_change": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "mode",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period change",
                pd_period_change,
                pl_period_change,
            ),
            "period_average": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period average",
                pd_period_average,
                pl_period_average,
            ),
            "period_cagr": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "periods_per_year",
                    "sign_policy",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period compound growth",
                pd_period_cagr,
                pl_period_cagr,
            ),
            "quarter_from_cumulative": Spec(
                "fundamental_period",
                ["x", "period_id", "fiscal_quarter", "revision_policy"],
                "strict cumulative-to-quarter conversion",
                pd_quarter_from_cumulative,
                pl_quarter_from_cumulative,
            ),
            "ttm_from_quarterly": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict consecutive-period TTM",
                pd_ttm_from_quarterly,
                pl_ttm_from_quarterly,
            ),
            "ttm_from_cumulative": Spec(
                "fundamental_period",
                ["x", "period_id", "fiscal_quarter", "revision_policy"],
                "strict cumulative-to-TTM",
                pd_ttm_from_cumulative,
                pl_ttm_from_cumulative,
            ),
            "yoy_by_period": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "denominator",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period growth",
                pd_yoy_by_period,
                pl_yoy_by_period,
            ),
        }
    )


register()

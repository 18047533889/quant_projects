"""Calendar-aware and rolling compounded-return risk metrics.

Calendar metrics consume the durable Data Access ``CalendarSnapshot`` as the
sole session authority.  They never infer missing sessions from the observed
return rows and never compress unknown observations out of a period.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from quant_evaluator.contracts.axis_refs import FactorAxisRef
from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact

__all__ = [
    "compute_worst_calendar_month",
    "compute_worst_calendar_quarter",
    "compute_worst_calendar_year",
    "compute_worst_rolling_return",
]


def _validated_returns(returns: Any, factor_ids: Sequence[str] | None = None) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("returns must have shape (T, F) or (T,)")
    if factor_ids is not None and len(factor_ids) != values.shape[1]:
        raise ValueError("factor_ids length must equal returns factor dimension")
    if np.any(np.isfinite(values) & (values < -1.0)):
        raise ValueError("finite capital returns must be >= -1")
    return values


def compute_worst_rolling_return(
    returns: Any,
    window: int = 21,
    min_periods: int = 1,
) -> np.ndarray:
    """Per-factor worst compounded return across matured full windows.

    ``min_periods`` is the required count of valid, fully matured windows; it
    never authorizes an expanding prefix. Every row in a candidate window must
    be finite for that factor. Invalid windows are skipped without compressing
    the time grid, and become eligible again only after the unknown row exits.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if (isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer))
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")
    values = _validated_returns(returns)
    rolling = np.full((max(0, values.shape[0] - int(window) + 1), values.shape[1]), np.nan)
    for offset, stop in enumerate(range(int(window), values.shape[0] + 1)):
        block = values[stop - int(window):stop]
        valid = np.all(np.isfinite(block), axis=0)
        rolling[offset, valid] = np.prod(1.0 + block[:, valid], axis=0) - 1.0
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for factor in range(values.shape[1]):
        candidates = rolling[np.isfinite(rolling[:, factor]), factor]
        if candidates.size >= int(min_periods):
            result[factor] = np.min(candidates)
    return result


def _local_session_dates(time_index: Sequence[Any], timezone_name: str) -> tuple[date, ...]:
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"calendar snapshot has unknown timezone {timezone_name!r}") from exc
    dates: list[date] = []
    for value in time_index:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("time_index entries must be timezone-aware datetime instants")
        dates.append(value.astimezone(timezone).date())
    if len(set(dates)) != len(dates):
        raise ValueError("time_index must contain at most one return row per local trading session")
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("time_index must be strictly increasing in calendar-local session dates")
    return tuple(dates)


def _period_key(day: date, frequency: str) -> tuple[int, ...]:
    if frequency == "month":
        return (day.year, day.month)
    if frequency == "quarter":
        return (day.year, (day.month - 1) // 3 + 1)
    return (day.year,)


def _period_id(key: tuple[int, ...], frequency: str) -> str:
    if frequency == "month":
        return f"{key[0]:04d}-{key[1]:02d}"
    if frequency == "quarter":
        return f"{key[0]:04d}-Q{key[1]}"
    return f"{key[0]:04d}"


def _calendar_metric(
    returns: Any,
    time_index: Sequence[Any],
    factor_ids: Sequence[str],
    calendar_snapshot: Any,
    *,
    frequency: str,
    partial_policy: str,
) -> ScalarMetricArtifact:
    # Lazy import keeps the standalone quant_evaluator wheel importable when
    # the optional data_access package is absent.
    from data_access.r30.calendar_snapshot import CalendarSnapshot
    if not isinstance(calendar_snapshot, CalendarSnapshot):
        raise TypeError("calendar_snapshot must be a Data Access CalendarSnapshot")
    if partial_policy not in {"exclude", "include"}:
        raise ValueError("partial_policy must be 'exclude' or 'include'")
    values = _validated_returns(returns, factor_ids)
    if len(time_index) != values.shape[0]:
        raise ValueError("time_index length must equal returns time dimension")
    local_dates = _local_session_dates(time_index, calendar_snapshot.timezone)
    try:
        expected_dates = tuple(date.fromisoformat(str(day)[:10]) for day in calendar_snapshot.trading_days)
    except ValueError as exc:
        raise ValueError("calendar_snapshot trading_days must contain ISO dates") from exc
    if not expected_dates or any(a >= b for a, b in zip(expected_dates, expected_dates[1:])):
        raise ValueError("calendar_snapshot trading_days must be non-empty and strictly increasing")
    expected_set = set(expected_dates)
    unknown_sessions = [day.isoformat() for day in local_dates if day not in expected_set]
    if unknown_sessions:
        raise ValueError(f"time_index contains sessions absent from calendar snapshot: {unknown_sessions}")

    observed_by_date = {day: row for day, row in zip(local_dates, values)}
    expected_by_period: dict[tuple[int, ...], list[date]] = {}
    for day in expected_dates:
        expected_by_period.setdefault(_period_key(day, frequency), []).append(day)

    period_rows: list[dict[str, Any]] = []
    period_values: list[np.ndarray] = []
    eligibility: list[np.ndarray] = []
    for key, sessions in expected_by_period.items():
        observed = [day for day in sessions if day in observed_by_date]
        bracketed = expected_dates[0] < sessions[0] and expected_dates[-1] > sessions[-1]
        sessions_complete = len(observed) == len(sessions)
        finite_counts = np.zeros(values.shape[1], dtype=np.int64)
        compounded = np.full(values.shape[1], np.nan, dtype=np.float64)
        if observed:
            block = np.stack([observed_by_date[day] for day in observed], axis=0)
            finite = np.all(np.isfinite(block), axis=0)
            finite_counts = np.sum(np.isfinite(block), axis=0)
            compounded[finite] = np.prod(1.0 + block[:, finite], axis=0) - 1.0
        complete_by_factor = sessions_complete & bracketed & (finite_counts == len(sessions))
        eligible = (finite_counts == len(observed)) & (len(observed) > 0)
        if partial_policy == "exclude":
            eligible &= complete_by_factor
        period_values.append(compounded)
        eligibility.append(eligible)
        period_rows.append({
            "period_id": _period_id(key, frequency),
            "expected_session_count": len(sessions),
            "observed_session_count": len(observed),
            "finite_return_counts": tuple(int(x) for x in finite_counts),
            "calendar_coverage_bracketed": bool(bracketed),
            "sessions_complete": bool(sessions_complete),
            "complete_by_factor": tuple(bool(x) for x in complete_by_factor),
            "partial": not bool(sessions_complete and bracketed),
            "included_by_factor": tuple(bool(x) for x in eligible),
            "compounded_returns": tuple(float(x) if np.isfinite(x) else None for x in compounded),
        })

    worst = np.full(values.shape[1], np.nan, dtype=np.float64)
    if period_values:
        matrix = np.stack(period_values)
        mask = np.stack(eligibility)
        for factor in range(values.shape[1]):
            candidates = matrix[mask[:, factor], factor]
            if candidates.size:
                worst[factor] = np.min(candidates)
    metric_id = f"worst_calendar_{frequency}"
    observation_counts = tuple(
        int(sum(bool(mask[factor]) for mask in eligibility))
        for factor in range(values.shape[1])
    )
    return ScalarMetricArtifact(
        metric_id=metric_id,
        domain="risk",
        values=worst,
        factor_axis=FactorAxisRef(factor_ids=tuple(factor_ids)),
        provenance={
            "calendar_snapshot_id": calendar_snapshot.snapshot_id,
            "calendar_market": calendar_snapshot.market,
            "calendar_timezone": calendar_snapshot.timezone,
            "calendar_source_version": calendar_snapshot.source_version,
            "partial_policy": partial_policy,
            "observation_counts": observation_counts,
            "period_rows": tuple(period_rows),
        },
    )


def compute_worst_calendar_month(returns, time_index, factor_ids, calendar_snapshot, partial_policy="exclude"):
    return _calendar_metric(returns, time_index, factor_ids, calendar_snapshot,
                            frequency="month", partial_policy=partial_policy)


def compute_worst_calendar_quarter(returns, time_index, factor_ids, calendar_snapshot, partial_policy="exclude"):
    return _calendar_metric(returns, time_index, factor_ids, calendar_snapshot,
                            frequency="quarter", partial_policy=partial_policy)


def compute_worst_calendar_year(returns, time_index, factor_ids, calendar_snapshot, partial_policy="exclude"):
    return _calendar_metric(returns, time_index, factor_ids, calendar_snapshot,
                            frequency="year", partial_policy=partial_policy)

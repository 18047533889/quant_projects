# -*- coding: utf-8 -*-
"""Point-in-time contracts for fundamental event data before factor calculation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

VisiblePeriodSelector = Literal["latest_visible_period", "annual_only", "quarterly_only"]


@dataclass(frozen=True)
class PITColumns:
    instrument: str = "instrument"
    period_end: str = "period_end"
    available_at: str = "available_at"
    revision_id: str = "revision_id"

# When an announcement becomes tradeable relative to its PubDate.  A-share
# disclosures (earnings, top-ten holders) carry a date, not a time-of-day; the
# same-day close of a bar dated PubDate cannot safely act on an after-close
# announcement, so the conservative production policy is ``next_trading_day``
# (the value is visible to decisions strictly after PubDate).  ``same_day``
# keeps the historical ``available_at <= decision`` behaviour.
AvailablePolicy = Literal["same_day", "next_trading_day"]


def four_layer_pit_allowed(
    *,
    field_pit_allowed: bool,
    table_pit_allowed: bool = True,
    dataset_pit_allowed: bool = True,
    operator_pit_allowed: bool = True,
) -> tuple[bool, dict[str, bool]]:
    """Combined four-layer PIT eligibility (round-7 WS-E #282).

    Production PIT eligibility = ``field ∧ table ∧ dataset ∧ operator``.  Each
    layer is a bool; the combined result is false if any layer is false.  Returns
    ``(combined, layer_status)`` so a caller can report which layer rejected the
    field.

    * ``field_pit_allowed`` — the field's ``strict_pit_allowed`` contract.
    * ``table_pit_allowed`` — the owning logical table's ``strict_pit_allowed``.
    * ``dataset_pit_allowed`` — the DataAccess COS contract's ``pit_policy``
      (``strict``) supports PIT reads.
    * ``operator_pit_allowed`` — operator-level PIT policy (caller-supplied).
    """
    layers = {
        "field_pit_allowed": bool(field_pit_allowed),
        "table_pit_allowed": bool(table_pit_allowed),
        "dataset_pit_allowed": bool(dataset_pit_allowed),
        "operator_pit_allowed": bool(operator_pit_allowed),
    }
    return (all(layers.values()), layers)


def _shift_available_to_next_decision(
    available: pd.Series,
    decisions: pd.DataFrame,
    *,
    decision_time: str,
) -> pd.Series:
    """Shift each event's ``available_at`` to the next decision timestamp.

    A report published on ``PubDate`` cannot inform a factor whose decision bar
    is dated ``PubDate`` (announcements land after close).  With the
    ``next_trading_day`` policy every event is made visible only to the first
    decision strictly after its announcement date.  The shift is computed from
    the actual decision grid, so sparse factors (e.g. monthly) naturally see the
    value at their next decision bar.
    """
    # Work in int64 ns-since-epoch: tz-aware datetime arrays cannot be compared
    # as raw numpy datetime64 (pandas returns an object array of Timestamps), but
    # their epoch offsets are directly searchsorted-comparable.
    decision_dates = np.unique(
        pd.to_datetime(decisions[decision_time], errors="raise", utc=True)
        .dt.normalize()
        .dt
        .tz_localize(None)
        .astype("int64")
        .to_numpy()
    )
    available_dates = (
        pd.to_datetime(available, errors="coerce", utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
        .astype("int64")
        .to_numpy()
    )
    idx = np.searchsorted(decision_dates, available_dates, side="right")
    isna = pd.isna(pd.to_datetime(available, errors="coerce", utc=True)).to_numpy()
    visible = (idx < len(decision_dates)) & ~isna
    out = pd.Series(pd.NaT, index=available.index, dtype="datetime64[ns]")
    if visible.any():
        out.iloc[visible] = pd.Index(
            [pd.Timestamp(decision_dates[i]) for i in idx[visible]]
        )
    # merge_asof requires matching key dtypes; the left decision keys are
    # tz-aware UTC, so the shifted availability must be too.
    return out.dt.tz_localize("UTC")


def validate_fundamental_events(events: pd.DataFrame, columns: PITColumns = PITColumns()) -> None:
    required = {columns.instrument, columns.period_end, columns.available_at}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"fundamental events missing PIT columns: {missing}")
    available = pd.to_datetime(events[columns.available_at], errors="coerce", utc=True)
    period_end = pd.to_datetime(events[columns.period_end], errors="coerce", utc=True)
    if available.isna().any():
        raise ValueError("available_at contains null or invalid timestamps")
    if period_end.isna().any():
        raise ValueError("period_end contains null or invalid timestamps")
    # A report cannot become available before its accounting period ends.
    if (available < period_end).any():
        raise ValueError("available_at precedes period_end; PIT leakage risk")
    keys = [columns.instrument, columns.period_end, columns.available_at]
    if columns.revision_id in events.columns:
        keys.append(columns.revision_id)
    if events.duplicated(keys).any():
        raise ValueError(f"duplicate fundamental event vintages on keys={keys}")


def pit_asof_join(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    max_age_days: int | None = 180,
    available_policy: AvailablePolicy = "next_trading_day",
) -> pd.DataFrame:
    """Backward as-of join enforcing ``available_at <= decision_timestamp``.

    Historical revision vintages must be retained in ``events``.  This function
    never deduplicates a report period to its final revised value before joining.

    ``available_policy`` controls when an event becomes visible:

    * ``next_trading_day`` (default, round-6 P0-03) — an announcement on
      ``PubDate`` is only visible to the first decision strictly after it.
      A-share earnings/top-ten filings land after close and carry a date, not a
      time-of-day; assuming ``00:00:00`` same-day availability would let an
      after-close announcement drive that day's close.  Conservative by default.
    * ``same_day`` — ``available_at <= decision`` (legacy behaviour, only when
      the caller asserts the events carry real timestamps such that the
      comparison is sound).
    """
    if available_policy not in {"same_day", "next_trading_day"}:
        raise ValueError(f"unknown available policy {available_policy!r}")
    validate_fundamental_events(events, columns)
    if decision_time not in decisions.columns:
        raise ValueError(f"decisions missing {decision_time!r}")
    if columns.instrument not in decisions.columns:
        raise ValueError(f"decisions missing {columns.instrument!r}")

    left = decisions.copy()
    right = events.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    right[columns.available_at] = pd.to_datetime(right[columns.available_at], errors="raise", utc=True)
    if available_policy == "next_trading_day":
        shifted = _shift_available_to_next_decision(
            right[columns.available_at], left, decision_time=decision_time
        )
        right[columns.available_at] = shifted
        # Events announced after the last decision can never be visible; a NaT
        # as-of key is invalid for merge_asof, so drop them explicitly.
        right = right.loc[shifted.notna()].copy()
    # #404 (review-8): ``pd.merge_asof(by=...)`` requires the asof key to be
    # *globally* monotonic across all ``by`` groups — sorting by
    # ``[instrument, decision_time]`` interleaves instruments (A: Jan1 Jan2,
    # B: Jan1 Jan2 → ``Jan1 Jan2 Jan1 Jan2``) and raises
    # ``ValueError: left keys must be sorted``.  Sort by the asof key first
    # (ties broken by instrument) and restore the caller's row order via an
    # explicit ``__pit_row_order__`` sentinel.
    left = left.copy()
    left["__pit_row_order__"] = range(len(left))
    left = left.sort_values([decision_time, columns.instrument]).reset_index(drop=True)
    right = right.sort_values([columns.available_at, columns.instrument]).reset_index(drop=True)

    joined = pd.merge_asof(
        left,
        right,
        left_on=decision_time,
        right_on=columns.available_at,
        by=columns.instrument,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_fundamental"),
    )
    joined = joined.sort_values("__pit_row_order__").drop(columns="__pit_row_order__").reset_index(drop=True)
    if max_age_days is not None:
        if int(max_age_days) < 0:
            raise ValueError("max_age_days must be non-negative or None")
        age = joined[decision_time] - joined[columns.available_at]
        stale = age > pd.Timedelta(days=int(max_age_days))
        event_columns = [c for c in events.columns if c != columns.instrument]
        joined.loc[stale, event_columns] = pd.NA
    return joined


def _period_mask(period_end: pd.Series, selector: VisiblePeriodSelector) -> pd.Series:
    dates = pd.to_datetime(period_end, errors="coerce")
    if selector == "latest_visible_period":
        return pd.Series(True, index=period_end.index)
    if selector == "annual_only":
        return (dates.dt.month == 12) & (dates.dt.day == 31)
    if selector == "quarterly_only":
        quarter_ends = dates.dt.is_quarter_end
        annual = (dates.dt.month == 12) & (dates.dt.day == 31)
        return quarter_ends & ~annual
    raise ValueError(f"unknown visible period selector {selector!r}")


def select_visible_row_bundles(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    selector: VisiblePeriodSelector = "latest_visible_period",
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    available_policy: AvailablePolicy = "next_trading_day",
) -> pd.DataFrame:
    """Select one whole visible report row for every decision.

    Rows are selected by availability first, then by the latest eligible report
    period and revision.  All value columns therefore come from the same row.

    ``available_policy`` mirrors :func:`pit_asof_join`: the default is
    ``next_trading_day`` (round-6 P0-03) so a date-level announcement is only
    visible to the first decision strictly after ``PubDate``; ``same_day`` is
    the legacy exact-match behaviour for callers with real timestamps.
    """
    if available_policy not in {"same_day", "next_trading_day"}:
        raise ValueError(f"unknown available policy {available_policy!r}")
    validate_fundamental_events(events, columns)
    if decision_time not in decisions.columns or columns.instrument not in decisions.columns:
        raise ValueError("decisions missing PIT decision columns")
    filtered = events.loc[_period_mask(events[columns.period_end], selector)].copy()
    left = decisions.copy()
    left["__pit_row_order__"] = range(len(left))
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    filtered[columns.available_at] = pd.to_datetime(
        filtered[columns.available_at], errors="raise", utc=True
    )
    filtered[columns.period_end] = pd.to_datetime(
        filtered[columns.period_end], errors="raise", utc=True
    )
    if available_policy == "next_trading_day":
        shifted = _shift_available_to_next_decision(
            filtered[columns.available_at], left, decision_time=decision_time
        )
        filtered[columns.available_at] = shifted
        filtered = filtered.loc[shifted.notna()].copy()

    selected_rows: list[dict[str, object]] = []
    event_columns = [c for c in filtered.columns if c != columns.instrument]
    for decision in left.to_dict("records"):
        visible = filtered.loc[
            (filtered[columns.instrument] == decision[columns.instrument])
            & (filtered[columns.available_at] <= decision[decision_time])
        ]
        if visible.empty:
            selected_rows.append({**decision, **{column: pd.NA for column in event_columns}})
            continue
        latest_period = visible[columns.period_end].max()
        candidates = visible.loc[visible[columns.period_end] == latest_period]
        sort_columns = [columns.available_at]
        if columns.revision_id in candidates.columns:
            sort_columns.append(columns.revision_id)
        chosen = candidates.sort_values(sort_columns, kind="stable").iloc[-1].to_dict()
        chosen.pop(columns.instrument, None)
        selected_rows.append({**decision, **chosen})
    return (
        pd.DataFrame(selected_rows)
        .sort_values("__pit_row_order__")
        .drop(columns="__pit_row_order__")
        .reset_index(drop=True)
    )

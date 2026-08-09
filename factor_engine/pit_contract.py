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

    .. note:: R9-P0-019 — this is the *decision-grid* approximation of market
       visibility.  ``market_visible_at`` and ``decision_at`` are DIFFERENT
       concepts: a Jan-2 announcement is market-visible on Jan-3 (next exchange
       session) but only reaches a monthly factor's next decision on Jan-31.
       This helper is retained as the fallback for callers that do not supply a
       market calendar; prefer :func:`_market_visible_shift` for production PIT.
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


def _market_visible_shift(
    knowledge: pd.Series,
    market_calendar: pd.DatetimeIndex | None,
    *,
    market_timezone: str = "UTC",
) -> pd.Series:
    """First MARKET-session day strictly AFTER the knowledge timestamp.

    R9-P0-019/021 (visibility is a market concept, not a decision-grid or UTC
    concept): ``market_visible_at`` is defined by the exchange calendar in the
    MARKET timezone — a US announcement dated ``2026-03-01 00:00 UTC`` lands on
    ``2026-02-28 19:00 ET`` (a different market date), which a UTC-normalised
    decision-grid shift would get wrong.  Returns the first market session date
    strictly after ``knowledge``, tz-aware UTC, or ``NaT`` when the calendar
    runs out / the input is null.  When ``market_calendar`` is ``None`` it
    returns the input unchanged so the caller falls back to the decision grid.
    """
    k = pd.to_datetime(knowledge, errors="coerce", utc=True)
    if market_calendar is None:
        return k
    cal = pd.DatetimeIndex(market_calendar)
    if getattr(cal.tz, "zone", None):
        cal_local = cal.tz_convert(market_timezone).tz_localize(None)
    else:
        cal_local = cal.tz_localize("UTC").tz_convert(market_timezone).tz_localize(None)
    cal_dates = np.unique(cal_local.normalize().astype("int64").to_numpy())
    km = (
        k.dt.tz_convert(market_timezone)
        .dt.normalize()
        .dt.tz_localize(None)
        .astype("int64")
        .to_numpy()
    )
    idx = np.searchsorted(cal_dates, km, side="right")
    out = pd.Series(pd.NaT, index=knowledge.index, dtype="datetime64[ns]")
    # NaT entries survive the int64 conversion as the NaT epoch sentinel, so the
    # validity mask must come from the ORIGINAL series, not from the converted
    # ints (which are never NaN).
    ok = (idx < cal_dates.size) & ~pd.isna(k).to_numpy()
    if ok.any():
        out.iloc[ok] = pd.Index([pd.Timestamp(cal_dates[i]) for i in idx[ok]])
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
    market_calendar: pd.DatetimeIndex | None = None,
    market_timezone: str = "UTC",
    staleness_basis: Literal["knowledge_at", "market_visible_at"] = "knowledge_at",
) -> pd.DataFrame:
    """Backward as-of join enforcing ``market_visible_at <= decision_timestamp``.

    Historical revision vintages must be retained in ``events``.  This function
    never deduplicates a report period to its final revised value before joining.

    **Three timestamps are kept distinct (R9-P0-019/020/021):**

    * ``knowledge_at`` — the original announcement time (unchanged from the
      events panel).
    * ``market_visible_at`` — when the market actually sees the event: the
      first exchange session day strictly after ``knowledge_at``, computed in
      ``market_timezone`` from ``market_calendar`` when provided.  **This is
      NOT the factor's next decision** — a monthly factor's Jan-31 decision
      does not change the fact the market saw a Jan-2 announcement on Jan-3.
      When ``market_calendar`` is ``None`` the join falls back to the
      decision-grid approximation (legacy behaviour) — documented, not silent.
    * ``decision_at`` — the factor's decision timestamp (the join key).

    PIT eligibility is ``market_visible_at <= decision_at``; the output carries
    both ``available_at`` (the join-visible time, backward compatible) and
    ``knowledge_at`` (original), and staleness is measured from
    ``staleness_basis`` (default ``knowledge_at`` — the OLD code measured age
    from the shifted time and systematically understated information age).

    ``available_policy`` controls when an event becomes visible:

    * ``next_trading_day`` (default, round-6 P0-03) — an announcement is only
      visible to the first exchange session / decision strictly after it.
      A-share earnings/top-ten filings land after close and carry a date, not a
      time-of-day; assuming ``00:00:00`` same-day availability would let an
      after-close announcement drive that day's close.  Conservative by default.
    * ``same_day`` — ``knowledge_at <= decision`` (legacy behaviour, only when
      the caller asserts the events carry real timestamps such that the
      comparison is sound).
    """
    if available_policy not in {"same_day", "next_trading_day"}:
        raise ValueError(f"unknown available policy {available_policy!r}")
    if staleness_basis not in {"knowledge_at", "market_visible_at"}:
        raise ValueError(f"unknown staleness basis {staleness_basis!r}")
    validate_fundamental_events(events, columns)
    if decision_time not in decisions.columns:
        raise ValueError(f"decisions missing {decision_time!r}")
    if columns.instrument not in decisions.columns:
        raise ValueError(f"decisions missing {columns.instrument!r}")

    left = decisions.copy()
    right = events.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    right[columns.available_at] = pd.to_datetime(right[columns.available_at], errors="raise", utc=True)
    # R9-P0-020: preserve the ORIGINAL announcement time under a distinct name —
    # the join key below is the market-visible time, so ``available_at`` in the
    # output stays the shifted join key (backward compatible) while the true
    # information age is measured from ``knowledge_at``.
    right["knowledge_at"] = right[columns.available_at]
    if available_policy == "next_trading_day":
        if market_calendar is not None:
            market_visible = _market_visible_shift(
                right[columns.available_at], market_calendar, market_timezone=market_timezone
            )
        else:
            # Fallback (legacy): shift to the next DECISION grid timestamp.  This
            # conflates market visibility with the factor's own decision cadence
            # and understates age; documented for backward compatibility — pass
            # ``market_calendar`` for production PIT.
            market_visible = _shift_available_to_next_decision(
                right[columns.available_at], left, decision_time=decision_time
            )
        right["market_visible_at"] = market_visible
        right[columns.available_at] = market_visible
        # Events that can never become visible have a NaT as-of key, which is
        # invalid for merge_asof — drop them explicitly.
        right = right.loc[market_visible.notna()].copy()
    else:
        right["market_visible_at"] = right[columns.available_at]
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
        # R9-P0-020: information age must be measured from the ORIGINAL
        # announcement (knowledge_at), NOT from the shifted market-visible time
        # — the old code measured ``decision - available_at`` AFTER the shift,
        # so a Jan-2 PubDate reaching a Jan-31 monthly decision looked 0 days
        # old instead of ~29.
        age_basis = joined.get("knowledge_at", joined[columns.available_at])
        if staleness_basis == "market_visible_at":
            age_basis = joined["market_visible_at"]
        age = joined[decision_time] - age_basis
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
    market_calendar: pd.DatetimeIndex | None = None,
    market_timezone: str = "UTC",
) -> pd.DataFrame:
    """Select one whole visible report row for every decision.

    Rows are selected by availability first, then by the latest eligible report
    period and revision.  All value columns therefore come from the same row.

    ``available_policy`` mirrors :func:`pit_asof_join`: the default is
    ``next_trading_day`` (round-6 P0-03) so a date-level announcement is only
    visible to the first exchange session (``market_calendar`` / market
    timezone, when provided) strictly after ``PubDate``; ``same_day`` is
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
    # R9-P0-019/021: preserve the original knowledge time; visibility is the
    # market-session time, distinct from the factor's decision cadence.
    filtered["knowledge_at"] = filtered[columns.available_at]
    if available_policy == "next_trading_day":
        if market_calendar is not None:
            market_visible = _market_visible_shift(
                filtered[columns.available_at], market_calendar, market_timezone=market_timezone
            )
        else:
            market_visible = _shift_available_to_next_decision(
                filtered[columns.available_at], left, decision_time=decision_time
            )
        filtered["market_visible_at"] = market_visible
        filtered[columns.available_at] = market_visible
        filtered = filtered.loc[market_visible.notna()].copy()
    else:
        filtered["market_visible_at"] = filtered[columns.available_at]

    selected_rows: list[dict[str, object]] = []
    event_columns = [c for c in filtered.columns if c != columns.instrument]
    for decision in left.to_dict("records"):
        visible = filtered.loc[
            (filtered[columns.instrument] == decision[columns.instrument])
            & (filtered["market_visible_at"] <= decision[decision_time])
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

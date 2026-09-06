# -*- coding: utf-8 -*-
"""Point-in-time contracts for fundamental event data before factor calculation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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


class AvailabilityPrecision(str, Enum):
    """R24-046..048: availability precision is a DECLARED source-metadata
    attribute, never inferred from data values (e.g. ``time == midnight``)."""

    DATE = "date"
    TIMESTAMP_SECOND = "timestamp_second"
    TIMESTAMP_MILLISECOND = "timestamp_millisecond"
    TIMESTAMP_NANOSECOND = "timestamp_nanosecond"
    SESSION_LABEL = "session_label"
    UNKNOWN = "unknown"

    @classmethod
    def is_timestamp(cls, precision: "AvailabilityPrecision | str | None") -> bool:
        """True only for a declared sub-day timestamp precision."""
        if precision is None:
            return False
        if isinstance(precision, AvailabilityPrecision):
            return precision in {cls.TIMESTAMP_SECOND, cls.TIMESTAMP_MILLISECOND, cls.TIMESTAMP_NANOSECOND}
        return str(precision) in {
            cls.TIMESTAMP_SECOND.value,
            cls.TIMESTAMP_MILLISECOND.value,
            cls.TIMESTAMP_NANOSECOND.value,
        }


class PITLayerVerdict(str, Enum):
    """R24-049..051: tri-state four-layer PIT gate.

    An omitted layer must be ``UNKNOWN`` — never ``True``.  Production rejects
    any layer that is not ``PROVEN_TRUE``.
    """

    PROVEN_TRUE = "proven_true"
    PROVEN_FALSE = "proven_false"
    UNKNOWN = "unknown"


def _verdict(value: bool | PITLayerVerdict | None) -> PITLayerVerdict:
    if value is None:
        return PITLayerVerdict.UNKNOWN
    if isinstance(value, PITLayerVerdict):
        return value
    return PITLayerVerdict.PROVEN_TRUE if bool(value) else PITLayerVerdict.PROVEN_FALSE


def four_layer_pit_allowed(
    *,
    field_pit_allowed: bool | PITLayerVerdict | None,
    table_pit_allowed: bool | PITLayerVerdict | None = None,
    dataset_pit_allowed: bool | PITLayerVerdict | None = None,
    operator_pit_allowed: bool | PITLayerVerdict | None = None,
    production: bool = False,
) -> tuple[bool, dict[str, PITLayerVerdict]]:
    """Combined four-layer PIT eligibility (round-7 WS-E #282).

    R24-049..051: every layer is tri-state.  ``None`` (an omitted/unsupplied
    layer) becomes ``UNKNOWN`` — it is NEVER treated as ``True``.  In
    production, ``UNKNOWN`` or ``PROVEN_FALSE`` on any layer rejects (combined
    ``False``); research treats ``UNKNOWN`` as an unresolved (reject) too —
    only a fully ``PROVEN_TRUE`` stack admits PIT.

    * ``field_pit_allowed`` — the field's ``strict_pit_allowed`` contract.
    * ``table_pit_allowed`` — the owning logical table's ``strict_pit_allowed``.
    * ``dataset_pit_allowed`` — the DataAccess COS contract's ``pit_policy``.
    * ``operator_pit_allowed`` — operator-level PIT policy (caller-supplied).
    """
    layers: dict[str, PITLayerVerdict] = {
        "field_pit_allowed": _verdict(field_pit_allowed),
        "table_pit_allowed": _verdict(table_pit_allowed),
        "dataset_pit_allowed": _verdict(dataset_pit_allowed),
        "operator_pit_allowed": _verdict(operator_pit_allowed),
    }
    combined = all(
        v == PITLayerVerdict.PROVEN_TRUE for v in layers.values()
    )
    if not combined:
        unknown = [k for k, v in layers.items() if v == PITLayerVerdict.UNKNOWN]
        if unknown:
            logger.warning(
                "four-layer PIT gate has UNKNOWN layers %s — never treated as "
                "True (R24-051); production rejects UNKNOWN", sorted(unknown),
            )
    return (combined, layers)


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
    # R10 #8: tz detection must be ``cal.tz is not None`` — for a UTC-aware
    # DatetimeIndex ``getattr(cal.tz, "zone", None)`` is None (zone is a pytz
    # concept), so the old check wrongly fell into ``tz_localize("UTC")`` on an
    # already-tz-aware index and raised "Already tz-aware".
    if cal.tz is not None:
        cal_local = cal.tz_convert(market_timezone).tz_localize(None)
    else:
        # A naive exchange calendar contains local session labels, not UTC
        # instants.  Keep that type distinction until the selected local
        # midnight is localized below.
        cal_local = cal
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
        local_midnights = pd.DatetimeIndex(
            [pd.Timestamp(cal_dates[i]) for i in idx[ok]]
        ).tz_localize(market_timezone, ambiguous="raise", nonexistent="raise")
        out.iloc[ok] = local_midnights.tz_localize(None)
    return out.dt.tz_localize(market_timezone).dt.tz_convert("UTC")


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


def _same_day_precision_check(
    available: pd.Series,
    *,
    production: bool,
    dataset: str,
    precision: AvailabilityPrecision | str | None = None,
) -> None:
    """R10 #46 + R24-046..048: ``same_day`` requires DECLARED timestamp
    precision.

    A date-only availability under ``same_day`` lets an after-close filing drive
    that same day's close (leak).  The precision MUST come from source metadata
    (``precision``) — it is NEVER inferred by inspecting data values (a
    midnight ``00:00:00`` timestamp is not proof of date-only semantics).
    ``UNKNOWN``/``DATE`` precision rejects in production and warns in research.
    """
    precision = AvailabilityPrecision(precision) if precision is not None else AvailabilityPrecision.UNKNOWN
    if precision == AvailabilityPrecision.UNKNOWN:
        if production:
            raise ValueError(
                f"same_day PIT on {dataset!r} requires a DECLARED "
                "timestamp AvailabilityPrecision; UNKNOWN cannot be inferred "
                "from the shape of production data."
            )
        # No declared precision.  Backward-compatible research fallback: use a
        # TIMESTAMP only if EVERY value carries a real time-of-day.
        avail = pd.to_datetime(available, errors="coerce", utc=True)
        midnight = pd.Timestamp("00:00:00").time()
        non_null = avail[avail.notna()]
        looks_timestamp = non_null.empty or bool((non_null.dt.time != midnight).all())
        inferred = AvailabilityPrecision.TIMESTAMP_NANOSECOND if looks_timestamp else AvailabilityPrecision.DATE
        logger.warning(
            "same_day PIT on %r has no DECLARED availability precision; "
            "inferred %s from data values (research-only fallback).",
            dataset, inferred.value,
        )
        return
    if AvailabilityPrecision.is_timestamp(precision):
        return
    # Declared DATE / SESSION_LABEL (or any non-timestamp) precision.
    if production:
        raise ValueError(
            f"same_day PIT on {dataset!r} requires TIMESTAMP-precision "
            f"available_at; declared precision is {precision.value!r}. "
            "use next_trading_day or provide real announcement timestamps."
        )
    logger.warning(
        "same_day PIT on %r uses declared precision %r; "
        "after-close filing could leak into the same day's close. "
        "production rejects this; research keeps legacy behaviour.",
        dataset, precision.value,
    )


def _pit_tie_break_columns(columns: PITColumns, frame: pd.DataFrame) -> list[str]:
    """Deterministic tie-break for same-instrument/same-available_at candidates.

    R10 #45: ``merge_asof`` picks the LAST row among exact key ties, so the
    ``right`` frame must be sorted by a total order — never left to original row
    order.  Preference: later ``period_end`` (newer report) wins; then a later
    ``revision_id`` (newer revision) wins.
    """
    cols: list[str] = [columns.available_at, columns.instrument]
    if columns.period_end in frame.columns:
        cols.append(columns.period_end)
    if columns.revision_id in frame.columns:
        cols.append(columns.revision_id)
    return cols


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
    production: bool = False,
    precision: AvailabilityPrecision | str | None = None,
) -> pd.DataFrame:
    """Backward as-of join enforcing ``market_visible_at <= decision_timestamp``.

    R24-040/041: this is the **event / knowledge-time** selector
    (``select_latest_event_by_knowledge_time``) — it picks the event with the
    latest visible knowledge time, NOT the latest fiscal-period state (see
    :func:`select_latest_fiscal_period_state`).  It is NOT the right tool for
    financial period-state reads.

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
    # R10 #8: ``next_trading_day`` defines market visibility via the exchange
    # calendar.  Without one the decision-grid fallback CONFLATES
    # ``market_visible_at`` with ``decision_at`` — production must reject, never
    # fall back.
    if (
        production
        and available_policy == "next_trading_day"
        and market_calendar is None
    ):
        raise ValueError(
            "production PIT with available_policy='next_trading_day' requires "
            "a real exchange calendar (market_calendar + market_timezone); the "
            "decision-grid fallback conflates market_visible_at with decision_at."
        )

    left = decisions.copy()
    right = events.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    right[columns.available_at] = pd.to_datetime(right[columns.available_at], errors="raise", utc=True)
    # 2026-08-29: the daily anchor index carries ``datetime64[s, UTC]`` while the
    # financial ``PubDate`` may be ``datetime64[us, UTC]``/``datetime64[ns, UTC]``.
    # ``pd.merge_asof`` requires the asof keys to be the SAME dtype/unit — normalize
    # both to nanoseconds so PIT joins never trip
    # ``MergeError: incompatible merge keys ... must be the same type``.
    if left[decision_time].dtype != right[columns.available_at].dtype:
        common_unit = "datetime64[ns, UTC]"
        left[decision_time] = left[decision_time].astype(common_unit)
        right[columns.available_at] = right[columns.available_at].astype(common_unit)
    # R10 #46 + R24-046..048: same_day requires a DECLARED timestamp precision
    # (source metadata), never inferred from midnight data values.
    if available_policy == "same_day":
        _same_day_precision_check(
            right[columns.available_at], production=production,
            dataset=columns.instrument, precision=precision,
        )
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
            # Fallback (legacy, research only): shift to the next DECISION grid
            # timestamp.  This conflates market visibility with the factor's own
            # decision cadence and understates age; documented for backward
            # compatibility — production requires ``market_calendar`` (R10 #8).
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
    # 2026-08-29: after the ``next_trading_day`` shift, ``available_at`` may be
    # rebuilt from a different source dtype — normalize BOTH asof keys to the
    # same unit right before merge_asof (the earlier equality check ran before
    # the shift reassigned ``right[available_at]``).
    if left[decision_time].dtype != right[columns.available_at].dtype:
        common_unit = "datetime64[ns, UTC]"
        left[decision_time] = left[decision_time].astype(common_unit)
        right[columns.available_at] = right[columns.available_at].astype(common_unit)
    # R10 #45: merge_asof returns the LAST row among exact key ties, so the
    # right frame must have a deterministic total order — newest period_end wins,
    # then newest revision_id.  Original-row order is never used to break ties.
    right = right.sort_values(
        _pit_tie_break_columns(columns, right)
    ).reset_index(drop=True)

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


# R24-056/057: explicit selector lattice.  A caller may only keep or tighten the
# registered field contract — never loosen a narrow contract to a broad one.
#   annual_only    ⊂ latest_visible_period
#   quarterly_only ⊂ latest_visible_period
_PERIOD_SELECTOR_LATTICE = {
    "latest_visible_period": frozenset({"latest_visible_period", "annual_only", "quarterly_only"}),
    "annual_only": frozenset({"annual_only"}),
    "quarterly_only": frozenset({"quarterly_only"}),
}


def selector_compatible(required: str, requested: str) -> bool:
    """R24-056/057: ``requested`` is allowed iff it keeps or tightens ``required``.

    ``requested in _PERIOD_SELECTOR_LATTICE[required]`` — a narrow contract
    (``annual_only``) cannot be relaxed to ``latest_visible_period``, while a
    broad contract (``latest_visible_period``) may be tightened to either
    annual or quarterly view.
    """
    if required not in _PERIOD_SELECTOR_LATTICE:
        raise ValueError(f"unknown required period selector {required!r}")
    if requested not in _PERIOD_SELECTOR_LATTICE:
        raise ValueError(f"unknown requested period selector {requested!r}")
    return requested in _PERIOD_SELECTOR_LATTICE[required]


def _period_mask(
    period_end: pd.Series,
    selector: VisiblePeriodSelector,
    timeframe: pd.Series | None = None,
    fiscal_quarter: pd.Series | None = None,
) -> pd.Series:
    """Select fiscal periods by declared timeframe / fiscal columns (R24-043..045).

    A US non-December fiscal year must NOT be identified by ``month==12 &&
    day==31``.  When the events carry a declared ``timeframe`` column (e.g.
    ``annual`` / ``quarterly`` / ``ttm``) or ``fiscal_quarter`` / ``fiscal_year``
    columns, those are authoritative.  The calendar ``12-31`` heuristic is only a
    last resort for panels that carry no fiscal metadata (documented, not ideal).
    """
    if selector == "latest_visible_period":
        return pd.Series(True, index=period_end.index)
    if timeframe is not None and timeframe.notna().any():
        tf = timeframe.astype(str).str.lower().str.strip()
        annual_flags = ("annual", "fy", "a", "year", "annual_report", "y")
        quarterly_flags = ("quarterly", "q", "quarter", "q1", "q2", "q3", "q4")
        if selector == "annual_only":
            return tf.isin(annual_flags)
        return tf.isin(quarterly_flags) & ~tf.isin(annual_flags)
    if fiscal_quarter is not None and selector == "quarterly_only":
        # fiscal_quarter in {1,2,3,4} marks a quarterly statement; annual has NaN.
        fq = pd.to_numeric(fiscal_quarter, errors="coerce")
        return fq.isin([1, 2, 3, 4])
    dates = pd.to_datetime(period_end, errors="coerce")
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
    production: bool = False,
    precision: AvailabilityPrecision | str | None = None,
    timeframe_column: str | None = None,
    fiscal_quarter_column: str | None = None,
) -> pd.DataFrame:
    """Select one whole visible report row for every decision.

    Rows are selected by availability first, then by the latest eligible report
    period and revision.  All value columns therefore come from the same row.

    R24-040/041: this is the **fiscal-period-state** selector
    (``select_latest_fiscal_period_state``).  It is semantically distinct from
    the event/knowledge-time selector (:func:`pit_asof_join` /
    ``select_latest_event_by_knowledge_time``): it picks the LATEST REPORT
    PERIOD visible at the decision, so a same-day restatement of an OLD period
    (R24-042) never moves the selected financial state backwards.

    ``available_policy`` mirrors :func:`pit_asof_join`: the default is
    ``next_trading_day`` (round-6 P0-03) so a date-level announcement is only
    visible to the first exchange session (``market_calendar`` / market
    timezone, when provided) strictly after ``PubDate``; ``same_day`` is
    the legacy exact-match behaviour for callers with real timestamps.
    ``production`` (R10 #8/#46) rejects next_trading_day without a market
    calendar and same_day on non-timestamp declared precision.
    ``precision`` (R24-046..048) must be the DECLARED availability precision
    from source metadata; ``timeframe_column`` / ``fiscal_quarter_column``
    (R24-043..045) drive the annual/quarterly mask when present.
    """
    if available_policy not in {"same_day", "next_trading_day"}:
        raise ValueError(f"unknown available policy {available_policy!r}")
    if (
        production
        and available_policy == "next_trading_day"
        and market_calendar is None
    ):
        raise ValueError(
            "production PIT with available_policy='next_trading_day' requires "
            "a real exchange calendar (market_calendar + market_timezone)."
        )
    validate_fundamental_events(events, columns)
    if decision_time not in decisions.columns or columns.instrument not in decisions.columns:
        raise ValueError("decisions missing PIT decision columns")
    tf = events[timeframe_column] if timeframe_column and timeframe_column in events.columns else None
    fq = events[fiscal_quarter_column] if fiscal_quarter_column and fiscal_quarter_column in events.columns else None
    filtered = events.loc[_period_mask(events[columns.period_end], selector, tf, fq)].copy()
    left = decisions.copy()
    left["__pit_row_order__"] = range(len(left))
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    if left.empty:
        # A zero-row decision request is a valid empty result, not malformed
        # PIT data.  Preserve the public output schema deterministically.
        event_columns = [
            c for c in events.columns if c != columns.instrument and c not in left.columns
        ]
        for name in ("knowledge_at", "market_visible_at"):
            if name not in event_columns and name not in left.columns:
                event_columns.append(name)
        return left.drop(columns="__pit_row_order__").reindex(
            columns=[*decisions.columns, *event_columns]
        )
    filtered[columns.available_at] = pd.to_datetime(
        filtered[columns.available_at], errors="raise", utc=True
    )
    filtered[columns.period_end] = pd.to_datetime(
        filtered[columns.period_end], errors="raise", utc=True
    )
    # R10 #46 + R24-046..048: same_day requires TIMESTAMP precision declared in
    # source metadata (never inferred from midnight data values).
    if available_policy == "same_day":
        _same_day_precision_check(
            filtered[columns.available_at], production=production,
            dataset=columns.instrument, precision=precision,
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

    selected_rows: list[dict[str, object] | None] = [None] * len(left)
    event_columns = [c for c in filtered.columns if c != columns.instrument]
    # FE-16: one monotone visibility sweep per instrument.  State is keyed by
    # fiscal period, so an old-period restatement updates that period's vintage
    # without replacing the latest visible fiscal period.
    empty_values = {column: pd.NA for column in event_columns}
    decision_records = left.to_dict("records")
    decision_positions: dict[object, list[int]] = {}
    for position, decision in enumerate(decision_records):
        decision_positions.setdefault(decision[columns.instrument], []).append(position)

    events_by_instrument = {
        instrument: group
        for instrument, group in filtered.groupby(columns.instrument, sort=False)
    }
    for instrument, positions in decision_positions.items():
        instrument_events = events_by_instrument.get(instrument, filtered.iloc[:0])
        # The canonical next-session path replaces available_at with
        # market_visible_at before selection.  Preserve its exact tie-break:
        # same-session candidates are ordered by revision, not original
        # knowledge time.
        event_sort = ["market_visible_at", columns.period_end]
        if columns.revision_id in instrument_events.columns:
            event_sort.append(columns.revision_id)
        instrument_events = instrument_events.sort_values(event_sort, kind="stable")
        event_records = instrument_events.to_dict("records")
        valid_positions: list[int] = []
        for position in positions:
            if pd.isna(decision_records[position][decision_time]):
                selected_rows[position] = {
                    **decision_records[position], **empty_values
                }
            else:
                valid_positions.append(position)
        ordered_positions = sorted(
            valid_positions, key=lambda pos: decision_records[pos][decision_time]
        )
        period_state: dict[object, dict[str, object]] = {}
        latest_period = None
        event_pos = 0
        for position in ordered_positions:
            decision = decision_records[position]
            decision_at = decision[decision_time]
            while (
                event_pos < len(event_records)
                and event_records[event_pos]["market_visible_at"] <= decision_at
            ):
                event = event_records[event_pos]
                # Event order is (visibility, period, revision);
                # overwrite therefore matches the reference's latest
                # available_at then latest revision tie-break for each period.
                period_state[event[columns.period_end]] = event
                if latest_period is None or event[columns.period_end] > latest_period:
                    latest_period = event[columns.period_end]
                event_pos += 1
            if not period_state:
                selected_rows[position] = {**decision, **empty_values}
                continue
            chosen = dict(period_state[latest_period])
            chosen.pop(columns.instrument, None)
            selected_rows[position] = {**decision, **chosen}
    return (
        pd.DataFrame(selected_rows)
        .sort_values("__pit_row_order__")
        .drop(columns="__pit_row_order__")
        .reset_index(drop=True)
    )


def select_latest_fiscal_period_state(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """R24-040: explicit fiscal-period-state selector (latest period → revision).

    Semantically identical to :func:`select_visible_row_bundles` — the name is
    the explicit API so a caller cannot mistake it for an event/knowledge-time
    selector.  An old-period restatement (R24-042) never moves the selected
    financial state backwards.
    """
    return select_visible_row_bundles(decisions, events, **kwargs)


def select_latest_event_by_knowledge_time(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """R24-040: explicit event/knowledge-time selector (latest visible event).

    Semantically identical to :func:`pit_asof_join` — the latest visible
    *event* by knowledge time.  This is NOT a fiscal-period-state selector; use
    :func:`select_latest_fiscal_period_state` for financial reads.
    """
    return pit_asof_join(decisions, events, **kwargs)


def select_specific_period_vintage(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    period_end: pd.Timestamp,
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    available_policy: AvailablePolicy = "next_trading_day",
    market_calendar: pd.DatetimeIndex | None = None,
    market_timezone: str = "UTC",
    production: bool = False,
    precision: AvailabilityPrecision | str | None = None,
) -> pd.DataFrame:
    """R24-040: select the visible vintage of ONE specific fiscal period.

    At each decision time, only rows whose ``period_end == period_end`` AND that
    are market-visible are candidates; the newest visible revision of that period
    is returned (one row per decision).  Used for old-period restatement analysis
    without contaminating the latest-period state.
    """
    if available_policy not in {"same_day", "next_trading_day"}:
        raise ValueError(f"unknown available policy {available_policy!r}")
    if (
        production
        and available_policy == "next_trading_day"
        and market_calendar is None
    ):
        raise ValueError(
            "production PIT with available_policy='next_trading_day' requires "
            "a real exchange calendar (market_calendar + market_timezone)."
        )
    validate_fundamental_events(events, columns)
    events = events.copy()
    events[columns.period_end] = pd.to_datetime(events[columns.period_end], errors="raise", utc=True)
    target = pd.Timestamp(period_end)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    filtered = events.loc[events[columns.period_end] == target].copy()
    filtered[columns.available_at] = pd.to_datetime(filtered[columns.available_at], errors="raise", utc=True)
    filtered["knowledge_at"] = filtered[columns.available_at]
    if available_policy == "next_trading_day":
        if market_calendar is not None:
            market_visible = _market_visible_shift(
                filtered[columns.available_at], market_calendar, market_timezone=market_timezone
            )
        else:
            market_visible = _shift_available_to_next_decision(
                filtered[columns.available_at], decisions, decision_time=decision_time
            )
        filtered["market_visible_at"] = market_visible
        filtered[columns.available_at] = market_visible
        filtered = filtered.loc[market_visible.notna()].copy()
    else:
        _same_day_precision_check(
            filtered[columns.available_at], production=production,
            dataset=columns.instrument, precision=precision,
        )
        filtered["market_visible_at"] = filtered[columns.available_at]
    left = decisions.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
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
        sort_cols = [columns.available_at]
        if columns.revision_id in visible.columns:
            sort_cols.append(columns.revision_id)
        chosen = visible.sort_values(sort_cols, kind="stable").iloc[-1].to_dict()
        chosen.pop(columns.instrument, None)
        selected_rows.append({**decision, **chosen})
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def select_revision_event_stream(
    events: pd.DataFrame,
    *,
    columns: PITColumns = PITColumns(),
    decision_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """R24-040: return the full visible revision event stream (never collapsed).

    Unlike :func:`select_latest_fiscal_period_state`, this does NOT select one
    row per decision — it returns every revision of every period that is visible
    by ``decision_time`` (or the whole panel when ``None``), ordered by
    (instrument, period_end, available_at, revision_id).  Used for revision
    latency / restatement analysis.
    """
    validate_fundamental_events(events, columns)
    out = events.copy()
    out[columns.available_at] = pd.to_datetime(out[columns.available_at], errors="raise", utc=True)
    out[columns.period_end] = pd.to_datetime(out[columns.period_end], errors="raise", utc=True)
    out["knowledge_at"] = out[columns.available_at]
    if decision_time is not None:
        cutoff = pd.Timestamp(decision_time)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        out = out.loc[out[columns.available_at] <= cutoff].copy()
    sort_cols = [columns.instrument, columns.period_end, columns.available_at]
    if columns.revision_id in out.columns:
        sort_cols.append(columns.revision_id)
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

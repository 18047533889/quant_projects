# -*- coding: utf-8 -*-
"""ClockSemantics — global operator clock / cadence metadata design (P11).

A factor operator's *clock* is the cadence at which its inputs are sampled and
its outputs are produced.  Two operators may be numerically identical but live
on different clocks (``ts_mean`` samples trading bars; ``event_*`` fire at event
time; ``fin_*``/``fiscal_*`` tick only when a new report period is published;
``intraday_*_slot`` measures a session slot).  Mixing clocks in one expression is
frequently the source of look-ahead / leakage bugs, so the engine needs a single
declarative answer to "what clock does this operator run on?".

This module is the FIRST step of the global ClockSemantics design: a canonical
metadata registry plus a prefix/rule-based resolver.  It is intentionally NOT
yet wired into every backend — Pandas / Polars / DuckDB must each implement the
SAME ``ClockSemantics`` for a given canonical (that is a documented contract,
enforced by future cross-backend tests), and the declaration surface
(``declare_clock``) lets a backend or a recipe override a family default.

``ClockSemantics`` alone is insufficient for operators that *change* cadence
between their inputs and their output (review item R9-P0-010): e.g.
``event_historical_response_mean`` consumes event markers but produces a value
that is naturally evaluated once per trading bar.  Production resolution is
therefore done through :class:`ClockContract` — a per-operator
``(input_clocks, output_clock, grain_transform, update_trigger)`` contract —
resolved by :func:`clock_contract_for`, which reads ONLY the explicit registry
keyed by canonical name and NEVER guesses a clock from a ``fin_``/``event_``/
``ts_`` name prefix.  For canonicals without an explicit entry it returns a
conservative same-clock default (``UNKNOWN`` input == output) instead of
prefix-guessing.

Semantics of the enum values:

* ``TRADING_BAR``   — one output per trading bar; inputs are OHLCV/bar panels.
* ``OBSERVATION``   — one output per observation row; the default for plain
                      elementwise / cross-sectional operators that do not change
                      cadence.
* ``EVENT``         — output advances at event arrival (event-response, level
                      survival, state transitions keyed by a marker series).
* ``SESSION_SLOT``  — output advances at intraday session-slot boundaries.
* ``FISCAL_PERIOD`` — output advances at report/fiscal-period publication; never
                      samples a bar that has not yet been reported.
* ``UNKNOWN``       — not yet classified; treated as unsafe for clock-sensitive
                      rewrites.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


class ClockSemantics(Enum):
    TRADING_BAR = "trading_bar"
    OBSERVATION = "observation"
    EVENT = "event"
    SESSION_SLOT = "session_slot"
    FISCAL_PERIOD = "fiscal_period"
    UNKNOWN = "unknown"


# Explicit per-canonical overrides.  These win over prefix rules so an
# ``intraday_*`` operator that is actually daily-frequency, or a ``ts_*``
# operator that is event-driven, can be pinned precisely.  Family defaults are
# handled by prefix rules in ``clock_for``; only exceptions and high-value
# canonicals need to live here.
_CLOCK_BY_CANONICAL: dict[str, ClockSemantics] = {
    # ts_* rolling family -> trading-bar cadence.
    "ts_mean": ClockSemantics.TRADING_BAR,
    "ts_std": ClockSemantics.TRADING_BAR,
    "ts_var": ClockSemantics.TRADING_BAR,
    "ts_sum": ClockSemantics.TRADING_BAR,
    "ts_max": ClockSemantics.TRADING_BAR,
    "ts_min": ClockSemantics.TRADING_BAR,
    "ts_median": ClockSemantics.TRADING_BAR,
    "ts_delay": ClockSemantics.TRADING_BAR,
    "ts_delta": ClockSemantics.TRADING_BAR,
    "ts_pct": ClockSemantics.TRADING_BAR,
    "ts_zscore": ClockSemantics.TRADING_BAR,
    "ts_rank": ClockSemantics.TRADING_BAR,
    "ts_sharpe": ClockSemantics.TRADING_BAR,
    "ts_autocorr": ClockSemantics.TRADING_BAR,
    "ts_corr": ClockSemantics.TRADING_BAR,
    "ts_cov": ClockSemantics.TRADING_BAR,
    "ts_beta": ClockSemantics.TRADING_BAR,
    "ts_regression_slope": ClockSemantics.TRADING_BAR,
    "ts_decay_linear": ClockSemantics.TRADING_BAR,
    "ts_decay_exp": ClockSemantics.TRADING_BAR,
    "rolling_beta": ClockSemantics.TRADING_BAR,
    "rolling_corr": ClockSemantics.TRADING_BAR,
    "rolling_cov": ClockSemantics.TRADING_BAR,
    "rolling_max": ClockSemantics.TRADING_BAR,
    "rolling_min": ClockSemantics.TRADING_BAR,
    "rolling_mean": ClockSemantics.TRADING_BAR,
    "rolling_std": ClockSemantics.TRADING_BAR,
    "rolling_sum": ClockSemantics.TRADING_BAR,
    "volatility": ClockSemantics.TRADING_BAR,
    # event_* family -> event-driven cadence.
    "event_response": ClockSemantics.EVENT,
    "event_response_effective_events": ClockSemantics.EVENT,
    "event_historical_response_mean": ClockSemantics.EVENT,
    "event_historical_response_sign_balance": ClockSemantics.EVENT,
    "event_uptrend_duration": ClockSemantics.EVENT,
    "event_downtrend_duration": ClockSemantics.EVENT,
    "event_survival_share": ClockSemantics.EVENT,
    "event_level_survival_share": ClockSemantics.EVENT,
    # fin_/fiscal_ family -> report-period cadence.
    "fin_lag": ClockSemantics.FISCAL_PERIOD,
    "fin_revision_count": ClockSemantics.FISCAL_PERIOD,
    "fin_revision_magnitude": ClockSemantics.FISCAL_PERIOD,
    "fin_revision_surprise": ClockSemantics.FISCAL_PERIOD,
    "fin_report_filing_delay": ClockSemantics.FISCAL_PERIOD,
    "fin_margin_persistence": ClockSemantics.FISCAL_PERIOD,
    "fiscal_yoy": ClockSemantics.FISCAL_PERIOD,
    "fiscal_quarter_avg": ClockSemantics.FISCAL_PERIOD,
    "fiscal_quarter_sum": ClockSemantics.FISCAL_PERIOD,
    "fiscal_component_score": ClockSemantics.FISCAL_PERIOD,
    "fiscal_asymmetric_elasticity": ClockSemantics.FISCAL_PERIOD,
    "fiscal_sign_consistency": ClockSemantics.FISCAL_PERIOD,
    "fiscal_sign_agreement": ClockSemantics.FISCAL_PERIOD,
    # intraday session-slot family.
    "intraday_slot": ClockSemantics.SESSION_SLOT,
    "intraday_slot_return": ClockSemantics.SESSION_SLOT,
    "intraday_slot_volume_share": ClockSemantics.SESSION_SLOT,
    "intraday_slot_volatility": ClockSemantics.SESSION_SLOT,
    "intraday_morning_return": ClockSemantics.SESSION_SLOT,
    "intraday_afternoon_return": ClockSemantics.SESSION_SLOT,
    "intraday_am_pm_gap": ClockSemantics.SESSION_SLOT,
    "intraday_activity_duration_curvature": ClockSemantics.SESSION_SLOT,
    # lag / horizon family -> observation cadence.
    "lag": ClockSemantics.OBSERVATION,
    "delay": ClockSemantics.OBSERVATION,
    "horizon_return": ClockSemantics.OBSERVATION,
    "horizon_volatility": ClockSemantics.OBSERVATION,
}


def _default_clock(canonical: str) -> ClockSemantics:
    """Prefix-based fallback for canonicals without an explicit table entry."""
    if canonical.startswith("fin_") or canonical.startswith("fiscal_"):
        return ClockSemantics.FISCAL_PERIOD
    if canonical.startswith("event_"):
        return ClockSemantics.EVENT
    if canonical.startswith("intraday_") or canonical.endswith("_slot"):
        return ClockSemantics.SESSION_SLOT
    if canonical.startswith("ts_"):
        return ClockSemantics.TRADING_BAR
    return ClockSemantics.OBSERVATION


def clock_for(canonical: str) -> ClockSemantics:
    """Resolve the ClockSemantics for a canonical operator name.

    Resolution order:
      1. explicit ``_CLOCK_BY_CANONICAL`` table (declared via ``declare_clock``
         or the built-in table);
      2. prefix rules: ``fin_``/``fiscal_`` -> FISCAL_PERIOD, ``event_`` -> EVENT,
         ``intraday_``/``*_slot`` -> SESSION_SLOT, ``ts_`` -> TRADING_BAR;
      3. OBSERVATION as the safe default for everything else.
    """
    key = str(canonical)
    explicit = _CLOCK_BY_CANONICAL.get(key)
    if explicit is not None:
        return explicit
    return _default_clock(key)


def clock_registry() -> dict[str, str]:
    """Return a stable, JSON-able view of every explicit clock declaration."""
    return {name: clock.value for name, clock in sorted(_CLOCK_BY_CANONICAL.items())}


def declare_clock(canonical: str, clock: ClockSemantics) -> None:
    """Declare (or override) the ClockSemantics for one canonical.

    Callers are responsible for keeping Pandas / Polars / DuckDB implementations
    in agreement — the global contract is that every backend implements the SAME
    ClockSemantics for a given canonical.  This is the extension point for
    recipes and backend bindings; it is intentionally process-local (persisted
    metadata lands in the operator manifest / evidence layer separately).
    """
    if not isinstance(clock, ClockSemantics):
        raise TypeError(f"clock must be a ClockSemantics, got {type(clock).__name__}")
    _CLOCK_BY_CANONICAL[str(canonical)] = clock


# ---------------------------------------------------------------------------
# ClockContract — per-operator input/output clock contract (R9-P0-010)
# ---------------------------------------------------------------------------
# A single ClockSemantics only answers "what cadence is this operator on?".  It
# is not enough once an operator *changes* cadence between its inputs and its
# output:
#
#   * ``event_historical_response_mean`` — inputs are event markers on the
#     EVENT clock; the output is a response mean naturally evaluated once per
#     trading bar (``event -> daily``).
#   * ``fin_announcement_lag`` — inputs (period-end / publication dates) update
#     only when a report is published (filing clock); the output is a natural-day
#     lag evaluated on the daily trading-bar clock (``filing -> daily``).
#
# The registry below is the single production source of truth.  It is keyed by
# canonical name and is intentionally NOT prefix-rule-based: adding a clock
# split for every ``fin_*``/``event_*``/``ts_*`` operator is exactly the kind of
# name-prefix guessing the review forbids.  Explicit entries are added per
# operator (usually alongside its operator metadata), and unlisted canonicals
# get a conservative same-clock default.

# Sentinel key used in ``ClockContract.input_clocks`` for canonicals whose exact
# input names are not known.  ``"*"`` records the conservative assumption that
# EVERY input shares the clock stored under this key (see
# ``_conservative_default_contract``).
ANY_INPUT = "*"


@dataclass(frozen=True)
class ClockContract:
    """Structured input/output clock contract for one canonical operator.

    Fields:
        input_clocks: per-input clock mapping (input name -> ClockSemantics).
            For unlisted canonicals a ``"*"`` wildcard key means "every input is
            conservatively assumed to share this clock".
        output_clock: cadence of the produced output.
        grain_transform: human-readable cadence transform, e.g. ``"event->daily"``,
            ``"filing->daily"``, or ``"same"`` when input and output share a clock.
        update_trigger: what event advances this operator's inputs, e.g.
            ``"bar_close"``, ``"event_detect"``, ``"filing"``.
    """

    input_clocks: dict[str, ClockSemantics] = field(default_factory=dict)
    output_clock: ClockSemantics = ClockSemantics.UNKNOWN
    grain_transform: str = "same"
    update_trigger: str = "unknown"

    def input_clock(self, name: str) -> ClockSemantics:
        """Resolve the clock for one named input.

        Unknown input names fall back to the output clock (conservative
        same-clock assumption); for unlisted canonicals the ``"*"`` wildcard
        entry answers for any name.  Callers therefore never need to know the
        full input list to ask "what clock is input X on?".
        """
        if name in self.input_clocks:
            return self.input_clocks[name]
        wildcard = self.input_clocks.get(ANY_INPUT)
        if wildcard is not None:
            return wildcard
        return self.output_clock


# Explicit per-canonical contracts.  This is the production surface: it is the
# ONLY place input/output clock differences may be declared.  Canonicals are
# added here (or via ``declare_contract``) as they are wired into backends; the
# resolver never falls back to prefix-guessing.
_CONTRACT_BY_CANONICAL: dict[str, ClockContract] = {
    # Event inputs -> daily trading-bar output.
    "event_historical_response_mean": ClockContract(
        input_clocks={
            "response": ClockSemantics.EVENT,
            "event": ClockSemantics.EVENT,
        },
        output_clock=ClockSemantics.TRADING_BAR,
        grain_transform="event->daily",
        update_trigger="event_detect",
    ),
    # Financial as-of: inputs update only on publication (filing clock), the
    # output lag is evaluated on the daily trading-bar clock.
    "fin_announcement_lag": ClockContract(
        input_clocks={
            "period_end_date": ClockSemantics.FISCAL_PERIOD,
            "pub_date": ClockSemantics.FISCAL_PERIOD,
        },
        output_clock=ClockSemantics.TRADING_BAR,
        grain_transform="filing->daily",
        update_trigger="filing",
    ),
}


def _conservative_default_contract() -> ClockContract:
    """Same-clock conservative default for canonicals without an explicit entry.

    Deliberately does NOT inspect the canonical name — no ``fin_``/``event_``/
    ``ts_`` prefix guessing.  The safe assumption for an unclassified operator
    is that it neither changes cadence nor exposes a known input list, so every
    input is conservatively marked ``UNKNOWN`` and the output shares that clock.
    """
    return ClockContract(
        input_clocks={ANY_INPUT: ClockSemantics.UNKNOWN},
        output_clock=ClockSemantics.UNKNOWN,
        grain_transform="same",
        update_trigger="unknown",
    )


def clock_contract_for(canonical: str) -> ClockContract:
    """Resolve the ClockContract for a canonical operator name.

    Resolution order:
      1. explicit ``_CONTRACT_BY_CANONICAL`` registry entry (declared via
         ``declare_contract`` or the built-in table);
      2. a conservative same-clock default (``UNKNOWN`` inputs == output) —
         never a name-prefix guess.

    Unlike :func:`clock_for` (the legacy single-clock API, which keeps prefix
    rules for backward compatibility), this production resolver derives the
    input/output clock split ONLY from the explicit registry — never from
    ``name.startswith(...)``.
    """
    key = str(canonical)
    contract = _CONTRACT_BY_CANONICAL.get(key)
    if contract is None:
        return _conservative_default_contract()
    # Defensive copy so callers cannot mutate a frozen registry entry's dict.
    return replace(contract, input_clocks=dict(contract.input_clocks))


def declare_contract(canonical: str, contract: ClockContract) -> None:
    """Declare (or override) the ClockContract for one canonical.

    This is the production extension point for recipes and backend bindings
    that know an operator's true input/output clocks.  Callers are responsible
    for keeping Pandas / Polars / DuckDB implementations in agreement, exactly
    as with :func:`declare_clock`.
    """
    if not isinstance(contract, ClockContract):
        raise TypeError(
            f"contract must be a ClockContract, got {type(contract).__name__}"
        )
    _CONTRACT_BY_CANONICAL[str(canonical)] = contract


def contract_registry() -> dict[str, dict]:
    """Return a stable, JSON-able view of every explicit clock contract."""
    return {
        name: {
            "input_clocks": {
                inp: clock.value for inp, clock in contract.input_clocks.items()
            },
            "output_clock": contract.output_clock.value,
            "grain_transform": contract.grain_transform,
            "update_trigger": contract.update_trigger,
        }
        for name, contract in sorted(_CONTRACT_BY_CANONICAL.items())
    }

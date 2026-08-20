# -*- coding: utf-8 -*-
"""R19 semantic policies — machine-readable declarations for temporal
topology / anchor / missing-value topology / current-row requirement / output
time-units / rank & quantile semantics (R19-075..093).

These declarations are the HISTORY / CERTIFICATION-side authority that
``runtime.execution_contract`` and the audits consult.  They live here (a
runtime-owned policy module) rather than inside each operator file so the
single-authority history layer can read them WITHOUT touching the operator
catalog.  Operators' own metadata MAY carry an equivalent declaration later
(e.g. ``metadata.min_effective_sample``); the lookup helpers in this module
stay pure map lookups so the two can be merged centrally without churn.

Design rules enforced here:

* ``AnchorPolicy`` (R19-075..077): every expanding / cumulative canonical
  declares an anchor.  ``campaign_start`` is NEVER an implicit default — the
  research/evaluation window is not a factor-definition anchor.
* ``MissingTopologyPolicy`` (R19-082/083): lag is physical-row by default;
  autocorr / directional-state / event-spacing operators must NOT re-bridge
  the two sides of a missing observation as neighbours.
* ``CurrentRowRequirement`` (R19-087): window statistics do not require the
  current cell to be non-NaN; correlation/regression pairs and event targets
  do.  Backends must not randomly apply ``.where(current.notna())``.
* ``TimeUnit`` (R19-084..086): days_since / age / duration / spacing /
  count operators declare their unit (trading bars / calendar days / session
  slots / events / reports).  A name that claims ``days`` but counts rows must
  be declared, not guessed.
* Rank/quantile semantics (R19-088..093): the 0-1-with-singleton-0.5 family
  and the 1/n..1 family are different canonicals; quantile ``p`` is strict
  ``[0, 1]`` and interpolation is fixed into the declaration.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# R19-075..077: anchor policy for expanding / cumulative (sample-start) ops
# ---------------------------------------------------------------------------
class AnchorPolicy(str, enum.Enum):
    """Where a cumulative/expanding operator's accumulation starts.

    The OUTPUT VALUE of an expanding/cumulative operator depends on the first
    observation of the series it sees.  The anchor declares WHICH start is the
    semantic one so the same factor definition produces the same output
    regardless of how the evaluation window is sliced.

    * ``listing_start``          — the instrument's first listed bar.
    * ``fixed_global_start``     — a fixed global calendar date.
    * ``campaign_start``         — the research/campaign start.  FORBIDDEN as a
      default (R19-077): an evaluation window must not silently become the
      factor's history anchor.
    * ``full_available_history`` — the full history available in the dataset.
    """

    LISTING_START = "listing_start"
    FIXED_GLOBAL_START = "fixed_global_start"
    CAMPAIGN_START = "campaign_start"
    FULL_AVAILABLE_HISTORY = "full_available_history"


# Every expanding_* / cum_* / cumulative_* canonical.  Their running values
# accumulate over the full available per-series history, so the honest anchor is
# ``full_available_history`` — never the evaluation/campaign start.
_EXPANDING_CUMULATIVE_CANONICALS: frozenset[str] = frozenset({
    "expanding_mean", "expanding_std", "expanding_rank", "expanding_zscore",
    "expanding_max", "expanding_min", "expanding_sum",
    "cum_avg", "cum_count", "cum_delta", "cum_first", "cum_last",
    "cum_max", "cum_min", "cum_positive_streak", "cum_prod", "cum_rank",
    "cum_std", "cum_sum", "cum_top_n_avg", "cum_top_n_sum",
    "cumulative_max", "cumulative_mean", "cumulative_min",
    "cumulative_returns",
})

_ANCHOR_POLICIES: dict[str, AnchorPolicy] = {
    canon: AnchorPolicy.FULL_AVAILABLE_HISTORY
    for canon in _EXPANDING_CUMULATIVE_CANONICALS
}

# Canonicals that declare a campaign-scoped anchor.  MUST stay empty: R19-077 —
# automatic mining must not let the campaign start date implicitly redefine a
# factor.  Kept as a machine-readable assertion the audits/tests can check.
_CAMPAIGN_ANCHORED_CANONICALS: frozenset[str] = frozenset()


def anchor_policy(canonical: str) -> AnchorPolicy | None:
    """Return the declared ``AnchorPolicy`` for ``canonical`` (``None`` when the
    operator is not sample-start-dependent and declares nothing)."""
    return _ANCHOR_POLICIES.get(canonical)


def declared_anchor_policies() -> dict[str, str]:
    """Snapshot of anchor declarations (canonical -> policy value)."""
    return {k: v.value for k, v in _ANCHOR_POLICIES.items()}


def anchor_belongs_to_campaign(canonical: str) -> bool:
    """R19-077: True when a canonical's anchor is campaign-scoped — the research
    window became the factor's history anchor, which must never happen for an
    automatically-mined factor.  The default declaration set never does."""
    return canonical in _CAMPAIGN_ANCHORED_CANONICALS


# ---------------------------------------------------------------------------
# R19-082/083: missing-value topology
# ---------------------------------------------------------------------------
class MissingTopologyPolicy(str, enum.Enum):
    """How an operator treats NaN/Inf gaps inside its input window.

    * ``skip_finite``            — drop non-finite cells, compute on the finite
      remainder (e.g. ``ts_mean``).
    * ``preserve_physical_lag``  — lag is PHYSICAL-ROW lag: ``lag=1`` reads the
      previous trading bar, NOT the previous non-missing observation.  A missing
      bar never bridges the two sides into neighbours.
    * ``break_episode``          — a missing observation ends the current
      episode/state (streaks, CUSUM state, directional state).
    * ``require_contiguous``     — the operator needs a trailing contiguous
      finite run (no drop-finite / re-connect).
    * ``pairwise_finite``        — pairwise deletion across multiple inputs
      (correlation/regression: a pair with either side non-finite is dropped).
    """

    SKIP_FINITE = "skip_finite"
    PRESERVE_PHYSICAL_LAG = "preserve_physical_lag"
    BREAK_EPISODE = "break_episode"
    REQUIRE_CONTIGUOUS = "require_contiguous"
    PAIRWISE_FINITE = "pairwise_finite"


# Lag / delay family: physical-row lag, never "previous non-missing".
_LAG_CANONICALS: frozenset[str] = frozenset({
    "ts_delay", "delay", "ts_delta", "deltas", "ts_pct", "ts_log_return",
    "MOM", "ROC", "prev", "ts_ratio", "cross_event", "cum_delta",
    "ts_lag_of_peak_corr",
})

# Autocorrelation: physical-row lag; a gap between the two compared rows must
# not be bridged into a new neighbour pair.
_AUTOCORR_CANONICALS: frozenset[str] = frozenset({
    "ts_autocorr", "volume_autocorr", "turnover_autocorr",
})

# Episode / state accumulators: a missing observation breaks the episode.
_EPISODE_CANONICALS: frozenset[str] = frozenset({
    "ts_cusum_pressure", "ts_cumulative_deviation_score",
    "ts_level_shift_score", "ts_vol_shift_score",
    "directional_change_state", "directional_change_extent",
    "state_since_sum", "state_since_mean", "state_since_count",
    "state_since_last", "state_since_trend_tstat",
    "state_episode_mfe", "state_episode_mae", "state_episode_efficiency",
    "state_episode_retrace_ratio", "state_episode_excursion_balance",
    "ts_positive_streak", "ts_negative_streak", "ts_max_buildup",
    "digital_count", "cum_positive_streak", "ts_time_since_change",
    "ts_threshold_cycle_period", "ts_threshold_cycle_asymmetry",
    "ts_interval_nesting_depth", "state_latch", "state_hold",
    "state_slew_limit", "state_deadband", "state_ewm_if", "event_refractory",
})

# Correlation / regression: pairwise finite across the (x, y) pair.
_PAIRWISE_CANONICALS: frozenset[str] = frozenset({
    "ts_corr", "ts_cov", "ts_beta", "rank_corr", "ts_regression",
    "ts_poly2_coeff", "ts_poly2_resid", "residual_momentum_capm",
    "tail_beta", "coskewness_to_market", "idio_skew", "idio_vol",
})

# Trailing-contiguous requirement (drop-finite / re-connect is wrong).
_CONTIGUOUS_CANONICALS: frozenset[str] = frozenset({
    "ts_cumulative_deviation_score", "ts_level_shift_score",
    "ts_vol_shift_score",
})

_MISSING_TOPOLOGY_POLICIES: dict[str, MissingTopologyPolicy] = {
    canon: MissingTopologyPolicy.PRESERVE_PHYSICAL_LAG for canon in _LAG_CANONICALS
}
for _canon in _AUTOCORR_CANONICALS:
    _MISSING_TOPOLOGY_POLICIES[_canon] = MissingTopologyPolicy.PRESERVE_PHYSICAL_LAG
for _canon in _EPISODE_CANONICALS:
    _MISSING_TOPOLOGY_POLICIES.setdefault(_canon, MissingTopologyPolicy.BREAK_EPISODE)
for _canon in _PAIRWISE_CANONICALS:
    _MISSING_TOPOLOGY_POLICIES[_canon] = MissingTopologyPolicy.PAIRWISE_FINITE
for _canon in _CONTIGUOUS_CANONICALS:
    _MISSING_TOPOLOGY_POLICIES[_canon] = MissingTopologyPolicy.REQUIRE_CONTIGUOUS


def missing_topology_policy(canonical: str) -> MissingTopologyPolicy:
    """Return the declared missing-topology policy for ``canonical``.  Defaults
    to ``SKIP_FINITE`` for undeclared operators (plain window statistics)."""
    return _MISSING_TOPOLOGY_POLICIES.get(canonical, MissingTopologyPolicy.SKIP_FINITE)


def declared_missing_topology_policies() -> dict[str, str]:
    """Snapshot of missing-topology declarations."""
    return {k: v.value for k, v in _MISSING_TOPOLOGY_POLICIES.items()}


# ---------------------------------------------------------------------------
# R19-087: current-row requirement
# ---------------------------------------------------------------------------
class CurrentRowRequirement(str, enum.Enum):
    """Whether the current row's (non-finite) value gates this operator's output
    at time ``t``.  Backends must not randomly apply ``.where(current.notna())``
    — the requirement is declared, not guessed.

    * ``not_required_for_window_stat`` — the window statistic (mean/std/…)
      emits at ``t`` from the window's finite samples; the current cell being
      non-finite does not suppress it.
    * ``required_as_target``           — the output is an error/forecast vs the
      actual current value, so the current cell must be finite.
    * ``required_as_pair``             — the output needs the current (x, y)
      pair finite (correlation/regression).
    * ``required_as_event``            — the output is anchored on an event at
      the current bar.
    """

    NOT_REQUIRED_FOR_WINDOW_STAT = "not_required_for_window_stat"
    REQUIRED_AS_TARGET = "required_as_target"
    REQUIRED_AS_PAIR = "required_as_pair"
    REQUIRED_AS_EVENT = "required_as_event"


_WINDOW_STAT_CANONICALS: frozenset[str] = frozenset({
    "ts_mean", "ts_std", "ts_var", "ts_sum", "ts_median", "ts_max", "ts_min",
    "ts_product", "ts_mad", "ts_skew", "ts_kurt", "ts_zscore", "ts_sharpe",
    "expanding_mean", "expanding_std", "expanding_zscore", "expanding_max",
    "expanding_min", "expanding_sum", "cum_avg", "cum_std", "cumulative_mean",
    "cum_sum", "cum_max", "cum_min", "cumulative_max", "cumulative_min",
    "cum_prod", "ts_decay_linear", "ts_sum_decay", "WMA", "ts_ema", "EMA",
})

_PAIR_CURRENT_ROW_CANONICALS: frozenset[str] = frozenset({
    "ts_corr", "ts_cov", "ts_beta", "rank_corr", "ts_autocorr",
    "ts_regression", "ts_poly2_coeff", "ts_poly2_resid",
    "residual_momentum_capm", "tail_beta", "coskewness_to_market",
    "idio_skew", "idio_vol",
})

_EVENT_CURRENT_ROW_CANONICALS: frozenset[str] = frozenset({
    "event_historical_response_mean", "event_historical_response_sign_balance",
    "event_refractory", "event_decay_asof", "directional_change_state",
    "directional_change_extent", "ts_cusum_pressure",
    "ts_cumulative_deviation_score", "ts_time_since_change",
})

_CURRENT_ROW_REQUIREMENTS: dict[str, CurrentRowRequirement] = {
    canon: CurrentRowRequirement.NOT_REQUIRED_FOR_WINDOW_STAT
    for canon in _WINDOW_STAT_CANONICALS
}
for _canon in _PAIR_CURRENT_ROW_CANONICALS:
    _CURRENT_ROW_REQUIREMENTS[_canon] = CurrentRowRequirement.REQUIRED_AS_PAIR
for _canon in _EVENT_CURRENT_ROW_CANONICALS:
    _CURRENT_ROW_REQUIREMENTS[_canon] = CurrentRowRequirement.REQUIRED_AS_EVENT
_CURRENT_ROW_REQUIREMENTS["ts_regression_forecast_error"] = (
    CurrentRowRequirement.REQUIRED_AS_TARGET
)


def current_row_requirement(canonical: str) -> CurrentRowRequirement | None:
    """Return the declared current-row requirement for ``canonical`` (``None``
    when undeclared — the operator does not gate on the current cell)."""
    return _CURRENT_ROW_REQUIREMENTS.get(canonical)


def declared_current_row_requirements() -> dict[str, str]:
    """Snapshot of current-row requirement declarations."""
    return {k: v.value for k, v in _CURRENT_ROW_REQUIREMENTS.items()}


# ---------------------------------------------------------------------------
# R19-084..086: explicit output time units
# ---------------------------------------------------------------------------
class TimeUnit(str, enum.Enum):
    """Explicit unit for days_since / age / duration / spacing / count outputs.

    * ``trading_bars``     — number of trading bars (physical rows).
    * ``calendar_days``    — wall-clock calendar days.
    * ``session_slots``    — intraday session slots (A-share lunch break
      11:30→13:01 is NOT a valid slot; never wall-clock minute difference).
    * ``events``           — number of event observations.
    * ``reports``          — number of report/fiscal periods.
    * ``wall_clock_minutes`` — explicit wall-clock minutes (only when the
      canonical declares it — never the implicit default for session work).
    """

    TRADING_BARS = "trading_bars"
    CALENDAR_DAYS = "calendar_days"
    SESSION_SLOTS = "session_slots"
    EVENTS = "events"
    REPORTS = "reports"
    WALL_CLOCK_MINUTES = "wall_clock_minutes"


_TIME_UNITS: dict[str, TimeUnit] = {
    # Pivot age / spacing: measured in trading bars (physical rows).
    "ts_pivot_high_age": TimeUnit.TRADING_BARS,
    "ts_pivot_low_age": TimeUnit.TRADING_BARS,
    "ts_nth_pivot_high_age": TimeUnit.TRADING_BARS,
    "ts_nth_pivot_low_age": TimeUnit.TRADING_BARS,
    "ts_pivot_high_spacing": TimeUnit.TRADING_BARS,
    "ts_pivot_low_spacing": TimeUnit.TRADING_BARS,
    "ts_swing_duration": TimeUnit.TRADING_BARS,
    "ts_threshold_cycle_period": TimeUnit.TRADING_BARS,
    "ts_interval_nesting_depth": TimeUnit.TRADING_BARS,
    # Count operators: events / observations.
    "ts_pivot_high_count": TimeUnit.EVENTS,
    "ts_pivot_low_count": TimeUnit.EVENTS,
    "cum_count": TimeUnit.EVENTS,
    "count": TimeUnit.EVENTS,
    "c_count": TimeUnit.EVENTS,
    "row_count": TimeUnit.EVENTS,
    "group_count": TimeUnit.EVENTS,
    "digital_count": TimeUnit.EVENTS,
    "ts_positive_streak": TimeUnit.EVENTS,
    "ts_negative_streak": TimeUnit.EVENTS,
    "cum_positive_streak": TimeUnit.EVENTS,
    # Financial / report-period operators: measured in report periods.
    "fin_lag": TimeUnit.REPORTS,
    "fin_diff": TimeUnit.REPORTS,
    "fin_pct_change": TimeUnit.REPORTS,
    "fin_log_change": TimeUnit.REPORTS,
    "fin_qoq": TimeUnit.REPORTS,
    "fin_yoy": TimeUnit.REPORTS,
    "fin_ttm": TimeUnit.REPORTS,
}

# Names that literally claim ``days``/``age``/``duration`` but are counted in
# physical rows (a rename/declaration obligation per R19-084..086).  The
# registry currently contains no ``days_since*`` canonical, so this is the
# audit's machine-readable empty-set assertion.
_DAYS_NAMED_ROW_COUNT: frozenset[str] = frozenset()


def time_unit(canonical: str) -> TimeUnit | None:
    """Return the declared output ``TimeUnit`` for ``canonical`` (``None`` when
    the operator's output is not time-quantified)."""
    return _TIME_UNITS.get(canonical)


def days_named_row_count_canonicals() -> frozenset[str]:
    """R19-084..086: canonicals whose NAME claims a time unit (``days``/``age``/
    ``duration``) but whose OUTPUT is a physical-row count — they must be
    renamed or explicitly declared.  Empty by default in this registry."""
    return _DAYS_NAMED_ROW_COUNT


def declared_time_units() -> dict[str, str]:
    """Snapshot of time-unit declarations."""
    return {k: v.value for k, v in _TIME_UNITS.items()}


# ---------------------------------------------------------------------------
# R19-088..093: rank / quantile family unification
# ---------------------------------------------------------------------------
class RankSemantic(str, enum.Enum):
    """The rank output family.  Two families that LOOK alike must not drift:
    ``(rank-1)/(n-1)`` with singleton→0.5 and ``rank/n`` with smallest→1/n are
    DIFFERENT canonical semantics (R19-088).  Tie methods are part of the
    declaration, never a per-backend choice."""

    # (rank - 1) / (n - 1), ties -> average, singleton -> 0.5, NaN/Inf excluded.
    NORMALIZED_01_SINGLETON_HALF = "normalized_01_singleton_half"
    # rank(pct=True) -> rank / n, smallest -> 1/n, largest -> 1.0.
    PCT_RANK_COUNT = "pct_rank_count"
    # raw integer ranks 1..n, ties -> average.
    RAW_RANK_AVERAGE = "raw_rank_average"
    # per-window (rolling) rank inside a trailing window (ts_rank).
    ROLLING_WINDOW = "rolling_window"
    # expanding rank over the series start.
    EXPANDING = "expanding"
    # group-normalised rank (0-1) within a group cross-section.
    GROUP = "group"
    # quantile-threshold membership / percentile value.
    PERCENTILE = "percentile"
    # matrix rank (a rank of the matrix, not a rank transform).
    MATRIX = "matrix"


# ``rank`` / ``cs_rank`` / ``c_rank`` resolve to ``cs_rank_01`` semantics.
_RANK_SEMANTICS: dict[str, RankSemantic] = {
    "cs_rank_01": RankSemantic.NORMALIZED_01_SINGLETON_HALF,
    "rank": RankSemantic.NORMALIZED_01_SINGLETON_HALF,
    "rank_pct": RankSemantic.PCT_RANK_COUNT,
    "cs_pct_rank": RankSemantic.PCT_RANK_COUNT,
    "panel_rank": RankSemantic.PCT_RANK_COUNT,
    "rank_transform": RankSemantic.PCT_RANK_COUNT,
    "rankavg_transform": RankSemantic.RAW_RANK_AVERAGE,
    "ts_rank": RankSemantic.ROLLING_WINDOW,
    "cum_rank": RankSemantic.EXPANDING,
    "expanding_rank": RankSemantic.EXPANDING,
    "group_rank": RankSemantic.GROUP,
    "group_rank_weighted_value": RankSemantic.GROUP,
    "group_percentile": RankSemantic.PERCENTILE,
    "c_percentile": RankSemantic.PERCENTILE,
    "mat_rank": RankSemantic.MATRIX,
}


def rank_semantics(canonical: str) -> RankSemantic | None:
    """Return the declared ``RankSemantic`` for ``canonical`` (``None`` when the
    operator is not a rank-family operator)."""
    return _RANK_SEMANTICS.get(canonical)


@dataclass(frozen=True)
class QuantilePolicy:
    """Fixed quantile contract: ``p`` is strict ``[p_min, p_max]``, the
    interpolation method is part of the identity (linear/nearest/lower/higher/
    midpoint) and the finite-sample policy is consistent across backends.

    ``bins`` is used by the ``quantile`` qcut family (a fixed bin count rather
    than a probability ``p``).
    """

    p_min: float = 0.0
    p_max: float = 1.0
    interpolation: str = "linear"  # linear | nearest | lower | higher | midpoint
    finite_policy: str = "drop_finite"  # drop_finite | pairwise | contiguous
    bins: int | None = None  # qcut bin count (``quantile`` family)


_QUANTILE_POLICIES: dict[str, QuantilePolicy] = {
    # qcut-binned quantile: labels 0..bins-1, no probability p.
    "quantile": QuantilePolicy(interpolation="linear", finite_policy="drop_finite", bins=10),
    # Cross-sectional quantile threshold: p in [0,1], linear.
    "cs_quantile": QuantilePolicy(p_min=0.0, p_max=1.0, interpolation="linear",
                                  finite_policy="drop_finite"),
    "c_percentile": QuantilePolicy(p_min=0.0, p_max=1.0, interpolation="linear",
                                   finite_policy="drop_finite"),
    # Rolling time-series quantile: p in [0,1], linear.
    "ts_quantile": QuantilePolicy(p_min=0.0, p_max=1.0, interpolation="linear",
                                  finite_policy="drop_finite"),
    # Group membership percentile: p in [0,1], boolean output.
    "group_percentile": QuantilePolicy(p_min=0.0, p_max=1.0, interpolation="linear",
                                       finite_policy="drop_finite"),
    # Distribution quantiles.
    "quantile_normal": QuantilePolicy(p_min=0.0, p_max=1.0, interpolation="linear",
                                      finite_policy="drop_finite"),
    "quantile_t": QuantilePolicy(p_min=0.0, p_max=1.0, interpolation="linear",
                                 finite_policy="drop_finite"),
}


def quantile_policy(canonical: str) -> QuantilePolicy | None:
    """Return the fixed ``QuantilePolicy`` for a quantile-family canonical
    (``None`` when the operator is not a quantile operator)."""
    return _QUANTILE_POLICIES.get(canonical)


def rank_quantile_declarations() -> dict[str, Any]:
    """Machine-readable snapshot of rank + quantile declarations (for audits /
    identity hashing)."""
    out: dict[str, Any] = {
        "rank": {k: v.value for k, v in _RANK_SEMANTICS.items()},
        "quantile": {k: v.__dict__ for k, v in _QUANTILE_POLICIES.items()},
    }
    return out


# ---------------------------------------------------------------------------
# combined snapshot for audits
# ---------------------------------------------------------------------------
def all_policy_declarations() -> dict[str, Any]:
    """Full machine-readable snapshot of every R19 semantic policy."""
    return {
        "anchor": declared_anchor_policies(),
        "missing_topology": declared_missing_topology_policies(),
        "current_row": declared_current_row_requirements(),
        "time_units": declared_time_units(),
        "rank_quantile": rank_quantile_declarations(),
    }


__all__ = [
    "AnchorPolicy",
    "MissingTopologyPolicy",
    "CurrentRowRequirement",
    "TimeUnit",
    "RankSemantic",
    "QuantilePolicy",
    "anchor_policy",
    "declared_anchor_policies",
    "anchor_belongs_to_campaign",
    "missing_topology_policy",
    "declared_missing_topology_policies",
    "current_row_requirement",
    "declared_current_row_requirements",
    "time_unit",
    "days_named_row_count_canonicals",
    "declared_time_units",
    "rank_semantics",
    "quantile_policy",
    "rank_quantile_declarations",
    "all_policy_declarations",
]

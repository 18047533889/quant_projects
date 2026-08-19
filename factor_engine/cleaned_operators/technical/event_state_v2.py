# -*- coding: utf-8 -*-
"""Event→Alpha and State→Alpha derivation layers (R20-EVENT-STATE-ALPHA).

Generic causal dimensionless canonicals that turn event-count and state-level
inputs into terminal-usable Direct Alpha signals.  Two input families:

* EVENT family — a strict EventBool panel (``1`` = event, ``0`` = confirmed
  no-event, ``NaN`` = unknown row).  Any other value (``-1``, ``0.2``, ``2``,
  ``±Inf``) is out-of-domain and raises ``ValueError`` at the call boundary —
  a mis-labelled mark must fail loudly, never invent an event (same policy as
  ``cleaned_operators/event_interval.py::_event_mask``; ±Inf additionally
  rejected here, matching ``strict_params`` "Inf is never a truthy").
* STATE family — a numeric state panel (integer or categorical code); any
  finite value is a legal state code, ``NaN`` is a missing row, ``±Inf`` is
  out-of-domain and raises ``ValueError``.

Duplicate audit (survey BEFORE implementing — documented skips / neighbors):
* ``event_frequency`` (alpha_language_events.py) is hits / finite-observations
  with skip-NaN ``min_periods`` semantics and treats ANY nonzero as an event —
  not fail-closed EventBool, not [0,1] events-per-bar under min_periods=window.
* ``ts_dc_event_rate`` (directional_change.py) is a DC-domain event rate over
  finite bars — different event definition, different domain.
* ``ts_event_spacing_mean/cv`` (common/polars_state_event.py) are gap mean /
  CV — NOT a z-score of the elapsed time since the last event.
* ``event_fano_excess`` (event_interval.py) is block-count Fano F-1
  (variance/mean dispersion over blocks) — ``event_cluster_score`` below is a
  count-vs-Poisson-expectation z-score under a trailing rate, not a
  block-variance statistic.
* ``ts_markov_persistence`` (markov_dynamics.py) is an estimated transition
  matrix's P_kk — ``state_dwell_pct`` below is the raw empirical dwell
  fraction of the CURRENT state, no Markov estimation.
* ``ts_markov_transition_surprisal`` (markov_dynamics.py) is -log P_ij from an
  estimated transition matrix — ``state_transition_surprise`` below is the
  observed transition COUNT vs a uniform-iid null expectation, no matrix.
* ``ts_state_age_percentile`` / ``ts_state_exit_hazard`` /
  ``ts_state_residual_life`` (stateful/survival.py) are episode-survival
  estimators over state episodes — distinct from dwell fraction and from
  transition-count surprise.
* ``ts_state_integral`` / ``ts_transition_intensity`` (alpha_language_state.py)
  are intensity-weighted accumulations — level-unit outputs, not the
  dimensionless persistence / surprise measures landed here.
No canonical name below existed anywhere in the tree before this module.

Family contract (all five): trailing windows END at t (prefix-causal; row r
uses rows <= r only); min_periods=window — any NaN inside the window makes the
output NaN (fail-closed, never zero-filled); degenerate denominators -> NaN
never 0; all outputs dimensionless; rolling-only bounded state (NOT in any
recursive/EWM governance set); deliberate documented all-NaN scenarios
(e.g. ``event_recency_z`` with fewer than 3 events in the window, or
``event_cluster_score`` when the trailing rate window contains zero events).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12


def _pi(v, name, minimum=1):
    if isinstance(v, bool):
        raise ValueError(f"{name} must be integer")
    v = int(v)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return v


def _frame(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _strict_event_bool(event: pd.DataFrame) -> np.ndarray:
    """Strict EventBool panel: legal set {0, 1, NaN}; anything else -> ValueError.

    ``NaN`` is an *unknown* row (fail-closed window semantics downstream);
    every other value — including ``-1``, ``0.2``, ``2`` and ``±Inf`` — is
    out-of-domain and rejected loudly instead of being silently interpreted.
    """
    try:
        arr = np.asarray(event.to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "event panel must be a numeric EventBool {0, 1, NaN} frame"
        ) from exc
    legal = np.isnan(arr) | (arr == 0.0) | (arr == 1.0)
    if not bool(np.all(legal)):
        bad = np.unique(arr[~legal])[:5]
        raise ValueError(
            "event panel must be strict EventBool {0, 1, NaN}; got "
            f"out-of-domain value(s) {bad!r}"
        )
    return arr


def _state_panel(state: pd.DataFrame) -> np.ndarray:
    """Numeric state panel: finite state codes or NaN (missing); ±Inf rejected."""
    try:
        arr = np.asarray(state.to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "state panel must be a numeric frame of state codes (NaN = missing)"
        ) from exc
    if not bool(np.all(np.isnan(arr) | np.isfinite(arr))):
        raise ValueError(
            "state panel must contain finite state codes or NaN; ±Inf is "
            "out-of-domain"
        )
    return arr


def _trailing_map(arr: np.ndarray, window: int, fn) -> np.ndarray:
    """Per-column trailing-window map ending at row r (prefix-causal).

    Rows with an incomplete window (warmup) stay NaN; ``fn`` receives the raw
    full-length window slice (guaranteed window rows, possibly containing NaN
    — each kernel applies its own fail-closed policy).
    """
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - window + 1)
            if r - lo + 1 < window:
                continue
            out[r, c] = fn(arr[lo : r + 1, c])
    return out


def event_rate_pct(event, window):
    """Trailing event count per bar, normalized by window — events/bar in [0,1]
    for a 0/1 indicator (dimensionless event intensity).

    Trailing window ENDS at t, min_periods=window: any NaN (unknown row)
    inside the window -> NaN (fail-closed, never skipped or zero-filled).
    Input must be strict EventBool {0, 1, NaN}; out-of-domain values raise
    ValueError.  Window >= 2 (a single-bar window is the identity indicator —
    a trivial search node, pruned up front)."""
    w = _pi(window, "window", 2)
    arr = _strict_event_bool(event)

    def _rate(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        return float(np.count_nonzero(chunk == 1.0)) / float(w)

    return _frame(event, _trailing_map(arr, w, _rate))


def event_recency_z(event, window):
    """Bars-since-last-event standardized against the trailing distribution of
    inter-arrival gaps (dimensionless "overdue event" z-score).

    At row t: recency = t - (position of the last event), and the gaps are the
    differences between consecutive event positions FULLY INSIDE the trailing
    window [t-w+1, t] (the boundary gap from a pre-window event is excluded —
    the operator's history is exactly the window).  z = (recency - mean(gaps))
    / std(gaps, ddof=1).

    Deliberate documented NaN scenarios: any NaN in the window (fail-closed);
    fewer than 3 events in the window (fewer than 2 complete gaps -> no ddof=1
    std — including the no-event-ever-in-window case); zero gap dispersion
    (perfectly periodic events — degenerate denominator, never 0).  Semantics:
    the current gap is right-censored, so z ~ 0 means the elapsed time matches
    a typical COMPLETE gap and positive z signals an overdue event.  Window
    >= 3 (fewer than 3 rows can never hold 3 events — an all-NaN search node,
    pruned up front)."""
    w = _pi(window, "window", 3)
    arr = _strict_event_bool(event)

    def _recency(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = np.flatnonzero(chunk == 1.0)
        if pos.size < 3:
            return np.nan
        gaps = np.diff(pos).astype(float)
        sd = float(np.std(gaps, ddof=1))
        if not np.isfinite(sd) or sd <= 0.0:
            return np.nan
        recency = float(chunk.size - 1 - int(pos[-1]))
        return (recency - float(np.mean(gaps))) / sd

    return _frame(event, _trailing_map(arr, w, _recency))


def event_cluster_score(event, window, rate_window):
    """Trailing event count vs Poisson-expected count under a trailing rate —
    dimensionless count-dispersion z-score (burstiness vs the event's own
    recent rate).

    At row t: K = event count in the count window [t-window+1, t];
    lambda = event count in the LONGER trailing rate window
    [t-rate_window+1, t] divided by rate_window (the trailing rate estimate —
    it includes the count window; documented overlap); expected count
    E = lambda * window; score = (K - E) / sqrt(E) (Poisson z).

    Deliberate documented NaN scenarios: any NaN inside the rate window
    (fail-closed — the rate window strictly contains the count window, so one
    check covers both); zero events in the rate window (lambda = 0 -> E = 0,
    degenerate denominator, never 0).  Requires rate_window > window
    (relational spec + runtime ValueError).  Distinct from event_fano_excess
    (block-variance F-1): this is a count-vs-expectation z-score, not a
    block-dispersion statistic."""
    w = _pi(window, "window", 2)
    rw = _pi(rate_window, "rate_window", 2)
    if rw <= w:
        raise ValueError(
            f"rate_window must be > window (window={w}, rate_window={rw})"
        )
    arr = _strict_event_bool(event)
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - rw + 1)
            if r - lo + 1 < rw:
                continue
            rate_chunk = arr[lo : r + 1, c]
            if np.any(np.isnan(rate_chunk)):
                continue
            count_chunk = arr[r - w + 1 : r + 1, c]
            k = float(np.count_nonzero(count_chunk == 1.0))
            lam = float(np.count_nonzero(rate_chunk == 1.0)) / float(rw)
            expect = lam * float(w)
            if not np.isfinite(expect) or expect <= _EPS:
                continue
            out[r, c] = (k - expect) / float(np.sqrt(expect))
    return _frame(event, out)


def state_dwell_pct(state, window):
    """Fraction of the trailing window spent in the CURRENT state — regime
    persistence in (0, 1] (dimensionless).

    At row t: dwell = count(state == state_t over [t-window+1, t]) / window.
    A NaN anywhere in the window (or a NaN current state) -> NaN (fail-closed);
    a constant window gives exactly 1.0; perfectly alternating states give 0.5
    on an even window.  Distinct from ts_markov_persistence: raw empirical
    dwell of the current state, no transition-matrix estimation.  Window >= 2
    (a single-bar window is identically 1.0 — a trivial search node, pruned
    up front)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _dwell(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        cur = chunk[-1]
        return float(np.count_nonzero(chunk == cur)) / float(w)

    return _frame(state, _trailing_map(arr, w, _dwell))


def state_transition_surprise(state, window):
    """Observed trailing transition count vs uniform-prior expectation —
    dimensionless regime-churn surprise (z-score).

    At row t over [t-window+1, t]: T = count of adjacent pairs with
    state_i != state_{i-1}; m = number of DISTINCT states observed in the
    window (trailing-window counts only — an unseen state simply never enters
    m, so there is no look-ahead and no support declaration needed).  Under
    the uniform-iid null (each bar drawn uniformly from the m observed
    states): p = 1 - 1/m, E[T] = (window-1) * p, Var[T] = (window-1) * p *
    (1-p); surprise = (T - E) / sqrt(Var).

    Deliberate documented NaN scenarios: any NaN in the window (fail-closed);
    a single distinct state in the window (m = 1 -> p = 0 -> degenerate null,
    NaN — never a fabricated 0).  Distinct from ts_markov_transition_surprisal
    (-log P_ij from an estimated Markov matrix): count-vs-uniform-null, no
    matrix estimation.  Window >= 2 (fewer than 2 rows hold no pair)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _surprise(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        m = float(np.unique(chunk).size)
        if m <= 1.0:
            return np.nan
        pairs = float(chunk.size - 1)
        p = 1.0 - 1.0 / m
        var = pairs * p * (1.0 - p)
        if not np.isfinite(var) or var <= _EPS:
            return np.nan
        transitions = float(np.count_nonzero(np.diff(chunk) != 0.0))
        expect = pairs * p
        return (transitions - expect) / float(np.sqrt(var))

    return _frame(state, _trailing_map(arr, w, _surprise))


# ---------------------------------------------------------------------------
# Registration (indicators_v2 pattern: spec-driven table + single loop).
# ---------------------------------------------------------------------------
_WIN_GE2 = ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON)
_WIN_GE3 = ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON)

_CLUSTER_REL = [
    RelationalParamSpec(
        expression="window < rate_window",
        message="require window < rate_window (window={window}, rate_window={rate_window})",
    )
]

_SPECS = [
    ("event_rate_pct", ["event", "window"], event_rate_pct,
     "Trailing event count per bar normalized by window; events/bar in [0,1] for a 0/1 indicator; strict EventBool {0,1,NaN} input, fail-closed NaN-in-window."),
    ("event_recency_z", ["event", "window"], event_recency_z,
     "Bars-since-last-event z-scored against the trailing inter-arrival gap distribution; dimensionless overdue-event signal; <3 events in window -> NaN (documented)."),
    ("event_cluster_score", ["event", "window", "rate_window"], event_cluster_score,
     "Trailing event count vs Poisson-expected count under a longer trailing rate window; dimensionless count-dispersion z-score; rate_window > window."),
    ("state_dwell_pct", ["state", "window"], state_dwell_pct,
     "Fraction of the trailing window spent in the current state; regime persistence in (0,1]; fail-closed NaN-in-window."),
    ("state_transition_surprise", ["state", "window"], state_transition_surprise,
     "Observed trailing transition count vs uniform-iid expectation over the window's distinct states; dimensionless regime-churn z-score; single-state window -> NaN."),
]

_PARAM_SPECS = {
    "event_rate_pct": {"window": _WIN_GE2},
    "event_recency_z": {"window": _WIN_GE3},
    "event_cluster_score": {"window": _WIN_GE2, "rate_window": _WIN_GE2},
    "state_dwell_pct": {"window": _WIN_GE2},
    "state_transition_surprise": {"window": _WIN_GE2},
}

_RELATIONAL_SPECS = {
    "event_cluster_score": _CLUSTER_REL,
}


def _register(name, params, fn, desc, *, param_specs=None, relational_specs=None):
    meta = OperatorMetadata(
        name=name,
        category="technical_signal",
        description=desc,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "production_extension"],
        param_specs=dict(param_specs or {}),
        relational_specs=list(relational_specs or []),
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"EventStateV2_{name}",
        (SeriesOperator,),
        {"metadata": meta, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="technical_signal",
        business_category="technical",
        canonical=name,
        source="technical_event_state_v2",
        backend="pandas_numpy",
        status="production",
    )(cls)


for _name, _params, _fn, _desc in _SPECS:
    _register(
        _name,
        _params,
        _fn,
        _desc,
        param_specs=_PARAM_SPECS.get(_name),
        relational_specs=_RELATIONAL_SPECS.get(_name),
    )

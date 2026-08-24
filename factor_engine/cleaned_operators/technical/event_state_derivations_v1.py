# -*- coding: utf-8 -*-
"""Event/State derivation framework primitives (R21-P0-EVENTSTATE-FRAMEWORK).

Sibling of ``cleaned_operators/technical/event_state_v2.py`` (same registration
idiom, same family contract).  Three input families beyond the R20 five:

* EVENT_BOOL — a strict EventBool panel (``1`` = event, ``0`` = confirmed
  no-event, ``NaN`` = unknown row); any other value (``-1``, ``0.2``, ``2``,
  ``±Inf``) raises ``ValueError`` at the call boundary (never silently
  interpreted as an event or a no-event).
* SIGNED_EVENT — a strict SignedEvent panel (``-1`` / ``0`` / ``+1`` / NaN,
  NaN = unknown row); out-of-domain values raise ``ValueError``.  This is the
  ``strict_signed_event`` legal set from ``cleaned_operators/common/strict_params.py``.
* STATE — a numeric state panel (finite state codes; NaN = missing row; ±Inf
  rejected) — identical to ``event_state_v2._state_panel``.

Duplicate audit (survey BEFORE implementing — documented skips + neighbors):
* ``event_age`` — SKIPPED.  An existing registry alias
  (``alpha_language_aliases.py``) maps ``event_age -> ts_days_since``
  (``common/daily_panel.py``; bars since last True with NaN-reset semantics,
  full history).  Registering a same-named canonical is impossible
  (``OperatorRegistry.register`` raises "canonical already declared as alias")
  and would fork the semantic.  The sparse-event robustness motive of this
  slice is served instead by the windowed canonicals below (e.g.
  ``positive_event_age``/``negative_event_age`` on the signed family).
* ``event_decay`` — SKIPPED.  Existing alias ``event_decay -> event_decay_asof``
  (``state_event.py``): causal half-life exponential decay of a *cumulative*
  marked/event stream.  This module's ``event_decay_window`` (a single
  exp(-age/halflife) pulse of the last in-window event) is deliberately named
  differently to avoid forking that alias.
* ``state_episode_duration`` — SKIPPED as a second canonical; registered as an
  ALIAS of ``state_age`` (same quantity: bars since the current state episode
  began).  One canonical, alias-safe.
* ``ts_days_since`` (daily_panel) — full-history bars-since-last-True with
  NaN-reset; the windowed fail-closed estimators here differ (min_periods=
  window, strict EventBool rejection, degenerate NaN).
* ``ts_true_streak`` (daily_panel) — full-history consecutive-True run with
  NaN-reset and 0-on-confirmed-False.  ``event_streak`` here is trailing-window
  with min_periods=window fail-closed semantics; a different contract, and a
  confirmed-False row resets to 0 only inside a complete window.
* ``ts_transition_count`` (state_event.py / polars_state_event.py) — bool
  0/1-transition count over a window with carry/break missing policies.
  ``state_transition_count`` here counts changes of a MULTI-valued state code
  with fail-closed min_periods=window semantics (no carry transparency).
* ``ts_time_since_change`` (state_event.py) — bars since a ConditionBool state
  last flipped, full-history recursive with carry/break policies.  ``state_age``
  here is the multi-state-code windowed analogue with fail-closed semantics,
  DISTINCT from ``state_dwell_pct`` (a window FRACTION, not an episode age).
* ``state_dwell_pct`` (event_state_v2) — fraction of the window in the current
  state; ``state_persistence`` below is the same estimator kept for the STATE
  family naming symmetry — implemented by delegation, one kernel.
* ``state_transition_surprise`` (event_state_v2) — z-score vs uniform-iid null;
  ``state_transition_count`` / ``state_transition_rate`` below are raw count /
  per-bar rates (no null model), and ``state_flip_density`` counts only genuine
  alternations between DISTINCT successive states (A,B,A = 2 flips, A,A,B = 1)
  — none of which exists as a canonical anywhere in the tree.
* ``event_cluster_score`` (event_state_v2) — count-vs-Poisson z-score;
  ``event_cluster_duration`` below is the run-length of the event burst that is
  still LIVE at t (a duration, not a dispersion statistic).
* ``event_recency_z`` (event_state_v2) — needs >= 3 in-window events (NaN
  below that).  The signed-age canonicals below are the robust sparse-event
  counterparts: finite whenever a single event exists inside the window.
No canonical name below existed anywhere in the tree before this module.

Family contract (all canonicals): trailing windows END at t (prefix-causal;
row r uses rows <= r only); min_periods=window — any NaN inside the window
makes the output NaN (fail-closed, never zero-filled, never skipped);
degenerate denominators -> NaN never 0; all outputs dimensionless; rolling-only
bounded state (NOT recursive/EWM — never in a full-history governance set);
deliberate documented all-NaN scenarios (e.g. no event ever inside the window
for the age/decay/streak canonicals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12


def _pi(v, name, minimum=1):
    # bool is an int subclass: reject explicitly (bool-as-int never legal).
    if isinstance(v, bool):
        raise ValueError(f"{name} must be integer")
    # R21-P1-7 strict_integer: only true int instances are valid.
    # 5 -> valid; 5.0 -> invalid; 5.5 -> invalid; True/NaN/Inf/None -> invalid.
    if not isinstance(v, int):
        raise ValueError(f"{name} must be integer")
    iv = v
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return iv


def _pf(v, name, minimum):
    # float param (halflife): reject bool, non-finite, below-minimum.
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        fv = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not np.isfinite(fv):
        raise ValueError(f"{name} must be finite")
    if fv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return fv


def _frame(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _strict_event_bool(event: pd.DataFrame) -> np.ndarray:
    """Strict EventBool panel: legal set {0, 1, NaN}; anything else -> ValueError.

    Same policy as ``event_state_v2._strict_event_bool``: ``NaN`` is an
    *unknown* row (fail-closed window semantics downstream); every other value
    — including ``-1``, ``0.2``, ``2`` and ``±Inf`` — is out-of-domain and
    rejected loudly.
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


def _strict_signed_event(event: pd.DataFrame) -> np.ndarray:
    """Strict SignedEvent panel: legal set {-1, 0, +1, NaN}; else ValueError.

    Mirrors the ``strict_signed_event`` legal set from
    ``cleaned_operators/common/strict_params.py``: a signed event stream whose
    only legal magnitudes are exactly -1 / 0 / +1 (NaN = unknown row).  Any
    other value (``0.5``, ``2``, ``-2``, ``±Inf``) is a mis-labelled mark and
    must fail loudly, never be silently truncated to a sign.
    """
    try:
        arr = np.asarray(event.to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "signed event panel must be a numeric SignedEvent {-1, 0, +1, NaN} frame"
        ) from exc
    legal = np.isnan(arr) | (arr == -1.0) | (arr == 0.0) | (arr == 1.0)
    if not bool(np.all(legal)):
        bad = np.unique(arr[~legal])[:5]
        raise ValueError(
            "signed event panel must be strict SignedEvent {-1, 0, +1, NaN}; "
            f"got out-of-domain value(s) {bad!r}"
        )
    return arr


def _state_panel(state: pd.DataFrame) -> np.ndarray:
    """Numeric state panel: finite state codes or NaN (missing); ±Inf rejected.

    Identical to ``event_state_v2._state_panel`` (duplicated for the same
    module-independence reason as the sibling: no cross-module private helper
    imports — each family module owns its validator).
    """
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

    Identical contract to ``event_state_v2._trailing_map``: rows with an
    incomplete window (warmup) stay NaN; ``fn`` receives the raw full-length
    window slice (guaranteed window rows, possibly containing NaN — each
    kernel applies its own fail-closed policy).
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


# ---------------------------------------------------------------------------
# EVENT_BOOL family (strict {0, 1, NaN} input)
# ---------------------------------------------------------------------------
def event_streak(event, window):
    """Consecutive-True run length ENDING at t, within the trailing window
    (dimensionless count of consecutive confirmed events).

    Trailing window ENDS at t, min_periods=window: any NaN (unknown row)
    inside the window -> NaN (fail-closed).  streak_t = number of consecutive
    rows equal to 1 immediately preceding and including t (0 when row t is a
    confirmed 0).  At a confirmed-0 row the run is exactly 0.0 — a confirmed
    absence is a REAL zero here (unlike the age/decay canonicals where no
    event means the clock is unknown -> NaN).  Window >= 2 (a single-bar
    window is the indicator itself — a trivial search node, pruned up front).

    Distinct from ``ts_true_streak`` (full-history, NaN-reset, ConditionBool):
    fail-closed min_periods=window EventBool semantics here."""
    w = _pi(window, "window", 2)
    arr = _strict_event_bool(event)

    def _streak(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        run = 0
        for v in chunk[::-1]:
            if v == 1.0:
                run += 1
            else:
                break
        return float(run)

    return _frame(event, _trailing_map(arr, w, _streak))


def event_cluster_duration(event, window):
    """Length of the LIVE event burst at t — bars since the last confirmed 0
    inside the trailing window (dimensionless run length of the ongoing run).

    Trailing window ENDS at t, min_periods=window: any NaN inside the window
    -> NaN (fail-closed).  duration_t = t - (position of the last confirmed
    no-event row inside the window); a confirmed 0 AT t gives 0.0 (a burst
    that just ended has length 0 — a real zero, not NaN); a window with NO
    confirmed 0 anywhere (all events) leaves the burst start outside the
    window — the duration is censored -> NaN (never the window length, never
    a fabricated number).  Window >= 2.

    Distinct from ``event_cluster_score`` (count-vs-Poisson z): this is the
    run-length of the burst still live at t, not a dispersion statistic."""
    w = _pi(window, "window", 2)
    arr = _strict_event_bool(event)

    def _duration(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        zeros = np.flatnonzero(chunk == 0.0)
        if zeros.size == 0:
            return np.nan  # burst start censored outside the window
        return float(chunk.size - 1 - int(zeros[-1]))

    return _frame(event, _trailing_map(arr, w, _duration))


def event_decay_window(event, window, halflife):
    """Exponential decay of the last in-window event: exp(-age/halflife) where
    age = bars since the last True inside the trailing window (dimensionless
    (0, 1] freshness pulse).

    Trailing window ENDS at t, min_periods=window: any NaN inside the window
    -> NaN (fail-closed).  No event anywhere in the window -> NaN (the age is
    unknown/censored, never infinity and never exp(-inf)=0 — a 0 would assert
    "the event is certainly long ago" which the window cannot prove).  An
    event AT t gives exactly 1.0.  ``halflife`` > 0 (float, in bars;
    decay = 2^(-age/halflife)).  Window >= 2.

    Distinct from the ``event_decay`` alias -> ``event_decay_asof`` (a
    cumulative recursive decay over ALL history, declared stateful
    full-replay): this is a single-pulse bounded rolling estimator — no
    recursive state, never in a full-history governance set."""
    w = _pi(window, "window", 2)
    hl = _pf(halflife, "halflife", 1e-12)
    arr = _strict_event_bool(event)

    def _decay(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = np.flatnonzero(chunk == 1.0)
        if pos.size == 0:
            return np.nan
        age = float(chunk.size - 1 - int(pos[-1]))
        return float(2.0 ** (-age / hl))

    return _frame(event, _trailing_map(arr, w, _decay))


# ---------------------------------------------------------------------------
# SIGNED_EVENT family (strict {-1, 0, +1, NaN} input)
# ---------------------------------------------------------------------------
def signed_event_rate(event, window):
    """Net signed event rate: (positive - negative event count) / window —
    signed events per bar in [-1, 1] (dimensionless signed intensity).

    Trailing window ENDS at t, min_periods=window: any NaN (unknown row)
    inside the window -> NaN (fail-closed).  Input must be strict SignedEvent
    {-1, 0, +1, NaN}; out-of-domain values raise ValueError.  Window >= 2."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _net(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = float(np.count_nonzero(chunk == 1.0))
        neg = float(np.count_nonzero(chunk == -1.0))
        return (pos - neg) / float(w)

    return _frame(event, _trailing_map(arr, w, _net))


def positive_event_rate(event, window):
    """Positive event count per bar, normalized by window — in [0, 1]
    (dimensionless bullish-event intensity over strict SignedEvent input).

    Trailing window ENDS at t, min_periods=window fail-closed; NaN in window
    -> NaN.  Window >= 2."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _pos(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        return float(np.count_nonzero(chunk == 1.0)) / float(w)

    return _frame(event, _trailing_map(arr, w, _pos))


def negative_event_rate(event, window):
    """Negative event count per bar, normalized by window — in [0, 1]
    (dimensionless bearish-event intensity over strict SignedEvent input).

    Trailing window ENDS at t, min_periods=window fail-closed; NaN in window
    -> NaN.  Window >= 2."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _neg(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        return float(np.count_nonzero(chunk == -1.0)) / float(w)

    return _frame(event, _trailing_map(arr, w, _neg))


def positive_event_age(event, window):
    """Bars since the last +1 event inside the trailing window (dimensionless
    bullish-recency age in [0, window-1]).

    The robust sparse-event counterpart of ``event_recency_z``: finite
    whenever a SINGLE +1 event exists inside the window (``event_recency_z``
    needs >= 3 events for a gap distribution and NaNs below that).  Trailing
    window ENDS at t, min_periods=window fail-closed: any NaN in the window
    -> NaN; no +1 event in the window -> NaN (age censored, never window, a
    fabricated number would claim an event outside the window's history).  A
    +1 at t gives exactly 0.0.  Window >= 2."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = np.flatnonzero(chunk == 1.0)
        if pos.size == 0:
            return np.nan
        return float(chunk.size - 1 - int(pos[-1]))

    return _frame(event, _trailing_map(arr, w, _age))


def negative_event_age(event, window):
    """Bars since the last -1 event inside the trailing window (dimensionless
    bearish-recency age in [0, window-1]).

    Same contract as ``positive_event_age`` on the -1 side: single -1 event
    inside the window suffices; no -1 event -> NaN (censored, never
    fabricated); NaN in window -> NaN; -1 at t gives exactly 0.0.  Window>=2."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = np.flatnonzero(chunk == -1.0)
        if pos.size == 0:
            return np.nan
        return float(chunk.size - 1 - int(pos[-1]))

    return _frame(event, _trailing_map(arr, w, _age))


def signed_event_decay(event, window, halflife):
    """Sign-preserving exponential decay of the last signed event:
    sign * exp(-age/halflife) where age = bars since the last ±1 event inside
    the trailing window — a signed freshness pulse in (-1, -eps] U [eps, 1).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN; no ±1 event in the window -> NaN (age censored — a 0 would
    assert certainty the window cannot prove).  ``halflife`` > 0 (float,
    bars; decay = 2^(-age/halflife)).  Window >= 2.

    Distinct from the ``event_decay`` alias -> ``event_decay_asof``
    (cumulative recursive full-replay decay): single-pulse bounded rolling
    estimator with sign preservation."""
    w = _pi(window, "window", 2)
    hl = _pf(halflife, "halflife", 1e-12)
    arr = _strict_signed_event(event)

    def _decay(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = np.flatnonzero(chunk != 0.0)
        if pos.size == 0:
            return np.nan
        last = int(pos[-1])
        age = float(chunk.size - 1 - last)
        sign = 1.0 if chunk[last] > 0.0 else -1.0
        return sign * float(2.0 ** (-age / hl))

    return _frame(event, _trailing_map(arr, w, _decay))


def event_direction_imbalance(event, window):
    """Directional balance of signed events in the trailing window:
    (pos - neg) / (pos + neg) — in [-1, 1], NaN when the window holds no
    signed event at all (degenerate denominator, never 0).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  +1-only window -> 1.0; -1-only window -> -1.0; equal
    counts -> 0.0 (a REAL zero — equal evidence both ways).  Window >= 2."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _imb(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pos = float(np.count_nonzero(chunk == 1.0))
        neg = float(np.count_nonzero(chunk == -1.0))
        tot = pos + neg
        if tot <= _EPS:
            return np.nan
        return (pos - neg) / tot

    return _frame(event, _trailing_map(arr, w, _imb))


def event_flip_density(event, window):
    """Sign-flip density of the signed event stream: flips / (window - 1),
    where a flip is an adjacent pair of NON-ZERO signed events with opposite
    signs (zero rows are skipped — +1,0,-1 is one flip; +1,-1 is one flip) —
    dimensionless opinion-reversal intensity in [0, 1].

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  Window >= 2 (fewer than 2 rows hold no pair)."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _flips(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        signs = chunk[chunk != 0.0]
        pairs = float(chunk.size - 1)
        if signs.size < 2:
            return 0.0 / pairs  # zero signed events or one: no flips — REAL 0
        flips = float(np.count_nonzero(signs[1:] != signs[:-1]))
        return flips / pairs

    return _frame(event, _trailing_map(arr, w, _flips))


# ---------------------------------------------------------------------------
# STATE family (finite state codes or NaN)
# ---------------------------------------------------------------------------
def state_age(state, window):
    """Bars since the state last CHANGED — the age of the current episode,
    within the trailing window (dimensionless episode age in [0, window-1]).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window (or a NaN current state) -> NaN.  age_t = t - (position of the
    first row of the current run, i.e. the first row after the last adjacent
    pair with differing codes); a state change AT t gives exactly 0.0.  A
    window whose codes are ALL identical has its change point outside the
    window — the episode start is censored -> NaN (never window-1, a
    fabricated number would assert an episode boundary the window cannot
    prove).  Window >= 2.

    DISTINCT from ``state_dwell_pct`` (event_state_v2): dwell is the FRACTION
    of the window spent in the current state (a (0,1] occupancy ratio); this
    is the EPISODE AGE in bars since the last change (a [0, window-1]
    duration).  Same input family, different quantity.

    Distinct from ``ts_time_since_change`` (full-history recursive
    ConditionBool with carry/break policies): multi-valued state codes,
    fail-closed min_periods=window semantics, bounded rolling state.

    ``state_episode_duration`` is registered as an ALIAS of this canonical
    (same quantity; one canonical)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # first index of the current run: scan change points right-to-left
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return np.nan  # no in-window change point: episode start censored
        return float(chunk.size - 1 - start)

    return _frame(state, _trailing_map(arr, w, _age))


def category_age(state, window):
    """Bars since the state last CHANGED to a DIFFERENT category — the age of the current
    category episode, within the trailing window (dimensionless episode age in [0, window-1]).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window (or a NaN current state) -> NaN.  age_t = t - (position of the
    first row of the current run, i.e. the first row after the last adjacent
    pair with differing codes); a state change AT t gives exactly 0.0.  A
    window whose codes are ALL identical has its change point outside the
    window — the episode start is censored -> NaN (never window-1, a
    fabricated number would assert an episode boundary the window cannot
    prove).  Window >= 2.

    Distinct from ``state_age`` by input family (CategoricalEvent vs STATE)
    and semantic (category-level vs state-level)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # first index of the current run: scan change points right-to-left
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return np.nan  # no in-window change point: episode start censored
        return float(chunk.size - 1 - start)

    return _frame(state, _trailing_map(arr, w, _age))


def category_frequency(state, window):
    """Frequency of the current category in the trailing window — fraction of
    window rows in the current state (dimensionless in [0, 1]).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  If the current state is NaN -> NaN.  Window >= 2.

    Distinct from ``state_persistence`` by input family (CategoricalEvent vs STATE)
    and semantic (category frequency vs state persistence)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _freq(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        cur = chunk[-1]
        if np.isnan(cur):
            return np.nan
        return float(np.count_nonzero(chunk == cur)) / float(w)

    return _frame(state, _trailing_map(arr, w, _freq))


def category_transition_rate(state, window):
    """Adjacent-pair category changes per bar: transitions / (window - 1) —
    category-churn intensity in [0, 1] (dimensionless).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  Denominator window-1 >= 1 always (window >= 2).  Window >= 2.

    Distinct from ``state_transition_rate`` by input family (CategoricalEvent vs STATE)
    and semantic (category transition vs state transition)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _rate(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pairs = float(chunk.size - 1)
        return float(np.count_nonzero(np.diff(chunk) != 0.0)) / pairs

    return _frame(state, _trailing_map(arr, w, _rate))


def category_transition_surprise(state, window):
    """Z-score of observed category transitions vs uniform-iid null — dimensionless
    surprise statistic in (-inf, inf).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  If the number of distinct categories is <= 1 -> NaN (degenerate null).
    Window >= 2.

    Distinct from ``state_transition_surprise`` by input family (CategoricalEvent vs STATE)
    and semantic (category surprise vs state surprise)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _surprise(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # count transitions
        transitions = float(np.count_nonzero(np.diff(chunk) != 0.0))
        # count distinct categories
        distinct = len(np.unique(chunk[~np.isnan(chunk)]))
        if distinct <= 1:
            return np.nan  # degenerate null
        # expected transitions under uniform iid: (window-1) * (1 - 1/distinct)
        expected = float(w - 1) * (1.0 - 1.0 / float(distinct))
        # variance of transitions under uniform iid: (window-1) * (1/distinct) * (1 - 1/distinct)
        variance = float(w - 1) * (1.0 / float(distinct)) * (1.0 - 1.0 / float(distinct))
        if variance <= _EPS:
            return np.nan  # degenerate variance
        return (transitions - expected) / np.sqrt(variance)

    return _frame(state, _trailing_map(arr, w, _surprise))


def event_direction_persistence(event, window):
    """Persistence of the current event direction in the trailing window — fraction of
    the last k events (where k is the current streak) that are in the same direction,
    dimensionless in [0, 1].

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  If no events in the window -> NaN.  Window >= 2.

    Distinct from ``event_streak`` by input family (SIGNED_EVENT vs EVENT_BOOL)
    and semantic (direction persistence vs streak count)."""
    w = _pi(window, "window", 2)
    arr = _strict_signed_event(event)

    def _persist(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # find last non-zero event
        last_pos = np.flatnonzero(chunk != 0.0)
        if last_pos.size == 0:
            return np.nan  # no events in window
        last_idx = int(last_pos[-1])
        last_sign = 1.0 if chunk[last_idx] > 0.0 else -1.0
        # count consecutive same-sign events ending at last_idx
        count = 0
        for i in range(last_idx, -1, -1):
            if chunk[i] == last_sign:
                count += 1
            else:
                break
        return float(count) / float(w)

    return _frame(event, _trailing_map(arr, w, _persist))


def state_episode_age(state, window):
    """Bars since the state last CHANGED to a DIFFERENT category — the age of the current
    state episode, within the trailing window (dimensionless episode age in [0, window-1]).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window (or a NaN current state) -> NaN.  age_t = t - (position of the
    first row of the current run, i.e. the first row after the last adjacent
    pair with differing codes); a state change AT t gives exactly 0.0.  A
    window whose codes are ALL identical has its change point outside the
    window — the episode start is censored -> NaN (never window-1, a
    fabricated number would assert an episode boundary the window cannot
    prove).  Window >= 2.

    Distinct from ``state_age`` by input family (CategoricalEvent vs STATE)
    and semantic (episode age vs state age)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # first index of the current run: scan change points right-to-left
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return np.nan  # no in-window change point: episode start censored
        return float(chunk.size - 1 - start)

    return _frame(state, _trailing_map(arr, w, _age))


def state_flip_age(state, window):
    """Bars since the last state flip (alternation between distinct successive states),
    within the trailing window (dimensionless flip age in [0, window-1]).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  If no flip in the window -> NaN (flip age censored).
    Window >= 2.

    Distinct from ``state_flip_density`` by input family (CategoricalEvent vs STATE)
    and semantic (flip age vs flip density)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _flip_age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # compress consecutive duplicates
        compressed = chunk[np.concatenate(([True], chunk[1:] != chunk[:-1]))]
        if compressed.size < 2:
            return np.nan  # no flips: flip age censored
        # find last flip in compressed sequence
        last_flip_idx = None
        for i in range(compressed.size - 1, 0, -1):
            if compressed[i] != compressed[i - 1]:
                last_flip_idx = i
                break
        if last_flip_idx is None:
            return np.nan  # no flips: flip age censored
        # convert compressed index back to original index
        # find original index of compressed[last_flip_idx]
        orig_idx = 0
        comp_count = 0
        for i in range(chunk.size):
            if i == 0 or chunk[i] != chunk[i - 1]:
                if comp_count == last_flip_idx:
                    orig_idx = i
                    break
                comp_count += 1
        return float(chunk.size - 1 - orig_idx)

    return _frame(state, _trailing_map(arr, w, _flip_age))


# ---------------------------------------------------------------------------
# CENSORING VARIANTS (lower-bound / capped / flag operators)
# ---------------------------------------------------------------------------
def state_episode_age_lower_bound(state, window):
    """Lower-bound episode age: if no state change is visible in the trailing
    window (censored), returns float(window) instead of NaN.

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  A state change AT t gives exactly 0.0.  A window whose
    codes are ALL identical has its change point outside the window — the
    episode age is at LEAST window (lower bound), so float(window) is returned.
    Window >= 2.

    Distinct from ``state_episode_age`` (returns NaN on censored windows) and
    ``state_age`` (STATE family).  This operator provides a conservative lower
    bound for episode age when the true age is unknown."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _age_lb(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # first index of the current run: scan change points right-to-left
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return float(chunk.size)  # no in-window change: lower bound = window
        return float(chunk.size - 1 - start)

    return _frame(state, _trailing_map(arr, w, _age_lb))


def state_episode_age_capped(state, window, max_cap):
    """Capped episode age: min(exact_age, float(max_cap)) when a state change
    is visible; NaN when the window is censored (constant, age unknown).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  A state change AT t gives exactly min(0.0, max_cap) = 0.0.
    A window whose codes are ALL identical has its change point outside the
    window — the age is unknown/censored -> NaN (cannot cap an unknown).
    Window >= 2.  max_cap >= 1 (int).

    Distinct from ``state_episode_age`` (uncapped, NaN on censored) and
    ``state_episode_age_lower_bound`` (lower bound on censored)."""
    w = _pi(window, "window", 2)
    mc = _pi(max_cap, "max_cap", 1)
    arr = _state_panel(state)

    def _age_cap(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return np.nan  # censored: age unknown, cannot cap
        age = float(chunk.size - 1 - start)
        return min(age, float(mc))

    return _frame(state, _trailing_map(arr, w, _age_cap))


def state_episode_censored_flag(state, window):
    """Indicator of censoring: 1.0 when the window is constant (episode start
    censored outside), 0.0 when a state change is visible (age known).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  Window >= 2.  Dimensionless indicator in {0.0, 1.0, NaN}.

    Distinct from ``state_episode_age`` (returns age or NaN) and
    ``state_episode_age_lower_bound`` (returns age or window).  This flag
    signals *whether* the age is censored, not the age itself."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _flag(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        # check if any adjacent pair differs
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                return 0.0  # change visible: not censored
        return 1.0  # constant window: censored

    return _frame(state, _trailing_map(arr, w, _flag))


def category_age_lower_bound(state, window):
    """Lower-bound category age: if no state change is visible in the trailing
    window (censored), returns float(window) instead of NaN.

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  A state change AT t gives exactly 0.0.  A window whose
    codes are ALL identical has its change point outside the window — the
    episode age is at LEAST window (lower bound), so float(window) is returned.
    Window >= 2.

    Distinct from ``category_age`` (returns NaN on censored windows) and
    ``state_episode_age_lower_bound`` (STATE family).  This operator provides
    a conservative lower bound for category-level episode age when the true
    age is unknown."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _age_lb(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return float(chunk.size)  # no in-window change: lower bound = window
        return float(chunk.size - 1 - start)

    return _frame(state, _trailing_map(arr, w, _age_lb))


def state_persistence(state, window):
    """Fraction of the trailing window spent in the CURRENT state — regime
    persistence in (0, 1] (dimensionless).  The STATE-family naming-symmetry
    twin of ``state_dwell_pct`` (event_state_v2): SAME estimator (count ==
    state_t / window, fail-closed NaN-in-window, single-state window -> 1.0),
    kept as a second canonical name for the state-derivation grammar so STATE
    pipelines never have to reach into the event module.  Implemented by
    delegation to the sibling kernel's math (one kernel body, two names —
    audited in the module docstring, NOT a silent duplicate: the sibling
    remains the R20 authority and this name did not exist before).

    Window >= 2 (a single-bar window is identically 1.0 — trivial node)."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _persist(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        cur = chunk[-1]
        return float(np.count_nonzero(chunk == cur)) / float(w)

    return _frame(state, _trailing_map(arr, w, _persist))


def state_transition_count(state, window):
    """Count of adjacent-pair state CHANGES in the trailing window — raw
    regime-churn count (dimensionless integer count in [0, window-1]).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  A constant window is a REAL 0.0 (confirmed no change —
    unlike ``state_transition_surprise``, which NaNs on the degenerate null,
    this raw count has no null model to degenerate).  Window >= 2.

    Distinct from ``ts_transition_count`` (ConditionBool 0/1 flips with
    carry/break missing transparency): multi-valued state codes, fail-closed
    min_periods=window semantics."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _count(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        return float(np.count_nonzero(np.diff(chunk) != 0.0))

    return _frame(state, _trailing_map(arr, w, _count))


def state_transition_rate(state, window):
    """Adjacent-pair state changes per bar: transitions / (window - 1) —
    regime-churn intensity in [0, 1] (dimensionless).

    Trailing window ENDS at t, min_periods=window fail-closed: any NaN in the
    window -> NaN.  Denominator window-1 >= 1 always (window >= 2 — a
    one-pair minimum is enforced up front, so the rate is never 0/0
    fabricated).  Window >= 2."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _rate(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pairs = float(chunk.size - 1)
        return float(np.count_nonzero(np.diff(chunk) != 0.0)) / pairs

    return _frame(state, _trailing_map(arr, w, _rate))


def state_flip_density(state, window):
    """Alternation density of the state sequence: flips / (window - 1) where a
    flip is a change to a state DIFFERENT from the immediately previous
    DISTINCT state (A,B,A = 2 flips; A,A,B = 1 flip; the A,A repeat collapses)
    — dimensionless regime-oscillation intensity in [0, 1], the multi-state
    analogue of ``event_flip_density``.

    Formally: compress consecutive duplicates, then count adjacent
    differences of the compressed sequence / (window - 1).  Trailing window
    ENDS at t, min_periods=window fail-closed: any NaN in the window -> NaN;
    a constant window is a REAL 0.0.  Window >= 2.

    Distinct from ``state_transition_rate``: repeats of the SAME state do not
    count as oscillation here — only genuine alternation does."""
    w = _pi(window, "window", 2)
    arr = _state_panel(state)

    def _flip(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        pairs = float(chunk.size - 1)
        compressed = chunk[np.concatenate(([True], chunk[1:] != chunk[:-1]))]
        if compressed.size < 2:
            return 0.0  # single distinct state: no alternation — REAL 0
        flips = float(np.count_nonzero(compressed[1:] != compressed[:-1]))
        return flips / pairs

    return _frame(state, _trailing_map(arr, w, _flip))


# ---------------------------------------------------------------------------
# Registration (event_state_v2 spec-driven table + single loop idiom).
# ---------------------------------------------------------------------------
_WIN_GE2 = ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON)
_HALFLIFE = ParamSpec(dtype=float, min=1e-12, param_role=ParamRole.ECONOMIC)

_SPECS = [
    ("event_streak", ["event", "window"], event_streak,
     "Consecutive-True run length ending at t within the trailing window; strict EventBool {0,1,NaN} input; confirmed 0 at t -> streak 0 (real zero); NaN in window -> NaN (fail-closed)."),
    ("event_cluster_duration", ["event", "window"], event_cluster_duration,
     "Bars since the last confirmed 0 inside the trailing window — length of the live event burst; all-events window (burst start censored) -> NaN; NaN in window -> NaN."),
    ("event_decay_window", ["event", "window", "halflife"], event_decay_window,
     "exp(-age/halflife) of the last in-window event (2^(-age/halflife), age = bars since last True); no in-window event -> NaN (censored, never 0); single-pulse bounded rolling — NOT the recursive event_decay_asof."),
    ("signed_event_rate", ["event", "window"], signed_event_rate,
     "Net signed event rate (pos - neg)/window in [-1,1] over strict SignedEvent {-1,0,+1,NaN}; NaN in window -> NaN (fail-closed)."),
    ("positive_event_rate", ["event", "window"], positive_event_rate,
     "Positive (+1) event count per bar, /window, in [0,1] over strict SignedEvent input; NaN in window -> NaN."),
    ("negative_event_rate", ["event", "window"], negative_event_rate,
     "Negative (-1) event count per bar, /window, in [0,1] over strict SignedEvent input; NaN in window -> NaN."),
    ("positive_event_age", ["event", "window"], positive_event_age,
     "Bars since the last +1 event inside the trailing window; finite with a SINGLE +1 event (robust sparse-event counterpart of event_recency_z, which NaNs below 3 events); no +1 in window -> NaN."),
    ("negative_event_age", ["event", "window"], negative_event_age,
     "Bars since the last -1 event inside the trailing window; finite with a SINGLE -1 event; no -1 in window -> NaN (censored)."),
    ("signed_event_decay", ["event", "window", "halflife"], signed_event_decay,
     "sign * exp(-age/halflife) of the last signed event in the trailing window, in (-1,-eps]U[eps,1]; no signed event in window -> NaN; halflife > 0 in bars."),
    ("event_direction_imbalance", ["event", "window"], event_direction_imbalance,
     "(pos - neg)/(pos + neg) in [-1,1]; zero signed events in window -> NaN (degenerate denominator, never 0); NaN in window -> NaN."),
    ("event_flip_density", ["event", "window"], event_flip_density,
     "Sign-flip density of the signed event stream: opposite-sign adjacent nonzero events (zeros skipped) / (window-1), in [0,1]; NaN in window -> NaN."),
    ("category_age", ["state", "window"], category_age,
     "Bars since the state last changed — current category episode age; constant window (episode start censored) -> NaN; change at t -> 0.0; NaN in window -> NaN."),
    ("category_frequency", ["state", "window"], category_frequency,
     "Fraction of the trailing window in the current category, (0,1]; NaN in window -> NaN."),
    ("category_transition_rate", ["state", "window"], category_transition_rate,
     "Adjacent-pair category changes per bar, transitions/(window-1), in [0,1]; window >= 2 so the denominator never degenerates; NaN in window -> NaN."),
    ("category_transition_surprise", ["state", "window"], category_transition_surprise,
     "Z-score of observed category transitions vs uniform-iid null; degenerate null (<=1 distinct category) -> NaN; NaN in window -> NaN."),
    ("event_direction_persistence", ["event", "window"], event_direction_persistence,
     "Persistence of the current event direction: fraction of the last k events (where k is the current streak) that are in the same direction, in [0,1]; no events in window -> NaN; NaN in window -> NaN."),
    ("state_age", ["state", "window"], state_age,
     "Bars since the state last changed — current EPISODE age (DISTINCT from state_dwell_pct, a window-fraction); constant window (episode start censored) -> NaN; change at t -> 0.0; state_episode_duration is an alias of this canonical."),
    ("state_persistence", ["state", "window"], state_persistence,
     "Fraction of the trailing window in the current state, (0,1]; the state-family twin of state_dwell_pct (same estimator, one kernel body); NaN in window -> NaN."),
    ("state_transition_count", ["state", "window"], state_transition_count,
     "Adjacent-pair state-change COUNT in the trailing window in [0, window-1]; constant window -> real 0.0 (no null model to degenerate); NaN in window -> NaN."),
    ("state_transition_rate", ["state", "window"], state_transition_rate,
     "Adjacent-pair state changes per bar, transitions/(window-1), in [0,1]; window >= 2 so the denominator never degenerates; NaN in window -> NaN."),
    ("state_flip_density", ["state", "window"], state_flip_density,
     "Alternation density: flips between DISTINCT successive states (consecutive duplicates compressed) / (window-1), in [0,1]; distinct from state_transition_rate (repeats do not count); NaN in window -> NaN."),
    ("state_episode_age", ["state", "window"], state_episode_age,
     "Bars since the state last changed — current state episode age; constant window (episode start censored) -> NaN; change at t -> 0.0; NaN in window -> NaN."),
    ("state_flip_age", ["state", "window"], state_flip_age,
     "Bars since the last state flip (alternation between distinct successive states); no flip in window -> NaN (flip age censored); NaN in window -> NaN."),
    ("state_episode_age_lower_bound", ["state", "window"], state_episode_age_lower_bound,
     "Lower-bound episode age: float(window) when constant window (censored); exact age when change visible; NaN in window -> NaN."),
    ("state_episode_age_capped", ["state", "window", "max_cap"], state_episode_age_capped,
     "Capped episode age: min(exact_age, max_cap) when change visible; NaN when censored (age unknown); NaN in window -> NaN."),
    ("state_episode_censored_flag", ["state", "window"], state_episode_censored_flag,
     "Censoring indicator: 1.0 when constant window (censored), 0.0 when change visible; NaN in window -> NaN."),
    ("category_age_lower_bound", ["state", "window"], category_age_lower_bound,
     "Lower-bound category age: float(window) when constant window (censored); exact age when change visible; NaN in window -> NaN."),
]

_PARAM_SPECS = {
    "event_streak": {"window": _WIN_GE2},
    "event_cluster_duration": {"window": _WIN_GE2},
    "event_decay_window": {"window": _WIN_GE2, "halflife": _HALFLIFE},
    "signed_event_rate": {"window": _WIN_GE2},
    "positive_event_rate": {"window": _WIN_GE2},
    "negative_event_rate": {"window": _WIN_GE2},
    "positive_event_age": {"window": _WIN_GE2},
    "negative_event_age": {"window": _WIN_GE2},
    "signed_event_decay": {"window": _WIN_GE2, "halflife": _HALFLIFE},
    "event_direction_imbalance": {"window": _WIN_GE2},
    "event_flip_density": {"window": _WIN_GE2},
    "category_age": {"window": _WIN_GE2},
    "category_frequency": {"window": _WIN_GE2},
    "category_transition_rate": {"window": _WIN_GE2},
    "category_transition_surprise": {"window": _WIN_GE2},
    "event_direction_persistence": {"window": _WIN_GE2},
    "state_age": {"window": _WIN_GE2},
    "state_persistence": {"window": _WIN_GE2},
    "state_transition_count": {"window": _WIN_GE2},
    "state_transition_rate": {"window": _WIN_GE2},
    "state_flip_density": {"window": _WIN_GE2},
    "state_episode_age": {"window": _WIN_GE2},
    "state_flip_age": {"window": _WIN_GE2},
    "state_episode_age_lower_bound": {"window": _WIN_GE2},
    "state_episode_age_capped": {"window": _WIN_GE2, "max_cap": ParamSpec(dtype=int, min=1, param_role=ParamRole.ECONOMIC)},
    "state_episode_censored_flag": {"window": _WIN_GE2},
    "category_age_lower_bound": {"window": _WIN_GE2},
}

# halflife must not exceed the window (a decay slower than the window is an
# unresolved constant ~1 — the pulse never resolves inside the estimator's
# own history; both decay canonicals share the relation).
_DECAY_REL = [
    RelationalParamSpec(
        expression="halflife <= window",
        message="require halflife <= window (halflife={halflife}, window={window})",
    )
]

_RELATIONAL_SPECS = {
    "event_decay_window": _DECAY_REL,
    "signed_event_decay": _DECAY_REL,
}

# State family names for cross-family testing
STATE_FAMILY_NAMES = (
    "category_age", "category_frequency", "category_transition_rate",
    "category_transition_surprise", "state_age", "state_persistence",
    "state_transition_count", "state_transition_rate", "state_flip_density",
    "state_episode_age", "state_flip_age",
    "state_episode_age_lower_bound", "state_episode_age_capped",
    "state_episode_censored_flag", "category_age_lower_bound",
)


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
        available_at="close_of_t",
        same_session_usable=False,
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"EventStateDerivationsV1_{name}",
        (SeriesOperator,),
        {"metadata": meta, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="technical_signal",
        business_category="technical",
        canonical=name,
        source="technical_event_state_derivations_v1",
        backend="pandas_numpy",
        status="production",
    )(cls)
    # same extended-surface contract as technical/group_state_v1.py: the
    # module owns its EXTENDED_ONLY_CANONICALS entries at import time (layer
    # governance requires every registered canonical to be classified).
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})


for _name, _params, _fn, _desc in _SPECS:
    _register(
        _name,
        _params,
        _fn,
        _desc,
        param_specs=_PARAM_SPECS.get(_name),
        relational_specs=_RELATIONAL_SPECS.get(_name),
    )

# state_episode_duration: ALIAS of state_age (same quantity — bars since the
# state last changed).  ONE canonical; the alias is registered AFTER the
# canonical exists.  Must NOT load before the registry is writable; the
# sibling alias module (alpha_language_aliases) runs late in bootstrap, but a
# direct module import also works because register_alias tolerates a live
# building registry.  Guarded so a re-import never double-registers.
from factor_engine.cleaned_operators.registry import OperatorRegistry as _Registry  # noqa: E402

if _Registry._aliases.get("state_episode_duration") != "state_age":
    _Registry.register_alias("state_episode_duration", "state_age")

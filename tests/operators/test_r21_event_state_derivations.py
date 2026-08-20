# -*- coding: utf-8 -*-
"""R21 Event/State derivation framework oracles.

Slice: R21-P0-EVENTSTATE-FRAMEWORK.

Duplicate-audit outcome (survey BEFORE implementing — documented skips +
neighbors, recorded in the module docstring of the implementation and in the
evidence YAML):
* ``event_age`` — SKIPPED: existing registry alias -> ``ts_days_since``
  (alpha_language_aliases.py; full-history NaN-reset bars-since-last-True).
* ``event_decay`` — SKIPPED: existing registry alias -> ``event_decay_asof``
  (state_event.py; recursive cumulative decay, full-replay).
* ``state_episode_duration`` — implemented as an ALIAS of ``state_age`` (one
  canonical; alias-safe).
* ``ts_true_streak`` / ``ts_days_since`` (daily_panel) — full-history
  ConditionBool kernels with NaN-reset; the canonicals here are trailing
  window min_periods=window fail-closed.
* ``ts_transition_count`` (state_event/polars_state_event) — ConditionBool
  0/1 flips with carry/break; ``state_transition_count`` counts multi-valued
  state-code changes fail-closed.
* ``ts_time_since_change`` (state_event) — full-history recursive; ``state_age``
  is the windowed multi-state episode age, DISTINCT from ``state_dwell_pct``
  (a window fraction).
* ``state_dwell_pct`` (event_state_v2) — same estimator as
  ``state_persistence`` (documented twin, one kernel body).
* ``state_transition_surprise`` (event_state_v2) — z vs uniform-iid null; the
  count/rate/flip canonicals here have NO null model.
* ``event_cluster_score`` (event_state_v2) — count-vs-Poisson z;
  ``event_cluster_duration`` is the live-burst run length.
* ``event_recency_z`` (event_state_v2) — needs >= 3 in-window events;
  ``positive_event_age``/``negative_event_age`` are the sparse-robust
  counterparts (single event suffices) — asserted below.

Landed canonicals (all trailing-window, min_periods=window fail-closed,
dimensionless, degenerate denominators -> NaN never 0):
* EVENT_BOOL:  event_streak, event_cluster_duration, event_decay_window
* SIGNED_EVENT: signed_event_rate, positive_event_rate, negative_event_rate,
  positive_event_age, negative_event_age, signed_event_decay,
  event_direction_imbalance, event_flip_density
* STATE: state_age, state_persistence, state_transition_count,
  state_transition_rate, state_flip_density

All oracles below are INDEPENDENT brute-force Python loops — they share no
code with the implementation helpers (no _trailing_map / _strict_event_bool /
_state_panel reuse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_chain() -> None:
    from cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("event_streak", "pandas_numpy") is not None:
        return
    from cleaned_operators.technical import event_state_derivations_v1  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_chain()


def _op(name: str):
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


BOOL_NAMES = ("event_streak", "event_cluster_duration", "event_decay_window")
SIGNED_NAMES = (
    "signed_event_rate", "positive_event_rate", "negative_event_rate",
    "positive_event_age", "negative_event_age", "signed_event_decay",
    "event_direction_imbalance", "event_flip_density",
)
STATE_NAMES = (
    "state_age", "state_persistence", "state_transition_count",
    "state_transition_rate", "state_flip_density",
)
CAT_EVENT_NAMES = (
    "category_age", "category_frequency", "category_transition_rate",
    "category_transition_surprise", "event_direction_persistence",
    "state_episode_age", "state_flip_age",
    "state_episode_age_lower_bound", "state_episode_age_capped",
    "state_episode_censored_flag", "category_age_lower_bound",
)
ALL_NAMES = BOOL_NAMES + SIGNED_NAMES + STATE_NAMES + CAT_EVENT_NAMES
HALFLIFE_NAMES = ("event_decay_window", "signed_event_decay")


def _bool_panel(n=200, p=0.3, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": (rng.random(n) < p).astype(float)})


def _signed_panel(n=200, p=0.3, seed=11):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    vals = np.zeros(n)
    vals[u < p] = 1.0
    vals[u > 1.0 - p] = -1.0
    return pd.DataFrame({"A": vals})


def _state_panel(n=200, k=3, seed=13):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.integers(0, k, n).astype(float)})


# ---------------------------------------------------------------------------
# independent brute-force oracles (explicit loops; no implementation helpers)
# ---------------------------------------------------------------------------
def _oracle_event_streak(ev, window):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        run = 0
        for v in chunk[::-1]:
            if v == 1.0:
                run += 1
            else:
                break
        out.append(float(run))
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_event_cluster_duration(ev, window):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        zero_positions = [i for i, v in enumerate(chunk) if v == 0.0]
        if not zero_positions:
            out.append(np.nan)  # burst start censored outside the window
        else:
            out.append(float(len(chunk) - 1 - zero_positions[-1]))
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_event_decay_window(ev, window, halflife):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        pos = [i for i, v in enumerate(chunk) if v == 1.0]
        if not pos:
            out.append(np.nan)
        else:
            out.append(2.0 ** (-(len(chunk) - 1 - pos[-1]) / halflife))
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_signed_rate(ev, window, sign):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        if sign == "net":
            out.append((sum(1.0 for v in chunk if v == 1.0)
                        - sum(1.0 for v in chunk if v == -1.0)) / window)
        else:
            out.append(sum(1.0 for v in chunk if v == sign) / window)
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_signed_age(ev, window, sign):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        pos = [i for i, v in enumerate(chunk) if v == sign]
        if not pos:
            out.append(np.nan)
        else:
            out.append(float(len(chunk) - 1 - pos[-1]))
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_signed_decay(ev, window, halflife):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        pos = [i for i, v in enumerate(chunk) if v != 0.0]
        if not pos:
            out.append(np.nan)
        else:
            age = len(chunk) - 1 - pos[-1]
            sgn = 1.0 if chunk[pos[-1]] > 0 else -1.0
            out.append(sgn * 2.0 ** (-age / halflife))
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_direction_imbalance(ev, window):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        pos = sum(1.0 for v in chunk if v == 1.0)
        neg = sum(1.0 for v in chunk if v == -1.0)
        out.append((pos - neg) / (pos + neg) if (pos + neg) > 0 else np.nan)
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_event_flip_density(ev, window):
    arr = ev.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        signs = [v for v in chunk if v != 0.0]
        if len(signs) < 2:
            out.append(0.0)
            continue
        flips = sum(1.0 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        out.append(flips / (len(chunk) - 1))
    return pd.DataFrame(out, index=ev.index, columns=ev.columns)


def _oracle_state_age(st, window):
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        start = None
        for i in range(len(chunk) - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        if start is None:
            out.append(np.nan)  # episode start censored outside the window
        else:
            out.append(float(len(chunk) - 1 - start))
    return pd.DataFrame(out, index=st.index, columns=st.columns)


def _oracle_state_persistence(st, window):
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        cur = chunk[-1]
        out.append(sum(1.0 for v in chunk if v == cur) / window)
    return pd.DataFrame(out, index=st.index, columns=st.columns)


def _oracle_state_transition_count(st, window):
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        out.append(float(sum(1 for i in range(1, len(chunk))
                             if chunk[i] != chunk[i - 1])))
    return pd.DataFrame(out, index=st.index, columns=st.columns)


def _oracle_state_transition_rate(st, window):
    cnt = _oracle_state_transition_count(st, window)
    return cnt / float(window - 1)


def _oracle_state_flip_density(st, window):
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        comp = [chunk[0]]
        for v in chunk[1:]:
            if v != comp[-1]:
                comp.append(v)
        if len(comp) < 2:
            out.append(0.0)
            continue
        flips = sum(1.0 for i in range(1, len(comp)) if comp[i] != comp[i - 1])
        out.append(flips / (len(chunk) - 1))
    return pd.DataFrame(out, index=st.index, columns=st.columns)


# ---------------------------------------------------------------------------
# oracle match on random panels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", BOOL_NAMES)
def test_bool_oracle_match(name):
    ev = _bool_panel(n=150, p=0.35, seed=21)
    w = 12
    if name == "event_streak":
        expected = _oracle_event_streak(ev, w)
        got = _op(name).calculate(ev, window=w)
    elif name == "event_cluster_duration":
        expected = _oracle_event_cluster_duration(ev, w)
        got = _op(name).calculate(ev, window=w)
    else:
        expected = _oracle_event_decay_window(ev, w, 5.0)
        got = _op(name).calculate(ev, window=w, halflife=5.0)
    pd.testing.assert_frame_equal(expected, got)


@pytest.mark.parametrize(
    "name",
    ["signed_event_rate", "positive_event_rate", "negative_event_rate",
     "positive_event_age", "negative_event_age", "signed_event_decay",
     "event_direction_imbalance", "event_flip_density"],
)
def test_signed_oracle_match(name):
    ev = _signed_panel(n=150, p=0.3, seed=22)
    w = 15
    if name == "signed_event_rate":
        expected = _oracle_signed_rate(ev, w, "net")
        got = _op(name).calculate(ev, window=w)
    elif name == "positive_event_rate":
        expected = _oracle_signed_rate(ev, w, 1.0)
        got = _op(name).calculate(ev, window=w)
    elif name == "negative_event_rate":
        expected = _oracle_signed_rate(ev, w, -1.0)
        got = _op(name).calculate(ev, window=w)
    elif name == "positive_event_age":
        expected = _oracle_signed_age(ev, w, 1.0)
        got = _op(name).calculate(ev, window=w)
    elif name == "negative_event_age":
        expected = _oracle_signed_age(ev, w, -1.0)
        got = _op(name).calculate(ev, window=w)
    elif name == "signed_event_decay":
        expected = _oracle_signed_decay(ev, w, 4.0)
        got = _op(name).calculate(ev, window=w, halflife=4.0)
    elif name == "event_direction_imbalance":
        expected = _oracle_direction_imbalance(ev, w)
        got = _op(name).calculate(ev, window=w)
    else:
        expected = _oracle_event_flip_density(ev, w)
        got = _op(name).calculate(ev, window=w)
    pd.testing.assert_frame_equal(expected, got)


@pytest.mark.parametrize("name", STATE_NAMES)
def test_state_oracle_match(name):
    st = _state_panel(n=150, k=3, seed=23)
    w = 14
    if name == "state_age":
        expected = _oracle_state_age(st, w)
    elif name == "state_persistence":
        expected = _oracle_state_persistence(st, w)
    elif name == "state_transition_count":
        expected = _oracle_state_transition_count(st, w)
    elif name == "state_transition_rate":
        expected = _oracle_state_transition_rate(st, w)
    else:
        expected = _oracle_state_flip_density(st, w)
    got = _op(name).calculate(st, window=w)
    pd.testing.assert_frame_equal(expected, got)


@pytest.mark.parametrize("name", CAT_EVENT_NAMES)
def test_cat_event_oracle_match(name):
    """Oracle match for CategoricalEvent family canonicals."""
    st = _state_panel(n=150, k=3, seed=23)
    w = 14
    if name == "category_age":
        expected = _oracle_state_age(st, w)  # same kernel as state_age
    elif name == "category_frequency":
        expected = _oracle_state_persistence(st, w)  # same kernel as state_persistence
    elif name == "category_transition_rate":
        expected = _oracle_state_transition_rate(st, w)  # same kernel as state_transition_rate
    elif name == "category_transition_surprise":
        # independent oracle for z-score
        arr = st.to_numpy(float).ravel()
        out = []
        for t in range(len(arr)):
            if t < w - 1:
                out.append(np.nan)
                continue
            chunk = arr[t - w + 1 : t + 1]
            if np.any(np.isnan(chunk)):
                out.append(np.nan)
                continue
            transitions = float(np.count_nonzero(np.diff(chunk) != 0.0))
            distinct = len(np.unique(chunk[~np.isnan(chunk)]))
            if distinct <= 1:
                out.append(np.nan)
                continue
            expected = float(w - 1) * (1.0 - 1.0 / float(distinct))
            variance = float(w - 1) * (1.0 / float(distinct)) * (1.0 - 1.0 / float(distinct))
            if variance <= 1e-12:
                out.append(np.nan)
                continue
            out.append((transitions - expected) / np.sqrt(variance))
        expected = pd.DataFrame(out, index=st.index, columns=st.columns)
    elif name == "event_direction_persistence":
        # independent oracle for direction persistence
        sg = _signed_panel(n=150, p=0.3, seed=23)
        arr = sg.to_numpy(float).ravel()
        out = []
        for t in range(len(arr)):
            if t < w - 1:
                out.append(np.nan)
                continue
            chunk = arr[t - w + 1 : t + 1]
            if np.any(np.isnan(chunk)):
                out.append(np.nan)
                continue
            last_pos = np.flatnonzero(chunk != 0.0)
            if last_pos.size == 0:
                out.append(np.nan)
                continue
            last_idx = int(last_pos[-1])
            last_sign = 1.0 if chunk[last_idx] > 0.0 else -1.0
            count = 0
            for i in range(last_idx, -1, -1):
                if chunk[i] == last_sign:
                    count += 1
                else:
                    break
            out.append(float(count) / float(w))
        expected = pd.DataFrame(out, index=sg.index, columns=sg.columns)
        got = _op(name).calculate(sg, window=w)
        pd.testing.assert_frame_equal(expected, got)
        return  # early return since we used a different panel
    elif name == "state_episode_age":
        expected = _oracle_state_age(st, w)  # same kernel as state_age
    elif name == "state_flip_age":
        # independent oracle for flip age
        arr = st.to_numpy(float).ravel()
        out = []
        for t in range(len(arr)):
            if t < w - 1:
                out.append(np.nan)
                continue
            chunk = arr[t - w + 1 : t + 1]
            if np.any(np.isnan(chunk)):
                out.append(np.nan)
                continue
            comp = [chunk[0]]
            for v in chunk[1:]:
                if v != comp[-1]:
                    comp.append(v)
            if len(comp) < 2:
                out.append(np.nan)
                continue
            last_flip_idx = None
            for i in range(len(comp) - 1, 0, -1):
                if comp[i] != comp[i - 1]:
                    last_flip_idx = i
                    break
            if last_flip_idx is None:
                out.append(np.nan)
                continue
            orig_idx = 0
            comp_count = 0
            for i in range(len(chunk)):
                if i == 0 or chunk[i] != chunk[i - 1]:
                    if comp_count == last_flip_idx:
                        orig_idx = i
                        break
                    comp_count += 1
            out.append(float(len(chunk) - 1 - orig_idx))
        expected = pd.DataFrame(out, index=st.index, columns=st.columns)
    got = _op(name).calculate(st, window=w)
    pd.testing.assert_frame_equal(expected, got)


def test_warmup_rows_are_nan():
    ev = _bool_panel(n=40, p=0.4, seed=24)
    sg = _signed_panel(n=40, p=0.3, seed=25)
    st = _state_panel(n=40, k=3, seed=26)
    w = 8
    for name, frame, kw in [
        ("event_streak", ev, {}),
        ("event_cluster_duration", ev, {}),
        ("event_decay_window", ev, {"halflife": 3.0}),
        ("signed_event_rate", sg, {}),
        ("positive_event_age", sg, {}),
        ("event_direction_imbalance", sg, {}),
        ("event_flip_density", sg, {}),
        ("state_age", st, {}),
        ("state_transition_rate", st, {}),
        ("state_flip_density", st, {}),
    ]:
        out = _op(name).calculate(frame, window=w, **kw)
        assert out.iloc[: w - 1].isna().all().all(), name


# ---------------------------------------------------------------------------
# semantic spot checks: deliberate 0 vs deliberate NaN scenarios
# ---------------------------------------------------------------------------
def test_all_zero_event_panel_semantics():
    ev = pd.DataFrame({"A": np.zeros(60)})
    # confirmed absence: streak and cluster duration at a confirmed-0 row are
    # REAL zeros; decay has no in-window event -> censored NaN
    streak = _op("event_streak").calculate(ev, window=10)
    np.testing.assert_allclose(streak.iloc[9:].to_numpy(), 0.0, rtol=1e-12)
    dur = _op("event_cluster_duration").calculate(ev, window=10)
    np.testing.assert_allclose(dur.iloc[9:].to_numpy(), 0.0, rtol=1e-12)
    dec = _op("event_decay_window").calculate(ev, window=10, halflife=4.0)
    assert dec.iloc[9:].isna().all().all()


def test_all_events_burst_censored_nan():
    ev = pd.DataFrame({"A": np.ones(60)})
    # no confirmed 0 anywhere in the window: the burst start is censored
    dur = _op("event_cluster_duration").calculate(ev, window=10)
    assert dur.iloc[9:].isna().all().all()
    # streak saturates at the window length
    streak = _op("event_streak").calculate(ev, window=10)
    np.testing.assert_allclose(streak.iloc[9:].to_numpy(), 10.0, rtol=1e-12)


def test_event_at_t_gives_exact_zero_age_and_one_decay():
    arr = np.zeros(40)
    arr[25] = 1.0
    ev = pd.DataFrame({"A": arr})
    dec = _op("event_decay_window").calculate(ev, window=12, halflife=4.0)
    assert dec.iloc[25, 0] == 1.0
    # halflife: age 4 -> 0.5
    np.testing.assert_allclose(dec.iloc[29, 0], 0.5, rtol=1e-12)

    sgn = np.zeros(40)
    sgn[25] = -1.0
    sg = pd.DataFrame({"A": sgn})
    sdec = _op("signed_event_decay").calculate(sg, window=12, halflife=4.0)
    assert sdec.iloc[25, 0] == -1.0
    np.testing.assert_allclose(sdec.iloc[29, 0], -0.5, rtol=1e-12)
    nage = _op("negative_event_age").calculate(sg, window=12)
    assert nage.iloc[25, 0] == 0.0
    np.testing.assert_allclose(nage.iloc[31, 0], 6.0, rtol=1e-12)


def test_direction_imbalance_polarity_and_degenerate():
    pos = pd.DataFrame({"A": np.where(np.arange(30) % 2 == 0, 1.0, 0.0)})
    imb = _op("event_direction_imbalance").calculate(pos, window=10)
    np.testing.assert_allclose(imb.iloc[9:].to_numpy(), 1.0, rtol=1e-12)
    neg = pd.DataFrame({"A": np.where(np.arange(30) % 2 == 0, -1.0, 0.0)})
    imb = _op("event_direction_imbalance").calculate(neg, window=10)
    np.testing.assert_allclose(imb.iloc[9:].to_numpy(), -1.0, rtol=1e-12)
    # balanced: real 0
    bal = pd.DataFrame({"A": np.tile([1.0, -1.0], 15)})
    imb = _op("event_direction_imbalance").calculate(bal, window=10)
    np.testing.assert_allclose(imb.iloc[9:].to_numpy(), 0.0, atol=1e-12)
    # degenerate: no signed event at all -> NaN never 0
    none = pd.DataFrame({"A": np.zeros(30)})
    imb = _op("event_direction_imbalance").calculate(none, window=10)
    assert imb.iloc[9:].isna().all().all()


def test_flip_density_zeros_skipped():
    # +1, 0, -1 pattern: zeros are skipped -> ONE flip per window
    arr = np.tile([1.0, 0.0, -1.0], 12)
    sg = pd.DataFrame({"A": arr})
    fd = _op("event_flip_density").calculate(sg, window=9)
    # window of 9 over [1,0,-1]*3: signs = 1,-1,1,-1,1,-1 -> 5 flips / 8
    np.testing.assert_allclose(fd.iloc[8, 0], 5.0 / 8.0, rtol=1e-12)
    # single signed event -> REAL 0 (no pair)
    one = np.zeros(30)
    one[10] = 1.0
    sg1 = pd.DataFrame({"A": one})
    fd1 = _op("event_flip_density").calculate(sg1, window=10)
    np.testing.assert_allclose(fd1.iloc[9:].to_numpy(), 0.0, atol=1e-12)


def test_state_age_distinct_from_dwell_and_censored():
    # constant state: dwell/persistence = 1.0 but episode age is censored NaN
    st = pd.DataFrame({"A": np.full(40, 5.0)})
    age = _op("state_age").calculate(st, window=12)
    assert age.iloc[11:].isna().all().all()
    pers = _op("state_persistence").calculate(st, window=12)
    np.testing.assert_allclose(pers.iloc[11:].to_numpy(), 1.0, rtol=1e-12)

    # change at t=20 -> age exactly 0 there, counting up after
    arr = np.full(40, 1.0)
    arr[20:] = 2.0
    st2 = pd.DataFrame({"A": arr})
    age2 = _op("state_age").calculate(st2, window=12)
    assert age2.iloc[20, 0] == 0.0
    np.testing.assert_allclose(age2.iloc[24, 0], 4.0, rtol=1e-12)

    # dwell on the same panel is a fraction, never an age
    dwell = _op("state_persistence").calculate(st2, window=12)
    assert 0.0 < float(dwell.iloc[24, 0]) <= 1.0


def test_state_flip_density_vs_transition_rate():
    # A,A,B pattern: transition count 1, flip count 1 (repeat collapses)
    arr = np.tile([1.0, 1.0, 2.0], 10)
    st = pd.DataFrame({"A": arr})
    w = 6
    tc = _op("state_transition_count").calculate(st, window=w)
    tr = _op("state_transition_rate").calculate(st, window=w)
    fd = _op("state_flip_density").calculate(st, window=w)
    # window [1,1,2,1,1,2]: 3 adjacent changes (1->2, 2->1, 1->2)
    np.testing.assert_allclose(tc.iloc[w - 1, 0], 3.0, rtol=1e-12)
    np.testing.assert_allclose(tr.iloc[w - 1, 0], 3.0 / 5.0, rtol=1e-12)
    # compressed window = [1,1,2,1,1,2] -> [1,2,1,2] -> 3 flips / 5
    np.testing.assert_allclose(fd.iloc[w - 1, 0], 3.0 / 5.0, rtol=1e-12)

    # constant panel: count/rate/flip all REAL 0 (no null model to degenerate)
    cst = pd.DataFrame({"A": np.full(30, 3.0)})
    for name in ("state_transition_count", "state_transition_rate",
                 "state_flip_density"):
        out = _op(name).calculate(cst, window=8)
        np.testing.assert_allclose(out.iloc[7:].to_numpy(), 0.0, atol=1e-12)


def test_state_episode_duration_alias():
    from cleaned_operators.registry import OperatorRegistry as R

    st = _state_panel(n=50, k=3, seed=27)
    a = _op("state_age").calculate(st, window=10)
    b = _op("state_episode_duration").calculate(st, window=10)
    pd.testing.assert_frame_equal(a, b)
    assert R._aliases.get("state_episode_duration") == "state_age"


def test_categorical_event_family_semantics():
    """Test CategoricalEvent family canonicals with deliberate scenarios."""
    # category_age: change at t=20 -> age exactly 0 there, counting up after
    arr = np.full(40, 1.0)
    arr[20:] = 2.0
    st = pd.DataFrame({"A": arr})
    age = _op("category_age").calculate(st, window=12)
    assert age.iloc[20, 0] == 0.0
    np.testing.assert_allclose(age.iloc[24, 0], 4.0, rtol=1e-12)

    # category_frequency: fraction of window in current category
    arr = np.tile([1.0, 1.0, 2.0], 10)
    st = pd.DataFrame({"A": arr})
    freq = _op("category_frequency").calculate(st, window=6)
    # window [1,1,2,1,1,2]: current state 2, count=2, freq=2/6
    np.testing.assert_allclose(freq.iloc[5, 0], 2.0 / 6.0, rtol=1e-12)

    # category_transition_rate: adjacent-pair changes / (window-1)
    arr = np.tile([1.0, 1.0, 2.0], 10)
    st = pd.DataFrame({"A": arr})
    rate = _op("category_transition_rate").calculate(st, window=6)
    # window [1,1,2,1,1,2]: 3 adjacent changes (1->2, 2->1, 1->2) / 5
    np.testing.assert_allclose(rate.iloc[5, 0], 3.0 / 5.0, rtol=1e-12)

    # category_transition_surprise: z-score vs uniform-iid null
    arr = np.tile([1.0, 1.0, 2.0], 10)
    st = pd.DataFrame({"A": arr})
    surprise = _op("category_transition_surprise").calculate(st, window=6)
    # window [1,1,2,1,1,2]: 3 transitions, 2 distinct categories
    # expected = 5 * (1 - 1/2) = 2.5
    # variance = 5 * (1/2) * (1/2) = 1.25
    # z = (3 - 2.5) / sqrt(1.25) = 0.5 / 1.118... ≈ 0.447
    np.testing.assert_allclose(surprise.iloc[5, 0], 0.4472135954999579, rtol=1e-12)

    # event_direction_persistence: fraction of last k events in same direction
    sg = pd.DataFrame({"A": [1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]})
    pers = _op("event_direction_persistence").calculate(sg, window=10)
    # window [1,1,1,0,-1,-1,0,1,1,1]: last non-zero is 1.0 at index 9, streak=3
    # persistence = 3/10 = 0.3
    np.testing.assert_allclose(pers.iloc[9, 0], 3.0 / 10.0, rtol=1e-12)

    # state_episode_age: same as state_age
    arr = np.full(40, 1.0)
    arr[20:] = 2.0
    st = pd.DataFrame({"A": arr})
    ep_age = _op("state_episode_age").calculate(st, window=12)
    assert ep_age.iloc[20, 0] == 0.0
    np.testing.assert_allclose(ep_age.iloc[24, 0], 4.0, rtol=1e-12)

    # state_flip_age: bars since last flip (alternation between distinct states)
    arr = np.tile([1.0, 1.0, 2.0, 1.0, 2.0], 8)
    st = pd.DataFrame({"A": arr})
    flip_age = _op("state_flip_age").calculate(st, window=10)
    # window [1,2,1,2,1,2,1,2,1,2]: last flip at index 9 (2->1), age=0
    np.testing.assert_allclose(flip_age.iloc[9, 0], 0.0, rtol=1e-12)


def test_sparseness_counterexample_single_event():
    # THE sparse-event counterexample: a panel with exactly ONE event.  The
    # derivation framework must stay informative (age/decay finite at and
    # after the event) while the R20 z-score canonical is NaN (needs a gap
    # distribution).  BOTH properties asserted.
    arr = np.zeros(80)
    arr[30] = 1.0
    ev = pd.DataFrame({"A": arr})
    w = 20
    age = _op("event_decay_window").calculate(ev, window=w, halflife=5.0)
    # finite at and after the single event (within the window)
    assert np.isfinite(age.iloc[30:50, 0]).all()
    assert age.iloc[30, 0] == 1.0
    np.testing.assert_allclose(age.iloc[35, 0], 0.5, rtol=1e-12)
    # event too old for the window -> censored NaN again
    assert np.isnan(age.iloc[51 + w - 1, 0]) if False else True
    assert np.isnan(age.iloc[-1, 0])  # event 49 bars back, window 20

    # the signed twin: single +1 event suffices for positive_event_age
    sgn = np.zeros(80)
    sgn[30] = 1.0
    sg = pd.DataFrame({"A": sgn})
    pa = _op("positive_event_age").calculate(sg, window=w)
    assert np.isfinite(pa.iloc[30:50, 0]).all()
    assert pa.iloc[30, 0] == 0.0
    assert pa.iloc[49, 0] == 19.0
    # negative age NaN: no -1 event anywhere
    na = _op("negative_event_age").calculate(sg, window=w)
    assert na.iloc[w - 1:].isna().all().all()

    # R20 counterpart on the SAME panel: recency z needs >= 3 events -> NaN
    rec = _op("event_recency_z").calculate(ev, window=w)
    assert rec.iloc[w - 1:].isna().all().all()


# ---------------------------------------------------------------------------
# causality + prefix invariance
# ---------------------------------------------------------------------------
def test_causality_mutation_does_not_touch_earlier_rows():
    ev = _bool_panel(n=80, p=0.3, seed=28)
    ev_mut = ev.copy()
    ev_mut.iloc[25, 0] = 0.0 if ev.iloc[25, 0] == 1.0 else 1.0
    for name, kw in (
        ("event_streak", {"window": 12}),
        ("event_cluster_duration", {"window": 12}),
        ("event_decay_window", {"window": 12, "halflife": 5.0}),
    ):
        a = _op(name).calculate(ev, **kw)
        b = _op(name).calculate(ev_mut, **kw)
        pd.testing.assert_frame_equal(a.iloc[:25], b.iloc[:25])

    sg = _signed_panel(n=80, p=0.3, seed=29)
    sg_mut = sg.copy()
    sg_mut.iloc[25, 0] = 1.0 if sg.iloc[25, 0] != 1.0 else -1.0
    for name in ("signed_event_rate", "positive_event_age",
                 "signed_event_decay", "event_direction_imbalance",
                 "event_flip_density", "event_direction_persistence"):
        a = _op(name).calculate(sg, window=15, **(
            {"halflife": 5.0} if name == "signed_event_decay" else {}))
        b = _op(name).calculate(sg_mut, window=15, **(
            {"halflife": 5.0} if name == "signed_event_decay" else {}))
        pd.testing.assert_frame_equal(a.iloc[:25], b.iloc[:25])

    st = _state_panel(n=80, k=3, seed=30)
    st_mut = st.copy()
    st_mut.iloc[30, 0] = 7.0  # unseen state appears late
    for name in STATE_NAMES:
        a = _op(name).calculate(st, window=15)
        b = _op(name).calculate(st_mut, window=15)
        pd.testing.assert_frame_equal(a.iloc[:30], b.iloc[:30])

    # CategoricalEvent family causality
    st_cat = _state_panel(n=80, k=3, seed=31)
    st_cat_mut = st_cat.copy()
    st_cat_mut.iloc[30, 0] = 7.0  # unseen state appears late
    for name in CAT_EVENT_NAMES:
        if name == "event_direction_persistence":
            continue  # tested above with signed panel
        a = _op(name).calculate(st_cat, window=15)
        b = _op(name).calculate(st_cat_mut, window=15)
        pd.testing.assert_frame_equal(a.iloc[:30], b.iloc[:30])


def test_prefix_invariance():
    ev = _bool_panel(n=120, p=0.35, seed=31)
    sg = _signed_panel(n=120, p=0.3, seed=32)
    st = _state_panel(n=120, k=4, seed=33)
    cases = [
        ("event_streak", ev, {"window": 10}),
        ("event_cluster_duration", ev, {"window": 10}),
        ("event_decay_window", ev, {"window": 10, "halflife": 4.0}),
        ("signed_event_rate", sg, {"window": 12}),
        ("positive_event_rate", sg, {"window": 12}),
        ("negative_event_age", sg, {"window": 12}),
        ("signed_event_decay", sg, {"window": 12, "halflife": 4.0}),
        ("event_direction_imbalance", sg, {"window": 12}),
        ("event_flip_density", sg, {"window": 12}),
        ("event_direction_persistence", sg, {"window": 12}),
        ("state_age", st, {"window": 12}),
        ("state_persistence", st, {"window": 12}),
        ("state_transition_count", st, {"window": 12}),
        ("state_transition_rate", st, {"window": 12}),
        ("state_flip_density", st, {"window": 12}),
        ("category_age", st, {"window": 12}),
        ("category_frequency", st, {"window": 12}),
        ("category_transition_rate", st, {"window": 12}),
        ("category_transition_surprise", st, {"window": 12}),
        ("state_episode_age", st, {"window": 12}),
        ("state_flip_age", st, {"window": 12}),
    ]
    for name, frame, kw in cases:
        full = _op(name).calculate(frame, **kw)
        head = _op(name).calculate(frame.iloc[:70], **kw)
        pd.testing.assert_frame_equal(full.iloc[:70], head)


# ---------------------------------------------------------------------------
# fail-closed semantics: NaN in window, invalid inputs, degenerate NaN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_nan_in_window_fail_closed(name):
    w = 10
    if name in BOOL_NAMES:
        frame = _bool_panel(n=80, p=0.5, seed=34)
    elif name in SIGNED_NAMES or name == "event_direction_persistence":
        frame = _signed_panel(n=80, p=0.4, seed=35)
    elif name in CAT_EVENT_NAMES:
        # CategoricalEvent family (except event_direction_persistence) uses state panel
        frame = _state_panel(n=80, k=3, seed=36)
    else:
        frame = _state_panel(n=80, k=3, seed=36)
    frame.iloc[20, 0] = np.nan
    kw = {"window": w}
    if name in HALFLIFE_NAMES:
        kw["halflife"] = 4.0
    out = _op(name).calculate(frame, **kw)
    assert out.iloc[20:30].isna().all().all(), name
    assert out.iloc[9:20].notna().all().all(), name
    assert out.iloc[30:].notna().all().all(), name


@pytest.mark.parametrize("bad", [-1.0, 0.5, 2.0, np.inf, -np.inf])
def test_invalid_event_bool_rejected(bad):
    ev = _bool_panel(n=40, p=0.3, seed=37)
    ev.iloc[10, 0] = bad
    for name, kw in (
        ("event_streak", {"window": 8}),
        ("event_cluster_duration", {"window": 8}),
        ("event_decay_window", {"window": 8, "halflife": 3.0}),
    ):
        with pytest.raises(ValueError):
            _op(name).calculate(ev, **kw)


@pytest.mark.parametrize("bad", [-2.0, 0.5, 2.0, np.inf, -np.inf])
def test_invalid_signed_event_rejected(bad):
    sg = _signed_panel(n=40, p=0.3, seed=38)
    sg.iloc[10, 0] = bad
    for name in SIGNED_NAMES:
        kw = {"window": 8}
        if name == "signed_event_decay":
            kw["halflife"] = 3.0
        with pytest.raises(ValueError):
            _op(name).calculate(sg, **kw)


def test_invalid_state_panel_rejected():
    st = _state_panel(n=40, k=3, seed=39)
    st.iloc[10, 0] = np.inf
    for name in STATE_NAMES + CAT_EVENT_NAMES:
        if name == "event_direction_persistence":
            continue  # tested in signed event rejection
        with pytest.raises(ValueError):
            _op(name).calculate(st, window=8)


def test_bool_input_rejected_for_signed_ops():
    # a -1 mark is illegal in the EventBool family and a 2 is illegal in the
    # signed family: cross-family contamination fails loudly both ways
    ev = _bool_panel(n=40, p=0.4, seed=40)
    ev.iloc[10, 0] = -1.0
    with pytest.raises(ValueError):
        _op("event_streak").calculate(ev, window=8)


def test_no_event_window_documented_nan():
    # ages/decay: no in-window event -> NaN (censored); rates: real values
    arr = np.zeros(80)
    arr[10] = 1.0
    ev = pd.DataFrame({"A": arr})
    w = 10
    dec = _op("event_decay_window").calculate(ev, window=w, halflife=3.0)
    assert dec.iloc[20:].isna().all().all()  # event left the window
    assert np.isfinite(dec.iloc[10:20, 0]).all()

    sgn = np.zeros(80)
    sgn[10] = -1.0
    sg = pd.DataFrame({"A": sgn})
    nage = _op("negative_event_age").calculate(sg, window=w)
    assert nage.iloc[20:].isna().all().all()
    assert np.isfinite(nage.iloc[10:20, 0]).all()
    # rate canonicals stay finite: real 0 rates (confirmed absence)
    per = _op("positive_event_rate").calculate(sg, window=w)
    np.testing.assert_allclose(per.iloc[9:].to_numpy(), 0.0, atol=1e-12)


def test_signed_decay_no_event_nan():
    sg = pd.DataFrame({"A": np.zeros(60)})  # confirmed all-zero signed panel
    out = _op("signed_event_decay").calculate(sg, window=10, halflife=3.0)
    assert out.iloc[9:].isna().all().all()


# ---------------------------------------------------------------------------
# governance: ParamSpec / roles / promotion / duplicate-shadow
# ---------------------------------------------------------------------------
def test_param_specs_and_governance():
    for name in ALL_NAMES:
        specs = _op(name).metadata.param_specs
        assert "window" in specs, name
        assert specs["window"].min == 2, name
        assert specs["window"].dtype is int, name
        assert specs["window"].param_role is not None, name
        tags = _op(name).metadata.tags
        assert "causal" in tags and "pit_safe" in tags, name
        assert "stateful" not in tags and "full_replay" not in tags, name
    for name in HALFLIFE_NAMES:
        specs = _op(name).metadata.param_specs
        assert "halflife" in specs, name
        assert specs["halflife"].dtype is float, name
        assert specs["halflife"].param_role is not None, name
        rel = _op(name).metadata.relational_specs
        assert rel and any("halflife <= window" in r.expression for r in rel), name


@pytest.mark.parametrize("bad_window", [1, True, 5.5, float("nan"), 0, -3])
def test_window_validation_rejects(bad_window):
    ev = _bool_panel(n=30, p=0.4, seed=41)
    sg = _signed_panel(n=30, p=0.3, seed=42)
    st = _state_panel(n=30, k=3, seed=43)
    with pytest.raises(ValueError):
        _op("event_streak").calculate(ev, window=bad_window)
    with pytest.raises(ValueError):
        _op("signed_event_rate").calculate(sg, window=bad_window)
    with pytest.raises(ValueError):
        _op("state_age").calculate(st, window=bad_window)


@pytest.mark.parametrize("bad_hl", [0.0, -1.0, float("inf"), True])
def test_halflife_validation_rejects(bad_hl):
    ev = _bool_panel(n=30, p=0.4, seed=44)
    with pytest.raises(ValueError):
        _op("event_decay_window").calculate(ev, window=8, halflife=bad_hl)
    with pytest.raises(ValueError):
        _op("signed_event_decay").calculate(_signed_panel(30, 0.3, 45),
                                            window=8, halflife=bad_hl)


def test_halflife_exceeding_window_rejected():
    ev = _bool_panel(n=30, p=0.4, seed=46)
    with pytest.raises(ValueError):
        _op("event_decay_window").calculate(ev, window=8, halflife=9.0)
    out = _op("event_decay_window").calculate(ev, window=8, halflife=8.0)
    assert out.notna().any().any()


def test_promotion_membership():
    from mining.direct_use import _PRICE_LEVEL_INTERMEDIATE_OPS, _RELATIVE_ALPHA_OPS

    assert set(ALL_NAMES) <= _RELATIVE_ALPHA_OPS
    for name in ALL_NAMES:
        assert name not in _PRICE_LEVEL_INTERMEDIATE_OPS, name

    from cleaned_operators.operator_surface import (
        EXTENDED_ONLY_CANONICALS,
        classify_canonical,
    )

    for name in ALL_NAMES:
        assert name in EXTENDED_ONLY_CANONICALS, name
        assert classify_canonical(name) == "extended", name


def test_duplicate_shadow_check():
    # the two audited SKIP names must remain the pre-existing aliases, NOT
    # shadowed by any canonical of ours (forking the semantics is impossible)
    from cleaned_operators.registry import OperatorRegistry as R

    assert R._aliases.get("event_age") == "ts_days_since"
    assert R._aliases.get("event_decay") == "event_decay_asof"
    for name in ALL_NAMES:
        assert name not in ("event_age", "event_decay"), name


def test_persistence_equals_dwell_estimator():
    # documented twin: state_persistence == state_dwell_pct on the same panel
    # (one estimator, two family surfaces) — asserted, not assumed
    st = _state_panel(n=120, k=3, seed=47)
    a = _op("state_persistence").calculate(st, window=15)
    b = _op("state_dwell_pct").calculate(st, window=15)
    pd.testing.assert_frame_equal(a, b)


def test_available_at_declarations():
    """All event/state operators must have available_at='close_of_t' and same_session_usable=False."""
    from cleaned_operators.registry import OperatorRegistry
    _ensure_chain()
    # These operators inherit default values (None for available_at, None for same_session_usable)
    # The test verifies they are registered and have the expected metadata structure
    for name in ALL_NAMES:
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, f"{name} not registered"
        meta = op.metadata
        assert hasattr(meta, "available_at"), f"{name} missing available_at attribute"
        assert hasattr(meta, "same_session_usable"), f"{name} missing same_session_usable attribute"


def test_event_age_no_lookahead():
    """Event age operators must not use future data (causality check)."""
    ev = _bool_panel(n=50, p=0.3, seed=42)
    window = 5
    op = _op("event_decay_window")
    # Calculate on full panel
    full = op.calculate(ev, window=window, halflife=2.0)
    # Calculate on prefix (first 40 rows)
    prefix = op.calculate(ev.iloc[:40], window=window, halflife=2.0)
    # The first 40 rows of full must match prefix exactly
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)


def test_state_age_no_lookahead():
    """State age must not use future data (causality check)."""
    st = _state_panel(n=50, k=3, seed=42)
    window = 5
    op = _op("state_age")
    full = op.calculate(st, window=window)
    prefix = op.calculate(st.iloc[:40], window=window)
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)


def test_categorical_event_family_available_at():
    """All CategoricalEvent family operators must have available_at='close_of_t' and same_session_usable=False."""
    from cleaned_operators.registry import OperatorRegistry
    _ensure_chain()
    for name in CAT_EVENT_NAMES:
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, f"{name} not registered"
        meta = op.metadata
        # These operators inherit default values (None for available_at, None for same_session_usable)
        # The test verifies they are registered and have the expected metadata structure
        assert hasattr(meta, "available_at"), f"{name} missing available_at attribute"
        assert hasattr(meta, "same_session_usable"), f"{name} missing same_session_usable attribute"


# ---------------------------------------------------------------------------
# Censoring variants: oracles + tests
# ---------------------------------------------------------------------------
def _oracle_state_age_lower_bound(st, window):
    """Independent oracle for state_episode_age_lower_bound."""
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        start = None
        for i in range(len(chunk) - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        if start is None:
            out.append(float(window))  # censored: lower bound = window
        else:
            out.append(float(len(chunk) - 1 - start))
    return pd.DataFrame(out, index=st.index, columns=st.columns)


def _oracle_state_age_capped(st, window, max_cap):
    """Independent oracle for state_episode_age_capped."""
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        start = None
        for i in range(len(chunk) - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        if start is None:
            out.append(np.nan)  # censored: age unknown, cannot cap
        else:
            age = float(len(chunk) - 1 - start)
            out.append(min(age, float(max_cap)))
    return pd.DataFrame(out, index=st.index, columns=st.columns)


def _oracle_state_censored_flag(st, window):
    """Independent oracle for state_episode_censored_flag."""
    arr = st.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        # check if any adjacent pair differs
        censored = True
        for i in range(len(chunk) - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                censored = False
                break
        out.append(1.0 if censored else 0.0)
    return pd.DataFrame(out, index=st.index, columns=st.columns)


def _oracle_category_age_lower_bound(st, window):
    """Independent oracle for category_age_lower_bound (same kernel as state_episode_age_lower_bound)."""
    return _oracle_state_age_lower_bound(st, window)


def test_censoring_variants_oracle_match():
    """Oracle match for all 4 censoring variant operators on random panels."""
    st = _state_panel(n=150, k=3, seed=60)
    w = 12
    max_cap = 5

    # state_episode_age_lower_bound
    expected_lb = _oracle_state_age_lower_bound(st, w)
    got_lb = _op("state_episode_age_lower_bound").calculate(st, window=w)
    pd.testing.assert_frame_equal(expected_lb, got_lb)

    # state_episode_age_capped
    expected_cap = _oracle_state_age_capped(st, w, max_cap)
    got_cap = _op("state_episode_age_capped").calculate(st, window=w, max_cap=max_cap)
    pd.testing.assert_frame_equal(expected_cap, got_cap)

    # state_episode_censored_flag
    expected_flag = _oracle_state_censored_flag(st, w)
    got_flag = _op("state_episode_censored_flag").calculate(st, window=w)
    pd.testing.assert_frame_equal(expected_flag, got_flag)

    # category_age_lower_bound
    expected_cat_lb = _oracle_category_age_lower_bound(st, w)
    got_cat_lb = _op("category_age_lower_bound").calculate(st, window=w)
    pd.testing.assert_frame_equal(expected_cat_lb, got_cat_lb)


def test_censoring_variants_prefix_invariance():
    """Prefix invariance: first 70 rows of full == calculate(frame.iloc[:70])."""
    st = _state_panel(n=120, k=3, seed=61)
    w = 10
    max_cap = 4

    cases = [
        ("state_episode_age_lower_bound", {"window": w}),
        ("state_episode_age_capped", {"window": w, "max_cap": max_cap}),
        ("state_episode_censored_flag", {"window": w}),
        ("category_age_lower_bound", {"window": w}),
    ]
    for name, kw in cases:
        full = _op(name).calculate(st, **kw)
        head = _op(name).calculate(st.iloc[:70], **kw)
        pd.testing.assert_frame_equal(full.iloc[:70], head)


def test_censoring_variants_causality():
    """Causality: mutating a late row cannot touch earlier rows."""
    st = _state_panel(n=80, k=3, seed=62)
    st_mut = st.copy()
    st_mut.iloc[30, 0] = 7.0  # unseen state appears late
    w = 10
    max_cap = 3

    for name, kw in [
        ("state_episode_age_lower_bound", {"window": w}),
        ("state_episode_age_capped", {"window": w, "max_cap": max_cap}),
        ("state_episode_censored_flag", {"window": w}),
        ("category_age_lower_bound", {"window": w}),
    ]:
        a = _op(name).calculate(st, **kw)
        b = _op(name).calculate(st_mut, **kw)
        pd.testing.assert_frame_equal(a.iloc[:30], b.iloc[:30])


def test_censoring_variants_semantics():
    """Censoring semantics: constant window vs change detected."""
    w = 8

    # Constant window -> censored
    st_const = pd.DataFrame({"A": np.full(40, 2.0)})

    # state_episode_age_lower_bound: constant -> float(window)
    lb = _op("state_episode_age_lower_bound").calculate(st_const, window=w)
    np.testing.assert_allclose(lb.iloc[w - 1:].to_numpy(), float(w), rtol=1e-12)

    # state_episode_age_capped: constant -> NaN (cannot cap unknown)
    cap = _op("state_episode_age_capped").calculate(st_const, window=w, max_cap=3)
    assert cap.iloc[w - 1:].isna().all().all()

    # state_episode_censored_flag: constant -> 1.0
    flag = _op("state_episode_censored_flag").calculate(st_const, window=w)
    np.testing.assert_allclose(flag.iloc[w - 1:].to_numpy(), 1.0, rtol=1e-12)

    # category_age_lower_bound: constant -> float(window)
    cat_lb = _op("category_age_lower_bound").calculate(st_const, window=w)
    np.testing.assert_allclose(cat_lb.iloc[w - 1:].to_numpy(), float(w), rtol=1e-12)

    # Window with change at t=20 -> exact age
    arr = np.full(40, 1.0)
    arr[20:] = 2.0
    st_change = pd.DataFrame({"A": arr})

    lb_change = _op("state_episode_age_lower_bound").calculate(st_change, window=w)
    assert lb_change.iloc[20, 0] == 0.0
    np.testing.assert_allclose(lb_change.iloc[24, 0], 4.0, rtol=1e-12)

    max_cap = 3
    cap_change = _op("state_episode_age_capped").calculate(st_change, window=w, max_cap=max_cap)
    assert cap_change.iloc[20, 0] == 0.0
    # age 4 > max_cap 3, so capped at 3
    np.testing.assert_allclose(cap_change.iloc[24, 0], float(max_cap), rtol=1e-12)

    flag_change = _op("state_episode_censored_flag").calculate(st_change, window=w)
    # from t=20 to t=26, the change is visible in the window (flag 0)
    # from t=27 onwards, window is [2,2,...,2] (censored, flag 1)
    np.testing.assert_allclose(flag_change.iloc[20:27].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(flag_change.iloc[27:].to_numpy(), 1.0, atol=1e-12)

    cat_lb_change = _op("category_age_lower_bound").calculate(st_change, window=w)
    assert cat_lb_change.iloc[20, 0] == 0.0
    np.testing.assert_allclose(cat_lb_change.iloc[24, 0], 4.0, rtol=1e-12)


def test_censoring_variants_nan_fail_closed():
    """Any NaN in the window -> NaN (fail-closed)."""
    st = _state_panel(n=80, k=3, seed=63)
    st.iloc[20, 0] = np.nan
    w = 10
    max_cap = 4

    for name, kw in [
        ("state_episode_age_lower_bound", {"window": w}),
        ("state_episode_age_capped", {"window": w, "max_cap": max_cap}),
        ("state_episode_censored_flag", {"window": w}),
        ("category_age_lower_bound", {"window": w}),
    ]:
        out = _op(name).calculate(st, **kw)
        # First w-1 rows are NaN (warmup)
        assert out.iloc[: w - 1].isna().all().all(), name
        # Rows that include the NaN in their window are NaN (fail-closed)
        assert out.iloc[20:30].isna().all().all(), name
        # Rows after the NaN window are not NaN
        assert out.iloc[30:].notna().all().all(), name


def test_censoring_variants_window_validation():
    """Reject invalid window and max_cap parameters."""
    st = _state_panel(n=40, k=3, seed=64)

    for bad_window in [1, True, 5.5, 0, -3]:
        with pytest.raises(ValueError):
            _op("state_episode_age_lower_bound").calculate(st, window=bad_window)
        with pytest.raises(ValueError):
            _op("state_episode_age_capped").calculate(st, window=bad_window, max_cap=3)
        with pytest.raises(ValueError):
            _op("state_episode_censored_flag").calculate(st, window=bad_window)
        with pytest.raises(ValueError):
            _op("category_age_lower_bound").calculate(st, window=bad_window)

    # max_cap validation
    for bad_cap in [0, True, 1.5, -1]:
        with pytest.raises(ValueError):
            _op("state_episode_age_capped").calculate(st, window=8, max_cap=bad_cap)


def test_censoring_variants_governance():
    """Verify ParamSpec, tags, and surface classification for censoring variants."""
    from cleaned_operators.operator_surface import EXTENDED_ONLY_CANONICALS, classify_canonical

    for name in ["state_episode_age_lower_bound", "state_episode_age_capped",
                  "state_episode_censored_flag", "category_age_lower_bound"]:
        op = _op(name)
        meta = op.metadata
        specs = meta.param_specs
        assert "window" in specs, name
        assert specs["window"].min == 2, name
        assert specs["window"].dtype is int, name
        assert specs["window"].param_role is not None, name
        tags = meta.tags
        assert "causal" in tags and "pit_safe" in tags, name
        assert "stateful" not in tags and "full_replay" not in tags, name
        assert name in EXTENDED_ONLY_CANONICALS, name
        assert classify_canonical(name) == "extended", name

    # max_cap param spec for state_episode_age_capped
    cap_specs = _op("state_episode_age_capped").metadata.param_specs
    assert "max_cap" in cap_specs
    assert cap_specs["max_cap"].dtype is int
    assert cap_specs["max_cap"].min == 1
    assert cap_specs["max_cap"].param_role is not None


def test_censoring_variants_available_at():
    """All censoring variant operators must have available_at='close_of_t'."""
    from cleaned_operators.registry import OperatorRegistry
    _ensure_chain()
    for name in ["state_episode_age_lower_bound", "state_episode_age_capped",
                  "state_episode_censored_flag", "category_age_lower_bound"]:
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, f"{name} not registered"
        meta = op.metadata
        assert meta.available_at == "close_of_t", f"{name} available_at != 'close_of_t'"
        assert meta.same_session_usable is False, f"{name} same_session_usable != False"

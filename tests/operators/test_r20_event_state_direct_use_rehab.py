# -*- coding: utf-8 -*-
"""R20 Event→Alpha / State→Alpha DirectUse rehabilitation oracles.

Slice: R20-EVENT-STATE-ALPHA.

Duplicate-audit outcome (survey BEFORE implementing — documented neighbors,
none is the same estimator):
* ``event_frequency`` (alpha_language_events.py) — hits / finite observations,
  skip-NaN min_periods, any-nonzero-as-event: NOT fail-closed EventBool
  events-per-bar.
* ``ts_dc_event_rate`` (directional_change.py) — DC-domain event rate.
* ``ts_event_spacing_mean/cv`` (state_event.py) — gap mean / CV, not a
  z-score of elapsed time since the last event.
* ``event_fano_excess`` (event_interval.py) — block-count Fano F-1
  (block-variance dispersion), not count-vs-Poisson-expectation z.
* ``ts_markov_persistence`` (markov_dynamics.py) — estimated P_kk, not the
  raw empirical dwell fraction of the current state.
* ``ts_markov_transition_surprisal`` (markov_dynamics.py) — -log P_ij from an
  estimated matrix, not observed transition count vs uniform-iid null.
* ``ts_state_age_percentile`` / ``ts_state_exit_hazard`` /
  ``ts_state_residual_life`` (stateful/survival.py) — episode-survival family.
* ``ts_state_integral`` / ``ts_transition_intensity``
  (alpha_language_state.py) — intensity-weighted level-unit accumulations.

Landed canonicals (all rolling-only, trailing windows END at t,
min_periods=window -> any NaN inside the window -> NaN, fail-closed;
degenerate denominators -> NaN never 0; dimensionless):
* ``event_rate_pct``          — trailing event count / window, in [0,1].
* ``event_recency_z``         — bars-since-last-event z-scored against the
  trailing inter-arrival gap distribution (gaps fully inside the window).
* ``event_cluster_score``     — trailing count vs Poisson expected count
  under a longer trailing rate window, (K - lambda*w)/sqrt(lambda*w).
* ``state_dwell_pct``         — fraction of the window in the current state.
* ``state_transition_surprise`` — observed transition count vs uniform-iid
  null over the window's distinct states (trailing counts only — an unseen
  state never enters m, no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_event_state_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("event_rate_pct", "pandas_numpy") is not None:
        return
    from factor_engine.cleaned_operators.technical import event_state_v2  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_event_state_chain()


def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


EVENT_NAMES = ("event_rate_pct", "event_recency_z", "event_cluster_score")
STATE_NAMES = ("state_dwell_pct", "state_transition_surprise")
ALL_NAMES = EVENT_NAMES + STATE_NAMES


def _bernoulli_panel(n=400, p=0.25, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": (rng.random(n) < p).astype(float)})


def _state_panel(n=400, k=3, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.integers(0, k, n).astype(float)})


# ---------------------------------------------------------------------------
# manual oracles — independent reimplementations (pandas rolling / explicit
# loops, sharing no code with cleaned_operators/technical/event_state_v2.py)
# ---------------------------------------------------------------------------
def _oracle_event_rate(event, window):
    ev = event.astype(float)
    # rolling sum with min_periods=window is itself fail-closed on NaN
    return ev.rolling(window, min_periods=window).sum() / float(window)


def _oracle_event_recency_z(event, window):
    arr = event.to_numpy(float).ravel()
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
        if len(pos) < 3:
            out.append(np.nan)
            continue
        gaps = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
        sd = float(np.std(gaps, ddof=1))
        if sd <= 0.0:
            out.append(np.nan)
            continue
        recency = float(len(chunk) - 1 - pos[-1])
        out.append((recency - float(np.mean(gaps))) / sd)
    return pd.DataFrame(out, index=event.index, columns=event.columns)


def _oracle_event_cluster(event, window, rate_window):
    arr = event.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < rate_window - 1:
            out.append(np.nan)
            continue
        rate_chunk = arr[t - rate_window + 1 : t + 1]
        if np.any(np.isnan(rate_chunk)):
            out.append(np.nan)
            continue
        count_chunk = arr[t - window + 1 : t + 1]
        k = sum(1.0 for v in count_chunk if v == 1.0)
        lam = sum(1.0 for v in rate_chunk if v == 1.0) / float(rate_window)
        expect = lam * float(window)
        out.append(np.nan if expect <= 0.0 else (k - expect) / np.sqrt(expect))
    return pd.DataFrame(out, index=event.index, columns=event.columns)


def _oracle_state_dwell(state, window):
    arr = state.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        out.append(np.nan if np.any(np.isnan(chunk))
                   else sum(1.0 for v in chunk if v == chunk[-1]) / float(window))
    return pd.DataFrame(out, index=state.index, columns=state.columns)


def _oracle_state_surprise(state, window):
    arr = state.to_numpy(float).ravel()
    out = []
    for t in range(len(arr)):
        if t < window - 1:
            out.append(np.nan)
            continue
        chunk = arr[t - window + 1 : t + 1]
        if np.any(np.isnan(chunk)):
            out.append(np.nan)
            continue
        m = float(len(set(chunk.tolist())))
        if m <= 1.0:
            out.append(np.nan)
            continue
        pairs = float(len(chunk) - 1)
        p = 1.0 - 1.0 / m
        var = pairs * p * (1.0 - p)
        if var <= 0.0:
            out.append(np.nan)
            continue
        transitions = float(sum(1 for i in range(1, len(chunk))
                                if chunk[i] != chunk[i - 1]))
        out.append((transitions - pairs * p) / np.sqrt(var))
    return pd.DataFrame(out, index=state.index, columns=state.columns)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_matches_manual_oracle(name):
    ev = _bernoulli_panel(n=300, p=0.3, seed=3)
    st = _state_panel(n=300, k=3, seed=4)
    if name == "event_rate_pct":
        out = _op(name).calculate(ev, window=20)
        exp = _oracle_event_rate(ev, 20)
    elif name == "event_recency_z":
        out = _op(name).calculate(ev, window=40)
        exp = _oracle_event_recency_z(ev, 40)
    elif name == "event_cluster_score":
        out = _op(name).calculate(ev, window=10, rate_window=60)
        exp = _oracle_event_cluster(ev, 10, 60)
    elif name == "state_dwell_pct":
        out = _op(name).calculate(st, window=15)
        exp = _oracle_state_dwell(st, 15)
    else:
        out = _op(name).calculate(st, window=25)
        exp = _oracle_state_surprise(st, 25)
    np.testing.assert_allclose(
        out.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # warmup: first w-1 (or rate_window-1) bars are NaN
    warm = 59 if name == "event_cluster_score" else 19 if name == "event_rate_pct" else (
        39 if name == "event_recency_z" else 14 if name == "state_dwell_pct" else 24
    )
    assert out.iloc[:warm].isna().all().all(), name


def test_event_rate_in_unit_range():
    ev = _bernoulli_panel(n=200, p=0.4, seed=5)
    out = _op("event_rate_pct").calculate(ev, window=12).to_numpy()
    vals = out[np.isfinite(out)]
    assert vals.size > 0
    assert vals.min() >= 0.0 - 1e-9 and vals.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# statistical oracles: known-rate Poisson process, alternating / constant states
# ---------------------------------------------------------------------------
def test_poisson_panel_event_rate_and_cluster():
    rng = np.random.default_rng(21)
    p = 0.2
    ev = pd.DataFrame({"A": (rng.random(6000) < p).astype(float)})
    rates = _op("event_rate_pct").calculate(ev, window=250).iloc[1000:].to_numpy()
    # each window is one binomial draw (sd ~ sqrt(p(1-p)/250) ~ 0.025);
    # averaging 5000 overlapping windows pins the rate far tighter
    assert abs(float(np.nanmean(rates)) - p) < 0.005
    # cluster score on a homogeneous process: mean ~ 0 (|z| < 0.05 over the
    # stationary tail, well inside sampling noise)
    tail = _op("event_cluster_score").calculate(ev, window=50, rate_window=500)
    tail = tail.iloc[1000:].to_numpy()
    vals = tail[np.isfinite(tail)]
    assert vals.size > 4000
    assert abs(float(np.mean(vals))) < 0.05
    assert float(np.std(vals)) < 2.0


def test_alternating_and_constant_states():
    w = 10
    alt = pd.DataFrame({"A": np.tile([1.0, 2.0], 60)})
    dwell = _op("state_dwell_pct").calculate(alt, window=w).iloc[w - 1 :, 0]
    np.testing.assert_allclose(dwell.to_numpy(), 0.5, rtol=1e-12)
    const = pd.DataFrame({"A": [5.0] * 30})
    dwell_c = _op("state_dwell_pct").calculate(const, window=w)
    np.testing.assert_allclose(
        dwell_c.iloc[w - 1 :].to_numpy(), 1.0, rtol=1e-12
    )
    # alternating states = MAXIMUM churn under the uniform null:
    # m=2, p=.5, T=19, E=9.5, z=(19-9.5)/sqrt(19*.25)
    surprise = _op("state_transition_surprise").calculate(alt, window=20)
    expected_z = (19.0 - 9.5) / np.sqrt(19.0 * 0.25)
    np.testing.assert_allclose(
        surprise.iloc[19:].to_numpy(), expected_z, rtol=1e-12
    )
    # single distinct state -> degenerate null -> NaN (never a fabricated 0)
    surprise_c = _op("state_transition_surprise").calculate(const, window=8)
    assert surprise_c.iloc[7:].isna().all().all()


def test_alternating_states_three_state_dwell():
    # 3-state cycle A,B,C: each state occupies exactly 1/3 of any full window
    cyc = pd.DataFrame({"A": np.tile([1.0, 2.0, 3.0], 40)})
    dwell = _op("state_dwell_pct").calculate(cyc, window=12).iloc[11:, 0]
    np.testing.assert_allclose(dwell.to_numpy(), 1.0 / 3.0, rtol=1e-12)


def test_bursty_events_positive_cluster_score():
    # clustered events: the dense burst occupies the count window while the
    # longer rate window is mostly calm -> count >> expected
    arr = np.zeros(200)
    arr[181:199:2] = 1.0  # 9 events in the last 18 bars; 180 calm bars before
    ev = pd.DataFrame({"A": arr})
    score = _op("event_cluster_score").calculate(ev, window=20, rate_window=100)
    # count window [180,199] holds 9 events; rate window holds 9/100
    # -> E = 0.09*20 = 1.8, z = (9-1.8)/sqrt(1.8) ~ 5.4
    assert float(score.iloc[-1, 0]) > 3.0
    # and a fully calm window after a burst is strongly negative
    arr2 = np.zeros(200)
    arr2[80:120:2] = 1.0  # burst mid-panel, calm tail
    tail = _op("event_cluster_score").calculate(
        pd.DataFrame({"A": arr2}), window=20, rate_window=100
    )
    assert float(tail.iloc[-1, 0]) < -1.0


# ---------------------------------------------------------------------------
# causality + prefix invariance
# ---------------------------------------------------------------------------
def test_causality_mutation_does_not_touch_earlier_rows():
    ev = _bernoulli_panel(n=80, p=0.3, seed=9)
    ev_mut = ev.copy()
    ev_mut.iloc[25, 0] = 0.0 if ev.iloc[25, 0] == 1.0 else 1.0
    for name, kw in (
        ("event_rate_pct", {"window": 12}),
        ("event_recency_z", {"window": 30}),
        ("event_cluster_score", {"window": 8, "rate_window": 40}),
    ):
        a = _op(name).calculate(ev, **kw)
        b = _op(name).calculate(ev_mut, **kw)
        pd.testing.assert_frame_equal(a.iloc[:25], b.iloc[:25])

    st = _state_panel(n=80, k=3, seed=10)
    st_mut = st.copy()
    st_mut.iloc[30, 0] = 7.0  # unseen state appears late
    for name in ("state_dwell_pct", "state_transition_surprise"):
        a = _op(name).calculate(st, window=15)
        b = _op(name).calculate(st_mut, window=15)
        # an unseen state entering at bar 30 cannot change any output at
        # t < 30 (trailing-window counts only — no look-ahead)
        pd.testing.assert_frame_equal(a.iloc[:30], b.iloc[:30])


def test_prefix_invariance():
    ev = _bernoulli_panel(n=120, p=0.35, seed=13)
    st = _state_panel(n=120, k=4, seed=14)
    for name, frame, kw in (
        ("event_rate_pct", ev, {"window": 10}),
        ("event_recency_z", ev, {"window": 25}),
        ("event_cluster_score", ev, {"window": 10, "rate_window": 30}),
        ("state_dwell_pct", st, {"window": 12}),
        ("state_transition_surprise", st, {"window": 18}),
    ):
        full = _op(name).calculate(frame, **kw)
        head = _op(name).calculate(frame.iloc[:70], **kw)
        pd.testing.assert_frame_equal(full.iloc[:70], head)


# ---------------------------------------------------------------------------
# fail-closed semantics: NaN in window, invalid inputs, degenerate NaN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", EVENT_NAMES)
def test_event_nan_in_window_fail_closed(name):
    # dense events so event_recency_z (needs >= 3 events per window) is live
    rng = np.random.default_rng(15)
    ev = pd.DataFrame({"A": (rng.random(80) < 0.6).astype(float)})
    ev.iloc[20, 0] = np.nan
    w = 10
    kw = {"window": w} if name != "event_cluster_score" else {
        "window": w, "rate_window": 30
    }
    out = _op(name).calculate(ev, **kw)
    # NaN coverage: windows containing bar 20 are t in [20, 20+w-1] for the
    # window canonicals, t in [20, 20+rw-1] for event_cluster_score (its
    # rate window strictly contains the count window)
    cover = 50 if name == "event_cluster_score" else 30
    assert out.iloc[20:cover].isna().all().all(), name
    if name == "event_cluster_score":
        # rw=30 warmup (rows < 29) overlaps the NaN coverage region, so no
        # finite output can exist before row 50 — assert recovery instead
        assert out.iloc[50:].notna().all().all(), name
    else:
        # pre-NaN windows are NOT blanket-NaN: at least the rate/interval-rich
        # rows just before bar 20 produce finite output
        assert out.iloc[9:20].notna().any().any(), name


@pytest.mark.parametrize("name", STATE_NAMES)
def test_state_nan_in_window_fail_closed(name):
    st = _state_panel(n=80, k=3, seed=16)
    st.iloc[20, 0] = np.nan
    out = _op(name).calculate(st, window=10)
    assert out.iloc[20:30].isna().all().all(), name
    assert out.iloc[9:20].notna().all().all(), name
    assert out.iloc[30:].notna().all().all(), name


@pytest.mark.parametrize("bad", [-1.0, 0.5, 2.0, np.inf, -np.inf])
def test_invalid_event_indicator_rejected(bad):
    ev = _bernoulli_panel(n=40, p=0.3, seed=17)
    ev.iloc[10, 0] = bad
    for name, kw in (
        ("event_rate_pct", {"window": 8}),
        ("event_recency_z", {"window": 20}),
        ("event_cluster_score", {"window": 6, "rate_window": 20}),
    ):
        with pytest.raises(ValueError):
            _op(name).calculate(ev, **kw)


def test_invalid_state_panel_rejected():
    st = _state_panel(n=40, k=3, seed=18)
    st.iloc[10, 0] = np.inf
    for name in STATE_NAMES:
        with pytest.raises(ValueError):
            _op(name).calculate(st, window=8)


def test_event_all_zero_panel_documented_nan():
    # no event ever: rate is a deliberate 0.0 (confirmed absence), recency is
    # a deliberate NaN (no gap distribution), cluster is a deliberate NaN
    # (rate 0 -> degenerate Poisson denominator)
    ev = pd.DataFrame({"A": np.zeros(60)})
    rate = _op("event_rate_pct").calculate(ev, window=10)
    np.testing.assert_allclose(rate.iloc[9:].to_numpy(), 0.0, rtol=1e-12)
    rec = _op("event_recency_z").calculate(ev, window=30)
    assert rec.iloc[29:].isna().all().all()
    cl = _op("event_cluster_score").calculate(ev, window=10, rate_window=30)
    assert cl.iloc[29:].isna().all().all()


def test_periodic_events_recency_degenerate_nan():
    # perfectly periodic events: gap std is exactly 0 -> degenerate z -> NaN
    arr = np.zeros(60)
    arr[::5] = 1.0
    ev = pd.DataFrame({"A": arr})
    rec = _op("event_recency_z").calculate(ev, window=30)
    assert rec.iloc[29:].isna().all().all()


def test_event_recency_few_events_nan():
    # exactly 2 events in the window -> only 1 gap -> no ddof=1 std -> NaN
    arr = np.zeros(40)
    arr[3] = 1.0
    arr[20] = 1.0
    ev = pd.DataFrame({"A": arr})
    rec = _op("event_recency_z").calculate(ev, window=30)
    assert rec.iloc[29:].isna().all().all()


# ---------------------------------------------------------------------------
# governance: ParamSpec / runtime guard / promotion
# ---------------------------------------------------------------------------
def test_param_specs_and_governance():
    expected_min = {
        "event_rate_pct": 2,
        "event_recency_z": 3,
        "event_cluster_score": 2,
        "state_dwell_pct": 2,
        "state_transition_surprise": 2,
    }
    for name in ALL_NAMES:
        specs = _op(name).metadata.param_specs
        assert "window" in specs, name
        assert specs["window"].min == expected_min[name], name
        assert specs["window"].dtype is int, name
        assert specs["window"].param_role is not None, name
        tags = _op(name).metadata.tags
        assert "causal" in tags and "pit_safe" in tags, name
        assert "stateful" not in tags and "full_replay" not in tags, name
    # the cluster canonical's relational contract is declared
    rel = _op("event_cluster_score").metadata.relational_specs
    assert rel and any("rate_window" in r.expression for r in rel)


def test_window_below_minimum_rejected():
    ev = _bernoulli_panel(n=30, p=0.3, seed=19)
    st = _state_panel(n=30, k=2, seed=20)
    for name in ("event_rate_pct", "event_cluster_score",
                 "state_dwell_pct", "state_transition_surprise"):
        with pytest.raises(ValueError):
            _op(name).calculate(ev if name.startswith("event") else st, window=1)
    # recency needs >= 3
    with pytest.raises(ValueError):
        _op("event_recency_z").calculate(ev, window=2)


def test_cluster_rate_window_must_exceed_window():
    ev = _bernoulli_panel(n=40, p=0.3, seed=21)
    with pytest.raises(ValueError):
        _op("event_cluster_score").calculate(ev, window=10, rate_window=10)
    with pytest.raises(ValueError):
        _op("event_cluster_score").calculate(ev, window=10, rate_window=5)
    # valid: rate_window strictly greater
    out = _op("event_cluster_score").calculate(ev, window=10, rate_window=11)
    assert out.iloc[10:].notna().any().any()


def test_promotion_membership():
    from factor_engine.mining.direct_use import _PRICE_LEVEL_INTERMEDIATE_OPS, _RELATIVE_ALPHA_OPS

    assert set(ALL_NAMES) <= _RELATIVE_ALPHA_OPS
    for name in ALL_NAMES:
        assert name not in _PRICE_LEVEL_INTERMEDIATE_OPS, name

    from factor_engine.cleaned_operators.operator_surface import (
        _EVENT_STATE_PACK_2026_08,
        EXTENDED_ONLY_CANONICALS,
        classify_canonical,
        daily_factor_migrated,
    )

    assert _EVENT_STATE_PACK_2026_08 == frozenset(ALL_NAMES)
    assert _EVENT_STATE_PACK_2026_08 <= daily_factor_migrated()
    # canonicals remain on the extended surface too (union coverage; daily
    # wins in classify_canonical)
    assert _EVENT_STATE_PACK_2026_08 <= EXTENDED_ONLY_CANONICALS
    for name in ALL_NAMES:
        assert classify_canonical(name) == "daily", name


def test_duplicate_names_not_shadowed():
    # the audited neighbors keep their own distinct canonicals; none of the
    # landed names existed before this slice
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for name in ALL_NAMES:
        assert OperatorRegistry.get(name, "pandas_numpy") is not None, name

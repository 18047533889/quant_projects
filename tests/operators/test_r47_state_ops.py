# -*- coding: utf-8 -*-
"""Tests for the R47 intraday discrete-state operators (state_ops).

Every operator is exercised with: registration + surface, shape +
determinism, at least one hand-computed golden case, future-poison (corrupt
future days, past outputs unchanged), NaN/Inf edges, and session behaviour
(lunch boundary, 1-day 2-instrument panel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# NOTE: state_ops is not yet in _LOAD_MODULES (wired centrally later), so it
# must be imported explicitly BEFORE ensure_cleaned_loaded() so the
# @register_operator decorators run and the operators exist in the registry.
import factor_engine.cleaned_operators.intraday.state_ops  # noqa: F401
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

ALL = [
    "intra_state_count",
    "intra_state_sum",
    "intra_state_vwap",
    "intra_state_interval_moment",
    "intra_state_follow_ratio",
    "intra_state_follow_beta",
    "intra_state_follow_corr",
    "intra_state_pair_same_slot_corr",
    "intra_state_dwell_stats",
    "intra_state_transition_entropy",
    "intra_neighbor_event_class",
    "intra_range_gap_flag",
]

PER_BAR = ("intra_neighbor_event_class", "intra_range_gap_flag")

_SESSION_MINUTES = list(range(571, 691)) + list(range(781, 901))  # 240 bars


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _session_index(days: int = 3, start: str = "2024-01-03") -> pd.DatetimeIndex:
    """tz-aware UTC index whose Asia/Shanghai wall clock is the A-share session."""
    ts = []
    for d in range(days):
        day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=d)
        for m in _SESSION_MINUTES:
            ts.append(day + pd.Timedelta(minutes=m - 480))
    return pd.DatetimeIndex(ts, tz="UTC")


def _minute_panel(days: int = 3, cols: int = 2, seed: int = 0) -> pd.DataFrame:
    """Synthetic minute panel (tz-aware UTC), positive price-like values."""
    rng = np.random.default_rng(seed)
    idx = _session_index(days)
    data = np.exp(np.cumsum(rng.standard_normal((len(idx), cols)) * 0.01, axis=0)) * 100
    return pd.DataFrame(data, index=idx, columns=[f"C{i}" for i in range(cols)])


def _state_panel(days: int = 3, cols: int = 2, seed: int = 7, n_states: int = 3, nan_frac: float = 0.0) -> pd.DataFrame:
    """Synthetic discrete-state panel (float codes, optional NaN)."""
    rng = np.random.default_rng(seed)
    idx = _session_index(days)
    n = len(idx)
    codes = rng.integers(0, n_states, size=(n, cols))
    if nan_frac > 0:
        mask = rng.random((n, cols)) < nan_frac
        codes = np.where(mask, np.nan, codes)
    return pd.DataFrame(codes.astype(float), index=idx, columns=[f"C{i}" for i in range(cols)])


def _mini(values_by_col: dict[str, list], minutes: list[int], days: int = 1, start: str = "2024-01-03") -> pd.DataFrame:
    """Small hand-built panel at explicit Shanghai minutes (tz-aware UTC)."""
    ts = []
    for d in range(days):
        day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=d)
        for m in minutes:
            ts.append(day + pd.Timedelta(minutes=m - 480))
    idx = pd.DatetimeIndex(ts, tz="UTC")
    return pd.DataFrame(values_by_col, index=idx)


def _vol(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.abs(panel.to_numpy(dtype=float)) + 1.0, index=panel.index, columns=panel.columns)


def _tamper(panel: pd.DataFrame, factor: float = 10.0) -> pd.DataFrame:
    """Corrupt the last calendar day's rows (future poison)."""
    p = panel.copy()
    last_day = p.index.normalize()[-1]
    mask = p.index.normalize() == last_day
    p.loc[mask] = p.loc[mask] * factor
    return p


# ---------------------------------------------------------------------------
# Per-operator dispatch
# ---------------------------------------------------------------------------

_DEFAULT_KW = {
    "intra_state_count": {"target": 1, "window_days": 2},
    "intra_state_sum": {"target": 1, "window_days": 2},
    "intra_state_vwap": {"target": 1, "window_days": 2},
    "intra_state_interval_moment": {"target": 1, "moment": "std", "window_days": 2, "min_events": 4},
    "intra_state_follow_ratio": {"target": 1, "lead_bars": 1, "window_days": 2},
    "intra_state_follow_beta": {"target": 1, "lead_bars": 1, "window_days": 2, "min_events": 8},
    "intra_state_follow_corr": {"target": 1, "lead_bars": 1, "window_days": 2, "min_events": 8},
    "intra_state_pair_same_slot_corr": {"window_days": 2, "min_slots": 8},
    "intra_state_dwell_stats": {"target_state": 1, "output": "mean", "min_slots": 20},
    "intra_state_transition_entropy": {"min_slots": 30, "normalize": True, "include_self": True},
    "intra_neighbor_event_class": {"radius": 1, "isolated_code": 1, "clustered_code": 2},
    "intra_range_gap_flag": {"neighbor_bars": 1},
}


def _call(name: str, panel: pd.DataFrame, state: pd.DataFrame, **overrides) -> pd.DataFrame:
    """Run operator ``name`` with a consistent panel/state signature."""
    kw = dict(_DEFAULT_KW[name])
    kw.update(overrides)
    op = OperatorRegistry.get(name)
    if name == "intra_state_count":
        return op.calculate(state, **kw)
    if name == "intra_state_sum":
        return op.calculate(panel, state, **kw)
    if name == "intra_state_vwap":
        return op.calculate(panel, _vol(panel), state, **kw)
    if name in (
        "intra_state_interval_moment",
        "intra_state_dwell_stats",
        "intra_state_transition_entropy",
        "intra_neighbor_event_class",
    ):
        return op.calculate(state, **kw)
    if name in ("intra_state_follow_ratio", "intra_state_follow_beta", "intra_state_follow_corr"):
        return op.calculate(panel, state, **kw)
    if name == "intra_state_pair_same_slot_corr":
        ta = kw.pop("target_a", 1)
        tb = kw.pop("target_b", 1)
        return op.calculate(state, ta, state, tb, **kw)
    if name == "intra_range_gap_flag":
        return op.calculate(panel, panel, state, **kw)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Registration + surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(ALL)))
def test_registered_and_classified(name: str) -> None:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


def test_canonical_count() -> None:
    import factor_engine.cleaned_operators.intraday.state_ops as m

    assert len(m._CANONICALS) == 12
    assert set(m._CANONICALS) == set(ALL)


# ---------------------------------------------------------------------------
# Shape + determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(ALL)))
def test_shape_and_determinism(name: str) -> None:
    panel = _minute_panel(days=3, cols=2, seed=1)
    state = _state_panel(days=3, cols=2, seed=11, n_states=3, nan_frac=0.02)
    first = _call(name, panel, state)
    second = _call(name, panel, state)
    if name in PER_BAR:
        assert first.shape == panel.shape, name
    else:
        assert first.shape == (3, 2), name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# Future-poison (corrupt future days, past outputs unchanged)
# ---------------------------------------------------------------------------

def test_future_poison() -> None:
    panel = _minute_panel(days=4, cols=2, seed=21)
    state = _state_panel(days=4, cols=2, seed=22, n_states=3, nan_frac=0.03)
    bars_per_day = len(panel) // 4
    for name in ALL:
        base = _call(name, panel, state)
        tampered = _call(name, _tamper(panel), _tamper(state))
        assert base.shape == tampered.shape, name
        if name in PER_BAR:
            pd.testing.assert_frame_equal(
                base.iloc[: 3 * bars_per_day],
                tampered.iloc[: 3 * bars_per_day],
                check_dtype=False,
            )
        else:
            pd.testing.assert_frame_equal(
                base.iloc[:3], tampered.iloc[:3], check_dtype=False
            )


# ---------------------------------------------------------------------------
# NaN / Inf edges
# ---------------------------------------------------------------------------

def test_all_nan_input_yields_all_nan() -> None:
    idx = _session_index(days=3)
    nan_panel = pd.DataFrame(np.full((len(idx), 2), np.nan), index=idx, columns=["C0", "C1"])
    for name in ALL:
        out = _call(name, nan_panel, nan_panel)
        assert out.isna().all().all(), name


def test_constant_input_nan_where_variance_required() -> None:
    idx = _session_index(days=2)
    const = pd.DataFrame(np.ones((len(idx), 2)), index=idx, columns=["C0", "C1"])
    state = _state_panel(days=2, cols=2, seed=5, n_states=3)
    # beta / corr need non-zero variance in the event values -> all NaN.
    for name in ("intra_state_follow_beta", "intra_state_follow_corr"):
        out = _call(name, const, state, target=1, window_days=1, min_events=2)
        assert out.isna().all().all(), name
    # pair-same-slot corr: identical constant counts across slots -> NaN.
    const_state = pd.DataFrame(np.ones((len(idx), 2)), index=idx, columns=["C0", "C1"])
    out = _call("intra_state_pair_same_slot_corr", const, const_state, window_days=1, min_slots=2)
    assert out.isna().all().all(), "intra_state_pair_same_slot_corr"


def test_zero_denominator_nan() -> None:
    idx = _session_index(days=2)
    zeros = pd.DataFrame(np.zeros((len(idx), 2)), index=idx, columns=["C0", "C1"])
    state = _state_panel(days=2, cols=2, seed=6, n_states=3)
    # follow_ratio: sum(x_event) == 0 -> zero denominator -> NaN.
    out = _call("intra_state_follow_ratio", zeros, state, target=1, window_days=1)
    assert out.isna().all().all(), "intra_state_follow_ratio"
    # vwap: volume all zero -> sum(volume) == 0 -> NaN.
    zero_vol = pd.DataFrame(np.zeros((len(idx), 2)), index=idx, columns=["C0", "C1"])
    out = OperatorRegistry.get("intra_state_vwap").calculate(
        zeros, zero_vol, state, target=1, window_days=1
    )
    assert out.isna().all().all(), "intra_state_vwap"


# ---------------------------------------------------------------------------
# Session behaviour
# ---------------------------------------------------------------------------

def test_lunch_boundary_does_not_bridge() -> None:
    # op 11: event at last morning bar, only neighbour is afternoon -> isolated.
    mask = _mini({"C0": [1.0, 1.0]}, minutes=[690, 781], days=1)
    out = OperatorRegistry.get("intra_neighbor_event_class").calculate(mask, radius=1)
    assert out["C0"].tolist() == [1.0, 1.0]

    # op 12: event at morning 690, neighbour bar crosses lunch -> NaN.
    h = _mini({"C0": [10.0, 10.0]}, minutes=[690, 781], days=1)
    l = _mini({"C0": [5.0, 5.0]}, minutes=[690, 781], days=1)
    m = _mini({"C0": [1.0, 0.0]}, minutes=[690, 781], days=1)
    out = OperatorRegistry.get("intra_range_gap_flag").calculate(h, l, m, neighbor_bars=1)
    assert np.isnan(out["C0"].iloc[0])
    assert out["C0"].iloc[1] == 0.0

    # op 4: gaps never span the lunch break (morning gap + afternoon gap only).
    state = _mini({"C0": [1, 1, 1, 1, 1]}, minutes=[571, 573, 577, 781, 787], days=1)
    out = OperatorRegistry.get("intra_state_interval_moment").calculate(
        state, target=1, moment="std", window_days=1, min_events=2
    )
    assert out["C0"].iloc[0] == pytest.approx(np.sqrt(8.0 / 3.0), rel=1e-9)


def test_single_day_two_instruments() -> None:
    panel = _minute_panel(days=1, cols=2, seed=41)
    state = _state_panel(days=1, cols=2, seed=42, n_states=3, nan_frac=0.02)
    for name in ALL:
        out = _call(name, panel, state)
        if name in PER_BAR:
            assert out.shape == panel.shape, name
        else:
            assert out.shape == (1, 2), name


# ---------------------------------------------------------------------------
# Golden hand-computed cases
# ---------------------------------------------------------------------------

def test_golden_intra_state_count() -> None:
    op = OperatorRegistry.get("intra_state_count")
    state = _mini({"C0": [1.0, 1.0, 2.0, np.nan, 2.0]}, minutes=[571, 572, 573, 574, 575])
    out = op.calculate(state, target=1)
    assert out["C0"].iloc[0] == 2.0
    out = op.calculate(state, target=2)
    assert out["C0"].iloc[0] == 2.0
    out = op.calculate(state)  # target None -> count all valid bars
    assert out["C0"].iloc[0] == 4.0

    # window_days>1 sums trailing completed days (shift(1) anchor).
    state2 = _mini(
        {"C0": [1, 1, 1, 1, 1, 2, 2, 2]},
        minutes=[571, 572, 573, 574],
        days=2,
    )
    out = op.calculate(state2, target=1, window_days=1)
    assert out["C0"].iloc[0] == 4.0
    assert out["C0"].iloc[1] == 1.0
    out = op.calculate(state2, target=1, window_days=2)
    assert np.isnan(out["C0"].iloc[0])
    assert out["C0"].iloc[1] == 4.0  # yesterday's count only


def test_golden_intra_state_sum() -> None:
    op = OperatorRegistry.get("intra_state_sum")
    x = _mini({"C0": [10.0, 20.0, np.nan, 40.0, 50.0]}, minutes=[571, 572, 573, 574, 575])
    state = _mini({"C0": [1.0, 1.0, 2.0, 2.0, 1.0]}, minutes=[571, 572, 573, 574, 575])
    out = op.calculate(x, state, target=1)
    assert out["C0"].iloc[0] == 80.0  # 10 + 20 + 50 (nan x ignored)


def test_golden_intra_state_vwap() -> None:
    op = OperatorRegistry.get("intra_state_vwap")
    price = _mini({"C0": [100.0, 101.0, 102.0, 103.0, 104.0]}, minutes=[571, 572, 573, 574, 575])
    volume = _mini({"C0": [1.0, 1.0, 0.0, 2.0, 1.0]}, minutes=[571, 572, 573, 574, 575])
    state = _mini({"C0": [1.0, 1.0, 2.0, 1.0, 1.0]}, minutes=[571, 572, 573, 574, 575])
    out = op.calculate(price, volume, state, target=1)
    # bars in state 1 with positive volume: (100,1),(101,1),(103,2),(104,1)
    assert out["C0"].iloc[0] == pytest.approx((100 + 101 + 206 + 104) / 5.0, rel=1e-9)


def test_golden_intra_state_interval_moment() -> None:
    op = OperatorRegistry.get("intra_state_interval_moment")
    state = _mini({"C0": [1, 1, 1, 1, 1]}, minutes=[571, 573, 577, 781, 787], days=1)
    out = op.calculate(state, target=1, moment="std", window_days=1, min_events=2)
    assert out["C0"].iloc[0] == pytest.approx(np.sqrt(8.0 / 3.0), rel=1e-9)
    out = op.calculate(state, target=1, moment="skew", window_days=1, min_events=2)
    assert out["C0"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    out = op.calculate(state, target=1, moment="kurtosis", window_days=1, min_events=2)
    assert out["C0"].iloc[0] == pytest.approx(1.5, rel=1e-9)
    # insufficient pooled gaps -> NaN
    out = op.calculate(state, target=1, moment="std", window_days=1, min_events=5)
    assert np.isnan(out["C0"].iloc[0])


def test_golden_intra_state_follow_ratio() -> None:
    op = OperatorRegistry.get("intra_state_follow_ratio")
    x = _mini({"C0": [10.0, 20.0, 30.0, 40.0, 50.0]}, minutes=[571, 572, 573, 574, 575])
    state = _mini({"C0": [1.0, 0.0, 0.0, 0.0, 1.0]}, minutes=[571, 572, 573, 574, 575])
    out = op.calculate(x, state, target=1, lead_bars=1, window_days=1)
    assert out["C0"].iloc[0] == pytest.approx(20.0 / 10.0, rel=1e-9)


def test_golden_intra_state_follow_beta() -> None:
    op = OperatorRegistry.get("intra_state_follow_beta")
    x = _mini({"C0": [0.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 0.0]}, minutes=list(range(571, 579)))
    state = _mini({"C0": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]}, minutes=list(range(571, 579)))
    out = op.calculate(x, state, target=1, lead_bars=1, window_days=1, min_events=2)
    # events at bars 1,3,5 -> xe=[1,2,4], xl=[2,4,8] -> OLS slope = 2.0
    assert out["C0"].iloc[0] == pytest.approx(2.0, rel=1e-9)


def test_golden_intra_state_follow_corr() -> None:
    op = OperatorRegistry.get("intra_state_follow_corr")
    x = _mini({"C0": [0.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 0.0]}, minutes=list(range(571, 579)))
    state = _mini({"C0": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]}, minutes=list(range(571, 579)))
    out = op.calculate(x, state, target=1, lead_bars=1, window_days=1, min_events=2)
    assert out["C0"].iloc[0] == pytest.approx(1.0, rel=1e-9)


def test_golden_intra_state_pair_same_slot_corr() -> None:
    op = OperatorRegistry.get("intra_state_pair_same_slot_corr")
    sa = _mini(
        {"C0": [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]},
        minutes=[571, 572, 573],
        days=4,
    )
    sb = _mini(
        {"C0": [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]},
        minutes=[571, 572, 573],
        days=4,
    )
    out = op.calculate(sa, 1, sb, 1, window_days=3, min_slots=2)
    # day3 pools days 0..2: ca=[3,2,1], cb=[3,1,0]
    assert out["C0"].iloc[3] == pytest.approx(np.sqrt(27.0 / 28.0), rel=1e-9)


def test_golden_intra_state_dwell_stats() -> None:
    op = OperatorRegistry.get("intra_state_dwell_stats")
    state = _mini({"C0": [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, np.nan, 1.0]}, minutes=list(range(571, 579)))
    # runs of state==1: [2, 1, 1]
    out = op.calculate(state, target_state=1, output="mean", min_slots=5)
    assert out["C0"].iloc[0] == pytest.approx(4.0 / 3.0, rel=1e-9)
    out = op.calculate(state, target_state=1, output="max", min_slots=5)
    assert out["C0"].iloc[0] == 2.0
    out = op.calculate(state, target_state=1, output="last", min_slots=5)
    assert out["C0"].iloc[0] == 1.0
    out = op.calculate(state, target_state=1, output="cv", min_slots=5)
    assert out["C0"].iloc[0] == pytest.approx(np.sqrt(2.0) / 4.0, rel=1e-9)
    out = op.calculate(state, target_state=1, output="share", min_slots=5)
    assert out["C0"].iloc[0] == pytest.approx(4.0 / 7.0, rel=1e-9)


def test_golden_intra_state_transition_entropy() -> None:
    op = OperatorRegistry.get("intra_state_transition_entropy")
    state = _mini({"C0": [1.0, 1.0, 2.0, 1.0, 2.0]}, minutes=list(range(571, 576)))
    # transitions: 1->1 (1), 1->2 (2), 2->1 (1)
    out = op.calculate(state, min_slots=4, normalize=True, include_self=False)
    expected = 0.6365141682 / np.log(2.0)
    assert out["C0"].iloc[0] == pytest.approx(expected, rel=1e-6)
    out = op.calculate(state, min_slots=4, normalize=False, include_self=False)
    assert out["C0"].iloc[0] == pytest.approx(0.6365141682, rel=1e-6)
    # include_self=True keeps the diagonal (1->1)
    out = op.calculate(state, min_slots=4, normalize=True, include_self=True)
    p = np.array([0.25, 0.5, 0.25])
    H = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    assert out["C0"].iloc[0] == pytest.approx(H / np.log(2.0), rel=1e-9)


def test_golden_intra_neighbor_event_class() -> None:
    op = OperatorRegistry.get("intra_neighbor_event_class")
    mask = _mini({"C0": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]}, minutes=list(range(571, 578)))
    out = op.calculate(mask, radius=1)
    assert out["C0"].tolist() == [0.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0]


def test_golden_intra_range_gap_flag() -> None:
    op = OperatorRegistry.get("intra_range_gap_flag")
    minutes = list(range(571, 578))
    mask = _mini({"C0": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}, minutes=minutes)
    # strict gap: pre range [5,10], post range [20,25] -> no overlap -> 1.0
    h = _mini({"C0": [10.0, 10.0, 10.0, 10.0, 25.0, 25.0, 25.0]}, minutes=minutes)
    l = _mini({"C0": [5.0, 5.0, 5.0, 5.0, 20.0, 20.0, 20.0]}, minutes=minutes)
    out = op.calculate(h, l, mask, neighbor_bars=1)
    assert out["C0"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # overlap/touch: pre [5,10], post [8,12] -> overlap -> 0.0
    h2 = _mini({"C0": [10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0]}, minutes=minutes)
    l2 = _mini({"C0": [5.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0]}, minutes=minutes)
    out = op.calculate(h2, l2, mask, neighbor_bars=1)
    assert out["C0"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

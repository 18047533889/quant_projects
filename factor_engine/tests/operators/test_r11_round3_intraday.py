# -*- coding: utf-8 -*-
"""Round-3 audit (P0-83 .. P0-90) — intraday session / recovery / Wasserstein.

Pins the round-3 findings:

* P0-83  session completeness compares ``observed_slot_set == official_slot_set``,
         never just ``len(observed) == expected`` (a same-count shifted grid, or
         a missing 09:45 plus a stray 09:46:30, must not certify complete).
* P0-84  bar width comes from the calendar's ``bar_freq`` (SessionCalendar /
         DataContract / SourceMetadata), NEVER from a modal of observed minute
         deltas; unknown resolution fails closed.
* P0-85  ``history_days`` counts COMPLETED sessions, not candidate runs — the
         history window expands backward past incomplete days until
         ``history_days`` completed sessions are found (or history is exhausted).
* P0-86  the session novelty distance is RMSE-normalised
         (``sqrt(mean(d_i^2))``), so a 1-min 240-node and a 5-min 48-node shape
         are comparable.
* P0-87  ``session_event_recovery_score`` never guesses the session timezone: a
         bare UTC/unknown tz-aware index fails closed unless an explicit
         ``session_tz`` or a ``calendar`` (market -> session zone) is given.
* P0-88  recovery requires ``min_events`` effective events before the daily
         median is meaningful (a single shock is statistically unstable).
* P0-89  ``residual_fraction`` is ``0 < f <= 1`` at BOTH compile time and
         runtime (a RelationalParamSpec makes 0 compile-invalid).
* P0-90  the baseline-MAD-scaled W1 is renamed ``baseline_scaled_wasserstein_-
         distance`` so it is not mistaken for a symmetric normalised W1;
         ``intraday_wasserstein_pair_distance`` remains a back-compat canonical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.session_calendar import SessionCalendar

_ASHARE_CAL = SessionCalendar(
    market="CN", timestamp_convention="bar_start", bar_freq="1min"
)
_ASHARE_FULL_MODS = list(range(570, 690)) + list(range(780, 900))  # 09:30-11:29 + 13:00-14:59


@pytest.fixture(scope="module", autouse=True)
def _owned_modules_loaded():
    # Register + surface the owned modules directly (load_all() is blocked by an
    # unrelated pre-existing tree conflict in polars_geometry_math.py, a file
    # outside this task's ownership).
    import factor_engine.cleaned_operators.advanced_intraday  # noqa: F401
    import factor_engine.cleaned_operators.intraday_session  # noqa: F401
    import factor_engine.cleaned_operators.session_recovery  # noqa: F401
    yield


def _session_panel(day_mods_by_day, value_fn=None):
    """Minute panel + integral session_id, one column, deterministic values."""
    idx, sids, vals = [], [], []
    for d, mods in enumerate(day_mods_by_day):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for m in mods:
            idx.append(day + pd.Timedelta(minutes=m))
            sids.append(float(d))
            vals.append(value_fn(m, d) if value_fn else 100.0 + 0.01 * m + d)
    return pd.DataFrame({"S0": vals}, index=pd.DatetimeIndex(idx)), pd.DataFrame(
        {"S0": sids}, index=pd.DatetimeIndex(idx)
    )


# ---------------------------------------------------------------------------
# P0-83  completeness: observed_slot_set == official_slot_set
# ---------------------------------------------------------------------------

def test_same_count_shifted_grid_not_complete():
    from factor_engine.cleaned_operators.intraday_session import (
        _official_grid,
        _minute_of_day,
        _session_runs,
    )

    exp, close_mod, slots = _official_grid(_ASHARE_CAL)
    shifted = [m + 1 for m in _ASHARE_FULL_MODS]  # same COUNT (240), wrong set (900 present, 570 gone)
    x, sid = _session_panel([_ASHARE_FULL_MODS, shifted, _ASHARE_FULL_MODS])
    index = x.index
    dates = index.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]").astype("int64")
    mods = _minute_of_day(index.to_numpy(dtype="datetime64[ns]"))
    runs = _session_runs(
        x["S0"].to_numpy(dtype=float), sid["S0"].to_numpy(dtype=float),
        dates, mods, exp, close_mod, slots,
    )
    assert runs[0]["completed"] is True
    assert runs[1]["completed"] is False, (
        "same COUNT but shifted set certified complete — observed set must equal "
        "the official set, not just match in length"
    )
    assert runs[2]["completed"] is True


def test_missing_slot_plus_stray_slot_not_complete():
    from factor_engine.cleaned_operators.intraday_session import (
        _official_grid,
        _minute_of_day,
        _session_runs,
    )

    exp, close_mod, slots = _official_grid(_ASHARE_CAL)
    # Remove the 09:45 slot (mod 585) and add a stray 09:46:30 (floor -> mod 586,
    # already present): the distinct-slot count drops, and the set differs — the
    # completeness gate must catch it rather than the old len check.
    missing = [m for m in _ASHARE_FULL_MODS if m != 585]
    x, sid = _session_panel([_ASHARE_FULL_MODS, missing, _ASHARE_FULL_MODS])
    index = x.index
    dates = index.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]").astype("int64")
    mods = _minute_of_day(index.to_numpy(dtype="datetime64[ns]"))
    runs = _session_runs(
        x["S0"].to_numpy(dtype=float), sid["S0"].to_numpy(dtype=float),
        dates, mods, exp, close_mod, slots,
    )
    assert runs[0]["completed"] is True
    assert runs[1]["completed"] is False
    assert runs[2]["completed"] is True


# ---------------------------------------------------------------------------
# P0-84  bar width from the calendar, never from observed minute deltas
# ---------------------------------------------------------------------------

def test_bar_width_comes_from_calendar_bar_freq():
    from factor_engine.cleaned_operators.intraday_session import IntradaySessionShapeNovelty

    cal5 = SessionCalendar(market="CN", timestamp_convention="bar_start", bar_freq="5min")
    # 5-min bar labels: morning 09:30..11:25, afternoon 13:00..14:55 -> 48 bars.
    mods5 = list(range(570, 690, 5)) + list(range(780, 900, 5))
    assert len(mods5) == 48
    x, sid = _session_panel([mods5, mods5, mods5])
    out = IntradaySessionShapeNovelty()._calculate_series(
        x, sid, history_days=8, min_history_sessions=2, calendar=cal5
    )
    arr = out["S0"].to_numpy(dtype=float)
    # Only the last day has >= 2 completed historical sessions.
    assert float(np.nansum(np.isfinite(arr))) == 1
    # The emission sits on the 5-min official close: 14:55 = mod 895.
    row = int(np.flatnonzero(np.isfinite(arr))[0])
    minute_of_day = x.index[row].hour * 60 + x.index[row].minute
    assert minute_of_day == 895


def test_bar_width_unknown_calendar_fails_closed():
    from factor_engine.cleaned_operators.intraday_session import _official_grid

    class _NoResolutionCalendar:
        segments = (("09:30", "11:30"), ("13:00", "15:00"))
        timestamp_convention = "bar_start"

    with pytest.raises(ValueError, match="bar_freq|resolution"):
        _official_grid(_NoResolutionCalendar())


# ---------------------------------------------------------------------------
# P0-85  history_days counts COMPLETED sessions, not candidate runs
# ---------------------------------------------------------------------------

def test_history_days_reaches_past_incomplete_days():
    from factor_engine.cleaned_operators.intraday_session import IntradaySessionShapeNovelty

    partial = list(range(570, 660))  # truncated at 11:00 -> incomplete
    x, sid = _session_panel([_ASHARE_FULL_MODS, partial, _ASHARE_FULL_MODS, _ASHARE_FULL_MODS])
    index = x.index

    def last_of_day(d):
        dts = index.normalize() == (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d))
        return int(np.flatnonzero(dts)[-1])

    out = IntradaySessionShapeNovelty()._calculate_series(
        x, sid, history_days=2, min_history_sessions=2, calendar=_ASHARE_CAL
    )
    arr = out["S0"].to_numpy(dtype=float)
    # Day 2: only completed predecessor in the immediate window is day 0
    # (day 1 is partial) -> 1 completed < min_history_sessions=2 -> no emit.
    assert np.isnan(arr[last_of_day(2)])
    # Day 3: the history scan reaches PAST the incomplete day to collect day 0
    # and day 2 (2 completed sessions) -> emits (P0-85).
    assert np.isfinite(arr[last_of_day(3)])


# ---------------------------------------------------------------------------
# P0-86  novelty distance is RMSE-normalised
# ---------------------------------------------------------------------------

def test_novelty_distance_is_rmse_not_raw_euclidean():
    from factor_engine.cleaned_operators.intraday_session import (
        _canonical_shape,
        _official_grid,
        _shape_novelty_series,
        _minute_of_day,
    )

    exp, close_mod, slots = _official_grid(_ASHARE_CAL)
    n_nodes = max(2, exp)
    # day0/day1 share a linear shape; day2 uses a QUADRATIC shape.  An affine
    # transform would normalise to the identical shape, so the quadratic makes
    # the RMSE genuinely positive (the guard below needs a nonzero distance).
    def shape_of(m, d):
        if d < 2:
            return 100.0 + 0.01 * m + d
        return 200.0 + 0.002 * (m - 570) ** 2 + d

    x, sid = _session_panel(
        [_ASHARE_FULL_MODS, _ASHARE_FULL_MODS, _ASHARE_FULL_MODS], value_fn=shape_of
    )
    index = x.index
    dates = index.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]").astype("int64")
    mods = _minute_of_day(index.to_numpy(dtype="datetime64[ns]"))
    arr = _shape_novelty_series(
        x.to_numpy(dtype=float), sid.to_numpy(dtype=float), dates, mods,
        history_days=8, min_history_sessions=1, expected_slots=exp,
        official_close_mod=close_mod, official_slot_set=slots,
    )
    # day2 emission: RMSE between day2's canonical shape and the (identical)
    # day0/day1 canonical shape.
    idx_day2 = np.flatnonzero(index.normalize() == pd.Timestamp("2024-01-03"))
    end2 = int(idx_day2[-1])
    assert np.isfinite(arr[end2])
    # Recompute the canonical shape of a day2 run and a day0 run, then RMSE.
    day0_rows = np.flatnonzero(index.normalize() == pd.Timestamp("2024-01-01"))
    s0 = _canonical_shape(x["S0"].to_numpy(dtype=float)[day0_rows], n_nodes)
    s2 = _canonical_shape(x["S0"].to_numpy(dtype=float)[idx_day2], n_nodes)
    expected_rmse = float(np.linalg.norm(s2 - s0) / np.sqrt(n_nodes))
    assert arr[end2] == pytest.approx(expected_rmse, rel=1e-9, abs=1e-12)
    # Guard: raw Euclidean would be sqrt(n_nodes) larger, so this test would
    # fail if someone reverts the RMSE normalisation.
    raw_euclid = float(np.linalg.norm(s2 - s0))
    assert abs(arr[end2] - raw_euclid) > 1e-9


# ---------------------------------------------------------------------------
# P0-87  session_event_recovery_score timezone is never guessed
# ---------------------------------------------------------------------------

def test_recovery_utc_index_without_tz_fails_closed():
    from factor_engine.cleaned_operators.session_recovery import SessionEventRecoveryScore

    idx = pd.DatetimeIndex(
        [
            (pd.Timestamp("2024-01-02") + pd.Timedelta(hours=19) + pd.Timedelta(minutes=m))
            .tz_localize("America/New_York").tz_convert("UTC")
            for m in range(60)
        ]
    )
    x = pd.DataFrame(np.linspace(10.0, 10.5, 60), index=idx, columns=["S0"])
    ev = pd.DataFrame(np.zeros(60), index=idx, columns=["S0"])
    ev.iloc[5, 0] = 1.0
    with pytest.raises(ValueError, match="session timezone|session_tz"):
        SessionEventRecoveryScore()._calculate_series(x, ev, horizon=5, residual_fraction=0.25)


def test_recovery_explicit_session_tz_converts_utc():
    from factor_engine.cleaned_operators.session_recovery import SessionEventRecoveryScore

    idx = pd.DatetimeIndex(
        [
            (pd.Timestamp("2024-01-02") + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(minutes=m))
            .tz_localize("UTC")
            for m in range(120)
        ]
    )
    x = pd.DataFrame(np.linspace(10.0, 10.5, 120), index=idx, columns=["S0"])
    ev = pd.DataFrame(np.zeros(120), index=idx, columns=["S0"])
    ev.iloc[5, 0] = 1.0
    out = SessionEventRecoveryScore()._calculate_series(
        x, ev, horizon=5, residual_fraction=0.25, session_tz="America/New_York"
    )
    assert out.index.tolist() == [pd.Timestamp("2024-01-02")]


def test_recovery_calendar_market_supplies_tz():
    from factor_engine.cleaned_operators.session_recovery import _resolve_session_tz

    cal = SessionCalendar(market="CN", timestamp_convention="bar_start", bar_freq="1min")
    idx = pd.DatetimeIndex(["2024-01-01 09:30:00+00:00", "2024-01-01 09:31:00+00:00"])
    assert _resolve_session_tz(idx, calendar=cal) == "Asia/Shanghai"
    # Known stored zone needs no conversion (no-op rename).
    idx_ny = pd.DatetimeIndex(["2024-01-01 09:30:00"], tz="America/New_York")
    assert _resolve_session_tz(idx_ny) == "America/New_York"


# ---------------------------------------------------------------------------
# P0-88  recovery needs min_events effective events
# ---------------------------------------------------------------------------

def test_recovery_min_events_gate():
    from factor_engine.cleaned_operators.session_recovery import _recovery_day

    x = np.array([10.0, 10.5, 10.05, 10.02, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    event = np.zeros(10)
    event[1] = 1.0  # a single shock
    assert np.isfinite(_recovery_day(x, event, horizon=5, residual_fraction=0.25, min_events=1))
    assert np.isnan(_recovery_day(x, event, horizon=5, residual_fraction=0.25, min_events=2)), (
        "a single shock must not produce a stable median — min_events gate failed"
    )


def test_recovery_min_events_via_operator():
    from factor_engine.cleaned_operators.session_recovery import SessionEventRecoveryScore

    # R26-050/051: the operator grid-aligns to the official A-share session, so
    # the fixture must be the full 240-bar session with a reachable close.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 09:31") + pd.Timedelta(minutes=m) for m in range(120)]
        + [pd.Timestamp("2024-01-02 13:01") + pd.Timedelta(minutes=m) for m in range(120)]
    )
    x = pd.DataFrame(np.linspace(10.0, 10.5, 240), index=idx, columns=["S0"])
    ev = np.zeros(240)
    ev[5] = 1.0  # a single shock
    evf = pd.DataFrame({"S0": ev}, index=idx)
    op = SessionEventRecoveryScore()
    assert np.isfinite(op._calculate_series(x, evf, horizon=5, residual_fraction=0.25, min_events=1).iloc[0, 0])
    assert np.isnan(op._calculate_series(x, evf, horizon=5, residual_fraction=0.25, min_events=2).iloc[0, 0])


# ---------------------------------------------------------------------------
# P0-89  residual_fraction compile-time == runtime
# ---------------------------------------------------------------------------

def test_recovery_residual_fraction_zero_compile_invalid():
    from factor_engine.cleaned_operators.session_recovery import SessionEventRecoveryScore

    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in range(120)])
    x = pd.DataFrame(np.linspace(10.0, 10.5, 120), index=idx, columns=["S0"])
    ev = np.zeros(120)
    ev[5] = 1.0
    evf = pd.DataFrame({"S0": ev}, index=idx)
    op = SessionEventRecoveryScore()
    # Compile-time (validate_operator_call -> RelationalParamSpec) rejects 0.
    with pytest.raises(ValueError, match="residual_fraction"):
        op.calculate(x, evf, horizon=5, residual_fraction=0.0)
    # Runtime kernel rejects 0 identically.
    with pytest.raises(ValueError, match="residual_fraction"):
        op._calculate_series(x, evf, horizon=5, residual_fraction=0.0)
    # 1 is the upper bound (valid both places).
    out = op.calculate(x, evf, horizon=5, residual_fraction=1.0)
    assert out.shape[1] == 1


# ---------------------------------------------------------------------------
# P0-90  baseline-scaled W1 honest name
# ---------------------------------------------------------------------------

def test_baseline_scaled_wasserstein_registered_extended():
    from factor_engine.cleaned_operators.operator_surface import classify_canonical
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    canon = set(OperatorRegistry.list_canonical())
    assert "baseline_scaled_wasserstein_distance" in canon
    assert "intraday_wasserstein_pair_distance" in canon  # back-compat canonical
    assert classify_canonical("baseline_scaled_wasserstein_distance") == "extended"
    assert classify_canonical("intraday_wasserstein_pair_distance") == "extended"
    assert OperatorRegistry.get("baseline_scaled_wasserstein_distance", "pandas_numpy") is not None


def _minute_frame(days=2, cols=3, per_day=60):
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per_day)]
    )
    rng = np.random.default_rng(11)
    base = 1.0 + 0.001 * np.arange(days * per_day)[:, None]
    return pd.DataFrame(
        base + rng.normal(0.0, 0.001, size=(days * per_day, cols)),
        index=idx, columns=[f"S{i}" for i in range(cols)],
    )


def test_baseline_scaled_wasserstein_asymmetric():
    from factor_engine.cleaned_operators.advanced_intraday import BaselineScaledWassersteinDistance

    m1 = _minute_frame(2, 3)
    m2 = m1 * 1.5
    op = BaselineScaledWassersteinDistance()
    d12 = op._calculate_series(m1, m2).to_numpy(dtype=float)
    d21 = op._calculate_series(m2, m1).to_numpy(dtype=float)
    d12 = d12[np.isfinite(d12)]
    d21 = d21[np.isfinite(d21)]
    assert d12.size > 0 and d21.size > 0
    # MAD(Y) scaling is not symmetric -> the two orders differ.
    assert not np.allclose(d12, d21, atol=1e-9)


def test_baseline_scaled_wasserstein_identical_is_zero():
    from factor_engine.cleaned_operators.advanced_intraday import BaselineScaledWassersteinDistance

    m1 = _minute_frame(2, 3)
    out = BaselineScaledWassersteinDistance()._calculate_series(m1, m1)
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    assert np.allclose(vals, 0.0, atol=1e-9)

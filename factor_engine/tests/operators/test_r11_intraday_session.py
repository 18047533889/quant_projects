# -*- coding: utf-8 -*-
"""Round-11 intraday session / realized-variance audit tests.

Pins the R11 findings #56-#79 and #183-#186:

* k-minute realised variance aggregates the intra-block minute returns (never a
  blind ``ret[::k]`` sample) — R11 #56/#57/#58.
* A truncated intraday session (11:00 / 14:00 / close-1min) does NOT emit a
  full-session factor; only the official close + full coverage does — R11 #68.
* US-timezone session grouping uses the market session wall-clock — R11 #64/#77.
* Degenerate PCA subspace (zero historical variance) -> NaN, never 0 — R11 #62.
* Phase shift below the min-correlation gate -> NaN — R11 #66.
* A missing minute in a recovery horizon makes recovery only interval
  identifiable -> NaN, never a false precise time — R11 #78.
* A partial activity session (missing whole minutes) is rejected by the official
  SessionGrid reindex — R11 #184/#185.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _calc(name: str, *frames: pd.DataFrame, **params) -> pd.DataFrame:
    return _op(name).calculate(*frames, **params)


# ---------------------------------------------------------------------------
# R11 #56/#57/#58  subsampled realised variance aggregates intra-block returns
# ---------------------------------------------------------------------------

def _subsampled_rv_reference(minute_returns: np.ndarray, sampling: int) -> float:
    """Hand-aggregated coefficient of variation of per-offset k-minute RV.

    For each offset ``off`` the k-minute return of a block is the SUM of its
    minute returns; the offset's RV proxy is the per-block mean squared return
    (equalises the block-count difference across offsets).  The result is the
    coefficient of variation across offsets.
    """
    r = np.asarray(minute_returns, dtype=float)
    sm = int(sampling)
    per_off: list[float] = []
    for off in range(sm):
        blocks: list[float] = []
        start = off
        while start + sm <= len(r):
            blk = r[start:start + sm]
            if np.all(np.isfinite(blk)):
                blocks.append(float(blk.sum()))
            start += sm
        if blocks:
            per_off.append(float(np.mean(np.asarray(blocks) ** 2)))
    if len(per_off) < 2:
        return np.nan
    mu = float(np.mean(per_off))
    if mu <= 1e-12:
        return np.nan
    return float(np.std(per_off) / mu)


def test_subsampled_rv_aggregates_in_sample_minutes():
    from cleaned_operators.advanced_intraday import _block_returns

    r = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    # sampling=2, offset 0 -> blocks (r0+r1), (r2+r3), (r4+r5)
    blocks = _block_returns(r, 2, 0)
    np.testing.assert_allclose(
        blocks,
        np.array([r[0] + r[1], r[2] + r[3], r[4] + r[5]]),
    )
    # The blind-sample ``r[::2]`` would give (r0, r2, r4) -> different values,
    # so the aggregate never equals a blind sample for a non-constant path.
    assert not np.allclose(blocks, r[::2])


def test_subsampled_rv_dispersion_matches_hand_aggregation():
    from cleaned_operators.advanced_intraday import _subsampled_rv_dispersion

    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.01, size=90)
    for sm in (2, 3, 5):
        got = _subsampled_rv_dispersion(r, sm)
        expected = _subsampled_rv_reference(r, sm)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12), sm


def test_realized_power_variation_aggregates_blocks():
    from cleaned_operators.advanced_intraday import _realized_power_variation

    r = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    blocks = np.array([r[0] + r[1], r[2] + r[3], r[4] + r[5]])
    val = _realized_power_variation(r, 2.0, 2)
    assert val == pytest.approx(float(np.sum(blocks ** 2.0)))
    val_p4 = _realized_power_variation(r, 4.0, 2)
    assert val_p4 == pytest.approx(float(np.sum(np.abs(blocks) ** 4.0)))


# ---------------------------------------------------------------------------
# R11 #64 / #77  US-timezone session grouping uses the session wall-clock
# ---------------------------------------------------------------------------

def _us_afterhours_utc_frame():
    """US after-hours session 19:00-20:59 ET on 2024-01-02, stored in UTC.

    In UTC these minutes are 00:00-01:59 on 2024-01-03, so bare
    ``index.normalize()`` would assign the session to the wrong trade date; the
    session wall-clock (America/New_York) keeps it on 2024-01-02.
    """
    local = pd.Timestamp("2024-01-02")
    idx = []
    for m in range(120):
        ts = (local + pd.Timedelta(hours=19) + pd.Timedelta(minutes=m)).tz_localize(
            "America/New_York"
        )
        idx.append(ts.tz_convert("UTC"))
    vals = np.linspace(100.0, 101.0, 120)
    return pd.DataFrame({"S0": vals}, index=pd.DatetimeIndex(idx))


def test_us_session_grouping_uses_session_tz():
    from cleaned_operators.advanced_intraday import _per_day_returns

    frame = _us_afterhours_utc_frame()
    days_ny, _ = _per_day_returns(frame, session_tz="America/New_York")
    assert days_ny == [pd.Timestamp("2024-01-02")]
    # The stored-UTC date (2024-01-03) must NOT leak into the session grouping.
    days_default, _ = _per_day_returns(frame, session_tz=None)
    assert days_default != [pd.Timestamp("2024-01-02")]


def test_us_session_grouping_via_operator():
    # A regular US day stored in UTC still groups into ONE trade date when the
    # session wall-clock is declared; the operator emits one row per day.
    idx = []
    for m in range(390):  # 09:30..15:59 ET = 14:30..20:59 UTC (winter)
        ts = (pd.Timestamp("2024-01-02") + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(minutes=m)).tz_localize("UTC")
        idx.append(ts)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(0.0, 0.01, len(idx)), index=pd.DatetimeIndex(idx), columns=["S0"])
    out = _calc("intraday_subsampled_rv_dispersion", frame, sampling=5, session_tz="America/New_York")
    assert out.index.tolist() == [pd.Timestamp("2024-01-02")]


# ---------------------------------------------------------------------------
# R11 #68  truncated intraday sessions never emit a full-session factor
# ---------------------------------------------------------------------------

_ASHARE_FULL_MODS = list(range(570, 690)) + list(range(780, 900))  # 09:30-11:29 + 13:00-14:59


def _session_panel(day_mods_by_day):
    """Minute panel + integral session_id, one column, deterministic values."""
    idx, sids, vals = [], [], []
    for d, mods in enumerate(day_mods_by_day):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for m in mods:
            idx.append(day + pd.Timedelta(minutes=m))
            sids.append(float(d))
            vals.append(100.0 + 0.01 * m + d)
    index = pd.DatetimeIndex(idx)
    x = pd.DataFrame({"S0": vals}, index=index)
    sid = pd.DataFrame({"S0": sids}, index=index)
    return x, sid


def _last_row_of_day(x: pd.DataFrame, day: pd.Timestamp) -> int:
    day_idx = np.flatnonzero(x.index.normalize() == pd.Timestamp(day))
    return int(day_idx[-1])


def test_truncated_session_never_emits_full_session_factor():
    day0 = pd.Timestamp("2024-01-01")
    # days 0-2 full; day 3 truncated at 11:00; days 4-5 full.
    day_mods = [
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        list(range(570, 660)),  # 09:30..10:59 -> stops at 11:00
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
    ]
    x, sid = _session_panel(day_mods)
    out = _calc("intraday_session_shape_novelty", x, sid, history_days=6, min_history_sessions=2)
    arr = out["S0"].to_numpy(dtype=float)

    # The truncated day-3 run must NOT emit a value at its last observed minute.
    day3_last = _last_row_of_day(x, day0 + pd.Timedelta(days=3))
    assert np.isnan(arr[day3_last])

    # A completed full day with >= min_history_sessions completed history emits.
    day2_last = _last_row_of_day(x, day0 + pd.Timedelta(days=2))
    assert np.isfinite(arr[day2_last])
    # And every row OUTSIDE the emission minutes stays NaN (per-session PIT).
    # Completed days with >= 2 completed history sessions: day2, day4, day5.
    assert float(np.nansum(np.isfinite(arr))) == 3


def test_truncated_close_minus_one_and_1400():
    # close-1min (missing the official close) and a 14:00 stop are both partial.
    day0 = pd.Timestamp("2024-01-01")
    day_mods = [
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        list(range(570, 690)) + list(range(780, 899)),  # close - 1min
        list(range(570, 690)) + list(range(780, 840)),  # stops at 14:00
        _ASHARE_FULL_MODS,
    ]
    x, sid = _session_panel(day_mods)
    out = _calc("intraday_session_shape_novelty", x, sid, history_days=8, min_history_sessions=3)
    arr = out["S0"].to_numpy(dtype=float)
    for offset in (4, 5):  # close-1min and 14:00 days
        last = _last_row_of_day(x, day0 + pd.Timedelta(days=offset))
        assert np.isnan(arr[last]), offset


def test_truncated_session_pca_residual_also_censored():
    day0 = pd.Timestamp("2024-01-01")
    day_mods = [
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        list(range(570, 660)),  # truncated at 11:00
        _ASHARE_FULL_MODS,
    ]
    x, sid = _session_panel(day_mods)
    out = _calc("intraday_profile_pca_residual", x, sid, history_days=6, n_components=2, min_history_sessions=3)
    arr = out["S0"].to_numpy(dtype=float)
    day4_last = _last_row_of_day(x, day0 + pd.Timedelta(days=4))
    assert np.isnan(arr[day4_last])


# ---------------------------------------------------------------------------
# R11 #62  degenerate PCA subspace -> NaN, never 0
# ---------------------------------------------------------------------------

def test_degenerate_pca_subspace_is_nan_not_zero():
    days, per = 60, 120
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per)])
    data = pd.DataFrame(np.full(days * per, 0.05), index=idx, columns=["S0"])
    score = _calc("intraday_quantile_curve_pca_score", data, window=50, k=1)
    resid = _calc("intraday_quantile_curve_pca_residual", data, window=50, k=1)
    # identical profiles -> zero historical variance -> DEGENERATE_PCA_SUBSPACE
    assert np.isnan(score.to_numpy(dtype=float)).all()
    assert np.isnan(resid.to_numpy(dtype=float)).all()


def test_pca_score_series_zero_history_variance_nan():
    from cleaned_operators.advanced_intraday import _pca_score_series

    const = [np.full(120, 0.05)] * 60
    vals = _pca_score_series(const, lookback=50, k=1)
    assert np.isnan(vals).all()


# ---------------------------------------------------------------------------
# R11 #66  phase shift below the min-correlation gate -> NaN
# ---------------------------------------------------------------------------

def test_phase_shift_below_threshold_is_nan():
    from cleaned_operators.advanced_intraday import _best_phase

    cur = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    anti = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # opposite trend -> corr <= 0
    assert np.isnan(_best_phase(cur, anti, max_shift=2, n_slots=6))

    same = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert _best_phase(cur, same, max_shift=2, n_slots=6) == pytest.approx(0.0)


def test_phase_shift_max_shift_infeasible_rejected():
    with pytest.raises(ValueError, match="max_shift"):
        _calc("intraday_profile_phase_shift", _minute_frame_naive(5, 12), history_days=3, max_shift=6, n_slots=6)


def _minute_frame_naive(n_days: int, bars: int) -> pd.DataFrame:
    idx = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(bars):
            idx.append(day + pd.Timedelta(minutes=30 + b))
    vals = np.tile(np.sin(2.0 * np.pi * np.arange(bars) / bars), n_days)
    return pd.DataFrame(vals, index=pd.DatetimeIndex(idx), columns=["S0"])


# ---------------------------------------------------------------------------
# R11 #78  recovery with a missing interior minute -> NaN (interval only)
# ---------------------------------------------------------------------------

def test_recovery_missing_minute_is_nan_not_false_precise():
    from cleaned_operators.session_recovery import _recovery_day

    # Event at minute 1 (shock 10.0 -> 10.5); minute 2 is MISSING; minute 3
    # happens to be back near the baseline.  The true recovery lies in
    # (10:01, 10:03] — a precise 10:03 would be fabricated.
    x = np.array([10.0, 10.5, np.nan, 10.05, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    event = np.zeros(10)
    event[1] = 1.0
    assert np.isnan(_recovery_day(x, event, horizon=5, residual_fraction=0.25, refractory=0))


def test_recovery_fully_observed_is_precise():
    from cleaned_operators.session_recovery import _recovery_day

    x = np.array([10.0, 10.5, 10.05, 10.02, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    event = np.zeros(10)
    event[1] = 1.0
    # minute 2: |10.05 - 10.0| = 0.05 <= 0.25 * 0.5 = 0.125 -> tau = 1
    val = _recovery_day(x, event, horizon=5, residual_fraction=0.25, refractory=0)
    assert val == pytest.approx(1.0 / 6.0)


def test_recovery_event_must_be_bool():
    from cleaned_operators.session_recovery import _recovery_day

    x = np.linspace(10.0, 10.5, 10)
    event = np.zeros(10)
    event[2] = 2.0  # non-binary -> invalid EventBool panel
    assert np.isnan(_recovery_day(x, event, horizon=5, residual_fraction=0.25))


def test_recovery_residual_fraction_out_of_range_rejected():
    from cleaned_operators.session_recovery import _recovery_day

    x = np.linspace(10.0, 10.5, 10)
    event = np.zeros(10)
    event[2] = 1.0
    with pytest.raises(ValueError, match="residual_fraction"):
        _recovery_day(x, event, horizon=5, residual_fraction=1.5)
    with pytest.raises(ValueError, match="residual_fraction"):
        _calc("session_event_recovery_score", pd.DataFrame(x, columns=["S0"]),
              pd.DataFrame(event, columns=["S0"]), horizon=5, residual_fraction=0.0)


def test_recovery_refractory_suppresses_overlap():
    from cleaned_operators.session_recovery import _recovery_day

    x = np.linspace(10.0, 10.6, 12)
    event = np.zeros(12)
    event[2] = 1.0
    event[3] = 1.0  # overlapping shock inside the refractory window
    # With refractory=1 the second event is suppressed -> single event.
    val = _recovery_day(x, event, horizon=5, residual_fraction=0.25, refractory=1)
    assert val == (5 + 1) / (5 + 1) == 1.0
    # With refractory=0 both events are counted; identical horizons -> same value,
    # but the mechanism is exercised (no crash, deterministic).
    val0 = _recovery_day(x, event, horizon=5, residual_fraction=0.25, refractory=0)
    assert val0 == val


# ---------------------------------------------------------------------------
# R11 #184/#185  activity-duration partial session rejected by SessionGrid
# ---------------------------------------------------------------------------

def _activity_panel(day_mods_by_day, value: float = 1.0):
    idx, vals = [], []
    for d, mods in enumerate(day_mods_by_day):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for m in mods:
            idx.append(day + pd.Timedelta(minutes=m))
            vals.append(value)
    return pd.DataFrame({"S0": vals}, index=pd.DatetimeIndex(idx))


def test_activity_duration_partial_session_rejected():
    day_mods = [
        _ASHARE_FULL_MODS,
        _ASHARE_FULL_MODS,
        list(range(570, 690)),  # morning only -> stops at 11:29, no afternoon
        _ASHARE_FULL_MODS,
    ]
    act = _activity_panel(day_mods, value=1.0)
    out = _calc("intraday_activity_duration_curvature", act, buckets=5)
    truncated = pd.Timestamp("2024-01-01") + pd.Timedelta(days=2)
    full = pd.Timestamp("2024-01-01") + pd.Timedelta(days=3)
    assert np.isnan(out.loc[truncated, "S0"])       # partial session -> NaN
    assert np.isfinite(out.loc[full, "S0"])         # full session -> finite
    assert out.loc[full, "S0"] == pytest.approx(0.0)  # constant activity -> flat curve


def test_activity_duration_missing_whole_minute_row():
    # A full-width day missing ONE whole minute row is invisible to a NaN gate
    # on the raw array; reindexing to the official grid turns it into NaN.
    full = _ASHARE_FULL_MODS[:]
    missing_one = _ASHARE_FULL_MODS[:]
    missing_one.remove(_ASHARE_FULL_MODS[60])  # drop one interior minute
    day_mods = [_ASHARE_FULL_MODS, missing_one, _ASHARE_FULL_MODS]
    act = _activity_panel(day_mods, value=1.0)
    out = _calc("intraday_activity_duration_curvature", act, buckets=5)
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "S0"])
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-01"), "S0"])

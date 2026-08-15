# -*- coding: utf-8 -*-
"""Hand-computed numeric semantics for the 2026-08 vertical-deepening pack.

Each test constructs a *deterministic* series whose answer is known by hand
(or by a closed-form reference) and asserts the operator recovers it within a
tight tolerance.  All inputs are trailing-causal; no future data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()


def _g(df: pd.DataFrame, name: str) -> np.ndarray:
    return OperatorRegistry.get(name, "pandas_numpy").calculate(df).to_numpy(dtype=float)


def _get(name: str):
    return OperatorRegistry.get(name, "pandas_numpy")


def test_markov_committor_upper_biased():
    # A monotone rising series: the transition matrix strongly favours the
    # upper state -> committor of any reached state should be near 1.
    x = pd.DataFrame(np.arange(300.0), index=pd.date_range("2024-01-01", periods=300))
    out = _get("ts_markov_committor").calculate(x, window=80, bins=3, lag=1).to_numpy(dtype=float)
    late = out[150:, 0]
    late = late[np.isfinite(late)]
    assert float(np.mean(late)) > 0.85


def test_markov_mean_first_passage_time_upper_small_for_high_state():
    # Same rising series: a state near the top should reach "upper" quickly.
    x = pd.DataFrame(np.arange(300.0) % 40.0, index=pd.date_range("2024-01-01", periods=300))
    out = _get("ts_markov_mean_first_passage_time").calculate(x, window=80, bins=3, lag=1, target="upper").to_numpy(dtype=float)
    late = out[150:, 0]
    late = late[np.isfinite(late)]
    assert float(np.mean(late)) < 60.0  # expected first passage stays bounded


def test_markov_spectral_gap_persistent_lower_than_random():
    rng = np.random.default_rng(4)
    persistent = pd.DataFrame(np.sin(np.linspace(0, 60, 400)),
                              index=pd.date_range("2024-01-01", periods=400))
    random = pd.DataFrame(rng.standard_normal((400, 1)), index=persistent.index)
    gap_p = _get("ts_markov_spectral_gap").calculate(persistent, window=120, bins=3, lag=1).to_numpy(dtype=float)
    gap_r = _get("ts_markov_spectral_gap").calculate(random, window=120, bins=3, lag=1).to_numpy(dtype=float)
    # Strong periodicity -> strong second eigenvalue -> small spectral gap.
    assert float(np.nanmean(gap_p[200:])) < float(np.nanmean(gap_r[200:]))


def test_markov_stationary_surprisal_rare_state():
    # 95% of the time the value is 1; 5% it is 100 -> the 100-state is rare.
    vals = np.ones(600)
    vals[::20] = 100.0
    x = pd.DataFrame(vals, index=pd.date_range("2024-01-01", periods=600))
    out = _get("ts_markov_stationary_surprisal").calculate(x, window=120, bins=3, lag=1).to_numpy(dtype=float)
    fin = out[200:].reshape(-1)
    rare = float(np.nanmax(fin))  # most surprising = the rare state
    assert rare > 1.5  # -log(0.05) ~ 3; with 3 bins the rarest bin is well under 1/3


def test_km_equilibrium_distance_attractor():
    # Mean-reverting process pinned near 0 with strong pull -> D1 crosses 0 at 0.
    rng = np.random.default_rng(5)
    n = 500
    xv = np.zeros(n)
    for t in range(1, n):
        xv[t] = 0.8 * xv[t - 1] + 0.05 * rng.standard_normal()
    x = pd.DataFrame(xv, index=pd.date_range("2024-01-01", periods=n))
    out = _get("ts_km_equilibrium_distance").calculate(x, window=120, bins=5, lag=1).to_numpy(dtype=float)
    late = out[200:][np.isfinite(out[200:])]
    # x* ~ 0, so the distance tracks the current level; the attractor-centred
    # z must be bounded and mean-reverting (not a raw drift).
    assert abs(float(np.nanmean(late))) < 1.0


def test_first_passage_hit_probability_certain():
    # Deterministic upward drift with a tiny barrier -> upper hit is near certain.
    logp = np.log(np.arange(1.0, 400.0))
    scale = np.ones_like(logp) * 0.001
    x = pd.DataFrame(logp, index=pd.date_range("2024-01-01", periods=logp.size))
    s = pd.DataFrame(scale, index=x.index)
    out = _get("ts_first_passage_hit_probability").calculate(x, s, window=120, barrier=1.0, horizon=5, side="upper").to_numpy(dtype=float)
    late = out[100:]
    late = late[np.isfinite(late)]
    assert float(np.mean(late)) > 0.5


def test_first_passage_conditional_time_between_0_and_1():
    rng = np.random.default_rng(6)
    n = 300
    x = pd.DataFrame(rng.standard_normal((n, 1)), index=pd.date_range("2024-01-01", periods=n))
    s = pd.DataFrame(np.ones((n, 1)), index=x.index)
    out = _get("ts_first_passage_conditional_time").calculate(x, s, window=120, barrier=1.0, horizon=10, side="upper").to_numpy(dtype=float)
    fin = out[np.isfinite(out)]
    assert np.all((fin > 0.0) & (fin <= 1.0))


def test_extremal_index_isolated_vs_clustered():
    n = 600
    iso = np.zeros(n); iso[::15] = 1.0
    clu = np.zeros(n); clu[50:60] = 1.0; clu[200:204] = 1.0
    x1 = pd.DataFrame(iso, index=pd.date_range("2024-01-01", periods=n))
    x2 = pd.DataFrame(clu, index=pd.date_range("2024-01-01", periods=n))
    t1 = _get("ts_extremal_index").calculate(x1, window=300, q=0.9).to_numpy(dtype=float)
    t2 = _get("ts_extremal_index").calculate(x2, window=300, q=0.9).to_numpy(dtype=float)
    assert float(np.nanmean(t1[300:])) > 0.9   # isolated -> theta ~ 1
    assert float(np.nanmean(t2[300:])) < 0.5   # clustered -> theta small


def test_gpd_shape_pwm_recovers_known_shape():
    from scipy import stats
    rng = np.random.default_rng(11)
    g = stats.genpareto.rvs(c=0.25, scale=1.0, size=20000, random_state=rng)
    x = pd.DataFrame(1.0 + g, index=pd.date_range("2000-01-01", periods=20000))
    out = _get("ts_gpd_shape_pwm").calculate(x, window=19000, tail_fraction=0.3, min_tail_count=100).to_numpy(dtype=float)
    est = float(np.nanmean(out[19000:]))
    assert abs(est - 0.25) < 0.05


def test_quantile_transport_slope_positive_on_tail_spread():
    rng = np.random.default_rng(9)
    n = 400
    # recent window is drawn from a mixture that widens the upper tail.
    base = rng.standard_normal(n)
    x = pd.DataFrame(base + np.where(np.arange(n) > 300, 2.0 * (base > 1.0), 0.0),
                     index=pd.date_range("2024-01-01", periods=n))
    out = _get("ts_quantile_transport_slope").calculate(x, recent_window=40, old_window=120).to_numpy(dtype=float)
    late = out[160:]
    late = late[np.isfinite(late)]
    assert float(np.nanmean(late)) > 0.0  # upper tail expanding -> positive slope


def test_mmd_identical_vs_shifted():
    rng = np.random.default_rng(10)
    n = 400
    x = pd.DataFrame(rng.standard_normal(n), index=pd.date_range("2024-01-01", periods=n))
    # shift applied only in the recent window via a deterministic ramp.
    off = np.zeros(n); off[300:] += np.linspace(0, 1.0, 100)
    y = pd.DataFrame(x.to_numpy() + off[:, None], index=x.index)
    same = _get("ts_mmd_rbf_shift").calculate(x, recent_window=40, old_window=120).to_numpy(dtype=float)
    diff = _get("ts_mmd_rbf_shift").calculate(y, recent_window=40, old_window=120).to_numpy(dtype=float)
    assert float(np.nanmean(same[160:])) < 0.02
    assert float(np.nanmean(diff[160:])) > float(np.nanmean(same[160:]))


def test_chord_excursion_area_hump_positive_dip_negative():
    w = 40
    hump = np.zeros(200)
    for k in range(w):
        hump[k] = 1.0 - abs(k - (w - 1) / 2.0) / ((w - 1) / 2.0)
    dip = -hump
    x = pd.DataFrame(np.column_stack([hump, dip]), index=pd.date_range("2024-01-01", periods=200))
    out = _get("ts_chord_excursion_area").calculate(x, window=w).to_numpy(dtype=float)
    # The full hump window sits entirely above its flat chord -> ~ +1.
    assert float(out[w - 1, 0]) > 0.9
    assert float(out[w - 1, 1]) < -0.9


def test_chord_max_excursion_vshape():
    w = 40
    v = np.zeros(200)
    for k in range(w):
        v[k] = -1.0 + abs(k - (w - 1) / 2.0) / ((w - 1) / 2.0)  # V: endpoints 0, trough -1
    x = pd.DataFrame(v, index=pd.date_range("2024-01-01", periods=200))
    out = _get("ts_max_chord_excursion").calculate(x, window=w).to_numpy(dtype=float)
    # Chord from 0 to 0; max deviation = 1; path length = 2 -> MCE ~ 0.5.
    assert abs(float(out[w - 1, 0]) - 0.5) < 0.02


def test_recurrence_rate_periodic_beats_random():
    rng = np.random.default_rng(12)
    per = pd.DataFrame(np.sin(np.linspace(0, 20 * np.pi, 400)), index=pd.date_range("2024-01-01", periods=400))
    rnd = pd.DataFrame(rng.standard_normal(400), index=per.index)
    rr_p = _get("ts_recurrence_rate").calculate(per, window=40).to_numpy(dtype=float)
    rr_r = _get("ts_recurrence_rate").calculate(rnd, window=40).to_numpy(dtype=float)
    div_p = _get("ts_recurrence_divergence").calculate(per, window=40).to_numpy(dtype=float)
    div_r = _get("ts_recurrence_divergence").calculate(rnd, window=40).to_numpy(dtype=float)
    assert float(np.nanmean(rr_p[100:])) > float(np.nanmean(rr_r[100:]))
    assert float(np.nanmean(div_p[100:])) < float(np.nanmean(div_r[100:]))


def test_transfer_entropy_peak_lag_detects_coupled_lag():
    rng = np.random.default_rng(13)
    n = 400
    s = rng.standard_normal(n)
    t = rng.standard_normal(n)
    for i in range(3, n):
        t[i] += 0.9 * s[i - 3]   # genuine coupling at lag 3
    sf = pd.DataFrame(s, index=pd.date_range("2024-01-01", periods=n))
    tf = pd.DataFrame(t, index=sf.index)
    out = _get("ts_transfer_entropy_peak_lag").calculate(tf, sf, window=150, bins=3).to_numpy(dtype=float)
    late = out[150:]
    late = late[np.isfinite(late)]
    assert abs(float(np.nanmean(late)) - 0.3) < 0.15  # l* = 3 / 10


def test_report_change_breadth_all_rising():
    n = 600
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    quarters: list[pd.Timestamp] = []
    y, q = 2019, 1
    for _ in range(40):
        quarters.append(pd.Timestamp(year=y, month=3 * q, day=28) if q in (1, 2)
                        else pd.Timestamp(year=y, month=3 * q - 2, day=30))
        q += 1
        if q == 5:
            q = 1
            y += 1
    per = np.array(quarters, dtype=object)
    period = pd.DataFrame(index=idx, columns=["S0"], dtype=object)
    col = np.full(n, None, dtype=object); col[0::15] = per[: len(col[0::15])]; period["S0"] = col
    base = np.linspace(100.0, 300.0, n)
    f1 = pd.DataFrame({"S0": base * 1.01}, index=idx)
    f2 = pd.DataFrame({"S0": base * 0.5}, index=idx)
    f3 = pd.DataFrame({"S0": 200.0 + base * 0.3}, index=idx)
    b = _get("report_change_breadth").calculate(f1, f2, f3, period, periods=1).to_numpy(dtype=float)
    c = _get("report_change_coherence").calculate(f1, f2, f3, period, periods=1).to_numpy(dtype=float)
    fin_b = b[np.isfinite(b)]
    fin_c = c[np.isfinite(c)]
    assert float(np.nanmean(fin_b)) > 0.5   # all rising -> breadth near +1
    assert float(np.nanmean(fin_c)) > 0.9   # dominant direction shared


def test_intraday_rv_signature_slope_noise_negative():
    rng = np.random.default_rng(3)
    n_min = 720
    lp = np.cumsum(rng.standard_normal(n_min) * 0.001)
    clean = np.exp(lp)
    noisy = np.exp(lp + 0.004 * rng.standard_normal(n_min))
    idx = pd.date_range("2024-01-02 09:30", periods=n_min, freq="min")
    x = pd.DataFrame(np.column_stack([clean, noisy]), index=idx)
    out = _get("intraday_rv_signature_slope").calculate(x).to_numpy(dtype=float)
    assert abs(float(out[0, 0])) < 0.3            # diffusion ~ flat signature
    assert float(out[0, 1]) < -0.3                # noise inflates 1-min RV

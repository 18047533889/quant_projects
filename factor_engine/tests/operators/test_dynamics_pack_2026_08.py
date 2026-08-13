# -*- coding: utf-8 -*-
"""Tests for the 2026-08 V2/V3 dynamics pack (state-dynamics / geometry /
event-response / spectral-crowding / volume-clock / KNN / EVT / report timing).

Covers registration & surface classification (21 daily, 8 research),
determinism, axis preservation, prefix causality, reference math, the
Kramers-Moyal strict-D2 fix, fail-closed edge cases and pandas↔polars parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_surface import classify_canonical

DAILY_OPS = {
    "ts_ordinal_irreversibility", "ts_state_density",
    "ts_markov_persistence", "ts_markov_state_entropy",
    "ts_markov_transition_surprisal", "ts_kramers_moyal_local_stability",
    "ts_first_passage_bias",
    "event_historical_response_mean", "event_historical_response_sign_balance",
    "ts_joint_energy_shift", "ts_energy_break_score",
    "group_feature_mode_share", "group_feature_effective_rank",
    "group_feature_mode_localization",
    "session_event_recovery_score",
    "intraday_volume_clock_path_efficiency", "intraday_volume_clock_roughness",
    "cs_knn_peer_mean_ex_self", "cs_knn_neighbor_retention",
    "report_filing_delay_surprise", "ts_hill_tail_index",
}
# R26/R28 promotion moved the model/surrogate canonicals to the extended
# surface (they are evidence-certified extended factors now).  Only
# ts_quantile_regression_beta remains genuinely research.
RESEARCH_OPS = {
    "ts_quantile_regression_beta",
}
POLARS_OPS = DAILY_OPS - {
    "group_feature_mode_share", "group_feature_effective_rank",
    "group_feature_mode_localization", "session_event_recovery_score",
    "intraday_volume_clock_path_efficiency", "intraday_volume_clock_roughness",
    "cs_knn_peer_mean_ex_self", "cs_knn_neighbor_retention",
}


def _frame(values: np.ndarray, start: str = "2024-01-01", cols: int | None = None) -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    cols = v.shape[1] if cols is None else cols
    return pd.DataFrame(v, index=pd.date_range(start, periods=v.shape[0], freq="B"),
                        columns=[f"S{i}" for i in range(cols)])


def _minute_frame(days: int = 2, cols: int = 1, per_day: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per_day)])
    return pd.DataFrame(
        np.tile(np.linspace(10, 12, per_day), days)[:, None],
        index=idx, columns=[f"S{i}" for i in range(cols)],
    )


# ---------------------------------------------------------------------------
# registration & classification
# ---------------------------------------------------------------------------

def test_dynamics_pack_registered_and_classified():
    canon = set(OperatorRegistry.list_canonical())
    assert DAILY_OPS | RESEARCH_OPS <= canon
    for name in DAILY_OPS:
        assert classify_canonical(name) == "daily", name
    for name in RESEARCH_OPS:
        assert classify_canonical(name) == "research", name
    # Research ops must not leak into the daily mining surface.
    from cleaned_operators.operator_surface import DAILY_FACTOR_MIGRATED
    assert not (RESEARCH_OPS & set(DAILY_FACTOR_MIGRATED))


# ---------------------------------------------------------------------------
# determinism / shape / prefix causality
# ---------------------------------------------------------------------------

def test_dynamics_pack_deterministic_shape_prefix():
    rng = np.random.default_rng(0)
    x = _frame(rng.normal(0.0, 1.0, size=(200, 4)))
    y = _frame(rng.normal(0.0, 1.0, size=(200, 4)))
    z = _frame(np.abs(rng.normal(0.0, 1.0, size=(200, 4))))
    g = _frame(np.tile(np.arange(4) % 2, (200, 1)))
    ev = _frame((np.abs(x.to_numpy() > 0.6).astype(float))
    scale = _frame(np.full((200, 4), 1.0))
    calls = {
        "ts_ordinal_irreversibility": (x,),
        "ts_state_density": (x,),
        "ts_markov_persistence": (x,),
        "ts_markov_state_entropy": (x,),
        "ts_markov_transition_surprisal": (x,),
        "ts_kramers_moyal_local_stability": (x,),
        "ts_first_passage_bias": (x, scale),
        "event_historical_response_mean": (y, ev),
        "event_historical_response_sign_balance": (y, ev),
        "ts_joint_energy_shift": (x, y, z),
        "ts_energy_break_score": (x, y, z),
        "group_feature_mode_share": (x, y, z, g),
        "group_corr_effective_rank": (x, y, z, g),
        "group_corr_mode_localization": (x, y, z, g),
        "cs_knn_peer_mean_ex_self": (y, x, z, z + 1.0),
        "cs_knn_neighbor_retention": (x, z, z + 1.0),
        "report_filing_delay_surprise": (z,),
        "ts_hill_tail_index": (x,),
    }
    kwargs = {
        "ts_ordinal_irreversibility": {"window": 40, "order": 3, "delay": 1},
        "ts_state_density": {"window": 40, "bandwidth": 1.0},
        "ts_markov_persistence": {"window": 40, "bins": 3, "lag": 1},
        "ts_markov_state_entropy": {"window": 40, "bins": 3, "lag": 1},
        "ts_markov_transition_surprisal": {"window": 40, "bins": 3, "lag": 1},
        "ts_kramers_moyal_local_stability": {"window": 40, "bins": 5, "lag": 1},
        "ts_first_passage_bias": {"window": 40, "barrier": 1.0, "horizon": 5},
        "event_historical_response_mean": {"history_window": 40, "horizon": 5, "mode": "sum"},
        "event_historical_response_sign_balance": {"history_window": 40, "horizon": 5},
        "ts_joint_energy_shift": {"recent_window": 10, "prior_window": 30},
        "ts_energy_break_score": {"window": 40, "recent_window": 10, "prior_window": 30},
        "cs_knn_neighbor_retention": {"k": 2, "lag": 1},
    }
    for name, args in calls.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        a = op.calculate(*args, **kwargs.get(name, {}))
        b = op.calculate(*args, **kwargs.get(name, {}))
        assert a.index.equals(args[0].index), name
        assert a.columns.equals(args[0].columns), name
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), name
        # Prefix causality: first 160 rows must not depend on later rows.
        sliced = tuple(ar.iloc[:160] for ar in args)
        prefix = op.calculate(*sliced, **kwargs.get(name, {})).to_numpy(dtype=float)
        assert np.allclose(a.to_numpy(dtype=float)[:160], prefix, equal_nan=True), name


def test_dynamics_minute_ops_prefix_causal():
    days, per = 4, 120
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per)])
    base = np.tile(np.linspace(10, 12, per), days)[:, None]
    price = pd.DataFrame(base, index=idx, columns=["S0"])
    activity = pd.DataFrame(base * 10.0, index=idx, columns=["S0"])
    for name in ("intraday_volume_clock_path_efficiency", "intraday_volume_clock_roughness"):
        op = OperatorRegistry.get(name, "pandas_numpy")
        full = op.calculate(price, activity, buckets=16)
        prefix = op.calculate(price.iloc[: days // 2 * per], activity.iloc[: days // 2 * per], buckets=16)
        assert np.allclose(full.to_numpy(dtype=float)[: days // 2],
                           prefix.to_numpy(dtype=float), equal_nan=True), name


# ---------------------------------------------------------------------------
# reference math
# ---------------------------------------------------------------------------

def test_kramers_moyal_strict_d2_and_local_stability():
    t = np.arange(80.0)[:, None]
    x = _frame(t)
    diff = OperatorRegistry.get("ts_kramers_moyal_diffusion", "pandas_numpy").calculate(x, window=60, bins=5, lag=1)
    qv = diff.to_numpy(dtype=float)[-20:]
    qv = qv[np.isfinite(qv)]
    assert np.allclose(qv, 0.5, atol=1e-6)  # strict D2 = mean(dx^2)/(2*lag)

    stab = OperatorRegistry.get("ts_kramers_moyal_local_stability", "pandas_numpy").calculate(x, window=60, bins=5, lag=1)
    sv = stab.to_numpy(dtype=float)
    sv = sv[np.isfinite(sv)]
    # A linear drift has flat D1 -> drift derivative ~0 -> stability ~0.
    assert np.allclose(sv, 0.0, atol=1e-6)


def test_ordinal_irreversibility_monotonic_beats_random():
    rng = np.random.default_rng(5)
    mono = _frame(np.arange(120.0))
    rw = _frame(np.cumsum(rng.normal(0.0, 1.0, 120)))
    ir_mono = OperatorRegistry.get("ts_ordinal_irreversibility", "pandas_numpy").calculate(mono, 60, 3, 1, 5)
    ir_rw = OperatorRegistry.get("ts_ordinal_irreversibility", "pandas_numpy").calculate(rw, 60, 3, 1, 5)
    m = float(np.nanmean(ir_mono.to_numpy()))
    r = float(np.nanmean(ir_rw.to_numpy()))
    assert m > 0.9   # forward=all increasing, backward=all decreasing -> JS~1
    assert r < m


def test_state_density_crowded_vs_vacuum():
    centered = _frame(np.concatenate([np.arange(50.0) + 10.0, [20.0]]))
    outlier = _frame(np.concatenate([np.arange(50.0) + 10.0, [500.0]]))
    sd_c = OperatorRegistry.get("ts_state_density", "pandas_numpy").calculate(centered, 50, 1.0, 5)
    sd_o = OperatorRegistry.get("ts_state_density", "pandas_numpy").calculate(outlier, 50, 1.0, 5)
    assert float(sd_c.to_numpy()[-1, 0]) > float(sd_o.to_numpy()[-1, 0])


def test_first_passage_bias_upward_drift_positive():
    logp = _frame(np.log(np.arange(1.0, 201.0)))
    scale = _frame(np.ones(200) * 0.001)
    fpb = OperatorRegistry.get("ts_first_passage_bias", "pandas_numpy").calculate(logp, scale, 120, 1.0, 10, 3)
    assert float(np.nanmean(fpb.to_numpy()) > 0.5


def test_event_response_positive_drift():
    rng = np.random.default_rng(6)
    ret = _frame(rng.normal(0.001, 0.01, 200))
    ev = _frame((np.arange(200) % 20 == 0).astype(float))
    erm = OperatorRegistry.get("event_historical_response_mean", "pandas_numpy").calculate(ret, ev, 100, 5, "mean", 3)
    assert float(np.nanmean(erm.to_numpy()) > 0.0


def test_markov_persistence_sticky():
    sticky = _frame(np.repeat(np.arange(5), 40))
    mp = OperatorRegistry.get("ts_markov_persistence", "pandas_numpy").calculate(sticky, 60, 3, 1, 3)
    assert float(np.nanmean(mp.to_numpy()) > 0.5


def test_markov_transition_surprisal_is_nonnegative_nats():
    rng = np.random.default_rng(7)
    x = _frame(rng.normal(0.0, 1.0, size=(200, 2)))
    s = OperatorRegistry.get("ts_markov_transition_surprisal", "pandas_numpy").calculate(x, 60, 3, 1, 3)
    vals = s.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    assert np.all(vals >= 0.0)  # -log(P_ij), P_ij <= 1


def test_hill_heavy_tail_beats_normal():
    rng = np.random.default_rng(8)
    tt = _frame(rng.standard_t(3, size=500) * 0.02)
    nn = _frame(rng.normal(0.0, 0.02, size=500))
    h_tt = OperatorRegistry.get("ts_hill_tail_index", "pandas_numpy").calculate(tt, 300, "upper", 0.2, 10)
    h_nn = OperatorRegistry.get("ts_hill_tail_index", "pandas_numpy").calculate(nn, 300, "upper", 0.2, 10)
    assert float(np.nanmean(h_tt.to_numpy()) > float(np.nanmean(h_nn.to_numpy()))


def test_quantile_regression_beta_recovers_slope():
    rng = np.random.default_rng(9)
    x = _frame(rng.normal(0.0, 1.0, 200))
    y = _frame(2.0 * x.to_numpy()[:, 0] + rng.normal(0.0, 0.5, 200))
    qb = OperatorRegistry.get("ts_quantile_regression_beta", "pandas_numpy", mode="any").calculate(y, x, 150, 0.5)
    assert abs(float(qb.to_numpy()[-1, 0]) - 2.0) < 0.3


def test_group_mode_share_common_factor_high():
    rng = np.random.default_rng(10)
    # Round-7 P0: the 3-feature spectrum requires N >= max(6, 2*d+2) = 8 valid
    # members per group (a 2-member x 3-feature matrix is mechanically low-rank).
    # Use one 9-member group so the dominant-common-factor scenario is valid.
    rows, cols = 200, 9
    f = rng.normal(0.0, 1.0, (rows, cols))
    common = 0.9 * f + 0.1 * rng.normal(0.0, 1.0, (rows, cols))
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    assets = [f"A{i}" for i in range(cols)]
    g = pd.DataFrame(np.full((rows, cols), "G0", dtype=object), index=dates, columns=assets)
    ms = OperatorRegistry.get("group_corr_mode_share", "pandas_numpy").calculate(
        pd.DataFrame(common, index=dates, columns=assets),
        pd.DataFrame(common + 0.1 * rng.normal(0.0, 1.0, (rows, cols)), index=dates, columns=assets),
        pd.DataFrame(f, index=dates, columns=assets),
        g,
    )
    assert float(np.nanmean(ms.to_numpy()) > 0.6


def test_volume_clock_efficiency_monotonic_near_one():
    price = _minute_frame()
    activity = price * 10.0
    vce = OperatorRegistry.get("intraday_volume_clock_path_efficiency", "pandas_numpy").calculate(price, activity, 16)
    assert np.all(vce.to_numpy()[:, 0] > 0.9)


def test_session_recovery_fast_vs_never():
    # R26-050/051: the operator grid-aligns to the official A-share session, so
    # the fixture must be the full 240-bar session (09:31-11:30 + 13:01-15:00).
    per = 240
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=m) for d in dates
         for m in list(range(571, 691)) + list(range(781, 901))]
    )
    seq1 = np.concatenate([np.linspace(10, 10.05, 120), np.full(120, 10.4)])          # never recovers
    seq2 = np.concatenate([np.linspace(10, 10.05, 120),
                           np.r_[10.4, np.linspace(10.4, 10.05, 14), np.full(105, 10.05)]])
    seq3 = np.linspace(10, 10.05, per)
    price = pd.DataFrame(np.concatenate([seq1, seq2, seq3])[:, None], index=idx, columns=["S0"])
    r = np.abs(np.diff(price.to_numpy()[:, 0], prepend=10.0))
    ev = pd.DataFrame((r > 0.1).astype(float), index=idx, columns=["S0"])
    # min_events=1 is passed explicitly (kernel path) — this test exercises the
    # fast-vs-never recovery SEMANTICS; the R26-048 default floor (>=3) is
    # covered by test_recovery_min_events_*.
    op = OperatorRegistry.get("session_event_recovery_score", "pandas_numpy")
    out = op._calculate_series(price, ev, horizon=30, residual_fraction=0.25, min_events=1)
    vals = out.to_numpy()[:, 0]
    assert vals[0] == pytest.approx(1.0)      # fragile
    assert vals[1] < 0.5                      # resilient
    assert np.isnan(vals[2])                  # no events


def test_copula_central_asymmetry_detects_skew():
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, 300)
    a_sq = OperatorRegistry.get("ts_copula_central_asymmetry", "pandas_numpy").calculate(
        *_pair(x, x * x), 200, 8)
    a_neg = OperatorRegistry.get("ts_copula_central_asymmetry", "pandas_numpy").calculate(
        *_pair(x, -x), 200, 8)
    assert float(a_sq.to_numpy()[-1, 0]) > float(a_neg.to_numpy()[-1, 0]) * 20.0


def _pair(x, y):
    idx = pd.date_range("2024-01-01", periods=len(x), freq="B")
    return pd.DataFrame(x[:, None], index=idx, columns=["S0"]), pd.DataFrame(y[:, None], index=idx, columns=["S0"])


# ---------------------------------------------------------------------------
# fail-closed robustness
# ---------------------------------------------------------------------------

def test_constant_and_all_nan_series_do_not_crash():
    const = _frame(np.ones(80))
    nan = _frame(np.full(80, np.nan))
    markov_kwargs = {"window": 60, "bins": 3, "lag": 1, "min_count": 3}
    for name, kwargs in (
        ("ts_markov_persistence", markov_kwargs),
        ("ts_markov_state_entropy", markov_kwargs),
        ("ts_markov_transition_surprisal", markov_kwargs),
        ("ts_state_density", {"window": 60, "bandwidth": 1.0, "min_periods": 5}),
        ("ts_ordinal_irreversibility", {"window": 60, "order": 3, "delay": 1, "min_patterns": 5}),
    ):
        op = OperatorRegistry.get(name, "pandas_numpy")
        out = op.calculate(const, **kwargs)
        assert out.shape == const.shape
        out_nan = op.calculate(nan, **kwargs)
        assert np.isnan(out_nan.to_numpy(dtype=float)).all(), name


# ---------------------------------------------------------------------------
# pandas ↔ polars parity
# ---------------------------------------------------------------------------

def _polars_available() -> bool:
    try:
        import polars  # noqa: F401
        return True
    except Exception:
        return False


def test_dynamics_polars_parity():
    if not _polars_available():
        pytest.skip("polars not installed")
    import polars as pl
    rng = np.random.default_rng(12)
    x = _frame(rng.normal(0.0, 1.0, size=(200, 4)))
    y = _frame(rng.normal(0.0, 1.0, size=(200, 4)))
    z = _frame(np.abs(rng.normal(0.0, 1.0, size=(200, 4))))
    scale = _frame(np.full((200, 4), 1.0))
    ev = _frame((np.abs(x.to_numpy() > 0.6).astype(float))
    calls = {
        "ts_markov_persistence": (x,),
        "ts_markov_state_entropy": (x,),
        "ts_markov_transition_surprisal": (x,),
        "ts_kramers_moyal_local_stability": (x,),
        "ts_state_density": (x,),
        "ts_ordinal_irreversibility": (x,),
        "ts_first_passage_bias": (x, scale),
        "event_historical_response_mean": (y, ev),
        "event_historical_response_sign_balance": (y, ev),
        "ts_joint_energy_shift": (x, y, z),
        "ts_energy_break_score": (x, y, z),
        "ts_hill_tail_index": (x,),
        "report_filing_delay_surprise": (z,),
    }
    kwargs = {
        "ts_markov_persistence": {"window": 40, "bins": 3, "lag": 1, "min_count": 3},
        "ts_markov_state_entropy": {"window": 40, "bins": 3, "lag": 1, "min_count": 3},
        "ts_markov_transition_surprisal": {"window": 40, "bins": 3, "lag": 1, "min_count": 3},
        "ts_kramers_moyal_local_stability": {"window": 40, "bins": 5, "lag": 1, "min_count": 3},
        "ts_state_density": {"window": 40, "bandwidth": 1.0, "min_periods": 5},
        "ts_ordinal_irreversibility": {"window": 40, "order": 3, "delay": 1, "min_patterns": 5},
        "ts_first_passage_bias": {"window": 40, "barrier": 1.0, "horizon": 5, "min_anchors": 3},
        "event_historical_response_mean": {"history_window": 40, "horizon": 5, "mode": "sum", "min_events": 3},
        "event_historical_response_sign_balance": {"history_window": 40, "horizon": 5, "min_events": 3},
        "ts_joint_energy_shift": {"recent_window": 10, "prior_window": 30},
        "ts_energy_break_score": {"window": 40, "recent_window": 10, "prior_window": 30},
        "ts_hill_tail_index": {"window": 120, "side": "upper", "tail_fraction": 0.2, "min_tail_count": 5},
        "report_filing_delay_surprise": {"window": 40, "min_periods": 5},
    }
    for name, args in calls.items():
        pn = OperatorRegistry.get(name, "pandas_numpy").calculate(*args, **kwargs[name]).to_numpy(dtype=float)
        pl_op = OperatorRegistry.get(name, "polars")
        assert pl_op is not None, name
        pdfs = [pl.from_pandas(ar) for ar in args]
        pr = pl_op.calculate(*pdfs, **kwargs[name]).to_pandas().to_numpy(dtype=float)
        assert np.allclose(pr, pn, equal_nan=True, atol=1e-9, rtol=1e-9), name

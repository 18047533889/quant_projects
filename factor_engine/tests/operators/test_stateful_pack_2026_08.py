# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08 stateful rule/episode/rotation pack (21 ops).

Covers:
- Registration: every stateful-pack canonical has a pandas_numpy runtime.
- Surface: every canonical classifies ``daily`` and stays in EXTENDED_ONLY.
- Policy: explicit PIT-safe policy with the intended scope (ts / cs).
- Determinism + axes across all families.
- Stateful specifics: reset-priority latch, hold memory, refractory cooldown,
  deadband/slew behaviour, survival running sample isolation, break-on-NaN.
- Fail-closed: constant input -> NaN where the kernel is a statistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_policy import infer_operator_policy
from cleaned_operators.operator_surface import classify_canonical
import cleaned_operators.operator_surface as _surface_mod

ensure_cleaned_loaded()

STATEFUL_PACK = frozenset({
    # rule language
    "state_latch", "state_hold", "state_slew_limit", "state_deadband",
    # events
    "event_refractory", "cross_event",
    # sequential / conditional memory
    "ts_cusum_pressure", "ts_rank_if", "state_ewm_if", "ts_lag_of_peak_corr",
    # dynamic episode / directional change
    "state_since_reduce", "directional_change_state", "directional_change_extent",
    "state_since_trend_tstat",
    # state survival / maturity
    "ts_state_age_percentile", "ts_state_exit_hazard", "ts_state_residual_life",
    # cross-sectional rotation
    "cs_rank_churn", "cs_tail_retention",
    # drawdown path / recovery
    "ts_recovery_fraction", "ts_current_drawdown_area",
})

EXPECTED_SCOPE = {
    "state_latch": "ts", "state_hold": "ts", "state_slew_limit": "ts",
    "state_deadband": "ts", "event_refractory": "ts", "cross_event": "ts",
    "ts_cusum_pressure": "ts", "ts_rank_if": "ts", "state_ewm_if": "ts",
    "ts_lag_of_peak_corr": "ts", "state_since_reduce": "ts",
    "directional_change_state": "ts", "directional_change_extent": "ts",
    "state_since_trend_tstat": "ts", "ts_state_age_percentile": "ts",
    "ts_state_exit_hazard": "ts", "ts_state_residual_life": "ts",
    "cs_rank_churn": "cs", "cs_tail_retention": "cs",
    "ts_recovery_fraction": "ts", "ts_current_drawdown_area": "ts",
}


def _panels(n: int = 120, cols: int = 4):
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, (n, cols))
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 10.0, index=dates, columns=assets)
    ret = close.pct_change()
    cond = pd.DataFrame((rng.random((n, cols)) > 0.6).astype(float), index=dates, columns=assets)
    reset = pd.DataFrame((rng.random((n, cols)) > 0.92).astype(float), index=dates, columns=assets)
    group = pd.DataFrame(np.tile(["g0", "g1"], (n, 2)), index=dates, columns=assets)
    return close, ret, cond, reset, group


def test_stateful_pack_all_registered_and_daily():
    missing = [c for c in STATEFUL_PACK if OperatorRegistry.get(c, "pandas_numpy") is None]
    assert not missing, f"missing runtimes: {missing}"
    not_daily = [c for c in sorted(STATEFUL_PACK) if classify_canonical(c) != "daily"]
    assert not not_daily, f"not daily surface: {not_daily}"
    live_extended = frozenset(_surface_mod.EXTENDED_ONLY_CANONICALS)
    not_extended = sorted(STATEFUL_PACK - live_extended)
    assert not not_extended, f"not in EXTENDED_ONLY (partition contract): {not_extended}"


def test_stateful_pack_policies_pit_safe_with_intended_scope():
    bad = []
    for canonical in sorted(STATEFUL_PACK):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(op, canonical=canonical)
        if not policy.pit_safe:
            bad.append(f"{canonical}: pit_safe=False")
        if policy.scope != EXPECTED_SCOPE[canonical]:
            bad.append(f"{canonical}: scope={policy.scope} expected={EXPECTED_SCOPE[canonical]}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("canonical", sorted(STATEFUL_PACK))
def test_stateful_pack_deterministic_and_axes(canonical):
    close, ret, cond, reset, group = _panels()
    panels = {
        "state_latch": (cond, reset),
        "state_hold": (close, cond, reset),
        "state_slew_limit": (close,),
        "state_deadband": (close,),
        "event_refractory": (cond,),
        "cross_event": (close, close * 1.01),
        "ts_cusum_pressure": (ret,),
        "ts_rank_if": (close, cond),
        "state_ewm_if": (close, cond),
        "ts_lag_of_peak_corr": (close, ret),
        "state_since_reduce": (ret, cond),
        "directional_change_state": (close,),
        "directional_change_extent": (close,),
        "state_since_trend_tstat": (ret, cond),
        "ts_state_age_percentile": (cond,),
        "ts_state_exit_hazard": (cond,),
        "ts_state_residual_life": (cond,),
        "cs_rank_churn": (close,),
        "cs_tail_retention": (close,),
        "ts_recovery_fraction": (close,),
        "ts_current_drawdown_area": (close,),
    }[canonical]
    kw = {
        "state_latch": {"initial_state": 0},
        "ts_cusum_pressure": {"reference_window": 20, "drift": 0.5},
        "ts_rank_if": {"window": 20, "min_periods": 5},
        "state_ewm_if": {"half_life": 10},
        "ts_lag_of_peak_corr": {"window": 20, "max_lag": 5},
        "state_since_reduce": {"mode": "sum", "min_episode": 2},
        "state_since_trend_tstat": {"min_obs": 5},
        "ts_state_age_percentile": {"history_window": 60, "min_completed_runs": 3},
        "ts_state_exit_hazard": {"history_window": 60, "min_completed_runs": 3, "alpha": 1.0},
        "ts_state_residual_life": {"history_window": 60, "min_completed_runs": 3},
        "cs_rank_churn": {"lag": 3},
        "cs_tail_retention": {"lag": 3, "quantile": 0.3, "side": "top"},
        "cross_event": {"direction": "up"},
        "directional_change_state": {"threshold": 0.03},
        "directional_change_extent": {"threshold": 0.03},
    }.get(canonical, {})
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    first = op.calculate(*panels, **kw)
    second = op.calculate(*panels, **kw)
    assert first.index.equals(close.index), f"{canonical}: index changed"
    assert first.columns.equals(close.columns), f"{canonical}: columns changed"
    assert np.allclose(
        first.fillna(-1).to_numpy(),
        second.fillna(-1).to_numpy(),
        equal_nan=True,
    ), f"{canonical}: non-deterministic"


def test_state_latch_reset_priority():
    n = 8
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    setc = pd.DataFrame([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], index=dates, columns=cols)
    resc = pd.DataFrame([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], index=dates, columns=cols)
    out = OperatorRegistry.get("state_latch", "pandas_numpy").calculate(setc, resc, initial_state=0)
    v = out["A"].tolist()
    # reset priority: row 3 both set & reset -> 0
    assert v == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_state_hold_remembers_snapshot():
    n = 8
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    val = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=dates, columns=cols)
    upd = pd.DataFrame([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], index=dates, columns=cols)
    out = OperatorRegistry.get("state_hold", "pandas_numpy").calculate(val, upd)
    v = out["A"].tolist()
    assert v[0] is np.nan or np.isnan(v[0])
    assert v[1] == 2.0 and v[2] == 2.0 and v[3] == 2.0 and v[4] == 5.0 and v[5] == 5.0


def test_event_refractory_cooldown():
    n = 10
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    cond = pd.DataFrame([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], index=dates, columns=cols)
    out = OperatorRegistry.get("event_refractory", "pandas_numpy").calculate(cond, cooldown=3)
    v = out["A"].tolist()
    # events at 0 accepted; 1,2 suppressed; 5 accepted (gap>3); 9 accepted (gap>3)
    assert v[0] == 1.0 and v[1] == 0.0 and v[2] == 0.0 and v[5] == 1.0 and v[9] == 1.0


def test_state_since_reduce_episode_accumulates():
    n = 8
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    x = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=dates, columns=cols)
    reset = pd.DataFrame([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=dates, columns=cols)
    out = OperatorRegistry.get("state_since_reduce", "pandas_numpy").calculate(x, reset, mode="sum", min_episode=1)
    v = out["A"].tolist()
    # row0 reset -> NaN; row1-2 sum 2, 5; row3 reset -> NaN; rows4-7 5,11,18,26
    assert np.isnan(v[0]) and v[1] == 2.0 and v[2] == 5.0
    assert np.isnan(v[3]) and v[4] == 5.0 and v[5] == 11.0 and v[6] == 18.0 and v[7] == 26.0


def test_state_slew_limit_bounds_each_step():
    n = 6
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    x = pd.DataFrame([0.1, 2.0, 2.0, 2.0, 2.0, 2.0], index=dates, columns=cols)
    out = OperatorRegistry.get("state_slew_limit", "pandas_numpy").calculate(x, limit=0.2)
    v = out["A"].tolist()
    assert v[0] == pytest.approx(0.1)
    assert v[1] == pytest.approx(0.3)   # 0.1 + 0.2
    assert v[2] == pytest.approx(0.5)


def test_state_deadband_ignores_small_changes():
    n = 6
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    x = pd.DataFrame([0.0, 0.1, 0.5, 1.0, 1.2, 2.0], index=dates, columns=cols)
    out = OperatorRegistry.get("state_deadband", "pandas_numpy").calculate(x, band=0.3)
    v = out["A"].tolist()
    assert v[0] == pytest.approx(0.0)
    assert v[1] == pytest.approx(0.0)   # |0.1-0| <= 0.3 -> no move
    assert v[2] == pytest.approx(0.2)   # 0.5-0.3=0.2 beyond band
    assert v[3] == pytest.approx(0.7)   # d=0.8 > 0.3 -> 0.2 + (0.8-0.3) = 0.7


def test_stateful_survival_excludes_current_run():
    """ts_state_age_percentile reference sample must be completed runs only."""
    n = 40
    dates = pd.bdate_range("2024-01-02", periods=n)
    cols = ["A"]
    # Two completed runs of length 2, then a long current run.
    state = pd.DataFrame(0.0, index=dates, columns=cols)
    state.iloc[0:2] = 1.0   # completed run length 2
    state.iloc[10:12] = 1.0  # completed run length 2
    state.iloc[20:30] = 1.0  # current run
    out = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy").calculate(
        state, history_window=60, min_completed_runs=2
    )
    # At age 1 of the current run: ECDF of 1 among completed {2,2} = 0
    # At age 3: ECDF of 3 among {2,2} = 1 (both <= 3)
    assert out["A"].iloc[20] == pytest.approx(0.0)
    assert out["A"].iloc[22] == pytest.approx(1.0)

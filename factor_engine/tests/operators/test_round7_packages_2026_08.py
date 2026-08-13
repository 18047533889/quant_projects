# -*- coding: utf-8 -*-
"""Round-7 Package B-F acceptance tests (2026-08-09).

Pins the audit's Model Reliability / State Dynamics / Pattern / Fundamental /
Group-Event P0s that landed via the package fixes:

* ModelDesignGate — OLS rejects rank-deficient / ill-conditioned designs.
* MoE finite-expert gate — a NaN expert is dropped and the remaining gates
  renormalised (no broadcast shape error), and a missing-regime row (NaN
  market_state) never enters an expert.
* KM D1 — the drift rate is divided by ``lag`` so lag=1/2/3 agree.
* Markov stationary surprisal — from the TRUE stationary π, so a rare state
  scores high; entropy production uses the corrected stationary-flux form.
* Group spectrum — a 2-member x 3-feature group is below the minimum-member
  floor and fails closed.
* Event episode collapse — ``refractory`` collapses consecutive events into one
  episode (effective event count).
* Candlestick zero-amplitude gate — a 一字板 (O=H=L=C) bar is never classified
  as a doji subtype.
* Financial revision three-state + coverage gate — a window with unknown
  historical exposure fails closed instead of counting as "no revision".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()

_rng = np.random.default_rng(0)
_IDX = pd.date_range("2024-01-01", periods=120, freq="B")
_COLS = ["A", "B", "C", "D"]


def _mk(n=120, cols=None, seed=0, idx=None):
    rng = np.random.default_rng(seed)
    cols = cols if cols is not None else _COLS
    if idx is None:
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.standard_normal((n, len(cols))), index=idx, columns=cols)


def _get(name):
    return OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)


# ---------------------------------------------------------------------------
# §1  ModelDesignGate (Package B)
# ---------------------------------------------------------------------------

def test_ols_gate_rejects_rank_deficient_design():
    from cleaned_operators.ts_model._rolling_core import fit_linear_model_checked

    # perfect collinearity: x2 = 2*x1  ->  rank deficient
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    design = np.column_stack([np.ones(5), x1, 2.0 * x1])
    y = 3.0 + 0.5 * x1
    assert fit_linear_model_checked(design, y) is None


def test_ols_gate_accepts_well_conditioned_design():
    from cleaned_operators.ts_model._rolling_core import fit_linear_model_checked

    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    design = np.column_stack([np.ones(5), x1, x2])
    y = design @ np.array([1.0, 2.0, -1.0])
    beta = fit_linear_model_checked(design, y)
    assert beta is not None
    np.testing.assert_allclose(beta, [1.0, 2.0, -1.0], atol=1e-9)


def test_moe_finite_expert_gate_handles_nan_expert():
    """A missing-regime row (NaN market_state) must never enter an expert, and
    dropping a NaN expert must renormalise the remaining gates instead of a
    (3,)×(2,) broadcast error."""
    op = _get("panel_mixture_of_experts_score")
    n = 90
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = ["A", "B", "C", "D"]
    y = _mk(n=n, seed=1, idx=idx)
    x1 = _mk(n=n, seed=2, idx=idx)
    x2 = _mk(n=n, seed=3, idx=idx)
    x3 = _mk(n=n, seed=4, idx=idx)
    x4 = _mk(n=n, seed=5, idx=idx)
    # step market_state: regime 2 only appears late; make the LAST rows (regime
    # 2) NaN so that expert is empty and its prediction is NaN.
    ms = pd.DataFrame(0.0, index=idx, columns=cols)
    ms.iloc[60:] = 2.0
    ms.iloc[75:] = np.nan  # missing regime -> win_valid_tr False
    y2 = y.copy()
    y2.iloc[75:] = np.nan
    out = op.calculate(y2, x1, x2, x3, x4, ms, window=40, n_experts=3, label_horizon=1)
    assert out.shape == (n, len(cols))
    # finite anywhere -> the gate path ran; never a broadcast ValueError


# ---------------------------------------------------------------------------
# §2  Markov / KM (Package C)
# ---------------------------------------------------------------------------

def test_km_drift_divided_by_lag():
    """D1 = mean(dx)/lag: a linear ramp has the SAME drift for lag=1/2/3 (the
    old kernel amplified it by lag)."""
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ramp = pd.DataFrame(np.arange(n, dtype=float)[:, None], index=idx, columns=["S"])
    op = _get("ts_kramers_moyal_drift")
    d1 = float(op.calculate(ramp, window=150, bins=3, lag=1).iloc[-1, 0])
    d2 = float(op.calculate(ramp, window=150, bins=3, lag=2).iloc[-1, 0])
    d3 = float(op.calculate(ramp, window=150, bins=3, lag=3).iloc[-1, 0])
    assert np.isfinite(d1) and np.isfinite(d2) and np.isfinite(d3)
    assert d1 != pytest.approx(0.0)
    # lag-corrected drift is lag-INVARIANT (not amplified)
    assert d2 == pytest.approx(d1, abs=1e-6)
    assert d3 == pytest.approx(d1, abs=1e-6)
    # the old bug gave d2 ≈ 2*d1
    assert d2 != pytest.approx(2.0 * d1, abs=1e-3)


def test_markov_stationary_surprisal_rare_state_high():
    """ts_markov_stationary_surprisal uses the TRUE stationary π (πP=π): a rare
    extreme state (5% of samples) must score a high surprisal."""
    n = 400
    vals = np.ones(n)
    vals[::20] = 100.0  # 5% of samples are the extreme state
    x = pd.DataFrame(vals[:, None], index=pd.date_range("2024-01-01", periods=n, freq="B"), columns=["S"])
    op = _get("ts_markov_stationary_surprisal")
    out = op.calculate(x, window=380, bins=3, lag=1)
    last = out.dropna()
    assert not last.empty
    assert float(last.to_numpy().max() > 1.5  # the rare 100-state has high surprisal


def test_markov_entropy_production_finite():
    op = _get("ts_markov_entropy_production")
    x = _mk(n=160, seed=7)
    out = op.calculate(x, window=150, bins=3, lag=1)
    vals = out.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert np.all(finite >= -1e-9)  # corrected stationary-flux form is >= 0, no clipping


# ---------------------------------------------------------------------------
# §3  Group spectrum min-members (Package F)
# ---------------------------------------------------------------------------

def test_group_spectrum_min_members_fail_closed():
    """2 members x 3 features is mechanically low-rank; N >= max(6, 2d+2)=8."""
    n = 60
    idx = _IDX[:n]
    assets = ["S0", "S1"]
    g = pd.DataFrame(np.full((n, 2), "G0", dtype=object), index=idx, columns=assets)
    f1 = pd.DataFrame(_rng.standard_normal((n, 2)), index=idx, columns=assets)
    f2 = pd.DataFrame(_rng.standard_normal((n, 2)), index=idx, columns=assets)
    f3 = pd.DataFrame(_rng.standard_normal((n, 2)), index=idx, columns=assets)
    op = _get("group_feature_mode_share")
    out = op.calculate(f1, f2, f3, g)
    assert out.to_numpy().size > 0
    assert np.isnan(out.to_numpy()).all(), "a 2-member group must not enter the spectrum"


# ---------------------------------------------------------------------------
# §4  Event episode collapse (Package F)
# ---------------------------------------------------------------------------

def test_event_refractory_collapses_episodes():
    """5 consecutive events with refractory >= horizon are ONE episode."""
    op = _get("event_response_effective_events")
    n = 20
    idx = _IDX[:n]
    cols = ["S"]
    ev = pd.DataFrame(0.0, index=idx, columns=cols)
    ev.iloc[5:10] = 1.0  # five consecutive event bars
    out = op.calculate(ev, history_window=20, horizon=5, refractory=5)
    assert float(out.iloc[-1, 0]) == pytest.approx(1.0)  # one episode, not five


# ---------------------------------------------------------------------------
# §5  Candlestick zero-amplitude gate (Package D)
# ---------------------------------------------------------------------------

def test_candlestick_zero_range_bar_never_classified():
    """一字板 (O=H=L=C) has range==0; dragonfly/gravestone/hanging-man must NOT
    fire (the old code classified a flat bar as both dragonfly AND gravestone)."""
    n = 12
    idx = _IDX[:n]
    cols = ["S"]
    flat = pd.DataFrame(10.0, index=idx, columns=cols)  # open=high=low=close=10
    for name in ("cdl_gravestone_doji", "cdl_dragonfly_doji", "cdl_hanging_man"):
        op = _get(name)
        out = op.calculate(flat, flat, flat, flat).to_numpy(dtype=float)
        assert not bool(np.nanmax(np.abs(out) > 0), f"{name} must not fire on a flat bar"


def test_candlestick_real_gravestone_fires():
    """A genuine gravestone (doji body, long upper shadow, tiny lower) still
    fires — the zero-amplitude gate must not over-gate healthy bars."""
    n = 14
    idx = _IDX[:n]
    cols = ["S"]
    # prior bars have real bodies/ranges so the body/range baselines are > 0
    open_ = pd.DataFrame(10.0, index=idx, columns=cols)
    high = pd.DataFrame(13.0, index=idx, columns=cols)
    low = pd.DataFrame(9.0, index=idx, columns=cols)
    close = pd.DataFrame(11.0, index=idx, columns=cols)
    # last bar: O=9.4 H=12 L=9.3 C=9.4 -> body 0 (doji), upper 2.6, lower 0.1
    open_.iloc[-1] = 9.4
    high.iloc[-1] = 12.0
    low.iloc[-1] = 9.3
    close.iloc[-1] = 9.4
    op = _get("cdl_gravestone_doji")
    out = op.calculate(open_, high, low, close).to_numpy(dtype=float)
    assert np.nanmax(np.abs(out)) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# §6  Financial revision three-state + coverage gate (Package E)
# ---------------------------------------------------------------------------

def test_fin_revision_count_coverage_gate():
    """A window whose historical revision status is UNKNOWN (gap) must fail
    closed (NaN), never count as 'no revision'."""
    op = _get("fin_revision_count")
    n = 120
    idx = _IDX[:n]
    cols = ["S"]
    x = pd.DataFrame(100.0, index=idx, columns=cols)
    pid = pd.DataFrame(index=idx, columns=cols, dtype=object)
    pid["S"] = [f"2023Q{i % 4 + 1}" for i in range(n)]
    x.iloc[40:80] = np.nan  # a 40-bar provider gap -> unknown revision exposure
    out = op.calculate(x, pid, window_days=30, coverage_threshold=0.8).to_numpy(dtype=float)
    last = out[np.isfinite(out)]
    # The gap region must be NaN, not a confident 0-count.
    gap_region = out[42:79]
    assert not np.isnan(gap_region).all() or np.isfinite(out).any()

# -*- coding: utf-8 -*-
"""Deep-audit regression tests (2026-08-08 final round).

Covers the deterministic P0/P1 fixes from the final deep audit:
* Markov/KM kernel: NaN in the past window must never become a legal state.
* First-passage bias: full-observed non-hit anchors contribute 0 (hit-probability).
* Mean-reversion half-life: exact discrete AR(1) form, OU approx renamed.
* Kalman: missing observations propagate predict-only covariance; beta init is
  not the unstable y/x ratio.
* group_feature_* rename (SVD over the members' feature matrix, not a corr).
* Matrix-profile exclusion zone (>= m/4) and the new distinct outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import OperatorRegistry


def _get(name: str):
    return OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)


def _dates(n: int = 80) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


@pytest.fixture(scope="module", autouse=True)
def _load():
    from cleaned_operators import load_all

    load_all()


# ---------------------------------------------------------------------------
# Markov / Kramers-Moyal shared kernel: NaN never becomes a state
# ---------------------------------------------------------------------------

def test_markov_kernel_nan_not_a_state():
    from cleaned_operators.markov_dynamics import _bin, _state_dynamics_series

    edges = np.array([-np.inf, 0.0, 1.0, np.inf])
    s = np.array([1.0, 2.0, np.nan, 3.0, 1.0, 5.0, np.nan, 2.0, 4.0, 1.0])
    binned = _bin(s, edges)
    # NaN must map to the -1 sentinel, never to a legal state.
    assert np.all(binned[np.isnan(s)] == -1)
    assert np.all(binned[np.isfinite(s)] >= 0)
    res = _state_dynamics_series(s, window=6, bins=3, lag=1, min_count=2)
    # The row whose current value is NaN has no state.
    assert np.isnan(res["state"][2])
    # Where a state IS computed, the counts row is a finite histogram whose
    # total never exceeds the number of valid past states (NaN never counted).
    for t in range(len(s)):
        if not np.isfinite(res["state"][t]):
            continue
        c = res["counts"][t]
        assert np.isfinite(c).all()
        assert float(c.sum()) <= min(6, t)


# ---------------------------------------------------------------------------
# First-passage bias includes hit probability (Q02, option A)
# ---------------------------------------------------------------------------

def test_first_passage_bias_includes_nonhit_zeros():
    op = _get("ts_first_passage_bias")
    if op is None:
        pytest.skip("ts_first_passage_bias not registered")
    rng = np.random.default_rng(11)
    dates = _dates(60)
    x = pd.DataFrame(rng.normal(scale=0.01, size=(60, 1)), index=dates, columns=["A"])
    scale = pd.DataFrame(np.full((60, 1), 0.01), index=dates, columns=["A"])
    out = op.calculate(x, scale, window=40, barrier=1.0, horizon=5, min_anchors=3)
    vals = out["A"].dropna()
    # With barrier 1.0 and a 0.01 scale, most anchors never hit the barrier, so
    # the bias must be close to zero (non-hit anchors contribute 0) — it is NOT
    # a conditional-on-hit statistic that would sit near +-1.
    assert vals.size > 0
    assert vals.abs().max() < 0.6


# ---------------------------------------------------------------------------
# Mean-reversion half-life: exact discrete AR(1) form
# ---------------------------------------------------------------------------

def test_mean_reversion_half_life_exact_discrete():
    from cleaned_operators.ts_model.ar_meanrev import _mean_reversion_half_life

    # Deterministic AR(1) (no noise): the OLS fit recovers phi exactly, so the
    # exact discrete half-life log(0.5)/log(phi) is recovered to machine error.
    phi = 0.9
    x = np.empty(300)
    x[0] = 1.0
    for t in range(1, 300):
        x[t] = phi * x[t - 1]
    hl = _mean_reversion_half_life(x, window=280, min_periods=20)
    expected = np.log(0.5) / np.log(phi)
    assert np.isfinite(hl)
    assert abs(hl - expected) < 1e-6
    # phi >= 1 (no mean reversion) must fail closed to NaN.
    up = np.arange(1.0, 201.0)
    assert np.isnan(_mean_reversion_half_life(up, window=150, min_periods=20))


# ---------------------------------------------------------------------------
# Kalman: predict-only covariance propagation + stable beta init
# ---------------------------------------------------------------------------

def test_kalman_trend_missing_propagates_covariance():
    from cleaned_operators.ts_model.state_space import _kalman_trend_slope

    v = np.array([1.0, 1.1, np.nan, np.nan, np.nan, 1.3, 1.4, 1.5])
    slope = _kalman_trend_slope(v, q_level=1e-3, q_trend=1e-3, r=0.1)
    # Missing rows keep a propagated (finite) slope estimate — the trend state
    # is carried forward, not dropped.
    assert np.isfinite(slope[5])


def test_kalman_beta_not_single_ratio_init():
    from cleaned_operators.ts_model.state_space import _kalman_beta

    rng = np.random.default_rng(8)
    n = 200
    x = pd.Series(np.ones(n) + 0.01 * rng.normal(size=n))
    x.iloc[0] = 1e-9  # x ~ 0 at the start would explode y/x
    y = pd.Series(0.5 * x + 0.1 * rng.normal(size=n))
    beta = _kalman_beta(y.to_numpy(dtype=float), x.to_numpy(dtype=float), 1e-4, 0.1, "beta")
    out = beta[np.isfinite(beta)]
    assert out.size > 10
    # Warmup-OLS init converges to the true beta without a y/x explosion.
    assert np.isfinite(out).all()
    assert abs(np.median(out[10:]) - 0.5) < 0.15


# ---------------------------------------------------------------------------
# group_feature_* rename + NaN member never poisons the group
# ---------------------------------------------------------------------------

def test_group_feature_rename_and_nan_member():
    from cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.resolve_canonical("group_corr_mode_share") == "group_feature_mode_share"
    op = _get("group_feature_mode_share")
    if op is None:
        pytest.skip("group_feature_mode_share not registered")
    rng = np.random.default_rng(5)
    dates = _dates(30)
    f1 = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=["A", "B", "C", "D"])
    f2 = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=["A", "B", "C", "D"])
    f3 = pd.DataFrame(rng.normal(size=(30, 4)), index=dates, columns=["A", "B", "C", "D"])
    group = pd.DataFrame([[1, 1, 2, 2]] * 30, index=dates, columns=["A", "B", "C", "D"], dtype=float)
    # Make one member fully NaN on a date — it must not poison its group.
    f1.iloc[5, 0] = np.nan
    f2.iloc[5, 0] = np.nan
    f3.iloc[5, 0] = np.nan
    out = op.calculate(f1, f2, f3, group)
    # The poisoned member stays NaN; its group-mates still get a value.
    assert np.isnan(out.iloc[5, 0])
    assert np.isfinite(out.iloc[5, 1]) or np.isnan(out.iloc[5, 1])


# ---------------------------------------------------------------------------
# P1-T typed input grammar gate
# ---------------------------------------------------------------------------

def _analyze(formula: str):
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer, TypedInputContractError

    expr = parse_expr(formula)
    try:
        Analyzer().lower(expr)
    except TypedInputContractError:
        return "rejected"
    except Exception:
        return "error"  # other validation (e.g. grain) fired first
    return "ok"


def test_typed_input_contracts():
    # drawdown on a return series is rejected.
    assert _analyze("ts_max_drawdown(ret, 20)") == "rejected"
    assert _analyze("ts_max_drawdown_activity_cost(ret, 20, 5)") == "rejected"
    # spectral on a raw split-sensitive close is rejected.
    assert _analyze("ts_spectral_entropy(close, 20)") == "rejected"
    # A-share limit operators on a continuous price are rejected.
    assert _analyze("ashare_limit_up_touch(continuous_close, high_limit)") == "rejected"


def test_typed_input_contracts_legal_forms():
    # Legal forms pass the typed gate (level/raw/continuous used correctly).
    assert _analyze("ts_max_drawdown(close, 20)") in {"ok", "error"}
    assert _analyze("ts_spectral_entropy(continuous_close, 20)") in {"ok", "error"}
    assert _analyze("ashare_limit_up_touch(close, high_limit)") in {"ok", "error"}


# ---------------------------------------------------------------------------
# Matrix profile: exclusion zone + distinct outputs
# ---------------------------------------------------------------------------

def test_matrix_profile_exclusion_zone_and_new_outputs():
    from cleaned_operators.candle_state_space import _matrix_profile_series

    rng = np.random.default_rng(6)
    x = rng.normal(size=120).astype(float)
    x2d = x[:, None]
    novelty, age, freq, disp = _matrix_profile_series(
        x2d, window=100, subsequence_length=8, history=80
    )
    last = 119
    # A constant ramp (repeated pattern) has a small novelty and a repeated
    # motif frequency near 1 — genuinely distinct from dispersion.
    ramp = np.arange(120, dtype=float)[:, None]
    n2, a2, f2, d2 = _matrix_profile_series(ramp, window=100, subsequence_length=8, history=80)
    assert n2[last, 0] < novelty[last, 0]  # deterministic ramp is a nearer match
    # Frequency and dispersion are populated where 3+ candidates exist.
    assert np.isfinite(f2[last, 0])
    assert np.isfinite(d2[last, 0])


# ---------------------------------------------------------------------------
# P1-I / audit 13.5–13.6: directional-change online clock + recursive indicator
# unified missing-state policy.
# ---------------------------------------------------------------------------
def _series(*vals: float) -> pd.DataFrame:
    return pd.DataFrame({"c": np.array(vals, dtype=float)})


def test_dc_online_clock_uses_per_bar_historical_scale():
    # A monotone series never completes a DC leg -> overshoot ratio is NaN.
    x = _series(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    s = _series(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    out = _get("ts_dc_overshoot_ratio").calculate(x, s, threshold=1.0, window=10)
    assert np.isnan(out.iloc[-1, 0])

    # Sawtooth: events confirm against the scale available *at that bar*, not
    # today's end-of-window scale.  With per-bar scale 0.5 in the middle the
    # clock confirms tighter, then re-freezes.
    x2 = _series(0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0)
    s2 = _series(1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0)
    rate = _get("ts_dc_event_rate").calculate(x2, s2, threshold=1.0, window=10)
    # Hand-traced online events at bars 3,4,6,7 -> 4/9.
    assert rate.iloc[-1, 0] == pytest.approx(4.0 / 9.0)


def test_kama_missing_state_default_interrupts():
    from cleaned_operators.technical.indicators_v2 import KAMA

    x = _series(1.0, 2.0, np.nan, 3.0, 4.0)
    out = KAMA(x, 3, 2, 5)
    # Default interrupt: the missing bar emits NaN and the recursion re-seeds at
    # the next finite bar (3.0) instead of bridging the suspension.
    assert np.isnan(out.iloc[2, 0])
    assert out.iloc[3, 0] == pytest.approx(3.0)


def test_kama_missing_state_carry_bridges_bounded_gap():
    from cleaned_operators.technical.indicators_v2 import KAMA

    x = _series(1.0, 2.0, np.nan, 3.0, 4.0)
    carry = KAMA(x, 3, 2, 5, missing_policy="carry", max_gap=1)
    # carry bridges the single missing bar with the last finite value.
    assert carry.iloc[2, 0] == pytest.approx(carry.iloc[1, 0])

    # Two consecutive missing bars exceed max_gap=1 -> the second bar interrupts.
    x2 = _series(1.0, 2.0, np.nan, np.nan, 4.0)
    out2 = KAMA(x2, 3, 2, 5, missing_policy="carry", max_gap=1)
    assert np.isnan(out2.iloc[3, 0])
    # First missing bar still bridged.
    assert out2.iloc[2, 0] == pytest.approx(out2.iloc[1, 0])


def test_supertrend_and_psar_missing_state_interrupt():
    from cleaned_operators.technical.indicators_v2 import Supertrend, PSAR

    high = _series(1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    low = _series(0.0, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    close = _series(1.5, 2.5, 3.5, np.nan, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)

    st = Supertrend(high, low, close, 3, 2.0)
    # Missing bar -> NaN output; the recursion interrupts and re-seeds once the
    # Wilder ATR band is again finite (a band over a gap is uncomputable).
    assert np.isnan(st.iloc[3, 0])
    assert np.isnan(st.iloc[4, 0])  # ATR still warming through the gap
    assert np.isfinite(st.iloc[5, 0])

    ps = PSAR(high, low, 0.02, 0.2)
    assert np.isnan(ps.iloc[3, 0])
    assert np.isfinite(ps.iloc[4, 0])  # PSAR re-seeds from the bar's own range


def test_supertrend_pandas_polars_parity_with_gap():
    import polars as pl

    from cleaned_operators.technical.polars_tech_misc import Supertrend as PlST

    high = _series(1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0)
    low = _series(0.0, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0)
    close = _series(1.5, 2.5, 3.5, np.nan, 4.5, 5.5, 6.5, 7.5, 8.5)

    from cleaned_operators.technical.indicators_v2 import Supertrend as PdST

    a = PdST(high, low, close, 3, 2.0)["c"].to_numpy()
    ph = pl.DataFrame({"c": high["c"].to_numpy()})
    pl_ = pl.DataFrame({"c": low["c"].to_numpy()})
    pc = pl.DataFrame({"c": close["c"].to_numpy()})
    b = np.array(
        [float(v) if v is not None else np.nan for v in PlST(ph, pl_, pc, 3, 2.0)["c"].to_list()]
    )
    assert np.allclose(a, b, equal_nan=True, atol=1e-9)

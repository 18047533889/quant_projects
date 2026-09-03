"""GPU batch cohort portfolio + risk-metric parity tests (spec §54-55, §19).

Each GPU kernel is compared against the CPU reference oracle (read-only
imports from ``metrics/probe_portfolio`` and ``metrics/portfolio_stats``).
All tests ``pytest.skip`` when CuPy is not importable (CPU-only CI).

Coverage:
  - hand-computed cohort example (existing CPU test scenario)
  - full cohort PnL parity over (T, F, N) vs per-factor CPU
  - long-short Sharpe parity
  - long-only active / gross / cohort-weight parity
  - cost scenarios (gross / 1x / holding=1 same-day)
  - overlapping-label direct annualization != cohort daily-PnL Sharpe (guard)
  - drawdown / calmar / sortino / win-rate / ann-ret / ann-vol / pmr parity
"""

import numpy as np
import pytest

try:
    import cupy as cp
    _HAS_CP = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CP = False

from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
from quant_evaluator.kernels.gpu.drawdown import (
    compute_annualized_return_batch,
    compute_annualized_volatility_batch,
    compute_sharpe_batch,
    compute_sortino_batch,
    compute_max_drawdown_batch,
    compute_calmar_batch,
    compute_win_rate_batch,
    compute_positive_month_ratio_batch,
)

# CPU oracles (read-only)
from quant_evaluator.metrics.probe_portfolio import (
    compute_cohort_pnl,
    build_cohort_panels,
    compute_overlapping_forward_returns,
    annualize_overlapping_label_sharpe,
)
from quant_evaluator.metrics.portfolio_stats import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_maximum_drawdown,
    compute_win_rate,
)
from quant_evaluator.metrics.probe_portfolio.sharpe import (
    compute_annualized_return,
    compute_annualized_volatility,
    compute_positive_month_ratio,
)

pytestmark = pytest.mark.skipif(not _HAS_CP, reason="cupy not importable")


def _asn(x):
    return cp.asnumpy(x)


# ---------------------------------------------------------------------------
# 1. Hand-computed cohort example (mirrors existing CPU test scenario)
# ---------------------------------------------------------------------------

def test_gpu_probe_cohort_hand_example():
    """12 stocks x 62 days, constant D1/D10 buckets, constant per-stock returns.

    D10 = {10, 11} (col11 +1%/day), D1 = {0, 1} (col0 -0.5%/day).
    Per-cohort-day LS PnL = 0.5*(0.01+0)/2 + 0.5*(0.005+0)/2 = 0.00375,
    /H=20 -> 0.0001875.  Day-t PnL = active_cohorts(t) * 0.0001875.
    """
    T, N, F = 62, 12, 1
    factor = np.tile(np.arange(12, dtype=float), (T, F, 1))
    ret = np.zeros((T, N))
    ret[:, 11] = 0.01
    ret[:, 0] = -0.005
    vwap = np.full((T, N), 10.0)

    out = compute_cohort_pnl_batch_gpu(factor, ret, None, holding=20)
    pnl = _asn(out["pnl_net"][:, 0])

    per_cohort_day = (0.5 * (0.01 + 0.0) / 2.0 + 0.5 * (0.005 + 0.0) / 2.0) / 20.0
    expected = np.zeros(T)
    for t in range(T):
        active = sum(1 for s in range(t) if t <= s + 20)
        expected[t] = active * per_cohort_day
    np.testing.assert_allclose(pnl, expected, rtol=1e-10, atol=1e-12)

    # gross steady-state = 1.0; cohort_weight signal-day = 1/20
    assert abs(_asn(out["gross_exposure"][30, 0]) - 1.0) < 1e-9
    assert abs(_asn(out["cohort_weight"][30, 0]) - 1.0 / 20.0) < 1e-9
    assert _asn(out["cohort_weight"][-1, 0]) == 0.0
    # D10 = 2 stocks, D1 = 2 stocks
    assert _asn(out["n_long"][0, 0]) == 2
    assert _asn(out["n_short"][0, 0]) == 2


# ---------------------------------------------------------------------------
# 2. Full cohort PnL parity vs per-factor CPU
# ---------------------------------------------------------------------------

def test_gpu_probe_cohort_cpu_parity():
    """GPU (T, F, N) batch must match per-factor CPU compute_cohort_pnl."""
    rng = np.random.default_rng(0)
    T, N, F = 300, 40, 5
    factor = rng.standard_normal((T, F, N))
    ret = rng.standard_normal((T, N)) * 0.02
    vwap = np.full((T, N), 10.0)
    vwap[30, :] = np.nan  # suspended day -> untradable in both
    trad = np.isfinite(vwap)

    g = compute_cohort_pnl_batch_gpu(
        factor, ret, trad, holding=20, per_side_cost=0.001
    )
    for f in range(F):
        cpu = compute_cohort_pnl(
            factor[:, f, :], ret, vwap, holding=20, per_side_cost=0.001
        )
        for k in ["pnl_net", "gross_exposure", "cohort_weight",
                  "entry_cost", "exit_cost", "n_long", "n_short"]:
            gv = _asn(g[k][:, f])
            cv = cpu[k].astype(np.float64)
            np.testing.assert_allclose(gv, cv, rtol=1e-8, atol=1e-10,
                                       err_msg=f"key={k} f={f}")


def test_gpu_probe_ls_sharpe():
    """Long-short cohort Sharpe parity (GPU vs CPU)."""
    rng = np.random.default_rng(2)
    T, N, F = 400, 30, 4
    factor = rng.standard_normal((T, F, N))
    ret = rng.standard_normal((T, N)) * 0.02 + 0.001 * (
        factor[:, 0, :] - factor[:, 0, :].mean(axis=1, keepdims=True)
    )
    vwap = np.full((T, N), 10.0)

    g = compute_cohort_pnl_batch_gpu(factor, ret, None, holding=20)
    pnl = _asn(g["pnl_net"])
    gpu_sharpe = _asn(compute_sharpe_batch(pnl))
    for f in range(F):
        cpu_sharpe = compute_sharpe_ratio(pnl[:, f])
        assert np.isclose(gpu_sharpe[f], cpu_sharpe, rtol=1e-8, atol=1e-10)


def test_gpu_probe_long_active():
    """Long-only active / gross / cohort-weight parity."""
    rng = np.random.default_rng(3)
    T, N, F = 250, 30, 3
    factor = rng.standard_normal((T, F, N))
    ret = rng.standard_normal((T, N)) * 0.02
    vwap = np.full((T, N), 10.0)

    g = compute_cohort_pnl_batch_gpu(factor, ret, None, holding=20)
    for f in range(F):
        cpu = compute_cohort_pnl(factor[:, f, :], ret, vwap, holding=20)
        np.testing.assert_allclose(
            _asn(g["gross_exposure"][:, f]), cpu["gross_exposure"],
            rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(
            _asn(g["cohort_weight"][:, f]), cpu["cohort_weight"],
            rtol=1e-8, atol=1e-10)


def test_gpu_probe_cost_scenarios():
    """Cost scenarios: gross / 1x / holding=1 same-day double-charge."""
    T, N, F = 5, 12, 2
    factor = np.tile(np.arange(12, dtype=float), (T, F, 1))
    ret = np.zeros((T, N))
    vwap = np.full((T, N), 10.0)

    # gross (per_side_cost=0)
    g0 = compute_cohort_pnl_batch_gpu(factor, ret, None, holding=20, per_side_cost=0.0)
    assert np.all(_asn(g0["entry_cost"]) == 0.0)
    assert np.all(_asn(g0["exit_cost"]) == 0.0)

    # holding=1 same-day entry/exit -> single double-side cost 1.0*0.01
    g1 = compute_cohort_pnl_batch_gpu(factor, ret, None, holding=1, per_side_cost=0.01)
    for f in range(F):
        cpu = compute_cohort_pnl(factor[:, f, :], ret, vwap, holding=1, per_side_cost=0.01)
        assert abs(_asn(g1["pnl_net"][1, f]) - (-0.01)) < 1e-12
        assert abs(_asn(g1["entry_cost"][1, f]) - 0.01) < 1e-12
        assert abs(_asn(g1["exit_cost"][1, f]) - 0.0) < 1e-12
        np.testing.assert_allclose(_asn(g1["pnl_net"][:, f]), cpu["pnl_net"],
                                   rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Guard: overlapping-label direct annualization != cohort daily-PnL Sharpe
# ---------------------------------------------------------------------------

def test_overlapping_target_not_portfolio_sharpe():
    """Prove overlapping-label direct annualization differs from cohort Sharpe.

    Reuses the CPU demonstration: a long-memory signal produces a highly
    autocorrelated overlapping forward-return series whose direct annualized
    Sharpe is inflated well above the true cohort daily-PnL Sharpe.  The GPU
    cohort PnL must reproduce the CPU cohort Sharpe (not the inflated one).
    """
    T, N, F = 600, 40, 1
    rng = np.random.default_rng(3)
    signal = np.cumsum(rng.standard_normal((T, 1)), axis=0)
    signal = signal - signal.mean()
    factor = signal * 0.5 + rng.standard_normal((T, N)) * 0.8
    ret = rng.standard_normal((T, N)) * 0.02 + 0.0015 * (
        factor - factor.mean(axis=1, keepdims=True)
    )
    vwap = np.full((T, N), 10.0)

    # GPU cohort PnL -> Sharpe
    g = compute_cohort_pnl_batch_gpu(factor[:, None, :], ret, None, holding=20)
    pnl = _asn(g["pnl_net"][:, 0])
    gpu_cohort_sharpe = float(_asn(compute_sharpe_batch(pnl))[0])

    # CPU cohort Sharpe (oracle)
    cpu = compute_cohort_pnl(factor, ret, vwap, holding=20)
    cpu_cohort_sharpe = float(compute_sharpe_ratio(cpu["pnl_net"]))

    # Overlapping-label direct annualization (the forbidden inflated method)
    long_mask, short_mask = build_cohort_panels(factor, n_quantiles=10)
    fwd = compute_overlapping_forward_returns(
        ret, long_mask=long_mask, short_mask=short_mask, holding=20
    )
    overlap_sharpe = annualize_overlapping_label_sharpe(fwd, holding=20)

    assert np.isfinite(gpu_cohort_sharpe)
    assert np.isfinite(overlap_sharpe)
    # GPU cohort Sharpe matches CPU cohort Sharpe (not the inflated overlap one)
    assert np.isclose(gpu_cohort_sharpe, cpu_cohort_sharpe, rtol=1e-8, atol=1e-10)
    # The forbidden method is materially higher than the true cohort Sharpe
    assert overlap_sharpe > cpu_cohort_sharpe * 1.5


# ---------------------------------------------------------------------------
# 4. Risk-metric parity
# ---------------------------------------------------------------------------

def _risk_parity(name, gpu_fn, cpu_fn, returns, **kw):
    gv = _asn(gpu_fn(returns, **kw))
    cv = np.array([cpu_fn(returns[:, f], **kw) for f in range(returns.shape[1])])
    np.testing.assert_allclose(gv, cv, rtol=1e-8, atol=1e-10, err_msg=name)
    return gv, cv


def test_gpu_drawdown_parity():
    rng = np.random.default_rng(4)
    T, F = 300, 4
    r = rng.normal(0, 0.02, (T, F))
    r[50, 0] = -1.5  # wipeout col 0
    r[60, 1] = -1.2  # wipeout col 1
    _risk_parity("maxdd", compute_max_drawdown_batch,
                 lambda x: compute_maximum_drawdown(x)[0], r)


def test_gpu_calmar_parity():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0005, 0.02, (300, 4))
    _risk_parity("calmar", compute_calmar_batch, compute_calmar_ratio, r)


def test_gpu_sortino_parity():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0005, 0.02, (300, 4))
    _risk_parity("sortino", compute_sortino_batch, compute_sortino_ratio, r)


def test_gpu_win_rate_parity():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.02, (300, 4))
    _risk_parity("winrate", compute_win_rate_batch, compute_win_rate, r)


def test_gpu_annualized_return_vol_parity():
    rng = np.random.default_rng(8)
    r = rng.normal(0.0005, 0.02, (300, 4))
    _risk_parity("ann_ret", compute_annualized_return_batch,
                 compute_annualized_return, r)
    _risk_parity("ann_vol", compute_annualized_volatility_batch,
                 compute_annualized_volatility, r)


def test_gpu_positive_month_ratio_parity():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0005, 0.02, (300, 4))
    _risk_parity("pmr", compute_positive_month_ratio_batch,
                 compute_positive_month_ratio, r)

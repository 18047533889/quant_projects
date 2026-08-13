# -*- coding: utf-8 -*-
"""Round-11 long-tail findings #50-#55 and #180-#182: fundamental flow semantics.

Covers:
- #50  Piotroski missing components never auto-score as a failed signal
       (partial score + observed count / normalized partial score).
- #51  FinancialFlowSemantics typed contract (``flow_type:*`` tags) on
       growth / persistence / volatility / accrual / financing-flow / cash-gap.
- #52  YTD cumulative must not be treated as a per-period growth rate:
       ``fin_growth``/``fin_yoy`` reject a declared ``CumulativeYTDFlow`` grain
       and the correct sequence is ``fin_quarter_from_cumulative`` first.
- #53  Quarter accrual minus YTD depreciation (incompatible flow grains) is
       rejected at the call boundary.
- #54  Fundamental strength normalises components (cross-sectional rank) before
       combining — component units/scale cannot decide the weight.
- #55  Altman / Zmijewski declare an ``ApplicableUniverse`` (non-financial) and
       fail closed when the universe mask marks names inapplicable.
- #180 report_yoy_lag matches the same fiscal slot of the prior fiscal year
       (quarterly / semiannual / annual), not a fixed periods_per_year lag.
- #181 report helper units (dimensionless / same_as:x).
- #182 relation distribution mobility units (rank steps / same_as:share).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.fundamental.accruals_scores import (
    ApplicableUniverse,
    _piotroski_observed_count,
    _piotroski_normalized_partial_score,
)

ensure_cleaned_loaded()


def _pid(periods, idx):
    return pd.DataFrame(np.asarray(periods, dtype=object)[:, None], index=idx, columns=["S0"])


# ---------------------------------------------------------------------------
# #50 Piotroski: a missing component must not be scored as a failed signal
# ---------------------------------------------------------------------------

def _piotroski_panels(n=10, *, ocf_nan_last_period=False):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    periods = np.array(["2024Q1"] * 3 + ["2024Q2"] * 2 + ["2025Q1"] * 3 + ["2025Q2"] * 2, dtype=object)
    pid = _pid(periods, idx)
    roa = pd.DataFrame(np.array([0.05] * 5 + [0.06] * 3 + [0.07] * 2)[:, None], index=idx, columns=["S0"])
    ocf = roa * 0.5
    if ocf_nan_last_period:
        ocf = ocf.copy()
        ocf.iloc[-2:, 0] = np.nan  # the whole last fiscal period is missing
    npf = roa * 0.4
    lev = pd.DataFrame(np.zeros((n, 1)), index=idx, columns=["S0"])
    one = pd.DataFrame(np.ones((n, 1)), index=idx, columns=["S0"])
    return roa, ocf, npf, lev, one, one, one, one, pid


def test_piotroski_missing_component_never_scores_as_failure():
    roa, ocf, npf, lev, cr, tc, gm, at, pid = _piotroski_panels(ocf_nan_last_period=True)
    score = OperatorRegistry.get("piotroski_f_score").calculate(roa, ocf, npf, lev, cr, tc, gm, at, pid)
    partial = OperatorRegistry.get("piotroski_partial_score").calculate(roa, ocf, npf, lev, cr, tc, gm, at, pid)
    observed = _piotroski_observed_count(roa, ocf, npf, lev, cr, tc, gm, at, pid)
    normalized = _piotroski_normalized_partial_score(roa, ocf, npf, lev, cr, tc, gm, at, pid)

    # The missing OCF period drops the observed count: ocf_pos + accrual are
    # unobserved, the other 7 signals remain observable.
    assert observed.iloc[-1, 0] == 7
    # A genuine full F-score requires ALL 9 signals observed — the incomplete
    # cell is NaN, never under-reported as a smaller passed count (#50, R11).
    assert np.isnan(score.iloc[-1, 0])
    # The partial score is the passed/observed fraction in [0, 1] — never
    # inflated to a full 9, never treated as a failure.
    assert 0.0 <= partial.iloc[-1, 0] <= 1.0
    # Normalised partial score is in [0, 1].
    assert 0.0 <= normalized.iloc[-1, 0] <= 1.0
    # A fully-observed panel scores all 9 observed and a finite full F-score.
    roa2, ocf2, npf2, lev2, cr2, tc2, gm2, at2, pid2 = _piotroski_panels(ocf_nan_last_period=False)
    observed_full = _piotroski_observed_count(roa2, ocf2, npf2, lev2, cr2, tc2, gm2, at2, pid2)
    assert observed_full.iloc[-1, 0] == 9
    score_full = OperatorRegistry.get("piotroski_f_score").calculate(
        roa2, ocf2, npf2, lev2, cr2, tc2, gm2, at2, pid2
    )
    assert np.isfinite(score_full.iloc[-1, 0])


# ---------------------------------------------------------------------------
# #51 / #52 / #53 FinancialFlowSemantics enforcement
# ---------------------------------------------------------------------------

def test_growth_ops_declare_single_period_flow_contract():
    for name in ("fin_growth", "fin_yoy", "fin_pct_change", "fin_cagr",
                 "fin_earnings_persistence", "fin_cashflow_persistence",
                 "fin_earnings_cash_gap_volatility", "fin_roe_cash_gap"):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, name
        assert "flow_type:SinglePeriodFlow" in set(op.metadata.tags), name


def test_ytd_cumulative_is_not_direct_period_growth():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    periods = np.array(["2024Q1"] * 3 + ["2024Q2"] * 3, dtype=object)
    pid = _pid(periods, idx)
    ytd = pd.DataFrame(np.array([100.0] * 3 + [250.0] * 3)[:, None], index=idx, columns=["S0"])
    q = pd.DataFrame(np.array([1.0] * 3 + [2.0] * 3)[:, None], index=idx, columns=["S0"])

    # Direct growth on YTD cumulative is NOT quarterly growth: Q2 YTD / Q1 YTD
    # = 150% while the true Q2 single-period growth is +50%.  A caller that
    # declares the YTD grain is rejected (finding #52).
    with pytest.raises(ValueError, match="CumulativeYTDFlow"):
        OperatorRegistry.get("fin_growth").calculate(ytd, pid, flow_type="CumulativeYTDFlow")
    with pytest.raises(ValueError, match="CumulativeYTDFlow"):
        OperatorRegistry.get("fin_yoy").calculate(ytd, pid, flow_type="CumulativeYTDFlow")

    # Correct sequence: QuarterFromCumulative first, then growth.
    from cleaned_operators.fundamental.flow_semantics_v2 import fin_quarter_from_cumulative
    quarterly = fin_quarter_from_cumulative(ytd, pid, q)
    growth = OperatorRegistry.get("fin_growth").calculate(quarterly, pid, periods=1)
    # Q2 single-period revenue = 250 - 100 = 150; growth = 150/100 - 1 = +50%.
    assert growth.iloc[-1, 0] == pytest.approx(0.5, abs=1e-9)


def test_incompatible_flow_grain_subtraction_rejected():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    a = pd.DataFrame({"S0": [1.0] * 4}, index=idx)
    b = pd.DataFrame({"S0": [2.0] * 4}, index=idx)
    aa = pd.DataFrame({"S0": [10.0] * 4}, index=idx)
    pid = _pid(["2024Q1"] * 2 + ["2024Q2"] * 2, idx)

    # Two-flow accrual / cash-gap subtraction must share one reporting grain.
    with pytest.raises(ValueError, match="mixing incompatible flow grains"):
        OperatorRegistry.get("fin_accrual_ratio").calculate(
            a, b, aa, flow_type=("SinglePeriodFlow", "CumulativeYTDFlow")
        )
    with pytest.raises(ValueError, match="mixing incompatible flow grains"):
        OperatorRegistry.get("fin_roe_cash_gap").calculate(
            a, b, aa, pid, flow_type=("SinglePeriodFlow", "CumulativeYTDFlow")
        )
    # Quarter accrual minus YTD depreciation (finding #53).
    with pytest.raises(ValueError, match="mixing incompatible flow grains"):
        OperatorRegistry.get("fin_total_operating_accruals").calculate(
            a, a, a, a, a, b, aa, pid, flow_type=("SinglePeriodFlow", "CumulativeYTDFlow")
        )
    # Same-grain declarations are accepted.
    out = OperatorRegistry.get("fin_accrual_ratio").calculate(
        a, b, aa, flow_type="SinglePeriodFlow"
    )
    assert out.shape == a.shape


# ---------------------------------------------------------------------------
# #54 Fundamental strength: normalize (rank) before combining
# ---------------------------------------------------------------------------

def test_fundamental_strength_score_normalizes_before_combining():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")

    def P(a_vals, b_vals):
        return pd.DataFrame({"A": a_vals, "B": b_vals}, index=idx)

    pid = P(["2024Q1", "2024Q1", "2024Q2", "2024Q2", "2024Q3"],
            ["2024Q1", "2024Q1", "2024Q2", "2024Q2", "2024Q3"])
    roa = P([0.05, 0.06, 0.07, 0.08, 0.09], [0.03, 0.04, 0.05, 0.06, 0.07])
    ocf = P([0.01, 0.02, 0.03, 0.04, 0.05], [0.005, 0.01, 0.015, 0.02, 0.025])
    gm = P([0.2, 0.25, 0.3, 0.35, 0.4], [0.15, 0.2, 0.25, 0.3, 0.35])
    at = P([0.5, 0.6, 0.7, 0.8, 0.9], [0.4, 0.5, 0.6, 0.7, 0.8])
    lev = P([0.4, 0.35, 0.3, 0.25, 0.2], [0.5, 0.45, 0.4, 0.35, 0.3])
    rt = P([1.0, 1.5, 2.0, 2.5, 3.0], [0.8, 1.2, 1.6, 2.0, 2.4])
    it = P([2.0, 2.5, 3.0, 3.5, 4.0], [1.5, 2.0, 2.5, 3.0, 3.5])
    rg = P([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.1, 0.15, 0.2, 0.25])
    # R11: cash_yield := OCF / AverageAssets — avg_assets is the 8th input.
    aa = P([10.0] * 5, [10.0] * 5)

    base = OperatorRegistry.get("fin_fundamental_strength_score").calculate(
        roa, ocf, gm, at, lev, rt, it, aa, rg, pid
    )
    # Cross-sectional rank is invariant to a monotone rescale of one component:
    # multiplying revenue growth by 1000 must NOT change the score (units/scale
    # cannot decide the weight — finding #54).
    rg_scaled = P([100.0, 200.0, 300.0, 400.0, 500.0], [50.0, 100.0, 150.0, 200.0, 250.0])
    scaled = OperatorRegistry.get("fin_fundamental_strength_score").calculate(
        roa, ocf, gm, at, lev, rt, it, aa, rg_scaled, pid
    )
    pd.testing.assert_frame_equal(base, scaled)
    # Each rank-normalised component lies in [-1, 1] and the score is the MEAN
    # over the observed components (R11), so it is bounded by [-1, 1].
    assert np.nanmax(np.abs(base.to_numpy(dtype=float))) <= 1.0


# ---------------------------------------------------------------------------
# #55 Altman / Zmijewski ApplicableUniverse
# ---------------------------------------------------------------------------

def test_altman_zmijewski_declare_applicable_universe():
    for name in ("altman_z_score", "zmijewski_score"):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, name
        assert f"applicable_universe:{ApplicableUniverse}" in set(op.metadata.tags), name

    # All-inapplicable universe (financials) fails closed to NaN via the mask.
    idx = pd.date_range("2024-01-01", periods=4, freq="B")

    def P(v):
        return pd.DataFrame({"S0": [float(v)] * 4}, index=idx)

    pid = _pid(["2024Q1"] * 2 + ["2024Q2"] * 2, idx)
    mask = pd.DataFrame({"S0": [0.0] * 4}, index=idx)  # 0 applicability = outside
    out = OperatorRegistry.get("altman_z_score").calculate(
        P(1), P(2), P(3), P(10), P(5), P(5), P(8), pid, mask
    )
    assert np.isnan(out["S0"].to_numpy(dtype=float)).all()
    outz = OperatorRegistry.get("zmijewski_score").calculate(P(1), P(10), P(5), P(4), P(3), pid, mask)
    assert np.isnan(outz["S0"].to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# #180 report_yoy_lag: same fiscal slot of the prior fiscal year
# ---------------------------------------------------------------------------

def test_report_yoy_lag_matches_prior_year_same_fiscal_slot():
    from cleaned_operators.alpha_language_events import _report_yoy_lag

    # Semiannual: with the default (misleading) periods_per_year=4 hint, the
    # same-slot match must still return 2023H2 for 2024H2.
    idx = pd.bdate_range("2024-01-01", periods=8)
    pid = _pid(["2023H1"] * 2 + ["2023H2"] * 2 + ["2024H1"] * 2 + ["2024H2"] * 2, idx)
    x = pd.DataFrame(np.array([10.0] * 2 + [20.0] * 2 + [15.0] * 2 + [30.0] * 2)[:, None],
                     index=idx, columns=["S0"])
    out = _report_yoy_lag(x, pid, periods_per_year=4)
    assert out.iloc[4, 0] == 10.0   # 2024H1 -> 2023H1
    assert out.iloc[-1, 0] == 20.0  # 2024H2 -> 2023H2

    # Annual: 2024A -> 2023A even with the default periods_per_year=4 hint.
    idx2 = pd.bdate_range("2024-01-01", periods=8)
    pid2 = _pid(["2022A"] * 2 + ["2023A"] * 2 + ["2024A"] * 4, idx2)
    x2 = pd.DataFrame(np.array([5.0] * 2 + [7.0] * 2 + [9.0] * 4)[:, None], index=idx2, columns=["S0"])
    out2 = _report_yoy_lag(x2, pid2, periods_per_year=4)
    assert out2.iloc[-1, 0] == 7.0  # 2024A -> 2023A

    # Quarterly: 2024Q3 -> 2023Q3, and no lookahead to 2024Q2.
    idx3 = pd.bdate_range("2024-01-01", periods=12)
    pid3 = _pid(["2023Q3"] * 3 + ["2023Q4"] * 3 + ["2024Q1"] * 3 + ["2024Q2"] * 3, idx3)
    x3 = pd.DataFrame(np.array([1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3)[:, None],
                      index=idx3, columns=["S0"])
    out3 = _report_yoy_lag(x3, pid3, periods_per_year=4)
    # 2024Q2 -> 2023Q2 is not visible, so NaN (fail closed), never 2023Q3.
    assert np.isnan(out3.iloc[-1, 0])


# ---------------------------------------------------------------------------
# #181 report helper units
# ---------------------------------------------------------------------------

def test_report_helper_units():
    expected = {
        "report_rolling_mean": "unit:same_as:x",
        "report_yoy_lag": "unit:same_as:x",
        "report_change_breadth": "unit:dimensionless",
        "report_change_coherence": "unit:dimensionless",
    }
    for name, unit in expected.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, name
        assert unit in set(op.metadata.tags), f"{name}: missing {unit}"


# ---------------------------------------------------------------------------
# #182 relation distribution mobility units
# ---------------------------------------------------------------------------

def test_relation_mobility_units():
    rank_op = OperatorRegistry.get("relation_rank_mobility", "pandas_numpy")
    assert "unit:rank" in set(rank_op.metadata.tags)
    assert rank_op.metadata.output_unit == "rank"
    share_op = OperatorRegistry.get("relation_share_mobility", "pandas_numpy")
    assert "unit:same_as:share" in set(share_op.metadata.tags)
    assert share_op.metadata.output_unit == "same_as:share"

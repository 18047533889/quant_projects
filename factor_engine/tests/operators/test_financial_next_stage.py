# -*- coding: utf-8 -*-
"""Tests for the 2026-08 next-stage fundamental / shareholder operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

FIN_CANONICALS = (
    "fin_working_capital_accruals fin_total_operating_accruals fin_delta_noa fin_roe_cash_gap "
    "fin_earnings_cash_gap_volatility fin_earnings_smoothness fin_earnings_persistence "
    "fin_cashflow_persistence fin_margin_persistence fin_core_earnings_ratio "
    "fin_noncore_income_ratio fin_fair_value_income_dependence fin_investment_income_dependence "
    "fin_other_earnings_dependence fin_comprehensive_income_gap fin_oci_to_equity "
    "fin_discontinued_operation_ratio fin_minority_profit_share fin_receivable_sales_divergence "
    "fin_inventory_sales_divergence fin_cash_sales_divergence fin_expense_sales_divergence "
    "fin_contract_asset_intensity fin_contract_asset_growth fin_contract_liability_intensity "
    "fin_contract_liability_growth fin_contract_asset_liability_gap fin_lease_intensity "
    "fin_lease_asset_liability_gap fin_goodwill_intensity fin_goodwill_risk_score "
    "fin_deferred_tax_gap fin_impairment_intensity fin_net_debt_issuance fin_borrowing_intensity "
    "fin_debt_repayment_intensity fin_net_borrowing_cashflow fin_equity_capital_growth "
    "fin_financing_gap fin_interest_coverage_proxy fin_debt_service_coverage_proxy "
    "fin_cash_burn_runway fin_capex_intensity fin_capex_growth fin_acquisition_cash_intensity "
    "fin_rd_total_intensity fin_rd_capitalization_ratio piotroski_f_score "
    "piotroski_partial_score piotroski_observed_count altman_z_score "
    "zmijewski_score fin_fundamental_strength_score fin_fundamental_strength_coverage"
).split()

HOLDER_CANONICALS = (
    "holder_weighted_churn holder_entry_share holder_exit_share holder_net_entry_share "
    "holder_rank_stability holder_pledge_ratio holder_freeze_ratio holder_pledge_concentration "
    "holder_freeze_concentration holder_pledged_holder_count holder_pledge_change holder_pledge_churn "
    "holder_float_concentration_gap holder_locked_share_ratio holder_concentration_slope "
    "holder_concentration_acceleration holder_class_entropy holder_nature_entropy "
    "holder_common_holding_peer_return holder_peer_return_breadth "
    "holder_shareholder_network_centrality holder_shareholder_overlap_ratio"
).split()

VALUATION_CANONICALS = (
    "valuation_pe_ttm_lyr_gap valuation_pcf_definition_gap free_float_ratio "
    "free_to_circulating_ratio a_share_cap_ratio free_float_turnover "
    "valuation_cashflow_disagreement valuation_growth_mismatch valuation_quality_mismatch "
    "market_cap_free_cap_gap capital_change_age capital_change_magnitude circulating_cap_unlock_proxy "
    "index_weight_gap_to_free_float index_reconstitution_churn multi_index_entry_intensity "
    "listing_age suspension_frequency index_event_decay"
).split()


def _period_frame(n: int = 120, seed: int = 0, cols: int = 2) -> tuple[pd.DataFrame, ...]:
    dates = pd.date_range("2023-04-01", periods=n, freq="D")
    periods = np.repeat(pd.to_datetime(["2022-12-31", "2023-03-31", "2023-06-30", "2023-09-30"]), n // 4)
    period_id = pd.DataFrame({"A": periods, "B": periods}, index=dates)
    rng = np.random.default_rng(seed)

    def P(lo, hi):
        return pd.DataFrame(rng.uniform(lo, hi, (n, cols)), index=dates, columns=["A", "B"])

    ta, ca, cl = P(300, 350), P(120, 160), P(50, 60)
    cash, std, tp = P(30, 40), P(10, 20), P(1, 5)
    aa, dep = P(200, 250), P(1, 3)
    tl, ltd = P(100, 150), P(20, 30)
    npf, ocf, equity = P(5, 15), P(3, 20), P(80, 120)
    return period_id, ta, ca, cl, cash, std, tp, aa, dep, tl, ltd, npf, ocf, equity


@pytest.mark.parametrize("name", sorted(set(FIN_CANONICALS + HOLDER_CANONICALS + VALUATION_CANONICALS)))
def test_registered(name: str) -> None:
    assert OperatorRegistry.get(name) is not None, name


def test_accrual_operators_shape() -> None:
    period_id, ta, ca, cl, cash, std, tp, aa, dep, tl, ltd, npf, ocf, equity = _period_frame(seed=1)
    # Altman / Zmijewski require an applicability mask (applicable_universe=
    # non_financial) — supply an all-applicable one for the shape smoke test.
    mask = pd.DataFrame(np.ones((120, 2)), index=period_id.index, columns=["A", "B"])
    calls = {
        "fin_working_capital_accruals": [ca, cash, cl, std, tp, aa, period_id],
        "fin_total_operating_accruals": [ta, cash, cl, std, tp, dep, aa, period_id],
        "fin_delta_noa": [ta, cash, tl, std, ltd, aa, period_id],
        "fin_roe_cash_gap": [npf, ocf, equity, period_id],
        "fin_earnings_cash_gap_volatility": [npf, ocf, aa, period_id],
        "fin_earnings_smoothness": [npf, ocf, period_id],
        "fin_earnings_persistence": [npf, period_id],
        "fin_core_earnings_ratio": [ocf, std, tp, cash, cl, ta, period_id],
        "fin_noncore_income_ratio": [std, tp, cash, cl, ca, cl, npf, period_id],
        "piotroski_f_score": [npf, ocf, npf, tl, cl, ta, ca, std, period_id],
        "piotroski_partial_score": [npf, ocf, npf, tl, cl, ta, ca, std, period_id],
        "piotroski_observed_count": [npf, ocf, npf, tl, cl, ta, ca, std, period_id],
        "altman_z_score": [ca, npf, ocf, ta, equity, tl, cl, period_id, mask],
        "zmijewski_score": [npf, ta, tl, ca, cl, period_id, mask],
    }
    for name, args in calls.items():
        out = OperatorRegistry.get(name).calculate(*args)
        assert out.shape == (120, 2), name
        assert np.isfinite(out.to_numpy()).any(), name


def test_piotroski_f_score_bounded() -> None:
    period_id, *_ = _period_frame(seed=2)
    rng = np.random.default_rng(2)
    dates = period_id.index
    comps = []
    for _ in range(8):
        comps.append(pd.DataFrame(rng.uniform(-1, 1, (120, 2)), index=dates, columns=["A", "B"]))
    out = OperatorRegistry.get("piotroski_f_score").calculate(*comps, period_id)
    valid = out.dropna().to_numpy()
    assert (valid >= 0).all() and (valid <= 9).all()


def test_shareholder_churn() -> None:
    # ID-matched ShareholderId form: 10 current ratios, 10 current ids,
    # 10 previous ratios, 10 previous ids.
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    cols = ["A", "B"]
    nan = pd.DataFrame(np.nan, index=dates, columns=cols)
    blank = pd.DataFrame(np.nan, index=dates, columns=cols, dtype=object)

    def _ratio(vals):
        f = pd.DataFrame(np.nan, index=dates, columns=cols)
        f["A"], f["B"] = vals[0], 0.2
        return f

    def _id(ids):
        f = pd.DataFrame(np.nan, index=dates, columns=cols, dtype=object)
        f["A"], f["B"] = ids[0], "SHARED"
        return f

    cur_r = [_ratio(v) for v in [(0.3,), (0.2,), (0.1,), (0.05,), (0.0,)]] + [nan] * 5
    cur_id = [_id(i) for i in [("S1",), ("S2",), ("S3",), ("S4",), ("NONE",)]] + [blank] * 5
    prev_r = [_ratio(v) for v in [(0.25,), (0.2,), (0.15,), (0.05,), (0.1,)]] + [nan] * 5
    prev_id = [_id(i) for i in [("S1",), ("S2",), ("S3",), ("S4",), ("E5",)]] + [blank] * 5
    args = cur_r + cur_id + prev_r + prev_id
    churn = OperatorRegistry.get("holder_weighted_churn").calculate(*args)
    assert churn["A"].iloc[-1] == pytest.approx(0.5 * 0.2, abs=1e-9)  # 0.5*Σ|ratio_cur - ratio_prev| = 0.5*0.2
    entry = OperatorRegistry.get("holder_entry_share").calculate(*args)
    assert entry["A"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    exit_ = OperatorRegistry.get("holder_exit_share").calculate(*args)
    assert exit_["A"].iloc[-1] == pytest.approx(0.1, abs=1e-9)  # E5 在上期不在本期 → 0.1
    net = OperatorRegistry.get("holder_net_entry_share").calculate(*args)
    assert net["A"].iloc[-1] == pytest.approx(-0.1, abs=1e-9)


def test_valuation_gap_operators() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.uniform(5, 30, (60, 2)), index=dates, columns=["A", "B"])
    b = pd.DataFrame(rng.uniform(5, 30, (60, 2)), index=dates, columns=["A", "B"])
    for name in ("valuation_pe_ttm_lyr_gap", "valuation_pcf_definition_gap"):
        out = OperatorRegistry.get(name).calculate(a, b)
        assert out.shape == (60, 2), name


def test_valuation_quality_mismatch_residual() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.standard_normal((60, 8)), index=dates, columns=[f"C{i}" for i in range(8)])
    y = 2.0 * x + 0.5
    w = pd.DataFrame(np.ones((60, 8)), index=dates, columns=[f"C{i}" for i in range(8)])
    out = OperatorRegistry.get("valuation_quality_mismatch").calculate(y, x, w)
    assert out.iloc[-1].abs().max() < 1e-6


def test_listing_age() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    ld = pd.DataFrame({"A": pd.Timestamp("2024-01-05"), "B": pd.Timestamp("2024-01-10")}, index=dates)
    out = OperatorRegistry.get("listing_age").calculate(ld)
    assert out["A"].iloc[-1] == 55
    assert out["B"].iloc[-1] == 50


def test_suspension_frequency() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    sus = pd.DataFrame({"A": np.zeros(60), "B": np.ones(60)}, index=dates)
    out = OperatorRegistry.get("suspension_frequency").calculate(sus, window=20)
    assert out["A"].iloc[-1] == 0.0
    assert out["B"].iloc[-1] == 1.0

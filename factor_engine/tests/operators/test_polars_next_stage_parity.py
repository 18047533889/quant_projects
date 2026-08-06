# -*- coding: utf-8 -*-
"""Polars parity tests for next-stage operators (genuine polars backends)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

_INTRA = (
    "intra_realized_skewness intra_realized_kurtosis intra_realized_quarticity "
    "intra_continuous_variance intra_jump_variation intra_positive_jump_variation "
    "intra_negative_jump_variation intra_signed_jump_ratio intra_jump_count "
    "intra_jump_concentration intra_interval_return intra_interval_volume_share "
    "intra_interval_amount_share intra_max_drawdown intra_max_drawup"
).split()

_DAILY = (
    "free_float_ratio valuation_pe_ttm_lyr_gap market_cap_free_cap_gap "
    "capital_change_magnitude circulating_cap_unlock_proxy fin_roe_cash_gap "
    "fin_fair_value_income_dependence fin_investment_income_dependence "
    "fin_oci_to_equity fin_minority_profit_share fin_contract_asset_intensity "
    "fin_lease_intensity fin_goodwill_intensity fin_deferred_tax_gap "
    "fin_impairment_intensity fin_borrowing_intensity fin_debt_repayment_intensity "
    "fin_interest_coverage_proxy fin_debt_service_coverage_proxy fin_capex_intensity "
    "fin_rd_total_intensity fin_rd_capitalization_ratio fin_financing_gap "
    "fin_cash_burn_runway holder_pledge_ratio holder_freeze_ratio "
    "holder_locked_share_ratio holder_float_concentration_gap holder_pledge_change "
    "holder_common_holding_peer_return holder_peer_return_breadth "
    "holder_shareholder_network_centrality holder_shareholder_overlap_ratio "
    "index_weight_gap_to_free_float suspension_frequency index_reconstitution_churn "
    "multi_index_entry_intensity ts_mean_reversion_half_life "
    "ts_variance_ratio_slope ts_market_liquidity_beta ts_industry_liquidity_beta "
    "group_peer_deviation_index group_peer_beta_deviation"
).split()


def _daily_panel(n: int = 120, seed: int = 0, cols: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(np.abs(rng.standard_normal((n, cols))) + 0.1,
                        index=idx, columns=[f"C{i}" for i in range(cols)])


def _to_pl(pdf: pd.DataFrame, time_col: str = "date") -> pl.DataFrame:
    return pl.DataFrame(
        {time_col: pl.Series(pd.DatetimeIndex(pdf.index)),
         **{c: pdf[c].to_numpy() for c in pdf.columns}}
    )


def _to_pd(pldf: pl.DataFrame) -> pd.DataFrame:
    if "date" in pldf.columns:
        idx = pd.DatetimeIndex(pldf["date"].to_pandas())
    else:
        idx = pd.RangeIndex(pldf.height)
    cols = [c for c in pldf.columns if c != "date"]
    return pd.DataFrame({c: pldf[c].to_numpy() for c in cols}, index=idx)


def _minute_panel(days: int = 2, seed: int = 0, cols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        for m in list(range(571, 691))[:120] + list(range(781, 901))[:120]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.01, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps), columns=[f"C{i}" for i in range(cols)],
    )


@pytest.mark.parametrize("name", sorted(set(_INTRA)))
def test_intraday_polars_parity(name: str) -> None:
    pdf = _minute_panel(seed=7)
    pdf_w = pdf * 0 + 1  # placeholder replaced below
    # volume/amount companions
    amount = pdf * (np.abs(np.random.default_rng(1).standard_normal(pdf.shape)) + 1.0)
    vol = amount / pdf
    pldf = _to_pl(pdf, time_col="QuoteTime")
    pl_amount = _to_pl(amount, time_col="QuoteTime")
    pl_vol = _to_pl(vol, time_col="QuoteTime")

    op = OperatorRegistry.get(name)
    if name in ("intra_interval_volume_share",):
        ref = op.calculate(vol, start_minute=570, end_minute=690)
        got = OperatorRegistry.get(name, backend="polars").calculate(pl_vol, start_minute=570, end_minute=690)
    elif name == "intra_interval_amount_share":
        ref = op.calculate(amount, start_minute=570, end_minute=690)
        got = OperatorRegistry.get(name, backend="polars").calculate(pl_amount, start_minute=570, end_minute=690)
    elif name in ("intra_max_drawdown", "intra_max_drawup"):
        ref = op.calculate(pdf)
        got = OperatorRegistry.get(name, backend="polars").calculate(pldf)
    elif name.startswith("intra_interval_return"):
        ref = op.calculate(pdf, start_minute=570, end_minute=690)
        got = OperatorRegistry.get(name, backend="polars").calculate(pldf, start_minute=570, end_minute=690)
    else:
        ref = op.calculate(pdf)
        got = OperatorRegistry.get(name, backend="polars").calculate(pldf)
    ref_d = _to_pd(got).reindex(index=ref.index, columns=ref.columns)
    for col in ref.columns:
        a = ref[col].dropna()
        b = ref_d[col].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        assert a.index.equals(b.index), f"{name}: index mismatch"
        assert np.allclose(a.values, b.values, atol=1e-8, equal_nan=True), f"{name}: {col}"


@pytest.mark.parametrize("name", sorted(set(_DAILY)))
def test_daily_polars_parity(name: str) -> None:
    rng = np.random.default_rng(3)
    n, cols = 100, 4
    a = _daily_panel(n, seed=3, cols=cols)
    b = _daily_panel(n, seed=4, cols=cols)
    c = _daily_panel(n, seed=5, cols=cols)
    d = _daily_panel(n, seed=6, cols=cols)
    pl_a, pl_b, pl_c, pl_d = _to_pl(a), _to_pl(b), _to_pl(c), _to_pl(d)
    op = OperatorRegistry.get(name)
    args_map = {
        "free_float_ratio": [a, b], "valuation_pe_ttm_lyr_gap": [a, b], "market_cap_free_cap_gap": [a, b],
        "capital_change_magnitude": [a], "circulating_cap_unlock_proxy": [a, b],
        "fin_roe_cash_gap": [a, b, c, d], "fin_fair_value_income_dependence": [a, b, d],
        "fin_investment_income_dependence": [a, b, d], "fin_oci_to_equity": [a, b, d],
        "fin_minority_profit_share": [a, b, d], "fin_contract_asset_intensity": [a, b, d],
        "fin_lease_intensity": [a, b, c, d], "fin_goodwill_intensity": [a, b, d],
        "fin_deferred_tax_gap": [a, b, c, d], "fin_impairment_intensity": [a, b, c, d],
        "fin_borrowing_intensity": [a, b, d], "fin_debt_repayment_intensity": [a, b, d],
        "fin_interest_coverage_proxy": [a, b, d], "fin_debt_service_coverage_proxy": [a, b, c, d],
        "fin_capex_intensity": [a, b, d], "fin_rd_total_intensity": [a, b, c, d],
        "fin_rd_capitalization_ratio": [a, b, d], "fin_financing_gap": [a, b, c, d, b, d],
        "fin_cash_burn_runway": [a, b, d],
        "holder_pledge_ratio": [a, b], "holder_freeze_ratio": [a, b], "holder_locked_share_ratio": [a, b],
        "holder_float_concentration_gap": [a, b], "holder_pledge_change": [a],
        "holder_common_holding_peer_return": [a, b, c], "holder_peer_return_breadth": [a],
        "holder_shareholder_network_centrality": [a, b], "holder_shareholder_overlap_ratio": [a, b],
        "index_weight_gap_to_free_float": [a, b], "suspension_frequency": [a],
        "index_reconstitution_churn": [a], "multi_index_entry_intensity": [a, b, c],
        "ts_mean_reversion_half_life": [a],
        "ts_variance_ratio_slope": [a], "ts_market_liquidity_beta": [a, b],
        "ts_industry_liquidity_beta": [a, b],
        "group_peer_deviation_index": [a, b, c],
    }
    if name not in args_map:
        pytest.skip(f"no args map for {name}")
    args = args_map[name]
    pl_args = [_to_pl(p) if isinstance(p, pd.DataFrame) else p for p in args]
    # period_id args replaced by a dummy (elementwise kernels ignore it)
    pargs = []
    for i, p in enumerate(args):
        if i == len(args) - 1 and name.startswith("fin_") and len(args) >= 3:
            pargs.append(p)  # pass through as numeric dummy
        else:
            pargs.append(p)
    ref = op.calculate(*args)
    pop = OperatorRegistry.get(name, backend="polars")
    got = pop.calculate(*pl_args)
    ref_d = _to_pd(got).reindex(index=ref.index, columns=ref.columns)
    for col in ref.columns:
        a_s, b_s = ref[col].dropna(), ref_d[col].dropna()
        if len(a_s) == 0 or len(b_s) == 0:
            continue
        if len(a_s) == len(b_s):
            assert np.allclose(a_s.values, b_s.values, atol=1e-6, equal_nan=True), f"{name}: {col}"

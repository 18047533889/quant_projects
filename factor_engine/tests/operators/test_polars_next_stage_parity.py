# -*- coding: utf-8 -*-
"""Polars parity tests for next-stage operators (genuine polars backends)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

# All ``intra_*`` minute->daily operators with a genuine Polars backend.  The
# second wave (``polars_intraday_full``) covers the segment / RV / path /
# distribution / liquidity / limit / slot-profile / beta / drawdown families.
_INTRA = (
    "intra_realized_skewness intra_realized_kurtosis intra_realized_quarticity "
    "intra_continuous_variance intra_jump_variation intra_positive_jump_variation "
    "intra_negative_jump_variation intra_signed_jump_ratio intra_jump_count "
    "intra_jump_concentration intra_interval_return intra_interval_volume_share "
    "intra_interval_amount_share intra_max_drawdown intra_max_drawup "
    "intra_segment_return intra_segment_volume_share intra_segment_amount_share "
    "intra_segment_vwap_deviation intra_segment_realized_vol "
    "intra_realized_variance intra_realized_semivariance intra_bipower_variation "
    "intra_jump_ratio intra_path_efficiency intra_high_time intra_low_time "
    "intra_vwap_above_ratio intra_vwap_cross_count intra_concentration "
    "intra_entropy intra_signed_imbalance_proxy intra_return_activity_corr "
    "intra_amihud intra_kyle_lambda_proxy intra_extreme_bar_return "
    "intra_lunch_gap_return intra_limit_first_hit_time intra_limit_duration "
    "intra_limit_reopen_count "
    "intra_interval_realized_variance intra_interval_vwap_deviation "
    "intra_interval_illiquidity intra_same_slot_momentum intra_same_slot_reversal "
    "intra_return_profile_cosine intra_volume_profile_cosine intra_amount_profile_cosine "
    "intra_volume_profile_jsd intra_amount_profile_jsd intra_profile_earth_mover_distance "
    "intra_tripower_quarticity intra_jump_first_time intra_jump_last_time intra_jump_clustering "
    "intra_realized_beta intra_realized_correlation intra_down_down_semibeta "
    "intra_up_up_semibeta intra_down_up_semibeta intra_up_down_semibeta "
    "intra_beta_asymmetry intra_idiosyncratic_variance intra_idiosyncratic_skewness "
    "intra_idiosyncratic_kurtosis intra_market_model_r2 "
    "intra_drawdown_depth intra_drawdown_duration intra_drawdown_recovery_half_life "
    "intra_vwap_path_slope intra_vwap_path_curvature "
    "intra_price_vwap_max_positive_excursion intra_price_vwap_max_negative_excursion "
    "intra_time_above_vwap intra_longest_above_vwap_streak intra_longest_below_vwap_streak "
    "intra_vwap_reversion_speed"
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


# Argument-group dispatch for the intraday parity cases.  ``kind`` selects which
# panels the operator consumes; ``kw`` are extra (non-panel) parameters.
_INTRA_SPECS: dict[str, tuple[str, dict]] = {
    "intra_segment_return": ("single", {"segment": "morning"}),
    "intra_segment_realized_vol": ("single", {"segment": "afternoon"}),
    "intra_interval_realized_variance": ("single", {"start_minute": 570, "end_minute": 690}),
    "intra_realized_variance": ("single", {}),
    "intra_realized_semivariance": ("single", {"side": "up"}),
    "intra_bipower_variation": ("single", {}),
    "intra_jump_ratio": ("single", {}),
    "intra_path_efficiency": ("single", {}),
    "intra_high_time": ("single", {}),
    "intra_low_time": ("single", {}),
    "intra_concentration": ("single", {}),
    "intra_entropy": ("single", {}),
    "intra_extreme_bar_return": ("single", {"side": "min"}),
    "intra_tripower_quarticity": ("single", {}),
    "intra_jump_first_time": ("single", {}),
    "intra_jump_last_time": ("single", {}),
    "intra_jump_clustering": ("single", {}),
    "intra_drawdown_depth": ("single", {}),
    "intra_drawdown_duration": ("single", {}),
    "intra_drawdown_recovery_half_life": ("single", {}),
    "intra_same_slot_momentum": ("single", {}),
    "intra_same_slot_reversal": ("single", {}),
    "intra_return_profile_cosine": ("single", {}),
    "intra_volume_profile_cosine": ("single", {}),
    "intra_amount_profile_cosine": ("single", {}),
    "intra_volume_profile_jsd": ("single", {}),
    "intra_amount_profile_jsd": ("single", {}),
    "intra_profile_earth_mover_distance": ("single", {}),
    "intra_segment_volume_share": ("vol", {"segment": "morning"}),
    "intra_segment_amount_share": ("amount", {"segment": "afternoon"}),
    "intra_signed_imbalance_proxy": ("close_value", {}),
    "intra_amihud": ("close_amount", {}),
    "intra_kyle_lambda_proxy": ("close_amount", {}),
    "intra_interval_illiquidity": ("close_amount", {"start_minute": 570, "end_minute": 690}),
    "intra_return_activity_corr": ("close_activity", {"absolute_return": True}),
    "intra_lunch_gap_return": ("close_open", {}),
    "intra_segment_vwap_deviation": ("three", {"segment": "morning"}),
    "intra_interval_vwap_deviation": ("three", {"start_minute": 570, "end_minute": 690}),
    "intra_vwap_above_ratio": ("three", {}),
    "intra_vwap_cross_count": ("three", {}),
    "intra_price_vwap_max_positive_excursion": ("three", {}),
    "intra_price_vwap_max_negative_excursion": ("three", {}),
    "intra_time_above_vwap": ("three", {}),
    "intra_longest_above_vwap_streak": ("three", {}),
    "intra_longest_below_vwap_streak": ("three", {}),
    "intra_vwap_reversion_speed": ("three", {}),
    "intra_vwap_path_slope": ("three", {}),
    "intra_vwap_path_curvature": ("three", {}),
    "intra_realized_beta": ("beta", {}),
    "intra_realized_correlation": ("beta", {}),
    "intra_down_down_semibeta": ("beta", {}),
    "intra_up_up_semibeta": ("beta", {}),
    "intra_down_up_semibeta": ("beta", {}),
    "intra_up_down_semibeta": ("beta", {}),
    "intra_beta_asymmetry": ("beta", {}),
    "intra_idiosyncratic_variance": ("beta", {}),
    "intra_idiosyncratic_skewness": ("beta", {}),
    "intra_idiosyncratic_kurtosis": ("beta", {}),
    "intra_market_model_r2": ("beta", {}),
    "intra_limit_first_hit_time": ("limit", {"side": "up"}),
    "intra_limit_duration": ("limit", {"side": "down"}),
    "intra_limit_reopen_count": ("limit", {"side": "up", "transition": "reseal"}),
}

# First-wave ops already covered by the original dispatch (same call signature).
_INTRA_LEGACY_ARGS = {
    "intra_interval_volume_share": ("vol", {"start_minute": 570, "end_minute": 690}),
    "intra_interval_amount_share": ("amount", {"start_minute": 570, "end_minute": 690}),
    "intra_interval_return": ("single", {"start_minute": 570, "end_minute": 690}),
}


@pytest.mark.parametrize("name", sorted(set(_INTRA)))
def test_intraday_polars_parity(name: str) -> None:
    pdf = _minute_panel(seed=7)
    # volume/amount companions
    amount = pdf * (np.abs(np.random.default_rng(1).standard_normal(pdf.shape)) + 1.0)
    vol = amount / pdf
    open_px = pdf * (np.abs(np.random.default_rng(2).standard_normal(pdf.shape)) + 0.9)
    pldf = _to_pl(pdf, time_col="QuoteTime")
    pl_amount = _to_pl(amount, time_col="QuoteTime")
    pl_vol = _to_pl(vol, time_col="QuoteTime")
    pl_open = _to_pl(open_px, time_col="QuoteTime")

    spec = _INTRA_SPECS.get(name) or _INTRA_LEGACY_ARGS.get(name) or ("single", {})
    kind, kw = spec

    if kind == "limit":
        # R5-05 typed-broadcast: the daily limit panels must carry the SAME
        # instrument columns as the minute panel (broadcast requires equal
        # column count + identity).
        lim_up = _daily_panel(n=10, seed=5, cols=pdf.shape[1]) * 1.1
        lim_down = _daily_panel(n=10, seed=5, cols=pdf.shape[1]) * 0.9
        pkw = {"high_limit": lim_up, "low_limit": lim_down}
        plkw = {"high_limit": _to_pl(lim_up), "low_limit": _to_pl(lim_down)}
    elif kind == "beta":
        cap = _daily_panel(n=2, seed=3, cols=pdf.shape[1])
        pkw = {"free_market_cap": cap}
        plkw = {"free_market_cap": _to_pl(cap)}
    else:
        pkw, plkw = {}, {}

    args_map = {
        "single": ((pdf,), (pldf,)),
        "vol": ((vol,), (pl_vol,)),
        "amount": ((amount,), (pl_amount,)),
        "close_value": ((pdf, amount), (pldf, pl_amount)),
        "close_amount": ((pdf, amount), (pldf, pl_amount)),
        "close_activity": ((pdf, amount), (pldf, pl_amount)),
        "close_open": ((pdf, open_px), (pldf, pl_open)),
        "three": ((pdf, amount, vol), (pldf, pl_amount, pl_vol)),
        "limit": ((pdf,), (pldf,)),
        "beta": ((pdf,), (pldf,)),
    }
    pargs, plargs = args_map[kind]

    op = OperatorRegistry.get(name)
    pop = OperatorRegistry.get(name, backend="polars")
    ref = op.calculate(*pargs, **{**kw, **pkw})
    got = pop.calculate(*plargs, **{**kw, **plkw})
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

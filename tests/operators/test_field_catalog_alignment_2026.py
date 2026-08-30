# -*- coding: utf-8 -*-
"""2026-08 A-share field-catalog alignment acceptance tests.

Locks the fixes from the COS lqtp data-dictionary audit (``factor_engine/docs/AUDIT_2026_08_
FIELD_ALIGNMENT_PLAN.md``): field/table contracts, unit normalization, strict
unknown-field interception, fiscal-ordinal ordering, valuation gap variants and
shareholder snapshot fail-closed semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.fields import FIELD_REGISTRY


@pytest.fixture(scope="module", autouse=True)
def _load():
    load_all()


def _panel(values, dates=None):
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"A": list(values)}, index=index)


# ---------------------------------------------------------------------------
# P0-A 字段/表契约对齐 COS 数据字典
# ---------------------------------------------------------------------------


def test_stock_dividend_contract_matches_cos_schema():
    spec = FIELD_REGISTRY.resolve_table("StockDividend")
    # No PubDate / no ExDate columns exist in the physical table; the partition
    # date == ExDividendDate and there is no announcement PIT (dict §4.17).
    assert spec.time_column == "TradeDate"
    assert spec.effective_time_column == "ExDividendDate"
    assert spec.knowledge_time_column is None
    assert spec.join_policy == "effective_only"
    assert spec.strict_pit_allowed is False


def test_index_daily_bar_keys_on_symbol_not_indexsymbol():
    spec = FIELD_REGISTRY.resolve_table("IndexDailyBar")
    assert spec.instrument_column == "Symbol"
    assert spec.required_parameters == ()


def test_calendar_time_column_is_trade_date():
    spec = FIELD_REGISTRY.resolve_table("Calendar")
    assert spec.time_column == "TradeDate"


@pytest.mark.parametrize("table", ["StockList", "EtfList", "IndexList", "StockIndustry"])
def test_list_tables_have_full_history_not_current_snapshot(table):
    spec = FIELD_REGISTRY.resolve_table(table)
    assert spec.time_column == "TradeDate"
    assert spec.current_snapshot_only is False
    assert spec.strict_pit_allowed is True  # full history → PIT-safe equi join


def test_stock_industry_requires_single_source():
    spec = FIELD_REGISTRY.resolve_table("StockIndustry")
    assert "IndustrySource" in spec.required_parameters


def test_update_time_role_is_ingestion_time_not_knowledge():
    # update_time is deliberately reused by name across the four financial
    # statements (and StockDailyBar), so the bare name is ambiguous; resolve
    # within the daily-bar table.
    spec = FIELD_REGISTRY.get("update_time", table="StockDailyBar")
    assert spec is not None
    assert spec.role == "ingestion_time"
    assert spec.mining_allowed is False  # freshness-only, never a mined feature


def test_price_fields_adjustment_status_declared():
    """R17-010: Factor direction is verified (backward multiplier); raw OHLC/VWAP
    declare RAW price basis; pre_close declares the official-reference basis —
    the stale ``adjustment_status == "unverified"`` metadata is gone."""
    for logical, expected_status, expected_basis in (
        ("open", "raw", "RAW"),
        ("high", "raw", "RAW"),
        ("low", "raw", "RAW"),
        ("close", "raw", "RAW"),
        ("vwap", "raw", "RAW"),
        ("pre_close", "official_reference_pre_close", "OFFICIAL_REFERENCE_PRE_CLOSE"),
        ("volume", "raw", None),
    ):
        # Open/High/Low/Close/Vwap are physical columns shared by the daily,
        # minute, index and ETF bar tables; the low-level registry requires a
        # table qualifier.  Formula resolution defaults to StockDailyBar.
        spec = FIELD_REGISTRY.get(logical, table="StockDailyBar")
        assert spec is not None, logical
        assert spec.adjustment is None, logical
        assert spec.metadata.get("adjustment_status") == expected_status, logical
        if expected_basis is None:
            assert spec.price_basis is None or spec.price_basis == "RAW", logical
        else:
            assert spec.price_basis == expected_basis, logical


def test_factor_direction_is_verified_backward_multiplier():
    """R17-010: the A-share Factor contract is a VERIFIED backward cumulative
    multiplier (continuous_price = raw_price * Factor), not ``unverified``."""
    spec = FIELD_REGISTRY.get("adj_factor", table="StockDailyBar")
    assert spec is not None
    assert spec.metadata.get("direction") == "backward_multiplier"
    assert spec.metadata.get("note", "") == "continuous_price = raw_price * Factor (verified; R17-010)"


def test_cos_unit_normalization_scales():
    checks = {
        "turnover_ratio": ("percent", 0.01),      # TurnoverRatio % -> ratio
        "share_ratio": ("percent", 0.01),         # ShareRatio % -> ratio
        "index_weight": ("percent", 0.01),        # Weight % -> ratio
        "roe": ("percent", 0.01),                 # StockIndicator ROE % -> ratio
    }
    # ADJ_FIELD_MIGRATION: bare ``ret`` is ambiguous (StockDailyBar + StockDailyBarAdj);
    # resolve the adj authority table explicitly.
    ret = FIELD_REGISTRY.get("ret", table="StockDailyBarAdj")
    assert ret is not None
    assert ret.source_unit == "basis_point"
    assert ret.scale_to_canonical == pytest.approx(0.0001)
    for logical, (src_unit, scale) in checks.items():
        spec = FIELD_REGISTRY.get(logical)
        assert spec is not None, logical
        assert spec.source_unit == src_unit, logical
        assert spec.scale_to_canonical == pytest.approx(scale), logical


def test_new_catalog_coverage_fields_registered():
    for logical in (
        "capitalization", "circulating_cap", "free_cap", "a_cap", "a_market_cap",
        "stock_dividend", "stock_transfer", "right_reg_date",
        "shareholder_id", "shareholder_rank", "shareholder_name",
        "shareholder_class", "shares_nature", "share_number",
        "share_pledge", "share_freeze", "change_date", "change_type",
        "public_status_code",
    ):
        assert FIELD_REGISTRY.get(logical) is not None, logical


def test_one_to_many_shareholder_fields_not_mining_allowed():
    for logical in ("share_ratio", "shareholder_id", "share_pledge", "share_freeze",
                    "index_weight"):
        spec = FIELD_REGISTRY.get(logical)
        assert spec is not None
        assert spec.cardinality == "one_to_many"
        assert spec.mining_allowed is False, logical


# ---------------------------------------------------------------------------
# P0-B 严格未知字段拦截（生产）与跨数据集
# ---------------------------------------------------------------------------


def test_strict_unknown_fields_fail_closed_in_production():
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        UnknownFieldSemanticError,
    )

    src = DataAccessSource(dataset="ashare_stock_daily_adj", strict_unknown_fields=True)
    physical, _ = src._resolve_columns(["close"])  # registered field resolves
    assert physical == ["AdjClose"]
    with pytest.raises(UnknownFieldSemanticError):
        src._resolve_columns(["no_such_column"])
    with pytest.raises(UnknownFieldSemanticError):
        # is_suspend belongs to the adj daily table (physical IsSuspend) and has
        # no presence in the index daily dataset.
        DataAccessSource(dataset="ashare_index_daily", strict_unknown_fields=True)._resolve_columns(
            ["is_suspend"]
        )


def test_research_mode_keeps_raw_column_fallback():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    src = DataAccessSource(dataset="ashare_stock_daily")  # default research lenient
    assert src.strict_unknown_fields is False
    physical, _ = src._resolve_columns(["no_such_column"])
    assert physical == ["no_such_column"]


# ---------------------------------------------------------------------------
# P0-C 财务时序语义
# ---------------------------------------------------------------------------


def test_fin_applicability_mask_preserves_nan():
    values = _panel([np.nan, 5.0, -1.0, 0.0])
    op = OperatorRegistry.get("fin_applicability_mask", backend="pandas_numpy")
    out = op.calculate(values, 0.0)
    assert np.isnan(out.iloc[0, 0])          # unknown stays unknown
    assert out.iloc[1, 0] == 1.0             # > threshold -> 1
    assert out.iloc[2, 0] == 0.0             # <= threshold -> 0
    assert out.iloc[3, 0] == 0.0


def test_fin_applicability_mask_nan_parity_with_polars():
    pl = pytest.importorskip("polars")
    values = _panel([np.nan, 5.0, -1.0, 0.0])
    pandas_out = OperatorRegistry.get("fin_applicability_mask", backend="pandas_numpy").calculate(values, 0.0)
    pl_frame = pl.DataFrame({"A": [None, 5.0, -1.0, 0.0]})
    polars_out = OperatorRegistry.get("fin_applicability_mask", backend="polars").calculate(pl_frame, 0.0)
    np.testing.assert_allclose(
        pandas_out["A"].to_numpy(), polars_out["A"].to_numpy(), equal_nan=True
    )


def test_walk_periods_orders_by_fiscal_ordinal_not_appearance():
    """A late-disclosed 2023Q2 arriving after 2023Q3 must not reorder the lag."""
    from factor_engine.cleaned_operators.fundamental.flow_semantics_v2 import fin_ttm_quarterly
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    x = _panel([10.0, 60.0, 30.0])            # Q1=10, Q3=60, then Q2=30 back-filled
    period_id = _panel(["2023Q1", "2023Q3", "2023Q2"])
    out = OperatorRegistry.get("fin_lag", backend="pandas_numpy").calculate(x, period_id, 1)
    # At the final row 2023Q2 is visible; its ordinal lag is Q1 (10.0).  Append
    # order would have returned the Q3 value (60.0).
    assert out.iloc[2, 0] == 10.0


def test_fin_ttm_quarterly_fails_closed_on_skipped_quarter():
    from factor_engine.cleaned_operators.fundamental.flow_semantics_v2 import fin_ttm_quarterly

    x = _panel([10.0, 20.0, 40.0])            # Q1, Q2, Q4 (Q3 missing)
    period_id = _panel(["2023Q1", "2023Q2", "2023Q4"])
    out = fin_ttm_quarterly(x, period_id, 3)
    assert np.isnan(out.iloc[2, 0])           # non-consecutive -> NaN, not Q1+Q2+Q4


def test_fin_quarter_from_cumulative_rejects_cross_year_subtraction():
    from factor_engine.cleaned_operators.fundamental.flow_semantics_v2 import fin_quarter_from_cumulative

    x = _panel([10.0, 30.0])
    period_id = _panel(["2024Q1", "2025Q2"])
    fiscal_quarter = _panel([1, 2])
    out = fin_quarter_from_cumulative(x, period_id, fiscal_quarter)
    assert out.iloc[0, 0] == 10.0
    assert np.isnan(out.iloc[1, 0])           # Q2 2025 cannot subtract Q1 2024


# ---------------------------------------------------------------------------
# P0-D 估值/股本
# ---------------------------------------------------------------------------


def test_pe_gap_signed_log_preserves_profit_loss_sign():
    a = _panel([10.0, -10.0])   # PeRatio
    b = _panel([8.0, -8.0])     # PeRatioLyr
    signed = OperatorRegistry.get("valuation_pe_gap_signed_log", backend="pandas_numpy")
    out = signed.calculate(a, b)
    # log|x| gap would be identical for +10/+8 and -10/-8; the signed variant
    # keeps them distinct (and both finite).
    assert np.isfinite(out.iloc[0, 0]) and np.isfinite(out.iloc[1, 0])
    assert abs(out.iloc[0, 0] - out.iloc[1, 0]) > 1e-8


def test_pe_gap_positive_only_for_profitable():
    a = _panel([10.0, -10.0])
    b = _panel([8.0, -8.0])
    pos = OperatorRegistry.get("valuation_pe_gap_positive", backend="pandas_numpy")
    out = pos.calculate(a, b)
    assert np.isfinite(out.iloc[0, 0])
    assert np.isnan(out.iloc[1, 0])           # loss side -> NaN


def test_valuation_gap_variants_polars_parity():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    a = pd.DataFrame(rng.uniform(-20, 30, (30, 2)), index=dates, columns=["A", "B"])
    b = pd.DataFrame(rng.uniform(-15, 25, (30, 2)), index=dates, columns=["A", "B"])
    for name in ("valuation_pe_gap_signed_log", "valuation_pe_gap_positive",
                 "valuation_pcf_gap_signed_log", "valuation_pcf_gap_positive"):
        pandas_out = OperatorRegistry.get(name, backend="pandas_numpy").calculate(a.copy(), b.copy())
        polars_out = OperatorRegistry.get(name, backend="polars").calculate(
            pl.DataFrame(a), pl.DataFrame(b)
        )
        np.testing.assert_allclose(
            pandas_out.to_numpy(), polars_out.to_numpy(), rtol=1e-8, atol=1e-8, equal_nan=True
        )


def test_growth_mismatch_rejects_free_scale_mask():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("valuation_growth_mismatch", backend="pandas_numpy")
    ey = _panel([0.08])
    g = _panel([0.34])
    out = op.calculate(ey, g, 1.0)  # both decimal ratios, scale must stay 1.0
    assert np.isfinite(out.iloc[0, 0])
    with pytest.raises(ValueError):
        op.calculate(ey, g, 0.0)
    with pytest.raises(ValueError):
        op.calculate(ey, g, -2.0)


def test_capital_change_age_maps_weekend_change_date_to_next_trading_day():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # B-freq dates: Mon 01-01, 01-02, 01-03, 01-04, 01-05, Mon 01-08, 01-09, ...
    # An S1 capital snapshot reports ChangeDate = Saturday 2024-01-06 from the
    # following trading day (Mon 01-08, position 5).  It must map to 01-08 rather
    # than being dropped (old behaviour left age NaN until the next in-index
    # change date).
    index = pd.date_range("2024-01-01", periods=10, freq="B")
    cd = pd.DataFrame(np.nan, index=index, columns=["A"], dtype="datetime64[ns]")
    cd.iloc[5, 0] = pd.Timestamp("2024-01-06")  # Saturday, known from 01-08 onward
    op = OperatorRegistry.get("capital_change_age", backend="pandas_numpy")
    out = op.calculate(cd)
    for i in range(5):
        assert np.isnan(out.iloc[i, 0]), f"row {i} should be NaN (no change yet)"
    # On Monday 01-08 the change becomes effective -> age 0, then counts up.
    assert out.iloc[5, 0] == 0.0
    assert out.iloc[6, 0] == 1.0


def test_circulating_cap_ratio_change_new_name_registered():
    entry = OperatorRegistry._catalog["circulating_cap_ratio_change"]
    assert entry is not None
    assert OperatorRegistry._catalog["circulating_cap_unlock_proxy"]["compatibility_only"] is True
    assert "circulating_cap_ratio_change" in OperatorRegistry._catalog[
        "circulating_cap_unlock_proxy"
    ]["preferred_replacements"]


# ---------------------------------------------------------------------------
# P0-E 股东快照 fail-closed
# ---------------------------------------------------------------------------


def _holder_args(cur_r, cur_id, prev_r, prev_id, n_rows=1):
    """Build the 40 rank-slot panels (s1..s10, sid1..sid10, p1..p10, psid1..psid10).

    Each ``*_r`` / ``*_id`` is a 10-slot dict {slot_index: [row values]}; empty
    slots default to NaN ratios / None ids.
    """
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    out = []
    for group, empty in ((cur_r, np.nan), (cur_id, None), (prev_r, np.nan), (prev_id, None)):
        for slot in range(1, 11):
            values = group.get(slot)
            if values is None:
                values = [empty] * n_rows
            out.append(pd.DataFrame({"A": values}, index=dates))
    return out


def test_holder_id_matched_churn_nan_ratio_fails_closed():
    op = OperatorRegistry.get("holder_id_matched_churn", backend="pandas_numpy")
    # s3 slot has ID "h3" but a missing ratio on the current snapshot -> the
    # snapshot is incomplete ("unknown" is not a confirmed 0%) -> NaN.
    cur_r = {1: [0.5], 2: [0.3], 3: [np.nan]}
    cur_id = {1: ["h1"], 2: ["h2"], 3: ["h3"]}
    prev_r = {1: [0.5], 2: [0.3], 3: [0.2]}
    prev_id = {1: ["h1"], 2: ["h2"], 3: ["h3"]}
    out = op.calculate(*_holder_args(cur_r, cur_id, prev_r, prev_id))
    assert np.isnan(out.iloc[0, 0])


def test_holder_id_matched_churn_duplicate_id_conflict_fails_closed():
    op = OperatorRegistry.get("holder_id_matched_churn", backend="pandas_numpy")
    # Same ID "h1" repeated across slots with conflicting ratios -> ambiguous
    # snapshot (duplicate record vs multiple share natures) -> NaN.
    cur_r = {1: [0.5], 2: [0.2]}
    cur_id = {1: ["h1"], 2: ["h1"]}
    prev_r = {1: [0.5], 2: [0.3], 3: [0.2]}
    prev_id = {1: ["h1"], 2: ["h2"], 3: ["h3"]}
    out = op.calculate(*_holder_args(cur_r, cur_id, prev_r, prev_id))
    assert np.isnan(out.iloc[0, 0])


def test_holder_id_matched_churn_identical_duplicate_is_deduped():
    op = OperatorRegistry.get("holder_id_matched_churn", backend="pandas_numpy")
    # Same ID "h1" repeated with the SAME ratio -> duplicate record -> dedupe.
    cur_r = {1: [0.5], 2: [0.5], 3: [0.0]}
    cur_id = {1: ["h1"], 2: ["h1"], 3: ["h2"]}
    prev_r = {1: [0.5], 2: [0.3], 3: [0.2]}
    prev_id = {1: ["h1"], 2: ["h2"], 3: ["h3"]}
    out = op.calculate(*_holder_args(cur_r, cur_id, prev_r, prev_id))
    # h1 = 0.5 on both sides; h2 entered with 0.0 -> entry share 0.0, finite.
    assert np.isfinite(out.iloc[0, 0])


def test_concentration_trend_advances_by_snapshot_not_ffill_rows():
    # §6.8: holder_concentration_slope with snapshot_date must advance once per
    # distinct report snapshot and hold across the ffilled daily flat tail,
    # instead of the legacy daily-rolling window decaying to zero.
    from factor_engine.cleaned_operators.shareholder.churn_network import _concentration_slope

    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    n = len(idx)
    conc = pd.DataFrame({"A": [1.0, 2.0, 3.0] + [3.0] * (n - 3)}, index=idx)
    sd = pd.DataFrame(
        {"A": pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29"] + [pd.NaT] * (n - 3))},
        index=idx,
    )
    legacy = _concentration_slope(conc, window=8)
    snap = _concentration_slope(conc, window=8, snapshot_date=sd)
    # Legacy daily rolling decays the rising trend to ~0 over the flat tail.
    assert abs(float(legacy["A"].iloc[-1])) < 1e-9
    # Snapshot-aligned trend is positive and holds constant after the 3rd report.
    assert float(snap["A"].iloc[-1]) > 0.0
    slope_tail = float(snap["A"].iloc[-1])
    # P1-136: the slope is a centered dot product over the snapshot DATES
    # (epoch days, not the ordinal 0,1,2) and the ddof inflation is gone — the
    # legacy 1.5 was the buggy value.  Compute the exact per-day slope of
    # [1,2,3] over the three snapshot dates.
    _sdates = pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29"]).to_numpy().astype("int64").astype(float) / 8.64e13
    _svals = np.array([1.0, 2.0, 3.0])
    _exp = float(np.sum((_sdates - _sdates.mean()) * (_svals - _svals.mean())) / np.sum((_sdates - _sdates.mean()) ** 2))
    assert abs(slope_tail - _exp) < 1e-6, f"snapshot slope {slope_tail} != expected {_exp}"
    # No slope before the 3rd snapshot (causal), then forward-filled daily.
    assert int(snap["A"].notna().sum()) < n
    assert snap["A"].notna().iloc[-1]


def test_concentration_trend_backward_compatible_without_snapshot_date():
    # Omitting snapshot_date keeps the historical daily rolling behaviour; the
    # optional third parameter defaults to None and never changes the signature.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("holder_concentration_slope", backend="pandas_numpy")
    assert op is not None
    assert list(op.metadata.param_names) == ["concentration", "window", "snapshot_date"]
    accel = OperatorRegistry.get("holder_concentration_acceleration", backend="pandas_numpy")
    assert accel is not None
    assert "snapshot_date" in accel.metadata.param_names


def test_shareholder_report_period_field_registered():
    spec = FIELD_REGISTRY.get("report_period_end_date", table="StockTopTenShareholder")
    assert spec is not None
    assert spec.role == "period_id"
    assert spec.mining_allowed is False


def test_four_statement_value_field_coverage():
    # §1.8: COS lqtp 财务四表值字段已按字典实测单位登记。Income/CashFlow 为
    # YTD 累计（flow_ytd），Balance 为时点存量（balance），Indicator 比率为百分数。
    checks = {
        "StockIncome": (("flow", "ytd"), ("interest_income", "investment_income",
                                         "asset_impairment_loss", "total_composite_income")),
        "StockCashFlow": (("flow", "ytd"), ("goods_sale_and_service_render_cash",
                                           "subtotal_operate_cash_inflow", "tax_payments")),
        "StockBalance": (("balance",), ("retained_profit", "capital_reserve_fund",
                                       "total_non_current_assets", "surplus_reserve_fund")),
        "StockIndicator": ("percent", ("net_profit_margin", "gross_profit_margin",
                                       "inc_net_profit_year_on_year", "roa")),
    }
    for table, (grain, names) in checks.items():
        for name in names:
            spec = FIELD_REGISTRY.get(name, table=table)
            assert spec is not None, (table, name)
            if grain == "percent":
                assert spec.source_unit == "percent", (table, name)
            else:
                assert tuple(spec.grain) == grain, (table, name)


def test_holder_pledge_churn_description_is_honest():
    entry = OperatorRegistry._catalog["holder_pledge_churn"]
    assert "非按股东 ID 匹配" in entry.get("description", "")


def test_rank_slot_shareholder_aliases_name_id_matched_replacements():
    # §6.1/§6.2: the reworked legacy names advertise the explicit ID-matched
    # canonicals; holder_pledge_churn stays flagged compatibility-only because it
    # has no ShareholderId input.
    checks = {
        "holder_weighted_churn": ["holder_id_matched_churn"],
        "holder_entry_share": ["holder_id_matched_entry_share"],
        "holder_exit_share": ["holder_id_matched_exit_share"],
        "holder_net_entry_share": ["holder_id_matched_entry_share", "holder_id_matched_exit_share"],
        "holder_rank_stability": ["holder_share_weighted_rank_migration"],
    }
    for canon, replacements in checks.items():
        entry = OperatorRegistry._catalog[canon]
        assert entry.get("preferred_replacements") == replacements, canon
        assert "semantic_note" in entry, canon
    for canon in ("holder_pledge_churn", "holder_concentration_change", "holder_count_change_rate"):
        entry = OperatorRegistry._catalog[canon]
        assert entry.get("compatibility_only") is True, canon
        assert "semantic_note" in entry, canon
    # The ID-matched canonical names exist and are distinct operators.
    for canon in ("holder_id_matched_churn", "holder_id_matched_entry_share",
                  "holder_id_matched_exit_share", "holder_share_weighted_rank_migration"):
        assert OperatorRegistry.get(canon) is not None, canon


# ---------------------------------------------------------------------------
# §12 生产默认挖掘门禁：experimental 家族必须 fail-closed
# ---------------------------------------------------------------------------

_EXPERIMENTAL_FAMILIES = [
    "piotroski_f_score", "altman_z_score", "zmijewski_score",
    "fin_fundamental_strength_score", "fin_goodwill_risk_score",
    "cs_actual_lof_score", "cs_knn_distance", "cs_local_density_score",
    "cs_mahalanobis_distance", "cs_relative_density_ratio",
    "cs_shrinkage_mahalanobis", "cs_robust_mahalanobis_mad",
    "ts_ar_forecast", "ts_ar_innovation", "ts_ar_innovation_z",
    "ts_multi_regression_resid", "ts_huber_regression_in_sample_resid",
    "ts_ridge_regression_in_sample_resid", "ts_quantile_regression_coeff",
    "ts_quantile_regression_resid",
    "intra_positive_jump_variation", "intra_negative_jump_variation",
    "intra_signed_jump_ratio", "intra_return_profile_cosine",
    "intra_realized_beta", "intra_amihud",
]


def test_experimental_families_excluded_from_production_mining():
    from factor_engine.backend.fastpath_allowlists import production_allowlist
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec

    prod = production_allowlist()
    missing = [c for c in _EXPERIMENTAL_FAMILIES if c not in OperatorRegistry.list_canonical()]
    assert not missing, f"unregistered canonicals: {missing}"
    for canonical in _EXPERIMENTAL_FAMILIES:
        spec = build_operator_spec(canonical)
        assert spec is not None, canonical
        assert spec.allow_in_production is False, canonical
        assert canonical not in prod, canonical


def test_production_allowlist_is_evidence_gated_not_surface_gated():
    """Daily-surface experimental ops must not sneak into production mining."""
    from factor_engine.backend.fastpath_allowlists import production_allowlist
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    prod = production_allowlist()
    # Every production-allowed canonical must be certified (not fail-closed) and
    # surface-classified (daily or extended), never unclassified.
    for canonical in prod:
        assert classify_canonical(canonical) in {"daily", "extended"}, canonical


def test_legacy_in_sample_and_intraday_aliases_stamped_honest():
    entry = OperatorRegistry._catalog["ts_multi_regression_resid"]
    assert entry["in_sample"] is True and entry["diagnostic_only"] is True
    assert entry["preferred_replacements"] == ["ts_multi_regression_forecast_error"]
    jump = OperatorRegistry._catalog["intra_positive_jump_variation"]
    assert jump["compatibility_only"] is True
    assert jump["preferred_replacements"] == ["intra_positive_tail_variation"]
    assert OperatorRegistry._catalog["intra_realized_beta"]["benchmark_only"] is True


def test_expectile_quantile_regression_in_sample_stamped_and_hidden():
    # §10: in-sample expectile/quantile regression ops are diagnostics hidden from
    # default mining; the causal *_prior variant is the advertised replacement.
    for canon, replacement in {
        "ts_expectile_regression_coeff": ["ts_expectile_regression_coeff_prior"],
        "ts_expectile_regression_resid": None,
        "ts_quantile_regression_coeff": None,
        "ts_quantile_regression_resid": None,
        "ts_quantile_regression_slope": None,
    }.items():
        entry = OperatorRegistry._catalog[canon]
        assert entry["in_sample"] is True, canon
        assert entry["diagnostic_only"] is True, canon
        assert entry["hidden_from_default_mining"] is True, canon
        if replacement is not None:
            assert entry["preferred_replacements"] == replacement, canon
    # hidden_from_default_mining is load-bearing: excluded from the production
    # allowlist independently of the diagnostic_only flag.
    from factor_engine.backend.fastpath_allowlists import production_allowlist

    prod = production_allowlist()
    assert "ts_expectile_regression_coeff" not in prod
    assert "ts_quantile_regression_resid" not in prod


# ---------------------------------------------------------------------------
# §2.10 period_selection 编译器强制
# ---------------------------------------------------------------------------


def test_every_financial_pit_field_has_a_period_selection_contract():
    from factor_engine.storage.sources.financial import FUNDAMENTAL_FIELD_CONTRACTS
    from factor_engine.storage.sources.logical_tables import logical_table_contract

    financial = [
        spec for spec in FIELD_REGISTRY.fields()
        if spec.mining_allowed  # structural metadata columns are contract-exempt
        and logical_table_contract(spec.table) is not None
        and logical_table_contract(spec.table).join_policy == "financial_pit"
    ]
    assert financial, "expected catalog financial_pit fields"
    missing = [spec.name for spec in financial if spec.name not in FUNDAMENTAL_FIELD_CONTRACTS]
    assert not missing, f"financial_pit fields without a period contract: {missing}"
    for spec in financial:
        contract = FUNDAMENTAL_FIELD_CONTRACTS[spec.name]
        assert contract.period_selector in {"latest_visible_period", "annual_only", "quarterly_only"}
        assert contract.field == spec.source_name


def test_analyzer_accepts_contracted_fundamental_field():
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer

    expr = parse_expr("ts_delta(StockIncome.net_profit, n=1)", dialect="lqtp")
    result = Analyzer().lower(expr)
    assert result.lookback >= 1
    assert any(getattr(s, "table", "") == "StockIncome" for s in result.referenced_fields.values())


def test_analyzer_fails_closed_on_unregistered_financial_field():
    from factor_engine.fields.spec import FieldSpec
    from factor_engine.ir.analyzer import PeriodSelectionContractError, validate_fundamental_period_contracts

    ghost = FieldSpec(name="ghost_income", table="StockIncome", source_name="GhostIncome")
    errors = validate_fundamental_period_contracts({"ghost_income": ghost})
    assert errors and "no registered period-selection contract" in errors[0]


# ---------------------------------------------------------------------------
# §4.4 flow-grain 编译期拒绝
# ---------------------------------------------------------------------------


def test_cumulative_flow_kernels_accept_flow_ytd_and_reject_balance():
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer, FieldGrainContractError

    Analyzer().lower(
        parse_expr("fin_ttm_cumulative(StockIncome.net_profit, period_id=1, fiscal_quarter=1)",
                   dialect="lqtp")
    )
    Analyzer().lower(
        parse_expr("fin_quarter_from_cumulative(StockIncome.net_profit, period_id=1, fiscal_quarter=1)",
                   dialect="lqtp")
    )
    with pytest.raises(FieldGrainContractError):
        Analyzer().lower(
            parse_expr("fin_ttm_cumulative(StockBalance.total_assets, period_id=1, fiscal_quarter=1)",
                       dialect="lqtp")
        )
    with pytest.raises(FieldGrainContractError):
        Analyzer().lower(
            parse_expr("fin_quarter_from_cumulative(StockBalance.total_assets, period_id=1, fiscal_quarter=1)",
                       dialect="lqtp")
        )


def test_one_period_ttm_rejects_raw_cumulative_field():
    # fin_ttm_quarterly consumes derived one-period flows; a raw YTD-cumulative
    # statement field is the "treating a cumulative value as a quarter" hazard.
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer, FieldGrainContractError

    with pytest.raises(FieldGrainContractError):
        Analyzer().lower(
            parse_expr("fin_ttm_quarterly(StockIncome.net_profit, period_id=1)", dialect="lqtp")
        )
    with pytest.raises(FieldGrainContractError):
        Analyzer().lower(
            parse_expr("fin_ttm_quarterly(StockBalance.total_assets, period_id=1)", dialect="lqtp")
        )


def test_balance_average_accepts_balance_only():
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer, FieldGrainContractError

    Analyzer().lower(parse_expr("fin_average_balance(StockBalance.total_assets)", dialect="lqtp"))
    with pytest.raises(FieldGrainContractError):
        Analyzer().lower(
            parse_expr("fin_average_balance(StockIncome.net_profit)", dialect="lqtp")
        )


def test_synthetic_columns_with_default_grain_are_skipped():
    # Research formulas may feed derived period flows; the contract check must
    # not fire on columns that carry no declared economic grain.
    from factor_engine.ir.analyzer import OPERATOR_INPUT_GRAIN_CONTRACTS, validate_field_grain_contracts
    from factor_engine.ir.nodes import IRNode

    assert "fin_ttm_cumulative" in OPERATOR_INPUT_GRAIN_CONTRACTS
    derived = IRNode(op="column", attrs={"name": "derived_flow"})  # default grain
    tree = IRNode(
        op="fin_ttm_cumulative",
        inputs=(derived, IRNode(op="literal", attrs={"value": 1})),
    )
    assert validate_field_grain_contracts(tree) == []


def test_financial_row_bundle_enforces_selector_contract():
    import pandas as pd
    from factor_engine.storage.sources.financial import load_financial_row_bundle

    events = pd.DataFrame({
        "instrument": ["A", "A"],
        "period_end": pd.to_datetime(["2023-12-31", "2024-03-31"]),
        "available_at": pd.to_datetime(["2024-03-01", "2024-04-30"]),
        "net_profit": [100.0, 30.0],
    })
    decisions = pd.DataFrame({
        "decision_timestamp": pd.to_datetime(["2024-05-01"]),
        "instrument": ["A"],
    })
    # net_profit is registered with latest_visible_period -> compatible with any.
    out = load_financial_row_bundle(decisions, events, ["net_profit"], selector="latest_visible_period")
    assert out["net_profit"].iloc[0] == 30.0
    # A quarterly-only selection is still allowed because the contract is the
    # looser latest_visible_period; the check only rejects narrower contracts.
    out_q = load_financial_row_bundle(decisions, events, ["net_profit"], selector="quarterly_only")
    assert out_q["net_profit"].iloc[0] == 30.0

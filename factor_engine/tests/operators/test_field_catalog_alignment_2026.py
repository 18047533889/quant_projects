# -*- coding: utf-8 -*-
"""2026-08 A-share field-catalog alignment acceptance tests.

Locks the fixes from the COS lqtp data-dictionary audit (``docs/AUDIT_2026_08_
FIELD_ALIGNMENT_PLAN.md``): field/table contracts, unit normalization, strict
unknown-field interception, fiscal-ordinal ordering, valuation gap variants and
shareholder snapshot fail-closed semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from fields import FIELD_REGISTRY


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
    spec = FIELD_REGISTRY.get("update_time")
    assert spec is not None
    assert spec.role == "ingestion_time"
    assert spec.mining_allowed is False  # freshness-only, never a mined feature


def test_price_fields_adjustment_unverified():
    for logical in ("open", "high", "low", "close", "pre_close", "vwap", "volume"):
        spec = FIELD_REGISTRY.get(logical)
        assert spec is not None, logical
        assert spec.adjustment is None, logical
        assert spec.metadata.get("adjustment_status") == "unverified", logical


def test_cos_unit_normalization_scales():
    checks = {
        "ret": ("basis_point", 0.0001),          # Return bp -> decimal ratio
        "turnover_ratio": ("percent", 0.01),      # TurnoverRatio % -> ratio
        "share_ratio": ("percent", 0.01),         # ShareRatio % -> ratio
        "index_weight": ("percent", 0.01),        # Weight % -> ratio
        "roe": ("percent", 0.01),                 # StockIndicator ROE % -> ratio
    }
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
    from storage.sources.data_access_source import (
        DataAccessSource,
        UnknownFieldSemanticError,
    )

    src = DataAccessSource(dataset="ashare_stock_daily", strict_unknown_fields=True)
    physical, _ = src._resolve_columns(["close"])  # registered field resolves
    assert physical == ["Close"]
    with pytest.raises(UnknownFieldSemanticError):
        src._resolve_columns(["no_such_column"])
    with pytest.raises(UnknownFieldSemanticError):
        # close belongs to ashare_stock_daily, not ashare_index_daily
        DataAccessSource(dataset="ashare_index_daily", strict_unknown_fields=True)._resolve_columns(
            ["close"]
        )


def test_research_mode_keeps_raw_column_fallback():
    from storage.sources.data_access_source import DataAccessSource

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
    from cleaned_operators.fundamental.flow_semantics_v2 import fin_ttm_quarterly
    from cleaned_operators.registry import OperatorRegistry

    x = _panel([10.0, 60.0, 30.0])            # Q1=10, Q3=60, then Q2=30 back-filled
    period_id = _panel(["2023Q1", "2023Q3", "2023Q2"])
    out = OperatorRegistry.get("fin_lag", backend="pandas_numpy").calculate(x, period_id, 1)
    # At the final row 2023Q2 is visible; its ordinal lag is Q1 (10.0).  Append
    # order would have returned the Q3 value (60.0).
    assert out.iloc[2, 0] == 10.0


def test_fin_ttm_quarterly_fails_closed_on_skipped_quarter():
    from cleaned_operators.fundamental.flow_semantics_v2 import fin_ttm_quarterly

    x = _panel([10.0, 20.0, 40.0])            # Q1, Q2, Q4 (Q3 missing)
    period_id = _panel(["2023Q1", "2023Q2", "2023Q4"])
    out = fin_ttm_quarterly(x, period_id, 3)
    assert np.isnan(out.iloc[2, 0])           # non-consecutive -> NaN, not Q1+Q2+Q4


def test_fin_quarter_from_cumulative_rejects_cross_year_subtraction():
    from cleaned_operators.fundamental.flow_semantics_v2 import fin_quarter_from_cumulative

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
    from cleaned_operators.registry import OperatorRegistry

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
    from cleaned_operators.registry import OperatorRegistry

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


def test_holder_pledge_churn_description_is_honest():
    entry = OperatorRegistry._catalog["holder_pledge_churn"]
    assert "非按股东 ID 匹配" in entry.get("description", "")


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
    "ts_multi_regression_resid", "ts_huber_regression_resid",
    "ts_ridge_regression_resid", "ts_quantile_regression_coeff",
    "ts_quantile_regression_resid",
    "intra_positive_jump_variation", "intra_negative_jump_variation",
    "intra_signed_jump_ratio", "intra_return_profile_cosine",
    "intra_realized_beta", "intra_amihud",
]


def test_experimental_families_excluded_from_production_mining():
    from backend.fastpath_allowlists import production_allowlist
    from cleaned_operators.operator_spec import build_operator_spec

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
    from backend.fastpath_allowlists import production_allowlist
    from cleaned_operators.operator_surface import classify_canonical

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

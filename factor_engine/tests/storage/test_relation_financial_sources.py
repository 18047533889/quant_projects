from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pit_contract import select_visible_row_bundles
from storage.sources.financial import load_financial_row_bundle
from storage.sources.logical_tables import ASHARE_LOGICAL_TABLES, logical_table_contract
from storage.sources.relation import (
    aggregate_holder_rows,
    effective_dividends,
    filter_index_constituents,
    filter_industry,
    top_ten_features_asof,
)


def _decisions(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"decision_timestamp": pd.to_datetime(dates), "instrument": ["A"] * len(dates)})


def test_financial_row_bundle_keeps_multiple_fields_on_one_visible_row() -> None:
    events = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "period_end": pd.to_datetime(["2023-12-31", "2024-03-31", "2023-12-31"]),
        "available_at": pd.to_datetime(["2024-03-01", "2024-04-30", "2024-05-10"]),
        "revision_id": [1, 1, 2],
        "revenue": [100.0, 30.0, 110.0],
        "profit": [10.0, 4.0, 11.0],
    })
    out = load_financial_row_bundle(
        _decisions("2024-05-01", "2024-05-11"), events, ["revenue", "profit"]
    )
    assert list(out["period_end"].dt.tz_localize(None)) == list(pd.to_datetime(["2024-03-31", "2024-03-31"]))
    assert list(zip(out["revenue"], out["profit"])) == [(30.0, 4.0), (30.0, 4.0)]


def test_financial_period_selectors_are_explicit() -> None:
    events = pd.DataFrame({
        "instrument": ["A", "A"],
        "period_end": pd.to_datetime(["2023-12-31", "2024-03-31"]),
        "available_at": pd.to_datetime(["2024-03-01", "2024-04-30"]),
        "value": [100.0, 30.0],
    })
    decisions = _decisions("2024-05-01")
    assert select_visible_row_bundles(decisions, events, selector="annual_only").loc[0, "value"] == 100.0
    assert select_visible_row_bundles(decisions, events, selector="quarterly_only").loc[0, "value"] == 30.0
    with pytest.raises(ValueError, match="unknown visible period selector"):
        select_visible_row_bundles(decisions, events, selector="monthly")  # type: ignore[arg-type]


def test_top_ten_aggregation_and_pit_visibility() -> None:
    rows = pd.DataFrame({
        "instrument": ["A"] * 12,
        "period_end": pd.to_datetime(["2024-03-31"] * 12),
        "available_at": pd.to_datetime(["2024-04-30"] * 12),
        "holding_amount": range(1, 13),
        "holding_ratio": np.arange(1, 13) / 100,
    })
    agg = aggregate_holder_rows(rows, ratio_unit="decimal")
    assert agg.loc[0, "top_ten_holder_count"] == 10
    assert agg.loc[0, "top_ten_holding_amount"] == sum(range(3, 13))
    assert np.isclose(agg.loc[0, "top_ten_holding_ratio"], sum(range(3, 13)) / 100)
    out = top_ten_features_asof(
        _decisions("2024-04-29", "2024-05-01"), rows,
        ratio_unit="decimal", allow_unbounded_staleness=True,
    )
    assert np.isnan(out.loc[0, "top_ten_holding_ratio"])
    assert np.isclose(out.loc[1, "top_ten_holding_ratio"], sum(range(3, 13)) / 100)
    # R24-033: snapshot age metadata is carried.
    assert "snapshot_available_at" in out.columns
    assert "snapshot_age_days" in out.columns


def test_holder_ratio_unit_never_inferred_from_data() -> None:
    # R24-021: raw percent values [0.3, 0.8] with a declared percent contract
    # are read as 0.3% / 0.8%, not 30% / 80%.
    rows = pd.DataFrame({
        "instrument": ["A"] * 2,
        "period_end": pd.to_datetime(["2024-03-31"] * 2),
        "available_at": pd.to_datetime(["2024-04-30"] * 2),
        "holding_amount": [100.0, 200.0],
        "holding_ratio": [0.3, 0.8],
    })
    agg = aggregate_holder_rows(rows, ratio_unit="percent")
    assert np.isclose(agg.loc[0, "top_ten_holding_ratio"], 0.003 + 0.008)


def test_holder_ratio_contract_violation_raises() -> None:
    # R24-022: a mixed out-of-contract value must raise a source-contract error,
    # not silently rescale the whole series.
    rows = pd.DataFrame({
        "instrument": ["A"] * 2,
        "period_end": pd.to_datetime(["2024-03-31"] * 2),
        "available_at": pd.to_datetime(["2024-04-30"] * 2),
        "holding_amount": [100.0, 200.0],
        "holding_ratio": [0.008, 2.0],
    })
    with pytest.raises(ValueError, match="source-contract error"):
        aggregate_holder_rows(rows, ratio_unit="decimal")


def test_holder_ratio_unit_is_required_no_guessing() -> None:
    rows = pd.DataFrame({
        "instrument": ["A"],
        "period_end": pd.to_datetime(["2024-03-31"]),
        "available_at": pd.to_datetime(["2024-04-30"]),
        "holding_amount": [100.0],
        "holding_ratio": [0.05],
    })
    with pytest.raises(ValueError, match="ratio_unit"):
        aggregate_holder_rows(rows)


def test_industry_is_single_source_and_index_symbol_is_exact() -> None:
    industry = pd.DataFrame({"IndustrySource": ["SW", "CITIC"], "code": ["A", "B"]})
    assert filter_industry(industry, "SW")["code"].tolist() == ["A"]
    with pytest.raises(ValueError, match="IndustrySource"):
        filter_industry(industry, "")
    constituents = pd.DataFrame({"IndexSymbol": ["000300.SH", "000300.SZ"], "Weight": [1.0, 2.0]})
    assert filter_index_constituents(constituents, "000300.SH")["Weight"].tolist() == [1.0]
    assert filter_index_constituents(constituents, "000300").empty


def test_dividends_are_effective_only_and_marked() -> None:
    rows = pd.DataFrame({"effective_at": pd.to_datetime(["2024-05-01", "2024-06-01"]), "cash": [1.0, 2.0]})
    out = effective_dividends(rows, decision_time="2024-05-15")
    assert out["cash"].tolist() == [1.0]
    assert out["effective_only"].eq(True).all()


def test_all_registered_ashare_tables_have_explicit_policy() -> None:
    expected = {
        "StockDailyBar", "StockMinuteBar", "StockValuationDaily", "StockCapitalDaily",
        "StockIndicator", "StockBalance", "StockIncome", "StockCashFlow", "StockDividend",
        "StockTopTenShareholder", "StockTopTenFloatShareholder", "StockIndustry", "StockStatus",
        "StockList", "IndexConstituent", "IndexDailyBar", "EtfDailyBar", "ETFList", "IndexList", "Calendar",
    }
    assert expected <= set(ASHARE_LOGICAL_TABLES)
    assert all(contract.join_policy != "special" for name, contract in ASHARE_LOGICAL_TABLES.items() if name in expected)
    assert logical_table_contract("StockIndustry").required_parameter == "IndustrySource"
    assert logical_table_contract("IndexConstituent").required_parameter == "IndexSymbol"
    assert logical_table_contract("StockDividend").join_policy == "effective_only"

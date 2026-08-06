from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fields import FIELD_REGISTRY, FieldRegistry, FieldSpec, TableSpec


def test_catalog_v2_is_deterministic_and_semantic() -> None:
    exported = FIELD_REGISTRY.export_catalog()
    assert exported["schema_version"] == "factor_engine.fields.v2"
    assert len(exported["tables"]) == 20
    assert FIELD_REGISTRY.catalog_hash() == FIELD_REGISTRY.catalog_hash()

    ret = FIELD_REGISTRY.require("ret")
    assert ret.source_name == "Return"
    assert ret.source_unit == "basis_point"
    assert ret.canonical_unit == "ratio"
    assert ret.scale_to_canonical == pytest.approx(0.0001)

    for name in ("turnover_ratio", "dividend_yield", "roe", "index_weight", "share_ratio"):
        spec = FIELD_REGISTRY.require(name)
        assert spec.source_unit == "percent"
        assert spec.canonical_unit == "ratio"
        assert spec.scale_to_canonical == pytest.approx(0.01)


def test_operator_expansion_fields_resolve_with_semantics() -> None:
    cost = FIELD_REGISTRY.require("operating_cost")
    assert cost.table == "StockIncome"
    assert cost.grain == ("flow", "ytd")
    assert cost.source_name == "OperatingCost"

    equity = FIELD_REGISTRY.require("total_owner_equities")
    assert equity.source_name == "TotalOwnerEquities"
    assert "total_equity" in equity.aliases

    capex = FIELD_REGISTRY.require("capex")
    assert capex.table == "StockCashFlow"
    assert capex.grain == ("flow", "ytd")
    assert capex.source_name == "FixIntanOtherAssetAcquiCash"

    net_income = FIELD_REGISTRY.require("net_income")
    assert net_income.name == "net_profit"
    assert net_income.source_name == "NetProfit"

    # 物理列名与 COS parquet 逐列核对（2026-08）。
    assert FIELD_REGISTRY.require("current_assets").source_name == "TotalCurrentAssets"
    assert FIELD_REGISTRY.require("current_liabilities").source_name == "TotalCurrentLiability"
    assert FIELD_REGISTRY.require("taxes_payable").source_name == "TaxsPayable"
    assert FIELD_REGISTRY.require("employee_payable").source_name == "SalariesPayable"
    assert FIELD_REGISTRY.require("operating_cash_flow").source_name == "NetOperateCashFlow"
    assert FIELD_REGISTRY.require("rd_expenses").source_name == "RdExpenses"
    assert FIELD_REGISTRY.require("total_liabilities").source_name == "TotalLiability"


def test_corrected_status_and_index_identities() -> None:
    status = FIELD_REGISTRY.require("listed_state")
    assert status.name == "public_status"
    assert status.source_name == "PublicStatus"
    index = FIELD_REGISTRY.require("index_code")
    assert index.name == "index_symbol"
    assert index.source_name == "IndexSymbol"


def test_one_to_many_fields_cannot_be_mined_directly() -> None:
    table = TableSpec("Relation", "relation", cardinality="one_to_many")
    with pytest.raises(ValueError, match="aggregated before mining"):
        FieldRegistry(
            [FieldSpec("weight", "Relation", "Weight", cardinality="one_to_many")],
            [table],
        )


def test_data_access_normalizes_registered_units_without_filling_nan(monkeypatch) -> None:
    from storage.sources.data_access_source import DataAccessSource

    source = DataAccessSource(dataset="ashare_stock_daily")
    values = pd.Series([10_000.0, np.nan], dtype=float)
    fetched = {"ret": values.copy()}
    source._normalize_contract_columns(fetched, ["ret"])
    assert fetched["ret"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(fetched["ret"].iloc[1])

    valuation = DataAccessSource(dataset="ashare_stock_valuation_daily")
    fetched = {"turnover_ratio": pd.Series([12.5, np.nan])}
    valuation._normalize_contract_columns(fetched, ["turnover_ratio"])
    assert fetched["turnover_ratio"].iloc[0] == pytest.approx(0.125)
    assert np.isnan(fetched["turnover_ratio"].iloc[1])


def test_data_access_resolves_catalog_physical_columns() -> None:
    from storage.sources.data_access_source import DataAccessSource

    daily = DataAccessSource(dataset="ashare_stock_daily")
    physical, output = daily._resolve_columns(["ret", "close"])
    assert physical == ["Return", "Close"]
    assert output == {"Return": "ret", "Close": "close"}

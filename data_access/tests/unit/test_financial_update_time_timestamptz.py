"""Production financial-table UpdateTime schema matches local Parquet footers."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.registry import load_registry
from data_access.registry.schema_validation import _is_compatible


_FINANCIAL_TABLES = (
    ("ashare_stock_indicator", "StockIndicator"),
    ("ashare_stock_cashflow", "StockCashFlow"),
    ("ashare_stock_income", "StockIncome"),
    ("ashare_stock_balance", "StockBalance"),
)


@pytest.mark.parametrize("dataset_name, directory", _FINANCIAL_TABLES)
def test_financial_update_time_is_timezone_aware_in_config_and_footer(
    dataset_name: str, directory: str
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    registry = load_registry(repo_root / "data_access/config/datasets.yaml")
    declared = registry.get(dataset_name).schema["UpdateTime"]
    assert declared == "timestamptz"
    assert _is_compatible(declared, "TIMESTAMP WITH TIME ZONE")

    parquet_root = repo_root / "data/a_share/lqtp_data" / directory
    parquet_path = next(parquet_root.rglob("*.parquet"), None)
    assert parquet_path is not None, f"no local Parquet evidence under {parquet_root}"

    # read_schema reads Parquet metadata/footer only; it does not materialize rows.
    arrow_schema = pq.read_schema(parquet_path)
    update_time = arrow_schema.field("UpdateTime").type
    assert pa.types.is_timestamp(update_time)
    assert update_time.tz is not None

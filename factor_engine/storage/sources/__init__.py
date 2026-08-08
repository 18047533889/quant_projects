"""数据源适配：Parquet / ClickHouse / data_access / Composite 等。"""

from .composite_source import CompositeDataSource
from .data_access_source import DataAccessSource
from .datasource import DataSource
from .field_plan import NormalizedFieldPlan
from .kline_parquet_source import KlineParquetSource
from .long_table_source import LongTableDataSource
from .logical_tables import ASHARE_LOGICAL_TABLES, LogicalTableContract, logical_table_contract
from .financial import FinancialFieldContract, load_financial_row_bundle
from .relation import (
    IndustrySelection,
    IndexSelection,
    aggregate_holder_rows,
    effective_dividends,
    filter_index_constituents,
    filter_industry,
    top_ten_features_asof,
)
from .parquet_source import ParquetSource
from .read_session import DataSourceReadSession

__all__ = [
    "CompositeDataSource",
    "DataAccessSource",
    "DataSource",
    "DataSourceReadSession",
    "KlineParquetSource",
    "LongTableDataSource",
    "NormalizedFieldPlan",
    "ASHARE_LOGICAL_TABLES",
    "LogicalTableContract",
    "logical_table_contract",
    "FinancialFieldContract",
    "load_financial_row_bundle",
    "IndustrySelection",
    "IndexSelection",
    "aggregate_holder_rows",
    "effective_dividends",
    "filter_index_constituents",
    "filter_industry",
    "top_ten_features_asof",
    "ParquetSource",
]

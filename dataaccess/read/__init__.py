"""读路径：契约、谓词、格式、预算、适配、SQL 逃生口。"""
from .read_contract import DataSnapshot, ReadResult, SqlReadResult
from .query_budget import QueryBudget, resolve_query_budget
from .scan_handle import ScanHandle
from .key_policy import KeyPolicy
from .adapters import arrow_table_to_multiindex_columns, arrow_to_multiindex_series
from .formats import (
    DataFormat,
    FormatSpec,
    default_glob_for_format,
    get_format_adapter,
)
from .predicate_ast import (
    And,
    Between,
    Eq,
    Filter,
    Ge,
    Gt,
    In,
    IsNotNull,
    IsNull,
    Le,
    Lt,
    Ne,
    Not,
    NotIn,
    Or,
    compile_filter_duckdb,
    compile_filter_polars,
    parse_filters,
)

__all__ = [
    "DataSnapshot",
    "ReadResult",
    "SqlReadResult",
    "QueryBudget",
    "resolve_query_budget",
    "ScanHandle",
    "KeyPolicy",
    "arrow_table_to_multiindex_columns",
    "arrow_to_multiindex_series",
    "DataFormat",
    "FormatSpec",
    "default_glob_for_format",
    "get_format_adapter",
    "Filter",
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "Between",
    "In",
    "NotIn",
    "IsNull",
    "IsNotNull",
    "And",
    "Or",
    "Not",
    "parse_filters",
    "compile_filter_duckdb",
    "compile_filter_polars",
]

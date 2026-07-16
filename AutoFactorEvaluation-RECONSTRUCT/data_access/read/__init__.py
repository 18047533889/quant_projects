"""读路径：契约、谓词、预算、适配、SQL 逃生口。"""
from .read_contract import DataSnapshot, ReadResult, SqlReadResult
from .query_budget import QueryBudget, resolve_query_budget
from .scan_handle import ScanHandle
from .key_policy import KeyPolicy
from .adapters import arrow_table_to_multiindex_columns, arrow_to_multiindex_series

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
]

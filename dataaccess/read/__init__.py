"""读路径：契约、谓词、格式、预算、适配、SQL 逃生口。"""
from .read_contract import DataSnapshot, ReadResult, SqlReadResult
from .query_budget import QueryBudget, resolve_query_budget
from .scan_handle import ScanHandle
from .key_policy import KeyPolicy
from .adapters import arrow_table_to_multiindex_columns, arrow_to_multiindex_series
from .read_handle import ReadHandle
from .relation_handle import RelationHandle
from .data_request import DataRequest, ReadPlan
from .aggregation import AggregationSpec, aggregate_minute_to_daily, parse_aggregation_spec
from .temporal_join import TemporalJoinSpec, join_spec_from_field, parse_join_spec
from .semantic_catalog import (
    SemanticField,
    SemanticFieldCatalog,
    get_semantic_catalog,
    normalize_table_units,
    parse_semantic_field,
    reset_semantic_catalog,
)
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
    "ReadHandle",
    "RelationHandle",
    "DataRequest",
    "ReadPlan",
    "AggregationSpec",
    "aggregate_minute_to_daily",
    "parse_aggregation_spec",
    "TemporalJoinSpec",
    "join_spec_from_field",
    "parse_join_spec",
    "SemanticField",
    "SemanticFieldCatalog",
    "get_semantic_catalog",
    "normalize_table_units",
    "parse_semantic_field",
    "reset_semantic_catalog",
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

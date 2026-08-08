"""IR static types and field-to-IR type inference."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ValueType(str, Enum):
    """编译期/运行期对因子结果形态的粗分类（目前 schema 层使用较少）。"""

    SCALAR = "scalar"   # 单值
    SERIES = "series"   # 一维序列（通常指单标的时序）
    PANEL = "panel"     # 二维面板 date × instrument


_DTYPE_ALIASES = {
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "date32": "date",
    "datetime": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
    "float": "float64",
    "double": "float64",
    "float64": "float64",
    "int": "int64",
    "integer": "int64",
    "int64": "int64",
    "uint64": "uint64",
    "str": "string",
    "string": "string",
    "object": "string",
}


def normalize_dtype(dtype: str | None) -> str:
    """Normalize source/catalog dtype spellings without requiring NumPy."""

    raw = str(dtype or "float64").strip().lower()
    if raw.startswith("timestamp") or raw.startswith("datetime"):
        return "datetime64[ns]"
    return _DTYPE_ALIASES.get(raw, raw)


def infer_field_type(value: Any, *, table: str | None = None):
    """Infer a registered ``FieldSpec`` for a name/ColumnRef/SourceRef.

    ``None`` means the legacy column has no semantic registration; callers
    should retain their historical float-panel fallback in that case.
    """

    from fields import resolve_field

    return resolve_field(value, table=table, strict=False)


class SemanticType(str, Enum):
    """Typed-IR semantic kind vocabulary (review P0-30).

    The analyzer propagates ``price_basis`` / ``flow_semantics`` /
    ``frequency`` / ``domain`` onto ``IRNode.semantic_attrs`` and maps them to
    one of these kinds so compile-time contract checks stop guessing from field
    names.  This is additive: the coarse :class:`ValueType` (scalar/series/
    panel) and the dtype aliases above are untouched.
    """

    PRICE_RAW = "PriceRaw"
    PRICE_CONTINUOUS = "PriceContinuous"
    OFFICIAL_LIMIT_PRICE = "OfficialLimitPrice"
    RETURN_DECIMAL = "ReturnDecimal"
    POSITIVE_LEVEL = "PositiveLevel"
    NON_NEGATIVE_ACTIVITY = "NonNegativeActivity"
    NON_NEGATIVE_WEIGHT = "NonNegativeWeight"
    EVENT_BOOL = "EventBool"
    STATE_SIGNED = "StateSigned"
    GROUP_KEY = "GroupKey"
    MINUTE_SERIES = "MinuteSeries"
    DAILY_SERIES = "DailySeries"
    FINANCIAL_STOCK = "FinancialStock"
    FINANCIAL_SINGLE_PERIOD_FLOW = "FinancialSinglePeriodFlow"
    FINANCIAL_CUMULATIVE_YTD_FLOW = "FinancialCumulativeYTDFlow"
    FINANCIAL_TTM_FLOW = "FinancialTTMFlow"
    RELATION_GRAPH = "RelationGraph"


# SEMANTIC_TYPE registry: string keys from the existing ``Schema``/field
# vocabulary -> semantic kind.  ``semantic_type_of`` below is the primary entry
# point; the registry is kept for introspection / reverse lookups.
SEMANTIC_TYPE: dict[str, SemanticType] = {
    "price_basis.raw": SemanticType.PRICE_RAW,
    "price_basis.continuous": SemanticType.PRICE_CONTINUOUS,
    "price_basis.raw_official_limit": SemanticType.OFFICIAL_LIMIT_PRICE,
    "price_basis.return": SemanticType.RETURN_DECIMAL,
    "flow_semantics.stock": SemanticType.FINANCIAL_STOCK,
    "flow_semantics.single_period_flow": SemanticType.FINANCIAL_SINGLE_PERIOD_FLOW,
    "flow_semantics.cumulative_ytd_flow": SemanticType.FINANCIAL_CUMULATIVE_YTD_FLOW,
    "flow_semantics.ttm_flow": SemanticType.FINANCIAL_TTM_FLOW,
    "frequency.minute": SemanticType.MINUTE_SERIES,
    "frequency.daily": SemanticType.DAILY_SERIES,
    "role.group_key": SemanticType.GROUP_KEY,
    "domain.relation": SemanticType.RELATION_GRAPH,
}

_PRICE_BASIS_TO_SEMANTIC: dict[str, SemanticType] = {
    "RAW": SemanticType.PRICE_RAW,
    "CONTINUOUS": SemanticType.PRICE_CONTINUOUS,
    "RAW_OFFICIAL_LIMIT": SemanticType.OFFICIAL_LIMIT_PRICE,
    "RETURN": SemanticType.RETURN_DECIMAL,
}

_FLOW_SEMANTICS_TO_SEMANTIC: dict[str, SemanticType] = {
    "stock": SemanticType.FINANCIAL_STOCK,
    "single_period_flow": SemanticType.FINANCIAL_SINGLE_PERIOD_FLOW,
    "cumulative_ytd_flow": SemanticType.FINANCIAL_CUMULATIVE_YTD_FLOW,
    "ttm_flow": SemanticType.FINANCIAL_TTM_FLOW,
}


def semantic_type_of(
    *,
    price_basis: str | None = None,
    flow_semantics: str | None = None,
    frequency: str | None = None,
    domain: str | None = None,
    role: str | None = None,
) -> SemanticType | None:
    """Map propagated semantic attributes to a :class:`SemanticType`.

    Precedence: explicit flow semantics (financial), then price basis, then the
    coarse shape descriptors.  ``None`` means the attributes do not pin down a
    semantic kind (generic derived numeric series).
    """
    if flow_semantics:
        kind = _FLOW_SEMANTICS_TO_SEMANTIC.get(str(flow_semantics).lower())
        if kind is not None:
            return kind
    if price_basis:
        kind = _PRICE_BASIS_TO_SEMANTIC.get(str(price_basis).upper())
        if kind is not None:
            return kind
    if role == "group_key":
        return SemanticType.GROUP_KEY
    if domain == "relation":
        return SemanticType.RELATION_GRAPH
    if str(frequency or "").lower() == "minute":
        return SemanticType.MINUTE_SERIES
    if str(frequency or "").lower() == "daily":
        return SemanticType.DAILY_SERIES
    return None


__all__ = [
    "SEMANTIC_TYPE",
    "SemanticType",
    "ValueType",
    "infer_field_type",
    "normalize_dtype",
    "semantic_type_of",
]

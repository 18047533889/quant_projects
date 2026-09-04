"""AlphaSchema-style 语义 Schema 空间（plan Task 12 / Part B2）。

``SchemaPlan``（contracts.py）定义九维语义；本模块提供：

- :func:`derive_schema_id`：同 schema 不同公式实现共享 schema_id（稳定版本化）。
- :func:`validate_schema_plan`：语义字段对照 DA taxonomy 校验（DataDomain
  必须 ∈ DA ``VALID_FIELD_DOMAINS``；其余枚举用保守白名单）；非法即拒
  （抛 ``SchemaValidationError``）。
- schema 性能统计独立于公式层：见 registry.py（``SchemaRegistry``）。

Part E 边界
-----------
- 本模块**不 import data_access**：DA taxonomy 词表以 ``DomainVocabulary``
  注入（生产绑 ``data_access.read.semantic_catalog.VALID_FIELD_DOMAINS``，
  测试绑 fake/子集）。不复制 taxonomy，只消费其词表做校验。
- Horizon / Normalization / Tradability / Direction 等维度无 DA 权威枚举，
  用本包保守白名单（缺省 unknown 放行、非法显式拒绝）——避免对非权威维度
  发明新语义。

说明（plan 契约九字段 与 arms.SCHEMA_DIMENSIONS 的关系）
---------------------------------------------------------
SchemaExploreArm 的 9 维 schema_tags 值（如 DataDomain=price/fundamental）
沿用既有表面；本模块的 ``DataDomain`` **校验词表用 DA taxonomy 大小写不敏感
匹配**（PRICE/FUNDAMENTAL.QUALITY…），把 lowercase 表面归一后校验；非法值
（如 ``FUNDAMENTAL.VOLATILITY`` 不在词表）→ 拒绝。Horizon 用
short/medium/long/fast/slow/unknown 保守集（与 search/dna 的 horizon bucket
表面一致）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from alphaprobe.research_space.contracts import SchemaPlan

logger = logging.getLogger(__name__)

__all__ = [
    "SchemaValidationError",
    "DomainVocabulary",
    "DEFAULT_DOMAIN_VOCABULARY",
    "validate_schema_plan",
    "validate_domain_token",
    "canonical_domain_token",
    "schema_canonical_tags",
    "HORIZON_VALUES",
    "NORMALIZATION_VALUES",
    "TRADABILITY_VALUES",
    "DIRECTION_VALUES",
    "OUTPUT_VALUES",
]

#: Horizon 保守白名单（surface 与 search/dna horizon_bucket 对齐：fast/medium/
#: slow + short/medium/long 兼容 + unknown）。非权威枚举，只做表面归一/非法拒。
HORIZON_VALUES: frozenset[str] = frozenset(
    {
        "unknown", "short", "medium", "long", "fast", "slow",
        "daily", "weekly", "monthly",
    }
)

#: Normalization 保守白名单（surface 与 SchemaExploreArm 对齐 + none/unknown）。
NORMALIZATION_VALUES: frozenset[str] = frozenset(
    {"unknown", "cross_section", "time_series", "none", "rank", "zscore"}
)

#: Tradability 保守白名单（surface 与 SchemaExploreArm 对齐 + unknown）。
TRADABILITY_VALUES: frozenset[str] = frozenset(
    {"unknown", "liquid", "illiquid", "cap_small", "cap_large", "any"}
)

#: Direction 保守白名单。
DIRECTION_VALUES: frozenset[str] = frozenset({"unknown", "long", "short", "both"})

#: Output 保守白名单。
OUTPUT_VALUES: frozenset[str] = frozenset(
    {"unknown", "rank", "zscore", "raw", "binary", "signal"}
)

#: DataDomain 大小写不敏感归一后的 DA taxonomy 词表（最小显式子集，与 DA
#: VALID_FIELD_DOMAINS 一致；生产注入完整词表用 DomainVocabulary）。
DEFAULT_DOMAIN_VOCABULARY: frozenset[str] = frozenset(
    {
        "PRICE",
        "VOLUME",
        "LIQUIDITY",
        "FUNDAMENTAL.VALUE",
        "FUNDAMENTAL.QUALITY",
        "FUNDAMENTAL.GROWTH",
        "FUNDAMENTAL.INVESTMENT",
        "FUNDAMENTAL.CASHFLOW",
        "FUNDAMENTAL.LEVERAGE",
        "EVENT",
        "FLOW_SENTIMENT",
        "MICROSTRUCTURE",
        "RISK",
        "CALENDAR",
        "ALTERNATIVE",
    }
)


class SchemaValidationError(ValueError):
    """SchemaPlan 语义字段非法（对照 DA taxonomy / 白名单校验失败）。"""


class DomainVocabulary(Protocol):
    """DA taxonomy 词表的注入面（``__contains__`` / ``__iter__``）。"""

    def __contains__(self, token: str) -> bool: ...

    def __iter__(self) -> Iterable[str]: ...


def canonical_domain_token(raw: str, vocabulary: DomainVocabulary) -> str | None:
    """把任意大小写 domain token 归一成词表里的大小写（找不到返回 None）。

    DA ``VALID_FIELD_DOMAINS`` 是规范大写；SchemaExploreArm 用 lowercase
    （price/fundamental）。消费方不需要猜：只需「词表里有谁就归一成谁」。
    """
    up = str(raw or "").strip().upper()
    if not up:
        return None
    # 1) 精确命中（大小写不敏感：词表本身大写，直接 in）
    if up in vocabulary:
        return up
    # 2) FUNDAMENTAL 宽前缀：expected_domains 允许 FUNDAMENTAL.QUALITY 之类，
    #    schema 的 DataDomain 允许写 FUNDAMENTAL（广义基本面）——归一为
    #    FUNDAMENTAL.VALUE? 不可猜具体子类，返回 None 由调用方判。
    if up == "FUNDAMENTAL":
        # 词表里没有裸 FUNDAMENTAL（都是 FUNDAMENTAL.*）。广义 fundamental
        # 由 domain 集合判定（has_fundamental_domain），单值校验不接受裸值。
        return None
    for token in vocabulary:
        if str(token).upper() == up:
            return str(token)
    return None


def validate_domain_token(raw: str, vocabulary: DomainVocabulary) -> str:
    """校验并归一一个 DataDomain token；非法 → SchemaValidationError。"""
    norm = canonical_domain_token(raw, vocabulary)
    if norm is None:
        allowed = ", ".join(sorted(str(v) for v in vocabulary))
        raise SchemaValidationError(
            f"DataDomain '{raw}' 不在 DA taxonomy 词表中（允许: {allowed}）。"
            "禁止自造 domain 语义（Part E：字段语义唯一权威 = DA taxonomy）。"
        )
    return norm


def schema_canonical_tags(plan: SchemaPlan, vocabulary: DomainVocabulary) -> dict[str, str]:
    """校验语义字段并返回 9 维 canonical tags（非法即拒）。

    只有 DataDomain 对照 DA taxonomy；其它枚举维度对照保守白名单
    （unknown 或空 = 不校验放行；显式非法值 = 拒绝）。
    """
    out: dict[str, str] = {}
    # DataDomain
    dd = str(plan.data_domain or "").strip()
    if dd:
        out["DataDomain"] = validate_domain_token(dd, vocabulary)
    # Horizon
    hz = str(plan.horizon or "").strip()
    if hz and hz not in HORIZON_VALUES:
        raise SchemaValidationError(
            f"Horizon '{hz}' 非法（应为 {sorted(HORIZON_VALUES)} 之一）"
        )
    if hz:
        out["Horizon"] = hz
    # Normalization
    nm = str(plan.normalization or "").strip()
    if nm and nm not in NORMALIZATION_VALUES:
        raise SchemaValidationError(
            f"Normalization '{nm}' 非法（应为 {sorted(NORMALIZATION_VALUES)} 之一）"
        )
    if nm:
        out["Normalization"] = nm
    # Tradability
    tr = str(plan.tradability or "").strip()
    if tr and tr not in TRADABILITY_VALUES:
        raise SchemaValidationError(
            f"Tradability '{tr}' 非法（应为 {sorted(TRADABILITY_VALUES)} 之一）"
        )
    if tr:
        out["Tradability"] = tr
    # Direction
    dr = str(plan.direction or "").strip()
    if dr and dr not in DIRECTION_VALUES:
        raise SchemaValidationError(
            f"Direction '{dr}' 非法（应为 {sorted(DIRECTION_VALUES)} 之一）"
        )
    if dr:
        out["Direction"] = dr
    # Output
    op = str(plan.output or "").strip()
    if op and op not in OUTPUT_VALUES:
        raise SchemaValidationError(
            f"Output '{op}' 非法（应为 {sorted(OUTPUT_VALUES)} 之一）"
        )
    if op:
        out["Output"] = op
    # 自由文本维（Event/Context/Qualities 不校验）
    for dim, attr in (
        ("Event", "event"),
        ("Context", "context"),
        ("Qualities", "qualities"),
    ):
        raw = str(getattr(plan, attr) or "").strip()
        if raw:
            out[dim] = raw
    return out


def validate_schema_plan(plan: SchemaPlan, vocabulary: DomainVocabulary | None = None) -> SchemaPlan:
    """校验 SchemaPlan 语义字段（对照注入的 DA taxonomy 词表）；非法即抛。

    返回原 plan（不修改；schema_id 派生统一走 ``derive_schema_id``，它调用
    本函数保证非法 schema 不能进 registry）。
    """
    vocab = vocabulary if vocabulary is not None else DEFAULT_DOMAIN_VOCABULARY
    schema_canonical_tags(plan, vocab)
    return plan

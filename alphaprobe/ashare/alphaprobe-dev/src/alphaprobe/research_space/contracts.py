"""research_space 五层研究空间的公共契约（plan Part B / Task 6 / Task 12）。

本模块是**纯 dataclass 契约层**：
- 不 import factor_engine / data_access / factor_assets（顶层零平台依赖，
  保证 OFFLINE_TEST 与纯单测可 import）；
- HypothesisSpec / AlignmentResult 契约与 plan.md Task 6 接口完全一致；
- SchemaPlan / ImplementationRecord / SchemaStats 契约与 plan.md Task 12
  九维 SchemaPlan 对齐（Event/Context/Qualities/Direction/Output/DataDomain/
  Horizon/Normalization/Tradability）。

Part E 边界：研究空间层只**消费**平台能力（FE static analysis artifact、
DA FieldTaxonomyProvider 的 descriptor、FE identity），不复制解析/taxonomy/
identity 能力。具体接线在 alignment.py / schema.py（注入式，顶层不 import）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "HypothesisSpec",
    "AlignmentResult",
    "DOMAIN_TOKEN_FUNDAMENTAL",
    "SCHEMA_DIMENSIONS",
    "SchemaPlan",
    "ImplementationRecord",
    "ImplementationOutcome",
    "SchemaStats",
    "SchemaIdentity",
]

# ---------------------------------------------------------------------------
# Hypothesis Alignment（plan Task 6）
# ---------------------------------------------------------------------------

#: 规范化 fundamental domain 前缀：FUNDAMENTAL.VALUE/QUALITY/GROWTH/INVESTMENT/
#: CASHFLOW/LEVERAGE 在判定「hypothesis 声称 fundamental」时都算命中。
DOMAIN_TOKEN_FUNDAMENTAL = "FUNDAMENTAL"


@dataclass(frozen=True)
class HypothesisSpec:
    """一条结构化经济假设（plan.md Task 6 契约）。

    Attributes
    ----------
    hypothesis_id : str
        稳定假设 id。
    logic_id / schema_id : str | None
        可选上层引用（五层研究空间的跨层 id）。
    text : str
        自然语言假设。
    expected_domains : tuple[str, ...]
        假设声称用到的 data-domain token（DA taxonomy 词表，
        如 ``PRICE`` / ``VOLUME`` / ``FUNDAMENTAL.QUALITY``）。
    expected_roles : tuple[str, ...]
        假设声称用到的 economic-role token（如 ``price_level`` /
        ``volume_traded``）。
    expected_horizon : str | None
        可选期望 horizon（如 ``medium``）；None 表示不约束。
    """

    hypothesis_id: str
    logic_id: str | None = None
    schema_id: str | None = None
    text: str = ""
    expected_domains: tuple[str, ...] = ()
    expected_roles: tuple[str, ...] = ()
    expected_horizon: str | None = None


@dataclass(frozen=True)
class AlignmentResult:
    """确定性 hypothesis↔implementation 对齐结果（plan.md Task 6 契约）。

    ``score`` ∈ [0, 1]：实现与假设的一致程度（1.0 = 完全一致）。
    ``hard_mismatch``：True = 领域矛盾（如假设声称 FUNDAMENTAL 但实现只用
    PRICE/VOLUME）——**评估前必须被拦截**（Non-negotiable #18）。
    ``missing_domains``：假设声称但实现没有的 domain。
    ``unexpected_domains``：实现使用但假设没有声称的 domain。
    ``timing_flags``：从实现侧（FE static analysis 或其扩展位）带来的
    timing/lookahead 风险标记；非空即表示不应进入市场评估。
    ``explanation_codes``：机器可读解释码（不解析自然语言）。
    """

    score: float = 1.0
    hard_mismatch: bool = False
    missing_domains: tuple[str, ...] = ()
    unexpected_domains: tuple[str, ...] = ()
    timing_flags: tuple[str, ...] = ()
    explanation_codes: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        """True = 该候选不应进入市场评估（领域矛盾或 timing 风险）。"""
        return self.hard_mismatch or bool(self.timing_flags)

    @property
    def ok(self) -> bool:
        """True = 通过确定性对齐检查。"""
        return not self.hard_mismatch and not self.timing_flags


# ---------------------------------------------------------------------------
# Schema Space（plan Task 12）
# ---------------------------------------------------------------------------

#: SchemaPlan 九维（与 search/arms.SCHEMA_DIMENSIONS 顺序对齐；值语义经
#: :func:`SchemaPlan.canonical_tags` 归一后参与 schema_id 派生）。
SCHEMA_DIMENSIONS: tuple[str, ...] = (
    "Event",
    "Context",
    "Qualities",
    "Direction",
    "Output",
    "DataDomain",
    "Horizon",
    "Normalization",
    "Tradability",
)


def _norm_tag(value: Any) -> str:
    s = str(value or "").strip()
    # DataDomain 大小写归一：PRICE / price / Price → PRICE
    return s.upper()


@dataclass(frozen=True)
class SchemaPlan:
    """机器可结构化的九维 Schema（plan.md Task 12 / Part B2）。

    ``schema_id`` 由九维语义稳定派生（与 LLM 起的名字无关；同语义 →
    同 id）。``name``/``description`` 仅展示，不参与 id。
    九字段全部可空字符串（未知维度 = ""，序列化稳定）。

    Schema ID 版本化：派生时带 ``SCHEMA_ID_VERSION``；语义字段值改变 →
    id 改变；仅 name/description/owner 改变 → id 不变（同一 schema 可被不同
    公式实现共享）。
    """

    event: str = ""
    context: str = ""
    qualities: str = ""
    direction: str = ""
    output: str = ""
    data_domain: str = ""
    horizon: str = ""
    normalization: str = ""
    tradability: str = ""
    #: 展示用（不参与 schema_id）
    name: str = ""
    description: str = ""
    owner: str = ""

    @classmethod
    def from_tags(cls, tags: Mapping[str, Any]) -> "SchemaPlan":
        """从 9 维 schema_tags dict（search arms/LLM 产物）构造。

        未知 key 忽略；已知 9 维 key 缺失 → 空串。多余 key 不进 SchemaPlan
        （避免把非 schema 标签混入语义 id）。
        """
        return cls(
            event=str(tags.get("Event") or ""),
            context=str(tags.get("Context") or ""),
            qualities=str(tags.get("Qualities") or ""),
            direction=str(tags.get("Direction") or ""),
            output=str(tags.get("Output") or ""),
            data_domain=str(tags.get("DataDomain") or ""),
            horizon=str(tags.get("Horizon") or ""),
            normalization=str(tags.get("Normalization") or ""),
            tradability=str(tags.get("Tradability") or ""),
        )

    def as_tags(self) -> dict[str, str]:
        """转回 9 维 schema_tags dict（空维度省略，兼容 search/arms）。"""
        return {
            k: str(getattr(self, _DIMENSION_ATTR[k]))
            for k in SCHEMA_DIMENSIONS
            if str(getattr(self, _DIMENSION_ATTR[k]) or "").strip()
        }

    def canonical_tags(self) -> dict[str, str]:
        """9 维语义规范化 tag（参与 schema_id；大小写/空白归一）。"""
        out: dict[str, str] = {}
        for dim in SCHEMA_DIMENSIONS:
            attr = _DIMENSION_ATTR[dim]
            raw = str(getattr(self, attr) or "").strip()
            if not raw:
                continue
            if dim == "DataDomain":
                out[dim] = _norm_tag(raw)
            else:
                out[dim] = raw
        return out

    def semantic_payload(self) -> dict[str, str]:
        """稳定版本化语义载荷（schema_id 输入；name/desc/owner 不进）。"""
        return {
            "version": SCHEMA_ID_VERSION,
            "dims": self.canonical_tags(),
        }

    def derive_schema_id(self) -> str:
        """确定性 schema_id：sha256('schemav1:' + 规范化 JSON)。"""
        payload = _schema_payload_bytes(self.semantic_payload())
        return "SCH_" + hashlib.sha256(payload, usedforsecurity=False).hexdigest()[:16].upper()

    def as_dict(self) -> dict[str, str]:
        d = {_DIMENSION_ATTR[k]: str(getattr(self, _DIMENSION_ATTR[k]) or "")
             for k in SCHEMA_DIMENSIONS}
        d.update(
            {
                "name": self.name,
                "description": self.description,
                "owner": self.owner,
            }
        )
        return d


#: schema_id 语义版本（参与哈希；语义字段表变化 → bump → 新旧 schema 不混）。
SCHEMA_ID_VERSION: str = "1"


#: SCHEMA_DIMENSIONS（CamelCase）→ SchemaPlan 字段名（snake_case）。
_DIMENSION_ATTR: dict[str, str] = {
    "Event": "event",
    "Context": "context",
    "Qualities": "qualities",
    "Direction": "direction",
    "Output": "output",
    "DataDomain": "data_domain",
    "Horizon": "horizon",
    "Normalization": "normalization",
    "Tradability": "tradability",
}


def _schema_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    import json

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


@dataclass(frozen=True)
class SchemaIdentity:
    """Schema 的稳定身份（schema_id + 派生信息）。"""

    schema_id: str
    semantic_version: str = SCHEMA_ID_VERSION
    canonical_tags: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_id": self.schema_id,
            "semantic_version": self.semantic_version,
            "canonical_tags": dict(self.canonical_tags),
        }


@dataclass
class ImplementationRecord:
    """一个公式实现（schema 的一个 implementation）。

    - ``implementation_id``：调用方给（如 factor_id / attempt_id）；
    - ``schema_id``：所属 schema；
    - ``formula``：FE DSL 文本；
    - 失败实现照常记录（failure_reason 非空），但**不自动把 schema 标记为
      failed**（plan Task 12：failed implementation ≠ failed schema）。
    """

    implementation_id: str
    schema_id: str
    formula: str
    status: str = "recorded"  # recorded / evaluated / promoted / failed
    outcome: "ImplementationOutcome | None" = None
    failure_reason: str = ""
    created_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "failed" or (
            self.outcome is not None and self.outcome.failed
        )


@dataclass(frozen=True)
class ImplementationOutcome:
    """一次实现的评估结果（schema 层聚合的输入，独立于公式层）。

    Attributes
    ----------
    reward : float
        实现层综合回报（0-1；缺样本时由 registry 层做置信收缩）。
    fitness / rank_ic / long_short_quality / stability : float | None
        可选的细分事实（registry 只聚合计数与均值，不重复权威口径——
        细分量纲由调用方统一）。
    elite : bool
        是否 elite（进入池/export）。
    n_eval : int
        该条结果的评估次数（多次评估聚合后的有效样本；默认 1）。
    """

    reward: float = 0.0
    fitness: float | None = None
    rank_ic: float | None = None
    long_short_quality: float | None = None
    stability: float | None = None
    elite: bool = False
    failed: bool = False
    n_eval: int = 1
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SchemaStats:
    """Schema 层聚合统计（**独立于公式层**，plan Task 12）。

    - 计数（n_impl / n_eval / n_success / n_elite / n_failed_impl）在 registry
      每次 record_outcome 时增量维护（版本化；同 schema 不同实现共享）；
    - ``aggregate_reward`` 是 schema 层的均值回报（**带置信收缩**，Part G
      #29：n_impl 小的 schema 收缩向中性 0.5，绝不当最好）；
    - ``elite_rate`` = n_elite / n_impl（无实现时 None，不编 0）；
    - ``is_failed`` 仅当显式 mark_schema_failed 或全部实现失败时才 True。
    """

    schema_id: str
    version: str = SCHEMA_ID_VERSION
    n_impl: int = 0
    n_eval: int = 0
    n_success: int = 0
    n_elite: int = 0
    n_failed_impl: int = 0
    reward_sum: float = 0.0
    reward_conf_sum: float = 0.0  # 置信收缩后 reward 的累计（聚合均值用）
    aggregate_reward: float = 0.5  # 无实现时的中性缺省（不静默当 0/1）
    is_failed: bool = False
    updated_at: float = 0.0

    @property
    def elite_rate(self) -> float | None:
        return self.n_elite / self.n_impl if self.n_impl else None

    @property
    def success_rate(self) -> float | None:
        return self.n_success / self.n_impl if self.n_impl else None

    @property
    def has_evidence(self) -> bool:
        return self.n_impl > 0 and self.n_eval > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "n_impl": self.n_impl,
            "n_eval": self.n_eval,
            "n_success": self.n_success,
            "n_elite": self.n_elite,
            "n_failed_impl": self.n_failed_impl,
            "aggregate_reward": self.aggregate_reward,
            "elite_rate": self.elite_rate,
            "success_rate": self.success_rate,
            "is_failed": self.is_failed,
            "updated_at": self.updated_at,
        }


#: registry 默认收缩强度（与 fitness/confidence.DEFAULT_SHRINKAGE_K=64 对齐语义；
#: 本地实现避免 import fitness 细节）。
DEFAULT_SHRINKAGE_K: float = 64.0


# ---------------------------------------------------------------------------
# （补充）Alignment 结果辅助常量：默认中性分数与允许的全局分数。
# ---------------------------------------------------------------------------

#: 确定性对齐通过时的默认满分（plan 契约无衰减即 1.0）。
ALIGNMENT_FULL_SCORE: float = 1.0
#: hard domain contradiction 的分数（0 分，必拦）。
ALIGNMENT_MISMATCH_SCORE: float = 0.0

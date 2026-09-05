"""Market Logic Library —— LogicNode / LogicLibrary（plan Task 13 / Part B1）。

AlphaLogics 风格的**市场逻辑记忆**：把历史因子里反复出现的可复用机制
（如 Liquidity Absorption Reversal / Overnight Gap Mean Reversion /
Low-volatility Compression → Expansion / Volume-price divergence / Fundamental
revision × price underreaction）表示为 LogicNode——稳定机器 id 与 LLM 起的
名字分离，一个 logic 不绑定单一公式（成员因子可概率、多标签映射）。

Non-negotiable / Part G 映射
----------------------------
- #23（当前版本 Test survival 不影响当前搜索）：``historical_survival`` 只
  接受 **frozen 上一研究版本**的 survival 事实。``record_logic_outcome``
  只在调用方显式传 ``survival_source_version < 当前研究版本`` 时才把 survival
  计入 historical_survival——当前版本 sealed test 的 survival 绝不进当前版本
  logic reward。
- #29（logic 统计版本化 + 置信收缩）：``support_count`` 小的 logic 的
  elite_rate / median_factor_fitness / median_long_short_quality /
  median_stability 全部向**全局先验**收缩（``shrunk_stats_of``）；每个
  logic_id 每个语义版本保留独立统计行，版本提升后旧统计不混。
- #24（大历史不物化进 DAG）：LogicLibrary 只存 **reference query / 聚合统计**
  （``member_factor_ids`` 可空，改用 ``reference_query``），不拉因子面板、
  不复制因子进 AlphaPROBE。

Part E 边界：本模块顶层零平台依赖（不 import factor_engine / data_access /
factor_assets / torch）。FactorAssets 的读取在 :mod:`logic_miner`（可注入
registry / adapter）；本模块只消费「因子事实行」（dataclass 或 dict 均可）。

确定性 / logic_id
-----------------
``LogicNode.signature`` 持有**机器可验证**的机制签名（mechanisms +
operator_motifs + field_families + horizon_bucket）；``derive_logic_id`` =
sha256('logicv1:' + 规范化签名 JSON)。**LLM 生成的 logic_text / aliases /
显示名不参与签名** → alias/名字变化不改 logic_id（机器 ID 与 LLM 名字分离）。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

#: logic_id 语义版本（参与哈希；签名维度表变化 → bump → 新旧 logic 不混）。
LOGIC_ID_VERSION: str = "1"

#: 签名维度键（排序参与 logic_id；展示字段不参与）。
LOGIC_SIGNATURE_DIMENSIONS: tuple[str, ...] = (
    "mechanisms",
    "operator_motifs",
    "field_families",
    "horizon_bucket",
)

#: 全局收缩强度 k（与 fitness/confidence.DEFAULT_SHRINKAGE_K=64 语义对齐；
#: 本地实现避免 import fitness 细节）。
DEFAULT_SHRINKAGE_K: float = 64.0

#: 中性缺省（无有效样本/无实现时不编 0/1）。
NEUTRAL_MEAN: float = 0.5


class LogicSignatureError(ValueError):
    """LogicNode 语义签名非法（空机制集/非法维度值）。"""


def _norm_token(value: Any) -> str:
    return str(value or "").strip().upper()


@dataclass(frozen=True)
class LogicSignature:
    """Logic 的机器可验证签名（参与 logic_id；展示字段不在内）。

    Attributes
    ----------
    mechanisms : tuple[str, ...]
        机制 DNA 标签（如 reversal / momentum_or_corr / liquidity /
        volatility / valuation）。**必须非空**——一条可复用市场逻辑至少要
        表达一个机制。
    operator_motifs : tuple[str, ...]
        算子/算子 motif（可选；同一机制可有多组算子实现）。
    field_families : tuple[str, ...]
        字段 family（可选；price_volume / liquidity / valuation /
        fundamental / intraday …）。
    horizon_bucket : str
        fast / medium / slow / unknown（可选）。
    """

    mechanisms: tuple[str, ...]
    operator_motifs: tuple[str, ...] = ()
    field_families: tuple[str, ...] = ()
    horizon_bucket: str = "unknown"

    def __post_init__(self) -> None:
        mechs = tuple(
            _norm_token(m) for m in self.mechanisms if str(m or "").strip()
        )
        if not mechs:
            raise LogicSignatureError(
                "LogicSignature.mechanisms must be non-empty "
                "(a market logic must express at least one reusable mechanism)"
            )
        ops = tuple(sorted({_norm_token(o) for o in self.operator_motifs if str(o or "").strip()}))
        fams = tuple(sorted({_norm_token(f) for f in self.field_families if str(f or "").strip()}))
        hz = _norm_token(self.horizon_bucket) or "unknown"
        # frozen：先 object.__setattr__
        object.__setattr__(self, "mechanisms", tuple(sorted(set(mechs))))
        object.__setattr__(self, "operator_motifs", ops)
        object.__setattr__(self, "field_families", fams)
        object.__setattr__(self, "horizon_bucket", hz)

    def canonical(self) -> dict[str, Any]:
        return {
            "version": LOGIC_ID_VERSION,
            "mechanisms": list(self.mechanisms),
            "operator_motifs": list(self.operator_motifs),
            "field_families": list(self.field_families),
            "horizon_bucket": self.horizon_bucket,
        }

    def derive_logic_id(self) -> str:
        """确定性 logic_id：**只 hash 机制集合**（market logic 锚点）。

        operator_motifs / field_families / horizon_bucket 是实现变体的
        surface 上下文，不参与 id——同一机制换算子/字段/horizon 实现仍是
        同一条逻辑（plan：Logic 不绑定单一公式；别名/名字变化不改 id）。
        """
        payload = json.dumps(
            {
                "version": LOGIC_ID_VERSION,
                "mechanisms": list(self.mechanisms),
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        return "LOG_" + hashlib.sha256(
            payload.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:16].upper()

    def as_dict(self) -> dict[str, Any]:
        return self.canonical()

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "LogicSignature":
        return cls(
            mechanisms=tuple(m.get("mechanisms") or ()),
            operator_motifs=tuple(m.get("operator_motifs") or ()),
            field_families=tuple(m.get("field_families") or ()),
            horizon_bucket=str(m.get("horizon_bucket") or "unknown"),
        )


def canonical_logic_signature(
    *,
    mechanisms: Sequence[str],
    operator_motifs: Sequence[str] | None = None,
    field_families: Sequence[str] | None = None,
    horizon_bucket: str | None = None,
) -> dict[str, Any]:
    """构造规范化签名 dict（供 derive_logic_id 使用）。"""
    return LogicSignature(
        mechanisms=tuple(mechanisms),
        operator_motifs=tuple(operator_motifs or ()),
        field_families=tuple(field_families or ()),
        horizon_bucket=horizon_bucket or "unknown",
    ).canonical()


def derive_logic_id(
    *,
    mechanisms: Sequence[str],
    operator_motifs: Sequence[str] | None = None,
    field_families: Sequence[str] | None = None,
    horizon_bucket: str | None = None,
) -> str:
    """确定性 logic_id：sha256('logicv1:' + 规范化签名 JSON) → 'LOG_' + 16 hex。"""
    sig = LogicSignature(
        mechanisms=tuple(mechanisms),
        operator_motifs=tuple(operator_motifs or ()),
        field_families=tuple(field_families or ()),
        horizon_bucket=horizon_bucket or "unknown",
    )
    return sig.derive_logic_id()


def logic_signature_match(node_signature: LogicSignature, candidate: LogicSignature) -> bool:
    """签名是否命中一个 logic（机制集合完全一致；其余维度宽松包含）。"""
    if set(candidate.mechanisms) != set(node_signature.mechanisms):
        return False
    if candidate.operator_motifs and node_signature.operator_motifs:
        if not set(candidate.operator_motifs).issuperset(node_signature.operator_motifs):
            return False
    if candidate.field_families and node_signature.field_families:
        if not set(candidate.field_families).issuperset(node_signature.field_families):
            return False
    # horizon：任一侧 unknown/空 = 无约束 → 放行；两侧都明确才比较。
    cand_hz = _norm_token(candidate.horizon_bucket)
    node_hz = _norm_token(node_signature.horizon_bucket)
    if cand_hz != "UNKNOWN" and node_hz != "UNKNOWN":
        if cand_hz != node_hz:
            return False
    return True

# ---------------------------------------------------------------------------
# LogicNode 契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicNode:
    """一条市场逻辑（AlphaLogics-style；plan Task 13 字段全清单）。

    Attributes
    ----------
    logic_id : str
        稳定机器 id（签名派生；alias/名字变化不改 id）。可显式传入，但必须
        与 ``signature.derive_logic_id()`` 一致（不一致 → ValueError，防止
        手工 id 绕过确定性签名）。
    signature : LogicSignature
        机器可验证签名（机制锚点）。**不可变**——机制改变 = 新 logic。
    logic_version : str
        语义版本（LOGIC_ID_VERSION；版本提升后旧统计保留）。
    logic_text : str
        LLM/人工可读的市场机制描述（**展示用，不参与 id**）。
    aliases : tuple[str, ...]
        历史别名/曾用名（展示用；改名不改 logic_id）。
    member_factor_ids : tuple[str, ...]
        已确认成员因子的 id 列表（**可空**——大历史用 reference_query 表达，
        不物化全量因子进 AlphaPROBE，#24）。
    reference_query : str
        成员因子的 reference query（FactorAssets 检索式）。可空。
    schema_distribution : dict[str, float]
        schema_id → 成员在该 schema 的聚合权重（多标签映射的 schema 侧证据）。
        空 = 未知（绝不当 0）。
    support_count : int
        支撑该 logic 的有效成员/实现样本数（n_eff 维度，参与收缩）。
    elite_count : int
        elite 成员计数（raw；收缩后见 ShrunkLogicStats.shrunk_elite_rate）。
    median_factor_fitness / median_long_short_quality / median_stability :
        float | None
        成员因子细分事实的中位数（raw；小样本向全局先验收缩）。
    historical_survival : float | None
        **frozen 上一研究版本**的 survival 事实（#23）。当前版本 sealed test
        survival 绝不进当前版本 logic reward。
    crowding : float | None
        拥挤度 ∈ [0, 1]。
    successful_implementations / failed_implementations : int
        该 logic 被尝试实现的成功/失败实现数（实现层计数）。
    """

    logic_id: str
    signature: LogicSignature
    logic_version: str = LOGIC_ID_VERSION
    logic_text: str = ""
    aliases: tuple[str, ...] = ()
    member_factor_ids: tuple[str, ...] = ()
    reference_query: str = ""
    schema_distribution: Mapping[str, float] = field(default_factory=dict)
    support_count: int = 0
    elite_count: int = 0
    median_factor_fitness: float | None = None
    median_long_short_quality: float | None = None
    median_stability: float | None = None
    historical_survival: float | None = None
    crowding: float | None = None
    successful_implementations: int = 0
    failed_implementations: int = 0

    def __post_init__(self) -> None:
        derived = self.signature.derive_logic_id()
        if self.logic_id != derived:
            raise ValueError(
                f"logic_id {self.logic_id!r} 与签名派生 id {derived!r} 不一致："
                "logic_id 必须由 LogicSignature 确定性派生，禁止手工 id 绕过签名。"
            )
        if not self.logic_id.startswith("LOG_"):
            raise ValueError(f"logic_id must start with LOG_, got {self.logic_id!r}")
        object.__setattr__(
            self, "aliases", tuple(dict.fromkeys(str(a) for a in self.aliases if str(a).strip()))
        )
        object.__setattr__(
            self, "member_factor_ids",
            tuple(dict.fromkeys(str(f) for f in self.member_factor_ids if str(f).strip())),
        )
        object.__setattr__(
            self, "schema_distribution",
            dict(self.schema_distribution or {}),
        )
        for attr in ("support_count", "elite_count", "successful_implementations",
                     "failed_implementations"):
            v = int(getattr(self, attr) or 0)
            object.__setattr__(self, attr, max(0, v))

    @property
    def elite_rate(self) -> float | None:
        """raw elite 比例（无成员 → None，不编 0）。"""
        if self.support_count <= 0:
            return None
        return self.elite_count / self.support_count

    @property
    def saturation(self) -> float | None:
        """饱和 = support/(support+k)；无成员 → None。"""
        if self.support_count <= 0:
            return None
        return self.support_count / (self.support_count + DEFAULT_SHRINKAGE_K)

    def with_stats(
        self,
        *,
        support_count: int | None = None,
        elite_count: int | None = None,
        median_factor_fitness: float | None = None,
        median_long_short_quality: float | None = None,
        median_stability: float | None = None,
        historical_survival: float | None = None,
        crowding: float | None = None,
        successful_implementations: int | None = None,
        failed_implementations: int | None = None,
    ) -> "LogicNode":
        """统计字段更新的复制工厂（frozen：每次更新返回新节点）。"""
        return replace(
            self,
            support_count=(
                self.support_count if support_count is None else max(0, int(support_count))
            ),
            elite_count=(
                self.elite_count if elite_count is None else max(0, int(elite_count))
            ),
            median_factor_fitness=(
                self.median_factor_fitness if median_factor_fitness is None else median_factor_fitness
            ),
            median_long_short_quality=(
                self.median_long_short_quality
                if median_long_short_quality is None else median_long_short_quality
            ),
            median_stability=(
                self.median_stability if median_stability is None else median_stability
            ),
            historical_survival=(
                self.historical_survival if historical_survival is None else historical_survival
            ),
            crowding=(self.crowding if crowding is None else crowding),
            successful_implementations=(
                self.successful_implementations
                if successful_implementations is None else max(0, int(successful_implementations))
            ),
            failed_implementations=(
                self.failed_implementations
                if failed_implementations is None else max(0, int(failed_implementations))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logic_id": self.logic_id,
            "logic_version": self.logic_version,
            "signature": self.signature.as_dict(),
            "logic_text": self.logic_text,
            "aliases": list(self.aliases),
            "member_factor_ids": list(self.member_factor_ids),
            "reference_query": self.reference_query,
            "schema_distribution": dict(self.schema_distribution),
            "support_count": self.support_count,
            "elite_count": self.elite_count,
            "elite_rate": self.elite_rate,
            "median_factor_fitness": self.median_factor_fitness,
            "median_long_short_quality": self.median_long_short_quality,
            "median_stability": self.median_stability,
            "historical_survival": self.historical_survival,
            "crowding": self.crowding,
            "saturation": self.saturation,
            "successful_implementations": self.successful_implementations,
            "failed_implementations": self.failed_implementations,
        }


# ---------------------------------------------------------------------------
# 因子事实行（miner 输入契约；LogicLibrary 聚合消费）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorFact:
    """一条历史因子的逻辑矿输入事实（确定性来源，非 LLM 产物）。

    Parameters
    ----------
    factor_id : str
        因子在 FactorAssets 的全局 factor_id（唯一身份，不复制公式）。
    formula : str
        factor_engine DSL 公式文本（miner 用它算 DNA；library 不存公式）。
    mechanisms : tuple[str, ...]
        机制 DNA（dna.mechanisms_of 输出；多标签）。
    operator_motifs : tuple[str, ...]
        算子/算子 motif DNA（dna operators 或 ast_motifs）。
    field_families : tuple[str, ...]
        字段 family DNA（dna.field_families_of 输出）。
    horizon_bucket : str
        fast / medium / slow / unknown。
    fitness / long_short_quality / stability : float | None
        评估事实（QE 口径由调用方保证；未知为 None——聚合时不编造）。
    elite : bool
        是否 elite（进入池/export）。
    schema_id : str | None
        可选 schema 归属（多标签映射到 logic 时的 schema 侧证据）。
    survival : SurvivalFact | None
        **frozen 上一研究版本**的 survival 事实（#23 隔离由 Library 强制）。
    """

    factor_id: str
    formula: str = ""
    mechanisms: tuple[str, ...] = ()
    operator_motifs: tuple[str, ...] = ()
    field_families: tuple[str, ...] = ()
    horizon_bucket: str = "unknown"
    fitness: float | None = None
    long_short_quality: float | None = None
    stability: float | None = None
    elite: bool = False
    schema_id: str | None = None
    survival: "SurvivalFact | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "formula": self.formula,
            "mechanisms": list(self.mechanisms),
            "operator_motifs": list(self.operator_motifs),
            "field_families": list(self.field_families),
            "horizon_bucket": self.horizon_bucket,
            "fitness": self.fitness,
            "long_short_quality": self.long_short_quality,
            "stability": self.stability,
            "elite": self.elite,
            "schema_id": self.schema_id,
            "survival": self.survival.to_dict() if self.survival is not None else None,
        }


@dataclass(frozen=True)
class SurvivalFact:
    """frozen 上一研究版本的 survival 事实（#23 版本隔离的承载）。"""

    version: str
    survival_rate: float | None = None
    support_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "survival_rate": self.survival_rate,
            "support_count": self.support_count,
        }


# ---------------------------------------------------------------------------
# 统计 / 收缩
# ---------------------------------------------------------------------------


def reliability_of(n_eff: int | float | None, *, k: float = DEFAULT_SHRINKAGE_K) -> float:
    """reliability = sqrt(n_eff/(n_eff+k))；n_eff<=0 → 0（收缩向中性）。"""
    n = float(n_eff) if n_eff is not None else 0.0
    n = max(0.0, n)
    if not math.isfinite(n):
        n = 0.0
    k = max(float(k), 1e-9)
    return math.sqrt(n / (n + k))


def shrink_mean(
    raw_mean: float | None, *, n_eff: int | float | None,
    prior: float = NEUTRAL_MEAN, k: float = DEFAULT_SHRINKAGE_K,
) -> float | None:
    """U_conf = prior + reliability * (raw - prior)；raw=None → None。

    小样本 → 向全局先验收缩（低 support 的 logic 的 elite_rate / 中位数等
    效应 → 偏移归零，绝不当最好）。n_eff=0 或 raw 无 → 中性 prior。
    raw 越界先 clip 到 [0, 1]（中位数在 0-1 口径；调用方保证量纲）。
    """
    if raw_mean is None or not math.isfinite(float(raw_mean)):
        return None
    r = reliability_of(n_eff, k=k)
    u = max(0.0, min(1.0, float(raw_mean)))
    return round(prior + r * (u - prior), 6)


def shrink_effect(
    raw_mean: float | None, *, n_eff: int | float | None,
    k: float = DEFAULT_SHRINKAGE_K,
) -> float:
    """对「相对中性先验的效应量」的收缩（供 survival/机会使用）。

    effect = (raw - 0.5) 经 reliability 缩放；低 support → 效应 → 0。
    原始值未知 → 0.0（不编造）。
    """
    if raw_mean is None or not math.isfinite(float(raw_mean)):
        return 0.0
    r = reliability_of(n_eff, k=k)
    # 不 round：消费方（survival/机会）要精确效应，round 会破坏公式一致性。
    return r * (float(raw_mean) - NEUTRAL_MEAN)


@dataclass
class ShrunkLogicStats:
    """LogicNode 的置信收缩后统计视图（检索/决策消费；Part G #29）。"""

    logic_id: str
    version: str = LOGIC_ID_VERSION
    support_count: int = 0
    shrunk_elite_rate: float | None = None
    shrunk_fitness: float | None = None
    shrunk_long_short_quality: float | None = None
    shrunk_stability: float | None = None
    shrunk_survival: float | None = None        # 只来自 frozen 上一版本 survival
    crowding: float | None = None
    saturation: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "logic_id": self.logic_id,
            "version": self.version,
            "support_count": self.support_count,
            "shrunk_elite_rate": self.shrunk_elite_rate,
            "shrunk_fitness": self.shrunk_fitness,
            "shrunk_long_short_quality": self.shrunk_long_short_quality,
            "shrunk_stability": self.shrunk_stability,
            "shrunk_survival": self.shrunk_survival,
            "crowding": self.crowding,
            "saturation": self.saturation,
        }


# ---------------------------------------------------------------------------
# LogicLibrary（存储 / 检索 / 统计）
# ---------------------------------------------------------------------------

DEFAULT_GLOBAL_ELITE_PRIOR: float = NEUTRAL_MEAN
DEFAULT_GLOBAL_MEDIAN_PRIOR: float = NEUTRAL_MEAN


@dataclass
class LogicLibrary:
    """市场逻辑库（进程内 + SQLite 可选持久化）。

    职责：
    - ``upsert_node``：登记/更新一个 LogicNode（按 logic_id 幂等）；
    - ``record_membership``：概率多标签映射因子 → logic（affinity ∈ (0, 1]；
      同一因子可映射多个 logic）；
    - ``record_logic_outcome``：记录一次实现结果并聚合 logic 级统计
      （版本化；survival 版本隔离 #23）；
    - ``shrunk_stats_of``：向全局先验收缩的检索视图；
    - ``members_of`` / ``schemas_of``：成员/聚合统计检索（#24，只引用不物化）。

    确定性、零 LLM、零模型、零网络；FactorAssets 只经 reference_query /
    factor_id 引用（#24）。
    """

    current_research_version: str = "1"
    #: 全局先验（miner 可在全量样本上先算再注入；缺省中性）
    global_elite_prior: float = DEFAULT_GLOBAL_ELITE_PRIOR
    global_median_prior: float = DEFAULT_GLOBAL_MEDIAN_PRIOR
    shrinkage_k: float = DEFAULT_SHRINKAGE_K

    _nodes: dict[str, LogicNode] = field(default_factory=dict)
    _memberships: dict[str, dict[str, float]] = field(default_factory=dict)  # logic_id -> factor_id -> affinity
    _factor_logics: dict[str, dict[str, float]] = field(default_factory=dict)  # factor_id -> logic_id -> affinity
    _med: dict[str, list[float]] = field(default_factory=dict)  # "lid" | "lsq:lid" | "stb:lid" -> raw 值列表
    _lock: Any = field(default=None)

    def __post_init__(self) -> None:
        if self._lock is None:
            self._lock = threading.RLock()

    # -- 登记 ---------------------------------------------------------------

    def upsert_node(self, node: LogicNode) -> str:
        """登记/覆盖一个 LogicNode（同 logic_id 幂等）。

        display 字段（logic_text / aliases / reference_query）以最新为准；
        统计字段保留旧值除非新值显式非零/非 None（upsert 不静默重置已聚合
        统计）。
        """
        if not isinstance(node, LogicNode):
            raise TypeError("upsert_node requires a LogicNode")
        with self._lock:
            old = self._nodes.get(node.logic_id)
            if old is None:
                self._nodes[node.logic_id] = node
                return node.logic_id
            merged = replace(
                old,
                logic_text=node.logic_text or old.logic_text,
                aliases=tuple(dict.fromkeys(list(old.aliases) + list(node.aliases))),
                reference_query=node.reference_query or old.reference_query,
                member_factor_ids=tuple(
                    dict.fromkeys(list(old.member_factor_ids) + list(node.member_factor_ids))
                ),
                schema_distribution={**dict(old.schema_distribution), **dict(node.schema_distribution)},
                crowding=node.crowding if node.crowding is not None else old.crowding,
            )
            self._nodes[node.logic_id] = merged
            return node.logic_id

    def node(self, logic_id: str) -> LogicNode | None:
        return self._nodes.get(logic_id)

    def logic_ids(self) -> list[str]:
        return sorted(self._nodes)

    # -- 概率多标签成员映射 ---------------------------------------------------

    def record_membership(
        self,
        *,
        factor_id: str,
        logic_id: str,
        affinity: float,
        schema_id: str | None = None,
        weight: float = 1.0,
        elite: bool = False,
        fitness: float | None = None,
        long_short_quality: float | None = None,
        stability: float | None = None,
    ) -> None:
        """把一个因子概率/多标签映射到一个 logic 节点。

        - ``affinity`` ∈ (0, 1]：该因子属于该 logic 的置信；同一因子可映射到
          多个 logic（**不是一对一硬归属**）；
        - 同一 (factor_id, logic_id) 重复调用：取最大 affinity（幂等）；重复
          不再累加 support_count；
        - ``schema_id`` 提供时累计到该 logic 的 schema_distribution；
        - ``elite`` 为 True 时 elite_count +1（仅在首次映射时累计）；
        - fitness / lsq / stability 提供时进入该 logic 的中位原始值池。
        """
        if not factor_id or not logic_id:
            raise ValueError("factor_id and logic_id are required")
        aff = float(affinity)
        if not (0.0 < aff <= 1.0):
            raise ValueError(f"affinity must be in (0, 1], got {affinity!r}")
        with self._lock:
            node = self._nodes.get(logic_id)
            if node is None:
                raise KeyError(f"logic_id {logic_id!r} 未登记（先 upsert_node）")
            mem = self._memberships.setdefault(logic_id, {})
            first = factor_id not in mem
            prior = mem.get(factor_id)
            mem[factor_id] = max(aff, float(prior or 0.0))
            self._factor_logics.setdefault(factor_id, {})[logic_id] = mem[factor_id]
            w = max(0.0, float(weight))
            dist = dict(node.schema_distribution)
            if schema_id and w > 0:
                dist[schema_id] = float(dist.get(schema_id, 0.0)) + w
            # member_factor_ids 只做「成员锚点」（首次出现的去重记录）；排序/affinity
            # 由 membership 表表达。每次映射都刷新 member_ids（幂等去重）。
            member_ids = tuple(dict.fromkeys(list(node.member_factor_ids) + [factor_id]))
            if first:
                if w > 0:
                    support = node.support_count + 1
                else:
                    support = node.support_count
                elite_count = node.elite_count + (1 if elite else 0)
            else:
                support = node.support_count
                elite_count = node.elite_count
            if fitness is not None and math.isfinite(float(fitness)):
                self._med.setdefault(logic_id, []).append(float(fitness))
            if long_short_quality is not None and math.isfinite(float(long_short_quality)):
                self._med.setdefault(f"lsq:{logic_id}", []).append(float(long_short_quality))
            if stability is not None and math.isfinite(float(stability)):
                self._med.setdefault(f"stb:{logic_id}", []).append(float(stability))
            self._nodes[logic_id] = replace(
                node,
                support_count=support,
                elite_count=elite_count,
                member_factor_ids=member_ids,
                schema_distribution=dist,
                median_factor_fitness=_median(self._med.get(logic_id)),
                median_long_short_quality=_median(self._med.get(f"lsq:{logic_id}")),
                median_stability=_median(self._med.get(f"stb:{logic_id}")),
            )

    def memberships_of(self, logic_id: str) -> dict[str, float]:
        return dict(self._memberships.get(logic_id, {}))

    def logics_of(self, factor_id: str) -> dict[str, float]:
        return dict(self._factor_logics.get(factor_id, {}))

    # -- 实现结果 / 统计聚合（版本化） --------------------------------------

    def record_logic_outcome(
        self,
        *,
        logic_id: str,
        success: bool,
        elite: bool = False,
        fitness: float | None = None,
        long_short_quality: float | None = None,
        stability: float | None = None,
        survival: SurvivalFact | None = None,
        survival_source_version: str | None = None,
    ) -> ShrunkLogicStats:
        """记录一次实现结果并聚合 logic 级统计。

        survival 版本隔离（#23）：
        - 仅当 ``survival`` 提供且 ``survival_source_version`` 显式 < 当前研究
          版本（``current_research_version``）→ 计入 historical_survival；
        - 否则（未提供 / 版本 >= 当前版本）→ **不计入**（当前版本 sealed test
          survival 绝不进当前版本 logic reward）。
        """
        node = self._nodes.get(logic_id)
        if node is None:
            raise KeyError(f"logic_id {logic_id!r} 未登记（先 upsert_node）")
        with self._lock:
            if success:
                n_success = node.successful_implementations + 1
                n_failed = node.failed_implementations
            else:
                n_success = node.successful_implementations
                n_failed = node.failed_implementations + 1
            support = node.support_count + 1
            elite_count = node.elite_count + (1 if elite else 0)
            if fitness is not None and math.isfinite(float(fitness)):
                self._med.setdefault(logic_id, []).append(float(fitness))
            if long_short_quality is not None and math.isfinite(float(long_short_quality)):
                self._med.setdefault(f"lsq:{logic_id}", []).append(float(long_short_quality))
            if stability is not None and math.isfinite(float(stability)):
                self._med.setdefault(f"stb:{logic_id}", []).append(float(stability))
            hist_surv = node.historical_survival
            if survival is not None and survival.survival_rate is not None:
                if (
                    survival_source_version is not None
                    and str(survival_source_version) < str(self.current_research_version)
                ):
                    hist_surv = max(0.0, min(1.0, float(survival.survival_rate)))
            self._nodes[logic_id] = node.with_stats(
                support_count=support,
                elite_count=elite_count,
                median_factor_fitness=_median(self._med.get(logic_id)),
                median_long_short_quality=_median(self._med.get(f"lsq:{logic_id}")),
                median_stability=_median(self._med.get(f"stb:{logic_id}")),
                historical_survival=hist_surv,
                successful_implementations=n_success,
                failed_implementations=n_failed,
            )
        return self.shrunk_stats_of(logic_id)

    # -- 检索 / 收缩视图 ----------------------------------------------------

    def shrunk_stats_of(self, logic_id: str) -> ShrunkLogicStats:
        """logic 统计的置信收缩视图（Part G #29）。

        - shrunk_elite_rate = shrink(elite_rate, n=support, prior=global_elite_prior)；
        - shrunk_fitness / lsq / stability = shrink(median, n=support, prior=global_median_prior)；
        - shrunk_survival = historical_survival（已版本隔离，#23）；
        - saturation = support/(support+k)（无成员 → None）。
        """
        node = self._nodes.get(logic_id)
        if node is None:
            raise KeyError(f"logic_id {logic_id!r} 未登记")
        n = node.support_count
        elite_rate = node.elite_count / n if n > 0 else None
        return ShrunkLogicStats(
            logic_id=logic_id,
            version=node.logic_version,
            support_count=n,
            shrunk_elite_rate=shrink_mean(
                elite_rate, n_eff=n, prior=self.global_elite_prior, k=self.shrinkage_k
            ),
            shrunk_fitness=shrink_mean(
                node.median_factor_fitness, n_eff=n,
                prior=self.global_median_prior, k=self.shrinkage_k,
            ),
            shrunk_long_short_quality=shrink_mean(
                node.median_long_short_quality, n_eff=n,
                prior=self.global_median_prior, k=self.shrinkage_k,
            ),
            shrunk_stability=shrink_mean(
                node.median_stability, n_eff=n,
                prior=self.global_median_prior, k=self.shrinkage_k,
            ),
            shrunk_survival=(
                max(0.0, min(1.0, float(node.historical_survival)))
                if node.historical_survival is not None else None
            ),
            crowding=node.crowding,
            saturation=node.saturation,
        )

    def summary(self) -> list[dict[str, Any]]:
        return [
            {"logic_id": lid, **self.shrunk_stats_of(lid).as_dict()}
            for lid in sorted(self._nodes)
        ]

    def members_of(self, logic_id: str, *, k: int | None = None) -> list[str]:
        """logic 的成员因子 id（按 affinity 降序）。

        - 显式 ``member_factor_ids``（miner 成桶锚点）语义上只是提示，排序/
          归属仍以 membership 表 affinity 为准；无 membership 记录时回退
          member_factor_ids 原序。
        - reference query 只存不物化（#24）。
        """
        mem = self.memberships_of(logic_id)
        if mem:
            ids = sorted(mem, key=lambda fid: (-float(mem[fid]), fid))
        else:
            node = self._nodes.get(logic_id)
            ids = list(node.member_factor_ids) if node is not None else []
        return ids[:k] if k is not None else ids

    def schemas_of(self, logic_id: str) -> dict[str, float]:
        node = self._nodes.get(logic_id)
        return dict(node.schema_distribution) if node is not None else {}

    # -- persistence（可选 SQLite；同 memory/ledger 风格） -------------------

    def persist_to(self, db_path: str) -> None:
        import os
        import sqlite3
        from pathlib import Path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(_SCHEMA_SQL)
            for lid, node in self._nodes.items():
                conn.execute(
                    "INSERT OR REPLACE INTO logic_nodes(logic_id, logic_version,"
                    " signature_json, logic_text, aliases_json, member_ids_json,"
                    " reference_query, schema_dist_json, support_count, elite_count,"
                    " median_fitness, median_lsq, median_stability, historical_survival,"
                    " crowding, n_success, n_failed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        lid, node.logic_version,
                        json.dumps(node.signature.as_dict(), sort_keys=True),
                        node.logic_text,
                        json.dumps(list(node.aliases)),
                        json.dumps(list(node.member_factor_ids)),
                        node.reference_query,
                        json.dumps(dict(node.schema_distribution), sort_keys=True),
                        node.support_count, node.elite_count,
                        node.median_factor_fitness, node.median_long_short_quality,
                        node.median_stability, node.historical_survival,
                        node.crowding, node.successful_implementations,
                        node.failed_implementations,
                    ),
                )
            for logic_id, mem in self._memberships.items():
                for fid, aff in mem.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO logic_membership(logic_id, factor_id, affinity)"
                        " VALUES (?,?,?)",
                        (logic_id, fid, float(aff)),
                    )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def load_from(cls, db_path: str, **kwargs: Any) -> "LogicLibrary":
        import sqlite3

        lib = cls(**kwargs)
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT logic_id, logic_version, signature_json, logic_text,"
                " aliases_json, member_ids_json, reference_query, schema_dist_json,"
                " support_count, elite_count, median_fitness, median_lsq,"
                " median_stability, historical_survival, crowding, n_success, n_failed"
                " FROM logic_nodes"
            ).fetchall()
            for r in rows:
                sig = LogicSignature.from_mapping(json.loads(r[2] or "{}"))
                node = LogicNode(
                    logic_id=r[0],
                    signature=sig,
                    logic_version=r[1] or LOGIC_ID_VERSION,
                    logic_text=r[3] or "",
                    aliases=tuple(json.loads(r[4] or "[]")),
                    member_factor_ids=tuple(json.loads(r[5] or "[]")),
                    reference_query=r[6] or "",
                    schema_distribution=json.loads(r[7] or "{}"),
                    support_count=int(r[8] or 0),
                    elite_count=int(r[9] or 0),
                    median_factor_fitness=r[10],
                    median_long_short_quality=r[11],
                    median_stability=r[12],
                    historical_survival=r[13],
                    crowding=r[14],
                    successful_implementations=int(r[15] or 0),
                    failed_implementations=int(r[16] or 0),
                )
                lib._nodes[node.logic_id] = node
            mrows = conn.execute(
                "SELECT logic_id, factor_id, affinity FROM logic_membership"
            ).fetchall()
            for logic_id, fid, aff in mrows:
                lib._memberships.setdefault(logic_id, {})[fid] = float(aff)
                lib._factor_logics.setdefault(fid, {})[logic_id] = float(aff)
        finally:
            conn.close()
        return lib


def _median(values: list[float] | None) -> float | None:
    if not values:
        return None
    return statistics.median(values)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS logic_nodes (
    logic_id TEXT PRIMARY KEY,
    logic_version TEXT NOT NULL,
    signature_json TEXT NOT NULL,
    logic_text TEXT,
    aliases_json TEXT,
    member_ids_json TEXT,
    reference_query TEXT,
    schema_dist_json TEXT,
    support_count INTEGER,
    elite_count INTEGER,
    median_fitness REAL,
    median_lsq REAL,
    median_stability REAL,
    historical_survival REAL,
    crowding REAL,
    n_success INTEGER,
    n_failed INTEGER
);
CREATE TABLE IF NOT EXISTS logic_membership (
    logic_id TEXT NOT NULL,
    factor_id TEXT NOT NULL,
    affinity REAL NOT NULL,
    PRIMARY KEY (logic_id, factor_id)
);
CREATE INDEX IF NOT EXISTS idx_lm_factor ON logic_membership(factor_id);
"""

__all__ = [
    "LOGIC_ID_VERSION",
    "LOGIC_SIGNATURE_DIMENSIONS",
    "DEFAULT_SHRINKAGE_K",
    "NEUTRAL_MEAN",
    "LogicSignatureError",
    "LogicSignature",
    "canonical_logic_signature",
    "derive_logic_id",
    "logic_signature_match",
    "LogicNode",
    "FactorFact",
    "SurvivalFact",
    "reliability_of",
    "shrink_mean",
    "shrink_effect",
    "ShrunkLogicStats",
    "LogicLibrary",
]

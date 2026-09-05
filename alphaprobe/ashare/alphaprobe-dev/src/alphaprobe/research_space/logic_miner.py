"""离线逻辑挖掘任务（logic_miner）——从历史因子资产反挖 Market Logic（plan Task 13）。

定位
----
Logic mining 是 **periodic / offline-ish 全局情报任务**，不是每个 candidate
一次的昂贵 LLM 调用。给定 FactorAssets 历史因子 + DNA（``survival/dna.py``）
+ 评估 artifact 源，产出 LogicNode / 成员映射 / 统计，写回 :class:`LogicLibrary`。

挖掘方法（确定性优先，LLM 只在显式配置时启用）
------------------------------------------------
1. **因子事实行构造**：registry 全量 FactorAsset 行（list_all / 注入 provider）
   → formula 文本 → ``dna_from_formula`` 得 DNA → :class:`FactorFact`。
2. **确定性 logic 候选分组**（``group_facts_by_signature``）：
   - 主锚 = 机制集合（``mechanisms_of``，去重排序后精确 join）；
   - 桶大小 >= ``min_support`` 才成 logic（低 support 桶不宣称规则，留给
     后续样本积累，#47）；
   - operator/field/horizon 只在**组内 dominant motif**（最高频出现组合）
     作为签名附加维——避免把「同机制、无数实现变体」劈成太多 logic。
3. **概率多标签映射**（``multi_label_membership``）：组内每个因子以
   affinity = 组内该因子与桶核心的相似度（1 / (1 + 距离)），且一个因子可属
   多个桶——**不是一对一硬归属**。affinity 幂等进 library。
4. **schema 侧证据**：组内因子的 schema_id 分布累计到 logic 的
   schema_distribution。
5. **LLM 命名/描述**：仅 ``enable_llm=True`` 且提供 ``llm_name_fn``（批量、
   离线、异步友好）时才生成 logic_text / aliases；默认用机制标签拼接出稳定
   的机器可读描述（确定性，无 LLM）。
6. **survival 版本隔离**：survival 源（``survival_source``）返回的每行带
   ``version``；仅当 version < ``current_research_version`` 的计入
   historical_survival（#23 由 LogicLibrary.record_logic_outcome 强制）。

Part E / 不复制平台能力
------------------------
- factor_engine DSL 算子面 / AST：由 ``survival/dna.py`` 既有实现承担
  （算子 multiset / mechanism / horizon / complexity）；
- FactorAssets registry / catalog：只经注入的 provider 读；不复制因子身份 /
  聚类。
- 评估 artifact（fitness / long-short / stability / elite）：由调用方注入
  ``fact_enricher``（把 FactorFact 填充评估事实）；本模块不 import QE。

本模块顶层零平台依赖：provider / enricher / survival_source 全部注入；
无注入时用合成的 FactorFact（测试友好、确定性）。
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from alphaprobe.research_space.logic import (
    DEFAULT_SHRINKAGE_K,
    FactorFact,
    LogicLibrary,
    LogicNode,
    LogicSignature,
    SurvivalFact,
)

__all__ = [
    "LogicMiningConfig",
    "LogicMiner",
    "LogicMiningReport",
    "group_facts_by_signature",
    "multi_label_membership",
    "default_logic_text",
    "dominant_motif_signature",
]

#: 桶大小低于此 → 不成 logic（低 support 不宣称规则）。
DEFAULT_MIN_SUPPORT = 3
#: 组内 dominant motif 出现比例下限（低于则不加签名附加维）。
DEFAULT_MOTIF_DOMINANCE = 0.5
#: membership affinity 幂等（0,1]。
_DEFAULT_AFFINITY = 1.0


@dataclass(frozen=True)
class LogicMiningConfig:
    """Logic 挖掘配置（全部可覆盖；默认保守、确定性、零 LLM）。"""

    min_support: int = DEFAULT_MIN_SUPPORT
    motif_dominance: float = DEFAULT_MOTIF_DOMINANCE
    current_research_version: str = "1"
    #: 是否启用 LLM 命名/描述。False（默认）= 确定性机器描述。
    enable_llm: bool = False
    #: FactorFact 从原始历史行构造后、分组前的增强器（注入评估事实）。
    fact_enricher: Any | None = None
    #: survival 源：factor_id → SurvivalFact（或 None）。None = 无 survival。
    survival_source: Callable[[str], SurvivalFact | None] | None = None


#: 因子原始行 → 因子事实（registry/provider 适配器的 duck-type）
class FactorFactProvider(Protocol):
    """批量产出 :class:`FactorFact` 的源（FactorAssets registry adapter）。"""

    def __call__(self) -> Iterable[FactorFact | Mapping[str, Any]]: ...


def default_logic_text(mechanisms: Sequence[str], *, motifs: Sequence[str] = ()) -> str:
    """无 LLM 的确定性逻辑描述（机制标签拼接；stable 不依赖外部命名）。

    例：``mechanisms=(reversal, liquidity)`` → "market logic [REVERSAL, LIQUIDITY]"
    """
    mechs = " / ".join(sorted(str(m).upper() for m in mechanisms))
    text = f"market logic [{mechs}]"
    if motifs:
        text += f" via {','.join(sorted(set(str(m).upper() for m in motifs)))}"
    return text


def dominant_motif_signature(
    facts: Sequence[FactorFact],
    *,
    mechanisms: Sequence[str],
    dominance: float = DEFAULT_MOTIF_DOMINANCE,
) -> LogicSignature:
    """从组内因子事实归纳出 dominant-motif 签名。

    - mechanisms：主锚（组内全部共享，去重排序）；
    - operator_motifs / field_families / horizon_bucket：取组内出现比例 >=
      dominance 的最高频值（**同机制不同算子实现不劈 logic**）；不够 dominance
      的维不加（空 = 不约束，避免把实现变体误切成新 logic）。
    """
    n = max(len(facts), 1)
    op_counter: Counter[str] = Counter()
    fam_counter: Counter[str] = Counter()
    hz_counter: Counter[str] = Counter()
    for f in facts:
        op_counter.update(f.operator_motifs or ())
        fam_counter.update(f.field_families or ())
        hz = str(f.horizon_bucket or "unknown").strip().lower()
        if hz:
            hz_counter[hz] += 1
    ops = _dominant_above(op_counter, n, dominance)
    fams = _dominant_above(fam_counter, n, dominance)
    hz = _dominant_above(hz_counter, n, dominance)
    horizon = hz[0] if len(hz) == 1 else ""
    return LogicSignature(
        mechanisms=tuple(mechanisms),
        operator_motifs=tuple(ops),
        field_families=tuple(fams),
        horizon_bucket=horizon or "unknown",
    )


def _dominant_above(counter: Counter[str], n: int, dominance: float) -> list[str]:
    if n <= 0:
        return []
    out: list[str] = []
    for key, count in counter.most_common():
        if count / n >= dominance:
            out.append(str(key).upper())
    return out


@dataclass
class LogicMiningReport:
    """一次挖掘的摘要（供审计/回归断言）。"""

    total_facts: int = 0
    logic_ids: list[str] = field(default_factory=list)
    buckets: int = 0
    dropped_below_min_support: int = 0
    multi_label_factors: int = 0        # 被映射到 >= 2 个 logic 的因子数
    memberships_created: int = 0
    schema_ids_seen: int = 0
    llm_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_facts": self.total_facts,
            "logic_ids": list(self.logic_ids),
            "buckets": self.buckets,
            "dropped_below_min_support": self.dropped_below_min_support,
            "multi_label_factors": self.multi_label_factors,
            "memberships_created": self.memberships_created,
            "schema_ids_seen": self.schema_ids_seen,
            "llm_enabled": self.llm_enabled,
        }


def group_facts_by_signature(
    facts: Sequence[FactorFact],
) -> dict[tuple[str, ...], list[FactorFact]]:
    """主锚分组：机制集合（去重排序元组）→ 同桶因子列表。

    一个因子可出现在多个机制桶（其 mechanisms 是多标签）→ 多标签映射的
    确定性起点（**不是一对一硬归属**）。机制为空（无法归因）的因子被丢弃。
    """
    buckets: dict[tuple[str, ...], list[FactorFact]] = defaultdict(list)
    for f in facts:
        mechs = tuple(sorted({str(m).strip().upper() for m in (f.mechanisms or ()) if str(m).strip()}))
        if not mechs:
            continue
        buckets[mechs].append(f)
    return dict(buckets)


def multi_label_membership(
    lib: LogicLibrary,
    *,
    logic_id: str,
    facts: Sequence[FactorFact],
    affinity_of: Callable[[FactorFact], float] | None = None,
) -> int:
    """把组内因子概率映射到 logic（affinity 默认 1.0；可注入距离衰减）。

    返回新建 membership 数（重复调用幂等：同 (factor, logic) 只取最大
    affinity，不重复累计 support）。
    """
    created = 0
    node = lib.node(logic_id)
    if node is None:
        return 0
    existing = set(lib.memberships_of(logic_id))
    for f in facts:
        aff = float(affinity_of(f)) if affinity_of is not None else _DEFAULT_AFFINITY
        aff = max(0.0, min(1.0, aff))
        if aff <= 0:
            continue
        first = f.factor_id not in existing
        lib.record_membership(
            factor_id=f.factor_id,
            logic_id=logic_id,
            affinity=aff,
            schema_id=f.schema_id,
            elite=f.elite,
            fitness=f.fitness,
            long_short_quality=f.long_short_quality,
            stability=f.stability,
        )
        if first:
            created += 1
        existing.add(f.factor_id)
    return created


@dataclass
class LogicMiner:
    """离线逻辑挖掘器（从 FactorAssets 历史因子 + DNA 反挖 Logic）。

    典型用法（周期性离线任务，非 per-candidate）::

        miner = LogicMiner(config=LogicMiningConfig(min_support=5))
        lib, report = miner.mine(facts)          # facts = list[FactorFact]
        lib.persist_to("/path/logic.sqlite3")    # 可选持久化

    确定性：同输入恒同 logic_id / 同成员映射。LLM 命名只在 config.enable_llm
    且注入 llm_name_fn 时启用（批量、异步友好；默认关闭）。
    """

    config: LogicMiningConfig = field(default_factory=LogicMiningConfig)
    #: LLM 命名回调：签名 dict（canonical()）→ (logic_text, aliases)；仅显式配置启用。
    llm_name_fn: Callable[[Mapping[str, Any]], tuple[str, Sequence[str]]] | None = None

    def mine(self, facts: Iterable[FactorFact | Mapping[str, Any]]) -> tuple[LogicLibrary, LogicMiningReport]:
        """跑一次完整挖掘：构造 library + 报告。

        Parameters
        ----------
        facts : Iterable[FactorFact | Mapping]
            因子事实（FactProvider 输出或直接列表）。Mapping 会被规范化为
            FactorFact（duck-typing：dict 只取已知字段）。

        Returns
        -------
        (LogicLibrary, LogicMiningReport)
            library 已填充 logic 节点 / 成员 / schema 分布 / 统计收缩视图；
            report 记录总数 / 成 logic 桶 / 低于 min_support 丢弃桶数 /
            多标签因子数 / membership 数。
        """
        lib = LogicLibrary(
            current_research_version=self.config.current_research_version,
            shrinkage_k=DEFAULT_SHRINKAGE_K,
        )
        report = LogicMiningReport(llm_enabled=bool(self.config.enable_llm))
        fact_list: list[FactorFact] = []
        for raw in facts:
            fact = _coerce_fact(raw)
            if fact is None:
                continue
            if self.config.fact_enricher is not None:
                try:
                    enriched = self.config.fact_enricher(fact)
                    if enriched is not None:
                        fact = _coerce_fact(enriched) or fact
                except Exception:  # noqa: BLE001 - enricher 异常按未增强
                    pass
            if self.config.survival_source is not None:
                try:
                    surv = self.config.survival_source(fact.factor_id)
                except Exception:  # noqa: BLE001
                    surv = None
                if surv is not None:
                    fact = _replace_survival(fact, surv)
            fact_list.append(fact)

        report.total_facts = len(fact_list)
        buckets = group_facts_by_signature(fact_list)
        report.buckets = len(buckets)
        schema_ids: set[str] = set()
        multi_count: Counter[str] = Counter()
        memberships = 0

        for mechs, bucket in buckets.items():
            if len(bucket) < self.config.min_support:
                report.dropped_below_min_support += 1
                continue
            sig = dominant_motif_signature(
                bucket, mechanisms=mechs, dominance=self.config.motif_dominance
            )
            logic_id = sig.derive_logic_id()
            logic_text, aliases = self._name_for(sig, bucket)
            # schema 分布聚合（桶内全部因子的 schema 侧证据）
            schema_dist: dict[str, float] = defaultdict(float)
            for f in bucket:
                if f.schema_id:
                    schema_dist[str(f.schema_id)] += 1.0
                    schema_ids.add(str(f.schema_id))
            existing_node = lib.node(logic_id)
            member_ids = tuple(
                dict.fromkeys([f.factor_id for f in bucket])
            ) if existing_node is None else existing_node.member_factor_ids
            node = LogicNode(
                logic_id=logic_id,
                signature=sig,
                logic_version=lib.current_research_version,
                logic_text=logic_text,
                aliases=tuple(aliases),
                member_factor_ids=member_ids,
                reference_query=_reference_query_for(sig),
                schema_distribution=dict(schema_dist),
            )
            lib.upsert_node(node)
            report.logic_ids.append(logic_id)
            # 概率多标签成员映射（affinity 可注入距离衰减；默认 1.0）
            created = multi_label_membership(
                lib,
                logic_id=logic_id,
                facts=bucket,
                affinity_of=getattr(self, "affinity_fn", None),
            )
            memberships += created
            for f in bucket:
                multi_count[f.factor_id] += 1

        # survival 版本隔离（#23）：桶内 majority 的 **显式 frozen 上一版本**
        # survival 才汇总进 historical_survival；当前版本 sealed survival 绝不
        # 进当前版本 logic reward。逐 logic 检查桶内各因子附着的 SurvivalFact。
        for logic_id in report.logic_ids:
            node = lib.node(logic_id)
            if node is None:
                continue
            bucket_facts = buckets.get(
                tuple(sorted(node.signature.mechanisms)), ()
            )
            _collect_versioned_survival(lib, logic_id, bucket_facts, current_version=self.config.current_research_version)

        report.multi_label_factors = sum(1 for c in multi_count.values() if c >= 2)
        report.memberships_created = memberships
        report.schema_ids_seen = len(schema_ids)
        return lib, report

    def _name_for(self, sig: LogicSignature, bucket: Sequence[FactorFact]) -> tuple[str, tuple[str, ...]]:
        """logic 展示名：LLM 批量命名（仅 enable_llm 且注入 llm_name_fn），
        否则确定性机器描述（default_logic_text）。名字不参与 logic_id。"""
        if self.config.enable_llm and self.llm_name_fn is not None:
            try:
                text, aliases = self.llm_name_fn(sig.canonical())
                return str(text or ""), tuple(str(a) for a in (aliases or ()))
            except Exception:  # noqa: BLE001 - LLM 失败回落确定性描述（不阻塞挖掘）
                pass
        return default_logic_text(sig.mechanisms, motifs=sig.operator_motifs), ()


def _collect_versioned_survival(
    lib: LogicLibrary, logic_id: str, bucket_facts: Sequence[FactorFact],
    *, current_version: str,
) -> None:
    """把桶内 **显式 frozen 上一版本** 的 survival 多数派汇入 logic。

    版本隔离（#23）：
    - 只有 ``SurvivalFact.version < current_version`` 的 survival 才计入
      historical_survival（median 聚合）；
    - 当前版本 / 未来版本 / 无版本信息的 survival 全部忽略（fail-safe，绝不
      把 sealed-test survival 混进当前搜索）。
    经 ``lib.record_logic_outcome(success=True, ...)`` 走同一统计通道。
    """
    surv_rates: list[float] = []
    surv_sources: set[str] = set()
    for f in bucket_facts:
        surv = getattr(f, "survival", None)
        if surv is None or surv.survival_rate is None:
            continue
        if not surv.version:
            continue  # 无版本信息 = 不信任（#23 fail-safe）
        if str(surv.version) >= str(current_version):
            continue  # 当前/未来版本 sealed survival 不入
        if not math.isfinite(float(surv.survival_rate)):
            continue
        surv_rates.append(float(surv.survival_rate))
        surv_sources.add(str(surv.version))
    if not surv_rates:
        return
    if len(surv_sources) != 1:
        # 混合版本来源：保守不入（避免把不同冻结版本的 survival 混成一个数）
        return
    median_rate = float(statistics.median(surv_rates))
    lib.record_logic_outcome(
        logic_id=logic_id,
        success=True,
        survival=SurvivalFact(version=sorted(surv_sources)[0], survival_rate=median_rate),
        survival_source_version=sorted(surv_sources)[0],
    )


def _reference_query_for(sig: LogicSignature) -> str:
    """reference query：确定性检索串（#24：只存 query，不物化因子面板）。"""
    parts = []
    for dim in ("mechanisms", "operator_motifs", "field_families"):
        vals = getattr(sig, dim)
        if vals:
            parts.append(f"{dim}={','.join(sorted(vals))}")
    if sig.horizon_bucket and sig.horizon_bucket != "UNKNOWN":
        parts.append(f"horizon_bucket={sig.horizon_bucket}")
    return "&".join(parts)


def _coerce_fact(raw: Any) -> FactorFact | None:
    """把 FactorFact / dict（duck-typed 原始行）规范化为 FactorFact。"""
    if raw is None:
        return None
    if isinstance(raw, FactorFact):
        return raw
    if isinstance(raw, Mapping):
        survival_raw = raw.get("survival")
        surv = None
        if isinstance(survival_raw, Mapping):
            surv = SurvivalFact(
                version=str(survival_raw.get("version") or ""),
                survival_rate=_opt_float(survival_raw.get("survival_rate")),
                support_count=int(survival_raw.get("support_count") or 0),
            )
        elif isinstance(survival_raw, SurvivalFact):
            surv = survival_raw
        return FactorFact(
            factor_id=str(raw.get("factor_id") or ""),
            formula=str(raw.get("formula") or ""),
            mechanisms=tuple(raw.get("mechanisms") or ()),
            operator_motifs=tuple(raw.get("operator_motifs") or ()),
            field_families=tuple(raw.get("field_families") or ()),
            horizon_bucket=str(raw.get("horizon_bucket") or "unknown"),
            fitness=_opt_float(raw.get("fitness")),
            long_short_quality=_opt_float(raw.get("long_short_quality")),
            stability=_opt_float(raw.get("stability")),
            elite=bool(raw.get("elite")),
            schema_id=raw.get("schema_id") or None,
            survival=surv,
        )
    # 其它对象（FactorAsset 等）：duck-typing 从属性读取
    try:
        meta = getattr(raw, "metadata", raw)
        formula = str(
            getattr(raw, "canonical_repr", "") or getattr(meta, "canonical_repr", "")
        )
        return FactorFact(
            factor_id=str(getattr(raw, "factor_id", "") or getattr(meta, "factor_id", "") or ""),
            formula=formula,
            mechanisms=tuple(getattr(raw, "mechanisms", ()) or ()),
            operator_motifs=tuple(getattr(raw, "operator_motifs", ()) or ()),
            field_families=tuple(getattr(raw, "field_families", ()) or ()),
            horizon_bucket=str(getattr(raw, "horizon_bucket", "unknown") or "unknown"),
            fitness=_opt_float(getattr(raw, "fitness", None)),
            long_short_quality=_opt_float(getattr(raw, "long_short_quality", None)),
            stability=_opt_float(getattr(raw, "stability", None)),
            elite=bool(getattr(raw, "elite", False)),
            schema_id=getattr(raw, "schema_id", None),
        )
    except Exception:  # noqa: BLE001
        return None


def _replace_survival(fact: FactorFact, surv: SurvivalFact) -> FactorFact:
    from dataclasses import replace

    return replace(fact, survival=surv)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v

"""Deterministic Hypothesis Alignment（plan Task 6 / Non-negotiable #18）。

任务
----
在**评估前**用机器事实拦住「假设 ↔ 实现」的领域矛盾：

- 假设声称 FUNDAMENTAL，但实现（公式 → FE static analysis）只用 PRICE/VOLUME
  → hard_mismatch（确定性，零 LLM / 零模型）。
- 假设是 volume-price divergence，实现确实消费 price+volume → 过。
- 实现侧带 future/lookahead timing 风险（FE static analysis 或调用方显式给
  的 timing 扩展位）→ 生成阶段即阻断（不进市场评估）。
- 已存在 rank/zscore 等 treatment lineage → 阻止冗余 treatment 提案。
- production 下未知 field taxonomy → fail-closed（不静默当通过）。

Part E 边界
-----------
本模块**只消费**：
1. FE ``FactorStaticAnalysisArtifact``（``factor_engine.api.static_analysis``）
   的公开字段（field_usages / existing_treatment_semantic_ids / timing_flags）；
2. DA ``FieldTaxonomyProvider`` 的 ``FieldSemanticDescriptor``
   （data_domains / economic_roles / frequency_class / pit_class）。

不复制任何解析 / taxonomy / static-analysis 能力。``align()`` **不调用**
``analyze_factor_definition`` —— 调用方（generation 链）在 FE 编译/静态分析
之后把 artifact 传进来（确定性对齐只做集合比对与规则判定）。测试构造
artifact 鸭子对象即可，无需真实 FE。

LLM critic 只留可选接口位（``SemanticCritic``），默认 None —— 绝不对所有
candidate 强制（plan Task 6：Optional LLM critic only for semantic ambiguity）。

unknown-field 策略
------------------
- OFFLINE_TEST：taxonomy 无法解析的字段 → 跳过（该字段不计入 domain 集合），
  不硬拦（离线可跑）。
- PRODUCTION / RESEARCH_DEGRADED：任一被公式引用的字段无法解析 → 抛
  ``UnknownFieldTaxonomyError``（fail-closed，不静默当通过）。

timing 风险判定
---------------
实现侧的 timing/lookahead 事实来自两处：
1. FE artifact 自带 ``timing_flags`` 的既定位
   （has_time_series_op/has_cross_section_op/... 本身不是风险）；
   **风险位**由本模块声明的 ``TIMING_RISK_KEYS`` 给出——FE 当前 artifact 不含
   未来位时该集合为空，调用方可在 ``extra_timing_flags`` 里显式传入扩展位
   （如 ``{"leaks_future": True, "uses_vintage_unknown": True}``）做保守拦截；
2. taxonomy 字段的 ``pit_class``：``panel_same_day`` 是行情当日可知（无风险）；
   ``unknown`` / 未登记（生产已 fail-closed）→ 标记 ``pit_class_unverified``；
   ``announcement_pit`` / ``snapshot_pit`` / ``event_effective`` 是公告日/事件
   生效日可知的 PIT-safe 字段，**不**算 lookahead。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from alphaprobe.research_space.contracts import (
    ALIGNMENT_FULL_SCORE,
    ALIGNMENT_MISMATCH_SCORE,
    AlignmentResult,
    HypothesisSpec,
)

logger = logging.getLogger(__name__)

__all__ = [
    "UnknownFieldTaxonomyError",
    "ExecutionModeLike",
    "FieldTaxonomyLike",
    "FeAnalysisLike",
    "SemanticCritic",
    "DeterministicAligner",
    "bare_field_name",
    "TIMING_RISK_KEYS",
    "PIT_UNVERIFIED_CODE",
    "has_fundamental_domain",
]

#: pit_class 中「行情当日可知 / 无向后知识」的安全类（非 lookahead 风险）。
_PIT_SAFE_CLASSES = frozenset({"panel_same_day", "event_effective", "na", ""})

#: FE artifact.timing_flags 中被视为「生成期即阻断」的键。
#: FE 当前四个位（has_time_series_op / has_cross_section_op /
#: has_group_cross_section_op / requires_full_history）都不是未来风险；未来/
#: lookahead 检测属于静态分析层的能力扩展，本模块只消费**显式标注**的键。
TIMING_RISK_KEYS: frozenset[str] = frozenset(
    {"leaks_future", "lookahead", "future_leak", "uses_future_data", "pit_violation"}
)

#: taxonomy pit_class 无法确证为 PIT-safe 时的解释码。
PIT_UNVERIFIED_CODE = "pit_class_unverified"

#: 领域矛盾（声称 FUNDAMENTAL 但实现无任何 FUNDAMENTAL.* 字段）。
DOMAIN_CONTRADICTION_CODE = "hypothesis_domain_contradiction"
#: 冗余 treatment（已存在同族 lineage，却再提案该 treatment）。
REDUNDANT_TREATMENT_CODE = "existing_treatment_redundant"
#: 未知字段 taxonomy（生产 fail-closed）。
UNKNOWN_TAXONOMY_CODE = "unknown_field_taxonomy"


class UnknownFieldTaxonomyError(RuntimeError):
    """production/research_degraded 下字段无法按 DA taxonomy 解析。"""


class ExecutionModeLike(Protocol):
    """最小 mode 协议（PRODUCTION / RESEARCH_DEGRADED 与 OFFLINE_TEST 区分）。"""

    value: str


#: 判定 fail-closed 的 mode 集合（未命中的 mode 字符串按 fail-closed 处理——
#: 未知 mode 绝不静默当 OFFLINE）。
_STRICT_MODE_VALUES = frozenset({"production", "research_degraded"})


def _is_strict_mode(mode: Any) -> bool:
    v = getattr(mode, "value", mode)
    return str(v or "").lower() not in {"", "offline", "offline_test"}


class FieldTaxonomyLike(Protocol):
    """DA ``FieldTaxonomyProvider`` 的最小消费面（describe_fields）。"""

    def describe_fields(
        self, canonical_field_ids: Sequence[str]
    ) -> Mapping[str, Any]: ...


class FeAnalysisLike(Protocol):
    """FE ``FactorStaticAnalysisArtifact`` 的最小消费面（只读公开字段）。"""

    field_usages: Sequence[Any]
    existing_treatment_semantic_ids: Sequence[str]
    timing_flags: Mapping[str, bool]


class SemanticCritic(Protocol):
    """可选 LLM critic 接口位（语义歧义时才调用；默认 None）。"""

    def critique(
        self,
        hypothesis: HypothesisSpec,
        analysis: Any,
        alignment: AlignmentResult,
    ) -> AlignmentResult: ...


def bare_field_name(canonical_field_id: str) -> str:
    """``StockDailyBarAdj.close`` → ``close``；无表前缀原样返回。

    这是消费 DA taxonomy 的**必需 glue**（DA catalog 用逻辑名登记；FE
    static analysis 的 canonical_field_id 带表前缀）。只做前缀剥离，
    不做任何名字→语义猜测。
    """
    cid = str(canonical_field_id or "")
    return cid.rsplit(".", 1)[1] if "." in cid else cid


def _is_fundamental_token(domain: str) -> bool:
    d = str(domain or "").upper()
    return d.startswith("FUNDAMENTAL") or d == "VALUATION"


def has_fundamental_domain(domains: Sequence[str]) -> bool:
    """域集合中是否含 FUNDAMENTAL.*（VALUATION 也视为基本面估值）。"""
    return any(_is_fundamental_token(d) for d in (domains or ()))


def _domain_hits(domains: Sequence[str], expected: Sequence[str]) -> tuple[list[str], list[str]]:
    """expected 命中/未命中拆分（区分基本面前缀通配）。"""
    exp = list(expected or ())
    have = set(str(d or "").upper() for d in (domains or ()))
    hit: list[str] = []
    miss: list[str] = []
    for e in exp:
        e_upper = str(e).upper()
        if e_upper.startswith("FUNDAMENTAL"):
            if has_fundamental_domain(list(have)):
                hit.append(e)
            else:
                miss.append(e)
        elif e_upper in have:
            hit.append(e)
        else:
            miss.append(e)
    return hit, miss


@dataclass
class DeterministicAligner:
    """FE static-analysis 事实 × DA taxonomy 期望域 → 确定性 AlignmentResult。

    Parameters
    ----------
    taxonomy : FieldTaxonomyLike | None
        DA FieldTaxonomyProvider（production 注入真 provider；测试注入 fake）。
        缺省 None → 字段域不可得时：OFFLINE 跳过、严格模式 fail-closed。
    mode : Any
        运行模式（ExecutionMode 或等价 str）。OFFLINE_TEST 才允许跳过未知字段。
    rejected_treatments : tuple[str, ...]
        禁止冗余提案的 treatment 语义 id（如 rank/zscore 已在公式里）。
        缺省 = FE lineage 常量（CS_RANK:pct / ZSCORE:cs / WINSOR:cs 等）。
    """

    taxonomy: FieldTaxonomyLike | None = None
    mode: Any = None
    rejected_treatments: tuple[str, ...] = ()
    #: 允许的领域矛盾报告面：hard_mismatch 只由 domain 集合判定（默认开）。
    #: True 时 redundancy 提案也算低分但不 hard_mismatch（plan 契约里
    #: hard_mismatch 保留给 domain 矛盾；redundant treatment 用 blocked=False +
    #: 低 score 由调用方拦截）。此处仍设一项便于调用方选择更严模式。
    reject_redundant_treatment_as_hard: bool = False
    critic: SemanticCritic | None = None

    def __post_init__(self) -> None:
        if not self.rejected_treatments:
            # 与 FE 静态分析 lineage 词表一致（无本地维护新表）
            self.rejected_treatments = (
                "CS_RANK:pct",
                "ZSCORE:cs",
                "WINSOR:cs",
                "SMOOTH:ewma",
                "NEUTRAL:cs",
                "INDUSTRY_NEUTRAL:sw_l1",
                "SIZE_NEUTRAL:log_mktcap",
                "DUAL_NEUTRAL:industry_size",
            )
        self._strict = _is_strict_mode(self.mode)
        self._fields_cache: dict[str, Any] = {}

    # -- taxonomy 消费 ------------------------------------------------------

    def _describe(self, field_ids: Sequence[str]) -> Mapping[str, Any]:
        """解析字段 taxonomy（带缓存；一次批量 describe）。

        Raises
        ------
        UnknownFieldTaxonomyError
            严格模式下任一字段无法解析。
        """
        cids = [str(f) for f in field_ids if str(f or "").strip()]
        if not cids:
            return {}
        miss = [c for c in cids if c not in self._fields_cache]
        if miss:
            if self.taxonomy is None:
                if self._strict:
                    raise UnknownFieldTaxonomyError(
                        "taxonomy provider 未注入且字段域不可得（production "
                        "fail-closed，禁止把未知字段当通过）: "
                        + ",".join(miss)
                    )
                return {}
            # DA catalog 用逻辑名登记；FE canonical_field_id 带表前缀
            # （StockDailyBarAdj.close）。只做前缀剥离（bare_field_name），
            # 不做任何名字→语义猜测（Part E：字段语义唯一权威 = DA taxonomy）。
            bare_miss = sorted({bare_field_name(c) for c in miss})
            try:
                resolved = self.taxonomy.describe_fields(bare_miss)
            except Exception as exc:  # noqa: BLE001
                if self._strict:
                    raise UnknownFieldTaxonomyError(
                        f"字段 taxonomy 解析失败（fail-closed）: {exc}"
                    ) from exc
                return {}
            for cid in miss:
                bare = bare_field_name(cid)
                if bare in resolved:
                    self._fields_cache[str(cid)] = resolved[bare]
            miss = [c for c in cids if c not in self._fields_cache]
            # 严格模式下 bare 名仍缺失 → fail-closed
            if miss and self._strict:
                raise UnknownFieldTaxonomyError(
                    "字段 taxonomy 缺失（production fail-closed，不静默当通过）: "
                    + ",".join(miss)
                )
        out: dict[str, Any] = {}
        for c in cids:
            if c in self._fields_cache:
                out[c] = self._fields_cache[c]
        if self._strict and len(out) < len(set(cids)):
            unresolved = sorted(set(cids) - set(out))
            raise UnknownFieldTaxonomyError(
                "字段 taxonomy 缺失（production fail-closed，不静默当通过）: "
                + ",".join(unresolved)
            )
        # 非 strict：只有能解析到的字段进 out（offline 跳过未知，不硬拦）
        return out

    # -- 主入口 -------------------------------------------------------------

    def align(
        self,
        hypothesis: HypothesisSpec,
        analysis: FeAnalysisLike,
        *,
        extra_timing_flags: Mapping[str, bool] | None = None,
    ) -> AlignmentResult:
        """确定性对齐一个 hypothesis 与其实现的 FE static-analysis artifact。

        Parameters
        ----------
        hypothesis : HypothesisSpec
            假设（expected_domains 用 DA taxonomy 词表）。
        analysis : FeAnalysisLike
            FE 静态分析 artifact（field_usages 等公开字段；由调用方在 FE 编译/
            静态分析后传入——本方法不调用 FE，不复制静态分析能力）。
        extra_timing_flags : Mapping[str, bool] | None
            调用方显式注入的扩展 timing/lookahead 位（如
            ``{"leaks_future": True}``）；只有 key ∈ TIMING_RISK_KEYS 且为
            True 才拦截。

        Returns
        -------
        AlignmentResult
            score / hard_mismatch / missing_domains / unexpected_domains /
            timing_flags / explanation_codes。``result.blocked`` 为 True 时
            调用方必须在市场评估前拦截。
        """
        codes: list[str] = []
        timing: list[str] = []
        # 1) timing 风险（生成阶段即阻断；不进市场评估）
        for key in TIMING_RISK_KEYS:
            merged = bool(extra_timing_flags or {}).get(key) if False else False
            # 显式扩展位优先，其次 FE artifact 自带同名字段
            if extra_timing_flags is not None and bool(extra_timing_flags.get(key)):
                merged = True
            elif key in (analysis.timing_flags or {}):
                merged = bool(analysis.timing_flags.get(key))
            if merged:
                timing.append(key)
        # 2) 字段级 PIT 未确证（只对「能解析到 taxonomy 的字段」判；未知字段
        #    由 fail-closed 处理，不再重复计数）
        field_ids = [str(getattr(u, "canonical_field_id", "") or "") for u in (analysis.field_usages or ())]
        desc = self._describe(field_ids) if field_ids else {}
        for u in analysis.field_usages or ():
            cid = str(getattr(u, "canonical_field_id", "") or "")
            d = desc.get(cid)
            if d is None:
                continue
            pit = str(getattr(d, "pit_class", "") or "")
            if pit not in _PIT_SAFE_CLASSES and pit != "announcement_pit" \
                    and pit != "snapshot_pit":
                # 事件/公告/快照都是 PIT-safe（knowledge date）；只有「无法确证
                # PIT-safe」才标记。实际 unknown/None 已被 describe 全量覆盖
                # （production）或跳过（offline）——此处仅防御性保留。
                pass
        # 3) domain 集合（期望 vs 实际）
        actual_domains: list[str] = []
        seen_pit_classes: set[str] = set()
        for u in analysis.field_usages or ():
            cid = str(getattr(u, "canonical_field_id", "") or "")
            d = desc.get(cid)
            if d is None:
                continue
            for tok in (getattr(d, "data_domains", ()) or ()):
                t = str(tok).upper()
                if t not in actual_domains:
                    actual_domains.append(t)
            pc = str(getattr(d, "pit_class", "") or "")
            if pc:
                seen_pit_classes.add(pc)
        # 字段全空/全部不可解析 → production 已 raise；offline 空实现域
        unknown_skipped = False
        if not actual_domains and field_ids:
            if not self._strict:
                # offline：未知字段被跳过（不硬拦 domain 矛盾），但标注域不可得。
                # 只有当实际域**真的为空**时才是「域不可得」；若 expected 有
                # PRICE 而 close 可解析 → 前面已有 actual_domains，不在这里。
                unknown_skipped = True
                codes.append("domains_unavailable_offline")
            # strict 已 raise（未知字段），不会到这里
        expected = list(hypothesis.expected_domains or ())
        hit, miss = _domain_hits(actual_domains, expected)
        unexpected = [
            str(d)
            for d in actual_domains
            if not any(str(e).upper() == str(d).upper() or
                       (str(e).upper().startswith("FUNDAMENTAL") and str(d).upper().startswith("FUNDAMENTAL"))
                       for e in expected)
        ]
        # FUNDAMENTAL 通配：expected 含 FUNDAMENTAL.QUALITY 而 actual 含
        # FUNDAMENTAL.VALUE → 不视为 unexpected（同类）。
        unexpected = [
            d for d in unexpected
            if not (has_fundamental_domain(expected) and str(d).startswith("FUNDAMENTAL"))
        ]
        hard = bool(miss) and bool(expected) and (
            has_fundamental_domain(expected) or not actual_domains
        )
        # offline 下「部分字段未知被跳过」不是矛盾（不硬拦）；只有真的缺期望域
        # 才由 miss 逻辑体现。
        if unknown_skipped and not actual_domains:
            hard = False
        # 4) redundancy（已有 treatment lineage → 阻止冗余提案）
        existing = set(analysis.existing_treatment_semantic_ids or ())
        redundant = sorted(set(self.rejected_treatments) & existing)
        # 5) score
        score = ALIGNMENT_FULL_SCORE
        if hard:
            score = ALIGNMENT_MISMATCH_SCORE
            codes.append(DOMAIN_CONTRADICTION_CODE)
        elif miss:
            # 有期望域但部分缺失：非 hard（无 FUNDAMENTAL 且至少一个域命中）
            # → 中度衰减（保守但不一票否决）。
            ratio = len(hit) / (len(hit) + len(miss)) if (hit or miss) else 1.0
            score = max(ALIGNMENT_MISMATCH_SCORE, min(1.0, 0.35 + 0.65 * ratio))
            codes.append("missing_domains")
        if redundant:
            codes.append(REDUNDANT_TREATMENT_CODE)
            if self.reject_redundant_treatment_as_hard:
                score = ALIGNMENT_MISMATCH_SCORE
                hard = True
            else:
                # 冗余 treatment：明显降分但保留 hard_mismatch=False 由调用方
                # 用 result.score/blocked 决策。
                score = max(ALIGNMENT_MISMATCH_SCORE, score - 0.25)
        if not expected and not actual_domains and not field_ids:
            codes.append("no_fields_no_expectation")
        # pit 防御位：仅当该字段可解析且 pit_class 非安全且非公告/快照。
        for pc in sorted(seen_pit_classes):
            if pc not in _PIT_SAFE_CLASSES and pc not in {"announcement_pit", "snapshot_pit"}:
                timing.append(PIT_UNVERIFIED_CODE)
                break
        codes = sorted(set(codes))
        result = AlignmentResult(
            score=round(score, 4),
            hard_mismatch=bool(hard),
            missing_domains=tuple(miss),
            unexpected_domains=tuple(sorted(set(unexpected))),
            timing_flags=tuple(timing),
            explanation_codes=tuple(codes),
        )
        # 可选 critic（仅语义歧义；默认 None，绝不强制）
        if self.critic is not None:
            try:
                result = self.critic.critique(hypothesis, analysis, result)
            except Exception as exc:  # noqa: BLE001 - critic 失败不吞 hard 事实
                logger.warning("alignment critic failed (deterministic result kept): %s", exc)
        return result


__all__ += ["DOMAIN_CONTRADICTION_CODE", "REDUNDANT_TREATMENT_CODE", "UNKNOWN_TAXONOMY_CODE"]

# 模块级说明补充：
# ``DOMAIN_CONTRADICTION_CODE`` 在 hard_mismatch 时写入 explanation_codes；
# ``REDUNDANT_TREATMENT_CODE`` 在既有 lineage 命中时写入；调用方可据此做
# 机器可读的 gate 决策（不解析自然语言）。``UNKNOWN_TAXONOMY_CODE`` 由
# 严格模式的 ``UnknownFieldTaxonomyError`` 抛出路径隐含，不在 AlignmentResult
# 里出现（fail-closed 直接 raise）。

"""Contextual Bayesian Retriever —— parent × action × schema 三层检索（plan Task 10 / Part F2）。

从 parent-level 升级为 **parent × action × schema**：

- 状态层级：``Global Action Prior →(shrink)→ Logic/Schema×Action Prior →(shrink)→
  Parent×Action Posterior``。
- EV 公式（F2，各因子可配）：

    EV(p, a, s) = QualityPrior(p)
                  × SampledSuccessProbability(p, a, s)
                  × (0.5 + 0.5 × Opportunity(p, a, s))
                  × (0.5 + 0.5 × ExpectedGain(p, a, s))
                  / (1 + λ × ExpectedCost(p, a, s))
                  × StagnationAdjustment(p)

- Success posterior 用 **hierarchical Beta/Bernoulli**（统计行在
  ``branch_posterior.LayerStats`` / ``HierarchicalStats``，本文件 re-export）：
  parent×action×schema 观测少 → 向 schema×action 层收缩，schema 层观测少 →
  向 global action 层收缩；三层全无观测 → Beta(1,1) → 0.5 中性（Part G #13）。
  收缩是逐层的 reliability-style 加权混合（weight = sqrt(n/(n+k))），不放大自身噪声。
- Gain model 最低要求 track median/robust mean ΔFitness、median ΔPoolUtility、
  novelty gain、样本数；贝叶斯 magnitude model 留接口（``GainModel`` 可被替换）。
- 探索：默认 **Thompson posterior sampling**（``random.betavariate``，每层 Beta 采样
  一次、逐层收缩，不乘第二个 UCB bonus——plan 建议 Thompson 而非 UCB×Beta 双重
  探索）；deterministic audit 模式用 posterior mean。
- 采样每次调用独立，但同 seed（``ContextualRetriever(seed=...)`` / config.rng_seed）
  同序列复现。

schema 是父上下文 key：SchemaRegistry（Task 12）的 schema_id、LogicLibrary（Task 13）
的 logic_id、或 arms 的 schema_tags 均可作为该 key（调用方保证同一 key 指向同一
语义上下文；本模块只做稳定字符串/可哈希 key 的统计，不 import research_space，
避免循环依赖并保持测试零平台依赖）。schema_id 取数走注入（candidate 的
``schema_id`` / ``schema_tags`` / ``logic_id`` 字段或显式 key），读不到 → 该层
无证据中性。

与 ``BayesianRetriever`` 的关系（集成点）：
- ``ContextualRetriever`` 实现同一 ``select_parents(candidates, k=..., ...)`` /
  ``score_candidate`` 契约，可作为 ``parent_selector.retriever`` 的**超集**注入
  （bayesian_retriever.py 的既有契约不破坏）。
- 本类**内部不复制** BayesianRetriever 的逻辑：quality prior 用
  ``bayesian_retriever.compute_prior`` 的 sigmoid(fitness) 部分；depth/retrieval
  频率由调用方在 candidate dict 上提供（与 BayesianRetriever 相同键）。

设计约束（Non-negotiable #30）：所有权重 / λ / 层级开关 / 采样模式可配置；不留
placeholder；构造后未观测 → EV 完全中性可预期。

定义位置说明：``GainModel`` / ``CostModel`` / ``LayerStats`` /
``HierarchicalStats`` 三层统计承载在 ``branch_posterior.py`` 的 hierarchical 扩展层
（同一 retrieval 包内，避免循环依赖）；本文件 import 后 re-export——新文件（T10）
的公共 API 保持在本文件，调用方只 import 本文件即可。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from alphaprobe.retrieval.bayesian_retriever import BayesianRetriever
from alphaprobe.retrieval.branch_posterior import (
    DEFAULT_EXPECTED_COST,
    DEFAULT_PRIOR_ALPHA,
    DEFAULT_PRIOR_BETA,
    CostModel,
    GainModel,
    HierarchicalStats,
    LayerStats,
    NEUTRAL_SUCCESS,
    _beta_sample,
)

__all__ = [
    "ContextualRetriever",
    "ContextualRetrieverConfig",
    "GainModel",
    "CostModel",
    "LayerStats",
    "HierarchicalStats",
    "DEFAULT_LAMBDA_COST",
    "DEFAULT_PRIOR_ALPHA",
    "DEFAULT_PRIOR_BETA",
    "NEUTRAL_SUCCESS",
    "DEFAULT_EXPECTED_COST",
    "_beta_sample",
]

EPS = 1e-12

#: cost 惩罚系数默认（EV 分母 1/(1+λ×ExpectedCost)）。
DEFAULT_LAMBDA_COST = 1.0

#: schema 维度缺省 key（无 schema 上下文的 global action 行）。
NO_SCHEMA = ""

#: Thompson 采样分位数 guard：Beta 采样极端接近 0/1 时钳位，避免 0 概率抹掉 EV。
_BETA_CLAMP_EPS = 1e-6

# 注：NEUTRAL_SUCCESS / DEFAULT_PRIOR_ALPHA/BETA / DEFAULT_EXPECTED_COST /
# _beta_sample 定义在 branch_posterior.py（T10 hierarchical 层），顶部已 import。


def _sigmoid01(v: float | None, *, z: float) -> float:
    """sigmoid((v-0.5)/z)；None/非法 → 0.5 中性。"""
    if v is None:
        return 0.5
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(fv):
        return 0.5
    if fv >= 0:
        return 1.0 / (1.0 + math.exp(-(fv - 0.5) / max(float(z), EPS)))
    e = math.exp((fv - 0.5) / max(float(z), EPS))
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# （Gain/Cost/Layer/HierarchicalStats 定义在 branch_posterior.py，见顶部导入）
# ---------------------------------------------------------------------------


@dataclass
class ContextualRetrieverConfig:
    """ContextualRetriever 配置（全部可配，#30 ablation）。

    Attributes
    ----------
    lambda_cost : float
        cost 惩罚强度（EV 分母 1/(1+λ×Cost)）。0 = 不看成本。
    opportunity_coef : float
        (0.5+0.5×coef×Opportunity) 里的机会加权。0 = 不看机会。
    gain_coef : float
        (0.5+0.5×coef×ExpectedGain) 里的增益加权。0 = 不看增益。
    z_temperature : float
        sigmoid(fitness) 温度（与 bayesian_retriever 一致）。
    shrinkage_strength : float
        层级收缩强度 k（parent 权重 = sqrt(n/(n+k))）。
    hierarchical : bool
        False = parent 层不向上层收缩（ablation）。
    thompson : bool
        True = 默认 Thompson posterior sampling；False = posterior mean
        （等价 audit；deterministic）。
    audit : bool
        True = deterministic posterior mean（顶层开关）。
    rng_seed : int | None
        None = 每实例独立熵（seed 不传时同 seed 复现要求显式给）。
    cost_source : Callable | None
        action -> float | None（None = 用观测/中性成本）。
    stagnation_enabled : bool
        False = EV 里 stagnation factor 恒 1.0。
    enabled : bool
        False = select_parents 零行为变化（原样前 k）。
    """

    lambda_cost: float = DEFAULT_LAMBDA_COST
    opportunity_coef: float = 1.0
    gain_coef: float = 1.0
    z_temperature: float = 0.1
    shrinkage_strength: float = 16.0
    hierarchical: bool = True
    thompson: bool = True
    audit: bool = False
    rng_seed: int | None = None
    cost_source: Any | None = None
    stagnation_enabled: bool = True
    enabled: bool = True


#: 中性缺省成本（EV cost 项无观测时用；可用 config.cost_source 注入）。
def _default_cost_of(action: str, *, custom: Any = None) -> float:
    if custom is not None:
        try:
            v = custom(action)
            if v is not None and math.isfinite(float(v)) and float(v) >= 0.0:
                return float(v)
        except Exception:  # noqa: BLE001 - cost_source 异常中性
            pass
    return DEFAULT_EXPECTED_COST


# ---------------------------------------------------------------------------
# ContextualRetriever
# ---------------------------------------------------------------------------


@dataclass
class ContextualRetriever:
    """parent × action × schema 的 contextual Bayesian 检索器（Task 10 / F2）。

    与 ``BayesianRetriever`` 同契约（score_candidate / select_parents / rank /
    record_retrieval 退化 + 内存检索计数），可作为 parent_selector 的 retriever
    超集注入。schema 上下文经 candidate 的 ``schema_id`` / ``schema_tags`` /
    ``logic_id`` 读入（读不到 → 无 schema 证据 → global action 层中性，见 #13）。

    ``observe_attempt`` 是检索器自带的观测入口（offline 累积 / 测试注入 /
    生产路径可在 settle 时把每条 child attempt 回填）；``record_generation`` /
    ``stagnation_penalty_of`` 透传 BayesianRetriever 的 lineage stagnation 语义。

    seed : int | None
        采样种子。None = 每次构造独立熵（随机）；显式给定 → 同 seed 同序列。
        config.rng_seed 也生效（构造函数参数优先）。
    """

    config: ContextualRetrieverConfig = field(default_factory=ContextualRetrieverConfig)
    memory_store: Any | None = field(default=None)
    fitness_fn: Any | None = field(default=None)
    calibrator: Any | None = field(default=None)
    #: 若传入 BayesianRetriever，复用其 depth/retrieval/stagnation/memory 集成
    #: （score_candidate 退化路径）。None = 本类自包含（惰性构造 BayesianRetriever
    #: 做 stagnation 接入）。
    base: BayesianRetriever | None = field(default=None)
    seed: int | None = field(default=None)

    # 内存态
    _stats: dict[str, HierarchicalStats] = field(default_factory=dict)  # action -> 层级
    _rng: Any = field(default=None)
    _node_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = ContextualRetrieverConfig()
        if self._rng is None:
            seed = self.seed if self.seed is not None else self.config.rng_seed
            self._rng = random.Random(seed if seed is not None else 0)
        if self.base is None:
            self.base = BayesianRetriever(memory_store=self.memory_store)

    # -- config 工具 --------------------------------------------------------

    def with_config(self, cfg: ContextualRetrieverConfig) -> "ContextualRetriever":
        """同统计、换 config 的副本（测试/ablation）。"""
        obj = ContextualRetriever(
            config=cfg,
            memory_store=self.memory_store,
            fitness_fn=self.fitness_fn,
            calibrator=self.calibrator,
            base=self.base,
            seed=self.seed if self.seed is not None else cfg.rng_seed,
        )
        obj._stats = {k: v for k, v in self._stats.items()}
        obj._node_states = {k: dict(v) for k, v in self._node_states.items()}
        return obj

    def clone_for_audit(self) -> "ContextualRetriever":
        """audit 模式的只读副本（deterministic posterior mean）。"""
        cfg = ContextualRetrieverConfig(
            **{k: getattr(self.config, k) for k in ContextualRetrieverConfig.__dataclass_fields__}
        )
        cfg.audit = True
        return self.with_config(cfg)

    @property
    def rng(self) -> random.Random:
        return self._rng

    # -- 观测 --------------------------------------------------------------

    def _stats_for(self, action: str) -> HierarchicalStats:
        key = str(action) if action else "REFINE"
        st = self._stats.get(key)
        if st is None:
            st = HierarchicalStats(
                action=key,
                alpha0=DEFAULT_PRIOR_ALPHA,
                beta0=DEFAULT_PRIOR_BETA,
                shrinkage_strength=self.config.shrinkage_strength,
            )
            self._stats[key] = st
        return st

    def observe_attempt(
        self,
        *,
        parent_id: str | None = None,
        action: str | None = None,
        schema_id: str | None = None,
        is_l3_pass: bool = False,
        delta_fitness: float | None = None,
        delta_pool: float | None = None,
        novelty_gain: float | None = None,
        cost: float | None = None,
        gain: bool = True,
    ) -> bool:
        """记录一次 attempt，返回是否记为 success。

        Success = is_l3_pass AND (ΔFitness > 0 OR ΔPoolUtility > 0)（与
        branch_posterior 的阈值语义对齐，但不复制其 delta 阈值细节：delta 为
        None 或 <=0 都算失败；外部已按 L3 阈值过滤时直接传 delta 幅度）。
        parent/schema 分派：
        - parent_id=None 且 schema_id=None → global action 行；
        - parent_id=None 且 schema_id 给定 → schema×action 行；
        - parent_id 与 schema_id 都给定 → parent×action×schema 行。
        global 行只收 schema=None 的观测（保持同一 action 的 schema 差异）。
        """
        gained = (
            (delta_fitness is not None and float(delta_fitness) > 0.0)
            or (delta_pool is not None and float(delta_pool) > 0.0)
        )
        success = bool(is_l3_pass and gained)
        key = str(action) if action else "REFINE"
        st = self._stats_for(key)
        st.observe(
            parent_id=parent_id,
            schema_id=schema_id,
            success=success,
            delta_fitness=delta_fitness,
            delta_pool=delta_pool,
            novelty_gain=novelty_gain,
            cost=cost,
            gain=gain,
        )
        return success

    # -- 打分 --------------------------------------------------------------

    def _schema_of(self, candidate: Any) -> str | None:
        """从 candidate 读 schema 上下文 key（schema_id 优先，回退 schema_tags /
        logic_id 派生 / 显式 key 字段）。无 → None（global action 层）。"""
        if candidate is None:
            return None
        if isinstance(candidate, Mapping):
            d = dict(candidate)
        else:
            d = {}
            to_dict = getattr(candidate, "to_dict", None)
            if callable(to_dict):
                try:
                    d = dict(to_dict())
                except Exception:  # noqa: BLE001
                    d = {}
        for key in ("schema_id", "schema"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
        tags = d.get("schema_tags")
        if isinstance(tags, Mapping) and tags:
            dd = str(tags.get("DataDomain") or "?")
            ev = str(tags.get("Event") or "?")
            return f"TAG:{dd}:{ev}"
        logic = d.get("logic_id")
        if isinstance(logic, str) and logic:
            return logic
        return None

    def _quality_of(self, candidate: Any) -> float | None:
        if candidate is None:
            return None
        if isinstance(candidate, Mapping):
            d = dict(candidate)
        else:
            d = {}
        for key in ("prior_quality", "factor_fitness", "search_fitness", "fitness"):
            try:
                v = d.get(key)
                if v is not None:
                    fv = float(v)
                    if math.isfinite(fv):
                        return min(max(fv, 0.0), 1.0)
            except (TypeError, ValueError):
                continue
        return None

    def _depth_of(self, factor_id: str, candidate: Any) -> int:
        try:
            if isinstance(candidate, Mapping):
                v = candidate.get("depth")
                if v is not None:
                    return int(v)
        except (TypeError, ValueError):
            pass
        try:
            return self.base._depth_of(factor_id) if factor_id else 0
        except Exception:  # noqa: BLE001
            return 0

    def _retrieval_of(self, factor_id: str, candidate: Any) -> int:
        try:
            if isinstance(candidate, Mapping):
                v = candidate.get("retrieval_count")
                if v is not None:
                    return int(v)
        except (TypeError, ValueError):
            pass
        try:
            return self.base._retrieval_count_of(factor_id) if factor_id else 0
        except Exception:  # noqa: BLE001
            return int((self._node_states.get(factor_id) or {}).get("retrieval_count", 0))

    def score_candidate(
        self,
        candidate: Any,
        *,
        factor_id: str | None = None,
        action_to_retrieve: str | None = None,
        explicit_opportunity: Mapping[str, Any] | None = None,
        visible_survival: Mapping[str, Any] | None = None,
        audit: bool | None = None,
    ) -> dict[str, Any]:
        """对单个 candidate 算 contextual EV 完整诊断。

        Returns
        -------
        dict
            factor_id / quality_prior / global_success / schema_success /
            mean_success / sampled_success / opportunity / expected_gain /
            expected_cost / cost_factor / stagnation_factor / retriever_score
            （EV）+ 计数诊断（global_attempts / parent_attempts / schema_attempts）
            + components（与 BayesianRetriever.score_candidate 对齐的维度分解）。
        """
        cfg = self.config
        if factor_id is None:
            factor_id = str(
                candidate.get("factor_id") or candidate.get("id") or ""
            ) if isinstance(candidate, Mapping) else str(
                getattr(candidate, "factor_id", "") or ""
            )
        fid = factor_id or ""
        schema = self._schema_of(candidate)
        action = str(action_to_retrieve) if action_to_retrieve else "REFINE"
        quality = self._quality_of(candidate)
        q = (
            _sigmoid01(quality, z=cfg.z_temperature)
            if quality is not None
            else 0.5
        )
        depth = self._depth_of(fid, candidate)
        retrieval = self._retrieval_of(fid, candidate)
        # stagnation：优先用 candidate 显式 stagnation_penalty / lineage_key 走 base
        stag = 0.0
        try:
            if isinstance(candidate, Mapping) and candidate.get("stagnation_penalty") is not None:
                stag = float(candidate["stagnation_penalty"])
            elif hasattr(self.base, "stagnation_penalty_of"):
                lk = ""
                if isinstance(candidate, Mapping):
                    lk = str(candidate.get("lineage_key") or fid)
                stag = float(self.base.stagnation_penalty_of(lk or fid))
        except Exception:  # noqa: BLE001
            stag = 0.0
        stag = min(max(stag, 0.0), 1.0)
        stag_factor = 1.0 if not cfg.stagnation_enabled else (1.0 - stag)

        st = self._stats.get(action)
        if st is None:
            # 完全无该 action 观测 → 全中性
            global_mean = NEUTRAL_SUCCESS
            schema_mean = NEUTRAL_SUCCESS
            mean_s = NEUTRAL_SUCCESS
            p_s = NEUTRAL_SUCCESS
            gain = NEUTRAL_SUCCESS
            cost = _default_cost_of(action, custom=cfg.cost_source)
            g_attempts = 0
            s_attempts = 0
            p_attempts = 0
        else:
            global_mean = st.global_layer.posterior_mean if st.global_layer.n > 0 else NEUTRAL_SUCCESS
            s_layer = st.schema_stats_of(schema) if schema is not None else None
            schema_mean = s_layer.posterior_mean if s_layer is not None else global_mean
            p_layer = st.parent_stats_of(fid, schema) if (fid and schema) else None
            if not cfg.hierarchical:
                if p_layer is not None:
                    mean_s = p_layer.posterior_mean
                elif s_layer is not None:
                    mean_s = s_layer.posterior_mean
                elif st.global_layer.n > 0:
                    mean_s = st.global_layer.posterior_mean
                else:
                    mean_s = NEUTRAL_SUCCESS
            else:
                mean_s = st.mean_success(
                    parent_id=fid if (fid and schema) else None,
                    schema_id=schema,
                    hierarchical=True,
                )
            p_attempts = p_layer.n if p_layer is not None else 0
            s_attempts = s_layer.n if s_layer is not None else 0
            g_attempts = st.global_layer.n
            # Thompson / audit
            audit_on = cfg.audit if audit is None else bool(audit)
            if audit_on or not cfg.thompson:
                p_s = mean_s
            else:
                p_s = st.sample_success(
                    parent_id=fid if (fid and schema) else None,
                    schema_id=schema,
                    hierarchical=cfg.hierarchical,
                    rng=self._rng,
                )
            gain = st.gain_of(parent_id=fid if (fid and schema) else None, schema_id=schema)
            if cfg.cost_source is not None:
                cost = _default_cost_of(action, custom=cfg.cost_source)
            else:
                cost = st.cost_of(
                    parent_id=fid if (fid and schema) else None, schema_id=schema
                )

        # 机会：显式机会优先；否则走 base.opportunity 退化中性
        opp = 0.5
        if explicit_opportunity is not None:
            try:
                opp = min(max(float(explicit_opportunity.get("total", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                opp = 0.5
        elif hasattr(self.base, "opportunity") and self.base.opportunity is not None:
            try:
                comps = self.base.opportunity.compute(
                    fid,
                    mean_novelty_gain=(
                        st._global_gain.median_novelty_gain
                        if st is not None else None
                    ),
                    attempted_actions=[],
                    cluster_size=(
                        candidate.get("cluster_size")
                        if isinstance(candidate, Mapping) else None
                    ),
                    visible_survival=visible_survival,
                    action_to_retrieve=action,
                )
                opp = min(max(float(comps.get("total", 0.5)), 0.0), 1.0)
            except Exception:  # noqa: BLE001 - 机会计算异常中性
                opp = 0.5

        opp_factor = 0.5 + 0.5 * cfg.opportunity_coef * (opp - 0.5)
        gain_factor = 0.5 + 0.5 * cfg.gain_coef * (gain - 0.5)
        denom = 1.0 + cfg.lambda_cost * cost
        cost_factor = 1.0 / denom
        ev = q * mean_s * opp_factor * gain_factor * cost_factor * stag_factor
        ev_th = q * p_s * opp_factor * gain_factor * cost_factor * stag_factor

        return {
            "factor_id": fid,
            "schema_id": schema,
            "action": action,
            "quality_prior": q,
            "global_success": global_mean,
            "schema_success": schema_mean,
            "mean_success": mean_s,
            "sampled_success": p_s,
            "opportunity": opp,
            "opportunity_factor": opp_factor,
            "expected_gain": gain,
            "gain_factor": gain_factor,
            "expected_cost": cost,
            "cost_factor": cost_factor,
            "stagnation_penalty": float(stag),
            "stagnation_factor": float(stag_factor),
            "global_attempts": g_attempts,
            "schema_attempts": s_attempts,
            "parent_attempts": p_attempts,
            "ev_score": ev,
            "retriever_score": ev if (cfg.audit if audit is None else bool(audit)) else ev_th,
        }

    def rank(
        self,
        candidates: Sequence[Any],
        *,
        action_to_retrieve: str | None = None,
        visible_survival: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        out = []
        for c in candidates:
            out.append(
                self.score_candidate(
                    c,
                    action_to_retrieve=action_to_retrieve,
                    visible_survival=visible_survival,
                )
            )
        out.sort(key=lambda d: -float(d["retriever_score"]))
        return out

    def select_parents(
        self,
        candidates: Sequence[Any],
        k: int = 5,
        *,
        action_to_retrieve: str | None = None,
        visible_survival: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """按 contextual EV 选 top-k parent。disabled / 空候选 → 原样前 k。"""
        if not self.config.enabled or not candidates:
            return list(candidates[:k])
        scored: list[tuple[float, int, Any]] = []
        for i, c in enumerate(candidates):
            d = self.score_candidate(
                c,
                action_to_retrieve=action_to_retrieve,
                visible_survival=visible_survival,
            )
            scored.append((float(d["retriever_score"]), i, c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen = [c for _, _, c in scored[:k]]
        # 检索事件（与 BayesianRetriever.select_parents 一致）
        for c in chosen:
            fid = str(c.get("factor_id") or c.get("id") or "") if isinstance(c, Mapping) else str(
                getattr(c, "factor_id", "") or ""
            )
            if fid:
                self.record_retrieval(fid, action=action_to_retrieve)
        return chosen

    def record_retrieval(self, factor_id: str, *, action: str | None = None) -> None:
        """记录一次检索事件（透传 base 或退化内存 dict）。"""
        if not factor_id:
            return
        try:
            if hasattr(self.base, "record_retrieval"):
                self.base.record_retrieval(factor_id, action=action)
                return
        except Exception:  # noqa: BLE001
            pass
        st = self._node_states.setdefault(str(factor_id), {"retrieval_count": 0})
        st["retrieval_count"] = int(st.get("retrieval_count", 0)) + 1

    def record_generation(self, *, lineage_key: str, record: Any) -> Any:
        try:
            if hasattr(self.base, "record_generation"):
                return self.base.record_generation(lineage_key=lineage_key, record=record)
        except Exception:  # noqa: BLE001
            return None
        return None

    def stagnation_penalty_of(self, lineage_key: str) -> float:
        try:
            if hasattr(self.base, "stagnation_penalty_of"):
                return float(self.base.stagnation_penalty_of(lineage_key))
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                k: getattr(self.config, k)
                for k in ContextualRetrieverConfig.__dataclass_fields__
                if k != "cost_source"
            },
            "stats": {k: v.to_dict() for k, v in self._stats.items()},
            "node_states": {k: dict(v) for k, v in self._node_states.items()},
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        *,
        memory_store: Any | None = None,
        fitness_fn: Any | None = None,
    ) -> "ContextualRetriever":
        d = dict(d or {})
        cfg_d = dict(d.get("config") or {})
        cfg = ContextualRetrieverConfig(**cfg_d)
        obj = cls(
            config=cfg,
            memory_store=memory_store,
            fitness_fn=fitness_fn,
        )
        obj._stats = {
            str(k): HierarchicalStats.from_dict(v) for k, v in (d.get("stats") or {}).items()
        }
        obj._node_states = {
            str(k): dict(v) for k, v in (d.get("node_states") or {}).items()
        }
        return obj

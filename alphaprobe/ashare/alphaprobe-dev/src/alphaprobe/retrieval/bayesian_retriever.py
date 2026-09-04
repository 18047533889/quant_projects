"""Bayesian Retriever V2（任务书 §19-§23 / §46 / §57）——主检索器。

保留原始 AlphaPROBE 结构 Factor Quality × Depth Decay × Retrieval Frequency Decay，
但 Qual(F) 用 FactorFitness（fitness/__init__.py 的 ``compute_search_fitness`` /
``factor_fitness_v2``）替换 ICIR 单指标。

公式：

    Prior(F) = sigmoid(z(FactorFitness(F))) × (1 - gamma)^depth(F) × (1 - omega)^retrieval_times(F)

    P_success(F) = Beta 分支成功后验（见 branch_posterior.BetaBranchPosterior）

    RetrieverScore(F) = Prior(F) × P_success(F) × (0.6 + 0.4 × SearchOpportunity(F))
                        × UncertaintyBonus(F)

    UncertaintyBonus(F) = 1 + beta_ucb × sqrt(log(1 + total_attempts) / (1 + node_attempts))

设计约束：
- 只 import fitness 公共函数（fitness 不反向 import retrieval）；
- 禁止在 FactorFitness 中重复惩罚相关性（N 只占 10%），Retriever 的 novelty 逻辑
  放 SearchOpportunity；
- memory/ 只调用公共 API（lineage_parents / exploration_of / get_node /
  rare_directions 等），读不到退化到内存 dict。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from alphaprobe.retrieval.branch_posterior import (
    DEFAULT_ALPHA0,
    DEFAULT_BETA0,
    BetaBranchPosterior,
)
from alphaprobe.retrieval.search_opportunity import (
    DEFAULT_ACTION_FAMILIES,
    SearchOpportunity,
    SearchOpportunityConfig,
)

EPS = 1e-9
#: 默认 sigmoid 温度（z 缩放）：Fitness 差 0.1 → z 差 1.0（温度敏感区间）
DEFAULT_Z_TEMPERATURE = 0.1
#: 默认深度衰减系数 gamma（每深一层先验乘 (1-gamma)）
DEFAULT_GAMMA = 0.15
#: 默认检索频率衰减系数 omega（每被检索一次先验乘 (1-omega)）
DEFAULT_OMEGA = 0.05
#: 默认 UCB 探索强度（UncertaintyBonus 系数）
DEFAULT_BETA_UCB = 1.0
#: memory 读不到时 depth 的退化默认
FALLBACK_DEPTH = 0
#: 停滞判据缺省（depth 只留小复杂度先验，见 stagnation.py）
DEFAULT_STAGNATION_PENALTY = 0.0
#: 缺省 stagnation 接入开关（默认 True：depth 硬指数衰减由 stagnation 取代）
DEFAULT_STAGNATION_ENABLED = True
#: depth 小先验（stagnation 开启时 depth 不再指数衰减，只留轻微复杂度先验）
DEFAULT_DEPTH_PRIOR_DECAY = 0.02


def _safe_fitness(value: Any) -> float | None:
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return fv


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def compute_prior(
    factor_fitness: float | None,
    depth: int,
    retrieval_times: int,
    *,
    gamma: float = DEFAULT_GAMMA,
    omega: float = DEFAULT_OMEGA,
    z_temperature: float = DEFAULT_Z_TEMPERATURE,
    stagnation_penalty: float = DEFAULT_STAGNATION_PENALTY,
    stagnation_enabled: bool = DEFAULT_STAGNATION_ENABLED,
    depth_prior_decay: float = DEFAULT_DEPTH_PRIOR_DECAY,
) -> float:
    """Prior(F) = sigmoid(z(Fitness)) × depth_factor × (1-omega)^retrieval_times。

    depth 项（plan Task 11：depth decay 替换为 Lineage Stagnation）：

    - ``stagnation_enabled=False``（ablation）→ 旧硬指数衰减
      ``(1-gamma)^depth``（历史行为完整保留，可对比）。
    - ``stagnation_enabled=True``（默认）→ depth 只留小复杂度先验
      ``(1-depth_prior_decay)^depth``（默认每层 0.02，远轻于旧 gamma=0.15），
      真正的「该不该继续挖」交给 ``stagnation_penalty``（0~1，来自
      stagnation.StagnationEvaluator）抑制停滞 branch；无停滞证据时中性。

    三因子分解各自单调：fitness↑→prior↑、stagnation_penalty↑→prior↓、
    retrieval↑→prior↓。Fitness 缺失（None）→ sigmoid(0) = 0.5（中性，
    不因缺指标额外惩罚）。
    """
    q = (
        _sigmoid((float(factor_fitness) - 0.5) / max(z_temperature, EPS))
        if factor_fitness is not None
        else 0.5
    )
    depth_n = max(0, int(depth))
    if stagnation_enabled:
        # depth 小复杂度先验：默认每层 0.02，深 20 层才衰减到 ~0.67——
        # 不再用旧 gamma=0.15 每层 15% 硬指数压制深 but fertile 的 branch。
        d = max(0.0, 1.0 - max(0.0, float(depth_prior_decay))) ** depth_n
        # stagnation 抑制（0 中性 → 1 全停）。gamma 参数在 stagnation 模式下
        # **仅用于 ablation/旧调用对比**，不再参与 depth 衰减；无停滞证据
        # （penalty=0）时 sp=1.0，fitness 中性（None）时 q=0.5 → prior=0.5。
        s = max(0.0, min(1.0, float(stagnation_penalty or 0.0)))
        sp = 1.0 - s
    else:
        # ablation（stagnation_enabled=False）：完整保留旧 depth 硬指数衰减
        # ``(1-gamma)^depth``，与 V3.1 之前的 compute_prior 逐字一致。
        d = max(0.0, 1.0 - max(0.0, float(gamma))) ** depth_n
        sp = 1.0
    r = max(0.0, 1.0 - max(0.0, float(omega))) ** max(0, int(retrieval_times))
    return max(0.0, q * d * sp * r)


def uncertainty_bonus(
    total_attempts: int,
    node_attempts: int,
    *,
    beta_ucb: float = DEFAULT_BETA_UCB,
) -> float:
    """UncertaintyBonus = 1 + beta_ucb × sqrt(log(1+total)/(1+node))。

    未尝试节点（node_attempts=0）→ +1×beta_ucb（最大探索加成）；老节点
    node≈total → 接近 1（不额外奖励）。
    """
    total = max(0, int(total_attempts))
    node = max(0, int(node_attempts))
    # 无 attempt 记录（total=0）：无信息 → 中性 1.0（不给加成也不惩罚）
    if total <= 0:
        return 1.0
    # 已完全探索（node == total）→ 无探索空间，返回 1.0
    if node >= total:
        return 1.0
    # 未尝试节点（node=0）→ 最大探索加成；老节点 node≈total → 接近 1。
    # 用 (1+node) 平滑，node=0 时也给出有限非 1 的加成。
    bonus = 1.0 + max(0.0, float(beta_ucb)) * math.sqrt(
        math.log(1.0 + total) / (1.0 + node)
    )
    return bonus


def compute_retriever_score(
    *,
    factor_fitness: float | None,
    depth: int,
    retrieval_times: int,
    posterior_success: float,
    search_opportunity: float,
    total_attempts: int,
    node_attempts: int,
    gamma: float = DEFAULT_GAMMA,
    omega: float = DEFAULT_OMEGA,
    z_temperature: float = DEFAULT_Z_TEMPERATURE,
    beta_ucb: float = DEFAULT_BETA_UCB,
    stagnation_penalty: float = DEFAULT_STAGNATION_PENALTY,
    stagnation_enabled: bool = DEFAULT_STAGNATION_ENABLED,
    depth_prior_decay: float = DEFAULT_DEPTH_PRIOR_DECAY,
) -> float:
    """RetrieverScore = Prior × P_success × (0.6 + 0.4×Opportunity) × UCB。"""
    prior = compute_prior(
        factor_fitness, depth, retrieval_times,
        gamma=gamma, omega=omega, z_temperature=z_temperature,
        stagnation_penalty=stagnation_penalty,
        stagnation_enabled=stagnation_enabled,
        depth_prior_decay=depth_prior_decay,
    )
    ps = min(max(float(posterior_success), 0.0), 1.0)
    opp = min(max(float(search_opportunity), 0.0), 1.0)
    ub = uncertainty_bonus(total_attempts, node_attempts, beta_ucb=beta_ucb)
    return max(0.0, prior * ps * (0.6 + 0.4 * opp) * ub)


def retriever_score_from_state(
    state: Any,
    *,
    gamma: float = DEFAULT_GAMMA,
    omega: float = DEFAULT_OMEGA,
    z_temperature: float = DEFAULT_Z_TEMPERATURE,
    beta_ucb: float = DEFAULT_BETA_UCB,
    stagnation_enabled: bool = DEFAULT_STAGNATION_ENABLED,
    stagnation_penalty: float = DEFAULT_STAGNATION_PENALTY,
) -> float:
    """从 BayesianNodeState 直接算 RetrieverScore（不重算 opportunity）。

    A5：``global_total_attempts``（全局池尝试总量）与 ``total_attempts``
    （本节点点级尝试量）是两个不同的计数器，必须分别传给 UCB 的 total /
    node 槽位——旧实现把 ``state.total_attempts`` 同时传两个位置，导致
    node>=total 时 UncertaintyBonus 恒退化为 1。

    历史 state 没有 global 字段（=0）→ 退化用节点计数当全局：无全局证据
    时不给探索加成（与旧版行为一致），绝不伪造探索空间。
    """
    total = int(getattr(state, "global_total_attempts", 0) or 0)
    node = int(getattr(state, "total_attempts", 0) or 0)
    if total <= 0:
        # 无全局尝试证据：退化为 node==total 的中性（UCB = 1.0）。
        # 注意不能把 node 直接当 total——那会重演 A5 恒退化，只是此时
        # 的语义是「无信息不给加成」而非「node 已挖满」。
        total = node
    return compute_retriever_score(
        factor_fitness=state.prior_quality,
        depth=state.depth,
        retrieval_times=state.retrieval_count,
        posterior_success=state.posterior_success,
        search_opportunity=state.search_opportunity,
        total_attempts=total,
        node_attempts=node,
        gamma=gamma,
        omega=omega,
        z_temperature=z_temperature,
        beta_ucb=beta_ucb,
        stagnation_enabled=stagnation_enabled,
        stagnation_penalty=stagnation_penalty,
    )


@dataclass
class BayesianRetrieverConfig:
    """Retriever 配置（全部可配，逻辑不许硬编码数值）。"""

    gamma: float = DEFAULT_GAMMA
    omega: float = DEFAULT_OMEGA
    z_temperature: float = DEFAULT_Z_TEMPERATURE
    beta_ucb: float = DEFAULT_BETA_UCB
    alpha0: float = DEFAULT_ALPHA0
    beta0: float = DEFAULT_BETA0
    delta_fitness: float = 0.005
    delta_pool: float = 0.01
    #: 0.6 + 0.4×Opportunity 的机会权重
    opportunity_weight: float = 0.4
    #: plan Task 11：depth decay → Lineage Stagnation（ablation 开关，Part G #30）。
    #: True（默认）= depth 只留小复杂度先验 + stagnation_penalty 抑制停滞 branch；
    #: False = 完全回退旧 ``(1-gamma)^depth`` 硬指数衰减。
    stagnation_enabled: bool = DEFAULT_STAGNATION_ENABLED
    #: stagnation 判据权重配置（透传 StagnationConfig；None = 默认）
    stagnation_config: Any | None = None
    enabled: bool = True
    action_families: tuple[str, ...] = DEFAULT_ACTION_FAMILIES


DEFAULT_RETRIEVER_CONFIG = BayesianRetrieverConfig()


@dataclass
class BayesianRetriever:
    """主检索器：为 candidates 算 RetrieverScore，提供 select_parents 排序。

    Parameters
    ----------
    config : BayesianRetrieverConfig | None
    memory_store : Any | None
        GlobalMemoryStore 兼容对象（lineage/深度/被检索次数从 memory store 读，
        有现成表；读不到退化到内存 dict）。
    fitness_fn : Callable[[Any], float | None] | None
        从 candidate/node 计算 FactorFitness 的注入函数。None 时尝试从
        candidate 里读 ``factor_fitness`` / ``search_fitness`` / ``fitness`` /
        ``rank_ic``（优先完整 V2 值，缺失退化为 0.5 中性）。
    opportunity : SearchOpportunity | None
        机会计算器。None 时惰性构造（cluster/survival 退化中性）。
    calibrator : Any | None
        可选注入；若 fitness_fn 需要 calibrator 计算 V2 fitness 时使用。
    """

    config: BayesianRetrieverConfig = field(default_factory=BayesianRetrieverConfig)
    memory_store: Any | None = field(default=None)
    fitness_fn: Callable[[Any], float | None] | None = field(default=None)
    opportunity: SearchOpportunity | None = field(default=None)
    calibrator: Any | None = field(default=None)

    # 内存态退化（memory 读不到时用）
    _node_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    _memory_ok: bool = True
    #: branch 停滞 evaluator（plan Task 11）：lineage key -> StagnationEvaluator
    _stagnation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = BayesianRetrieverConfig()
        if self.opportunity is None:
            try:
                self.opportunity = SearchOpportunity.from_memory_store(self.memory_store)
            except Exception:  # noqa: BLE001 - 本地退化中性
                self.opportunity = SearchOpportunity(
                    config=SearchOpportunityConfig(
                        w_novelty=0.40, w_cluster=0.25, w_action=0.20, w_survival=0.15,
                    )
                )
        if self.fitness_fn is None:
            self.fitness_fn = self._default_fitness_fn

    # ------------------------------------------------------------------
    # plan Task 11：lineage stagnation 观测与积分
    # ------------------------------------------------------------------

    def record_generation(
        self,
        *,
        lineage_key: str,
        record: Any,
    ) -> Any:
        """累积一条 branch 生成记录并返回最新 StagnationResult。

        与旧 depth 硬指数衰减正交：retriever 不再用 depth 直接罚深 branch，
        而是把最近 K 代有没有产出（stagnation_penalty）作为先验抑制项。
        lineage_key 可为 factor_id（branch 根）或 lineage 路径 key。
        """
        if not self.config.stagnation_enabled:
            return None
        from alphaprobe.search.stagnation import StagnationEvaluator

        ev = self._stagnation.get(str(lineage_key))
        if ev is None:
            ev = StagnationEvaluator(config=self.config.stagnation_config)
            self._stagnation[str(lineage_key)] = ev
        return ev.observe(record)

    def stagnation_penalty_of(self, lineage_key: str) -> float:
        """branch 当前停滞分（0 = 中性/未接入，1 = 全停）。"""
        ev = self._stagnation.get(str(lineage_key))
        if ev is None or not self.config.stagnation_enabled:
            return 0.0
        res = ev.evaluate()
        if res is None:
            return 0.0
        return float(res.stagnation)

    # ------------------------------------------------------------------
    # fitness 提取
    # ------------------------------------------------------------------

    def _default_fitness_fn(self, candidate: Any) -> float | None:
        if isinstance(candidate, Mapping):
            d = dict(candidate)
        else:
            d = getattr(candidate, "to_dict", None)
            d = d() if callable(d) else {}
        for key in ("factor_fitness", "search_fitness", "fitness"):
            v = d.get(key)
            fv = _safe_fitness(v)
            if fv is not None:
                return fv
        for key in ("rank_ic", "single_ic", "norm_ic"):
            v = d.get(key)
            fv = _safe_fitness(v)
            if fv is not None:
                return max(0.0, min(1.0, fv * 50.0))
        return None

    def _fitness_from_memory(self, factor_id: str) -> float | None:
        if self.memory_store is None:
            return None
        try:
            if hasattr(self.memory_store, "get_evaluations"):
                for ev in self.memory_store.get_evaluations(factor_id):
                    mb = ev.get("metric_bundle") or {}
                    for key in ("factor_fitness", "search_fitness", "fitness", "single_ic"):
                        fv = _safe_fitness(mb.get(key))
                        if fv is not None:
                            return fv
            node = None
            if hasattr(self.memory_store, "get_node"):
                node = self.memory_store.get_node(factor_id)
            if node is not None:
                fv = _safe_fitness(node.get("schema_json", {}).get("fitness")) if isinstance(
                    node.get("schema_json"), Mapping
                ) else None
                if fv is None:
                    fv = _safe_fitness(node.get("fitness"))
                if fv is not None:
                    return fv
        except Exception:  # noqa: BLE001 - memory 不可用退化 None
            return None
        return None

    # ------------------------------------------------------------------
    # memory 读取（全部公共 API；读不到退化内存 dict）
    # ------------------------------------------------------------------

    def _depth_of(self, factor_id: str) -> int:
        if self.memory_store is not None:
            try:
                from alphaprobe.lineage import LineageView

                view = LineageView(self.memory_store)
                if hasattr(view, "depth_of"):
                    return int(view.depth_of(factor_id))
            except Exception:  # noqa: BLE001 - 退化内存 dict
                pass
        return int((self._node_states.get(factor_id) or {}).get("depth", FALLBACK_DEPTH))

    # P0-B：检索频率必须是「被检索多少次」而不是「几个孩子」。record_retrieval
    # 由 orchestrator/pipeline 在每次 select_parents 命中（选为 parent）时调用；
    # memory API 可用 → 落 retrieval_events 轻量表；不可用 → 同步维护内存 dict
    # 退化态（_node_states[factor_id]["retrieval_count"] += 1）。

    def record_retrieval(
        self, factor_id: str, *, action: str | None = None, memory: bool = True
    ) -> None:
        """记录一次检索事件（该 factor 被选为 parent / 检索起点）。"""
        if not factor_id:
            return
        if memory and self.memory_store is not None:
            try:
                if hasattr(self.memory_store, "record_retrieval"):
                    self.memory_store.record_retrieval(factor_id=factor_id, action=action)
                    return
            except Exception:  # noqa: BLE001 - 记失败不阻塞检索，退化内存计数
                pass
        st = self._node_states.setdefault(
            str(factor_id), {"retrieval_count": 0, "depth": FALLBACK_DEPTH}
        )
        st["retrieval_count"] = int(st.get("retrieval_count", 0)) + 1
        if action is not None:
            st.setdefault("attempted_actions", [])
            if action not in st["attempted_actions"]:
                st["attempted_actions"].append(action)

    def _retrieval_count_of(self, factor_id: str) -> int:
        # P0-B：被检索次数 = 真·检索事件计数（record_retrieval 落表），
        # 不再是 lineage_parents 的「繁殖代数」——后者会随成功繁殖而增长，
        # 造成越成功越被降权的反向激励。memory API 不可用 → 内存 dict 退化态。
        if self.memory_store is not None:
            try:
                if hasattr(self.memory_store, "retrieval_count_of"):
                    return int(self.memory_store.retrieval_count_of(factor_id))
            except Exception:  # noqa: BLE001 - 退化内存 dict
                pass
        return int((self._node_states.get(factor_id) or {}).get("retrieval_count", 0))

    def _attempted_actions_of(self, factor_id: str) -> list[str]:
        if self.memory_store is not None:
            try:
                if hasattr(self.memory_store, "exploration_of"):
                    return [str(e.get("action_family")) for e in self.memory_store.exploration_of(factor_id) if e.get("action_family")]
            except Exception:  # noqa: BLE001 - 退化内存 dict
                pass
        return list((self._node_states.get(factor_id) or {}).get("attempted_actions", []))

    def _node_stats_of(self, factor_id: str) -> dict[str, Any]:
        if self.memory_store is not None:
            try:
                if hasattr(self.memory_store, "exploration_of"):
                    rows = self.memory_store.exploration_of(factor_id)
                    if rows:
                        total = sum(int(r.get("attempts", 0)) for r in rows)
                        succ = sum(int(r.get("successes", 0)) for r in rows)
                        md = [
                            float(r.get("mean_delta_fitness", 0.0) or 0.0)
                            for r in rows if r.get("mean_delta_fitness") is not None
                        ]
                        mean_delta = sum(md) / len(md) if md else 0.0
                        return {"total_attempts": total, "total_successes": succ, "mean_delta_fitness": mean_delta}
            except Exception:  # noqa: BLE001 - 退化内存 dict
                pass
        st = self._node_states.get(factor_id) or {}
        return {
            "total_attempts": int(st.get("total_attempts", 0)),
            "total_successes": int(st.get("total_successes", 0)),
            "mean_delta_fitness": float(st.get("mean_delta_fitness", 0.0)),
        }

    # ------------------------------------------------------------------
    # 单节点打分
    # ------------------------------------------------------------------

    def _fitness_of(self, candidate: Any, factor_id: str | None = None) -> float | None:
        if self.fitness_fn is not None:
            try:
                fv = self.fitness_fn(candidate)
                if fv is not None:
                    return max(0.0, min(1.0, float(fv)))
            except Exception:  # noqa: BLE001 - fitness_fn 异常退化
                pass
        if factor_id:
            fv = self._fitness_from_memory(factor_id)
            if fv is not None:
                return max(0.0, min(1.0, float(fv)))
        return None

    def score_candidate(
        self,
        candidate: Any,
        *,
        factor_id: str | None = None,
        explicit_fitness: float | None = None,
        explicit_success_probability: float | None = None,
        explicit_opportunity: Mapping[str, Any] | None = None,
        visible_survival: Mapping[str, Any] | None = None,
        action_to_retrieve: str | None = None,
    ) -> dict[str, Any]:
        """对单个 candidate（node）算完整检索诊断 + RetrieverScore。

        Returns
        -------
        dict
            factor_id / fitness / prior / posterior_success / search_opportunity /
            uncertainty_bonus / retriever_score / components（四维分解）。
        """
        if factor_id is None:
            if isinstance(candidate, Mapping):
                factor_id = str(
                    candidate.get("factor_id") or candidate.get("id") or ""
                )
            else:
                factor_id = str(getattr(candidate, "factor_id", "") or "")
        fid = factor_id or ""

        fitness = explicit_fitness if explicit_fitness is not None else self._fitness_of(candidate, fid)
        if explicit_fitness is None and fitness is None:
            fitness = 0.5  # 缺指标中性，不额外惩罚

        # 从 candidate 本身读 depth/retrieval_count（测试/内存态显式传入时优先）；
        # 否则从 memory/内存退化读。
        cand_depth: int | None = None
        cand_retrieval: int | None = None
        cand_attempts: int | None = None
        cand_global_attempts: int | None = None
        cand_node_attempts: int | None = None
        cand_successes: int | None = None
        cand_stagnation: float | None = None
        cand_lineage_key: str | None = None
        if isinstance(candidate, Mapping):
            cd = dict(candidate)
        else:
            cd = {}
        if "depth" in cd:
            try:
                cand_depth = int(cd["depth"])
            except (TypeError, ValueError):
                cand_depth = None
        if "retrieval_count" in cd:
            try:
                cand_retrieval = int(cd["retrieval_count"])
            except (TypeError, ValueError):
                cand_retrieval = None
        if "total_attempts" in cd:
            try:
                cand_attempts = int(cd["total_attempts"])
            except (TypeError, ValueError):
                cand_attempts = None
        # A5：显式 global_total_attempts 优先；否则 total_attempts 退化为
        # 「本节点尝试规模」（node 维度）；node_attempts 独立读取（可为 None）。
        if "global_total_attempts" in cd:
            try:
                cand_global_attempts = int(cd["global_total_attempts"])
            except (TypeError, ValueError):
                cand_global_attempts = None
        if "node_attempts" in cd:
            try:
                cand_node_attempts = int(cd["node_attempts"])
            except (TypeError, ValueError):
                cand_node_attempts = None
        if "total_successes" in cd:
            try:
                cand_successes = int(cd["total_successes"])
            except (TypeError, ValueError):
                cand_successes = None
        # plan Task 11：停滞分显式传入优先；否则按 lineage key 从内部累积查
        if "stagnation_penalty" in cd:
            try:
                cand_stagnation = float(cd["stagnation_penalty"])
            except (TypeError, ValueError):
                cand_stagnation = None
        if "lineage_key" in cd:
            cand_lineage_key = str(cd["lineage_key"]) or None
        # 显式 total_attempts 缺失时：retrieval_count（被检索次数）就是该节点
        # 的尝试规模代理（保持 §19 核心场景的语义：被挖 200 次 = 200 attempts）。
        if cand_attempts is None and cand_retrieval is not None:
            cand_attempts = cand_retrieval

        depth = cand_depth if cand_depth is not None else (self._depth_of(fid) if fid else 0)
        retrieval_times = (
            cand_retrieval
            if cand_retrieval is not None
            else (self._retrieval_count_of(fid) if fid else 0)
        )
        # plan Task 11：stagnation 抑制项（0~1）。显式传入优先；否则该 candidate
        # 的 lineage_key（缺省用 factor_id）查内部累积的 StagnationEvaluator。
        if cand_stagnation is not None:
            stagnation_penalty = cand_stagnation
        else:
            stagnation_penalty = self.stagnation_penalty_of(
                cand_lineage_key or fid or ""
            )
        stats = self._node_stats_of(fid) if fid else {}
        attempted = self._attempted_actions_of(fid) if fid else []

        total_attempts = (
            cand_attempts
            if cand_attempts is not None
            else int(stats.get("total_attempts", 0))
        )
        # A5：UCB 的 total 槽位取「全局池尝试总量」：显式 global_total_attempts
        # 优先；candidate 仍给 total_attempts 且未给 node_attempts（旧版契约，
        # total_attempts 即节点尝试规模）→ 节点规模就是全局规模，不加成。
        if cand_global_attempts is not None:
            global_attempts = cand_global_attempts
        else:
            # 无全局证据：旧版语义 total_attempts 即节点尝试量 → global=node
            # （不给探索加成），绝不虚构全局规模制造假的探索空间。
            global_attempts = (
                cand_attempts
                if cand_attempts is not None
                else int(stats.get("total_attempts", 0))
            )
        # node 维度的尝试规模：显式 node_attempts 优先；否则退化为节点总尝试。
        if cand_node_attempts is not None:
            node_attempts = cand_node_attempts
        else:
            node_attempts = total_attempts
        successes = (
            cand_successes
            if cand_successes is not None
            else int(stats.get("total_successes", 0))
        )

        # 检索频率（retrieval_times）是 node 被「选为 parent / 生成」的次数；
        # 尝试规模（total_attempts）是 node 作为 parent 的生成尝试数。
        # node_attempts（本节点已尝试数）缺失时 = total_attempts；
        # total_attempts 缺失但 retrieval_times 存在 → 用 retrieval_times 作代理
        # （被挖 200 次 ≈ 200 attempts，§19 核心场景语义）。
        # global_attempts（UCB 的 total 槽位）至少 = node_attempts：全局池规模
        # ≥ 节点规模；node_attempts 全缺失 → node=0 → UCB 对未尝试节点给最大
        # 探索加成（无信息时不给加成也不惩罚是另一条路径，见下方 node_attempts
        # 显式置 0 注释——此处 total 有全局证据才成立）。
        if total_attempts <= 0 and retrieval_times > 0:
            total_attempts = retrieval_times
        if node_attempts <= 0 and retrieval_times > 0:
            node_attempts = retrieval_times
        if global_attempts <= 0 and retrieval_times > 0:
            global_attempts = retrieval_times
        if global_attempts < node_attempts:
            global_attempts = node_attempts
        if total_attempts < node_attempts:
            total_attempts = node_attempts
        if total_attempts < global_attempts:
            total_attempts = global_attempts
        # node_attempts 全缺失（retrieval_times=0 且无 total）→ node=0：
        # 无信息 → UCB 中性 1.0（不给加成也不惩罚）。
        if node_attempts <= 0:
            node_attempts = 0
        if global_attempts <= 0:
            global_attempts = 0

        if explicit_success_probability is not None:
            ps = explicit_success_probability
        else:
            bp = BetaBranchPosterior(
                alpha0=self.config.alpha0,
                beta0=self.config.beta0,
                delta_fitness=self.config.delta_fitness,
                delta_pool=self.config.delta_pool,
            )
            bp.n_attempt = total_attempts
            bp.n_success = successes
            ps = bp.posterior_success

        if explicit_opportunity is not None:
            comps = {
                "novelty": explicit_opportunity.get("novelty", 0.0),
                "cluster_rarity": explicit_opportunity.get("cluster_rarity", 0.0),
                "unexplored_action": explicit_opportunity.get("unexplored_action", 0.0),
                "survival": explicit_opportunity.get("survival", 0.0),
            }
            opp = float(explicit_opportunity.get("total", 0.0))
        else:
            comps = self.opportunity.compute(
                fid,
                mean_novelty_gain=stats.get("mean_delta_fitness"),
                attempted_actions=attempted,
                cluster_size=cd.get("cluster_size") if cd else None,
                visible_survival=visible_survival,
                action_to_retrieve=action_to_retrieve,
            )
            opp = float(comps["total"])

        ub = uncertainty_bonus(global_attempts, node_attempts, beta_ucb=self.config.beta_ucb)

        score = compute_retriever_score(
            factor_fitness=fitness,
            depth=depth,
            retrieval_times=retrieval_times,
            posterior_success=ps,
            search_opportunity=opp,
            total_attempts=global_attempts,
            node_attempts=node_attempts,
            gamma=self.config.gamma,
            omega=self.config.omega,
            z_temperature=self.config.z_temperature,
            beta_ucb=self.config.beta_ucb,
            stagnation_penalty=stagnation_penalty,
            stagnation_enabled=self.config.stagnation_enabled,
        )
        return {
            "factor_id": fid,
            "fitness": fitness,
            "prior": compute_prior(
                fitness, depth, retrieval_times,
                gamma=self.config.gamma,
                omega=self.config.omega,
                z_temperature=self.config.z_temperature,
                stagnation_penalty=stagnation_penalty,
                stagnation_enabled=self.config.stagnation_enabled,
            ),
            "posterior_success": ps,
            "search_opportunity": opp,
            "uncertainty_bonus": ub,
            "stagnation_penalty": float(stagnation_penalty),
            "retriever_score": score,
            "components": comps,
        }

    # ------------------------------------------------------------------
    # 排序选择（orchestrator / round_manager 接线）
    # ------------------------------------------------------------------

    def select_parents(
        self,
        candidates: Sequence[Any],
        k: int = 5,
        *,
        action_to_retrieve: str | None = None,
        visible_survival: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """按 RetrieverScore 排序选前 k 个 parent（不打乱原对象，返回子列表）。

        disabled 时直接返回前 k 个（顺序不变，零行为变化）。

        P0-B：选中的 k 个 parent 会被作为检索起点繁殖 —— 这里即「被检索」的
        真时点，命中即记一次检索事件（record_retrieval），驱动频率衰减计数。
        disabled / 空候选 → 没有真正发生检索选择，不记。
        """
        if not self.config.enabled or not candidates:
            return list(candidates[:k])
        scored: list[tuple[float, int, Any]] = []
        for i, c in enumerate(candidates):
            d = self.score_candidate(
                c, action_to_retrieve=action_to_retrieve, visible_survival=visible_survival
            )
            scored.append((float(d["retriever_score"]), i, c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen = [c for _, _, c in scored[:k]]
        for c in chosen:
            fid = ""
            if isinstance(c, Mapping):
                fid = str(c.get("factor_id") or c.get("id") or "")
            else:
                fid = str(getattr(c, "factor_id", "") or "")
            if fid:
                self.record_retrieval(fid, action=action_to_retrieve)
        return chosen

    def rank(
        self,
        candidates: Sequence[Any],
        *,
        action_to_retrieve: str | None = None,
        visible_survival: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """返回带 score 的完整排序诊断（测试/审计用）。"""
        if not candidates:
            return []
        out = []
        for c in candidates:
            out.append(
                self.score_candidate(
                    c, action_to_retrieve=action_to_retrieve, visible_survival=visible_survival
                )
            )
        out.sort(key=lambda d: -float(d["retriever_score"]))
        return out

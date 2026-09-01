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
) -> float:
    """Prior(F) = sigmoid(z(Fitness)) × (1-gamma)^depth × (1-omega)^retrieval_times。

    三因子分解各自单调：fitness↑→prior↑、depth↑→prior↓、retrieval↑→prior↓。
    Fitness 缺失（None）→ sigmoid(0) = 0.5（中性，不因缺指标额外惩罚）。
    """
    q = (
        _sigmoid((float(factor_fitness) - 0.5) / max(z_temperature, EPS))
        if factor_fitness is not None
        else 0.5
    )
    d = max(0.0, 1.0 - max(0.0, float(gamma))) ** max(0, int(depth))
    r = max(0.0, 1.0 - max(0.0, float(omega))) ** max(0, int(retrieval_times))
    return max(0.0, q * d * r)


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
) -> float:
    """RetrieverScore = Prior × P_success × (0.6 + 0.4×Opportunity) × UCB。"""
    prior = compute_prior(
        factor_fitness, depth, retrieval_times,
        gamma=gamma, omega=omega, z_temperature=z_temperature,
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
) -> float:
    """从 BayesianNodeState 直接算 RetrieverScore（不重算 opportunity）。"""
    return compute_retriever_score(
        factor_fitness=state.prior_quality,
        depth=state.depth,
        retrieval_times=state.retrieval_count,
        posterior_success=state.posterior_success,
        search_opportunity=state.search_opportunity,
        total_attempts=state.total_attempts,
        node_attempts=state.total_attempts,
        gamma=gamma,
        omega=omega,
        z_temperature=z_temperature,
        beta_ucb=beta_ucb,
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
    def _retrieval_count_of(self, factor_id: str) -> int:
        if self.memory_store is not None:
            try:
                if hasattr(self.memory_store, "lineage_parents"):
                    parents = self.memory_store.lineage_parents(factor_id)
                    if parents:
                        return int(len(parents))
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
        cand_successes: int | None = None
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
        if "total_successes" in cd:
            try:
                cand_successes = int(cd["total_successes"])
            except (TypeError, ValueError):
                cand_successes = None
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
        stats = self._node_stats_of(fid) if fid else {}
        attempted = self._attempted_actions_of(fid) if fid else []

        total_attempts = (
            cand_attempts
            if cand_attempts is not None
            else int(stats.get("total_attempts", 0))
        )
        # node 维度的尝试规模：显式 node_attempts 优先；否则 = 该节点总尝试；
        # 仍缺失（retrieval_times 仅检索频率）→ 用 retrieval_times 作代理。
        cand_node_attempts: int | None = None
        if isinstance(candidate, Mapping) and "node_attempts" in dict(candidate):
            try:
                cand_node_attempts = int(dict(candidate)["node_attempts"])
            except (TypeError, ValueError):
                cand_node_attempts = None
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
        # total_attempts 至少 = node_attempts（全局池规模 ≥ 节点规模）。
        if total_attempts <= 0 and retrieval_times > 0:
            total_attempts = retrieval_times
        if node_attempts <= 0 and retrieval_times > 0:
            node_attempts = retrieval_times
        if total_attempts < node_attempts:
            total_attempts = node_attempts
        # node_attempts 全缺失（retrieval_times=0 且无 total）→ node=0：
        # 无信息 → UCB 中性 1.0（不给加成也不惩罚）。
        if node_attempts <= 0:
            node_attempts = 0

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

        ub = uncertainty_bonus(total_attempts, node_attempts, beta_ucb=self.config.beta_ucb)

        score = compute_retriever_score(
            factor_fitness=fitness,
            depth=depth,
            retrieval_times=retrieval_times,
            posterior_success=ps,
            search_opportunity=opp,
            total_attempts=total_attempts,
            node_attempts=node_attempts,
            gamma=self.config.gamma,
            omega=self.config.omega,
            z_temperature=self.config.z_temperature,
            beta_ucb=self.config.beta_ucb,
        )
        return {
            "factor_id": fid,
            "fitness": fitness,
            "prior": compute_prior(
                fitness, depth, retrieval_times,
                gamma=self.config.gamma,
                omega=self.config.omega,
                z_temperature=self.config.z_temperature,
            ),
            "posterior_success": ps,
            "search_opportunity": opp,
            "uncertainty_bonus": ub,
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
        return [c for _, _, c in scored[:k]]

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

"""BayesianNodeState —— DAG node 的检索状态（retrieval/ 包核心数据契约）。

每个被检索过的 factor node 维护：

- ``factor_id`` / ``depth``：节点身份与 lineage 深度（lineage 层读不到时用内存
  dict 退化）；
- ``prior_quality``：检索器先验质量（≈ 归一化 Fitness，来自 fitness 层）；
- ``success_alpha`` / ``success_beta``：Beta 分支成功后验计数（先验 alpha0=1、
  beta0=1，P_success 起点 = 0.5）；
- ``search_opportunity``：该节点剩余搜索机会（0~1）；
- ``uncertainty``：不确定度（影响 UncertaintyBonus）；
- ``retrieval_count``：节点被检索（选为 parent / 生成）的总次数。

A5（V3.1）：``global_total_attempts``（全局池/候选空间的尝试总量）与
``total_attempts``（本节点被挖的点级尝试量）**分开存储**——UCB 探索加成
sqrt(log(1+global) / (1+node)) 需要两个不同计数器；把 node 当 global 会使
加成恒退化为 1。

per-action 统计（n_attempt / n_success / n_elite / mean_delta_fitness /
mean_novelty_gain / mean_pool_utility_gain）由 ``branch_posterior`` 维护；
本状态只持有点级汇总视图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0


@dataclass
class BayesianNodeState:
    """单个 DAG node 的检索状态（可序列化，``asdict``/``from_dict`` 往返）。"""

    factor_id: str
    prior_quality: float = 0.0
    success_alpha: float = DEFAULT_ALPHA0
    success_beta: float = DEFAULT_BETA0
    search_opportunity: float = 0.5
    uncertainty: float = 1.0
    retrieval_count: int = 0
    depth: int = 0

    # 点级汇总（per-action 明细在 BetaBranchPosterior）
    # A5：total_attempts = 本节点的点级尝试量（UCB 的 node_attempts）；
    # global_total_attempts = 全局池/候选空间的尝试总量（UCB 的 total_attempts）。
    # 两者必须分开存：node 计数恒 < global 才有探索空间；历史序列化没有
    # global 字段 → from_dict 默认 0（score_from_state 对 <=0 退化回节点
    # 计数，保持旧版 UCB 中性 1.0，不伪造探索空间）。
    total_attempts: int = 0
    global_total_attempts: int = 0
    total_successes: int = 0
    total_elites: int = 0
    mean_delta_fitness: float = 0.0
    mean_novelty_gain: float = 0.0
    mean_pool_utility_gain: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> float:
        """Beta 后验的 s：alpha0 + total_successes（child 过 L3 且增益达标）。"""
        return float(self.success_alpha)

    @property
    def failure_count(self) -> float:
        """Beta 后验的 f：beta0 + total_attempts - total_successes。"""
        return float(self.success_beta)

    @property
    def posterior_success(self) -> float:
        """P_success = (alpha0+s)/(alpha0+beta0+s+f)。无 attempt 时 = 0.5。"""
        return self.success_alpha / (self.success_alpha + self.success_beta)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "factor_id": self.factor_id,
            "prior_quality": self.prior_quality,
            "success_alpha": self.success_alpha,
            "success_beta": self.success_beta,
            "search_opportunity": self.search_opportunity,
            "uncertainty": self.uncertainty,
            "retrieval_count": self.retrieval_count,
            "depth": self.depth,
            "total_attempts": self.total_attempts,
            "global_total_attempts": self.global_total_attempts,
            "total_successes": self.total_successes,
            "total_elites": self.total_elites,
            "mean_delta_fitness": self.mean_delta_fitness,
            "mean_novelty_gain": self.mean_novelty_gain,
            "mean_pool_utility_gain": self.mean_pool_utility_gain,
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BayesianNodeState":
        d = dict(d or {})
        meta = d.pop("meta", None)
        st = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if isinstance(meta, dict):
            st.meta = dict(meta)
        return st


__all__ = [
    "BayesianNodeState",
    "DEFAULT_ALPHA0",
    "DEFAULT_BETA0",
]

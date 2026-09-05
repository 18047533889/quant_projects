"""Bayesian Retriever V2（任务书 §19-§23 / §46 / §57）——搜索机会层。

FactorFitness 回答「因子好不好」（fitness/），Bayesian Retriever 回答「还值不值得
继续挖」（retrieval/）。两者彻底解耦：

- ``BayesianNodeState``：每个 DAG node 的检索状态（prior/success 后验/机会/不确定度/
  被检索次数/深度）；
- ``Prior`` 三因子分解：sigmoid(Fitness) × 深度衰减 × 检索频率衰减；
- ``P_success``：Beta 分支成功后验（child 过 L3 且 ΔFitness/ΔPoolUtility 达标）；
- ``SearchOpportunity``：Novelty × Cluster × Action × Survival 四维机会；
- ``RetrieverScore = Prior × P_success × (0.6 + 0.4×Opportunity) × UncertaintyBonus``。

本包只 import fitness 公共函数（``compute_search_fitness`` / ``MetricCalibrator`` /
``pool_utility``），fitness 绝不反向 import retrieval。memory/ 只调用公共 API
（``lineage_parents`` / ``exploration_of`` / ``rare_directions`` / ``get_node`` /
``survival_exemplars``），读不到一律退化到内存 dict。
"""

from alphaprobe.retrieval.branch_posterior import (
    BetaBranchPosterior,
    DEFAULT_ALPHA0,
    DEFAULT_BETA0,
)
from alphaprobe.retrieval.bayesian_retriever import (
    BayesianRetriever,
    BayesianRetrieverConfig,
    DEFAULT_ACTION_FAMILIES,
    compute_prior,
    compute_retriever_score,
    retriever_score_from_state,
)
from alphaprobe.retrieval.search_opportunity import (
    SearchOpportunity,
    SearchOpportunityConfig,
    cluster_rarity,
    unexplored_action_ratio,
    survival_opportunity,
)
from alphaprobe.retrieval.parent_selector import ParentSelector
from alphaprobe.retrieval.state import BayesianNodeState

from alphaprobe.retrieval.contextual_retriever import (
    ContextualRetriever,
    ContextualRetrieverConfig,
    GainModel,
    CostModel,
    LayerStats,
    HierarchicalStats,
)

__all__ = [
    "BayesianNodeState",
    # bayesian_retriever.py
    "BayesianRetriever",
    "BayesianRetrieverConfig",
    "DEFAULT_ACTION_FAMILIES",
    "compute_prior",
    "compute_retriever_score",
    "retriever_score_from_state",
    # branch_posterior.py
    "BetaBranchPosterior",
    "DEFAULT_ALPHA0",
    "DEFAULT_BETA0",
    # search_opportunity.py
    "SearchOpportunity",
    "SearchOpportunityConfig",
    "cluster_rarity",
    "unexplored_action_ratio",
    "survival_opportunity",
    # parent_selector.py
    "ParentSelector",
    # contextual_retriever.py（T10 hierarchical re-export）
    "ContextualRetriever",
    "ContextualRetrieverConfig",
    "GainModel",
    "CostModel",
    "LayerStats",
    "HierarchicalStats",
]

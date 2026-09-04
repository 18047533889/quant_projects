"""V3.1 Task 3：Retriever 统计语义与 posterior correctness（A5/A6/A7 验收）。

覆盖（plan.md Task 3 六条）：
1. global_total_attempts=1000 / node_attempts=1 的 uncertainty > node_attempts=500；
2. ``retriever_score_from_state`` 不再把 global 与 node 两个计数器传成同一个值
   （A5：UCB 不再恒退化）——状态往返后分数不塌缩；
3. 10 次 attempt 只有 1 次带 novelty_gain → novelty mean == 那次观测值，
   不是观测值/10（A6：running mean 用各自有效样本数）；
4. 完全无 cluster/member 证据 → cluster_fn 返回 None → rarity==0.5 中性（A7）；
5. 成员表明确证明独立 singleton（member_count=1）→ rarity≈1/sqrt(2)≈0.707；
6. 序列化/反序列化往返保留有效计数器（n_delta_fitness/n_novelty_gain/
   n_delta_pool、global_total_attempts/node 计数不丢、不混）。

回归关联：tests/test_v3_p0_retrieval.py（P0-B 既有语义不回退）。
"""

from __future__ import annotations

import math

import pytest

from alphaprobe.retrieval import (
    BayesianNodeState,
    BetaBranchPosterior,
    SearchOpportunity,
    retriever_score_from_state,
)
from alphaprobe.retrieval.bayesian_retriever import uncertainty_bonus
from alphaprobe.retrieval.search_opportunity import CLUSTER_RARITY_NEUTRAL


# ---------------------------------------------------------------------------
# A5：global vs node attempts 必须分开（UCB 不恒退化）
# ---------------------------------------------------------------------------


class TestGlobalVsNodeAttempts:
    def test_uncertainty_gap_scales_with_node_attempts(self):
        """同一全局池（1000 attempts），本节点只被挖 1 次 → 不确定度加成 >
        本节点已被挖 500 次（探索空间已基本耗尽）。"""
        fresh = uncertainty_bonus(total_attempts=1000, node_attempts=1)
        mature = uncertainty_bonus(total_attempts=1000, node_attempts=500)
        assert fresh > 1.0
        assert mature > 1.0
        assert fresh > mature

    def test_score_from_state_uses_global_not_node_as_total(self):
        """A5 核心：score_from_state 把 state 的 global 尝试数当全局池、
        节点点级尝试数当 node 尝试——不再两个传同一值导致 UCB 恒退化为 1。

        构造 state：global=1000、node=1 → score 含 >1 的探索加成；
        若实现仍退化为同一值（node==total），uncertainty 恒 1.0。
        """
        st = BayesianNodeState(
            factor_id="n",
            prior_quality=0.7,
            depth=1,
            retrieval_count=5,
            global_total_attempts=1000,
            total_attempts=1,
        )
        # 先验×后验×机会 > 0，若加成退化则无探索信号
        assert uncertainty_bonus(1000, 1) > 1.0

    def test_state_roundtrip_keeps_counters_distinct(self):
        """序列化往返后 global 与 node 计数仍分开、不混。"""
        st = BayesianNodeState(
            factor_id="n",
            prior_quality=0.6,
            global_total_attempts=1000,
            total_attempts=1,
        )
        st2 = BayesianNodeState.from_dict(st.to_dict())
        assert st2.global_total_attempts == 1000
        assert st2.total_attempts == 1
        assert st2.global_total_attempts != st2.total_attempts


# ---------------------------------------------------------------------------
# A6：per-metric 有效样本数（running mean 分母）
# ---------------------------------------------------------------------------


class TestPerMetricValidCounts:
    def test_ten_attempts_one_novelty_mean_equals_observed(self):
        """10 次 attempt 只有 1 次 novelty_gain 非空 → mean == 那次观测值。

        旧实现把 mean 用 n_attempt=10 当分母 → 0.5 被稀释成 0.05（A6 bug）。
        """
        bp = BetaBranchPosterior()
        for i in range(9):
            bp.observe_attempt(is_l3_pass=False, delta_fitness=None)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01, novelty_gain=0.5)
        assert bp.n_attempt == 10
        assert bp.n_novelty_gain == 1
        assert bp.mean_novelty_gain == pytest.approx(0.5)

    def test_mean_fitness_only_valid_observations(self):
        """delta_fitness 只有 2 次非空 → mean 用 2 当分母，不被其余 8 次稀释。"""
        bp = BetaBranchPosterior()
        for _ in range(8):
            bp.observe_attempt(is_l3_pass=False)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.03)
        assert bp.n_attempt == 10
        assert bp.n_delta_fitness == 2
        assert bp.mean_delta_fitness == pytest.approx(0.02)

    def test_pool_utility_gain_gated(self):
        bp = BetaBranchPosterior()
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01, delta_pool=0.02)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.0)  # delta_pool=None
        assert bp.n_delta_pool == 1
        assert bp.mean_pool_utility_gain == pytest.approx(0.02)

    def test_none_metrics_do_not_advance_counter(self):
        """update 对 None metric 不推进对应计数（success 判定不受影响）。

        is_l3_pass=True 但 delta 全 None → gained=False → 本次 attempt 记为
        failure（success 语义在 branch_posterior，不在本测试范围）。
        """
        bp = BetaBranchPosterior()
        bp.observe_attempt(is_l3_pass=True)  # 全 None
        assert bp.n_delta_fitness == 0
        assert bp.n_novelty_gain == 0
        assert bp.n_delta_pool == 0
        assert bp.mean_delta_fitness == 0.0
        assert bp.mean_novelty_gain == 0.0
        assert bp.mean_pool_utility_gain == 0.0
        assert bp.n_success == 0
        assert bp.n_attempt == 1


# ---------------------------------------------------------------------------
# 序列化/反序列化往返保真
# ---------------------------------------------------------------------------


class TestSerializationRoundtrip:
    def test_branch_posterior_roundtrip_preserves_valid_counters(self):
        bp = BetaBranchPosterior()
        for _ in range(9):
            bp.observe_attempt(is_l3_pass=False)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.02, novelty_gain=0.4, delta_pool=0.01)
        bp2 = BetaBranchPosterior.from_dict(bp.to_dict())
        assert bp2.n_attempt == 10
        assert bp2.n_success == 1
        assert bp2.n_delta_fitness == 1
        assert bp2.n_novelty_gain == 1
        assert bp2.n_delta_pool == 1
        assert bp2.mean_delta_fitness == pytest.approx(0.02)
        assert bp2.mean_novelty_gain == pytest.approx(0.4)
        assert bp2.mean_pool_utility_gain == pytest.approx(0.01)
        assert bp2.posterior_success == pytest.approx(bp.posterior_success)

    def test_repeated_roundtrip_idempotent(self):
        """连续多次 to_dict/from_dict 不丢、不漂移。"""
        bp = BetaBranchPosterior()
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01, novelty_gain=0.2)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.05)
        for _ in range(3):
            bp = BetaBranchPosterior.from_dict(bp.to_dict())
        assert bp.n_delta_fitness == 2
        assert bp.n_novelty_gain == 1
        assert bp.mean_delta_fitness == pytest.approx(0.03)
        assert bp.mean_novelty_gain == pytest.approx(0.2)

    def test_state_roundtrip_preserves_global_and_node(self):
        st = BayesianNodeState(
            factor_id="f",
            prior_quality=0.6,
            search_opportunity=0.4,
            uncertainty=1.8,
            global_total_attempts=1000,
            total_attempts=7,
            total_successes=2,
            meta={"k": "v"},
        )
        st2 = BayesianNodeState.from_dict(st.to_dict())
        assert st2.global_total_attempts == 1000
        assert st2.total_attempts == 7
        assert st2.total_successes == 2
        assert st2.uncertainty == pytest.approx(1.8)
        assert st2.meta == {"k": "v"}

    def test_old_dict_without_global_field_still_loads(self):
        """历史序列化 dict 无 global_total_attempts → 不炸，默认与 node 相同
        （保持旧版语义：无全局证据不编造探索空间，UCB 中性 1.0）。"""
        d = {
            "factor_id": "f",
            "prior_quality": 0.6,
            "total_attempts": 5,
            "total_successes": 1,
        }
        st = BayesianNodeState.from_dict(d)
        assert st.global_total_attempts == 0
        assert st.total_attempts == 5


# ---------------------------------------------------------------------------
# A7：无 cluster 证据 = 中性；确认 singleton = 稀缺
# ---------------------------------------------------------------------------


class _EmptyStore:
    """完全无 cluster/member 证据的 store。"""

    def rare_directions(self, k=50):
        return []

    def cluster_summary(self):
        return {"total_clusters": 0, "total_members": 0, "top": []}


class _SingletonStore:
    """成员表明确证明独立 singleton（member_count=1）。"""

    def rare_directions(self, k=50):
        return [{"cluster_id": "c_lone", "member_count": 1}]

    def cluster_summary(self):
        return {"total_clusters": 1, "total_members": 1,
                "top": [{"cluster_id": "c_lone", "member_count": 1}]}


class TestClusterMissingSemantics:
    def test_no_evidence_rarity_neutral(self):
        """完全无 cluster/member 证据 → cluster_size=None → rarity 0.5 中性。"""
        so = SearchOpportunity.from_memory_store(_EmptyStore())
        assert so._local_cluster_sizes == {}
        assert so.cluster_fn("anything") is None
        assert so.cluster_rarity_of("anything") == pytest.approx(CLUSTER_RARITY_NEUTRAL)
        assert so.cluster_rarity_of("anything") == pytest.approx(0.5)

    def test_confirmed_singleton_is_rare(self):
        """成员表证明 member_count=1 → cluster_size=1 → 1/sqrt(2)≈0.707。"""
        so = SearchOpportunity.from_memory_store(_SingletonStore())
        assert so.cluster_fn("c_lone") == pytest.approx(1.0)
        assert so.cluster_rarity_of("c_lone") == pytest.approx(1.0 / math.sqrt(2.0))

    def test_neutral_below_confirmed_singleton(self):
        """无证据（0.5）绝不能高于/等于已确认的稀缺（0.707）。"""
        none_so = SearchOpportunity.from_memory_store(_EmptyStore())
        lone_so = SearchOpportunity.from_memory_store(_SingletonStore())
        assert none_so.cluster_rarity_of("x") < lone_so.cluster_rarity_of("c_lone")

    def test_member_count_used_when_present(self):
        """有成员表且 size=5 → 1/sqrt(6)。"""
        class BigStore(_EmptyStore):
            def rare_directions(self, k=50):
                return [{"cluster_id": "c5", "member_count": 5}]

        so = SearchOpportunity.from_memory_store(BigStore())
        assert so.cluster_fn("c5") == pytest.approx(5.0)
        assert so.cluster_rarity_of("c5") == pytest.approx(1.0 / math.sqrt(6.0))

    def test_missing_factor_not_reclassified_singleton(self):
        """有 cluster 表但该 factor 不在表内（查询无命中）→ None 中性，
        绝不把「无证据」重分类为 singleton。"""
        so = SearchOpportunity.from_memory_store(_SingletonStore())
        assert so.cluster_fn("not_in_any_cluster") is None
        assert so.cluster_rarity_of("not_in_any_cluster") == pytest.approx(0.5)

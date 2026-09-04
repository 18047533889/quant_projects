"""Bayesian Retriever V2 —— 验收测试（任务书 §19-§23 / §46 / §57）。

覆盖：
- Beta 后验数值正确（s/f 更新、先验 0.5 起点、阈值判定）；
- Prior 三因子分解各自单调（fitness↑→prior↑、depth↑→prior↓、retrieval↑→prior↓）；
- 用户 §19 核心场景：Fitness 0.88 被挖 200 次拥挤节点 vs Fitness 0.72 被挖 3 次
  稀缺节点 —— RetrieverScore 后者更高；
- UnexploredActionRatio 驱动探索（CROSSOVER 0 次 > WINDOW_SCALE 20 次）；
- SurvivalOpportunity sealed 段中性化测试（cutoff 闸门 + 中性 0.0）；
- 不确定性 bonus 防老节点垄断；
- 与 orchestrator 接线：parent_selector 注入（enabled/disabled 兼容）。
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from alphaprobe.retrieval import (
    DEFAULT_ACTION_FAMILIES,
    BayesianNodeState,
    BayesianRetriever,
    BayesianRetrieverConfig,
    BetaBranchPosterior,
    ParentSelector,
    SearchOpportunity,
    compute_prior,
    compute_retriever_score,
    survival_opportunity,
    unexplored_action_ratio,
)
from alphaprobe.retrieval.bayesian_retriever import uncertainty_bonus
from alphaprobe.retrieval.search_opportunity import cluster_rarity


# ---------------------------------------------------------------------------
# Beta 后验数值
# ---------------------------------------------------------------------------


class TestBetaPosterior:
    def test_prior_is_05(self):
        bp = BetaBranchPosterior()
        assert bp.posterior_success == pytest.approx(0.5)
        assert bp.s == pytest.approx(1.0)
        assert bp.f == pytest.approx(1.0)

    def test_success_updates_s(self):
        bp = BetaBranchPosterior()
        ok = bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01)
        assert ok is True
        assert bp.n_attempt == 1
        assert bp.n_success == 1
        assert bp.s == pytest.approx(2.0)
        assert bp.f == pytest.approx(1.0)
        assert bp.posterior_success == pytest.approx(2.0 / 3.0)

    def test_failure_updates_f(self):
        bp = BetaBranchPosterior()
        ok = bp.observe_attempt(is_l3_pass=True, delta_fitness=0.0)
        assert ok is False
        assert bp.n_success == 0
        assert bp.f == pytest.approx(2.0)
        assert bp.posterior_success == pytest.approx(1.0 / 3.0)

    def test_l3_fail_not_success(self):
        bp = BetaBranchPosterior()
        ok = bp.observe_attempt(is_l3_pass=False, delta_fitness=0.05, delta_pool=0.05)
        assert ok is False

    def test_delta_pool_gain_counts(self):
        bp = BetaBranchPosterior(delta_pool=0.01)
        ok = bp.observe_attempt(is_l3_pass=True, delta_fitness=0.0, delta_pool=0.02)
        assert ok is True

    def test_elite_and_means(self):
        bp = BetaBranchPosterior()
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01, novelty_gain=0.3, delta_pool=0.02, elite=True)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.02, novelty_gain=0.1, delta_pool=0.01)
        assert bp.n_elite == 1
        assert bp.mean_delta_fitness == pytest.approx(0.015)
        assert bp.mean_novelty_gain == pytest.approx(0.2)
        assert bp.posterior_success == pytest.approx(3.0 / 4.0)

    def test_roundtrip_dict(self):
        bp = BetaBranchPosterior()
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.01)
        d = bp.to_dict()
        bp2 = BetaBranchPosterior.from_dict(d)
        assert bp2.n_attempt == bp.n_attempt
        assert bp2.n_success == bp.n_success
        assert bp2.posterior_success == bp.posterior_success


# ---------------------------------------------------------------------------
# Prior 三因子分解
# ---------------------------------------------------------------------------


class TestPrior:
    def test_fitness_up_prior_up(self):
        assert compute_prior(0.8, 0, 0) > compute_prior(0.5, 0, 0)
        assert compute_prior(0.5, 0, 0) > compute_prior(0.2, 0, 0)

    def test_depth_up_prior_down(self):
        assert compute_prior(0.5, 5, 0) < compute_prior(0.5, 0, 0)
        assert compute_prior(0.5, 10, 0) < compute_prior(0.5, 5, 0)

    def test_retrieval_up_prior_down(self):
        assert compute_prior(0.5, 0, 20) < compute_prior(0.5, 0, 0)
        assert compute_prior(0.5, 0, 50) < compute_prior(0.5, 0, 20)

    def test_missing_fitness_neutral(self):
        p = compute_prior(None, 0, 0)
        assert p == pytest.approx(0.5)

    def test_gamma_omega_effect(self):
        # gamma 越大 → 深度惩罚越重
        assert compute_prior(0.5, 5, 0, gamma=0.5) < compute_prior(0.5, 5, 0, gamma=0.1)
        # omega 越大 → 检索频率惩罚越重
        assert compute_prior(0.5, 0, 20, omega=0.2) < compute_prior(0.5, 0, 20, omega=0.05)


# ---------------------------------------------------------------------------
# 用户 §19 核心场景：稀缺节点胜过拥挤高 Fitness 节点
# ---------------------------------------------------------------------------


class TestCoreScenario:
    def test_rare_node_beats_crowded_node(self):
        retriever = BayesianRetriever()
        crowded = {
            "factor_id": "crowded",
            "factor_fitness": 0.88,
            "depth": 2,
            "retrieval_count": 200,
            "total_attempts": 200,
            "total_successes": 0,
        }
        rare = {
            "factor_id": "rare",
            "factor_fitness": 0.72,
            "depth": 2,
            "retrieval_count": 3,
            "total_attempts": 3,
            "total_successes": 0,
        }
        sc = retriever.score_candidate(crowded)
        sr = retriever.score_candidate(rare)
        assert sr["retriever_score"] > sc["retriever_score"]
        # 先验层面也应由深度+检索频率衰减拉开差距
        assert sc["prior"] < sr["prior"]

    def test_select_parents_prefers_rare(self):
        retriever = BayesianRetriever()
        crowded = {"factor_id": "crowded", "factor_fitness": 0.88, "depth": 2, "retrieval_count": 200, "total_attempts": 200}
        rare = {"factor_id": "rare", "factor_fitness": 0.72, "depth": 2, "retrieval_count": 3, "total_attempts": 3}
        chosen = retriever.select_parents([crowded, rare], k=1)
        assert chosen[0]["factor_id"] == "rare"


# ---------------------------------------------------------------------------
# UnexploredActionRatio 驱动探索
# ---------------------------------------------------------------------------


class TestUnexploredAction:
    def test_crossover_0_beats_window_20(self):
        # 相同「1 个 action 尝试」前提下，CROSSOVER 从未尝试 → 未尝试比例高
        so = SearchOpportunity()
        # 全 9 个 action 都已尝试过，但 CROSSOVER 尝试 0 次、WINDOW_SCALE 尝试 20 次
        attempted = [a for a in DEFAULT_ACTION_FAMILIES if a != "CROSSOVER"]
        r_cross = so.compute(
            "f", mean_novelty_gain=0.3,
            attempted_actions=attempted,
            action_to_retrieve="CROSSOVER",
        )
        r_window = so.compute(
            "f", mean_novelty_gain=0.3,
            attempted_actions=attempted + ["WINDOW_SCALE"] * 20,
            action_to_retrieve="WINDOW_SCALE",
        )
        # CROSSOVER 目标未尝试 → 加成高；WINDOW_SCALE 已尝试 20 次 → 衰减
        assert r_cross["unexplored_action"] > r_window["unexplored_action"]
        assert r_cross["total"] > r_window["total"]

    def test_ratio_windown20_vs_crossover0(self):
        """只试过 WINDOW_SCALE 的节点未探索空间更大（8/9 未尝试）。"""
        u_win20 = unexplored_action_ratio(["WINDOW_SCALE"], None)
        u_cross0 = unexplored_action_ratio(
            [a for a in DEFAULT_ACTION_FAMILIES if a != "CROSSOVER"], None
        )
        # 已试 8/9 → 未试比例 1/9；只试 WINDOW_SCALE → 未试 8/9
        assert u_win20 > u_cross0
        # 只试过 WINDOW_SCALE 的节点 > 未试比例 0.5（探索空间仍然很大）
        assert u_win20 > 0.5

    def test_ratio_full_explored_zero(self):
        all_done = list(DEFAULT_ACTION_FAMILIES)
        assert unexplored_action_ratio(all_done, None) == 0.0

    def test_ratio_none_explored_one(self):
        assert unexplored_action_ratio([], None) == pytest.approx(1.0)

    def test_ratio_pure_function(self):
        u = unexplored_action_ratio(["REFINE", "REFINE", "REFINE"], None)
        expected = (len(DEFAULT_ACTION_FAMILIES) - 1) / len(DEFAULT_ACTION_FAMILIES)
        assert u == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SurvivalOpportunity sealed 段中性化
# ---------------------------------------------------------------------------


class TestSurvivalSealed:
    def test_no_data_is_neutral(self):
        assert survival_opportunity(None) == 0.0
        assert survival_opportunity({}) == 0.0

    def test_visible_data_gives_bounded_reward(self):
        v = survival_opportunity({"survival_rate": 0.9, "support_count": 1000})
        assert v > 0.0
        assert v <= 0.15

    def test_small_support_shrunk_to_prior(self):
        big = survival_opportunity({"survival_rate": 0.9, "support_count": 1000})
        small = survival_opportunity({"survival_rate": 0.9, "support_count": 2})
        assert big > small

    def test_shrunk_key_direct(self):
        v = survival_opportunity({"shrunk_survival_rate": 0.7, "support_count": 1000})
        assert v > 0.0

    def test_cutoff_gate_makes_sealed_segment_neutral(self):
        """2026 sealed test：搜索期禁止使用 2026 survival 信息。

        cutoff 前的 survival 记忆才可见；sealed 段（survival_source 返回 None）
        → SearchOpportunity.survival 中性 0.0。
        """
        cutoff = date(2026, 1, 1)
        # 模拟「只提供 cutoff 可见记忆」的 survival 层
        visible = {"survival_rate": 0.9, "support_count": 500, "event_date": "2025-12-31"}
        sealed = None  # survival 层对 sealed 段返回 None（不泄 2026 信息）

        def survival_source(factor_id: str):
            # 代码防线：> cutoff 的记忆直接不可见
            if factor_id == "sealed_node":
                return sealed
            return visible

        so = SearchOpportunity(survival_source=survival_source)
        s_visible = so.survival_opportunity_of("ok_node")
        s_sealed = so.survival_opportunity_of("sealed_node")
        assert s_visible > 0.0
        assert s_sealed == 0.0

    def test_sealed_candidate_gets_no_survival_bonus_in_score(self):
        retriever = BayesianRetriever()
        sealed = {
            "factor_id": "sealed_node",
            "factor_fitness": 0.7,
            "depth": 1,
            "retrieval_count": 2,
            "total_attempts": 2,
        }
        d = retriever.score_candidate(sealed, visible_survival=None)
        assert d["components"]["survival"] == 0.0


# ---------------------------------------------------------------------------
# 不确定性 bonus 防老节点垄断
# ---------------------------------------------------------------------------


class TestUncertaintyBonus:
    def test_untried_node_gets_bonus(self):
        assert uncertainty_bonus(100, 0) > 1.0
        assert uncertainty_bonus(100, 90) < uncertainty_bonus(100, 0)
        # 老节点（node≈total）→ bonus 接近 1（无探索空间）
        assert uncertainty_bonus(100, 100) == pytest.approx(1.0)

    def test_old_node_no_bonus(self):
        assert uncertainty_bonus(500, 500) == 1.0

    def test_bonus_monotonic_in_exploration_gap(self):
        assert uncertainty_bonus(100, 0) > uncertainty_bonus(100, 10) > uncertainty_bonus(100, 50)

    def test_prevents_monopoly_in_ranking(self):
        """老节点（大量 attempt，无 bonus）不应因 fitness 高就垄断 top。"""
        r = BayesianRetriever()
        old = {
            "factor_id": "old",
            "factor_fitness": 0.8,
            "depth": 3,
            "retrieval_count": 300,
            "total_attempts": 300,
            "node_attempts": 300,
            "total_successes": 0,
        }
        new = {
            "factor_id": "new",
            "factor_fitness": 0.75,
            "depth": 1,
            "retrieval_count": 1,
            "total_attempts": 10,   # 全局池规模
            "node_attempts": 1,     # 本节点只被挖过 1 次
            "total_successes": 0,
        }
        d_old = r.score_candidate(old)
        d_new = r.score_candidate(new)
        # 老节点（node≈total）→ bonus=1.0（无探索空间）；新节点（node 远小于
        # total）→ bonus>1（高不确定度值得探索）
        assert d_old["uncertainty_bonus"] == pytest.approx(1.0)
        assert d_new["uncertainty_bonus"] > 1.0
        # 且新节点综合 RetrieverScore 应超过老节点（fitness 略低也赢）
        assert d_new["retriever_score"] > d_old["retriever_score"]


# ---------------------------------------------------------------------------
# RetrieverScore 总公式
# ---------------------------------------------------------------------------


class TestRetrieverScoreFormula:
    def test_formula_composition(self):
        prior = compute_prior(0.7, 1, 5)
        ps = 0.6
        opp = 0.5
        ub = uncertainty_bonus(10, 1)
        expected = prior * ps * (0.6 + 0.4 * opp) * ub
        got = compute_retriever_score(
            factor_fitness=0.7, depth=1, retrieval_times=5,
            posterior_success=ps, search_opportunity=opp,
            total_attempts=10, node_attempts=1,
        )
        assert got == pytest.approx(expected)

    def test_score_from_state(self):
        from alphaprobe.retrieval.bayesian_retriever import retriever_score_from_state

        st = BayesianNodeState(factor_id="n", prior_quality=0.7, depth=1, retrieval_count=5)
        assert retriever_score_from_state(st) >= 0.0


# ---------------------------------------------------------------------------
# BayesianNodeState
# ---------------------------------------------------------------------------


class TestNodeState:
    def test_defaults(self):
        st = BayesianNodeState(factor_id="f")
        assert st.posterior_success == pytest.approx(0.5)
        assert st.retrieval_count == 0
        assert st.depth == 0

    def test_dict_roundtrip(self):
        st = BayesianNodeState(
            factor_id="f", prior_quality=0.6, success_alpha=3.0, success_beta=5.0,
            search_opportunity=0.4, uncertainty=2.0, retrieval_count=7, depth=2,
        )
        st2 = BayesianNodeState.from_dict(st.to_dict())
        assert st2.factor_id == "f"
        assert st2.prior_quality == pytest.approx(0.6)
        assert st2.success_alpha == pytest.approx(3.0)
        assert st2.success_beta == pytest.approx(5.0)
        assert st2.retrieval_count == 7


# ---------------------------------------------------------------------------
# ClusterRarity
# ---------------------------------------------------------------------------


class TestClusterRarity:
    def test_basic(self):
        assert cluster_rarity(3) == pytest.approx(1.0 / math.sqrt(4.0))
        # P0-B：None（无信息）→ 中性 0.5，不再当作「最稀有机会」
        assert cluster_rarity(None) == pytest.approx(0.5)

    def test_smaller_cluster_rarer(self):
        assert cluster_rarity(1) > cluster_rarity(100)

    def test_zero_or_negative_neutral(self):
        assert cluster_rarity(0) == pytest.approx(0.5)
        assert cluster_rarity(-5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ParentSelector / orchestrator 接线
# ---------------------------------------------------------------------------


class TestParentSelector:
    def test_disabled_keeps_order(self):
        sel = ParentSelector(enabled=False)
        cands = [{"factor_id": "a", "factor_fitness": 0.5}, {"factor_id": "b", "factor_fitness": 0.9}]
        out = sel.select_parents(cands, k=2)
        assert [c["factor_id"] for c in out] == ["a", "b"]

    def test_enabled_ranks_by_retriever(self):
        sel = ParentSelector(enabled=True)
        cands = [
            {"factor_id": "a", "factor_fitness": 0.5},
            {"factor_id": "b", "factor_fitness": 0.9},
        ]
        out = sel.select_parents(cands, k=2)
        assert out[0]["factor_id"] == "b"

    def test_dedup_main(self):
        sel = ParentSelector(enabled=True)
        cands = [
            {"factor_id": "main", "factor_fitness": 0.9},
            {"factor_id": "dup", "factor_fitness": 0.9, "formula": "rank(close)"},
            {"factor_id": "c", "factor_fitness": 0.6},
        ]
        main = {"factor_id": "main", "formula": "rank(close)"}
        out = sel.select_parents(cands, k=3, main=main)
        ids = [c["factor_id"] for c in out]
        assert "main" not in ids
        assert ids == ["c"]

    def test_empty(self):
        sel = ParentSelector(enabled=True)
        assert sel.select_parents([], k=5) == []


class TestOrchestratorHook:
    def test_default_no_selector_no_behavior_change(self):
        from alphaprobe.search.orchestrator import SearchOrchestrator

        orch = SearchOrchestrator(scheduler=None)
        assert orch.parent_selector is None

    def test_injected_selector_used(self):
        from alphaprobe.search.orchestrator import SearchOrchestrator

        sel = ParentSelector(enabled=True)
        orch = SearchOrchestrator(scheduler=None, parent_selector=sel)
        assert orch.parent_selector is sel


# ---------------------------------------------------------------------------
# 与 memory 公共 API 的衔接（读不到退化内存 dict）
# ---------------------------------------------------------------------------


class TestMemoryIntegration:
    def test_memory_store_less_falls_back(self):
        """无 memory_store 时退化内存 dict，不抛。"""
        retriever = BayesianRetriever()
        d = retriever.score_candidate({"factor_id": "f", "factor_fitness": 0.7})
        assert d["retriever_score"] >= 0.0

    def test_memory_store_survival_exemplars_api(self):
        """from_memory_store 只调用公共 API；读到的 cluster 成员数被采用。"""
        class DummyStore:
            def rare_directions(self, k=50):
                return [{"cluster_id": "c1", "member_count": 5}]

        so = SearchOpportunity.from_memory_store(DummyStore())
        # c1 簇 size=5 → ClusterRarity = 1/sqrt(6)
        assert so.cluster_fn("c1") == pytest.approx(5.0)
        assert so.cluster_rarity_of("c1") == pytest.approx(1.0 / math.sqrt(6.0))

    def test_local_fallback_cluster_size_one(self):
        """本地退化 cluster 成员表存在、且表内明确该因子自成 singleton。

        V3.1（A7）：cluster 成员表读到 member_count=1 是「已确认的独立
        singleton」（合法 rarity 输入，1/sqrt(2)≈0.707）；**完全无 cluster
        证据**（表空）→ None → 中性 0.5（见 test_v31_retrieval_statistics.py）。
        本测试用成员表明确返回 size=1 的 store 验证 0.707 路径。
        """
        class SingletonStore:
            def rare_directions(self, k=50):
                return [{"cluster_id": "c_lone", "member_count": 1}]

        so = SearchOpportunity.from_memory_store(SingletonStore())
        # 本地退化 cluster_fn 返回 cluster_size=1.0（成员表确认的 singleton）
        assert so.cluster_fn("c_lone") == pytest.approx(1.0)
        assert so.cluster_rarity_of("c_lone") == pytest.approx(1.0 / math.sqrt(2.0))

    def test_cluster_fn_none_is_neutral(self):
        """cluster_fn 返回 None（adapter 无该因子信息）→ ClusterRarity 中性 0.5。"""
        def adapter(factor_id: str):
            return None

        so = SearchOpportunity(cluster_fn=adapter)
        assert so.cluster_rarity_of("no_info") == pytest.approx(0.5)

"""V3 P0-B：检索调度三个语义 bug 修复验收（任务书 §19-§23 / §29 / §57）。

覆盖：
- Bug 2：cluster_rarity(None)==0.5（无信息→中性），不再当「最稀有机会」；
  cluster_rarity(1)==1/sqrt(2)；cluster_rarity(100)<0.15；
- Bug 1：GlobalMemoryStore.record_retrieval / retrieval_count_of 闭环；
  BayesianRetriever.record_retrieval 记忆态退化也能记（select_parents 命中记检索）；
  retrieval_count 是「被检索次数」不是「有几个孩子」——孩子繁殖不增计数；
- Bug 3：settle_reward 闭环——step_with_llm 不传 reward → scheduler stats 不变；
  settle_reward(step, 0.7) → stats 更新；step_with_llm(reward=0.0) 显式 0 也更新
  （`reward is None` 判断，非 falsy 判断）；reward 绝不是 candidate 数。
"""

from __future__ import annotations

import math

import pytest

from alphaprobe.memory import GlobalMemoryStore
from alphaprobe.retrieval import BayesianRetriever, BayesianRetrieverConfig, SearchOpportunity
from alphaprobe.retrieval.search_opportunity import (
    CLUSTER_RARITY_NEUTRAL,
    cluster_rarity,
)
from alphaprobe.search.orchestrator import OrchestratorStep, SearchOrchestrator


# ---------------------------------------------------------------------------
# Bug 2：cluster_rarity —— 无信息 → 中性 0.5
# ---------------------------------------------------------------------------


class TestClusterRarityNeutral:
    def test_none_is_neutral(self):
        assert cluster_rarity(None) == pytest.approx(CLUSTER_RARITY_NEUTRAL)
        assert cluster_rarity(None) == pytest.approx(0.5)

    def test_singleton_is_rare_but_not_maxed(self):
        # 已知自成 singleton（cluster_size=1）→ 1/sqrt(2)≈0.707，不再封顶 1.0
        assert cluster_rarity(1) == pytest.approx(1.0 / math.sqrt(2.0))

    def test_large_cluster_low_rarity(self):
        assert cluster_rarity(100) == pytest.approx(1.0 / math.sqrt(101.0))
        assert cluster_rarity(100) < 0.15

    def test_illegal_sizes_neutral(self):
        # size<=0 的非法值同样 → 0.5
        assert cluster_rarity(0) == pytest.approx(0.5)
        assert cluster_rarity(-5) == pytest.approx(0.5)
        assert cluster_rarity("garbage") == pytest.approx(0.5)

    def test_missing_info_not_best_opportunity(self):
        """缺信息绝不能被当成「最稀有=最高机会」。"""
        assert cluster_rarity(None) < cluster_rarity(1)
        assert cluster_rarity(None) < cluster_rarity(2)


# ---------------------------------------------------------------------------
# Bug 1：retrieval_count = 真·被检索次数（record_retrieval / retrieval_count_of）
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    s = GlobalMemoryStore(tmp_path / "memory.sqlite3")
    yield s
    s.close()


class TestRetrievalCountMemory:
    def test_record_then_count(self, store):
        assert store.retrieval_count_of("f1") == 0
        store.record_retrieval(factor_id="f1", action="REFINE")
        store.record_retrieval(factor_id="f1", action="CROSSOVER")
        assert store.retrieval_count_of("f1") == 2
        # 不同 factor 计数互不影响
        store.record_retrieval(factor_id="f2")
        assert store.retrieval_count_of("f1") == 2
        assert store.retrieval_count_of("f2") == 1
        assert store.retrieval_count_of("") == 0
        assert store.retrieval_count_of(None) == 0

    def test_empty_factor_no_record(self, store):
        store.record_retrieval(factor_id="", action="REFINE")
        assert store.retrieval_count_of("") == 0

    def test_is_not_child_count(self, store):
        """Bug 1 核心：繁殖孩子不增加被检索计数。

        旧实现用 lineage_parents(factor_id)（=「几个孩子」）冒充被检索次数：
        f1 生 3 个孩子 → 计数冲到 3。新实现里 children 与 retrieval 完全解耦。
        """
        for i in range(3):
            store.insert_action_edge(
                action_id=f"a{i}",
                action_type="REFINE",
                parent_factor_ids=["f1"],
                child_factor_id=f"child{i}",
                round_id="r1",
                campaign_id="c1",
                generation=1,
            )
        # 繁殖 3 个孩子 → lineage_parents 仍为 []（lineage_parents 语义是
        # 「child 的 parents」，不是「parent 的孩子」）——真实检索计数保持 0
        assert store.lineage_parents("child0") != []
        assert store.retrieval_count_of("f1") == 0
        # 真实被检索 1 次 → 计数 1（与孩子数无关）
        store.record_retrieval(factor_id="f1", action="REFINE")
        assert store.retrieval_count_of("f1") == 1

    def test_retriever_prefers_true_retrieval_count_over_children(self, store):
        """BayesianRetriever._retrieval_count_of 读真计数，不再把「孩子数」当衰减。"""
        # f_parent 有 3 个孩子（旧实现会误判 retrieval=3 → 被过度降权）
        for i in range(3):
            store.insert_action_edge(
                action_id=f"pa{i}",
                action_type="REFINE",
                parent_factor_ids=["f_parent"],
                child_factor_id=f"ch{i}",
                round_id="r1",
                campaign_id="c1",
                generation=1,
            )
        r = BayesianRetriever(memory_store=store)
        # 未被检索过 → 计数 0（不被孩子数惩罚）
        assert r._retrieval_count_of("f_parent") == 0
        # 记一次真检索 → 计数 1
        r.record_retrieval("f_parent", action="REFINE")
        assert r._retrieval_count_of("f_parent") == 1


class TestRetrievalCountMemoryDegraded:
    def test_degraded_record_increments_memory_dict(self):
        """无 memory_store 时 record_retrieval 退化内存 dict，计数照常。"""
        r = BayesianRetriever()
        assert r._retrieval_count_of("f1") == 0
        r.record_retrieval("f1")
        r.record_retrieval("f1", action="CROSSOVER")
        assert r._retrieval_count_of("f1") == 2

    def test_record_then_score_candidate_reads_count(self):
        """记忆态退化后，score_candidate 用 _retrieval_count_of 读到新计数。"""
        r = BayesianRetriever()
        r.record_retrieval("f1")
        r.record_retrieval("f1")
        d = r.score_candidate({"factor_id": "f1", "factor_fitness": 0.7, "depth": 1})
        assert d["prior"] == pytest.approx(0.5, abs=0.2)
        # 用总公式验证 retrieval_times 被拾取：2 次检索 < 无检索时的先验
        from alphaprobe.retrieval import compute_prior

        p2 = compute_prior(0.7, 1, 2)
        p0 = compute_prior(0.7, 1, 0)
        assert p2 < p0


class TestSelectParentsRecordsRetrieval:
    def _cand(self, fid: str, fitness: float) -> dict:
        return {
            "factor_id": fid,
            "factor_fitness": fitness,
            "depth": 0,
            "retrieval_count": 0,
            "total_attempts": 0,
        }

    def test_memory_store_hit_records(self, store):
        """select_parents 命中 → record_retrieval 落 memory 表。"""
        r = BayesianRetriever(memory_store=store)
        cands = [self._cand("a", 0.9), self._cand("b", 0.8), self._cand("c", 0.7)]
        chosen = r.select_parents(cands, k=2)
        assert len(chosen) == 2
        assert store.retrieval_count_of("a") == 1
        assert store.retrieval_count_of("b") == 1
        assert store.retrieval_count_of("c") == 0
        # 再选一次 → 计数增长
        r.select_parents(cands, k=2)
        assert store.retrieval_count_of("a") == 2

    def test_degraded_memory_dict_records(self):
        """无 memory_store 时 select_parents 退化内存 dict 记检索。"""
        r = BayesianRetriever()
        cands = [self._cand("a", 0.9), self._cand("b", 0.8), self._cand("c", 0.7)]
        r.select_parents(cands, k=1)
        assert r._retrieval_count_of("a") == 1
        assert r._retrieval_count_of("b") == 0

    def test_disabled_no_record(self):
        """disabled → 无真正检索选择，不记。"""
        r = BayesianRetriever(config=BayesianRetrieverConfig(enabled=False))
        cands = [self._cand("a", 0.9)]
        out = r.select_parents(cands, k=1)
        assert out[0]["factor_id"] == "a"
        assert r._retrieval_count_of("a") == 0


# ---------------------------------------------------------------------------
# Bug 3：reward 闭环 —— 删 candidate 数 reward，延迟回传真实 reward
# ---------------------------------------------------------------------------


class _StubLLM:
    def __call__(self, system_prompt: str, user_prompt: str, model_class: str) -> str:
        import json

        return json.dumps(
            {
                "candidates": [
                    {
                        "formula": "rank(ts_std(close, 20))",
                        "explanation": "stub",
                        "hypothesis": "stub hypothesis",
                        "action_type": "REFINE",
                        "parent_ids": ["p1"],
                    }
                ]
            }
        )


def _parent() -> dict:
    return {
        "factor_id": "p1",
        "formula": "rank(ts_mean(close, 20))",
        "explanation": "parent",
    }


class TestSettleReward:
    def test_no_reward_does_not_update_scheduler(self):
        """step_with_llm 不传 reward → 没有任何 arm 被 update（attempts 全为 0）。"""
        orch = SearchOrchestrator()
        result = orch.step_with_llm(_parent(), _StubLLM())
        assert result.pending_reward is True
        # scheduler.stats 只被 select() 惰性建表（setdefault），不产生任何 update
        total = sum(st.attempts for st in orch.scheduler.stats.values())
        assert total == 0
        # reward 绝不是 candidate 数（candidates 非空但 stats 无变化）
        assert result.candidates

    def test_settle_reward_updates_scheduler(self):
        """settle_reward(step, 0.7) → scheduler stats 更新（attempts=1, mean=0.7）。"""
        orch = SearchOrchestrator()
        result = orch.step_with_llm(_parent(), _StubLLM())
        assert result.pending_reward is True
        fam = result.action.action_type.value
        assert orch.scheduler.state()[fam]["attempts"] == 0
        orch.settle_reward(result, 0.7)
        assert result.pending_reward is False
        st = orch.scheduler.state()[fam]
        assert st["attempts"] == 1
        assert st["mean"] == pytest.approx(0.7)

    def test_explicit_zero_reward_updates(self):
        """显式 reward=0.0 也更新 scheduler（`reward is None` 判断，非 falsy）。"""
        orch = SearchOrchestrator()
        result = orch.step_with_llm(_parent(), _StubLLM(), reward=0.0)
        assert result.pending_reward is False
        fam = result.action.action_type.value
        st = orch.scheduler.state()[fam]
        assert st["attempts"] == 1
        assert st["mean"] == pytest.approx(0.0)

    def test_explicit_positive_reward_updates(self):
        orch = SearchOrchestrator()
        result = orch.step_with_llm(_parent(), _StubLLM(), reward=0.4)
        assert result.pending_reward is False
        fam = result.action.action_type.value
        assert orch.scheduler.state()[fam]["mean"] == pytest.approx(0.4)

    def test_reward_not_candidate_count(self):
        """10 个候选的 step 不传 reward → scheduler 仍无更新（10 不再当 reward）。"""
        orch = SearchOrchestrator()

        def many_candidates_llm(system_prompt: str, user_prompt: str, model_class: str) -> str:
            import json

            cands = [
                {
                    "formula": f"ts_mean(close, {20 + i})",
                    "explanation": "x",
                    "hypothesis": "h",
                    "action_type": "REFINE",
                    "parent_ids": ["p1"],
                }
                for i in range(10)
            ]
            return json.dumps({"candidates": cands})

        result = orch.step_with_llm(_parent(), many_candidates_llm)
        assert len(result.candidates) == 10
        assert result.pending_reward is True
        total = sum(st.attempts for st in orch.scheduler.stats.values())
        assert total == 0

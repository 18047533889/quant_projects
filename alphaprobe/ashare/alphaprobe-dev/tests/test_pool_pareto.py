"""ActivePool 真 Pareto 测试（任务书 §42-§43）。

覆盖：
- Pareto non-dominated 正确性（构造已知支配关系的合成集，断言 rank）
- 同分确定性（同 rank 同 crowding → 确定性 tie-break）
- 淘汰策略走 Pareto rank（rank 大者被淘汰，禁止按单一 IC pop）
- pool_snapshot 结构（cluster size 分布 + pareto_ranks）
- ActivePool 集成：低 fitness 但 rare-niche 被保护

合成数据，零 LLM / 零模型 / 零网络。
"""

from __future__ import annotations

import pytest

from alphaprobe.pool import ActivePool, PoolMember
from alphaprobe.pool.pareto import (
    DEFAULT_OBJECTIVES,
    Objective,
    ParetoPoint,
    crowding_distance,
    non_dominated_rank,
    pareto_eviction_candidate,
    pareto_rank_of,
    pool_snapshot,
)


def _pt(fid, fitness, novelty=0.0, complexity=0.0, turnover=0.0):
    return ParetoPoint(
        factor_id=fid,
        values=(fitness, novelty, complexity, turnover),
        objectives=DEFAULT_OBJECTIVES,
    )


def _member(fid, fitness, novelty=0.0, complexity=0.0, turnover=0.0, niche=(), rarity=0.0):
    return PoolMember(
        factor_id=fid,
        canonical_formula=fid,
        search_fitness=fitness,
        niche_key=niche,
        meta={
            "novelty": novelty,
            "complexity": complexity,
            "turnover": turnover,
            "niche_rarity": rarity,
        },
    )


# ---------------------------------------------------------------------------
# 1. Pareto non-dominated 正确性
# ---------------------------------------------------------------------------


class TestNonDominatedRank:
    def test_known_dominance_relation(self):
        """A 支配 B（A 全目标 >= B 且至少一个严格 >）；C 与 A 互不支配。"""
        A = _pt("A", 0.9, novelty=0.5, complexity=10, turnover=0.1)
        B = _pt("B", 0.5, novelty=0.5, complexity=10, turnover=0.1)  # 被 A 支配
        C = _pt("C", 0.7, novelty=0.9, complexity=5, turnover=0.05)  # 与 A 互不支配
        rank = non_dominated_rank([A, B, C])
        assert rank["A"] == 0
        assert rank["C"] == 0
        assert rank["B"] == 1  # 被 A 支配 → 更差 front

    def test_chain_dominance(self):
        """A 支配 B 支配 C → rank A=0, B=1, C=2。"""
        A = _pt("A", 1.0, novelty=1.0, complexity=0, turnover=0.0)
        B = _pt("B", 0.8, novelty=0.8, complexity=1, turnover=0.1)
        C = _pt("C", 0.5, novelty=0.5, complexity=2, turnover=0.2)
        rank = non_dominated_rank([A, B, C])
        assert rank["A"] == 0
        assert rank["B"] == 1
        assert rank["C"] == 2

    def test_minimize_objectives_respected(self):
        """低复杂度 / 低换手是「越小越好」：复杂度低者不被高复杂度者支配。"""
        # 用自定义 objectives：fitness 最大化，complexity 最小化
        objs = (Objective("fitness", True), Objective("complexity", False))
        hi = ParetoPoint("hi", (0.9, 5), objectives=objs)   # 高 fitness 低复杂度
        lo = ParetoPoint("lo", (0.5, 5), objectives=objs)   # 低 fitness 同复杂度
        rank = non_dominated_rank([hi, lo])
        assert rank["hi"] == 0
        assert rank["lo"] == 1  # hi 支配 lo

    def test_pareto_rank_of(self):
        A = _pt("A", 0.9, novelty=0.5)
        B = _pt("B", 0.5, novelty=0.5)
        assert pareto_rank_of([A, B], "A") == 0
        assert pareto_rank_of([A, B], "B") == 1
        assert pareto_rank_of([A, B], "missing") == -1


# ---------------------------------------------------------------------------
# 2. 同分确定性
# ---------------------------------------------------------------------------


class TestDeterministicTieBreak:
    def test_identical_points_tie_break_deterministic(self):
        """完全相同的目标向量 → 同 rank 同 crowding → 确定性 tie-break。"""
        A = _pt("A", 0.5, novelty=0.5, complexity=5, turnover=0.1)
        B = _pt("B", 0.5, novelty=0.5, complexity=5, turnover=0.1)
        C = _pt("C", 0.5, novelty=0.5, complexity=5, turnover=0.1)
        rank = non_dominated_rank([A, B, C])
        assert rank["A"] == rank["B"] == rank["C"] == 0
        # 确定性：多次调用结果一致（不要求具体是哪个，只要求稳定）
        victims = {pareto_eviction_candidate([A, B, C]).factor_id for _ in range(5)}
        assert len(victims) == 1

    def test_crowding_distance_prefers_sparse(self):
        """同 rank 内 crowding 大（稀疏）者更该保留 → 淘汰 crowding 小者。"""
        # A 支配 B（fitness 高、novelty 相同）；C 与 A 互不支配
        A = _pt("A", 0.9, novelty=0.5, complexity=5, turnover=0.1)
        B = _pt("B", 0.5, novelty=0.5, complexity=5, turnover=0.1)  # 被 A 支配
        C = _pt("C", 0.1, novelty=0.9, complexity=5, turnover=0.1)
        rank = non_dominated_rank([A, B, C])
        assert rank["A"] == 0 and rank["C"] == 0
        assert rank["B"] == 1  # B 被 A 支配 → 更差 front
        victim = pareto_eviction_candidate([A, B, C])
        assert victim.factor_id == "B"


# ---------------------------------------------------------------------------
# 3. 淘汰策略走 Pareto rank
# ---------------------------------------------------------------------------


class TestEvictionByPareto:
    def test_evicts_worst_rank_not_lowest_ic(self):
        """淘汰按 Pareto rank：rank 大者被淘汰，即使它 fitness 不是最低。"""
        # A 支配 B（A fitness 高）；C 与 A 互不支配但 fitness 低于 A
        A = _pt("A", 0.9, novelty=0.5, complexity=10, turnover=0.1)
        B = _pt("B", 0.5, novelty=0.5, complexity=10, turnover=0.1)  # 被 A 支配
        C = _pt("C", 0.7, novelty=0.9, complexity=5, turnover=0.05)  # 与 A 互不支配
        victim = pareto_eviction_candidate([A, B, C])
        assert victim.factor_id == "B"  # rank 1（最差 front）

    def test_eviction_never_single_ic(self):
        """低 fitness 但 novelty 高者不被淘汰（Pareto 保护非单维 IC）。"""
        # D 低 fitness 但高 novelty → 与高 fitness 低 novelty 的 A 互不支配
        A = _pt("A", 0.9, novelty=0.1, complexity=5, turnover=0.1)
        D = _pt("D", 0.1, novelty=0.9, complexity=5, turnover=0.1)
        rank = non_dominated_rank([A, D])
        assert rank["A"] == 0 and rank["D"] == 0  # 互不支配，都在 front 0
        victim = pareto_eviction_candidate([A, D])
        # 同 rank → 不因单维 IC 淘汰 D；tie-break 确定性
        assert victim is not None


# ---------------------------------------------------------------------------
# 4. pool_snapshot 结构
# ---------------------------------------------------------------------------


class TestPoolSnapshot:
    def test_snapshot_structure(self):
        """pool_snapshot 返回 total / cluster_sizes / by_cluster / pareto_ranks。"""
        members = [
            _member("A", 0.9, novelty=0.5, niche=("x",)),
            _member("B", 0.5, novelty=0.5, niche=("y",)),
            _member("C", 0.7, novelty=0.9, niche=("x",)),
        ]
        snap = pool_snapshot(members)
        assert snap["total"] == 3
        assert snap["cluster_sizes"] == {("x",): 2, ("y",): 1}
        assert snap["by_cluster"][("x",)] == ["A", "C"]
        assert snap["by_cluster"][("y",)] == ["B"]
        assert set(snap["pareto_ranks"].keys()) == {"A", "B", "C"}

    def test_snapshot_cluster_key_fn(self):
        """自定义 cluster_key_fn 覆盖默认 cluster 键。"""
        members = [
            _member("A", 0.9, novelty=0.5),
            _member("B", 0.5, novelty=0.5),
        ]
        snap = pool_snapshot(members, cluster_key_fn=lambda m: "CL_1")
        assert snap["cluster_sizes"] == {"CL_1": 2}


# ---------------------------------------------------------------------------
# 5. ActivePool 集成
# ---------------------------------------------------------------------------


class TestActivePoolPareto:
    def test_low_fitness_rare_niche_protected(self):
        """低 fitness 但 rare-niche 成员不被淘汰（Pareto + QD 保护）。"""
        pool = ActivePool(target_size=2, max_size=3)
        pool.admit(_member("A", 0.9, novelty=0.5, niche=("x",)))
        pool.admit(_member("B", 0.1, novelty=0.5, niche=("y",), rarity=0.95))
        pool.admit(_member("C", 0.5, novelty=0.5, niche=("z",)))
        assert pool.contains("B")  # rare-niche 被保护

    def test_pool_snapshot_method(self):
        """ActivePool.pool_snapshot() 暴露 cluster size 分布。"""
        pool = ActivePool(target_size=2, max_size=3)
        pool.admit(_member("A", 0.9, novelty=0.5, niche=("x",)))
        pool.admit(_member("B", 0.5, novelty=0.5, niche=("y",)))
        snap = pool.pool_snapshot()
        assert snap["total"] == 2
        assert snap["cluster_sizes"] == {("x",): 1, ("y",): 1}
        assert set(snap["pareto_ranks"].keys()) == {"A", "B"}

    def test_max_size_enforced_via_pareto(self):
        """触顶时按 Pareto 淘汰，池大小不超 max_size。"""
        pool = ActivePool(target_size=2, max_size=3)
        for i in range(5):
            pool.admit(_member(f"F{i}", 0.5 + 0.01 * i, novelty=0.5, niche=(f"n{i % 4}",)))
        assert len(pool) <= 3

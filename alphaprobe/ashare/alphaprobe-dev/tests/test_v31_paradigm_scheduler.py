"""plan Task 14：AlphaBench-style macro search paradigm scheduler（CoE/ToT/EA/…）。

覆盖 plan Task 14 Tests + 追加验收（Non-negotiable #27 / #13 / #30）：
1. fertile 低饱和 lineage 偏向 COE（Chain of Experience 利用）；
2. 停滞但 parent 本身好 → 偏向 TOT（发散分支）；
3. 互补 cluster 存在 → EA（mutation + 互补 crossover）eligible；
4. 无第二个 parent 时 EA 绝不可选（不合成假 parent）；
5. SCHEMA paradigm 在 schema 稀有 / gap 大时被选中；
6. 历史 paradigm reward/cost 进选择（bandit 语义：每 paradigm 有 attempts /
   mean_reward；sampled 选择可复现）；
7. ActionScheduler 保持 micro 层不动（范式调度不接管 action 选择）；
8. Paradigm enum = plan 全集 COE/TOT/EA/SCHEMA/TRAJECTORY_REPAIR/LOGIC_EXPLORE；
9. #30：范式/repair 全部有开关（enabled / repair_enabled / schema_enabled /
   ea_enabled / temperature / mode）；
10. #27 行为区分：同一上下文在是否给第二 parent / 是否给停滞证据时，
    eligible 集与高分范式不同——COE/TOT/EA 对同一上下文给出不同选择。

全部合成数据、零 LLM / 零模型 / 零网络；不 import torch / faiss。
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from alphaprobe.search.paradigms import (
    PARADIGM_ORDER,
    PARADIGM_REQUIREMENTS,
    Paradigm,
    ParadigmBanditStats,
    ParadigmContext,
    ParadigmContextRequirements,
    ParadigmFamily,
    ParadigmSelectionConfig,
    eligibility_of,
    paradigm_bias,
    softmax_distribution,
)
from alphaprobe.search.paradigm_scheduler import (
    ParadigmDecision,
    ParadigmScheduler,
)
from alphaprobe.search.arms import RefineArm, BranchArm, EvolutionArm
from alphaprobe.search import ActionScheduler


def _ctx(**kw: Any) -> ParadigmContext:
    base: dict[str, Any] = {}
    base.update(kw)
    return ParadigmContext(**base)


# ---------------------------------------------------------------------------
# 0. Paradigm 全集 + 需求声明
# ---------------------------------------------------------------------------


class TestParadigmEnumFullSet:
    def test_enum_has_plan_full_set(self):
        values = {p.value for p in Paradigm}
        assert values == {
            "COE", "TOT", "EA", "SCHEMA",
            "TRAJECTORY_REPAIR", "LOGIC_EXPLORE",
        }

    def test_plan_order_first_three(self):
        assert PARADIGM_ORDER[:3] == (Paradigm.COE, Paradigm.TOT, Paradigm.EA)

    def test_every_paradigm_has_declared_requirements(self):
        for p in Paradigm:
            assert p in PARADIGM_REQUIREMENTS
            assert isinstance(PARADIGM_REQUIREMENTS[p], ParadigmContextRequirements)


# ---------------------------------------------------------------------------
# 1. fertile 低饱和 lineage → COE
# ---------------------------------------------------------------------------


class TestFertileLowSaturationFavorsCOE:
    def test_fertile_lineage_selects_coe(self):
        """低停滞 + 正趋势 + parent 好 → COE eligible 且被选。"""
        ctx = _ctx(
            parent_fitness=0.7,
            lineage_stagnation=0.05,
            parent_fitness_trend=0.02,
            cluster_saturation=0.1,
        )
        sch = ParadigmScheduler()
        d = sch.select(ctx)
        assert Paradigm.COE in d.eligible
        # 高停滞范式（TOT）在 fertile 低停滞下不可选
        assert Paradigm.TOT not in d.eligible
        # COE 分数领先（low stagnation → 高分）
        assert d.scores[Paradigm.COE.value] > 0.5
        assert d.paradigm is Paradigm.COE

    def test_high_stagnation_makes_coe_ineligible(self):
        """停滞 ≥ 0.3 → COE 不再 eligible（利用 fertile 链的前提不成立）。"""
        ctx = _ctx(lineage_stagnation=0.8, parent_fitness=0.8)
        assert eligibility_of(Paradigm.COE, ctx) is False
        assert Paradigm.COE not in ParadigmScheduler().select(ctx).eligible

    def test_bias_monotonic_in_fertility(self):
        """COE 强度分随停滞升高而下降（低停滞 = 更 fertile）。"""
        low = paradigm_bias(Paradigm.COE, _ctx(lineage_stagnation=0.0))
        high = paradigm_bias(Paradigm.COE, _ctx(lineage_stagnation=0.29))
        assert low > high


# ---------------------------------------------------------------------------
# 2. 停滞但 parent 本身好 → TOT
# ---------------------------------------------------------------------------


class TestStagnantGoodParentFavorsTOT:
    def test_stagnant_selects_tot(self):
        ctx = _ctx(
            parent_fitness=0.9,
            lineage_stagnation=0.8,
            cluster_saturation=0.9,
        )
        sch = ParadigmScheduler()
        d = sch.select(ctx)
        assert Paradigm.TOT in d.eligible
        assert Paradigm.COE not in d.eligible  # 停滞 → 不再利用
        # TOT 分数领先（停滞高 + parent 好）
        assert d.scores[Paradigm.TOT.value] > d.scores.get(Paradigm.COE.value, 0.0)
        assert d.paradigm is Paradigm.TOT

    def test_fertile_lineage_never_selects_tot(self):
        """低停滞（fertile）→ TOT 不可选（不打断好链去发散）。"""
        ctx = _ctx(lineage_stagnation=0.0, parent_fitness_trend=0.03)
        assert eligibility_of(Paradigm.TOT, ctx) is False
        assert Paradigm.TOT not in ParadigmScheduler().select(ctx).eligible


# ---------------------------------------------------------------------------
# 3. 互补 cluster → EA eligible（行为可区分）
# ---------------------------------------------------------------------------


class TestComplementaryClusterEA:
    def test_complementary_parent_makes_ea_eligible(self):
        ctx = _ctx(
            parent_fitness=0.8,
            lineage_stagnation=0.5,
            has_complementary_parent=True,
            n_second_parents=2,
            cluster_saturation=0.85,
            meta={"second_parent_fitness": 0.8},
        )
        sch = ParadigmScheduler()
        d = sch.select(ctx)
        assert Paradigm.EA in d.eligible

    def test_ea_never_without_second_parent(self):
        """无第二个 parent → EA 绝不可选（不合成假 parent，#27）。"""
        ctx = _ctx(
            parent_fitness=0.9,
            lineage_stagnation=0.9,
            cluster_saturation=0.9,
        )
        assert eligibility_of(Paradigm.EA, ctx) is False
        d = ParadigmScheduler().select(ctx)
        assert Paradigm.EA not in d.eligible
        # 即使 has_complementary_parent=True 但 n_second_parents=0（矛盾证据）
        # → EA 仍不可选（真值以显式 parent 数为准）
        assert eligibility_of(
            Paradigm.EA,
            _ctx(has_complementary_parent=True, n_second_parents=0),
        ) is False

    def test_ea_requires_second_parent_requirement_flag(self):
        assert PARADIGM_REQUIREMENTS[Paradigm.EA].needs_second_parent is True

    def test_behavior_distinct_coe_tot_ea_same_base(self):
        """#27：同一基准上下文，是否给停滞 / 是否给第二 parent → 三个不同选择。

        - 低停滞 fertile → COE（EA/TOT 都不可选）；
        - 停滞无第二 parent → TOT（EA 不可选）；
        - 停滞 + 互补第二 parent → EA/TOT 竞争且 EA 常胜（高互补强度）。
        """
        fertile = _ctx(lineage_stagnation=0.05, parent_fitness_trend=0.02)
        stagnant_no2 = _ctx(
            parent_fitness=0.8, lineage_stagnation=0.8, cluster_saturation=0.8
        )
        sch = ParadigmScheduler()
        d_fertile = sch.select(fertile)
        assert d_fertile.paradigm is Paradigm.COE
        d_stag = sch.select(stagnant_no2)
        assert d_stag.paradigm is Paradigm.TOT
        assert Paradigm.EA not in d_stag.eligible
        # EA 上下文：eligible 里含 EA；argmax 下 EA 分数必须高于 TOT
        # （互补 cluster 饱和 0.99 + 第二 parent 好）
        ctx_ea = _ctx(
            parent_fitness=0.9,
            lineage_stagnation=0.9,
            has_complementary_parent=True,
            n_second_parents=2,
            cluster_saturation=0.99,
            schema_gap=0.0,
            schema_rarity=0.0,
            logic_gap=0.0,
            meta={"second_parent_fitness": 0.95},
        )
        sch_am = ParadigmScheduler(config=ParadigmSelectionConfig(mode="argmax"))
        d_ea = sch_am.select(ctx_ea)
        assert Paradigm.EA in d_ea.eligible
        assert d_ea.scores[Paradigm.EA.value] > d_ea.scores[Paradigm.TOT.value]
        assert d_ea.paradigm is Paradigm.EA


# ---------------------------------------------------------------------------
# 4. 无第二个 parent 时 EA 不可选（专项）
# ---------------------------------------------------------------------------


class TestNoSyntheticParent:
    def test_select_never_returns_ea_without_second_parent(self):
        for _ in range(100):
            ctx = _ctx(
                parent_fitness=0.6 + random.random() * 0.4,
                lineage_stagnation=random.random(),
                has_complementary_parent=False,
                n_second_parents=0,
            )
            d = ParadigmScheduler().select(ctx)
            assert d.paradigm is not Paradigm.EA

    def test_ea_disabled_by_default_on_empty_context(self):
        d = ParadigmScheduler().select(ParadigmContext())
        assert Paradigm.EA not in d.eligible


# ---------------------------------------------------------------------------
# 5. SCHEMA paradigm 在 schema 稀有 / gap 大时被选中
# ---------------------------------------------------------------------------


class TestSchemaParadigm:
    def test_large_schema_gap_and_rarity_selects_schema(self):
        ctx = _ctx(
            schema_gap=0.9,
            schema_rarity=0.85,
            schema_n_impl=1,
            parent_fitness=0.5,
        )
        sch = ParadigmScheduler(config=ParadigmSelectionConfig(mode="argmax"))
        d = sch.select(ctx)
        assert Paradigm.SCHEMA in d.eligible
        assert d.paradigm is Paradigm.SCHEMA

    def test_small_schema_gap_not_eligible(self):
        ctx = _ctx(schema_gap=0.0, schema_rarity=0.0)
        assert eligibility_of(Paradigm.SCHEMA, ctx) is False
        assert Paradigm.SCHEMA not in ParadigmScheduler().select(ctx).eligible

    def test_schema_requires_declared_gap_threshold(self):
        req = PARADIGM_REQUIREMENTS[Paradigm.SCHEMA]
        assert req.min_schema_gap > 0.0 and req.min_schema_rarity > 0.0


# ---------------------------------------------------------------------------
# 6. 历史 paradigm reward/cost 进选择（bandit 语义）
# ---------------------------------------------------------------------------


class TestHistoricalRewardBandit:
    def test_mean_reward_neutral_without_attempts(self):
        st = ParadigmBanditStats()
        assert st.attempts == 0
        assert st.mean_reward == pytest.approx(0.5)  # #13：没试过 → 中性
        assert st.mean_cost == pytest.approx(0.0)

    def test_mean_reward_from_attempts(self):
        st = ParadigmBanditStats(attempts=4, reward_sum=3.2, cost_sum=1.0)
        assert st.mean_reward == pytest.approx(0.8)
        assert st.mean_cost == pytest.approx(0.25)

    def test_high_history_lifts_paradigm_bias(self):
        ctx = _ctx(schema_gap=0.9, schema_rarity=0.85)  # SCHEMA 上下文
        stats_good = ParadigmBanditStats(attempts=20, reward_sum=19.0)
        stats_bad = ParadigmBanditStats(attempts=20, reward_sum=2.0)
        good = paradigm_bias(Paradigm.SCHEMA, ctx, stats_good)
        bad = paradigm_bias(Paradigm.SCHEMA, ctx, stats_bad)
        assert good > bad

    def test_high_cost_penalizes_bias(self):
        ctx = _ctx(schema_gap=0.9, schema_rarity=0.85)
        no_cost = paradigm_bias(
            Paradigm.SCHEMA, ctx, ParadigmBanditStats(attempts=20, reward_sum=19.0, cost_sum=0.0)
        )
        high_cost = paradigm_bias(
            Paradigm.SCHEMA, ctx, ParadigmBanditStats(attempts=20, reward_sum=19.0, cost_sum=10.0)
        )
        assert high_cost < no_cost

    def test_injected_history_updates_scheduler_stats(self):
        sch = ParadigmScheduler(
            history={
                Paradigm.COE.value: {"attempts": 10, "reward_sum": 9.0},
                Paradigm.TOT.value: {"attempts": 10, "reward_sum": 1.0},
            }
        )
        st = sch.stats()
        assert st[Paradigm.COE.value]["attempts"] == 10
        assert st[Paradigm.COE.value]["mean_reward"] == pytest.approx(0.9)

    def test_sampled_selection_reproducible(self):
        """同一 seed + 同一上下文 → 同一次选择（可复现）。"""
        ctx = _ctx(
            parent_fitness=0.8,
            lineage_stagnation=0.5,
            has_complementary_parent=True,
            n_second_parents=1,
            cluster_saturation=0.7,
        )
        s1 = ParadigmScheduler(rng=random.Random(42)).select(ctx)
        s2 = ParadigmScheduler(rng=random.Random(42)).select(ctx)
        assert s1.paradigm == s2.paradigm
        assert s1.distribution == s2.distribution

    def test_update_reward_record_decision(self):
        sch = ParadigmScheduler()
        sch.record_decision(Paradigm.COE)
        sch.update_reward(Paradigm.COE, 0.9, cost=0.1)
        assert sch.stats()[Paradigm.COE.value]["attempts"] == 1
        assert sch.stats()[Paradigm.COE.value]["mean_reward"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 7. ActionScheduler 保持 micro 层不动
# ---------------------------------------------------------------------------


class TestActionSchedulerUntouched:
    def test_micro_action_scheduler_unchanged_behavior(self):
        """范式调度与 ActionScheduler 解耦：ActionScheduler 仍是 arm 层。"""
        sch = ActionScheduler()
        # 零尝试先轮询探索（cold-start 规则，与原版一致）
        fam = sch.select(["REFINE", "CROSSOVER", "SCHEMA_EXPLORE"])
        assert fam == "REFINE"
        # paradigm_scheduler 不消费 arm 层
        from alphaprobe.search.paradigm_scheduler import ParadigmScheduler
        ps = ParadigmScheduler()
        d = ps.select(ParadigmContext())
        assert d.paradigm is not None

    def test_arms_remain_micro_generators(self):
        """COE/ToT/EA 的微层实现仍是 arms 的 REFINE/STATE_CONDITION/CROSSOVER。"""
        parent = {"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}
        refine = RefineArm().generate([parent])
        branch = BranchArm().generate([parent])
        evolve = EvolutionArm().generate([parent, {"formula": "if_else(gt(ts_mean(close,20), ts_mean(close,60)), 1.0, 0.0)", "factor_id": "p2"}])
        assert refine[0].action_type == "REFINE"
        assert branch[0].action_type == "STATE_CONDITION"
        assert any(c.action_type == "CROSSOVER" for c in evolve)


# ---------------------------------------------------------------------------
# 8. #30 ablation 开关
# ---------------------------------------------------------------------------


class TestAblationSwitches:
    def test_repair_switch_removes_repair(self):
        ctx = _ctx(
            trajectory_repair_requested=True, trajectory_stagnation=0.9
        )
        on = ParadigmScheduler().select(ctx)
        assert Paradigm.TRAJECTORY_REPAIR in on.eligible
        off = ParadigmScheduler(
            config=ParadigmSelectionConfig(repair_enabled=False)
        ).select(ctx)
        assert Paradigm.TRAJECTORY_REPAIR not in off.eligible

    def test_schema_switch_removes_schema(self):
        ctx = _ctx(schema_gap=0.9, schema_rarity=0.9)
        off = ParadigmScheduler(
            config=ParadigmSelectionConfig(schema_enabled=False)
        ).select(ctx)
        assert Paradigm.SCHEMA not in off.eligible

    def test_ea_switch_removes_ea(self):
        ctx = _ctx(
            has_complementary_parent=True,
            n_second_parents=1,
            cluster_saturation=0.9,
        )
        off = ParadigmScheduler(
            config=ParadigmSelectionConfig(ea_enabled=False)
        ).select(ctx)
        assert Paradigm.EA not in off.eligible

    def test_global_disabled_returns_coe_decision(self):
        off = ParadigmScheduler(config=ParadigmSelectionConfig(enabled=False))
        d = off.select(ParadigmContext())
        assert d.paradigm is Paradigm.COE
        assert d.distribution == {Paradigm.COE.value: 1.0}

    def test_argmax_deterministic(self):
        ctx = _ctx(schema_gap=0.9, schema_rarity=0.9)
        s = ParadigmScheduler(config=ParadigmSelectionConfig(mode="argmax"))
        assert s.select(ctx).paradigm is Paradigm.SCHEMA
        assert s.select(ctx).paradigm is Paradigm.SCHEMA


# ---------------------------------------------------------------------------
# 9. 分布/诊断工具
# ---------------------------------------------------------------------------


class TestDistributionHelpers:
    def test_softmax_sum_one(self):
        dist = softmax_distribution({"COE": 0.9, "TOT": 0.5}, eligible=["COE", "TOT"])
        assert sum(dist.values()) == pytest.approx(1.0)
        assert set(dist) == {Paradigm.COE, Paradigm.TOT}

    def test_softmax_argmax_on_temperature_zero(self):
        dist = softmax_distribution(
            {"COE": 0.2, "TOT": 0.8}, temperature=0, eligible=["COE", "TOT"]
        )
        assert dist[Paradigm.TOT] == 1.0
        assert dist[Paradigm.COE] == 0.0

    def test_empty_distribution(self):
        assert softmax_distribution({}) == {}

    def test_decision_to_dict_roundtrip(self):
        d = ParadigmDecision(
            paradigm=Paradigm.COE,
            eligible=(Paradigm.COE,),
            scores={"COE": 0.8},
            distribution={"COE": 1.0},
        )
        dd = d.to_dict()
        assert dd["paradigm"] == "COE"
        assert dd["eligible"] == ["COE"]

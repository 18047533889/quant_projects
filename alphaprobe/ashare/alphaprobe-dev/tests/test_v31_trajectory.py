"""plan Task 15：QuantaAlpha-style Search Trajectory Credit Assignment。

覆盖 plan Task 15 Tests + Non-negotiable #28 / #30 / #13：
1. F1--a1-->F2--a2-->F3--a3-->F4 前两步改进第三步崩塌 → repair target = 第 3 条
   边（critic 能隔离坏边，不丢整个成功前缀，#28）；
2. 重复失败 suffix 提高局部 action 饱和度，但不判 parent 全局死亡
   （action 级与 parent 级分离）；
3. 两个兼容 trajectory fragment 可组合（action/schema 层）；schema/domain
   不兼容则拒绝组合；
4. repair 只在停滞或显式 critic 条件后触发（不是每个 candidate）；
5. per-step ΔFitness/ΔPool/novelty/complexity/cost/failure_reason 持久化
   （to_dict/from_dict roundtrip）；
6. trajectory crossover 的最终表达式仍由 FE AST 编译——本模块只记录组合
   plan（parent ids / fragments / mode），plan 契约断言「不产公式、编译留给
   生成链」。

全部合成数据、零 LLM / 零模型 / 零网络；不 import torch / faiss。
"""

from __future__ import annotations

import pytest

from alphaprobe.search.trajectory import (
    CriticConfig,
    CriticVerdict,
    SearchTrajectory,
    TrajectoryCritic,
    TrajectoryFragment,
    TrajectoryStep,
    compose_fragments,
    repair_target_of,
)


def _step(
    factor_id: str = "",
    *,
    parent_id: str = "",
    action_type: str = "REFINE",
    df: float | None = None,
    dp: float | None = None,
    nov: float | None = None,
    cx: int = 0,
    cost: float = 0.0,
    failure: str = "",
) -> TrajectoryStep:
    return TrajectoryStep(
        factor_id=factor_id,
        parent_id=parent_id,
        action_type=action_type,
        delta_fitness=df,
        delta_pool=dp,
        novelty=nov,
        complexity=cx,
        cost=cost,
        failure_reason=failure,
    )


# ---------------------------------------------------------------------------
# 1. 前两步改进、第三步崩塌 → repair target = 第三条边
# ---------------------------------------------------------------------------


class TestCriticIsolatesBadEdge:
    def _good_good_collapse(self) -> SearchTrajectory:
        t = SearchTrajectory(lineage_key="L1")
        t.append(
            _step("F2", parent_id="F1", action_type="REFINE", df=0.02, dp=0.01, nov=0.05)
        )
        t.append(
            _step("F3", parent_id="F2", action_type="REFINE", df=0.03, dp=0.01, nov=0.02)
        )
        t.append(_step("F4", parent_id="F3", action_type="CROSSOVER", df=-0.05))
        return t

    def test_bad_edge_is_third(self):
        t = self._good_good_collapse()
        v = repair_target_of(t, stagnation=0.6)
        assert v.bad_edge_index == 2  # 第三条边（0-based）
        assert 2 in v.low_value_indexes
        # 前两条好边不在低价值集（成功前缀保留，#28）
        assert 0 not in v.low_value_indexes
        assert 1 not in v.low_value_indexes
        assert v.suggests_repair is True

    def test_prefix_not_discarded(self):
        """修复目标是坏边而不是丢弃整个前缀：低价值集只有坏边。"""
        t = self._good_good_collapse()
        v = repair_target_of(t, stagnation=0.6)
        assert v.low_value_indexes == (2,)

    def test_reason_codes(self):
        t = self._good_good_collapse()
        v = repair_target_of(t, stagnation=0.6)
        assert "fitness_collapse" in v.reasons
        assert v.score > 0.5


# ---------------------------------------------------------------------------
# 2. 重复失败 suffix → 局部 action 饱和度（不判 parent 死亡）
# ---------------------------------------------------------------------------


class TestLocalActionSaturationVsParentDeath:
    def test_repeated_failure_raises_local_action_saturation(self):
        t = SearchTrajectory(lineage_key="L2")
        for _ in range(3):
            t.append(
                _step("", parent_id="P", action_type="CROSSOVER", failure="FE_INVALID")
            )
        assert t.local_action_saturation["CROSSOVER"] == 3

    def test_saturation_does_not_kill_parent(self):
        """失败只抬升 action 局部饱和度；轨迹本身无 parent 死亡判定。"""
        t = SearchTrajectory(lineage_key="L2")
        for _ in range(5):
            t.append(_step("", parent_id="P", action_type="CROSSOVER", failure="FE_INVALID"))
        # action 级饱和度 5（局部），但 lineage 上不存在「parent 已死」标志
        assert t.local_action_saturation["CROSSOVER"] == 5
        assert t.repaired_at == -1  # 没有外部判 dead/repair 的副作用

    def test_gain_resets_action_saturation(self):
        """同 action 成功后 → 局部饱和度清零（action 级与 parent 级分离）。"""
        t = SearchTrajectory(lineage_key="L3")
        t.append(_step("F2", parent_id="F1", action_type="REFINE", failure="x"))
        t.append(_step("F3", parent_id="F2", action_type="REFINE", failure="x"))
        assert t.local_action_saturation["REFINE"] == 2
        t.append(
            _step("F4", parent_id="F3", action_type="REFINE", df=0.02, nov=0.1)
        )
        assert t.local_action_saturation["REFINE"] == 0

    def test_no_evidence_step_neutral(self):
        """无证据 step 不动饱和度（#13：没测过不算失败）。"""
        t = SearchTrajectory(lineage_key="L4")
        t.append(_step("F2", parent_id="F1", action_type="REFINE"))
        assert t.local_action_saturation.get("REFINE", 0) == 0


# ---------------------------------------------------------------------------
# 3. 兼容 fragment 可组合；不兼容拒绝
# ---------------------------------------------------------------------------


class TestComposeFragments:
    def _frag(self, fid: str, *, schema: str, domain: tuple[str, ...]) -> TrajectoryFragment:
        return TrajectoryFragment(
            fragment_id=fid,
            steps=(
                _step(f"{fid}2", parent_id=f"{fid}1", action_type="REFINE",
                      df=0.02, nov=0.05),
            ),
            anchor_factor_id=f"{fid}2",
            schema_id=schema,
            domain_tags=domain,
        )

    def test_compatible_fragments_compose_prefix_concat(self):
        a = self._frag("A", schema="SCH_PV", domain=("PRICE", "VOLUME"))
        b = self._frag("B", schema="SCH_PV", domain=("PRICE",))
        plan = compose_fragments(a, b)
        assert plan is not None
        assert plan.mode == "prefix_concat"
        # parents = 两个 fragment 的锚点因子（DAG multi-parent 契约）
        assert set(plan.parent_factor_ids) == {"A2", "B2"}
        assert plan.prefix_fragment is a and plan.suffix_fragment is b

    def test_compatible_fragments_compose_schema_cross(self):
        a = self._frag("A", schema="SCH_PV", domain=("PRICE", "VOLUME"))
        b = self._frag("B", schema="SCH_PV", domain=("PRICE",))
        plan = compose_fragments(a, b, mode="schema_cross")
        assert plan is not None
        assert plan.mode == "schema_cross"

    def test_domain_incompatible_refused(self):
        a = self._frag("A", schema="SCH_PV", domain=("PRICE",))
        c = self._frag("C", schema="SCH_FUND", domain=("FUNDAMENTAL",))
        assert compose_fragments(a, c) is None  # domain 不兼容 → 拒绝
        assert compose_fragments(c, a) is None
        assert compose_fragments(a, c, mode="schema_cross") is None

    def test_schema_incompatible_refused_without_override(self):
        a = self._frag("A", schema="SCH_PV", domain=("PRICE",))
        b = self._frag("B", schema="SCH_VOL", domain=("PRICE", "VOLUME"))
        assert compose_fragments(a, b) is None  # 不同 schema 未放行 → 拒绝
        # 显式 compatible_schemas=True（调用方已做语义对齐）→ 允许
        assert compose_fragments(a, b, compatible_schemas=True) is not None

    def test_empty_fragment_refused(self):
        a = self._frag("A", schema="SCH_PV", domain=("PRICE",))
        empty = TrajectoryFragment(fragment_id="E", schema_id="SCH_PV", domain_tags=("PRICE",))
        assert compose_fragments(a, empty) is None
        assert compose_fragments(empty, a) is None

    def test_unknown_mode_fails_closed(self):
        a = self._frag("A", schema="SCH_PV", domain=("PRICE",))
        b = self._frag("B", schema="SCH_PV", domain=("PRICE",))
        assert compose_fragments(a, b, mode="bogus") is None

    def test_plan_contract_no_formula(self):
        """组合 plan 不含最终公式（最终表达式由 FE AST 编译）。"""
        a = self._frag("A", schema="SCH_PV", domain=("PRICE",))
        b = self._frag("B", schema="SCH_PV", domain=("PRICE",))
        plan = compose_fragments(a, b)
        assert plan is not None
        d = plan.to_dict()
        # plan 只记录 fragments / parents / mode —— 无拼接后的 formula 字符串
        assert "formula" not in d
        assert set(d) == {"prefix_fragment", "suffix_fragment",
                          "parent_factor_ids", "mode", "schema_tags"}


# ---------------------------------------------------------------------------
# 4. repair 只在停滞或显式 critic 条件后触发
# ---------------------------------------------------------------------------


class TestRepairTriggerGating:
    def _traj_with_bad_edge(self) -> SearchTrajectory:
        t = SearchTrajectory(lineage_key="L5")
        t.append(_step("F2", parent_id="F1", action_type="REFINE", df=0.02))
        t.append(_step("F3", parent_id="F2", action_type="REFINE", df=0.02))
        t.append(_step("F4", parent_id="F3", action_type="REFINE", df=-0.05))
        return t

    def test_no_stagnation_no_explicit_condition_no_repair(self):
        """坏边存在但无停滞证据 + 无 critic 条件 → 不触发 repair。"""
        v = repair_target_of(self._traj_with_bad_edge())
        assert v.has_bad_edge  # 坏边仍被标出（credit 判定独立）
        assert v.suggests_repair is False  # 但 repair 不触发

    def test_stagnation_above_threshold_triggers_repair(self):
        v = repair_target_of(self._traj_with_bad_edge(), stagnation=0.5)
        assert v.suggests_repair is True

    def test_explicit_critic_condition_triggers_repair(self):
        v = repair_target_of(
            self._traj_with_bad_edge(), critic_condition=True
        )
        assert v.suggests_repair is True

    def test_stagnation_below_threshold_no_repair(self):
        v = repair_target_of(self._traj_with_bad_edge(), stagnation=0.05)
        assert v.suggests_repair is False

    def test_no_bad_edge_never_triggers_repair(self):
        t = SearchTrajectory(lineage_key="L6")
        t.append(_step("F2", parent_id="F1", action_type="REFINE", df=0.02))
        v = repair_target_of(t, stagnation=0.9)
        assert v.has_bad_edge is False
        assert v.suggests_repair is False

    def test_critic_condition_can_be_toggled_in_config(self):
        """#30：critic 阈值/开关在 config 里可调。"""
        cfg = CriticConfig(min_stagnation_for_repair=0.8)
        critic = TrajectoryCritic(cfg)
        v = critic.judge(self._traj_with_bad_edge(), stagnation=0.6)
        assert v.has_bad_edge
        assert v.suggests_repair is False  # 0.6 < 0.8 → 不触发


# ---------------------------------------------------------------------------
# 5. per-step 字段持久化（ΔFitness/ΔPool/novelty/complexity/cost/failure）
# ---------------------------------------------------------------------------


class TestStepPersistence:
    def test_step_dict_roundtrip(self):
        s = _step(
            "F2", parent_id="F1", action_type="CROSSOVER",
            df=0.015, dp=-0.001, nov=0.2, cx=3, cost=0.42,
            failure="",
        )
        d = s.to_dict()
        assert d["delta_fitness"] == 0.015
        assert d["delta_pool"] == -0.001
        assert d["novelty"] == 0.2
        assert d["complexity"] == 3
        assert d["cost"] == 0.42
        assert d["failure_reason"] == ""
        assert d["action_type"] == "CROSSOVER"
        s2 = TrajectoryStep.from_dict(d)
        assert s2 == s

    def test_step_dict_failure_reason(self):
        s = _step("", parent_id="P", action_type="REFINE", failure="FE_INVALID")
        s2 = TrajectoryStep.from_dict(s.to_dict())
        assert s2.failed is True
        assert s2.failure_reason == "FE_INVALID"

    def test_trajectory_dict_roundtrip(self):
        t = SearchTrajectory(lineage_key="LK")
        t.append(_step("F2", parent_id="F1", action_type="REFINE", df=0.02, nov=0.05))
        t.append(_step("F3", parent_id="F2", action_type="REFINE", failure="x"))
        d = t.to_dict()
        assert d["lineage_key"] == "LK"
        assert len(d["steps"]) == 2
        assert d["local_action_saturation"]["REFINE"] == 1
        t2 = SearchTrajectory.from_dict(d)
        assert t2.lineage_key == "LK"
        assert t2.local_action_saturation == {"REFINE": 1}
        assert t2.steps[0].delta_fitness == 0.02

    def test_has_gain_and_evidence(self):
        good = _step("F2", parent_id="F1", df=0.01)
        assert good.has_gain is True
        neutral = _step("F2", parent_id="F1")
        assert neutral.has_gain is False
        assert neutral.has_evidence is False  # 无证据（#13）
        neg = _step("F2", parent_id="F1", df=-0.02)
        assert neg.has_gain is False
        assert neg.has_evidence is True


# ---------------------------------------------------------------------------
# 6. critic 窗口 / 可配置
# ---------------------------------------------------------------------------


class TestCriticConfig:
    def test_window_only_recent_steps(self):
        """窗口外的坏边不算：早期失败被窗口截断后不判坏边。"""
        t = SearchTrajectory(lineage_key="L7")
        # 前 3 步全失败（会被 window=3 截断在窗外）
        for i in range(3):
            t.append(_step("", parent_id="P", action_type="REFINE", failure="f"))
        # 最近 1 步成功
        t.append(_step("F4", parent_id="P3", action_type="REFINE", df=0.02, nov=0.1))
        cfg = CriticConfig(window=3)
        v = TrajectoryCritic(cfg).judge(t, stagnation=0.9)
        # 窗口 = 最近 3 步（含成功那步）：失败比例 2/3 < fail_threshold? 不触发
        assert v.has_bad_edge is False or v.bad_edge_index >= len(t) - cfg.window

    def test_high_failed_ratio_detected(self):
        t = SearchTrajectory(lineage_key="L8")
        for _ in range(5):
            t.append(_step("", parent_id="P", action_type="REFINE", failure="f"))
        cfg = CriticConfig(window=4, fail_threshold=0.5)
        v = TrajectoryCritic(cfg).judge(t, stagnation=0.5)
        assert v.has_bad_edge is True
        assert "failed_step" in v.reasons or "failure_streak" in v.reasons

    def test_mark_repaired(self):
        t = SearchTrajectory(lineage_key="L9")
        t.append(_step("F2", parent_id="F1", action_type="REFINE", df=0.02))
        t.mark_repaired(0)
        assert t.repaired_at == 0


# ---------------------------------------------------------------------------
# 7. lineage 接线：LineageTrajectoryTracker（plan Task 15 Modify lineage/）
# ---------------------------------------------------------------------------


class TestLineageTrackerWiring:
    def _tracker(self):
        from alphaprobe.lineage import LineageTrajectoryTracker
        return LineageTrajectoryTracker()

    def test_observe_records_step(self):
        tk = self._tracker()
        tk.observe(
            lineage_key="LK1",
            factor_id="F2", parent_id="F1", action_type="REFINE",
            delta_fitness=0.02, delta_pool=0.01, novelty=0.05,
            cost=0.1,
        )
        t = tk.trajectory_of("LK1")
        assert len(t) == 1
        assert t.steps[0].delta_fitness == 0.02
        assert t.steps[0].cost == 0.1

    def test_repair_requested_gated_on_stagnation(self):
        """无停滞证据时即使有坏边 repair 也不请求（#13 / 不是每 candidate 触发）。"""
        tk = self._tracker()
        for df in (0.02, 0.02, -0.05):
            tk.observe(lineage_key="LK2", action_type="REFINE", delta_fitness=df)
        assert tk.repair_requested("LK2") is False  # 无停滞证据
        assert tk.repair_requested("LK2", stagnation=0.5) is True

    def test_repair_requested_with_critic_condition(self):
        tk = self._tracker()
        for df in (0.02, 0.02, -0.05):
            tk.observe(lineage_key="LK3", action_type="REFINE", delta_fitness=df)
        assert tk.repair_requested("LK3", critic_condition=True) is True

    def test_observe_updates_local_saturation(self):
        tk = self._tracker()
        for _ in range(3):
            tk.observe(
                lineage_key="LK4", action_type="CROSSOVER",
                failure_reason="FE_INVALID",
            )
        t = tk.trajectory_of("LK4")
        assert t.local_action_saturation["CROSSOVER"] == 3

    def test_tracker_does_not_judge_parent_death(self):
        """tracker 只做 action 级饱和度/repair 判定——绝不暴露 parent 死亡。"""
        tk = self._tracker()
        for _ in range(6):
            tk.observe(
                lineage_key="LK5", action_type="REFINE", failure_reason="x"
            )
        t = tk.trajectory_of("LK5")
        assert t.local_action_saturation["REFINE"] == 6
        # 无「parent_dead」属性/接口（action 级与 parent 级分离）
        assert not hasattr(t, "parent_dead")

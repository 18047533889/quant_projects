"""Phase 9（多臂搜索 / Scheduler / LLM 结构化输出）测试。

覆盖任务书验收点：
- GenerationRepairArm 修括号不平衡
- parse_generated：合法 JSON / markdown 包裹 / 坏 JSON fallback 三种路径
- orchestrator stub llm 跑 3 步不抛，scheduler stats 增长
- EvolutionArm crossover 输出含双方成分且非简单 "(A) + (B)" 字符串拼接
- model_route：CROSSOVER→strong、REFINE→cheap
"""

from __future__ import annotations

import json
import warnings

import pytest

from alphaprobe.search import GenerationRepairArm, model_route
from alphaprobe.search.arms import (
    BranchArm,
    EvolutionArm,
    RefineArm,
    SchemaExploreArm,
    infer_formula_roles,
)
from alphaprobe.search.orchestrator import OrchestratorStep, SearchOrchestrator
from alphaprobe.search.structured_llm import (
    GeneratedCandidate,
    candidates_to_json,
    parse_generated,
)


# ---------------------------------------------------------------------------
# §25.5 GenerationRepairArm：修括号不平衡
# ---------------------------------------------------------------------------

class TestGenerationRepair:
    def test_missing_close_paren(self):
        repaired, note = GenerationRepairArm().repair("rank(ts_mean(close, 20)")
        assert repaired == "rank(ts_mean(close, 20))"
        assert note == "syntax"

    def test_missing_open_paren(self):
        repaired, note = GenerationRepairArm().repair("rank(ts_mean(close, 20)))")
        assert repaired == "(rank(ts_mean(close, 20)))"
        assert note == "syntax"

    def test_balanced_untouched(self):
        repaired, _ = GenerationRepairArm().repair("rank(ts_mean(close, 20))")
        assert repaired == "rank(ts_mean(close, 20))"

    def test_unrepairable_returns_none(self):
        repaired, _ = GenerationRepairArm().repair("((rank(close)")
        assert repaired is None


# ---------------------------------------------------------------------------
# §31 parse_generated：合法 JSON / markdown 包裹 / 坏 JSON fallback
# ---------------------------------------------------------------------------

class TestParseGenerated:
    def test_valid_json(self):
        text = candidates_to_json(
            [
                GeneratedCandidate(
                    formula="rank(ts_mean(close, 20))",
                    explanation="x",
                    hypothesis="h",
                    action_type="REFINE",
                    parent_ids=["p1"],
                )
            ]
        )
        cands = parse_generated(text, expected_num=1)
        assert len(cands) == 1
        assert cands[0].formula == "rank(ts_mean(close, 20))"
        assert cands[0].parent_ids == ["p1"]

    def test_markdown_wrapped_json(self):
        text = (
            "Here is the result:\n"
            "```json\n"
            '{"candidates": [{"formula": "rank(ts_std(close, 10))", '
            '"explanation": "md", "hypothesis": "h1", '
            '"action_type": "REFINE", "parent_ids": ["p1"], "schema_tags": {}}]}\n'
            "```\n"
        )
        cands = parse_generated(text, expected_num=1)
        assert len(cands) == 1
        assert cands[0].formula == "rank(ts_std(close, 10))"

    def test_bad_json_regex_fallback(self):
        text = 'prose here {"candidates": [{"formula": "rank(close)", ' \
               '"explanation": "fb", "hypothesis": "h", ' \
               '"action_type": "REFINE", "parent_ids": ["p1"]}]} trailing'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cands = parse_generated(text, expected_num=1)
        assert len(cands) == 1
        assert cands[0].formula == "rank(close)"

    def test_garbage_returns_empty_no_raise(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cands = parse_generated("not json at all [{{{{", expected_num=1)
        assert cands == []

    def test_invalid_items_skipped_not_raised(self):
        text = json.dumps(
            {
                "candidates": [
                    {"formula": "rank(close)", "explanation": "ok"},
                    {"formula": "", "explanation": "empty formula"},
                    "not a dict",
                    {"no_formula_key": 1},
                ]
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cands = parse_generated(text, expected_num=1)
        assert [c.formula for c in cands] == ["rank(close)"]


# ---------------------------------------------------------------------------
# §25.3 EvolutionArm crossover：含双方成分且非简单字符串拼接
# ---------------------------------------------------------------------------

class TestEvolutionCrossover:
    def _parent(self, fid: str, formula: str) -> dict:
        return {"factor_id": fid, "formula": formula, "explanation": fid}

    def test_crossover_contains_both_and_no_plain_concat(self):
        arm = EvolutionArm()
        a = self._parent("a1", "rank(ts_mean(close, 20))")
        b = self._parent("b1", "if_else(gt(volume, 0.5), rank(close), 0.0)")
        cands = arm.generate([a, b], None, llm_fn=None)
        cross = [c for c in cands if c.action_type == "CROSSOVER"]
        assert cross, "crossover 候选缺失"
        c = cross[0]
        # 双方成分都在
        assert "ts_mean(close, 20)" in c.formula or "rank(ts_mean(close, 20))" in c.formula
        assert "volume" in c.formula
        # 不是简单 "(A) + (B)" 字符串拼接
        assert "+" not in c.formula
        assert c.formula.count("(") == c.formula.count(")")
        # 角色感知 hypothesis 说明出处
        assert "A" in c.hypothesis or "B" in c.hypothesis or "role" in c.explanation

    def test_role_heuristics(self):
        assert infer_formula_roles("rank(close)")["quality_filter"] == "rank(close)"
        assert (
            infer_formula_roles("if_else(gt(volume, 0.5), rank(close), 0.0)")[
                "context_condition"
            ]
        )
        assert infer_formula_roles("ts_mean(close, 20)")["signal_core"]

    def test_two_parents_role_aware_if_else(self):
        """两 parents：产出 if_else 形式；条件来自 B（context），核心来自 A（signal_core）。"""
        arm = EvolutionArm()
        a = self._parent("a1", "rank(ts_mean(close, 20))")
        b = self._parent("b1", "if_else(gt(volume, 0.5), rank(close), 0.0)")
        cands = arm.generate([a, b], None, llm_fn=None)
        cross = [c for c in cands if c.action_type == "CROSSOVER"]
        assert cross, "crossover 候选缺失"
        c = cross[0]
        assert c.formula.startswith("if_else(")
        assert c.formula.endswith(", 0.0)")
        assert "gt(volume, 0.5)" in c.formula  # 条件来自 B
        assert "ts_mean(close, 20)" in c.formula  # 核心来自 A
        assert c.formula.count("(") == c.formula.count(")")
        # parent_ids 同时含 A 与 B
        assert "a1" in c.parent_ids and "b1" in c.parent_ids

    def test_single_parent_no_crossover_fallback(self):
        """单 parent：不抛，仅 mutation；无 crossover 也非空。"""
        arm = EvolutionArm()
        a = self._parent("a1", "rank(ts_mean(close, 20))")
        cands = arm.generate([a], None, llm_fn=None)
        assert cands
        assert not [c for c in cands if c.action_type == "CROSSOVER" and "if_else" in c.formula]

    def test_two_parents_both_no_context_falls_back_no_raise(self):
        """parents=2 但都无 context 角色：回落单 parent 路径，不抛。"""
        arm = EvolutionArm()
        a = self._parent("a1", "ts_mean(close, 20)")
        b = self._parent("b1", "rank(close)")
        cands = arm.generate([a, b], None, llm_fn=None)
        assert cands
        # 不产出双 parent if_else 组合（B 无 context 段）
        assert not any(
            c.action_type == "CROSSOVER" and "if_else(" in c.formula
            for c in cands
        )

    def test_multi_parent_all_with_context_each_pair(self):
        """parents=3 全带 context：每个补充 parent 都产出 if_else 组合。"""
        arm = EvolutionArm()
        a = self._parent("a1", "rank(ts_mean(close, 20))")
        b1 = self._parent("b1", "if_else(gt(volume, 0.5), rank(close), 0.0)")
        b2 = self._parent("b2", "if_else(gt(amount, 0.5), rank(close), 0.0)")
        cands = arm.generate([a, b1, b2], None, llm_fn=None)
        crosses = [c for c in cands if c.action_type == "CROSSOVER" and "if_else(" in c.formula]
        assert len(crosses) == 2
        conds = [c.formula.split("if_else(")[1].split(",", 1)[0].strip() for c in crosses]
        assert any(cd.startswith("gt(volume") for cd in conds)
        assert any(cd.startswith("gt(amount") for cd in conds)
    def test_empty_parents_returns_empty(self):
        arm = EvolutionArm()
        assert arm.generate([], None, llm_fn=None) == []


# ---------------------------------------------------------------------------
# §30 model_route：CROSSOVER→strong、REFINE→cheap
# ---------------------------------------------------------------------------

class TestModelRoute:
    def test_strong_actions(self):
        assert model_route("CROSSOVER") == "strong"
        assert model_route("SCHEMA_EXPLORE") == "strong"

    def test_cheap_actions(self):
        assert model_route("REFINE") == "cheap"
        assert model_route("GENERATION_REPAIR") == "cheap"


# ---------------------------------------------------------------------------
# §26-§32 SearchOrchestrator：stub llm 跑 3 步不抛，scheduler stats 增长
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def _stub_llm(self, system_prompt: str, user_prompt: str, model_class: str) -> str:
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

    def _parent(self) -> dict:
        return {
            "factor_id": "p1",
            "formula": "rank(ts_mean(close, 20))",
            "explanation": "parent",
        }

    def test_three_steps_no_throw_stats_grow(self):
        from alphaprobe.search import ACTION_FAMILIES

        orch = SearchOrchestrator(scheduler=None)
        seen_types = set()
        for _ in range(3):
            result = orch.step_with_llm(self._parent(), self._stub_llm)
            assert isinstance(result, OrchestratorStep)
            seen_types.add(result.action.action_type.value)
        assert seen_types <= set(ACTION_FAMILIES)
        total_attempts = sum(
            st.attempts for st in orch.scheduler.stats.values()
        )
        assert total_attempts == 3

    def test_candidates_parsed_from_stub(self):
        orch = SearchOrchestrator()
        result = orch.step_with_llm(self._parent(), self._stub_llm)
        assert result.candidates
        assert result.candidates[0].formula == "rank(ts_std(close, 20))"

    def test_step_records_attempt(self):
        orch = SearchOrchestrator()
        result = orch.step_with_llm(self._parent(), self._stub_llm)
        assert result.attempts
        att = result.attempts[0]
        assert att.action.action_id
        assert att.latency_ms >= 0
        assert att.llm_cost == 0.0

    def test_generation_repair_path(self):
        orch = SearchOrchestrator()
        # 强制 scheduler 先选中 GENERATION_REPAIR：直接喂一个零尝试的 scheduler
        from alphaprobe.search import ACTION_FAMILIES, ArmStats

        # 手动构造：让 GENERATION_REPAIR 是唯一零尝试臂
        for fam in ACTION_FAMILIES:
            if fam != "GENERATION_REPAIR":
                orch.scheduler.stats[fam] = ArmStats(attempts=1, reward_sum=0.0, reward_sq_sum=0.0)
        parent = self._parent()
        parent["formula"] = "rank(ts_mean(close, 20)"
        result = orch.step(parent, self._stub_llm)
        assert result.action.action_type.value == "GENERATION_REPAIR"
        assert all(c.formula.count("(") == c.formula.count(")") for c in result.candidates)

    def test_memory_packet_in_prompt(self):
        from alphaprobe.memory import MemoryPacket

        class _Mem:
            def build_memory_packet(self, *, parent_node):
                return MemoryPacket(
                    parent=parent_node,
                    structural_neighbors=[{"formula": "rank(close)"}],
                    unexplored_actions=["CROSSOVER"],
                )

        orch = SearchOrchestrator(memory=_Mem())
        sys_p, usr_p = orch.build_prompt([self._parent()], "REFINE", 3)
        assert "MemoryPacket" in usr_p
        assert "rank(close)" in usr_p

"""V2-H 结构化 generation 测试（任务书 §25 / §27 / §37）。

覆盖：
① action JSON → AST 构造正确性（zscore/rank/加权和）
② 非白名单 op 拒绝并记录
③ FE validate 失败的 action 不进候选池
④ stub llm_fn 确定性（同 seed 同输出）
⑤ lineage 记录完整
⑥ 旧文本输出兼容路径（identity action fallback）
⑦ multi-parent wiring 启用后 parents 来自 select_parents（monkeypatch）

合成数据；不跑真实 LLM / 模型训练。
"""

from __future__ import annotations

import json

import pytest

from alphaprobe.generation.structured import (
    DEFAULT_ACTION_OP_WHITELIST,
    StructuredGenerator,
    build_formula,
    fe_operator_surface,
    make_structured_stub_llm_fn,
    normalize_llm_output,
    validate_action,
)


def _fe_importable() -> bool:
    try:
        import alphaprobe.fe_bridge.paths as _p

        _p.ensure_factor_engine_importable()
        import factor_engine.api  # noqa: F401

        return True
    except Exception:
        return False


FE_AVAILABLE = _fe_importable()


def _parents() -> list[dict]:
    return [
        {"factor_id": "p1", "formula": "rank(close)"},
        {"factor_id": "p2", "formula": "ts_mean(close, 20)"},
    ]


# ---------------------------------------------------------------------------
# ① action JSON → AST 构造正确性
# ---------------------------------------------------------------------------


class TestBuildFormula:
    def test_transform_zscore(self):
        f, reason = build_formula(
            {"action": "transform", "op": "zscore", "target": "parent_0"}, _parents()
        )
        assert f == "zscore(rank(close))"
        assert reason == ""

    def test_transform_rank(self):
        f, _ = build_formula(
            {"action": "transform", "op": "rank", "target": "parent_1"}, _parents()
        )
        assert f == "rank(ts_mean(close, 20))"

    def test_transform_window_op(self):
        f, _ = build_formula(
            {
                "action": "transform",
                "op": "ts_mean",
                "target": "parent_1",
                "params": {"window": 20},
            },
            _parents(),
        )
        assert f == "ts_mean(ts_mean(close, 20), 20)"

    def test_combine_weighted_sum(self):
        f, _ = build_formula(
            {
                "action": "combine",
                "op": "add",
                "parents": ["p1", "p2"],
                "weights": [0.5, 0.5],
            },
            _parents(),
        )
        assert f == "add(multiply(0.5, rank(close)), multiply(0.5, ts_mean(close, 20)))"

    def test_combine_equal_weight(self):
        f, _ = build_formula(
            {"action": "combine", "op": "add", "parents": ["p1", "p2"]}, _parents()
        )
        assert f == "add(rank(close), ts_mean(close, 20))"

    def test_combine_requires_two_parents(self):
        f, reason = build_formula(
            {"action": "combine", "op": "add", "parents": ["p1"]}, _parents()
        )
        assert f is None
        assert ">=2 parents" in reason

    def test_identity_action(self):
        f, _ = build_formula({"action": "identity"}, _parents())
        assert f == "rank(close)"


# ---------------------------------------------------------------------------
# ② 非白名单 op 拒绝并记录
# ---------------------------------------------------------------------------


class TestNonWhitelistOp:
    def test_unknown_op_rejected(self):
        f, reason = build_formula(
            {"action": "transform", "op": "foo_bar", "target": "parent_0"}, _parents()
        )
        assert f is None
        assert "not in whitelist" in reason

    def test_validate_action_rejects_unknown_op(self):
        v = validate_action({"action": "transform", "op": "foo_bar", "target": "parent_0"})
        assert not v.ok
        assert "not in whitelist" in v.reason

    def test_whitelist_ops_are_in_fe_surface(self):
        """白名单 op 必须都在 FE 表面（禁止发明 FE 没有的算子语义）。"""
        if not FE_AVAILABLE:
            pytest.skip("FE not importable")
        surface = fe_operator_surface()
        for op in DEFAULT_ACTION_OP_WHITELIST:
            assert op in surface, f"op {op!r} not in FE surface"

    def test_generator_rejects_and_records(self):
        gen = StructuredGenerator()
        res = gen.generate(
            [
                {"action": "transform", "op": "zscore", "target": "parent_0"},
                {"action": "transform", "op": "foo_bar", "target": "parent_0"},
            ],
            _parents(),
        )
        assert len(res.candidates) == 1
        assert len(res.rejected) == 1
        assert res.rejected[0]["stage"] == "build"
        assert "not in whitelist" in res.rejected[0]["reason"]


# ---------------------------------------------------------------------------
# ③ FE validate 失败的 action 不进候选池
# ---------------------------------------------------------------------------


class TestFeValidateReject:
    def test_fe_validate_failure_not_in_pool(self):
        """FE 校验失败（语法无效）的 action 不进候选池，记录原因。"""
        gen = StructuredGenerator()
        # 构造一个 build 成功但 FE validate 失败的公式：zscore(rank(close) 缺右括号
        # （V31：parent 公式语法无效在 FE typed transform 阶段即被拒（build
        # stage）；两者都要求 candidates == [] 且 rejected 记录原因）。
        bad_parents = [{"factor_id": "p1", "formula": "rank(close"}]
        res = gen.generate(
            [{"action": "transform", "op": "zscore", "target": "parent_0"}],
            bad_parents,
        )
        assert res.candidates == []
        assert len(res.rejected) == 1
        assert res.rejected[0]["stage"] in ("build", "fe_validate")
        assert res.rejected[0]["reason"]

    def test_injected_validator_rejects(self):
        """注入假 validator 返回 False → 不进候选池。"""
        gen = StructuredGenerator(validator=lambda f: (False, "injected reject"))
        res = gen.generate(
            [{"action": "transform", "op": "zscore", "target": "parent_0"}], _parents()
        )
        assert res.candidates == []
        assert res.rejected[0]["stage"] == "fe_validate"
        assert "injected reject" in res.rejected[0]["reason"]

    def test_injected_validator_accepts(self):
        gen = StructuredGenerator(validator=lambda f: (True, "OK"))
        res = gen.generate(
            [{"action": "transform", "op": "zscore", "target": "parent_0"}], _parents()
        )
        assert len(res.candidates) == 1
        assert res.candidates[0].formula == "zscore(rank(close))"


# ---------------------------------------------------------------------------
# ④ stub llm_fn 确定性（同 seed 同输出）
# ---------------------------------------------------------------------------


class TestStubDeterminism:
    def test_same_seed_same_output(self):
        f1 = make_structured_stub_llm_fn(seed=42)
        f2 = make_structured_stub_llm_fn(seed=42)
        prompt = "## Parent factor(s)\n- rank(close) :: p\n- ts_mean(close, 20) :: q"
        assert f1(prompt, prompt, "cheap") == f2(prompt, prompt, "cheap")

    def test_different_seed_different_output(self):
        f1 = make_structured_stub_llm_fn(seed=1)
        f2 = make_structured_stub_llm_fn(seed=2)
        prompt = "## Parent factor(s)\n- rank(close) :: p"
        assert f1(prompt, prompt, "cheap") != f2(prompt, prompt, "cheap")

    def test_forced_actions_fixed(self):
        forced = [{"action": "transform", "op": "zscore", "target": "parent_0"}]
        f = make_structured_stub_llm_fn(seed=0, forced_actions=forced)
        out = f("sys", "user", "cheap")
        assert out == {"actions": list(forced)}

    def test_stub_output_normalizes_to_actions(self):
        f = make_structured_stub_llm_fn(seed=7)
        out = f("sys", "## Parent factor(s)\n- rank(close) :: p", "cheap")
        actions = normalize_llm_output(out)
        assert actions
        assert all("action" in a for a in actions)


# ---------------------------------------------------------------------------
# ⑤ lineage 记录完整
# ---------------------------------------------------------------------------


class TestLineage:
    def test_transform_lineage_single_parent(self):
        gen = StructuredGenerator()
        res = gen.generate(
            [{"action": "transform", "op": "zscore", "target": "parent_0"}], _parents()
        )
        assert res.lineages == [["p1"]]
        assert res.candidates[0].parent_ids == ["p1"]

    def test_combine_lineage_multi_parent(self):
        gen = StructuredGenerator()
        res = gen.generate(
            [
                {
                    "action": "combine",
                    "op": "add",
                    "parents": ["p1", "p2"],
                    "weights": [0.5, 0.5],
                }
            ],
            _parents(),
        )
        assert res.lineages == [["p1", "p2"]]
        assert res.candidates[0].parent_ids == ["p1", "p2"]

    def test_lineage_dedup_preserves_order(self):
        gen = StructuredGenerator()
        res = gen.generate(
            [
                {
                    "action": "combine",
                    "op": "add",
                    "parents": ["p2", "p1", "p2"],
                }
            ],
            _parents(),
        )
        assert res.lineages == [["p2", "p1"]]


# ---------------------------------------------------------------------------
# ⑥ 旧文本输出兼容路径（identity action fallback）
# ---------------------------------------------------------------------------


class TestLegacyCompatibility:
    def test_legacy_text_to_identity_actions(self):
        legacy = json.dumps(
            {
                "candidates": [
                    {
                        "formula": "rank(ts_std(close, 20))",
                        "action_type": "REFINE",
                        "parent_ids": ["p1"],
                    }
                ]
            }
        )
        actions = normalize_llm_output(legacy)
        assert len(actions) == 1
        assert actions[0]["action"] == "identity"
        assert actions[0]["formula"] == "rank(ts_std(close, 20))"

    def test_legacy_identity_flows_through_generator(self):
        """旧文本 → identity action → generator 产出原公式候选。"""
        legacy = json.dumps(
            {
                "candidates": [
                    {
                        "formula": "rank(ts_std(close, 20))",
                        "action_type": "REFINE",
                        "parent_ids": ["p1"],
                    }
                ]
            }
        )
        actions = normalize_llm_output(legacy)
        gen = StructuredGenerator()
        res = gen.generate(actions, _parents())
        assert len(res.candidates) == 1
        assert res.candidates[0].formula == "rank(ts_std(close, 20))"

    def test_none_output_empty(self):
        assert normalize_llm_output(None) == []

    def test_garbage_text_empty(self):
        assert normalize_llm_output("not json at all") == []


# ---------------------------------------------------------------------------
# ⑦ multi-parent wiring：启用后 parents 来自 select_parents（monkeypatch）
# ---------------------------------------------------------------------------


class TestMultiParentWiring:
    def test_structured_generation_uses_parent_selector(self, monkeypatch):
        """structured_generation=True 时，parents 来自 parent_selector 的 top-k 采样。"""
        from alphaprobe.search.orchestrator import SearchOrchestrator

        captured = {}

        class FakeSelector:
            def select_parents(self, candidates, k=5, **kw):
                captured["k"] = k
                captured["main"] = kw.get("main")
                captured["action"] = kw.get("action_to_retrieve")
                # 返回 top-k（这里固定返回前 k 个）
                return list(candidates[:k])

        orch = SearchOrchestrator(
            scheduler=None, parent_selector=FakeSelector(), structured_generation=True
        )
        llm = make_structured_stub_llm_fn(seed=7)
        parent = {"factor_id": "p1", "formula": "rank(close)", "fitness": 0.5}
        extra = [
            {"factor_id": "p2", "formula": "ts_mean(close, 20)", "fitness": 0.9},
            {"factor_id": "p3", "formula": "ts_std(close, 20)", "fitness": 0.6},
        ]
        step = orch.step(parent, llm, extra_parents=extra, top_k=2)
        # select_parents 被调用，且 k=2
        assert captured["k"] == 2
        assert captured["main"] == parent
        # 候选的 lineage 覆盖了来自 select_parents 的补充 parent
        all_parent_ids = {pid for c in step.candidates for pid in c.parent_ids}
        assert "p1" in all_parent_ids
        assert "p2" in all_parent_ids
        assert "p3" in all_parent_ids

    def test_structured_generation_default_off_no_behavior_change(self):
        """默认 structured_generation=False：不触发结构化路径，行为与旧版一致。"""
        from alphaprobe.search.orchestrator import SearchOrchestrator

        orch = SearchOrchestrator(scheduler=None)
        assert orch.structured_generation is False
        assert orch.structured_generator is None

    def test_structured_generation_off_uses_legacy_parse(self):
        """关闭时走旧 parse_generated 路径（文本输出）。"""
        from alphaprobe.search.orchestrator import SearchOrchestrator

        orch = SearchOrchestrator(scheduler=None)

        def _legacy_llm(sys_p, usr_p, model_class):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "formula": "rank(ts_std(close, 20))",
                            "action_type": "REFINE",
                            "parent_ids": ["p1"],
                        }
                    ]
                }
            )

        step = orch.step(
            {"factor_id": "p1", "formula": "rank(ts_mean(close, 20))"}, _legacy_llm
        )
        assert step.candidates
        assert step.candidates[0].formula == "rank(ts_std(close, 20))"

    def test_pipeline_config_structured_flag(self):
        """PipelineConfig.structured_generation 透传到 orchestrator。"""
        from alphaprobe.pipeline import PipelineConfig, SearchPipeline

        cfg = PipelineConfig(structured_generation=True)
        sp = SearchPipeline(experiment=None, data_train=None, config=cfg)
        assert sp.orchestrator.structured_generation is True

        cfg2 = PipelineConfig()
        sp2 = SearchPipeline(experiment=None, data_train=None, config=cfg2)
        assert sp2.orchestrator.structured_generation is False

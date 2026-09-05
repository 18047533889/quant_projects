"""plan.md Task 16：CogAlpha-style multi-stage generation（AlphaPROBE 差异化）。

覆盖 plan 三条 + Non-negotiable #18/#30：
1. 畸形 LLM action 永远到不了 FE 执行（stage 1 action schema 即拦）；
2. hypothesis 在 implementation 失败后存活，可生成替代 implementation
   （hypothesis 与实现解耦，失败不杀 hypothesis）；
3. 确定性 alignment 能在**没有第二次 LLM 调用**的情况下拒绝明显 domain 矛盾
   （stage 5 确定性对齐，不调 LLM）；
4. 多阶段/预算截断均可配置开关（ablation switch，#30）；
5. 九阶段骨架以 pipeline hook/状态机实现，不造 20 个常驻 agent。

实现文件：research_space/hypothesis.py（新建）+ search/orchestrator.py 多阶段 hook。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from alphaprobe.research_space.contracts import (
    ALIGNMENT_FULL_SCORE,
    ALIGNMENT_MISMATCH_SCORE,
    AlignmentResult,
    HypothesisSpec,
)
from alphaprobe.research_space.hypothesis import (
    DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS,
    HypothesisRecord,
    HypothesisRegistry,
    ImplementationAttempt,
    Stage,
    StageHooks,
    run_stage_pipeline,
)
from alphaprobe.search.orchestrator import SearchOrchestrator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spec(**kw: Any) -> HypothesisSpec:
    base: dict[str, Any] = {
        "hypothesis_id": "h1",
        "text": "price-volume divergence alpha",
        "expected_domains": ("PRICE", "VOLUME"),
        "expected_roles": ("price_level", "volume_traded"),
        "expected_horizon": "medium",
    }
    base.update(kw)
    return HypothesisSpec(**base)


def _hook(which: str) -> StageHooks:
    """返回只启用某一个 stage 的 hooks（其余返回未变 context）。"""
    h = StageHooks()
    for attr in ("on_stage_1", "on_stage_2", "on_stage_3", "on_stage_4",
                 "on_stage_5", "on_stage_6", "on_stage_7", "on_stage_8",
                 "on_stage_9"):
        base = getattr(h, attr)

        def make(attr: str = attr, base: Callable = base) -> Callable:
            def wrapper(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
                if attr == which:
                    return base(stage, ctx, rec)
                return dict(ctx)
            return wrapper

        setattr(h, attr, make())
    return h


def _blocker(result: AlignmentResult) -> Callable[[str, dict[str, Any], Any], dict[str, Any]]:
    def on_stage_5(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
        new_ctx = dict(ctx)
        new_ctx["alignment_result"] = result
        return new_ctx
    return on_stage_5


# ---------------------------------------------------------------------------
# 1. 畸形 LLM action 永远到不了 FE 执行
# ---------------------------------------------------------------------------


def test_malformed_action_never_reaches_fe_execution():
    registry = HypothesisRegistry()
    spec = _spec()
    record = registry.register(spec)
    fe_calls: list[str] = []
    actions = [
        {"action": "transform", "op": "not_a_real_op", "target": "parent_0"},
        {"action": "combine", "parents": ["parent_0"]},  # <2 parents
        {"action": "bogus"},  # 非 transform/combine/identity
        {"action": "transform", "op": "zscore", "target": "parent_0"},  # 唯一合法
    ]
    hooks = StageHooks()
    # stage 4 = FE 编译执行（把公式文本塞进 ctx["formula"] 以证明“执行了”）
    def on_stage_4(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
        new_ctx = dict(ctx)
        new_ctx["formula"] = f"zscore({ctx.get('parent_formula', 'close')})"
        fe_calls.append(str(new_ctx.get("action")))
        return new_ctx
    hooks.on_stage_4 = on_stage_4

    out = run_stage_pipeline(
        actions=actions,
        hypothesis=record,
        parents=[{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}],
        hooks=hooks,
        op_whitelist=frozenset({"zscore"}),
    )
    # 只有合法 action 到达 FE 执行（stage 3 白名单把未知 op 拦下；stage 4
    # 回调只被合法 action 触发）
    assert len(fe_calls) == 1
    assert out["implemented"] == [3]  # 只有下标 3 的 action（唯一合法）被实现
    # 其余三条被拦在 stage 1/3（未到 FE）
    assert len(out["rejected"]) == 3
    rejected_types = {r["reason"] for r in out["rejected"]}
    assert rejected_types  # 每条都有 reason


def test_malformed_action_stage1_intercept_count_no_stage4_calls():
    """所有 action 都畸形时 FE 零调用（stage 1 即拦）。"""
    registry = HypothesisRegistry()
    record = registry.register(_spec())
    fe_calls: list[str] = []
    hooks = StageHooks()
    hooks.on_stage_4 = lambda st, ctx, rec: fe_calls.append("x") or dict(ctx)
    out = run_stage_pipeline(
        actions=[{"op": "zscore", "target": "parent_0"}, {"action": "combine"}],
        hypothesis=record,
        parents=[{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}],
        hooks=hooks,
    )
    assert fe_calls == []
    assert len(out["implemented"]) == 0
    assert record.state == "proposed"  # 未实现 → 仍在 proposed
    assert record.implementation_attempts == 0


# ---------------------------------------------------------------------------
# 2. hypothesis 在 implementation 失败后存活，可生成替代实现
# ---------------------------------------------------------------------------


def test_hypothesis_survives_implementation_failure_and_alternative():
    registry = HypothesisRegistry()
    record = registry.register(_spec())
    parents = [{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}]

    def compile_ok(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
        new_ctx = dict(ctx)
        new_ctx["formula"] = f"zscore({ctx.get('parent_formula', 'close')})"
        return new_ctx

    def fe_reject(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
        """stage 7 FE 静态分析/校验：拒绝实现 #1，放行实现 #2。"""
        new_ctx = dict(ctx)
        if rec.implementation_attempts <= 1:  # 第一次实现
            new_ctx["fe_validation"] = (False, "lookahead risk")
        else:
            new_ctx["fe_validation"] = (True, "")
        return new_ctx

    # 第一次尝试：实现失败
    hooks = StageHooks()
    hooks.on_stage_4 = compile_ok
    hooks.on_stage_7 = fe_reject
    out1 = run_stage_pipeline(
        actions=[{"action": "transform", "op": "zscore", "target": "parent_0"}],
        hypothesis=record,
        parents=parents,
        hooks=hooks,
    )
    assert out1["implemented"] == []
    assert len(out1["rejected"]) == 1
    assert record.state == "implementation_failed"  # 失败但不死
    assert record.implementation_attempts == 1

    # 第二次尝试：替代实现成功（同一 hypothesis 记录）
    out2 = run_stage_pipeline(
        actions=[{"action": "transform", "op": "zscore", "target": "parent_0"}],
        hypothesis=record,
        parents=parents,
        hooks=hooks,
    )
    assert len(out2["implemented"]) == 1
    assert record.state == "implemented"
    assert record.implementation_attempts == 2
    assert record.best_formula == out2["contexts"][0]["formula"]


def test_hypothesis_max_alternative_implementations_cap():
    """替代实现尝试有上限（默认 3），用尽后不再自动重试。"""
    registry = HypothesisRegistry()
    record = registry.register(_spec())
    parents = [{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}]

    def always_reject(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
        new_ctx = dict(ctx)
        new_ctx["fe_validation"] = (False, "always fails")
        return new_ctx

    hooks = StageHooks()
    hooks.on_stage_4 = lambda st, ctx, rec: dict(ctx, formula="zscore(close)")
    hooks.on_stage_7 = always_reject
    for _ in range(DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS + 2):
        out = run_stage_pipeline(
            actions=[{"action": "transform", "op": "zscore", "target": "parent_0"}],
            hypothesis=record,
            parents=parents,
            hooks=hooks,
        )
    # 尝试次数被 cap 住（默认 3）
    assert record.implementation_attempts <= DEFAULT_MAX_ALTERNATIVE_IMPLEMENTATIONS
    # 即使调用方再多试，attempt 计数不超 cap；hypothesis 记录仍在（未被删除）
    assert record.state == "implementation_failed"


# ---------------------------------------------------------------------------
# 3. 确定性 alignment 拒绝 domain 矛盾（无第二次 LLM 调用）
# ---------------------------------------------------------------------------


def test_alignment_rejects_domain_contradiction_without_llm():
    """假设声称 FUNDAMENTAL，实现只用 PRICE → 拦截（不用第二次 LLM 判定）。"""
    registry = HypothesisRegistry()
    spec = _spec(expected_domains=("FUNDAMENTAL.QUALITY",))
    record = registry.register(spec)
    llm_calls: list[str] = []
    parents = [{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}]

    hooks = StageHooks()
    hooks.on_stage_4 = lambda st, ctx, rec: dict(ctx, formula="zscore(close)")
    hooks.on_stage_6 = lambda st, ctx, rec: llm_calls.append("critic") or dict(ctx)  # 可选 critic
    # stage 5 确定性对齐：真实 P2-A DeterministicAligner + PRICE 行情 taxonomy，
    # 实现只用 close → 与 FUNDAMENTAL.QUALITY 声称矛盾 → hard_mismatch。
    def on_stage_5(stage: str, ctx: dict[str, Any], rec: Any) -> dict[str, Any]:
        new_ctx = dict(ctx)
        from alphaprobe.research_space.alignment import (
            DeterministicAligner,
            bare_field_name,
        )

        class _Usage:
            canonical_field_id = "StockDailyBarAdj.close"

        class _Analysis:
            field_usages = [_Usage()]
            existing_treatment_semantic_ids = []
            timing_flags = {}

        class _Descriptor:
            data_domains = ("PRICE",)
            pit_class = "panel_same_day"
            economic_roles = ()

        class _Taxonomy:
            def describe_fields(self, ids):
                return {
                    str(c): _Descriptor() for c in ids if bare_field_name(str(c)) == "close"
                }

        analysis = _Analysis()
        aligner = DeterministicAligner(
            taxonomy=_Taxonomy(), mode="offline_test", critic=None
        )
        new_ctx["alignment_result"] = aligner.align(rec.spec, analysis)
        return new_ctx

    hooks.on_stage_5 = on_stage_5

    out = run_stage_pipeline(
        actions=[{"action": "transform", "op": "zscore", "target": "parent_0"}],
        hypothesis=record,
        parents=parents,
        hooks=hooks,
    )
    assert out["implemented"] == []  # 被对齐拦截
    assert "hypothesis_domain_contradiction" in out["blocked"]
    # 关键：critic 没有被调用（确定性拦截后无第二次 LLM）
    assert llm_calls == []
    # hypothesis 记录被标记 blocked（不进入后续实现）
    assert record.state == "blocked"


def test_alignment_pass_goes_to_fe_and_implements():
    """确定性对齐通过 → 继续 stage 6+（critic 可选被调用）→ FE 验证 → implemented。"""
    registry = HypothesisRegistry()
    record = registry.register(_spec())  # PRICE+VOLUME 声称
    parents = [{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}]

    hooks = StageHooks()
    hooks.on_stage_4 = lambda st, ctx, rec: dict(ctx, formula="zscore(volume)")
    hooks.on_stage_5 = lambda st, ctx, rec: dict(
        ctx, alignment_result=AlignmentResult(score=ALIGNMENT_FULL_SCORE)
    )
    hooks.on_stage_7 = lambda st, ctx, rec: dict(ctx, fe_validation=(True, ""))
    out = run_stage_pipeline(
        actions=[{"action": "transform", "op": "zscore", "target": "parent_0"}],
        hypothesis=record,
        parents=parents,
        hooks=hooks,
    )
    assert len(out["implemented"]) == 1
    assert record.state == "implemented"


# ---------------------------------------------------------------------------
# 4. 可配置开关（ablation switch，#30）
# ---------------------------------------------------------------------------


def test_multistage_can_be_toggled_off():
    """多阶段生成可整体关闭（搜索侧配置开关默认 False → 行为与旧版一致）。"""
    orch = SearchOrchestrator()
    assert orch.multistage_generation is False  # ablation switch 默认关闭


# ---------------------------------------------------------------------------
# 5. 状态机与 stage 常量
# ---------------------------------------------------------------------------


def test_stage_constants_nine_stages():
    assert list(Stage) == [
        "STAGE_1_HYPOTHESIS",
        "STAGE_2_SCHEMA_PLAN",
        "STAGE_3_AST_ACTION_PLAN",
        "STAGE_4_FE_COMPILE",
        "STAGE_5_DETERMINISTIC_ALIGNMENT",
        "STAGE_6_OPTIONAL_CRITIC",
        "STAGE_7_FE_VALIDATION",
        "STAGE_8_DEDUP",
        "STAGE_9_EVALUATION_FEEDBACK",
    ]


def test_stage_transitions_record_history():
    registry = HypothesisRegistry()
    record = registry.register(_spec())
    parents = [{"formula": "rank(ts_mean(close, 20))", "factor_id": "p1"}]
    hooks = StageHooks()
    hooks.on_stage_4 = lambda st, ctx, rec: dict(ctx, formula="zscore(close)")
    hooks.on_stage_5 = lambda st, ctx, rec: dict(
        ctx, alignment_result=AlignmentResult(score=1.0)
    )
    hooks.on_stage_7 = lambda st, ctx, rec: dict(ctx, fe_validation=(True, ""))
    run_stage_pipeline(
        actions=[{"action": "transform", "op": "zscore", "target": "parent_0"}],
        hypothesis=record,
        parents=parents,
        hooks=hooks,
    )
    # history 里记录推进过的 stage
    stages_seen = [s for s, _, _ in record.history]
    assert "STAGE_5_DETERMINISTIC_ALIGNMENT" in stages_seen
    assert record.state == "implemented"


# ---------------------------------------------------------------------------
# 额外：registry 持久化与 load
# ---------------------------------------------------------------------------


def test_registry_roundtrip(tmp_path):
    reg1 = HypothesisRegistry(db_path=tmp_path / "h.sqlite3")
    rec = reg1.register(_spec(text="persist me"))
    reg1.record_impl_failure(rec.hypothesis_id, reason="FE invalid")
    reg2 = HypothesisRegistry(db_path=tmp_path / "h.sqlite3")
    loaded = reg2.get(rec.hypothesis_id)
    assert loaded is not None
    assert loaded.state == "implementation_failed"
    assert loaded.implementation_attempts == 1
    assert loaded.spec.text == "persist me"

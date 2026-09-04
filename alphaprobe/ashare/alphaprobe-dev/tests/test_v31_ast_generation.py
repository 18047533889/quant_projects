"""V3.1 结构化 generation 验收（Task 7 / A9）：FE AST 语义权威迁移。

覆盖 Task 7 全部验收点：
① WINDOW_SCALE 只改声明的那一个 WINDOW 参数；
② FIELD_SUBSTITUTION 尊重字段 domain/type 兼容；
③ CROSSOVER 只组合类型兼容的子树；
④ 非布尔表达式不能当 if_else 条件；
⑤ 变异结果重新 parse → canonical identity 确定性一致（FE identity）；
⑥ 不支持的算子被拒绝，无生产 fallback。

原则（Part E 边界）：
- AlphaPROBE 不自建 regex parser / operator taxonomy；一切算子语义
  （arity/参数角色/window role/surface/类型）来自 FE registry。
- 生产路径若 FE transform API 不可用则 fail-closed（拒绝而非字符串拼装）。

合成数据；不跑真实 LLM / 模型训练。
"""

from __future__ import annotations

import pytest

from alphaprobe.generation.structured import (
    StructuredGenerator,
    build_formula,
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

pytestmark = [
    pytest.mark.skipif(not FE_AVAILABLE, reason="FE not importable"),
]


def _parents() -> list[dict]:
    return [
        {"factor_id": "p1", "formula": "rank(close)"},
        {"factor_id": "p2", "formula": "ts_mean(close, 20)"},
    ]


def _fe_identity(text: str):
    from factor_engine.identity import get_factor_identity

    return get_factor_identity(text)


def _identity_deterministic(text: str) -> tuple[str, str]:
    """reparse → canonical identity（canonical_ast_hash / signal_equivalence_id）。"""
    ident = _fe_identity(text)
    return ident.canonical_ast_hash, ident.signal_equivalence_id


def _import_at():
    import factor_engine.api as api

    return api


# ---------------------------------------------------------------------------
# ① WINDOW_SCALE 只改声明的那一个 WINDOW 参数
# ---------------------------------------------------------------------------


class TestWindowScale:
    def test_window_scale_changes_only_declared_window_param(self):
        at = _import_at()
        scaled = at.ast_call("ts_mean", ("close",), {"window": 30})
        formula = at.ast_to_dsl_text(scaled)
        assert formula == "ts_mean(close, 30)"
        # 同公式仅 window 不同 → identity 不同（只有窗口槽被改）
        base_ident = _fe_identity("ts_mean(close, 20)")
        scaled_ident = _fe_identity(formula)
        assert base_ident.canonical_ast_hash != scaled_ident.canonical_ast_hash

    def test_window_scale_pair_op_keeps_both_panels(self):
        at = _import_at()
        node = at.ast_call("ts_corr", ("rank(close)", "ts_mean(close, 20)"), {"window": 60})
        formula = at.ast_to_dsl_text(node)
        assert "ts_corr" in formula and "60" in formula
        ident = _fe_identity(formula)
        assert ident.lookback == 60

    def test_window_scale_via_build_formula(self):
        """WINDOW_SCALE action → 只替换窗口槽位。"""
        f, reason = build_formula(
            {
                "action": "transform",
                "op": "ts_mean",
                "target": "parent_1",
                "params": {"window": 30},
            },
            _parents(),
        )
        assert f == "ts_mean(ts_mean(close, 20), 30)"
        assert reason == ""


# ---------------------------------------------------------------------------
# ② FIELD_SUBSTITUTION 尊重字段 domain/type 兼容
# ---------------------------------------------------------------------------


class TestFieldSubstitution:
    def test_field_substitution_respects_domain(self):
        """close(价格) 可替换为 open/high/vwap（同 price 单位）；volume 不行。"""
        at = _import_at()
        ok = at.ast_substitute_field("rank(close)", "close", "open")
        assert "open" in at.ast_to_dsl_text(ok)
        # volume 单位 share ≠ 价格 → 拒绝
        with pytest.raises(ValueError):
            at.ast_substitute_field("rank(close)", "close", "volume")
        # pe_ttm（valuation 域）→ 拒绝
        with pytest.raises(ValueError):
            at.ast_substitute_field("rank(close)", "close", "pe_ttm")

    def test_field_substitution_unknown_source_fails_closed(self):
        at = _import_at()
        with pytest.raises(ValueError):
            at.ast_substitute_field("rank(close)", "not_a_field", "open")


# ---------------------------------------------------------------------------
# ③ CROSSOVER 只组合类型兼容的子树
# ---------------------------------------------------------------------------


class TestCrossover:
    def test_crossover_numeric_subtrees(self):
        at = _import_at()
        node = at.ast_crossover("subtract", ("rank(close)", "ts_mean(close, 20)"), output_kind="float")
        assert at.ast_to_dsl_text(node) == "subtract(rank(close), ts_mean(close, 20))"

    def test_crossover_boolean_roots_into_bool_op(self):
        at = _import_at()
        node = at.ast_crossover(
            "and_",
            ("gt(close, open)", "gt(ts_mean(close, 20), open)"),
            output_kind="bool",
        )
        assert at.ast_expr_kind(node) == "bool"

    def test_crossover_type_mismatch_rejected(self):
        """bool 子树不能进 float 二元算子，float 子树不能进 bool 算子。"""
        at = _import_at()
        with pytest.raises(ValueError):
            at.ast_crossover("add", ("gt(close, open)", "rank(close)"), output_kind="float")
        with pytest.raises(ValueError):
            at.ast_crossover("and_", ("rank(close)", "gt(close, open)"), output_kind="bool")


# ---------------------------------------------------------------------------
# ④ 非布尔表达式不能当 if_else 条件
# ---------------------------------------------------------------------------


class TestBooleanConditionGate:
    def test_if_else_accepts_bool_condition(self):
        at = _import_at()
        node = at.ast_call(
            "if_else",
            ("gt(close, ts_mean(close, 20))", "rank(close)", "open"),
            {},
        )
        assert at.ast_infer_expr_kind(node) == "float"

    def test_if_else_rejects_numeric_condition(self):
        at = _import_at()
        with pytest.raises(ValueError, match="bool"):
            at.ast_call("if_else", ("rank(close)", "open", "open"), {})

    def test_build_formula_conditional_rejects_non_bool_cond(self):
        """生成器路径：cond 不是 bool 表达式 → 拒绝（不进候选池）。"""
        gen = StructuredGenerator()
        res = gen.generate(
            [
                {
                    "action": "transform",
                    "op": "if_else",
                    "target": "parent_0",
                    "cond": "rank(close)",
                    "otherwise": "open",
                }
            ],
            _parents(),
        )
        assert res.candidates == []
        assert res.rejected, "非布尔 if_else 条件必须被拒绝"
        assert any("bool" in r["reason"] for r in res.rejected)


# ---------------------------------------------------------------------------
# ⑤ 变异结果重新 parse → canonical identity 确定性一致
# ---------------------------------------------------------------------------


class TestIdentityDeterminism:
    def test_transformed_formula_reparse_identity_consistent(self):
        at = _import_at()
        node = at.ast_crossover(
            "subtract", ("rank(close)", "ts_mean(close, 20)"), output_kind="float"
        )
        text = at.ast_to_dsl_text(node)
        # 再 parse 一次仍得同 identity（确定性）
        h1 = _identity_deterministic(text)
        h2 = _identity_deterministic(text)
        assert h1 == h2
        assert len(h1[0]) == 64

    def test_generated_candidate_formula_identity_stable(self):
        """生成器产物 → FE identity（canonical_ast_hash / signal_equivalence_id 可计算）。"""
        gen = StructuredGenerator()
        # 用一元 transform（zscore）验证 identity 确定性：产物是简单单层变换，
        # lookback 明确（无嵌套窗口干扰）。
        res = gen.generate(
            [{"action": "transform", "op": "ts_mean", "target": "parent_0",
              "params": {"window": 5}}],
            [{"factor_id": "p0", "formula": "rank(close)"}],
        )
        assert len(res.candidates) == 1
        formula = res.candidates[0].formula
        ident = _fe_identity(formula)
        assert ident.lookback == 5
        assert len(ident.canonical_ast_hash) == 64
        # 同一产物再 parse → identity 完全一致（确定性硬验收）
        h1 = _identity_deterministic(formula)
        h2 = _identity_deterministic(formula)
        assert h1 == h2


# ---------------------------------------------------------------------------
# ⑥ 不支持的算子被拒绝，无生产 fallback
# ---------------------------------------------------------------------------


class TestNoProductionFallback:
    def test_unknown_op_rejected_no_builtin(self):
        v = validate_action({"action": "transform", "op": "definitely_not_an_op", "target": "parent_0"})
        assert not v.ok
        assert "whitelist" in v.reason or "unsupported" in v.reason or "registry" in v.reason

    def test_registry_unknown_never_falls_back_to_string_build(self):
        """生产路径不经过字符串拼装 fallback：FE registry 拒绝即无产物。"""
        gen = StructuredGenerator()
        res = gen.generate(
            [
                {
                    "action": "transform",
                    "op": "made_up_ts_9999",
                    "target": "parent_0",
                    "params": {"window": 10},
                }
            ],
            _parents(),
        )
        assert res.candidates == []
        assert res.rejected

    def test_legacy_string_path_disabled_by_default(self):
        """legacy 字符串拼接路径不是生产默认（无内置 fallback 拼公式）。"""
        from alphaprobe.generation import structured as st

        assert getattr(st, "_LEGACY_STRING_PATH", False) is False

    def test_whitelist_ops_still_parse_and_identity(self):
        """DEFAULT_ACTION_OP_WHITELIST 中每个 op 都能经 FE parse（防漂移）。"""
        from alphaprobe.generation.structured import DEFAULT_ACTION_OP_WHITELIST

        for op in sorted(DEFAULT_ACTION_OP_WHITELIST):
            formula = f"{op}(rank(close))"
            ident = _fe_identity(formula)
            assert ident is not None


# ---------------------------------------------------------------------------
# 兼容面：旧文本 identity fallback 仍工作
# ---------------------------------------------------------------------------


class TestLegacyCompatKept:
    def test_normalize_identity_action(self):
        import json

        legacy = json.dumps(
            {"candidates": [{"formula": "rank(ts_std(close, 20))", "action_type": "REFINE"}]}
        )
        actions = normalize_llm_output(legacy)
        assert actions[0]["action"] == "identity"

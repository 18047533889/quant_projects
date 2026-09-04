# -*- coding: utf-8
"""``factor_engine.api.ast_transform`` —— typed AST transform 公共 API 测试。

覆盖 Task 7 / A9 需要的通用、带类型校验的 Expr transform 原语：

- ``resolve_op_semantics``：operator arity / 参数角色 / window role 全部来自
  registry / signature 契约（禁止调用方自带本地 taxonomy）。
- ``call``：按声明的 panel-arity 绑定位置参数；窗口参数只允许正整数 literal。
- ``infer_expr_kind`` / ``expr_kind``：输入/输出类型推断（float/bool/group）。
- ``WINDOW_SCALE``：只改声明为 WINDOW 角色的参数，绝不改动其他参数。
- ``FIELD_SUBSTITUTION``：字段域/类型兼容才替换（价格域 → 价格域）；
  不兼容（close → volume 跨语义域）显式拒绝。
- ``CROSSOVER``：只组合类型兼容子树；根输出类型显式声明。
- ``if_else`` 条件必须是 bool Series（非布尔表达式不能当条件）。
- transform 结果 serialize → ``parse_expr`` → canonical identity 确定性一致。
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    """与 tests/api 既有约定一致：先 load_all 再使用 registry。"""
    from factor_engine.cleaned_operators import load_all

    load_all()


def _idents_equal(formula_a: str, formula_b: str) -> bool:
    """两公式 canonical identity（canonical_ast_hash / signal_equivalence_id）一致。"""
    from factor_engine.identity import get_factor_identity

    ia = get_factor_identity(formula_a)
    ib = get_factor_identity(formula_b)
    return (
        ia.canonical_ast_hash == ib.canonical_ast_hash
        and ia.signal_equivalence_id == ib.signal_equivalence_id
    )


class TestOpSemanticsFromRegistry:
    """operator 语义唯一来自 registry/signature——FE 侧权威。"""

    def test_unary_op_arity(self, _loaded):
        from factor_engine.api import ast_transform as at

        info = at.resolve_op_semantics("zscore")
        assert info.canonical == "zscore"
        assert info.panel_count == 1
        assert info.panel_names == ("x",)
        assert info.window_names == ()

    def test_ts_mean_window_role_from_param_role(self, _loaded):
        from factor_engine.api import ast_transform as at

        info = at.resolve_op_semantics("ts_mean")
        assert info.canonical == "ts_mean"
        assert info.panel_count == 1
        # ts_mean 的 window 参数带 ParamRole.HORIZON（搜索角色）+ 历史语义
        assert "window" in info.window_names
        assert "min_periods" not in info.window_names

    def test_where_conditional_arity(self, _loaded):
        from factor_engine.api import ast_transform as at

        info = at.resolve_op_semantics("where")
        assert info.canonical == "where"
        assert info.panel_count == 3

    def test_add_variadic_panel_arity(self, _loaded):
        from factor_engine.api import ast_transform as at

        info = at.resolve_op_semantics("add")
        assert info.canonical == "add"
        assert info.panel_count == 2
        assert info.allow_variadic is True

    def test_unknown_operator_fails_closed(self, _loaded):
        from factor_engine.api import ast_transform as at

        with pytest.raises(ValueError, match="unsupported operator"):
            at.resolve_op_semantics("no_such_op_v31")


class TestCallValidation:
    """call() 按声明的 arity/类型校验；非法输入 fail-closed。"""

    def test_call_ts_mean_window_positional(self, _loaded):
        from factor_engine.api import ast_transform as at

        node = at.call("ts_mean", ("close",), {"window": 20})
        assert node.op == "ts_mean"
        assert len(node.args) == 2

    def test_call_window_must_be_positive_int_literal(self, _loaded):
        from factor_engine.api import ast_transform as at

        with pytest.raises(ValueError, match="window"):
            at.call("ts_mean", ("close",), {"window": 0})
        with pytest.raises(ValueError, match="window"):
            at.call("ts_mean", ("close",), {"window": -3})
        with pytest.raises(ValueError, match="window"):
            at.call("ts_mean", ("close",), {"window": 2.5})

    def test_call_window_changes_only_window_slot(self, _loaded):
        """WINDOW_SCALE：只改声明的 WINDOW 参数；其余位置原样保留。"""
        from factor_engine.api import ast_transform as at

        base = at.call("ts_mean", ("close",), {"window": 5})
        scaled = at.call("ts_mean", ("close",), {"window": 21})
        assert base.args[0] == scaled.args[0]
        assert base.args[1] != scaled.args[1]

    def test_call_rejects_extra_positional_beyond_panel(self, _loaded):
        from factor_engine.api import ast_transform as at

        # rank 只有 1 个 panel 输入；多给位置参数必须拒绝
        with pytest.raises(ValueError, match="arity|panel"):
            at.call("rank", ("close", "open"), {})

    def test_call_pair_op_two_panels(self, _loaded):
        from factor_engine.api import ast_transform as at

        node = at.call("ts_corr", ("close", "open"), {"window": 20})
        assert node.op == "ts_corr"
        assert len(node.args) == 3


class TestExprTypeInference:
    """输入类型推断 + where/if_else 布尔条件门。"""

    def test_infer_numeric_series(self, _loaded):
        from factor_engine.api import ast_transform as at

        assert at.infer_expr_kind("close") == "float"
        assert at.infer_expr_kind("ts_mean(close, 5)") == "float"

    def test_gt_output_bool(self, _loaded):
        from factor_engine.api import ast_transform as at

        assert at.infer_expr_kind("gt(close, open)") == "bool"

    def test_where_accepts_bool_condition(self, _loaded):
        from factor_engine.api import ast_transform as at

        node = at.call(
            "where",
            ("gt(close, open)", "close", "open"),
            {},
        )
        assert node.op == "where"
        assert len(node.args) == 3

    def test_where_rejects_numeric_condition(self, _loaded):
        """非布尔表达式不能当 if_else/where 条件。"""
        from factor_engine.api import ast_transform as at

        with pytest.raises(ValueError, match="condition.*bool"):
            at.call("where", ("close", "open", "open"), {})

    def test_if_else_alias_resolves_where(self, _loaded):
        from factor_engine.api import ast_transform as at

        info = at.resolve_op_semantics("if_else")
        assert info.canonical == "where"
        assert info.cond_slot_names == ("condition",)
        with pytest.raises(ValueError, match="condition.*bool"):
            at.call("if_else", ("ts_mean(close, 5)", "open", "open"), {})


class TestCrossoverCompose:
    """CROSSOVER 只组合类型兼容的子树。"""

    def test_crossover_numeric_pair(self, _loaded):
        from factor_engine.api import ast_transform as at

        expr = at.crossover(
            "subtract",
            ("rank(close)", "ts_mean(close, 20)"),
            output_kind="float",
        )
        assert expr.op == "subtract"
        assert len(expr.args) == 2
        assert at.expr_kind(expr) == "float"

    def test_crossover_boolean_pair(self, _loaded):
        from factor_engine.api import ast_transform as at

        expr = at.crossover(
            "and_",
            ("gt(close, open)", "gt(volume, 0)"),
            output_kind="bool",
        )
        assert expr.op == "and_"
        assert at.expr_kind(expr) == "bool"

    def test_crossover_bool_input_to_float_op_rejected(self, _loaded):
        from factor_engine.api import ast_transform as at

        with pytest.raises(ValueError, match="bool|float|type"):
            at.crossover("add", ("gt(close, open)", "rank(close)"), output_kind="float")

    def test_crossover_numeric_into_bool_op_rejected(self, _loaded):
        from factor_engine.api import ast_transform as at

        with pytest.raises(ValueError, match="bool|float|type"):
            at.crossover("and_", ("rank(close)", "gt(close, open)"), output_kind="bool")


class TestFieldSubstitutionCompat:
    """FIELD_SUBSTITUTION 尊重字段 domain/type 兼容。"""

    def test_same_domain_substitution(self, _loaded):
        from factor_engine.api import ast_transform as at

        # close → open（同为行情价格域 numeric）应当允许
        result = at.substitute_field("rank(close)", "close", "open")
        assert at.infer_expr_kind(result) == "float"
        assert "open" in str(result)

    def test_cross_domain_substitution_rejected(self, _loaded):
        from factor_engine.api import ast_transform as at

        # close(价格域) → pe_ttm/volume(不同域) 不兼容 → 拒绝
        with pytest.raises(ValueError, match="domain|incompatible"):
            at.substitute_field("rank(close)", "close", "volume")
        with pytest.raises(ValueError, match="domain|incompatible"):
            at.substitute_field("rank(close)", "close", "pe_ttm")

    def test_unknown_source_field_fails_closed(self, _loaded):
        from factor_engine.api import ast_transform as at

        with pytest.raises(ValueError, match="unknown field|not a registered"):
            at.substitute_field("rank(close)", "not_a_field", "open")


class TestSerializationDeterminism:
    """transform 产物 → DSL 文本 → reparse → canonical identity 确定性。"""

    def test_reparse_identity_after_transform(self, _loaded):
        from factor_engine.api import ast_transform as at

        base = at.call("ts_mean", ("close",), {"window": 20})
        formula = at.to_dsl_text(base)
        assert _idents_equal("ts_mean(close, 20)", formula)

    def test_identity_deterministic_across_construction(self, _loaded):
        from factor_engine.api import ast_transform as at

        f1 = at.to_dsl_text(at.call("ts_mean", ("close",), {"window": 20}))
        f2 = "ts_mean(close, 20)"
        assert _idents_equal(f1, f2)

    def test_transformed_canonical_ast_hash_stable(self, _loaded):
        from factor_engine.api import ast_transform as at
        from factor_engine.identity import get_factor_identity

        node = at.crossover(
            "subtract", ("rank(close)", "ts_mean(close, 20)"), output_kind="float"
        )
        ident = get_factor_identity(at.to_dsl_text(node))
        assert len(ident.canonical_ast_hash) == 64
        assert len(ident.signal_equivalence_id) == 64

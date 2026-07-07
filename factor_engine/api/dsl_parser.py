"""因子 DSL 解析：字符串公式 → ``Expr`` 树。

支持的语法（受限 Python ``eval`` 子集）
--------------------------------------
- 四则运算、比较（``>`` ``<`` ``==`` 等会 lower 为 ``gt``/``lt``/``eq`` 算子）、一元 ``-``；
- 函数调用：函数名必须在 ``build_dsl_allowlist()`` 内（``col``、``rank``、``ts_mean`` 等）；
- 字面量：数值、字符串（字段名由 ``col('...')`` 引用，勿裸写未定义变量）。

不支持：任意 import、属性访问、列表推导、lambda、未白名单的第三方函数。

典型流程::

    from api.dsl_parser import parse_expr
    expr = parse_expr("rank(ts_mean(col('close'), 20))")
    # → CleanedCall 嵌套树，再经 IR → PlanNode → PandasBackend 执行
"""

from __future__ import annotations

import ast
from typing import Any

from api.columns import col
from api.factor import Factor
from api.operator_registry import build_dsl_allowlist
from expr.base import Expr


class DSLParseError(ValueError):
    """公式语法不合法，或使用了未白名单的函数/结构。"""


class _ExprBuilder:
    """遍历 ``ast``，把调用/比较/四则运算还原为 ``CleanedCall`` / ``Expr`` 树。"""

    def __init__(self) -> None:
        # 允许出现的函数名 → 工厂（来自 operator_registry + cleaned_operators）
        self._allowed = build_dsl_allowlist()

    def build(self, text: str) -> Expr:
        try:
            parsed = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise DSLParseError(f"Invalid expression syntax: {text}") from exc

        expr = self._visit(parsed.body)
        if not isinstance(expr, Expr):
            raise DSLParseError("Expression must evaluate to an Expr object.")
        return expr

    def _visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Call):
            return self._visit_call(node)

        if isinstance(node, ast.Compare):
            return self._visit_compare(node)

        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._visit(node.operand)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool)) or node.value is None:
                return node.value
            raise DSLParseError(f"Unsupported literal: {node.value!r}")

        if isinstance(node, ast.Name):
            if node.id in self._allowed:
                raise DSLParseError(
                    f"Bare name {node.id!r} is not a column reference; "
                    f"use {node.id}(...) for operators."
                )
            if not _is_field_identifier(node.id):
                raise DSLParseError(f"Unsupported name: {node.id}")
            return col(node.id)

        raise DSLParseError(f"Unsupported syntax node: {type(node).__name__}")

    def _visit_call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in self._allowed:
                raise DSLParseError(f"Unsupported function: {name}")
            func = self._allowed[name]
        else:
            func = self._visit(node.func)

        args = [self._visit(arg) for arg in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise DSLParseError("Keyword-only **kwargs are not supported.")
            kwargs[kw.arg] = self._visit(kw.value)

        if not callable(func):
            raise DSLParseError("Call target is not callable.")
        return func(*args, **kwargs)

    def _visit_compare(self, node: ast.Compare) -> Expr:
        """单条比较 ``left op right``（不支持链式 ``a < b < c``）。"""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise DSLParseError(
                "Chained comparisons are not supported; use one comparison only."
            )
        left = self._visit(node.left)
        right = self._visit(node.comparators[0])
        op = node.ops[0]
        from api.cleaned_ops import make_cleaned_call_factory

        if isinstance(op, ast.Lt):
            return make_cleaned_call_factory("lt")(left, right)
        if isinstance(op, ast.LtE):
            return make_cleaned_call_factory("le")(left, right)
        if isinstance(op, ast.Eq):
            return make_cleaned_call_factory("eq")(left, right)
        if isinstance(op, ast.Gt):
            return make_cleaned_call_factory("gt")(left, right)
        if isinstance(op, ast.GtE):
            return make_cleaned_call_factory("ge")(left, right)
        if isinstance(op, ast.NotEq):
            return make_cleaned_call_factory("ne")(left, right)
        raise DSLParseError(f"Unsupported comparison: {type(op).__name__}")

    def _visit_binop(self, node: ast.BinOp) -> Any:
        left = self._visit(node.left)
        right = self._visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right

        raise DSLParseError(f"Unsupported binary operator: {type(node.op).__name__}")


def _is_field_identifier(name: str) -> bool:
    """挖掘侧 DSL 字段名：蛇形标识符（如 ``close``、``ret_price``）。"""
    if not name or name[0].isdigit():
        return False
    return all(c.isalnum() or c == "_" for c in name)


def parse_expr(text: str) -> Expr:
    """解析单条表达式字符串为 ``Expr``。"""
    return _ExprBuilder().build(text)


def parse_factor(
    text: str,
    *,
    name: str = "factor",
    freq: str = "1d",
    universe: str | None = None,
    description: str | None = None,
) -> Factor:
    """解析表达式并包成带元数据的 :class:`api.factor.Factor`。"""
    return Factor(
        name=name,
        expr=parse_expr(text),
        freq=freq,
        universe=universe,
        description=description,
    )

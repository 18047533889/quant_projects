"""R39 #67 —— DerivedFieldCompiler：把 ``derived_expression`` 编译成可执行表达式。

背景
    语义 catalog 允许声明 derived 字段（``derived_expression``），但执行链长期
    未落地，Planner 遇到就拒绝。本模块实现编译 + 求值：

    - :func:`parse_expression`：安全解析简单算术表达式（``+ - * /``、一元负号、
      括号、数字字面量、``dataset.column`` / ``column`` 引用），返回 AST；
    - :class:`DerivedFieldCompiler`：把 ``SemanticField.derived_expression``
      编译成 AST（plan 期 fail-fast）；
    - :func:`evaluate_expression`：在 Arrow Table 上求值（pyarrow.compute）。

安全性
    - 只接受白名单 token（数字 / 标识符 / ``+-*/()``），不执行任意代码；
    - 不支持 / 解析失败的表达式 → ``UnsupportedFeatureError``（typed，非 generic
      reject）；列引用在结果表里不唯一 → ``AmbiguousFieldError``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from data_access.core.exceptions import (
    AmbiguousFieldError,
    UnsupportedFeatureError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Ref:
    dataset: str | None
    column: str


@dataclass(frozen=True)
class Bin:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Any


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_NUMBER = r"(?:\d+\.?\d*|\.\d+)"
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_TOKEN_RE = re.compile(rf"\s*(?:(?P<num>{_NUMBER})|(?P<ident>{_IDENT})|(?P<op>[+\-*/()]))")


def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise UnsupportedFeatureError(
                f"derived_expression 含非法字符（位置 {pos}）: {expr!r}"
            )
        pos = m.end()
        if m.group("num") is not None:
            tokens.append(("num", m.group("num")))
        elif m.group("ident") is not None:
            tokens.append(("ident", m.group("ident")))
        else:
            tokens.append(("op", m.group("op")))
    tokens.append(("eof", ""))
    return tokens


class _Parser:
    """递归下降：expr := term (('+'|'-') term)*；term := factor (('*'|'/') factor)*。"""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._i = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._i]

    def _advance(self) -> tuple[str, str]:
        tok = self._tokens[self._i]
        self._i += 1
        return tok

    def _expect_op(self, op: str) -> None:
        kind, value = self._advance()
        if kind != "op" or value != op:
            raise UnsupportedFeatureError(
                f"derived_expression 语法错误：期望 '{op}'，收到 {value!r}"
            )

    def parse(self) -> Any:
        node = self._expr()
        kind, value = self._peek()
        if kind != "eof":
            raise UnsupportedFeatureError(
                f"derived_expression 语法错误：多余 token {value!r}"
            )
        return node

    def _expr(self) -> Any:
        node = self._term()
        while True:
            kind, value = self._peek()
            if kind == "op" and value in ("+", "-"):
                self._advance()
                node = Bin(value, node, self._term())
            else:
                return node

    def _term(self) -> Any:
        node = self._factor()
        while True:
            kind, value = self._peek()
            if kind == "op" and value in ("*", "/"):
                self._advance()
                node = Bin(value, node, self._factor())
            else:
                return node

    def _factor(self) -> Any:
        kind, value = self._peek()
        if kind == "op" and value == "-":
            self._advance()
            return Unary("-", self._factor())
        return self._primary()

    def _primary(self) -> Any:
        kind, value = self._peek()
        if kind == "num":
            self._advance()
            return Num(float(value))
        if kind == "ident":
            self._advance()
            if "." in value:
                ds, col = value.split(".", 1)
            else:
                ds, col = None, value
            if not col:
                raise UnsupportedFeatureError(
                    f"derived_expression 引用非法: {value!r}"
                )
            return Ref(ds, col)
        if kind == "op" and value == "(":
            self._advance()
            node = self._expr()
            self._expect_op(")")
            return node
        raise UnsupportedFeatureError(
            f"derived_expression 语法错误：意外的 {value!r}"
        )


def parse_expression(expr: str) -> Any:
    """安全解析 derived_expression → AST。失败抛 ``UnsupportedFeatureError``。"""
    if not expr or not str(expr).strip():
        raise UnsupportedFeatureError("derived_expression 为空")
    return _Parser(_tokenize(str(expr))).parse()


class DerivedFieldCompiler:
    """把 ``SemanticField.derived_expression`` 编译成可执行 AST（plan 期 fail-fast）。"""

    def __init__(self, field: Any) -> None:
        self.field = field
        self.logical_name = str(getattr(field, "logical_name", ""))
        expr = getattr(field, "derived_expression", None)
        if not expr:
            raise UnsupportedFeatureError(
                f"字段 '{self.logical_name}' 不是 derived 字段（缺 derived_expression）"
            )
        self.expression = str(expr)
        self.ast = parse_expression(self.expression)

    def describe(self) -> str:
        return f"{self.logical_name} := {self.expression}"


def _resolve_column(table: pa.Table, dataset: str | None, column: str) -> pa.ChunkedArray:
    """把 ``(dataset, column)`` 引用映射到结果表列。

    结果表列名是物理列名（read_joined 多表输出=physical_name）。若结果表里同名
    物理列出现多次（跨数据集重复列）→ ``AmbiguousFieldError``。
    """
    names = table.column_names
    matches = [c for c in names if c == column]
    if not matches:
        qual = f"{dataset}.{column}" if dataset else column
        raise ValidationError(
            f"derived 字段引用 {qual!r} 在结果表里找不到列（可用列: "
            f"{sorted(set(names))[:20]}）"
        )
    if len(matches) > 1:
        raise AmbiguousFieldError(
            f"derived 字段引用 {dataset}.{column} 在结果表里有多个同名物理列"
            f"（{len(matches)}），无法确定用哪一列"
        )
    return table.column(column)


def _evaluate_node(node: Any, table: pa.Table) -> Any:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Ref):
        return _resolve_column(table, node.dataset, node.column)
    if isinstance(node, Unary):
        operand = _evaluate_node(node.operand, table)
        if isinstance(operand, (int, float)):
            return -float(operand)
        return pc.negate(operand)
    if isinstance(node, Bin):
        left = _evaluate_node(node.left, table)
        right = _evaluate_node(node.right, table)
        if node.op == "+":
            return pc.add(left, right)
        if node.op == "-":
            return pc.subtract(left, right)
        if node.op == "*":
            return pc.multiply(left, right)
        if node.op == "/":
            return pc.divide(left, right)
        raise UnsupportedFeatureError(f"不支持的运算符: {node.op!r}")
    raise UnsupportedFeatureError(f"未知表达式节点: {node!r}")


def evaluate_expression(ast: Any, table: pa.Table) -> pa.Array:
    """在 Arrow Table 上求值 AST → 单列 ``pa.Array``/``ChunkedArray``。"""
    result = _evaluate_node(ast, table)
    if isinstance(result, (int, float)):
        # 整表达式退化成常量（如纯数字）→ 广播成行数一致的列。
        return pa.array([float(result)] * table.num_rows)
    if isinstance(result, pa.ChunkedArray):
        return result.combine_chunks()
    if isinstance(result, pa.Array):
        return result
    raise UnsupportedFeatureError(f"derived 求值返回未知类型: {type(result).__name__}")


def apply_derived_fields(
    table: pa.Table,
    derived_fields: Sequence[Any],
) -> pa.Table:
    """在结果表上追加全部 derived 字段列（列名=logical_name）。

    derived 列依赖的物理列必须已经出现在 ``table``（plan 阶段已把它们展开进
    扫描/join）。只做追加投影，不删除依赖列。
    """
    if not derived_fields:
        return table
    out = table
    for df in derived_fields:
        compiler = DerivedFieldCompiler(df)
        col = evaluate_expression(compiler.ast, out)
        out = out.append_column(df.logical_name, col)
    return out


__all__ = [
    "DerivedFieldCompiler",
    "parse_expression",
    "evaluate_expression",
    "apply_derived_fields",
]

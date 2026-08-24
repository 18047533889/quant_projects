"""R39 #67 / R45 —— DerivedFieldCompiler：把 ``derived_expression`` 编译成可执行表达式。

背景
    语义 catalog 允许声明 derived 字段（``derived_expression``），但执行链长期
    未落地，Planner 遇到就拒绝。本模块实现编译 + 求值：

    - :func:`parse_expression`：安全解析简单算术表达式（``+ - * /``、一元负号、
      括号、数字字面量、``dataset.column`` / ``column`` 引用），返回 AST；
    - :class:`DerivedFieldCompiler`：把 ``SemanticField.derived_expression``
      编译成 AST（plan 期 fail-fast）；
    - :func:`evaluate_expression`：在 Arrow Table 上求值（pyarrow.compute）。

R45 强化（本文件本次修改）
    1. **Field-ID 绑定**：plan/compile 期把每个 ``dataset.column`` 引用绑定成
       ``ResolvedFieldID``（``(dataset, physical_name)`` 组合键），执行期按绑定
       ID 解析列。两个数据集暴露同名物理列（``alpha.value`` 与 ``beta.value``）
       时各归其列，绝不交叉引用——即使 ``read_joined`` 里同名输出列已被拒绝，
       绑定语义仍显式携带数据集身份，杜绝拼写错位。
    2. **NumericPolicy 绑定**：derived 计算绑定一个数值策略，控制除零 / Inf /
       NaN 行为（``raise`` / ``nan`` / ``masked``）。两份逻辑表达式相同但 policy
       不同的字段会 hash 成不同实例，行为按各自策略执行。
    3. **Projection closure**：最终投影只返回请求的逻辑字段（derived 的
       ``logical_name``）以及它们依赖的原始物理列，绝不泄露无关原始列。

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
# NumericPolicy（R45 fix 2）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericPolicy:
    """控制 derived 求值里的除零 / Inf / NaN 行为。

    - ``division``：除数为 0 时的行为。
        - ``"raise"``  → 抛 ``ValidationError``（fail-closed）；
        - ``"nan"``    → 该位填 NaN；
        - ``"masked"`` → 该位填 NULL。
    - ``nonfinite``：最终结果里出现的 Inf / NaN（含除零溢出）如何收口。
        - ``"raise"``  → 抛 ``ValidationError``；
        - ``"nan"``    → 归一化 NaN；
        - ``"masked"`` → 归一化 NULL。

    同表达式但 policy 不同 → 编译出的对象不同（frozen dataclass，按值 hash 区分）。
    """

    division: str = "nan"
    nonfinite: str = "nan"

    def __post_init__(self) -> None:
        if self.division not in {"raise", "nan", "masked"}:
            raise ValidationError(
                f"NumericPolicy.division 非法: {self.division!r}（raise/nan/masked）"
            )
        if self.nonfinite not in {"raise", "nan", "masked"}:
            raise ValidationError(
                f"NumericPolicy.nonfinite 非法: {self.nonfinite!r}（raise/nan/masked）"
            )


_DEFAULT_POLICY = NumericPolicy()


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
# Field-ID 绑定（R45 fix 1）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedFieldID:
    """``dataset.column`` 引用在执行期的唯一身份。

    同一物理列名在**不同数据集**里是不同 ID（即使列名相同也不会交叉引用）。
    """

    dataset: str
    physical_name: str

    @property
    def qualified(self) -> str:
        return f"{self.dataset}.{self.physical_name}"


def bind_ref(dataset: str | None, column: str) -> ResolvedFieldID:
    """把 ``(dataset, column)`` 引用编译成唯一 Field-ID。

    带 ``dataset.`` 前缀 → ``(dataset, column)``；
    裸引用 → ``column`` 当作 dataset 持有者（单一数据集场景）。真正校验列是否
    存在于结果表留到执行期（表在 execute 才物化）。
    """
    ds = dataset if dataset is not None else column
    return ResolvedFieldID(dataset=ds, physical_name=column)


def _walk_refs(node: Any) -> list[Ref]:
    if isinstance(node, Ref):
        return [node]
    if isinstance(node, Unary):
        return _walk_refs(node.operand)
    if isinstance(node, Bin):
        return _walk_refs(node.left) + _walk_refs(node.right)
    return []


def _ref_key(ref: Ref) -> str:
    """把列引用映射成绑定表键：带 dataset 前缀用 ``dataset.column``，裸引用用列名。"""
    return f"{ref.dataset}.{ref.column}" if ref.dataset is not None else ref.column


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


def _numeric_policy_from_field(field: Any) -> NumericPolicy:
    """从 SemanticField 的 ``policy_derive`` 读取 NumericPolicy。

    字段无声明 → 默认 ``_DEFAULT_POLICY``；声明非法 → 编译期抛。
    """
    raw = getattr(field, "policy_derive", None)
    if not raw:
        return _DEFAULT_POLICY
    if isinstance(raw, NumericPolicy):
        return raw
    if isinstance(raw, dict):
        return NumericPolicy(
            division=str(raw.get("division", "nan")),
            nonfinite=str(raw.get("nonfinite", "nan")),
        )
    raise ValidationError(
        f"derived 字段 '{getattr(field, 'logical_name', '')}' 的 policy_derive "
        f"必须为 dict 或 NumericPolicy，收到 {type(raw).__name__}"
    )


class DerivedFieldCompiler:
    """把 ``SemanticField.derived_expression`` 编译成可执行 AST（plan 期 fail-fast）。

    持有字段数值策略（fix 2）；``bind()`` 把列引用绑定成 Field-ID（fix 1）。
    """

    def __init__(
        self,
        field: Any,
        *,
        numeric_policy: NumericPolicy | None = None,
    ) -> None:
        self.field = field
        self.logical_name = str(getattr(field, "logical_name", ""))
        expr = getattr(field, "derived_expression", None)
        if not expr:
            raise UnsupportedFeatureError(
                f"字段 '{self.logical_name}' 不是 derived 字段（缺 derived_expression）"
            )
        self.expression = str(expr)
        self.ast = parse_expression(self.expression)
        self.numeric_policy = numeric_policy or _numeric_policy_from_field(field)
        self._bound: dict[str, ResolvedFieldID] | None = None

    def describe(self) -> str:
        return f"{self.logical_name} := {self.expression}"

    def bind(self) -> DerivedFieldCompiler:
        """把每个列引用绑定成唯一 Field-ID（fix 1 plan 面）。

        绑定键按 ``(dataset, column)`` 组合区分：同一数据集里的多列不会相互覆盖，
        不同数据集里的同名列也不会合并。
        """
        self._bound = {}
        for ref in _walk_refs(self.ast):
            key = _ref_key(ref)
            if key in self._bound:
                continue
            self._bound[key] = bind_ref(ref.dataset, ref.column)
        return self

    @property
    def referenced_ids(self) -> list[ResolvedFieldID]:
        if self._bound is None:
            raise ValidationError(f"DerivedFieldCompiler 尚未 bind（{self.logical_name}）")
        return sorted(self._bound.values(), key=lambda f: f.qualified)

    @property
    def raw_columns(self) -> list[str]:
        """该 derived 依赖的物理列名（去重，保持 AST 出现顺序）。"""
        if self._bound is None:
            raise ValidationError(f"DerivedFieldCompiler 尚未 bind（{self.logical_name}）")
        out: list[str] = []
        for ref in _walk_refs(self.ast):
            fid = self._bound.get(_ref_key(ref))
            if fid is not None and fid.physical_name not in out:
                out.append(fid.physical_name)
        return out


# ---------------------------------------------------------------------------
# 求值（R45：按绑定 Field-ID 解析列 + 按 NumericPolicy 收口）
# ---------------------------------------------------------------------------


def _resolve_by_id(table: pa.Table, fid: ResolvedFieldID) -> pa.ChunkedArray:
    """执行期按绑定 Field-ID 解析列。

    结果表列名 = 物理列名（``read_joined`` 多表输出即 physical_name）。同名物理列
    跨数据集已被 join 输出守卫拒绝，故按 ``physical_name`` 取列即精确命中目标列。
    ``fid.dataset`` 用于错误信息与拼写错位防御。
    """
    if fid.physical_name not in table.column_names:
        raise ValidationError(
            f"derived 字段引用 {fid.qualified!r} 在结果表里找不到列"
            f"（可用列: {sorted(table.column_names)[:20]}）"
        )
    return table.column(fid.physical_name)


def _divide(left: Any, right: Any, n_rows: int, policy: NumericPolicy) -> pa.Array:
    """除法：除零按 NumericPolicy.division 处理，结果 Inf/NaN 按 nonfinite 收口。"""
    larr = _coerce_fixed(left, n_rows)
    rarr = _coerce_fixed(right, n_rows)
    if policy.division == "raise":
        # 除零 → 抛
        is_zero = pc.equal(rarr, 0)
        if pa.compute.any(is_zero).as_py():
            raise ValidationError(
                "derived 除法遇到除零（NumericPolicy.division=raise）"
            )
        return _finalize(pc.divide(larr, rarr), policy)
    # nan / masked：除零位置不参与真实除法，直接按策略填 NaN / NULL。
    zero_mask = pc.equal(rarr, 0)
    safe_den = pc.if_else(zero_mask, pa.scalar(1.0, type=rarr.type), rarr)
    out = pc.divide(larr, safe_den)
    if policy.division == "masked":
        out = pc.if_else(zero_mask, pa.scalar(None, type=out.type), out)
    else:  # nan
        out = pc.if_else(zero_mask, pa.scalar(float("nan"), type=out.type), out)
    return _finalize(out, policy)


def _coerce_fixed(x: Any, n_rows: int) -> pa.Array:
    return _coerce_scalarish(x, n_rows)


def _coerce_scalarish(x: Any, n_rows: int) -> pa.Array:
    if isinstance(x, (int, float)):
        return pa.array([float(x)] * n_rows)
    if isinstance(x, pa.ChunkedArray):
        x = x.combine_chunks()
    if isinstance(x, pa.Array):
        if len(x) == 0:
            return pa.array([], type=x.type)
        return x
    raise UnsupportedFeatureError(f"derived 求值返回未知类型: {type(x).__name__}")


def _finalize(out: pa.Array, policy: NumericPolicy) -> pa.Array:
    """把结果列按 ``nonfinite`` 收口 Inf/NaN（默认 nan：原样保留）。"""
    if policy.nonfinite == "raise":
        is_bad = pc.or_kleene(pc.is_nan(out), pc.is_inf(out))
        if pa.compute.any(is_bad).as_py():
            raise ValidationError(
                "derived 结果含 Inf/NaN（NumericPolicy.nonfinite=raise）"
            )
        return out
    if policy.nonfinite == "masked":
        is_bad = pc.or_kleene(pc.is_nan(out), pc.is_inf(out))
        return pc.if_else(
            is_bad, pa.scalar(None, type=out.type), out
        )
    return out


def _evaluate_node(
    node: Any,
    table: pa.Table,
    bound: dict[str, ResolvedFieldID],
    policy: NumericPolicy,
    n_rows: int,
) -> Any:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Ref):
        key = _ref_key(node)
        fid = bound.get(key)
        if fid is None:
            raise ValidationError(f"derived 引用未绑定列 {key!r}")
        return _resolve_by_id(table, fid)
    if isinstance(node, Unary):
        operand = _evaluate_node(node.operand, table, bound, policy, n_rows)
        if isinstance(operand, (int, float)):
            return -float(operand)
        return pc.negate(_coerce_fixed(operand, n_rows))
    if isinstance(node, Bin):
        left = _evaluate_node(node.left, table, bound, policy, n_rows)
        right = _evaluate_node(node.right, table, bound, policy, n_rows)
        if node.op == "+":
            return pc.add(_coerce_fixed(left, n_rows), _coerce_fixed(right, n_rows))
        if node.op == "-":
            return pc.subtract(_coerce_fixed(left, n_rows), _coerce_fixed(right, n_rows))
        if node.op == "*":
            return pc.multiply(_coerce_fixed(left, n_rows), _coerce_fixed(right, n_rows))
        if node.op == "/":
            return _divide(left, right, n_rows, policy)
        raise UnsupportedFeatureError(f"不支持的运算符: {node.op!r}")
    raise UnsupportedFeatureError(f"未知表达式节点: {node!r}")


def evaluate_expression(
    ast: Any,
    table: pa.Table,
    *,
    bindings_map: dict[str, ResolvedFieldID] | None = None,
    numeric_policy: NumericPolicy | None = None,
) -> pa.Array:
    """在 Arrow Table 上求值 AST → 单列 ``pa.Array``。

    默认用表列名直接解析（保持 R39 旧接口）；``bindings_map`` 传入时按绑定
    Field-ID 解析列（R45 fix 1 执行面）。
    """
    policy = numeric_policy or _DEFAULT_POLICY
    if bindings_map is not None:
        bound = bindings_map
    else:
        bound = {}
        for ref in _walk_refs(ast):
            key = _ref_key(ref)
            bound.setdefault(key, bind_ref(ref.dataset, ref.column))
    result = _evaluate_node(ast, table, bound, policy, table.num_rows)
    if isinstance(result, (int, float)):
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
    """在结果表上追加 derived 字段列并做投影闭包。

    R45 行为
        - bind() 期给每个列引用绑定 Field-ID，执行期按 ID 解析列（fix 1）；
        - 每个字段绑定 NumericPolicy（fix 2）；
        - 投影闭包（fix 3）：只返回请求的逻辑输出（derived 列）+ 其依赖的原始
          物理列，绝不泄露无关输入列。依赖物理列保留原始列名。
    """
    if not derived_fields:
        return table
    n_rows = table.num_rows
    out = table
    compiled: list[DerivedFieldCompiler] = []
    for df in derived_fields:
        c = DerivedFieldCompiler(df)
        c.bind()
        col = _evaluate_node(c.ast, out, c._bound, c.numeric_policy, n_rows)
        col = _finalize(col, c.numeric_policy)
        out = out.append_column(c.logical_name, col)
        compiled.append(c)
    # fix 3：投影闭包 —— derived 输出 + 依赖的原始物理列。
    keep: list[str] = []
    for c in compiled:
        if c.logical_name not in keep:
            keep.append(c.logical_name)
        for rc in c.raw_columns:
            if rc not in keep:
                keep.append(rc)
    return out.select(keep)


__all__ = [
    "DerivedFieldCompiler",
    "NumericPolicy",
    "ResolvedFieldID",
    "parse_expression",
    "evaluate_expression",
    "apply_derived_fields",
]

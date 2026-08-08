"""
data_access.read.predicate_ast —— 通用过滤表达式（Filter AST）

职责
    1. 提供结构化过滤的表达式树：Eq/Ne/Lt/Le/Gt/Ge/Between/In/NotIn/IsNull/
       IsNotNull 与 And/Or/Not
    2. 把调用方友好的输入（``filters={"col": v}`` / ``{"col": [..]}`` /
       ``{"col": {"gt": .., "lte": ..}}`` / ``Filter`` 实例）归一化为 AST
    3. 编译成 DuckDB WHERE 片段 / Polars 表达式——语义共享，一份 AST 多后端

设计要点
    1. 永远用 DuckDB 参数绑定（? 占位符），不把用户值拼进 SQL。
    2. 列名来自调用方，统一走 ``_quote_ident`` 防注入；strict 场景下
       ``allowed_columns`` 只放行 registry schema 声明的列。
    3. 与 ``Predicate``（time_range/instrument_filter/hive_filters）互补：
       AST 处理任意列的等值/范围/集合/空值过滤。

非职责
    不做类型推断（DuckDB/Polars 执行时报）；不负责 SELECT 投影（store.py）。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class Filter:
    """过滤表达式基类。"""

    def __and__(self, other: "Filter") -> "And":
        return And([self, other])

    def __or__(self, other: "Filter") -> "Or":
        return Or([self, other])

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class ColumnFilter(Filter):
    column: str


@dataclass(frozen=True)
class Eq(ColumnFilter):
    value: Any


@dataclass(frozen=True)
class Ne(ColumnFilter):
    value: Any


@dataclass(frozen=True)
class Lt(ColumnFilter):
    value: Any


@dataclass(frozen=True)
class Le(ColumnFilter):
    value: Any


@dataclass(frozen=True)
class Gt(ColumnFilter):
    value: Any


@dataclass(frozen=True)
class Ge(ColumnFilter):
    value: Any


@dataclass(frozen=True)
class Between(ColumnFilter):
    lower: Any
    upper: Any
    lower_inclusive: bool = True
    upper_inclusive: bool = True


@dataclass(frozen=True)
class In(ColumnFilter):
    values: tuple[Any, ...]


@dataclass(frozen=True)
class NotIn(ColumnFilter):
    values: tuple[Any, ...]


@dataclass(frozen=True)
class IsNull(ColumnFilter):
    pass


@dataclass(frozen=True)
class IsNotNull(ColumnFilter):
    pass


@dataclass(frozen=True)
class And(Filter):
    children: tuple[Filter, ...]


@dataclass(frozen=True)
class Or(Filter):
    children: tuple[Filter, ...]


@dataclass(frozen=True)
class Not(Filter):
    child: Filter


# ---------------------------------------------------------------------------
# 规范化序列化 / 指纹（缓存 key、审计用）—— 不依赖 repr()
# ---------------------------------------------------------------------------


def _canon_value(value: Any) -> Any:
    """把过滤值规范化为可 JSON 序列化的稳定表示（日期→iso，numpy→标量）。"""
    import datetime as _dt

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _dt.datetime):
        return f"dt:{value.isoformat()}"
    if isinstance(value, _dt.date):
        return f"d:{value.isoformat()}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canon_value(v) for v in value]
    try:
        import numpy as _np  # 本地 import：只在非原生标量出现时触发

        if isinstance(value, _np.generic):
            return _canon_value(value.item())
    except ImportError:
        pass
    return str(value)


def canonical_filter_ast(filters: Any) -> dict[str, Any]:
    """把 Filter AST 序列化成规范化 dict（不排序版本；And/Or 子节点排序）。

    供 ``canonical_filter_hash`` 用。``filters`` 接受 parse_filters 的所有
    输入形态（Filter / dict / list / None）。
    """
    node = parse_filters(filters) if filters is not None else None
    if node is None:
        return {"kind": "all"}
    import json

    def _ser(f: Filter) -> dict[str, Any]:
        cls = type(f).__name__
        if isinstance(f, ColumnFilter):
            base: dict[str, Any] = {"node": cls, "column": f.column}
            if cls in ("Eq", "Ne", "Lt", "Le", "Gt", "Ge"):
                base["value"] = _canon_value(f.value)
            elif cls in ("In", "NotIn"):
                base["values"] = sorted(
                    (_canon_value(v) for v in f.values), key=str
                )
            elif cls == "Between":
                base["lower"] = _canon_value(f.lower)
                base["upper"] = _canon_value(f.upper)
                base["lower_inclusive"] = f.lower_inclusive
                base["upper_inclusive"] = f.upper_inclusive
            return base
        if isinstance(f, (And, Or)):
            # #P1-final closure 15：And/Or 子节点按**完整 canonical JSON** 排序——
            # 旧 key 只按 node|column，同列不同值（Eq(col,1) vs Eq(col,2)）的 tie
            # 保留输入顺序，交换书写顺序仍会 miss cache。用完整序列化排序后，
            # 语义等价的过滤（列顺序/And 顺序/同列多值顺序不同）得到同一 hash。
            return {
                "node": cls,
                "children": sorted(
                    (_ser(c) for c in f.children),
                    key=lambda d: json.dumps(
                        d, sort_keys=True, separators=(",", ":"), default=str
                    ),
                ),
            }
        if isinstance(f, Not):
            return {"node": "Not", "child": _ser(f.child)}
        return {"node": cls}

    return _ser(node)


def canonical_filter_hash(filters: Any) -> str:
    """Filter AST 的规范化指纹（sha256[:24]）。None/空 → "no_filter"。

    两个语义等价的过滤（列顺序/And 顺序不同）得到同一个 hash——缓存 key 不因
    调用方书写顺序不同而 miss。
    """
    if filters is None:
        return "no_filter"
    import hashlib
    import json

    payload = json.dumps(
        canonical_filter_ast(filters),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------

_COMPARISON_OPS = {
    "eq": Eq,
    "ne": Ne,
    "lt": Lt,
    "le": Le,
    "gt": Gt,
    "ge": Ge,
    "lte": Le,
    "gte": Ge,
    "==": Eq,
    "!=": Ne,
    "<": Lt,
    "<=": Le,
    ">": Gt,
    ">=": Ge,
}


def _build_column_filter(column: str, value: Any) -> Filter:
    if isinstance(value, Filter):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return In(column, tuple(value))
    if isinstance(value, dict):
        # {op: value} 或 {"between": [lo, hi]} / {"in": [...]}
        clauses: list[Filter] = []
        for op, arg in value.items():
            op_s = str(op).strip().lower()
            if op_s in {"between"}:
                if not isinstance(arg, (list, tuple)) or len(arg) != 2:
                    raise ValueError(
                        f"column '{column}': between 需要 [lower, upper]，收到 {arg!r}"
                    )
                clauses.append(Between(column, arg[0], arg[1]))
            elif op_s in {"in", "notin", "not_in"}:
                values = list(arg) if isinstance(arg, (list, tuple, set)) else [arg]
                cls = NotIn if op_s in {"notin", "not_in"} else In
                clauses.append(cls(column, tuple(values)))
            elif op_s in {"isnull", "is_null", "null"}:
                # #P0-6 boolean-aware：isnull:true → IS NULL，isnull:false → IS NOT NULL。
                # 之前识别 operator 却忽略布尔值，isnull:false 仍生成 IS NULL（错语义）。
                # None（YAML null）按「请求空值检查」处理 = True（向后兼容既有写法）。
                if arg is None:
                    arg = True
                if not isinstance(arg, bool):
                    raise ValueError(
                        f"column '{column}': {op_s} 的值必须是布尔 true/false，收到 {arg!r}"
                    )
                clauses.append(IsNull(column) if arg else IsNotNull(column))
            elif op_s in {"isnotnull", "is_not_null", "notnull", "not_null"}:
                if arg is None:
                    arg = True
                if not isinstance(arg, bool):
                    raise ValueError(
                        f"column '{column}': {op_s} 的值必须是布尔 true/false，收到 {arg!r}"
                    )
                clauses.append(IsNotNull(column) if arg else IsNull(column))
            elif op_s in _COMPARISON_OPS:
                clauses.append(_COMPARISON_OPS[op_s](column, arg))
            else:
                raise ValueError(
                    f"column '{column}': 未知过滤操作符 {op!r}；"
                    "支持 eq/ne/lt/le/gt/ge/between/in/notin/isnull/isnotnull"
                )
        if not clauses:
            raise ValueError(f"column '{column}': 空过滤 dict")
        return And(tuple(clauses)) if len(clauses) > 1 else clauses[0]
    # 标量 → 等值
    return Eq(column, value)


def parse_filters(filters: Any) -> Filter | None:
    """把任意调用方输入归一化为 Filter AST；None/空 返回 None。

    支持：
        - ``Filter`` 实例（原样返回）
        - ``{col: scalar}``      → Eq
        - ``{col: [..]}``        → In
        - ``{col: {op: v, ...}}``→ And(...)（between/in/isnull/比较符）
        - ``[filter1, filter2]`` → And(...)
    """
    if filters is None:
        return None
    if isinstance(filters, Filter):
        return filters
    if isinstance(filters, dict):
        clauses: list[Filter] = []
        for col, value in filters.items():
            clauses.append(_build_column_filter(str(col), value))
        if not clauses:
            return None
        return And(tuple(clauses)) if len(clauses) > 1 else clauses[0]
    if isinstance(filters, (list, tuple)):
        clauses = [parse_filters(f) for f in filters]
        clauses = [c for c in clauses if c is not None]
        if not clauses:
            return None
        return And(tuple(clauses)) if len(clauses) > 1 else clauses[0]
    raise ValueError(
        f"filters 必须是 Filter/dict/list，收到 {type(filters).__name__}"
    )


def filter_columns(f: Filter | None) -> set[str]:
    """收集 AST 里引用到的所有列名（用于 strict 模式 schema 校验）。"""
    if f is None:
        return set()
    if isinstance(f, ColumnFilter):
        return {f.column}
    if isinstance(f, And):
        out: set[str] = set()
        for c in f.children:
            out |= filter_columns(c)
        return out
    if isinstance(f, Or):
        out = set()
        for c in f.children:
            out |= filter_columns(c)
        return out
    if isinstance(f, Not):
        return filter_columns(f.child)
    return set()


def filter_restricts_column(f: Filter | None, column: str) -> bool:
    """保守证明 filter 是否把 ``column`` 限制到某个具体取值/区间子集。

    required_filters / allowed_filter_values 用（#14/#15）：只靠「列名出现在
    条件里」证明不了维度被真正限定——``timeframe != 'quarterly'``、
    ``timeframe IS NOT NULL``、``timeframe='quarterly' OR price > 0`` 都能让
    结果越出所需维度。

    规则（充分条件，宁严勿松）：
        - Eq(col, v) / In(col, [v...]) / Between(col, lo, hi) → True（正向限制）
        - And：任一子句限制即成立（AND 链中任意一支都能保证子集）
        - Or：所有子句都限制才成立（未限制分支会让整条 OR 通过越界行）
        - Ne / NotIn / IsNull / IsNotNull / Not / 单边范围 → False（不能证明）
        - 其它列的子句 → 不影响本列
    """
    return filter_constraint_status(f, column) == "positive"


def filter_constraint_status(f: Filter | None, column: str) -> str:
    """报告 ``column`` 在 filter 里的约束强度：absent / positive / weak。

    - ``absent``   ：列未出现在 AST 中（没有施加任何约束）
    - ``positive`` ：存在等值/包含/区间字面约束，``filter_column_values`` 可取到
      被允许的值集合（结果 ⊆ 该集合可验证）
    - ``weak``     ：列被提及但无法证明子集（Ne/NotIn/IsNull/IsNotNull/Not/
      OR 部分分支）——strict 下禁止作为契约维度 gate 的依据
    """
    if f is None:
        return "absent"
    if isinstance(f, ColumnFilter):
        if f.column != column:
            return "absent"
        if isinstance(f, (Eq, In, Between)):
            return "positive"
        return "weak"
    if isinstance(f, (And, Or)):
        statuses = [filter_constraint_status(c, column) for c in f.children]
        if isinstance(f, And):
            # AND：任一分支 positive 即成立
            if "positive" in statuses:
                return "positive"
            if any(s != "absent" for s in statuses):
                return "weak"
            return "absent"
        # OR：必须所有分支都 positive，否则未限制分支可绕过
        if all(s == "positive" for s in statuses):
            return "positive"
        if any(s != "absent" for s in statuses):
            return "weak"
        return "absent"
    if isinstance(f, Not):
        return "weak" if filter_constraint_status(f.child, column) != "absent" else "absent"
    return "absent"


def filter_column_values(f: Filter | None) -> dict[str, set[Any]]:
    """收集 AST 里每个列的「候选取值」集合（#9 allowed_filter_values 值校验用）。

    只对等值/包含（Eq/In）和 Between 端点提取字面值；范围/空值过滤不产生
    强候选（返回空集合，调用方跳过该校验）。
    """
    if f is None:
        return {}
    out: dict[str, set[Any]] = {}

    def _add(col: str, values: Sequence[Any]) -> None:
        vals = out.setdefault(col, set())
        for v in values:
            if v is not None:
                vals.add(v)

    if isinstance(f, Eq):
        _add(f.column, [f.value])
    elif isinstance(f, In):
        _add(f.column, list(f.values))
    elif isinstance(f, Between):
        _add(f.column, [f.lower, f.upper])
    elif isinstance(f, (And, Or)):
        for c in f.children:
            for col, vals in filter_column_values(c).items():
                _add(col, vals)
    elif isinstance(f, Not):
        for col, vals in filter_column_values(f.child).items():
            _add(col, vals)
    return out


# ---------------------------------------------------------------------------
# DuckDB 编译器
# ---------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def compile_filter_duckdb(
    f: Filter | None,
    *,
    allowed_columns: set[str] | None = None,
) -> tuple[str, list[Any]]:
    """把 Filter AST 编译成 (where_clause, params)。

    where_clause 不含 "WHERE" 前缀；空 AST 返回 ("", [])。
    ``allowed_columns`` 提供时，未声明的列直接抛 ValueError（strict 治理）。
    """
    if f is None:
        return "", []
    _validate_columns(f, allowed_columns=allowed_columns)
    sql, params = _compile_node(f)
    return sql, params


def _validate_columns(f: Filter, *, allowed_columns: set[str] | None) -> None:
    if allowed_columns is None:
        return
    for col in filter_columns(f):
        if col not in allowed_columns:
            raise ValueError(
                f"过滤列 '{col}' 不在数据集 schema 声明中（允许: "
                f"{sorted(allowed_columns) or '无'}）。请先登记 schema 或去掉该过滤。"
            )


def _compile_node(f: Filter) -> tuple[str, list[Any]]:
    if isinstance(f, Eq):
        return f"{_quote_ident(f.column)} = ?", [f.value]
    if isinstance(f, Ne):
        return f"{_quote_ident(f.column)} != ?", [f.value]
    if isinstance(f, Lt):
        return f"{_quote_ident(f.column)} < ?", [f.value]
    if isinstance(f, Le):
        return f"{_quote_ident(f.column)} <= ?", [f.value]
    if isinstance(f, Gt):
        return f"{_quote_ident(f.column)} > ?", [f.value]
    if isinstance(f, Ge):
        return f"{_quote_ident(f.column)} >= ?", [f.value]
    if isinstance(f, Between):
        lo_op = ">=" if f.lower_inclusive else ">"
        hi_op = "<=" if f.upper_inclusive else "<"
        return (
            f"{_quote_ident(f.column)} {lo_op} ? AND {_quote_ident(f.column)} {hi_op} ?",
            [f.lower, f.upper],
        )
    if isinstance(f, In):
        return f"{_quote_ident(f.column)} IN ?", [list(f.values)]
    if isinstance(f, NotIn):
        return f"{_quote_ident(f.column)} NOT IN ?", [list(f.values)]
    if isinstance(f, IsNull):
        return f"{_quote_ident(f.column)} IS NULL", []
    if isinstance(f, IsNotNull):
        return f"{_quote_ident(f.column)} IS NOT NULL", []
    if isinstance(f, And):
        return _compile_logical(f.children, "AND")
    if isinstance(f, Or):
        return _compile_logical(f.children, "OR")
    if isinstance(f, Not):
        inner_sql, inner_params = _compile_node(f.child)
        if not inner_sql:
            return "", []
        return f"NOT ({inner_sql})", inner_params
    raise ValueError(f"未知 Filter 节点: {type(f).__name__}")


def _compile_logical(children: Sequence[Filter], op: str) -> tuple[str, list[Any]]:
    parts: list[str] = []
    params: list[Any] = []
    for child in children:
        sql, child_params = _compile_node(child)
        if not sql:
            continue
        parts.append(sql)
        params.extend(child_params)
    if not parts:
        return "", []
    if len(parts) == 1:
        return parts[0], params
    return f"({f' {op} '.join(f'({p})' for p in parts)})", params


# ---------------------------------------------------------------------------
# PyArrow 编译器（pyarrow.compute 表达式；arrow/feather 引擎用）
# ---------------------------------------------------------------------------


def compile_filter_arrow(f: Filter | None, *, pc: Any = None):
    """把 Filter AST 编译成 pyarrow.compute 表达式。

    ``pc`` 不传时惰性 import；返回可传给 ``table.filter(expr)`` 的表达式。
    """
    if f is None:
        return None
    if pc is None:
        import pyarrow.compute as _pc

        pc = _pc
    import pyarrow as _pa

    def field(name: str):
        return pc.field(name)

    def scalar(value: Any):
        return _pa.scalar(value)

    def value_set(values) -> Any:
        return _pa.array(list(values))

    if isinstance(f, Eq):
        return pc.equal(field(f.column), scalar(f.value))
    if isinstance(f, Ne):
        return pc.not_equal(field(f.column), scalar(f.value))
    if isinstance(f, Lt):
        return pc.less(field(f.column), scalar(f.value))
    if isinstance(f, Le):
        return pc.less_equal(field(f.column), scalar(f.value))
    if isinstance(f, Gt):
        return pc.greater(field(f.column), scalar(f.value))
    if isinstance(f, Ge):
        return pc.greater_equal(field(f.column), scalar(f.value))
    if isinstance(f, Between):
        lo = pc.greater_equal(field(f.column), scalar(f.lower)) if f.lower_inclusive else pc.greater(field(f.column), scalar(f.lower))
        hi = pc.less_equal(field(f.column), scalar(f.upper)) if f.upper_inclusive else pc.less(field(f.column), scalar(f.upper))
        return pc.and_(lo, hi)
    if isinstance(f, In):
        return pc.is_in(field(f.column), value_set(f.values))
    if isinstance(f, NotIn):
        return pc.invert(pc.is_in(field(f.column), value_set(f.values)))
    if isinstance(f, IsNull):
        return pc.is_null(field(f.column))
    if isinstance(f, IsNotNull):
        return pc.is_valid(field(f.column))
    if isinstance(f, And):
        expr = None
        for child in f.children:
            sub = compile_filter_arrow(child, pc=pc)
            expr = sub if expr is None else pc.and_kleene(expr, sub)
        return expr
    if isinstance(f, Or):
        expr = None
        for child in f.children:
            sub = compile_filter_arrow(child, pc=pc)
            expr = sub if expr is None else pc.or_kleene(expr, sub)
        return expr
    if isinstance(f, Not):
        return pc.invert(compile_filter_arrow(f.child, pc=pc))
    raise ValueError(f"未知 Filter 节点: {type(f).__name__}")


# ---------------------------------------------------------------------------
# Polars 编译器
# ---------------------------------------------------------------------------


def compile_filter_polars(f: Filter | None, *, pl: Any):
    """把 Filter AST 编译成 Polars 表达式；用于 scan_polars 的 filter。"""
    if f is None:
        return None
    return _compile_polars_node(f, pl)


def _compile_polars_node(f: Filter, pl: Any):
    col = lambda c: pl.col(c)  # noqa: E731
    if isinstance(f, Eq):
        return col(f.column) == f.value
    if isinstance(f, Ne):
        return col(f.column) != f.value
    if isinstance(f, Lt):
        return col(f.column) < f.value
    if isinstance(f, Le):
        return col(f.column) <= f.value
    if isinstance(f, Gt):
        return col(f.column) > f.value
    if isinstance(f, Ge):
        return col(f.column) >= f.value
    if isinstance(f, Between):
        expr = col(f.column) >= f.lower if f.lower_inclusive else col(f.column) > f.lower
        hi = col(f.column) <= f.upper if f.upper_inclusive else col(f.column) < f.upper
        return expr & hi
    if isinstance(f, In):
        return col(f.column).is_in(list(f.values))
    if isinstance(f, NotIn):
        return ~col(f.column).is_in(list(f.values))
    if isinstance(f, IsNull):
        return col(f.column).is_null()
    if isinstance(f, IsNotNull):
        return col(f.column).is_not_null()
    if isinstance(f, And):
        expr = None
        for child in f.children:
            sub = _compile_polars_node(child, pl)
            expr = sub if expr is None else (expr & sub)
        return expr
    if isinstance(f, Or):
        expr = None
        for child in f.children:
            sub = _compile_polars_node(child, pl)
            expr = sub if expr is None else (expr | sub)
        return expr
    if isinstance(f, Not):
        return ~_compile_polars_node(f.child, pl)
    raise ValueError(f"未知 Filter 节点: {type(f).__name__}")

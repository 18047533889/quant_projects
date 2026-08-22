"""
data_access.predicate —— 结构化谓词生成 SQL

职责：
    把 read API 里的 time_range / instrument_filter 等结构化参数翻译成
    DuckDB 可执行的 WHERE 片段 + 参数绑定列表。

设计要点：
    1. 永远用 DuckDB 的参数绑定（? 占位符），不把用户值拼进 SQL
    2. 只开放有限几种谓词，不提供 where_sql 原始字符串入口（PR1 决策）
    3. 时间列/标的列名来自 registry，不是调用方传入 —— 避免列名注入

非职责：
    不做参数值的类型校验（pandas/DuckDB 会在执行时报）。
    不负责 SELECT 的列投影（那是 store.py 在 SQL 拼接时的事）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from data_access.core.exceptions import ValidationError


def ensure_sequence_arg(value: Any, *, name: str) -> Any:
    """#P0-5 拒绝 str/bytes 冒充 Sequence（``instrument_filter="AAPL"`` 会被逐字符
    展开成 A/A/P/L）。所有 sequence 型 API 边界统一用它：
    instrument_filter / columns / partition_by / upsert_on / order_by / factor_ids。
    """
    if isinstance(value, (str, bytes)):
        raise ValidationError(
            f"{name} 必须是序列（list/tuple/set），收到 str/bytes {value!r}。"
            "字符串会被误当成字符序列逐项过滤，是典型的 silent semantic inversion。"
        )
    return value


def strict_sequence(
    value: Any,
    *,
    name: str,
    element_type: type = str,
    allow_none: bool = True,
    allow_empty: bool = True,
    ordered: bool = False,
) -> tuple | None:
    """#P0-C12 严格序列解析（DataRequest / SemanticField 共用）。

    拒绝裸 str/bytes（会被误当字符序列逐项拆成 c/l/o/s/e，是典型的 silent
    semantic inversion）；非序列容器 → ValidationError；元素类型不对 → 立即
    ValidationError（不再静默 str() 化非法对象 / 返回空 tuple 丢语义）。

    #7 拆规则（set/dict/generator 之前会被 ``tuple(value)`` 吞掉）：
        - ``ordered=True``（fields / order_by / revision_order / primary_key，
          顺序是语义）：只接受 list/tuple；set/frozenset/dict/generator 拒绝。
        - ``ordered=False``（instruments 等成员语义）：set/frozenset 接受，但转成
          deterministic canonical 排序（repr key），不再依赖 set 迭代序；
          dict/generator 拒绝。

    - None → None（``allow_none=True``）或 ()（``allow_none=False``）
    - str/bytes → ValidationError
    - 其余 → tuple，逐元素校验 ``element_type``
    """
    if value is None:
        return None if allow_none else ()
    if isinstance(value, (str, bytes)):
        raise ValidationError(
            f"{name} 必须是序列（list/tuple/set），收到 str/bytes {value!r}。"
            "裸字符串会被误当成字符序列逐项拆开。"
        )
    if ordered:
        if not isinstance(value, (list, tuple)):
            raise ValidationError(
                f"{name} 必须是 list/tuple（顺序是语义），"
                f"收到 {type(value).__name__} {value!r}；set/dict/generator 会丢顺序"
            )
        items = value
    elif isinstance(value, dict):
        raise ValidationError(
            f"{name} 不接受 mapping/dict（会退化成键列表），收到 {value!r}"
        )
    elif isinstance(value, (set, frozenset)):
        # 成员语义：set 可以，但转 deterministic canonical order（repr key）。
        items = sorted(value, key=repr)
    elif isinstance(value, collections.abc.Iterator):
        # #收官轮 P0（Integration）：generator/iterator 是一次性迭代器，``tuple()``
        # 会静默消费掉——docstring 已声明「generator 拒绝」，实现必须兑现，
        # 否则 adapter 层 ``instrument_filter=(x for x in ...)`` 会悄悄被吞掉。
        raise ValidationError(
            f"{name} 不接受 generator/iterator（一次性迭代器，会被静默消费），"
            f"收到 {type(value).__name__} {value!r}"
        )
    else:
        try:
            items = tuple(value)
        except TypeError:
            raise ValidationError(
                f"{name} 必须是序列（list/tuple/set），收到 {type(value).__name__} {value!r}"
            )
    if not allow_empty and not items:
        raise ValidationError(f"{name} 不能为空序列")
    for item in items:
        if not isinstance(item, element_type):
            raise ValidationError(
                f"{name} 的元素必须是 {element_type.__name__}，"
                f"收到 {type(item).__name__} {item!r}"
            )
    return tuple(items)


def _normalize_time_range(value: Any) -> Any:
    """#P0-4 ``(None, None)`` → None（两个端点都空 = 无时间约束，不能生成空 WHERE）。"""
    if isinstance(value, tuple) and len(value) == 2:
        if value[0] is None and value[1] is None:
            return None
    if isinstance(value, list) and len(value) == 2:
        if value[0] is None and value[1] is None:
            return None
    return value


def _normalize_sequence(value: Any, *, name: str) -> Any:
    """序列型参数：None 保持 None；str/bytes 拒绝；list/tuple/set/frozenset 原样。"""
    if value is None:
        return None
    ensure_sequence_arg(value, name=name)
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    raise ValidationError(
        f"{name} 必须是序列（list/tuple/set），收到 {type(value).__name__}"
    )


@dataclass
class Predicate:
    """结构化谓词。None 表示该维度不过滤。

    #P0-2/#P0-3 空集合语义（silent semantic inversion 修复）：
        - ``instrument_filter=None``   → 不限制标的
        - ``instrument_filter=[]``     → **空股票池 → WHERE FALSE**（不是全市场！）
        - ``instrument_filter=["A"]``  → IN (...)
        空 hive filter ``{"year": []}`` 同理 → WHERE FALSE。
    """

    time_range: tuple[Any, Any] | None = None           # (start, end)，闭区间
    instrument_filter: Sequence[str] | None = None      # 标的白名单
    hive_filters: dict[str, Sequence[Any]] | None = None  # hive 分区列 IN 过滤
    filters: "Filter | None" = None                     # 通用 Filter AST（任意列等值/范围/集合/空值）
    extra: list[dict] | None = None
    time_column_is_timestamp: bool = False              # 时间列为 timestamp 时做 end-of-day 展开
    time_lower_exclusive: bool = False                  # start 用严格 >（PIT seed 分支）
    time_upper_exclusive: bool = False                  # end 用严格 <（PIT seed 分支）

    def __post_init__(self) -> None:
        # #P0-4 归一化 (None, None) → None
        object.__setattr__(self, "time_range", _normalize_time_range(self.time_range))
        # #P0-5 拒绝 str/bytes；非 None 必须是序列
        object.__setattr__(
            self,
            "instrument_filter",
            _normalize_sequence(self.instrument_filter, name="instrument_filter"),
        )
        if self.hive_filters is not None:
            if not isinstance(self.hive_filters, dict):
                raise ValidationError(
                    f"hive_filters 必须是 dict（分区列→值列表），收到 {type(self.hive_filters).__name__}"
                )
            normalized = {}
            for col, values in self.hive_filters.items():
                normalized[col] = _normalize_sequence(values, name=f"hive_filters.{col}")
            object.__setattr__(self, "hive_filters", normalized)

    def is_empty(self) -> bool:
        """无任何过滤维度。注意空集合（``[]``）不是空——它是 WHERE FALSE。"""
        return (
            self.time_range is None
            and self.instrument_filter is None
            and not self.hive_filters
            and self.filters is None
            and not self.extra
        )


@dataclass
class CompiledPredicate:
    """编译后的 WHERE 片段和参数。参数顺序必须和片段里的 ? 一一对应。"""
    where_sql: str            # "" 表示无谓词
    params: list[Any] = field(default_factory=list)


def expand_end_bound(
    end: Any, *, time_column_is_timestamp: bool
) -> tuple[Any, str]:
    """#27B 收官轮：timestamp 时间列上 date-only end 必须包含**完整一天**。

    对 timestamp 时间列，日期型 end 应包含整天——**`< next_day`**（严格小于次日
    零点），而不是旧 ``<= next_day - 1µs``。nanosecond timestamp 会漏掉当天最后
    999ns 范围的数据；``< next_day`` 对任意时间精度都包含完整一天。只对「午夜/
    纯日期」值展开，带时间的值原样保留（``<=``）。DuckDB 与 Polars 两条读取路径
    共用同一语义——三 backend 结果必须一致。

    返回 ``(bound_value, hi_op)``。
    """
    if not time_column_is_timestamp:
        return end, "<="
    try:
        ts = pd.Timestamp(end)
        if ts == ts.normalize():
            return ts + pd.Timedelta(days=1), "<"
    except (ValueError, TypeError):
        pass
    return end, "<="


def compile_predicate(
    predicate: Predicate,
    *,
    time_column: str,
    instrument_column: str,
) -> CompiledPredicate:
    """把结构化谓词编译成 DuckDB WHERE 子句。

    参数：
        predicate: 调用方传入的过滤条件
        time_column: registry 里该数据集登记的时间列名
        instrument_column: 同上，标的列名

    WHY 参数顺序：
        DuckDB list binding: `WHERE col IN ?` 搭配 `params=[[a, b, c]]`，
        list 本身作为一个参数；这里严格按 SQL 里 ? 出现的顺序 append。
    """
    if predicate.is_empty():
        return CompiledPredicate(where_sql="")

    clauses: list[str] = []
    params: list[Any] = []

    # 时间列/标的列可选（generic table）；对应的过滤维度没列就明确报错。
    if predicate.time_range is not None:
        if not time_column:
            raise ValidationError("该数据集未声明 time_column，无法应用 time_range 过滤")
        t_col = _quote_ident(time_column)
        start, end = predicate.time_range
        if start is not None:
            lo_op = ">" if predicate.time_lower_exclusive else ">="
            clauses.append(f"{t_col} {lo_op} ?")
            params.append(start)
        if end is not None:
            end_value, hi_op = expand_end_bound(
                end, time_column_is_timestamp=predicate.time_column_is_timestamp
            )
            if predicate.time_upper_exclusive:
                hi_op = "<"
            clauses.append(f"{t_col} {hi_op} ?")
            params.append(end_value)

    # #P0-2 空股票池（[]）→ WHERE FALSE，绝不能等价于全市场。
    if predicate.instrument_filter is not None:
        if len(predicate.instrument_filter) == 0:
            clauses.append("1 = 0")  # 空股票池 → 空结果（无需 instrument 列）
        else:
            if not instrument_column:
                raise ValidationError(
                    "该数据集未声明 instrument_column，无法应用 instrument_filter 过滤"
                )
            i_col = _quote_ident(instrument_column)
            # list 作为单个参数绑定到 IN ?
            clauses.append(f"{i_col} IN ?")
            params.append(list(predicate.instrument_filter))

    if predicate.hive_filters:
        for col_name in sorted(predicate.hive_filters.keys()):
            values = predicate.hive_filters[col_name]
            if values is None:
                continue  # 该分区列无约束
            values = list(values)
            # #P0-3 空 hive filter → WHERE FALSE（不是跳过 = 全年份扫描）。
            if len(values) == 0:
                clauses.append("1 = 0")
                continue
            clauses.append(f"{_quote_ident(col_name)} IN ?")
            params.append(values)

    if predicate.filters is not None:
        from data_access.read.predicate_ast import compile_filter_duckdb

        filter_sql, filter_params = compile_filter_duckdb(predicate.filters)
        if filter_sql:
            clauses.append(f"({filter_sql})")
            params.extend(filter_params)

    if predicate.extra:
        # 旧占位；如真有人传了，说明是在用尚未定义的扩展
        raise ValidationError("predicate.extra 未启用；请使用 filters= 通用过滤表达式")

    where_sql = "WHERE " + " AND ".join(clauses)
    return CompiledPredicate(where_sql=where_sql, params=params)


def _arrow_bound_value(value: Any) -> Any:
    """pyarrow 需要**类型化** bound：str → python datetime。

    DuckDB 会对 VARCHAR 字面量隐式 cast 到 date/timestamp；pyarrow 的 dataset
    expression 不会（`timestamp[us] >= string` 直接
    ``ArrowNotImplementedError``）。转成 ``datetime.datetime`` 后 date32 与
    timestamp 列的比较 kernel 都能接受。
    """
    if isinstance(value, str):
        try:
            return pd.Timestamp(value).to_pydatetime()
        except (ValueError, TypeError):
            return value
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):  # pandas.Timestamp / np.datetime64 兼容
        return to_py()
    return value


def compile_predicate_arrow(
    predicate: Predicate,
    *,
    time_column: str,
    instrument_column: str,
    pc: Any = None,
):
    """把结构化谓词编译成 pyarrow.compute 表达式（真 PyArrow backend parity）。

    与 ``compile_predicate``（DuckDB SQL）共用**同一份**语义，三 backend 结果必须
    一致：

    * ``time_range`` 的 end 走 ``expand_end_bound``——timestamp 列上 date-only
      end 变成 ``< next_day``（完整一天），不再 ``<= end 00:00`` 丢最后一天白天；
    * ``instrument_filter=[]`` → 假表达式（空股票池 → 0 行），**绝不**等于全市场；
    * ``hive_filters`` 空集同理 → 假表达式。

    返回 pyarrow 表达式，或 ``None``（无任何谓词）。``pc`` 不传时惰性 import。
    """
    if predicate.is_empty():
        return None
    if pc is None:
        import pyarrow.compute as _pc

        pc = _pc

    import pyarrow.dataset as pa_ds

    def _false_expr() -> Any:
        # 空集合恒假谓词：``col != col`` → 对每行都 false（null 行被 scanner 按
        # keep-null 语义丢弃）→ 0 行，且类型无关（``isin([])`` 在 string 列会报
        # "values set: null"）。pyarrow 没有列无关的常量 Expression（scalar 比较
        # 返回 BooleanScalar 不是 Expression，scanner 拒绝）。空股票池/空 hive
        # filter 在 DuckDB 是 ``1=0``，这里等价为恒假表达式。
        col = instrument_column or time_column
        if not col:
            raise ValidationError(
                "空集合过滤（instrument_filter=[] / 空 hive filter）需要数据集声明 "
                "instrument_column/time_column 之一来构造恒假谓词"
            )
        return pc.not_equal(pc.field(col), pc.field(col))

    exprs: list[Any] = []

    if predicate.time_range is not None:
        if not time_column:
            raise ValidationError(
                "该数据集未声明 time_column，无法应用 time_range 过滤"
            )
        start, end = predicate.time_range
        if start is not None:
            exprs.append(
                pa_ds.field(time_column) >= _arrow_bound_value(start)
            )
        if end is not None:
            end_value, hi_op = expand_end_bound(
                end, time_column_is_timestamp=predicate.time_column_is_timestamp
            )
            end_value = _arrow_bound_value(end_value)
            if hi_op == "<":
                exprs.append(pa_ds.field(time_column) < end_value)
            else:
                exprs.append(pa_ds.field(time_column) <= end_value)

    # #P0-2 空股票池（[]）→ 假表达式，绝不能等价于全市场。
    if predicate.instrument_filter is not None:
        if len(predicate.instrument_filter) == 0:
            return _false_expr()
        if not instrument_column:
            raise ValidationError(
                "该数据集未声明 instrument_column，无法应用 instrument_filter 过滤"
            )
        exprs.append(
            pa_ds.field(instrument_column).isin(list(predicate.instrument_filter))
        )

    if predicate.hive_filters:
        for col_name in sorted(predicate.hive_filters.keys()):
            values = predicate.hive_filters[col_name]
            if values is None:
                continue
            values = list(values)
            # #P0-3 空 hive filter → 假表达式（不是跳过 = 全量扫描）。
            if len(values) == 0:
                return _false_expr()
            exprs.append(pa_ds.field(col_name).isin(values))

    if predicate.filters is not None:
        from data_access.read.predicate_ast import compile_filter_arrow

        expr = compile_filter_arrow(predicate.filters, pc=pc)
        if expr is not None:
            exprs.append(expr)

    if predicate.extra:
        raise ValidationError(
            "predicate.extra 未启用；请使用 filters= 通用过滤表达式"
        )
    if not exprs:
        return None
    combined = exprs[0]
    for e in exprs[1:]:
        combined = combined & e
    return combined


def _quote_ident(name: str) -> str:
    """DuckDB 标识符引用：用双引号包起来，内部双引号转义。

    WHY：列名从 registry 来，可控；但加一层就能防住 registry 里某个维护者
    手滑写了带空格的列名。
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'

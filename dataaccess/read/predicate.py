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

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from data_access.core.exceptions import ValidationError


@dataclass
class Predicate:
    """结构化谓词。None 表示该维度不过滤。"""
    time_range: tuple[Any, Any] | None = None           # (start, end)，闭区间
    instrument_filter: Sequence[str] | None = None      # 标的白名单
    hive_filters: dict[str, Sequence[Any]] | None = None  # hive 分区列 IN 过滤
    extra: list[dict] | None = None
    time_column_is_timestamp: bool = False              # 时间列为 timestamp 时做 end-of-day 展开

    def is_empty(self) -> bool:
        return (
            self.time_range is None
            and not self.instrument_filter
            and not self.hive_filters
            and not self.extra
        )


@dataclass
class CompiledPredicate:
    """编译后的 WHERE 片段和参数。参数顺序必须和片段里的 ? 一一对应。"""
    where_sql: str            # "" 表示无谓词
    params: list[Any] = field(default_factory=list)


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

    # 列名用 quote_ident，防止列名里有奇怪字符（实际上不会有，但便宜的防御）
    t_col = _quote_ident(time_column)
    i_col = _quote_ident(instrument_column)

    clauses: list[str] = []
    params: list[Any] = []

    if predicate.time_range is not None:
        start, end = predicate.time_range
        if start is not None:
            clauses.append(f"{t_col} >= ?")
            params.append(start)
        if end is not None:
            end_value = end
            if predicate.time_column_is_timestamp:
                # 对 timestamp 时间列，日期型 end 应包含整天；展开为当日末。
                # 只对"午夜/纯日期"值展开，带时间的值原样保留。
                try:
                    ts = pd.Timestamp(end)
                    if ts == ts.normalize():
                        end_value = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                except (ValueError, TypeError):
                    end_value = end
            clauses.append(f"{t_col} <= ?")
            params.append(end_value)

    if predicate.instrument_filter:
        # list 作为单个参数绑定到 IN ?
        clauses.append(f"{i_col} IN ?")
        params.append(list(predicate.instrument_filter))

    if predicate.hive_filters:
        for col_name in sorted(predicate.hive_filters.keys()):
            values = list(predicate.hive_filters[col_name] or [])
            if not values:
                continue
            clauses.append(f"{_quote_ident(col_name)} IN ?")
            params.append(values)

    if predicate.extra:
        # 预留给后续扩展，PR1 不处理；有人传了就抛错提醒
        raise ValidationError("predicate.extra 在 PR1 未启用")

    where_sql = "WHERE " + " AND ".join(clauses)
    return CompiledPredicate(where_sql=where_sql, params=params)


def _quote_ident(name: str) -> str:
    """DuckDB 标识符引用：用双引号包起来，内部双引号转义。

    WHY：列名从 registry 来，可控；但加一层就能防住 registry 里某个维护者
    手滑写了带空格的列名。
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'

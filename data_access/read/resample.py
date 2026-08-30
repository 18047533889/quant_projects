"""R39 #68 —— standalone ``frequency``：对返回帧做 resample。

DataRequest 的 ``frequency`` 以前只是「计划元数据」（只在 explain 显示、单独设置
被拒绝）。本模块实现真实的返回帧降频：

    - 按 ``(instrument, 时间桶)`` 分组；数值列聚合（默认 mean），其余列取 first；
    - 支持 daily / weekly / monthly / quarterly / yearly / hourly / minute；
    - 不支持 / 时间列缺失 → typed ``UnsupportedFeatureError`` / ``ValidationError``。

实现（PERF-3，2026-08-28）：
    原实现 ``df.groupby([inst, pd.Grouper(key=time, freq=alias)]).agg(...)`` 在
    pandas 里对每个数值列走 Python-level 聚合 dispatch，分钟级大表（单日
    122 万行 × 多列）较慢。新实现：

    * 首选 **polars ``group_by_dynamic``**（惰性表达式，向量化）：``every`` 用
      polars 时长别名，``start_by='window'`` 与 pandas ``Grouper(freq=alias)``
      的窗口左边界一致；分桶后把窗口标签重标到 pandas 的桶标签（monthly/
      quarterly/yearly 用月末/季末/年末，weekly 用周日，daily/hourly/minute
      本身对齐）；数值列 ``mean`` 前先把 ``NaN`` 显式转 ``null``，保证与
      ``numpy`` 的 ``skipna`` 语义一致；其余列 ``first``。
    * pandas 路径保留为兜底（polars 不可用时/空表退化），语义与旧实现逐字节
      相同（同一 ``groupby+Grouper``）。

    输出回到 pyarrow Table（列序、dtype、行序与旧实现一致）。
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from data_access.core.exceptions import UnsupportedFeatureError, ValidationError

#: frequency 字符串 → pandas offset alias。
_FREQ_MAP = {
    "daily": "D",
    "day": "D",
    "weekly": "W",
    "week": "W",
    "monthly": "M",
    "month": "M",
    "quarterly": "Q",
    "quarter": "Q",
    "yearly": "Y",
    "annual": "Y",
    "year": "Y",
    "hourly": "h",
    "hour": "h",
    "minute": "T",
    "min": "T",
}

#: frequency 字符串 → polars ``group_by_dynamic`` ``every`` 时长。
_POLARS_EVERY = {
    "D": "1d",
    "W": "1w",
    "M": "1mo",
    "Q": "1q",
    "Y": "1y",
    "h": "1h",
    "T": "1m",
}

#: polars 窗口左边界（pandas Grouper 同频 label）→ 需要重标到 pandas 桶标签的
#: 频率。daily/hourly/minute 的窗口左边界 == pandas 标签，无需重标。
_LABEL_RELABEL = {"M": "month", "Q": "quarter", "Y": "year", "W": "week"}


def _relabel_window(ts_expr: Any, alias: str) -> Any:
    """把 polars ``group_by_dynamic`` 的窗口左边界重标到 pandas 桶标签。

    - monthly (M): 月末最后一天（``dt.month_end()``）；
    - quarterly (Q): 季末最后一天（季度号 ×3 月的最后一天）；
    - yearly (Y): 年末 12-31；
    - weekly (W): pandas ``W`` = 周日结束周 → 周一标签 +6 天；
    - daily/hourly/minute: 本身对齐，原样返回。
    """
    import polars as pl

    if alias not in _LABEL_RELABEL:
        return ts_expr
    kind = _LABEL_RELABEL[alias]
    if kind == "month":
        return ts_expr.dt.month_end()
    if kind == "year":
        return ts_expr.dt.replace(month=12, day=31)
    if kind == "week":
        # pandas freq W 的桶标签是周日（周结束）；polars 1w 从周一开始。
        return ts_expr + pl.duration(days=6)
    # quarterly: 由窗口左边界（季初）推季末
    return ts_expr.dt.replace(
        month=ts_expr.dt.quarter() * 3,
        day=ts_expr.dt.replace(month=ts_expr.dt.quarter() * 3, day=1).dt.days_in_month(),
    )


def _agg_exprs_polars(numeric_cols: list[str], other_cols: list[str]):
    """构造 polars 聚合表达式：数值列 mean（NaN→null），其余列 first。

    顺序与 pandas ``df.groupby(...).agg(agg_dict)`` 的列序一致：数值列先
    （按原表列序）、其余列后（按原表列序）。
    """
    import polars as pl

    exprs: list[Any] = []
    for col in numeric_cols:
        exprs.append(
            pl.col(col)
            .replace(float("nan"), None)
            .mean()
            .alias(col)
        )
    for col in other_cols:
        exprs.append(pl.col(col).first().alias(col))
    return exprs


def _apply_frequency_polars(
    table: pa.Table,
    *,
    alias: str,
    time_column: str,
    instrument_column: str | None,
) -> pa.Table | None:
    """polars ``group_by_dynamic`` 向量化降频路径；语义不匹配/不可用 → None。

    返回 None 时调用方回退 pandas 实现（语义逐字节一致）。
    """
    import polars as pl

    every = _POLARS_EVERY.get(alias)
    if every is None:
        return None
    pf = pl.from_arrow(table)
    # 时间列必须是 datetime 才能用 group_by_dynamic（Arrow timestamp 已满足）。
    ts = pf.schema.get(time_column)
    if not isinstance(ts, (pl.Datetime, pl.Date)):
        return None

    group_by: list[Any] = []
    if instrument_column and instrument_column in pf.columns:
        group_by.append(instrument_column)

    # 数值列（与 pandas select_dtypes(number) 一致：float/int，不含 bool/timedelta）。
    # 顺序保持原表列序（pandas agg dict 列序 = 数值列先行、其余列后行）。
    numeric_cols: list[str] = []
    other_cols: list[str] = []
    for name in pf.columns:
        if name == time_column or (instrument_column and name == instrument_column):
            continue
        dtype = pf.schema[name]
        if isinstance(dtype, (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)):
            numeric_cols.append(name)
        else:
            other_cols.append(name)

    lf = pf.lazy().sort(time_column)
    agg_exprs = _agg_exprs_polars(numeric_cols, other_cols)
    if not agg_exprs:
        # 退化：全表只有 time/instrument 列 → 等价 pandas ``.size()`` 去重
        # （每 (inst, 时间桶) 一组一行）。
        gb = lf.group_by_dynamic(
            time_column,
            every=every,
            group_by=group_by or None,
            start_by="window",
            closed="left",
        ).agg(pl.len())
        out = gb.drop("len").collect()
    else:
        gb = lf.group_by_dynamic(
            time_column,
            every=every,
            group_by=group_by or None,
            start_by="window",
            closed="left",
        ).agg(agg_exprs)
        out = gb.collect()

    # 重标窗口标签到 pandas 桶标签。
    out = out.with_columns(_relabel_window(pl.col(time_column), alias).alias(time_column))
    # 行序与 pandas ``sort_values(time_column)`` 一致（仅按时间列排）。
    out = out.sort(time_column).unique(subset=[time_column, *(group_by or [])], keep="first")
    return out.to_arrow()


def _apply_frequency_pandas(
    table: pa.Table,
    *,
    alias: str,
    time_column: str,
    instrument_column: str | None,
) -> pa.Table:
    """旧实现（groupby + pandas Grouper），逐字节语义保留，作为兜底。"""
    import pandas as pd

    df = table.to_pandas()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    groupers: list[Any] = []
    if instrument_column and instrument_column in df.columns:
        groupers.append(instrument_column)
    groupers.append(pd.Grouper(key=time_column, freq=alias))

    numeric_cols = set(df.select_dtypes(include="number").columns)
    agg: dict[str, str] = {}
    for col in df.columns:
        if col == time_column or (instrument_column and col == instrument_column):
            continue
        agg[col] = "mean" if col in numeric_cols else "first"
    if not agg:
        # 全表只有 time/instrument 列：退化成分组计数也无意义，直接按组去重。
        out = df.groupby(groupers, dropna=False).size().reset_index(name="__n__")
        out = out.drop(columns=["__n__"])
    else:
        out = df.groupby(groupers, dropna=False).agg(agg).reset_index()
    out = out.sort_values(time_column).reset_index(drop=True)
    return pa.Table.from_pandas(out, preserve_index=False)


def apply_frequency(
    table: pa.Table,
    *,
    frequency: str,
    time_column: str | None,
    instrument_column: str | None = None,
) -> pa.Table:
    """把 Arrow Table 降频到目标 frequency。

    - ``time_column`` 缺失 → 无法 resample，抛 ``ValidationError``（不静默 no-op）；
    - 数值列聚合 ``mean``，其它非时间/标序列取 ``first``。
    """
    alias = _FREQ_MAP.get(str(frequency).strip().lower())
    if alias is None:
        raise UnsupportedFeatureError(
            f"standalone frequency={frequency!r} 不支持；合法: "
            f"{sorted(set(_FREQ_MAP))}"
        )
    if table.num_rows == 0:
        return table
    if not time_column or time_column not in table.column_names:
        raise ValidationError(
            "standalone frequency 需要数据集声明 time_column 且结果表包含该列；"
            f"time_column={time_column!r}，结果列={table.column_names}"
        )

    try:
        out = _apply_frequency_polars(
            table,
            alias=alias,
            time_column=time_column,
            instrument_column=instrument_column,
        )
    except Exception:
        # polars 路径任何失败都回退 pandas 兜底（语义权威），绝不抛 polars 内部错。
        out = None
    if out is None:
        out = _apply_frequency_pandas(
            table,
            alias=alias,
            time_column=time_column,
            instrument_column=instrument_column,
        )
    return out


__all__ = ["apply_frequency"]

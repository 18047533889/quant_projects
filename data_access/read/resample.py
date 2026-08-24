"""R39 #68 —— standalone ``frequency``：对返回帧做 resample。

DataRequest 的 ``frequency`` 以前只是「计划元数据」（只在 explain 显示、单独设置
被拒绝）。本模块实现真实的返回帧降频：

    - 按 ``(instrument, 时间桶)`` 分组；数值列聚合（默认 mean），其余列取 first；
    - 支持 daily / weekly / monthly / quarterly / yearly / hourly / minute；
    - 不支持 / 时间列缺失 → typed ``UnsupportedFeatureError`` / ``ValidationError``。

实现用 pandas（依赖已有）做 groupby+Grouper；输出回到 pyarrow Table。
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
    import pandas as pd

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


__all__ = ["apply_frequency"]

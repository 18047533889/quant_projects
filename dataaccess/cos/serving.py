"""
data_access.cos.serving —— Source / Serving / Semantic View 三层模型（#31）

物理布局分层
    - **Source Layer**：COS 原始 date/period 文件，负责可追溯性（不改变）。
    - **Optimized Serving Layer**：为性能改变 partition/sort/rowgroup；可以是
      filing_year/month 分区的 PIT 优化湖，或高扇出表的日频预聚合。
    - **Semantic View Layer**：PIT、单位、市场适配、derived field。

本模块提供两个可落地的 serving 工具：
    1. ``route_minute_storage``：分钟级 date-major vs bucket-major 路由（#17）。
    2. ``materialize_daily_aggregate``：高扇出源表（TopTen/Industry）→ 日频
       预聚合 serving 数据集（#30）。

原则：COS 原始布局不动；优化只发生在本地 serving 层；预聚合自动维护避免
因子挖掘重复 groupby。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError


def route_minute_storage(
    *,
    instrument_count: int | None = None,
    history_days: int | None = None,
    market: str | None = None,
    full_market_threshold: int = 2000,
    narrow_threshold: int = 200,
    long_history_days: int = 250,
) -> str:
    """分钟级存储路由：#17 全市场日内扫描走 date-major，窄股票池长历史走
    bucket-major（``Symbol%N``，32/64/128 buckets），Cost Router 自动选择。

    返回：
        - ``date_major``  ：按 {YYYY-MM-DD}.parquet 布局（全市场单日一个文件）
        - ``bucket_major``：按 bucket 分区（窄池长历史减少扫描）
        - ``current``      ：保持现状（边界情况）
    """
    if instrument_count is None:
        return "current"
    if instrument_count >= full_market_threshold:
        return "date_major"
    if (
        instrument_count <= narrow_threshold
        and history_days is not None
        and history_days >= long_history_days
    ):
        return "bucket_major"
    if history_days is not None and history_days >= long_history_days * 2:
        return "bucket_major"
    return "date_major"


def materialize_daily_aggregate(
    store: Any,
    *,
    source_dataset: str,
    serving_dataset: str,
    time_column: str,
    instrument_column: str,
    groupby: Sequence[str],
    value_columns: Mapping[str, Sequence[str]],
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    mode: str = "overwrite",
    partition_by: Sequence[str] | None = None,
    **params: Any,
) -> dict[str, Any]:
    """#30 高扇出源表 → 日频预聚合 serving 数据集。

    ``source_dataset``（如 ashare_stock_topten_shareholder）读出后按
    ``groupby``（通常含 time + instrument）聚合成每日一行，写入
    ``serving_dataset``（如 ashare_topten_daily_serving）。

    ``value_columns``：``{输出列名: {"columns": [...], "agg": "sum|max|min|mean"}}``
    或简写 ``{输出列名: [源列...]}``（默认单列 max / 多列 sum；如需 top10
    concentration = SUM(单列) 请用 dict 形式显式声明 agg="sum"）。

    返回写结果 + 行数。
    """
    def _qi(name: str) -> str:
        return f'"{str(name).replace(chr(34), chr(34) * 2)}"'

    _AGG = {"sum": "SUM", "max": "MAX", "min": "MIN", "mean": "AVG"}

    def _normalize_value(cols: Any) -> tuple[list[str], str]:
        if isinstance(cols, Mapping):
            srcs = [str(c) for c in cols.get("columns", [])]
            agg = str(cols.get("agg", "max")).lower()
            if agg not in _AGG:
                raise ValidationError(f"agg={agg!r} 不支持；合法: {sorted(_AGG)}")
            return srcs, agg
        srcs = [str(c) for c in cols]
        return srcs, ("sum" if len(srcs) > 1 else "max")

    if not value_columns:
        raise ValidationError("materialize_daily_aggregate: value_columns 不能为空")
    ds = store._registry.get(source_dataset)
    if not time_column or not instrument_column:
        raise ValidationError("需要 time_column / instrument_column")

    cols = list(dict.fromkeys([time_column, instrument_column, *groupby]))
    for out, vc in value_columns.items():
        srcs, _ = _normalize_value(vc)
        cols.extend(srcs)
    table = store.read_arrow(
        source_dataset,
        columns=cols,
        time_range=time_range,
        instrument_filter=instrument_filter,
        allow_sparse=True,
        **params,
    )
    if table.num_rows == 0:
        return {"rows": 0, "path": None, "mode": mode}

    # DuckDB 内 groupby + 聚合（source 扇出 → 每日一行）
    group_list = ", ".join(_qi(c) for c in dict.fromkeys([time_column, instrument_column, *groupby]))
    agg_parts: list[str] = []
    for out, vc in value_columns.items():
        srcs, agg = _normalize_value(vc)
        expr = " + ".join(_qi(c) for c in srcs) if len(srcs) > 1 else _qi(srcs[0])
        agg_parts.append(f"{_AGG[agg]}({expr}) AS {_qi(out)}")
    sql = (
        f"SELECT {group_list}, {', '.join(agg_parts)} "
        f"FROM read_parquet(?) GROUP BY {group_list}"
    )
    paths = store._prepare_dataset_read(
        ds, time_range=time_range, params=dict(params), instrument_filter=instrument_filter
    )
    if not paths:
        return {"rows": 0, "path": None, "mode": mode}
    path_param = paths if len(paths) > 1 else paths[0]
    agg_table = store._engine.execute_arrow(sql, [path_param], deadline_ms=None)

    result = store.write_arrow(
        serving_dataset, agg_table, mode=mode, partition_by=partition_by
    )
    result["source_dataset"] = source_dataset
    result["rows_computed"] = agg_table.num_rows
    return result


# 三层模型（文档化，供 introspection）
LAYERS = {
    "source": "COS 原始布局（date/period 文件，可追溯性）",
    "serving": "优化层（可改 partition/sort/rowgroup，如 PIT 湖 / 预聚合）",
    "semantic_view": "语义层（PIT、单位、市场适配、derived field）",
}


def describe_layers() -> dict[str, str]:
    """返回三层模型的说明（#31 文档化入口）。"""
    return dict(LAYERS)


__all__ = [
    "route_minute_storage",
    "materialize_daily_aggregate",
    "describe_layers",
    "LAYERS",
]

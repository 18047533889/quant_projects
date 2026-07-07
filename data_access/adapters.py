"""
data_access.adapters —— Arrow 到各种下游类型的转换器

职责：
    把 DuckDB/Arrow 结果转成 factor_engine 等历史代码期望的类型，
    主要是 `(timestamp, instrument)` MultiIndex Series。

WHY 单独一个模块：
    原代码里各处自己写 `df.set_index([...]).sort_index()`，格式略有差异
    （有的带 sort、有的带 normalize、有的带 dedup）。统一在这里，改一处
    生效全局，也方便写测试验证契约。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def arrow_to_multiindex_series(
    table: pa.Table,
    *,
    timestamp_column: str,
    instrument_column: str,
    value_column: str,
    output_name: str | None = None,
    sort_index: bool = True,
    dedup: bool = True,
    normalize_timestamp: bool = False,
    timestamp_unit: str | None = None,
):
    """把 Arrow Table 转成 `(timestamp, instrument)` MultiIndex Series。

    参数：
        table: DuckDB 查出来的 Arrow Table，至少含 timestamp_column、
               instrument_column、value_column 三列
        value_column: 要作为 Series 值的列
        output_name: 结果 Series 的 name；默认用 value_column
        sort_index: 是否 sort_index()（默认 True，和原 ParquetSource 一致）
        dedup: 是否按 (ts, instrument) 去重保留最后一条（原代码默认行为）
        normalize_timestamp: 是否 .dt.normalize()（日线数据有时需要）
        timestamp_unit: 如果 timestamp 列是数值（epoch），用哪个单位转 datetime；
                        None 表示不转，信任 Arrow schema

    返回：
        pd.Series，索引名固定为 ["timestamp", "instrument"]，name=output_name

    兼容性：
        原 ParquetSource.load_column 返回的 Series 长这样：
            MultiIndex [timestamp: datetime64[ns] (tz-naive),
                        instrument: str]
            values: 原列类型
        本函数的默认参数就是为了对齐这个契约。
    """
    import pandas as pd

    name = output_name or value_column
    batch = arrow_table_to_multiindex_columns(
        table,
        timestamp_column=timestamp_column,
        instrument_column=instrument_column,
        value_columns=[value_column],
        output_names={value_column: name},
        sort_index=sort_index,
        dedup=dedup,
        normalize_timestamp=normalize_timestamp,
        timestamp_unit=timestamp_unit,
    )
    return batch[name]


def arrow_table_to_multiindex_columns(
    table: pa.Table,
    *,
    timestamp_column: str,
    instrument_column: str,
    value_columns: list[str],
    output_names: dict[str, str] | None = None,
    sort_index: bool = True,
    dedup: bool = True,
    normalize_timestamp: bool = False,
    timestamp_unit: str | None = None,
) -> dict[str, Any]:
    """一次 Arrow→pandas 转换，批量产出多列 MultiIndex Series（load_columns 热路径）。"""
    import pandas as pd

    if not value_columns:
        return {}

    output_names = output_names or {}
    needed_cols = list(
        dict.fromkeys([timestamp_column, instrument_column, *value_columns])
    )
    missing = [c for c in needed_cols if c not in table.column_names]
    if missing:
        raise KeyError(
            f"Arrow Table 缺少必要列 {missing}，实际列：{table.column_names}"
        )

    df = table.select(needed_cols).to_pandas(self_destruct=True, split_blocks=True)

    ts_series = df[timestamp_column]
    if timestamp_unit is not None:
        ts_numeric = pd.to_numeric(ts_series, errors="coerce")
        ts_series = pd.to_datetime(ts_numeric, unit=timestamp_unit, utc=True, errors="coerce")
    elif not pd.api.types.is_datetime64_any_dtype(ts_series):
        ts_series = pd.to_datetime(ts_series, utc=True, errors="coerce")

    if getattr(ts_series.dt, "tz", None) is not None:
        ts_series = ts_series.dt.tz_convert(None)
    if normalize_timestamp:
        ts_series = ts_series.dt.normalize()

    def _normalize_instrument(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return text or None

    inst_series = df[instrument_column].map(_normalize_instrument)

    frame = pd.DataFrame(
        {
            "timestamp": ts_series,
            "instrument": inst_series,
        }
    )
    for col in value_columns:
        frame[col] = df[col]

    frame = frame[frame["timestamp"].notna() & frame["instrument"].notna()]

    if dedup:
        frame = frame.drop_duplicates(subset=["timestamp", "instrument"], keep="last")

    indexed = frame.set_index(["timestamp", "instrument"])
    indexed.index = indexed.index.set_names(["timestamp", "instrument"])
    if sort_index:
        indexed = indexed.sort_index()

    out: dict[str, Any] = {}
    for col in value_columns:
        target = output_names.get(col, col)
        series = indexed[col]
        series.name = target
        out[target] = series
    return out

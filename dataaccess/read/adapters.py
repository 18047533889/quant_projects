"""
data_access.read.adapters —— Arrow 到各种下游类型的转换器

职责：
    把 DuckDB/Arrow 结果转成 factor_engine 等历史代码期望的类型，
    主要是 `(timestamp, instrument)` MultiIndex Series。

WHY 单独一个模块：
    原代码里各处自己写 `df.set_index([...]).sort_index()`，格式略有差异
    （有的带 sort、有的带 normalize、有的带 dedup）。统一在这里，改一处
    生效全局，也方便写测试验证契约。

维护人：quant 基础平台组    最后更新：2026-07-11
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from data_access.core.exceptions import ValidationError
from .key_policy import KeyPolicy, resolve_key_policy


def arrow_to_multiindex_series(
    table: pa.Table,
    *,
    timestamp_column: str,
    instrument_column: str,
    value_column: str,
    output_name: str | None = None,
    sort_index: bool = True,
    dedup: bool = True,
    key_policy: KeyPolicy | None = None,
    normalize_timestamp: bool = False,
    timestamp_unit: str | None = None,
):
    """把 Arrow Table 转成 `(timestamp, instrument)` MultiIndex Series。

    ``key_policy`` 是正式键契约。``dedup`` 仅保留为旧调用兼容参数：当策略允许
    ``keep_last`` 时决定是否执行旧式去重；production/strict 模式默认直接拒绝
    无效键和重复键。
    """
    name = output_name or value_column
    batch = arrow_table_to_multiindex_columns(
        table,
        timestamp_column=timestamp_column,
        instrument_column=instrument_column,
        value_columns=[value_column],
        output_names={value_column: name},
        sort_index=sort_index,
        dedup=dedup,
        key_policy=key_policy,
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
    key_policy: KeyPolicy | None = None,
    normalize_timestamp: bool = False,
    timestamp_unit: str | None = None,
) -> dict[str, Any]:
    """一次 Arrow→pandas 转换，批量产出多列 MultiIndex Series。"""
    import pandas as pd

    if not value_columns:
        return {}

    output_names = output_names or {}
    policy = resolve_key_policy(key_policy)
    needed_cols = list(
        dict.fromkeys([timestamp_column, instrument_column, *value_columns])
    )
    if policy.duplicate_resolution and policy.duplicate_resolution not in needed_cols:
        needed_cols.append(policy.duplicate_resolution)
    missing = [c for c in needed_cols if c not in table.column_names]
    if missing:
        raise KeyError(
            f"Arrow Table 缺少必要列 {missing}，实际列：{table.column_names}"
        )

    df = table.select(needed_cols).to_pandas(self_destruct=True, split_blocks=True)

    ts_series = df[timestamp_column]
    if timestamp_unit is not None:
        ts_numeric = pd.to_numeric(ts_series, errors="coerce")
        ts_series = pd.to_datetime(
            ts_numeric, unit=timestamp_unit, utc=True, errors="coerce"
        )
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
        },
        index=df.index,
    )
    for col in value_columns:
        frame[col] = df[col]
    if policy.duplicate_resolution:
        frame["_revision"] = df[policy.duplicate_resolution]

    null_mask = frame["timestamp"].isna() | frame["instrument"].isna()
    if null_mask.any():
        if policy.invalid_key == "error":
            raise ValidationError(
                f"Arrow→MultiIndex 转换发现 {int(null_mask.sum())} 行无效键"
                "（timestamp 或 instrument 为空）；production 模式禁止静默丢弃。"
            )
        frame = frame.loc[~null_mask].copy()

    if policy.duplicate_resolution:
        # 版本列只有在存在同一主键多条记录时才参与决议，但所有有效键行都必须
        # 具备版本值，否则“最新一条”无法被确定性定义。
        revision_null = frame["_revision"].isna()
        if revision_null.any():
            raise ValidationError(
                f"duplicate_resolution={policy.duplicate_resolution!r} 存在 "
                f"{int(revision_null.sum())} 行 NULL，无法确定性去重"
            )

        duplicate_key_mask = frame.duplicated(
            subset=["timestamp", "instrument"], keep=False
        )
        if duplicate_key_mask.any():
            duplicate_version_mask = frame.loc[duplicate_key_mask].duplicated(
                subset=["timestamp", "instrument", "_revision"], keep=False
            )
            if duplicate_version_mask.any():
                raise ValidationError(
                    "同一 (timestamp, instrument, revision) 出现重复记录；"
                    "版本列无法提供唯一顺序，拒绝依赖文件扫描顺序"
                )

            # mergesort 保证稳定；先按 key 排序，再按 revision 降序，选择最新版本。
            frame = frame.sort_values(
                ["timestamp", "instrument", "_revision"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            frame = frame.drop_duplicates(
                subset=["timestamp", "instrument"], keep="first"
            )
        frame = frame.drop(columns=["_revision"])
    else:
        dup_mask = frame.duplicated(
            subset=["timestamp", "instrument"], keep=False
        )
        if dup_mask.any():
            if policy.duplicate_key == "error":
                raise ValidationError(
                    f"Arrow→MultiIndex 转换发现 {int(dup_mask.sum())} 行重复键"
                    "（timestamp, instrument）；production 模式禁止静默 dedup。"
                )
            if dedup:
                # 仅为历史研究模式保留。没有 revision/order 列时 keep_last 不是
                # production 语义，严格模式会在上面的 policy 分支直接失败。
                frame = frame.drop_duplicates(
                    subset=["timestamp", "instrument"], keep="last"
                )

    indexed = frame.set_index(["timestamp", "instrument"])
    indexed.index = indexed.index.set_names(["timestamp", "instrument"])
    if sort_index:
        indexed = indexed.sort_index()

    out: dict[str, Any] = {}
    for col in value_columns:
        target = output_names.get(col, col)
        series = _coerce_factor_value_series(indexed[col])
        series.name = target
        out[target] = series
    return out


def _coerce_factor_value_series(series):
    """SQL/Arrow 路径常见 Decimal→object；统一为 float64 以对齐 pandas 因子值。"""
    import numpy as np
    import pandas as pd
    from decimal import Decimal

    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")

    if pd.api.types.is_numeric_dtype(series):
        if series.dtype != np.float64:
            return series.astype("float64", copy=False)
        return series

    if series.dtype == object:
        non_null = series.dropna()
        if non_null.empty:
            return series.astype("float64")
        sample = non_null.head(64)
        if all(isinstance(v, Decimal) for v in sample):
            return series.map(
                lambda v: float(v) if isinstance(v, Decimal) else v,
                na_action="ignore",
            ).astype("float64")
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() >= non_null.notna().sum():
            return converted.astype("float64")
    return series

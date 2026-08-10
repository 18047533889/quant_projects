# -*- coding: utf-8 -*-
"""因子长表 / 宽表 / MultiIndex Series 互转（与 materializer 落盘格式对齐）。"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

LONG_KEY_COLUMNS = ("datetime", "asset")
VALUE_COLUMN = "value"


def series_to_long_table(
    series: pd.Series,
    *,
    timestamp_col: str = "datetime",
    asset_col: str = "asset",
    value_col: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """MultiIndex Series 转因子长表 DataFrame。
    
    参数:
        series: MultiIndex Series
        timestamp_col: 见函数签名（可选）
        asset_col: 见函数签名（可选）
        value_col: 见函数签名（可选）
    
    返回:
        pd.DataFrame
    """
    if not isinstance(series, pd.Series):
        raise ValueError(f"期望 pd.Series，实际 {type(series).__name__}")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise ValueError("期望 MultiIndex(timestamp, instrument) Series")

    frame = series.reset_index()
    cols = list(frame.columns)
    frame = frame.rename(
        columns={
            cols[0]: timestamp_col,
            cols[1]: asset_col,
            cols[-1]: value_col,
        }
    )
    keep = [timestamp_col, asset_col, value_col]
    out = frame[keep].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out[asset_col] = out[asset_col].astype("string")
    out[value_col] = out[value_col].astype("float64")
    return out


def long_table_to_series(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    asset_col: str = "asset",
    value_col: str = VALUE_COLUMN,
    sort: bool = True,
) -> pd.Series:
    """因子长表转 MultiIndex Series。
    
    参数:
        frame: 长表 DataFrame
        timestamp_col: 见函数签名（可选）
        asset_col: 见函数签名（可选）
        value_col: 见函数签名（可选）
        sort: 见函数签名（可选）
    
    返回:
        pd.Series
    """
    missing = {timestamp_col, asset_col, value_col} - set(frame.columns)
    if missing:
        raise ValueError(f"长表缺少列: {sorted(missing)}")

    work = frame[[timestamp_col, asset_col, value_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col])
    if sort:
        work = work.sort_values([timestamp_col, asset_col])
    indexed = work.set_index([timestamp_col, asset_col])[value_col]
    indexed.index.names = [timestamp_col, asset_col]
    # 与引擎中间态对齐：instrument 层用 object，避免 string dtype 不兼容
    lvl_asset = indexed.index.get_level_values(1).astype(object)
    indexed.index = pd.MultiIndex.from_arrays(
        [indexed.index.get_level_values(0), lvl_asset],
        names=indexed.index.names,
    )
    return indexed.astype("float64")


def pivot_long_to_wide(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    asset_col: str = "asset",
    value_col: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """单因子长表转宽表 panel。
    
    参数:
        frame: 长表 DataFrame
        timestamp_col: 见函数签名（可选）
        asset_col: 见函数签名（可选）
        value_col: 见函数签名（可选）
    
    返回:
        pd.DataFrame
    """
    missing = {timestamp_col, asset_col, value_col} - set(frame.columns)
    if missing:
        raise ValueError(f"长表缺少列: {sorted(missing)}")

    work = frame[[timestamp_col, asset_col, value_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col])
    panel = work.pivot(index=timestamp_col, columns=asset_col, values=value_col)
    panel.index.name = timestamp_col
    panel.columns.name = asset_col
    return panel.sort_index()


def unpivot_wide_to_long(
    panel: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    asset_col: str = "asset",
    value_col: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """宽表 panel 转因子长表。
    
    参数:
        panel: 见函数签名
        timestamp_col: 见函数签名（可选）
        asset_col: 见函数签名（可选）
        value_col: 见函数签名（可选）
    
    返回:
        pd.DataFrame
    """
    if panel.index.name is None:
        panel = panel.copy()
        panel.index.name = timestamp_col
    asset_name = panel.columns.name or asset_col
    frame = panel.reset_index().melt(
        id_vars=[panel.index.name],
        var_name=asset_name,
        value_name=value_col,
    )
    frame = frame.rename(
        columns={panel.index.name: timestamp_col, asset_name: asset_col}
    )
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame[asset_col] = frame[asset_col].astype("string")
    # R32-P0-024: 宽表→长表保留原 panel 精度。绝不强制 downcast 成 float32 ——
    # 否则历史 float64 面板在 wide 路径做一次 upsert 就被永久降精度。空 panel
    # 保持 float64 默认。
    if not frame[value_col].empty:
        frame[value_col] = frame[value_col].astype(panel.dtypes.iloc[0])
    else:
        frame[value_col] = frame[value_col].astype("float64")
    return frame.dropna(subset=[value_col]).reset_index(drop=True)


def pivot_multi_factor_long_to_wide(
    frame: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    timestamp_col: str = "datetime",
    asset_col: str = "asset",
) -> pd.DataFrame:
    """多因子长表转 MultiIndex 列宽表。
    
    参数:
        frame: 长表 DataFrame
        factor_columns: 见函数签名
        timestamp_col: 见函数签名（可选）
        asset_col: 见函数签名（可选）
    
    返回:
        pd.DataFrame
    """
    factor_cols = list(factor_columns)
    if not factor_cols:
        raise ValueError("factor_columns 不能为空")

    panels: list[pd.DataFrame] = []
    for fid in factor_cols:
        if fid not in frame.columns:
            raise ValueError(f"缺少因子列: {fid}")
        sub = frame[[timestamp_col, asset_col, fid]].rename(columns={fid: VALUE_COLUMN})
        panel = pivot_long_to_wide(
            sub,
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            value_col=VALUE_COLUMN,
        )
        panel.columns = pd.MultiIndex.from_product([[fid], panel.columns.astype(str)])
        panels.append(panel)

    merged = panels[0]
    for panel in panels[1:]:
        merged = merged.join(panel, how="outer")
    merged.columns.names = ["factor_id", "asset"]
    return merged.sort_index()

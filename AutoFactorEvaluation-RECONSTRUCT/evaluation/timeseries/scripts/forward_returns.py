"""未来收益率计算与缓存。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：按 t+1 -> t+1+N 窗口计算 N 期目标收益，并支持缓存复用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import (
    COL_ASSET,
    COL_DATETIME,
    COL_FORWARD_RETURN,
    COL_HORIZON,
)
from .validation import Severity, ValidationBuffer


def _market_fingerprint(market_data: pd.DataFrame, horizons: list[int], price_col: str) -> str:
    """为 forward return 输入构造稳定缓存键。

    入参：
        market_data: 至少包含 `datetime`、`asset` 与价格列。
        horizons: 目标收益持有期列表。
        price_col: 选定价格列（`vwap` 或 `open`）。

    出参：
        str：缓存文件名使用的短 SHA256 摘要。
    """
    payload = market_data[[COL_DATETIME, COL_ASSET, price_col]].copy()
    payload = payload.sort_values([COL_ASSET, COL_DATETIME]).reset_index(drop=True)
    row_hash = pd.util.hash_pandas_object(payload, index=False).to_numpy(dtype="uint64", copy=False).tobytes()
    meta = json.dumps({"horizons": sorted(int(h) for h in horizons), "price_col": price_col}).encode("utf-8")
    h = hashlib.sha256()
    h.update(row_hash)
    h.update(meta)
    return h.hexdigest()[:16]


def _read_cached_forward_returns(path: Path) -> pd.DataFrame:
    """读取 forward return 缓存文件。"""
    return pd.read_parquet(path)


def _write_cached_forward_returns(df: pd.DataFrame, path: Path) -> None:
    """写出 forward return 缓存文件。"""
    df.to_parquet(path, index=False)


def pick_price_column(market_data: pd.DataFrame, vb: ValidationBuffer) -> str | None:
    """选择 forward return 的价格列。

    规则：优先 `vwap`，其次 `open`；两者都缺失时记录 ERROR 并返回 `None`。
    """
    if "vwap" in market_data.columns:
        return "vwap"
    if "open" in market_data.columns:
        vb.add(Severity.WARNING, "vwap_missing_fallback_open", "vwap not provided, fallback to open")
        return "open"
    vb.add(Severity.ERROR, "price_column_missing", "market_data requires either vwap or open column")
    return None


def compute_forward_returns(
    market_data: pd.DataFrame,
    horizons: list[int],
    vb: ValidationBuffer,
    *,
    price_col: str | None = None,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame | None, str | None]:
    """按无未来函数定义计算 forward return。

    公式：
        对于 t 时点暴露、持有期 N，
        `forward_return(t, N) = price(t + 1 + N) / price(t + 1) - 1`。

    说明：
        当未来价格不足、分母接近 0 或收益非有限值时，相关行会被置 NaN 后丢弃。
    """
    selected_price_col = price_col or pick_price_column(market_data, vb)
    if selected_price_col is None:
        return None, None

    base = market_data[[COL_DATETIME, COL_ASSET, selected_price_col]].copy()
    base = base.sort_values([COL_ASSET, COL_DATETIME]).reset_index(drop=True)

    out_parts: list[pd.DataFrame] = []
    for h in sorted(int(x) for x in horizons):
        part = base.copy()
        g = part.groupby(COL_ASSET, sort=False)[selected_price_col]
        start = g.shift(-1)
        end = g.shift(-(h + 1))
        part[COL_HORIZON] = h
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_return = end / start - 1.0
        valid = (
            np.isfinite(start.to_numpy(dtype=float, copy=False))
            & np.isfinite(end.to_numpy(dtype=float, copy=False))
            & (np.abs(start.to_numpy(dtype=float, copy=False)) > float(eps))
            & np.isfinite(raw_return.to_numpy(dtype=float, copy=False))
        )
        part[COL_FORWARD_RETURN] = np.where(valid, raw_return, np.nan)
        out_parts.append(part[[COL_DATETIME, COL_ASSET, COL_HORIZON, COL_FORWARD_RETURN]])

    out = pd.concat(out_parts, ignore_index=True)
    out = out.dropna(subset=[COL_FORWARD_RETURN]).copy()
    return out, selected_price_col


def compute_or_load_forward_returns(
    market_data: pd.DataFrame,
    horizons: list[int],
    cache_dir: Path,
    vb: ValidationBuffer,
    *,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame | None, str | None, bool]:
    """优先读取缓存，否则计算并落盘 forward return。

    入参：
        market_data: 行情表。
        horizons: 目标收益持有期列表。
        cache_dir: 缓存目录。
        vb: 验证缓冲区。

    出参：
        tuple[pd.DataFrame | None, str | None, bool]：
        `(forward_return_df, 价格列名, 是否命中缓存)`。
    """
    price_col = pick_price_column(market_data, vb)
    if price_col is None:
        return None, None, False
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _market_fingerprint(market_data, horizons, price_col)
    cache_file = cache_dir / f"forward_returns_{key}.parquet"
    if cache_file.exists():
        fr = _read_cached_forward_returns(cache_file)
        return fr, price_col, True

    fr, resolved_price_col = compute_forward_returns(
        market_data,
        horizons,
        vb,
        price_col=price_col,
        eps=eps,
    )
    if fr is None:
        return None, None, False
    _write_cached_forward_returns(fr, cache_file)
    return fr, resolved_price_col, False


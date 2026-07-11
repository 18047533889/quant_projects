"""AutoFactorEvaluation 共享缓存。

行情读取统一走 ``data_access``；本模块只保留进程内 DataFrame 缓存，并以
``DataSnapshot.snapshot_id`` 自动失效，不再创建私有合并 Parquet 缓存。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from integrations.quant_platform import get_market_data_spec, load_market_frame

logger = logging.getLogger("cache_utils")

_global_market_data: pd.DataFrame | None = None
_global_cache_key: str | None = None
_global_snapshot_id: str | None = None


def preload_market_data(
    market_data_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    dataset: str | None = None,
    market: str = "ashare",
    start_date: str | None = None,
    end_date: str | None = None,
    instrument_filter: Sequence[str] | None = None,
    fields: Iterable[str] | None = None,
    rename_columns: dict[str, str] | None = None,
    lowercase: bool = False,
) -> pd.DataFrame:
    """在主进程预加载 DataAccess 行情，供 fork worker 通过 COW 共享。"""
    return load_market_data_cached(
        market_data_path,
        cache_dir,
        dataset=dataset,
        market=market,
        start_date=start_date,
        end_date=end_date,
        instrument_filter=instrument_filter,
        fields=fields,
        rename_columns=rename_columns,
        lowercase=lowercase,
    )


def load_market_data_cached(
    market_data_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    *,
    dataset: str | None = None,
    market: str = "ashare",
    start_date: str | None = None,
    end_date: str | None = None,
    instrument_filter: Sequence[str] | None = None,
    fields: Iterable[str] | None = None,
    force_reload: bool = False,
    rename_columns: dict[str, str] | None = None,
    lowercase: bool = False,
) -> pd.DataFrame:
    """经 DataAccess 读取行情并按 snapshot 维护进程内缓存。

    ``market_data_path`` / ``cache_dir`` 仅兼容旧调用；路径不再直接扫描。
    可将 ``market_data_path`` 传成已登记 dataset 名，或显式传 ``dataset=``。
    """
    del cache_dir
    global _global_market_data, _global_cache_key, _global_snapshot_id

    resolved_market, resolved_dataset = _resolve_dataset(
        market=market,
        dataset=dataset,
        market_data_path=market_data_path,
    )
    spec = get_market_data_spec(resolved_market, dataset=resolved_dataset)
    logical_fields = tuple(fields or spec.fields.keys())
    request_key = _request_key(
        market=resolved_market,
        dataset=spec.dataset,
        start_date=start_date,
        end_date=end_date,
        instrument_filter=instrument_filter,
        fields=logical_fields,
        rename_columns=rename_columns,
        lowercase=lowercase,
    )

    from data_access import get_store

    snapshot = get_store().describe_dataset(
        spec.dataset,
        instrument_filter=list(instrument_filter) if instrument_filter else None,
    )
    if (
        not force_reload
        and _global_market_data is not None
        and _global_cache_key == request_key
        and _global_snapshot_id == snapshot.snapshot_id
    ):
        return _global_market_data

    frame, read_snapshot = load_market_frame(
        market=resolved_market,
        dataset=spec.dataset,
        fields=logical_fields,
        start_date=start_date,
        end_date=end_date,
        instrument_filter=instrument_filter,
    )
    # 兼容旧模块期望的轴列。
    if resolved_market == "ashare":
        frame["TradeDate"] = frame["datetime"]
        frame["Symbol"] = frame["asset"]
    else:
        frame["TradeDate"] = frame["datetime"]
        frame["Ticker"] = frame["asset"]
    if rename_columns:
        frame = frame.rename(columns=rename_columns)
    if lowercase:
        frame.columns = [str(column).lower() for column in frame.columns]

    _global_market_data = frame
    _global_cache_key = request_key
    _global_snapshot_id = read_snapshot.snapshot_id
    logger.info(
        "DataAccess 行情缓存更新 dataset=%s rows=%d snapshot=%s",
        spec.dataset,
        len(frame),
        read_snapshot.snapshot_id,
    )
    return frame


def clear_market_data_cache() -> None:
    global _global_market_data, _global_cache_key, _global_snapshot_id
    _global_market_data = None
    _global_cache_key = None
    _global_snapshot_id = None


def market_data_cache_info() -> dict[str, object]:
    return {
        "rows": 0 if _global_market_data is None else len(_global_market_data),
        "request_key": _global_cache_key,
        "snapshot_id": _global_snapshot_id,
    }


def _resolve_dataset(
    *,
    market: str,
    dataset: str | None,
    market_data_path: str | Path | None,
) -> tuple[str, str | None]:
    if dataset:
        return market, str(dataset)
    configured = os.environ.get("AUTOFACTOR_MARKET_DATASET", "").strip()
    if configured:
        return market, configured
    if market_data_path is not None:
        value = str(market_data_path)
        if value in {"ashare_stock_daily", "us_stock_daily", "us_stocks_sip_day_aggs"}:
            inferred_market = "us" if value.startswith("us_") else "ashare"
            return inferred_market, value
        lowered = value.lower()
        if "us_stock" in lowered or "massive" in lowered:
            return "us", "us_stock_daily"
    return market, None


def _request_key(**payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def get_forward_return_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """前向收益属于评估中间产物，仍使用显式本地缓存目录。"""
    if cache_dir is not None:
        directory = Path(cache_dir)
    else:
        from config_manager import ConfigManager

        try:
            directory = ConfigManager().path("timeseries_forward_return_cache")
        except Exception as exc:
            raise ValueError(
                "forward return cache_dir 必须从外部传入或在 ConfigManager 中配置"
            ) from exc
    if not directory.exists():
        raise FileNotFoundError(f"前向收益缓存目录不存在: {directory}")
    return directory

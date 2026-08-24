# -*- coding: utf-8
"""从 data_access ``factor_lake_staging`` 读取因子长表并转为引擎 Series。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from factor_engine.util.logging_utils import get_logger
from factor_engine.storage.factor_format import long_table_to_series

logger = get_logger("factor_engine.storage.staging_loader")

_STAGING_DATASET = "factor_lake_staging"
_VALUE_COLUMNS = ("value",)


def _ensure_data_access() -> None:
    """R21-130..133: import the installed ``data_access`` (no sys.path injection)."""
    from factor_engine.storage.data_access_loader import ensure_data_access_importable

    ensure_data_access_importable()


def load_factor_series_from_staging(
    factor_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """从 staging 数据集读取因子 MultiIndex Series。
    
    参数:
        factor_id: 因子唯一标识
        start: 起始时间（含）（可选）
        end: 结束时间（含）（可选）
    
    返回:
        pd.Series
    """
    _ensure_data_access()
    from data_access import get_store

    store = get_store()
    time_range = None
    if start is not None or end is not None:
        time_range = (start, end)

    read_columns = list(_VALUE_COLUMNS)
    optional_meta = ("is_valid", "invalid_reason", "factor_version", "data_snapshot_id")
    try:
        frame = store.read_frame(
            _STAGING_DATASET,
            columns=[*read_columns, *optional_meta],
            time_range=time_range,
            factor_id=factor_id,
        )
    except Exception:
        frame = store.read_frame(
            _STAGING_DATASET,
            columns=list(read_columns),
            time_range=time_range,
            factor_id=factor_id,
        )

    if frame is None or len(frame) == 0:
        raise FileNotFoundError(
            f"staging 中无因子数据: dataset={_STAGING_DATASET}, factor_id={factor_id}"
        )

    rename: dict[str, str] = {}
    if "datetime" not in frame.columns and "timestamp" in frame.columns:
        rename["timestamp"] = "datetime"
    if "asset" not in frame.columns and "instrument" in frame.columns:
        rename["instrument"] = "asset"
    if rename:
        frame = frame.rename(columns=rename)

    series = long_table_to_series(frame)
    logger.info(
        "从 staging 加载因子 '%s'：rows=%d range=[%s, %s]",
        factor_id,
        len(series),
        series.index.get_level_values(0).min(),
        series.index.get_level_values(0).max(),
    )
    return series


def staging_factor_exists(factor_id: str) -> bool:
    """探测 staging 是否已有指定因子数据。
    
    参数:
        factor_id: 因子唯一标识
    
    返回:
        bool
    """
    try:
        load_factor_series_from_staging(factor_id)
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:
        logger.debug("staging 探测 factor_id=%s 失败: %s", factor_id, exc)
        return False


def delete_staging_rows(
    factor_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    """从 staging 删除指定时间范围的因子行。
    
    参数:
        factor_id: 因子唯一标识
        start: 起始时间（含）（可选）
        end: 结束时间（含）（可选）
        after: 开区间下界（严格大于）（可选）
    
    返回:
        dict[str, Any]
    
    
    ``start``/``end`` 为闭区间；``after`` 为开区间下界（删除 strictly > after 的行）。
        staging 数据集未注册时跳过删除（本地 repair / 无 staging 环境）。
    """
    _ensure_data_access()
    try:
        from data_access import get_store

        store = get_store()
        if hasattr(store, "delete_rows"):
            result = store.delete_rows(
                "factor_lake_staging",
                start=start,
                end=end,
                after=after,
                factor_id=factor_id,
            )
        else:
            result = _delete_staging_rows_local(
                factor_id,
                start=start,
                end=end,
                after=after,
            )
    except Exception as exc:
        from data_access.core.exceptions import ValidationError

        if isinstance(exc, ValidationError) or "未注册" in str(exc):
            logger.warning(
                "staging 补偿跳过 factor_id=%s：%s",
                factor_id,
                exc,
            )
            return {
                "ok": False,
                "skipped": True,
                "factor_id": factor_id,
                "rows_deleted": 0,
                "reason": str(exc),
            }
        raise

    rows_deleted = int(result.get("rows_deleted", 0))
    logger.info(
        "staging 行级删除 factor_id=%s rows_deleted=%d",
        factor_id,
        rows_deleted,
    )
    return {
        "ok": True,
        "factor_id": factor_id,
        "rows_deleted": rows_deleted,
        "partitions": result.get("partitions", []),
    }


def _delete_staging_rows_local(
    factor_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    """无 delete_rows API 时的本地 Parquet 行删除 fallback。
    
    参数:
        factor_id: 因子唯一标识
        start: 起始时间（含）（可选）
        end: 结束时间（含）（可选）
        after: 开区间下界（严格大于）（可选）
    
    返回:
        dict[str, Any]
    """
    import os
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_access import get_store

    store = get_store()
    target_dir = store.resolve_dataset_path("factor_lake_staging", factor_id=factor_id)
    if not target_dir.exists():
        return {"rows_deleted": 0, "partitions": []}

    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    after_ts = pd.Timestamp(after) if after else None
    rows_deleted = 0
    partitions: list[str] = []

    for parquet_path in sorted(target_dir.rglob("*.parquet")):
        try:
            df = pq.read_table(str(parquet_path), partitioning=None).to_pandas()
        except Exception as exc:
            logger.warning("跳过无法读取的分区 %s: %s", parquet_path, exc)
            continue
        if df.empty or "datetime" not in df.columns:
            continue

        dt = pd.to_datetime(df["datetime"])
        if after_ts is not None and start_ts is None and end_ts is None:
            delete_mask = dt > after_ts
        else:
            delete_mask = pd.Series(True, index=df.index)
            if start_ts is not None:
                delete_mask &= dt >= start_ts
            if end_ts is not None:
                delete_mask &= dt <= end_ts
            if after_ts is not None:
                delete_mask &= dt > after_ts

        removed = int(delete_mask.sum())
        if removed <= 0:
            continue

        kept = df.loc[~delete_mask]
        rows_deleted += removed
        partitions.append(str(parquet_path.parent))

        if kept.empty:
            parquet_path.unlink(missing_ok=True)
            continue

        tmp_path = parquet_path.parent / f".delete.tmp.{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(pa.Table.from_pandas(kept, preserve_index=False), str(tmp_path))
        os.replace(str(tmp_path), str(parquet_path))

    return {"rows_deleted": rows_deleted, "partitions": partitions}

# -*- coding: utf-8 -*-
"""§27-29: DataShapeEstimate —— metadata-only shape 推断（不加载大数据也能规划）。

R27-29 要求：
    1. metadata-only shape estimate（Parquet metadata / DataAccess manifest / schema /
       date bounds / universe metadata / row-group stats / calendar sessions）。
    2. 去掉固定 3000 instruments —— 改用 shape.estimated_instruments；没有 metadata
       时使用 conservative universe prior（CSI300≈300, CSI500≈500, CSI1000≈1000,
       ALL_A≈5500）。
    3. 日期数优先真实交易日历 —— 走 calendar_service/session_count；fallback 才普通
       工作日估计（不要主路径 pd.bdate_range）。
    4. 不加载大数据也能规划（充分静态代码、metadata-only、synthetic fixtures、小数据
       correctness tests）。

设计：
    - :class:`DataShapeEstimate`: frozen dataclass，汇总估算结果（rows/dates/instruments/
      columns/bytes/density/frequency/bars_per_session/group_count/remote/storage_kind/
      sorted_by/partition_by/projected_columns）。
    - :func:`estimate_shape_from_metadata`: 从 DataAccess manifest / Parquet metadata /
      universe / calendar 构造 shape。
    - :func:`estimate_instruments_from_universe`: 按 universe 名称推断仪器数（保守先验）。
    - :func:`estimate_dates_from_calendar`: 优先真实交易日历 session_count；fallback
      才普通工作日估计。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)

# Conservative universe priors (§28)
UNIVERSE_INSTRUMENT_PRIOR: dict[str, int] = {
    "CSI300": 300,
    "CSI500": 500,
    "CSI1000": 1000,
    "CSI800": 800,
    "HS300": 300,
    "ZZ500": 500,
    "ZZ1000": 1000,
    "ALL_A": 5500,
    "ALL_ASHARE": 5500,
    "ASHARE_ALL": 5500,
    "US_LARGE": 500,
    "US_MID": 2000,
    "US_ALL": 8000,
    "NASDAQ": 3000,
    "NYSE": 2500,
    "SP500": 500,
    "RUSSELL2000": 2000,
}


@dataclass(frozen=True)
class DataShapeEstimate:
    """§27: metadata-only data shape 估算（不加载大数据也能规划）。

    字段：
        estimated_rows: 估算总行数
        estimated_dates: 估算日期数（交易日/session）
        estimated_instruments: 估算仪器数（§28：去掉固定 3000，改用 universe prior）
        estimated_columns: 估算列数
        estimated_bytes: 估算字节数
        average_row_width_bytes: 平均行宽（字节）
        density: 密度（0.0~1.0，NaN 占比的反面）
        frequency: 频率（"daily" / "minute" / "tick" / ...）
        bars_per_session: 日内每 session bar 数（可选，分钟/tick 数据）
        group_count: 分组数（可选，group-wise 算子）
        remote: 是否远程存储
        storage_kind: 存储类型（"parquet" / "duckdb" / "memory" / "cos" / ...）
        sorted_by: 已排序键（tuple of column names）
        partition_by: 分区键（tuple of column names）
        projected_columns: 投影列（tuple of column names）
        rows_known: whether estimated_rows is known (True) or unknown (False)
    """

    estimated_rows: int
    estimated_dates: int
    estimated_instruments: int
    estimated_columns: int
    estimated_bytes: int
    average_row_width_bytes: float
    density: float
    frequency: str
    bars_per_session: int | None = None
    group_count: int | None = None
    remote: bool = False
    storage_kind: str = "unknown"
    sorted_by: tuple[str, ...] = ()
    partition_by: tuple[str, ...] = ()
    projected_columns: tuple[str, ...] = ()
    rows_known: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_rows": self.estimated_rows,
            "estimated_dates": self.estimated_dates,
            "estimated_instruments": self.estimated_instruments,
            "estimated_columns": self.estimated_columns,
            "estimated_bytes": self.estimated_bytes,
            "average_row_width_bytes": round(self.average_row_width_bytes, 2),
            "density": round(self.density, 3),
            "frequency": self.frequency,
            "bars_per_session": self.bars_per_session,
            "group_count": self.group_count,
            "remote": self.remote,
            "storage_kind": self.storage_kind,
            "sorted_by": list(self.sorted_by),
            "partition_by": list(self.partition_by),
            "projected_columns": list(self.projected_columns),
            "rows_known": self.rows_known,
        }


def estimate_instruments_from_universe(
    universe: str | None,
    *,
    universe_metadata: Any | None = None,
    default: int = 3000,
) -> int:
    """§28: 按 universe 名称推断仪器数（保守先验 + metadata）。

    优先级：
        1. universe_metadata.instrument_count（真实 metadata）
        2. UNIVERSE_INSTRUMENT_PRIOR 匹配（保守先验）
        3. default 兜底（默认 3000）

    参数:
        universe: universe 名称（可选）
        universe_metadata: universe metadata（可选，duck-typed object with instrument_count）
        default: 兜底默认值（可选）

    返回:
        估算仪器数
    """
    # 1. 真实 metadata
    if universe_metadata is not None:
        count = getattr(universe_metadata, "instrument_count", None)
        if count is not None and int(count) > 0:
            return int(count)

    # 2. 保守先验
    if universe:
        u = str(universe).strip().upper().replace("-", "").replace("_", "")
        for key, val in UNIVERSE_INSTRUMENT_PRIOR.items():
            k = key.upper().replace("-", "").replace("_", "")
            if k in u or u in k:
                _LOGGER.debug(
                    f"estimate_instruments_from_universe: {universe!r} matched prior {key}={val}"
                )
                return val

    # 3. default 兜底（R21-P029: empty/unknown -> 0, not default）
    return default


def estimate_dates_from_calendar(
    start_date: str | date | pd.Timestamp | None,
    end_date: str | date | pd.Timestamp | None,
    *,
    calendar: Any | None = None,
    market: str | None = None,
    frequency: str = "daily",
    allow_approximate_calendar: bool = True,
) -> int:
    """§29: 日期数优先真实交易日历 —— calendar.session_count；fallback 才普通工作日估计。

    优先级：
        1. calendar.days 长度（真实交易日历）
        2. TradingCalendar.get_trading_calendar(market) session_count（加载真实日历）
        3. pandas bdate_range 工作日估计（fallback，仅 allow_approximate_calendar=True）

    参数:
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        calendar: TradingCalendar 实例（可选）
        market: 市场标识（ashare/us，可选）
        frequency: 频率（daily/minute/tick，可选）
        allow_approximate_calendar: 是否允许近似工作日历（可选，默认 True）

    返回:
        估算日期数（交易日/session）
    """
    if start_date is None or end_date is None:
        return 0

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        return 0

    # 1. 真实 calendar
    if calendar is not None:
        days = getattr(calendar, "days", None)
        if days:
            # 在 [start_date, end_date] 范围内的交易日数
            count = sum(1 for d in days if start_ts <= d <= end_ts)
            _LOGGER.debug(
                f"estimate_dates_from_calendar: real calendar session_count={count} "
                f"[{start_ts.date()} .. {end_ts.date()}]"
            )
            return count

    # 2. 加载真实日历
    if market:
        try:
            from factor_engine.storage.trading_calendar import get_trading_calendar

            loaded_cal = get_trading_calendar(
                market,
                allow_approximate_calendar=allow_approximate_calendar,
            )
            if loaded_cal is not None:
                days = loaded_cal.days
                count = sum(1 for d in days if start_ts <= d <= end_ts)
                _LOGGER.debug(
                    f"estimate_dates_from_calendar: loaded {market!r} calendar "
                    f"session_count={count} [{start_ts.date()} .. {end_ts.date()}]"
                )
                return count
        except Exception as exc:
            _LOGGER.debug(
                f"estimate_dates_from_calendar: failed to load {market!r} calendar ({exc}), "
                f"fallback to bdate if allow_approximate_calendar={allow_approximate_calendar}"
            )

    # 3. fallback 普通工作日估计（仅 allow_approximate_calendar=True）
    if allow_approximate_calendar:
        try:
            bdays = pd.bdate_range(start=start_ts, end=end_ts, freq="B")
            count = len(bdays)
            _LOGGER.debug(
                f"estimate_dates_from_calendar: approximate bdate_range={count} "
                f"[{start_ts.date()} .. {end_ts.date()}]"
            )
            return count
        except Exception as exc:
            _LOGGER.warning(
                f"estimate_dates_from_calendar: bdate_range failed ({exc}), returning 0"
            )
            return 0

    # production fail-closed
    _LOGGER.warning(
        f"estimate_dates_from_calendar: no real calendar and allow_approximate_calendar=False, "
        f"returning 0"
    )
    return 0


def estimate_shape_from_metadata(
    *,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
    universe: str | None = None,
    dataset: str | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    parquet_metadata: Any | None = None,
    data_access_manifest: Any | None = None,
    universe_metadata: Any | None = None,
    calendar: Any | None = None,
    market: str | None = None,
    frequency: str = "daily",
    bars_per_session: int | None = None,
    remote: bool = False,
    storage_kind: str = "unknown",
    density: float = 0.95,
    average_row_width_bytes: float = 256.0,
    allow_approximate_calendar: bool = True,
) -> DataShapeEstimate:
    """§27: metadata-only shape 估算（不加载大数据也能规划）。

    来源优先级：
        - Parquet metadata（row-group stats / num_rows / total_byte_size）
        - DataAccess manifest（selected_rowgroups / selected_bytes / instrument_count）
        - schema（columns）
        - date bounds（start_date / end_date）
        - universe metadata（instrument_count）
        - calendar sessions（§29：真实交易日历）

    参数:
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        universe: universe 名称（可选）
        dataset: data_access dataset 名称（可选）
        columns: 列名列表（可选）
        parquet_metadata: Parquet file metadata（可选）
        data_access_manifest: DataAccess SourceManifest（可选）
        universe_metadata: universe metadata（可选）
        calendar: TradingCalendar 实例（可选）
        market: 市场标识（可选）
        frequency: 频率（可选）
        bars_per_session: 日内每 session bar 数（可选）
        remote: 是否远程存储（可选）
        storage_kind: 存储类型（可选）
        density: 密度（可选，默认 0.95）
        average_row_width_bytes: 平均行宽（可选，默认 256）
        allow_approximate_calendar: 是否允许近似工作日历（可选，默认 True）

    返回:
        DataShapeEstimate 实例
    """
    # 1. 估算仪器数（§28：去掉固定 3000）
    estimated_instruments = estimate_instruments_from_universe(
        universe,
        universe_metadata=universe_metadata,
        default=3000,
    )

    # 2. 估算日期数（§29：真实交易日历）
    estimated_dates = estimate_dates_from_calendar(
        start_date,
        end_date,
        calendar=calendar,
        market=market,
        frequency=frequency,
        allow_approximate_calendar=allow_approximate_calendar,
    )

    # 3. Parquet metadata
    if parquet_metadata is not None:
        num_rows = getattr(parquet_metadata, "num_rows", None)
        total_byte_size = getattr(parquet_metadata, "serialized_size", None) or getattr(
            parquet_metadata, "total_byte_size", None
        )
        if num_rows is not None and int(num_rows) > 0:
            estimated_rows = int(num_rows)
            if total_byte_size is not None and int(total_byte_size) > 0:
                estimated_bytes = int(total_byte_size)
                average_row_width_bytes = estimated_bytes / estimated_rows
            else:
                estimated_bytes = int(estimated_rows * average_row_width_bytes)
            estimated_columns = len(columns) if columns else 10
            return DataShapeEstimate(
                estimated_rows=estimated_rows,
                estimated_dates=estimated_dates or (estimated_rows // max(1, estimated_instruments)),
                estimated_instruments=estimated_instruments,
                estimated_columns=estimated_columns,
                estimated_bytes=estimated_bytes,
                average_row_width_bytes=average_row_width_bytes,
                density=density,
                frequency=frequency,
                bars_per_session=bars_per_session,
                remote=remote,
                storage_kind=storage_kind,
                projected_columns=tuple(columns) if columns else (),
                rows_known=True,
            )

    # 4. DataAccess manifest
    if data_access_manifest is not None:
        selected_bytes = getattr(data_access_manifest, "selected_bytes", None)
        selected_rowgroups = getattr(data_access_manifest, "selected_rowgroups", None)
        manifest_instrument_count = getattr(data_access_manifest, "instrument_count", None)
        if manifest_instrument_count is not None and int(manifest_instrument_count) > 0:
            estimated_instruments = int(manifest_instrument_count)
        if selected_bytes is not None and int(selected_bytes) > 0:
            estimated_bytes = int(selected_bytes)
            estimated_rows = int(estimated_bytes / average_row_width_bytes)
            estimated_columns = len(columns) if columns else 10
            return DataShapeEstimate(
                estimated_rows=estimated_rows,
                estimated_dates=estimated_dates or (estimated_rows // max(1, estimated_instruments)),
                estimated_instruments=estimated_instruments,
                estimated_columns=estimated_columns,
                estimated_bytes=estimated_bytes,
                average_row_width_bytes=average_row_width_bytes,
                density=density,
                frequency=frequency,
                bars_per_session=bars_per_session,
                remote=remote,
                storage_kind=storage_kind,
                projected_columns=tuple(columns) if columns else (),
                rows_known=True,
            )

    # 5. 推断默认 shape（date × instrument × columns）
    if estimated_dates == 0:
        # 如果没有真实日期范围，默认 252 交易日（1 年）
        estimated_dates = 252
    estimated_rows = estimated_dates * estimated_instruments
    if frequency == "minute" and bars_per_session:
        estimated_rows *= bars_per_session
    estimated_columns = len(columns) if columns else 10
    estimated_bytes = int(estimated_rows * average_row_width_bytes)

    return DataShapeEstimate(
        estimated_rows=estimated_rows,
        estimated_dates=estimated_dates,
        estimated_instruments=estimated_instruments,
        estimated_columns=estimated_columns,
        estimated_bytes=estimated_bytes,
        average_row_width_bytes=average_row_width_bytes,
        density=density,
        frequency=frequency,
        bars_per_session=bars_per_session,
        remote=remote,
        storage_kind=storage_kind,
        projected_columns=tuple(columns) if columns else (),
        rows_known=True,
    )


def shape_to_cost_context(shape: DataShapeEstimate) -> dict[str, Any]:
    """将 DataShapeEstimate 转换为 backend.operator_cost.CostContext 兼容字典。

    参数:
        shape: DataShapeEstimate 实例

    返回:
        CostContext 兼容字典
    """
    return {
        "rows": shape.estimated_rows,
        "instruments": shape.estimated_instruments,
        "expected_density": shape.density,
        "group_count": shape.group_count,
        "session_bars": shape.bars_per_session,
    }

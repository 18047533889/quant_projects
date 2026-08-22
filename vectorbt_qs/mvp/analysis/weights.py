"""Portfolio and benchmark weight extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .risk_model import _normalize_frame


def realized_close_weights(pf: Any) -> pd.DataFrame:
    """Return end-of-day asset value divided by total portfolio NAV."""
    asset_value = pf.asset_value(group_by=False)
    if isinstance(asset_value, pd.Series):
        asset_value = asset_value.to_frame()
    total_value = pf.value()
    if isinstance(total_value, pd.DataFrame):
        if total_value.shape[1] != 1:
            raise ValueError("暴露分析要求共享资金池或单列组合价值")
        total_value = total_value.iloc[:, 0]
    total_value = pd.Series(total_value, copy=False).reindex(asset_value.index)
    if total_value.isna().any() or (total_value == 0.0).any():
        raise ValueError("组合价值包含空值或零，无法计算实际持仓权重")
    return _normalize_frame(asset_value.div(total_value, axis=0))


def extract_portfolio_weights(pf: Any, source: str = "realized_close") -> pd.DataFrame:
    """Extract one supported portfolio-weight view."""
    normalized = str(source).lower()
    if normalized == "realized_close":
        return realized_close_weights(pf)
    attr_by_source = {
        "scheduled_target": "_qs_scheduled_targets",
        "execution_target": "_qs_execution_targets",
    }
    if normalized not in attr_by_source:
        raise ValueError(
            "weight_source 仅支持 realized_close、scheduled_target、execution_target"
        )
    attr = attr_by_source[normalized]
    frame = getattr(pf, attr, None)
    if frame is None:
        raise ValueError(f"当前回测对象不包含 {normalized} 权重")
    realized = realized_close_weights(pf)
    targets = _normalize_frame(pd.DataFrame(frame)).reindex(
        index=realized.index,
        columns=realized.columns,
    )
    return targets.ffill().fillna(0.0)


def load_benchmark_weights(
    data_root: str | Path,
    index_symbol: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Load point-in-time index constituent weights from lqtp_data."""
    root = Path(data_root).resolve()
    directory = root / "IndexConstituent"
    if not directory.is_dir():
        raise FileNotFoundError(f"缺少指数成分目录: {directory}")
    requested = pd.DatetimeIndex(dates).normalize()
    date_frame = pd.DataFrame({"date": requested.strftime("%Y-%m-%d")})
    glob_path = (directory / "*.parquet").as_posix().replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.register("requested_dates", date_frame)
        frame = connection.execute(
            f"""
            SELECT
                strftime(i.TradeDate, '%Y-%m-%d') AS date,
                i.Symbol AS asset,
                SUM(i.Weight) / 100.0 AS weight
            FROM read_parquet('{glob_path}') i
            JOIN requested_dates d
              ON strftime(i.TradeDate, '%Y-%m-%d') = d.date
            WHERE i.IndexSymbol = ?
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            [str(index_symbol)],
        ).df()
    finally:
        connection.close()
    if frame.empty:
        raise ValueError(f"指数 {index_symbol} 在风险分析区间没有成分权重")
    frame["date"] = pd.to_datetime(frame["date"])
    weights = (
        frame.pivot(index="date", columns="asset", values="weight")
        .reindex(index=requested)
        .fillna(0.0)
    )
    weights.index.name = "date"
    weights.columns.name = "asset"
    sums = weights.sum(axis=1)
    invalid = sums[(sums < 0.98) | (sums > 1.02)]
    if not invalid.empty:
        first = invalid.index[0]
        raise ValueError(
            f"{index_symbol} {first.date()} 成分权重和异常: {invalid.iloc[0]:.6f}"
        )
    return weights


def load_industry_name_map(
    data_root: str | Path,
    date: pd.Timestamp,
) -> dict[str, str]:
    """Return ``industry_<code> -> Chinese name`` for one lqtp snapshot."""
    root = Path(data_root).resolve()
    directory = root / "StockIndustry"
    if not directory.is_dir():
        return {}
    requested = pd.Timestamp(date).normalize()
    exact = directory / f"{requested.date()}.parquet"
    if exact.is_file():
        frame = pd.read_parquet(
            exact,
            columns=["IndustrySource", "IndustryCode", "IndustryName"],
        )
    else:
        glob_path = (directory / "*.parquet").as_posix().replace("'", "''")
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                f"""
                SELECT IndustrySource, IndustryCode, IndustryName
                FROM read_parquet('{glob_path}')
                WHERE TradeDate <= ? AND IndustrySource = 'sw_l1'
                QUALIFY TradeDate = MAX(TradeDate) OVER ()
                """,
                [requested.date()],
            ).df()
        finally:
            connection.close()
    frame = frame.loc[frame["IndustrySource"].eq("sw_l1")].drop_duplicates(
        "IndustryCode"
    )
    return {
        f"industry_{code}": str(name)
        for code, name in zip(frame["IndustryCode"], frame["IndustryName"])
    }

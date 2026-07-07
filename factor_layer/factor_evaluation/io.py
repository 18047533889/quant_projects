from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_layer.data import load_factor_long

from factor_layer.factor_evaluation.config import SourceConfig


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(series, utc=True, errors="coerce")
    if isinstance(normalized.dtype, pd.DatetimeTZDtype):
        normalized = normalized.dt.tz_convert(None)
    return normalized


def _read_table(path: Path, *, columns: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=lambda c: c in columns)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported file type: {suffix}")


def _collect_table_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"数据路径不存在: {path}")

    parquet_paths = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.pq"))
    csv_paths = sorted(path.rglob("*.csv"))
    paths = parquet_paths + csv_paths
    if not paths:
        raise FileNotFoundError(f"目录下没有可读取的数据文件: {path}")
    return paths


def _finalize_keyed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = _normalize_timestamp(out["timestamp"])
    out["symbol"] = out["symbol"].astype("string").str.strip()
    out = out.dropna(subset=["timestamp", "symbol"]).copy()
    out = out.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def load_factor_data(
    lake_root: str | Path,
    factor_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    frame = load_factor_long(lake_root, factor_id, start=start, end=end)
    return frame.rename(columns={"value": "factor_raw"})


def load_market_data(
    source: SourceConfig,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    path = Path(source.market_data_path)
    read_columns = [
        source.market_timestamp_col,
        source.market_symbol_col,
        source.market_price_col,
    ]
    frames = []
    for table_path in _collect_table_paths(path):
        frame = _read_table(table_path, columns=read_columns)
        frame = frame.rename(
            columns={
                source.market_timestamp_col: "timestamp",
                source.market_symbol_col: "symbol",
                source.market_price_col: "price",
            }
        )
        frames.append(frame)
    market = _finalize_keyed_frame(pd.concat(frames, ignore_index=True))
    market["price"] = pd.to_numeric(market["price"], errors="coerce")
    market = market.dropna(subset=["price"]).copy()
    if start is not None:
        market = market.loc[market["timestamp"] >= pd.Timestamp(start)].copy()
    if end is not None:
        market = market.loc[market["timestamp"] <= pd.Timestamp(end)].copy()
    return market.reset_index(drop=True)


def load_universe_data(
    source: SourceConfig,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame | None:
    if source.universe_path is None:
        return None

    path = Path(source.universe_path)
    read_columns = [source.universe_timestamp_col, source.universe_symbol_col]
    if source.universe_membership_col:
        read_columns.append(source.universe_membership_col)

    frames = []
    for table_path in _collect_table_paths(path):
        frame = _read_table(table_path, columns=read_columns)
        frame = frame.rename(
            columns={
                source.universe_timestamp_col: "timestamp",
                source.universe_symbol_col: "symbol",
            }
        )
        frames.append(frame)
    universe = _finalize_keyed_frame(pd.concat(frames, ignore_index=True))

    if source.universe_membership_col:
        membership = universe[source.universe_membership_col]
        if source.universe_include_values:
            allowed = {str(value) for value in source.universe_include_values}
            mask = membership.astype("string").isin(allowed)
        elif pd.api.types.is_bool_dtype(membership) or pd.api.types.is_numeric_dtype(membership):
            mask = membership.fillna(0).astype(bool)
        else:
            normalized = membership.astype("string").str.lower().fillna("")
            mask = normalized.isin({"1", "true", "yes", "y", "member", "in"})
        universe = universe.loc[mask, ["timestamp", "symbol"]].copy()
    else:
        universe = universe.loc[:, ["timestamp", "symbol"]].copy()

    if start is not None:
        universe = universe.loc[universe["timestamp"] >= pd.Timestamp(start)].copy()
    if end is not None:
        universe = universe.loc[universe["timestamp"] <= pd.Timestamp(end)].copy()
    return universe.reset_index(drop=True)
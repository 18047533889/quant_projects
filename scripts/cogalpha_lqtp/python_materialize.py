#!/usr/bin/env python3
"""Materialize CogAlpha Python factors (non-DSL) on wide panel, symbol-chunked."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import default_ashare_pv_data_source_config  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from runtime.config import DataSourceConfig  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from storage.cache import CacheManager  # noqa: E402
from storage.factory import build_data_source  # noqa: E402

from scripts.cogalpha_lqtp.materialize import _normalize_symbol  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import release_memory  # noqa: E402

PANEL_COLS = ["open", "high", "low", "close", "volume", "amount", "ret", "preclose", "vwap"]


def _build_data_source(*, start: str, end: str):
    cfg = default_ashare_pv_data_source_config(start_date=start, end_date=end)
    ds_type = str(cfg.pop("type", "data_access"))
    return build_data_source(DataSourceConfig(type=ds_type, options=cfg))


def load_symbol_panel(
    *,
    start: str,
    end: str,
    symbol_chunk: int = 400,
) -> list[tuple[list[str], dict[str, pd.DataFrame]]]:
    """Return chunked symbol panels as {lowercase_col: date-indexed Series}."""
    ds = _build_data_source(start=start, end=end)
    if hasattr(ds, "load_columns"):
        ds.load_columns(PANEL_COLS)
    wide: dict[str, pd.DataFrame] = {}
    for col in PANEL_COLS:
        series = ds.get_column(col)
        if hasattr(series, "to_pandas"):
            series = series.to_pandas()
        if isinstance(series, pd.Series) and isinstance(series.index, pd.MultiIndex):
            frame = series.unstack(level=-1)
            frame.index = pd.to_datetime(frame.index)
            wide[col] = frame
        elif isinstance(series, pd.DataFrame):
            wide[col] = series
        else:
            raise RuntimeError(f"unexpected column shape for {col}: {type(series)}")
    symbols = sorted(set(map(str, wide["close"].columns)))
    chunks: list[tuple[list[str], dict[str, pd.DataFrame]]] = []
    for i in range(0, len(symbols), symbol_chunk):
        syms = symbols[i : i + symbol_chunk]
        chunk = {col: frame[syms].copy() for col, frame in wide.items()}
        chunks.append((syms, chunk))
    release_memory(ds, wide)
    return chunks


def _series_to_long(values: pd.Series, asset: str) -> pd.DataFrame:
    frame = values.rename("value").reset_index()
    frame.columns = ["datetime", "value"]
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["asset"] = _normalize_symbol(asset)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    return frame[["datetime", "asset", "value"]]


def materialize_python_factor(
    *,
    function_name: str,
    python_code: str,
    start: str,
    end: str,
    lake_root: Path,
    symbol_chunk: int = 400,
) -> Path:
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    exec(python_code, namespace)  # noqa: S102
    if function_name not in namespace:
        raise RuntimeError(f"{function_name} not found in python_code")
    func = namespace[function_name]

    parts: list[pd.DataFrame] = []
    for syms, panel in load_symbol_panel(start=start, end=end, symbol_chunk=symbol_chunk):
        for sym in syms:
            df = pd.DataFrame({col: panel[col][sym] for col in panel})
            df = df.reset_index().rename(columns={"index": "date"})
            if "date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "date"})
            out = func(df.copy())
            if isinstance(out, pd.DataFrame):
                out = out.iloc[:, 0]
            if not isinstance(out, pd.Series):
                raise RuntimeError(f"{function_name} returned {type(out)}")
            out.index = df["date"].values
            parts.append(_series_to_long(out, sym))
        release_memory(panel)

    if not parts:
        raise RuntimeError(f"{function_name} produced no rows")
    long_df = pd.concat(parts, ignore_index=True)
    out_dir = lake_root / function_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "values.parquet"
    long_df.to_parquet(out_path, index=False)
    release_memory(parts, long_df)
    return out_path

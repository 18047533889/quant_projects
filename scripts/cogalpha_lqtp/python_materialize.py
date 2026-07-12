#!/usr/bin/env python3
"""Materialize CogAlpha Python factors on wide A-share panel (chunked + vectorized prep)."""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import default_ashare_pv_data_source_config  # noqa: E402
from runtime.config import DataSourceConfig  # noqa: E402
from storage.factory import build_data_source  # noqa: E402

from scripts.cogalpha_lqtp.materialize import _normalize_symbol  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import release_memory  # noqa: E402
from scripts.cogalpha_lqtp.python_runtime import (  # noqa: E402
    build_python_exec_namespace,
    column_from_source,
    prepare_factor_dataframe,
)

PANEL_COLS = ["open", "high", "low", "close", "volume", "amount", "ret", "preclose", "vwap"]


def _build_data_source(
    *,
    start: str,
    end: str,
    instrument_filter: list[str] | None = None,
):
    cfg = default_ashare_pv_data_source_config(start_date=start, end_date=end)
    if instrument_filter:
        cfg["instrument_filter"] = sorted(instrument_filter)
    ds_type = str(cfg.pop("type", "data_access"))
    return build_data_source(DataSourceConfig(type=ds_type, options=cfg))


def load_symbol_panel(
    *,
    start: str,
    end: str,
    symbol_chunk: int = 400,
    allowed_symbols: set[str] | None = None,
    instrument_filter: list[str] | None = None,
) -> list[tuple[list[str], dict[str, pd.DataFrame]]]:
    """Return chunked symbol panels as {field: date x symbol DataFrame}."""
    ds = _build_data_source(start=start, end=end, instrument_filter=instrument_filter)
    if hasattr(ds, "load_columns"):
        ds.load_columns(PANEL_COLS)
    wide: dict[str, pd.DataFrame] = {}
    for col in PANEL_COLS:
        series = column_from_source(ds, col)
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
    if allowed_symbols:
        symbols = sorted(
            sym for sym in symbols if _normalize_symbol(sym) in allowed_symbols
        )
    if not symbols:
        raise RuntimeError("no symbols left after LQTP universe filter")
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
    frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["value"])
    return frame[["datetime", "asset", "value"]]


def _symbol_frame(sym: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.DataFrame({col: panel[col][sym] for col in panel})
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    return df


def _eval_one_symbol(
    func: Callable[..., Any],
    *,
    sym: str,
    panel: dict[str, pd.DataFrame],
    function_name: str,
) -> pd.DataFrame:
    df = _symbol_frame(sym, panel)
    out = func(prepare_factor_dataframe(df))
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    if not isinstance(out, pd.Series):
        raise RuntimeError(f"{function_name} returned {type(out)} for {sym}")
    out.index = df["date"].values
    return _series_to_long(out, sym)


def _eval_chunk_parallel(
    func: Callable[..., Any],
    *,
    syms: list[str],
    panel: dict[str, pd.DataFrame],
    function_name: str,
    workers: int = 4,
) -> pd.DataFrame:
    """Evaluate one symbol-chunk; uses thread pool (numpy/pandas releases GIL in rolling ops)."""
    if not syms:
        return pd.DataFrame(columns=["datetime", "asset", "value"])
    workers = max(1, min(workers, len(syms)))
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _eval_one_symbol,
                func,
                sym=sym,
                panel=panel,
                function_name=function_name,
            ): sym
            for sym in syms
        }
        for fut in as_completed(futures):
            part = fut.result()
            if not part.empty:
                frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["datetime", "asset", "value"])
    return pd.concat(frames, ignore_index=True)


def _merge_parquet_parts(part_paths: list[Path], out_path: Path) -> None:
    if not part_paths:
        raise RuntimeError("no parquet parts to merge")
    if len(part_paths) == 1:
        part_paths[0].replace(out_path)
        return
    import pyarrow.parquet as pq

    table = pq.read_table([str(p) for p in part_paths])
    pq.write_table(table, out_path, compression="snappy")
    for path in part_paths:
        path.unlink(missing_ok=True)


def materialize_python_factor(
    *,
    function_name: str,
    python_code: str,
    start: str,
    end: str,
    lake_root: Path,
    symbol_chunk: int = 400,
    workers: int = 4,
    allowed_symbols: set[str] | None = None,
) -> Path:
    namespace = build_python_exec_namespace()
    exec(python_code, namespace)  # noqa: S102
    if function_name not in namespace:
        raise RuntimeError(f"{function_name} not found in python_code")
    func = namespace[function_name]

    out_dir = lake_root / function_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "values.parquet"
    part_paths: list[Path] = []

    filter_list = sorted(allowed_symbols) if allowed_symbols else None
    for chunk_idx, (syms, panel) in enumerate(
        load_symbol_panel(
            start=start,
            end=end,
            symbol_chunk=symbol_chunk,
            allowed_symbols=allowed_symbols,
            instrument_filter=filter_list,
        )
    ):
        chunk_df = _eval_chunk_parallel(
            func,
            syms=syms,
            panel=panel,
            function_name=function_name,
            workers=workers,
        )
        release_memory(panel)
        if chunk_df.empty:
            continue
        part_path = out_dir / f"_part_{chunk_idx:04d}.parquet"
        chunk_df.to_parquet(part_path, index=False)
        part_paths.append(part_path)
        release_memory(chunk_df)

    if not part_paths:
        raise RuntimeError(f"{function_name} produced no rows")
    _merge_parquet_parts(part_paths, out_path)
    return out_path

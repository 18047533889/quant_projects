#!/usr/bin/env python3
"""Shared data_access / factor_engine panel helpers for cogalpha_lqtp."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

DEFAULT_INDUSTRY_SOURCE = "sw_l1"


def ashare_materialize_data_source_config(
    *,
    start_date: str,
    end_date: str,
    instrument_filter: list[str] | None = None,
    max_files: int | None = None,
    parquet_root: Path | None = None,
) -> dict[str, Any]:
    """A-share PV config aligned with factor_engine ``default_ashare_pv_data_source_config``."""
    from api.mining_integration import default_ashare_pv_data_source_config

    cfg = default_ashare_pv_data_source_config(
        max_files=max_files,
        start_date=start_date,
        end_date=end_date,
    )
    # Alphasage / LQTP formulas may reference limit prices and pre_close.
    extra_fields = {
        "preclose": "PreClose",
        "pre_close": "PreClose",
        "high_limit": "HighLimit",
        "low_limit": "LowLimit",
    }
    if cfg.get("type") == "data_access" and isinstance(cfg.get("fields"), dict):
        cfg["fields"] = {**cfg["fields"], **extra_fields}
    elif cfg.get("type") == "parquet" and isinstance(cfg.get("fields"), dict):
        cfg["fields"] = {**cfg["fields"], **extra_fields}
    if parquet_root is not None and max_files is not None:
        cfg["root"] = str(parquet_root)
    if instrument_filter:
        cfg["instrument_filter"] = sorted(instrument_filter)
    return cfg


def build_materialize_data_source(cfg: dict[str, Any]):
    from storage.factory import build_data_source

    return build_data_source(cfg)


def build_materialize_backend():
    from backend.factory import build_backend

    backend_type = os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "auto").strip() or "auto"
    return build_backend(backend_type)


def resolve_factor_values_parquet(work_dir: Path, factor_name: str) -> Path | None:
    """Locate factor values: cogalpha ``factor_lake`` or hive ``year=*/data.parquet``."""
    candidates = [
        work_dir / "factor_lake" / factor_name / "values.parquet",
        work_dir / "factors" / factor_name / "values.parquet",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    for root in (
        work_dir / "factor_lake" / factor_name,
        work_dir / "factors" / factor_name,
    ):
        if root.is_dir() and any(root.glob("year=*/data.parquet")):
            return root
    return None


def factor_parquet_read_spec(factor_path: Path) -> tuple[str, bool]:
    """Return (duckdb read_parquet path, hive_partitioning)."""
    if factor_path.is_file():
        return factor_path.as_posix().replace("'", "''"), False
    if factor_path.is_dir() and any(factor_path.glob("year=*/data.parquet")):
        glob_path = (factor_path / "year=*" / "data.parquet").as_posix().replace("'", "''")
        return glob_path, True
    raise FileNotFoundError(f"factor parquet not found: {factor_path}")


def factor_values_cte_sql(factor_path: Path) -> str:
    """DuckDB CTE for factor long table (trade_date, symbol, value)."""
    import pyarrow.parquet as pq

    read_path, hive = factor_parquet_read_spec(factor_path)
    hive_kw = ", hive_partitioning=true" if hive else ""
    if factor_path.is_dir():
        sample = next(factor_path.glob("year=*/data.parquet"), None)
        if sample is None:
            raise FileNotFoundError(factor_path)
        cols = set(pq.ParquetFile(sample).schema.names)
    else:
        cols = set(pq.ParquetFile(factor_path).schema.names)

    if "trade_date" in cols and "symbol" in cols:
        return f"""
        SELECT trade_date::INTEGER AS trade_date,
               symbol::VARCHAR AS symbol,
               value::DOUBLE AS value
        FROM read_parquet('{read_path}'{hive_kw}, union_by_name=true)
        """
    return f"""
    SELECT (year(datetime) * 10000 + month(datetime) * 100 + day(datetime))::INTEGER AS trade_date,
           asset::VARCHAR AS symbol,
           value::DOUBLE AS value
    FROM read_parquet('{read_path}'{hive_kw}, union_by_name=true)
    """


def build_close_returns_cache_data_access(
    out_path: Path,
    *,
    start: str,
    end: str,
    instrument_filter: list[str] | None = None,
) -> Path:
    """Build ``(trade_date, symbol, value)`` decimal close returns via data_access."""
    import duckdb

    from data_access import get_store

    store = get_store()
    kwargs: dict[str, Any] = {"time_range": (start[:10], end[:10])}
    if instrument_filter:
        kwargs["instrument_filter"] = list(instrument_filter)

    table = store.read_cos_panel(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol"],
        normalize_returns=True,
        return_output_column="value",
        **kwargs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("_ret_arrow", table)
        con.execute(
            f"""
            COPY (
              SELECT
                (year("TradeDate") * 10000 + month("TradeDate") * 100 + day("TradeDate"))::INTEGER AS trade_date,
                "Symbol"::VARCHAR AS symbol,
                value::DOUBLE AS value
              FROM _ret_arrow
              WHERE value IS NOT NULL
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()
    return out_path


def build_vwap_returns_cache_data_access(
    out_path: Path,
    *,
    start: str,
    end: str,
    instrument_filter: list[str] | None = None,
) -> Path:
    """Build daily VWAP→VWAP returns: ``Vwap_t / Vwap_{t-1} - 1`` labeled on day t."""
    import duckdb

    from data_access import get_store

    store = get_store()
    kwargs: dict[str, Any] = {"time_range": (start[:10], end[:10])}
    if instrument_filter:
        kwargs["instrument_filter"] = list(instrument_filter)

    table = store.read_cos_panel(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol", "Vwap"],
        **kwargs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("_vwap_arrow", table)
        con.execute(
            f"""
            COPY (
              WITH base AS (
                SELECT
                  (year("TradeDate") * 10000 + month("TradeDate") * 100 + day("TradeDate"))::INTEGER AS trade_date,
                  "Symbol"::VARCHAR AS symbol,
                  "Vwap"::DOUBLE AS vwap
                FROM _vwap_arrow
                WHERE "Vwap" IS NOT NULL AND "Vwap" > 0
              ),
              ret AS (
                SELECT
                  trade_date,
                  symbol,
                  vwap / lag(vwap) OVER (PARTITION BY symbol ORDER BY trade_date) - 1.0 AS value
                FROM base
              )
              SELECT trade_date, symbol, value
              FROM ret
              WHERE value IS NOT NULL AND isfinite(value)
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()
    return out_path


def build_eval_aux_parquets_data_access(
    *,
    industry_out: Path,
    mcap_out: Path,
    start: str,
    end: str,
    industry_source: str = DEFAULT_INDUSTRY_SOURCE,
) -> None:
    """Normalize industry + market cap panels through data_access (COS mirror handled by store)."""
    import duckdb

    from data_access import get_store

    store = get_store()
    time_range = (start[:10], end[:10])

    ind = store.read_cos_panel(
        "ashare_stock_industry",
        columns=["TradeDate", "Symbol", "IndustryCode"],
        time_range=time_range,
        semantic_filters={"IndustrySource": industry_source},
    )
    mcap = store.read_cos_panel(
        "ashare_stock_valuation_daily",
        columns=["TradeDate", "Symbol", "MarketCap"],
        time_range=time_range,
    )

    industry_out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("_ind", ind)
        con.execute(
            f"""
            COPY (
              SELECT
                (year("TradeDate") * 10000 + month("TradeDate") * 100 + day("TradeDate"))::INTEGER AS trade_date,
                "Symbol"::VARCHAR AS symbol,
                "IndustryCode"::VARCHAR AS industry_code
              FROM _ind
              WHERE "IndustryCode" IS NOT NULL
            ) TO '{industry_out.as_posix()}' (FORMAT PARQUET)
            """
        )
        con.register("_mcap", mcap)
        con.execute(
            f"""
            COPY (
              SELECT
                (year("TradeDate") * 10000 + month("TradeDate") * 100 + day("TradeDate"))::INTEGER AS trade_date,
                "Symbol"::VARCHAR AS symbol,
                ln(greatest("MarketCap", 1.0))::DOUBLE AS log_market_cap
              FROM _mcap
              WHERE "MarketCap" IS NOT NULL AND "MarketCap" > 0
            ) TO '{mcap_out.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()

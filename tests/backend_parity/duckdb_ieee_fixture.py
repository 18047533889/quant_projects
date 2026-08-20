# -*- coding: utf-8
"""DuckDB 原生 IEEE fixture：不经 Pandas/Parquet 将 NaN 写成 SQL NULL。"""
from __future__ import annotations

from pathlib import Path

import duckdb


def write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
    Open: double
    Ret: double
    Numer: double
    Denom: double
    Neg: double
    Flag: double
    Grp: int64
    Volume: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def seed_duckdb_ieee_panel(root: Path) -> None:
    """写入含 IEEE NaN/Inf 与 SQL NULL 的 parquet（DuckDB 原生 COPY）。"""
    root.mkdir(parents=True, exist_ok=True)
    out = (root / "ieee_panel.parquet").as_posix()
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            ('2024-01-02'::DATE, 'A'::VARCHAR, 42.0::DOUBLE, 1.0::DOUBLE, 0.01::DOUBLE,
             1.0::DOUBLE, 1.0::DOUBLE, -1.0::DOUBLE, 1.0::DOUBLE, 1::BIGINT, 100.0::DOUBLE),
            ('2024-01-02'::DATE, 'B'::VARCHAR, 'NaN'::DOUBLE, 2.0::DOUBLE, 0.02::DOUBLE,
             2.0::DOUBLE, 1.0::DOUBLE, -2.0::DOUBLE, 0.0::DOUBLE, 1::BIGINT, 110.0::DOUBLE),
            ('2024-01-02'::DATE, 'C'::VARCHAR, NULL::DOUBLE, 3.0::DOUBLE, 0.03::DOUBLE,
             3.0::DOUBLE, 1.0::DOUBLE, -3.0::DOUBLE, 1.0::DOUBLE, 1::BIGINT, 120.0::DOUBLE),
            ('2024-01-02'::DATE, 'D'::VARCHAR, 'Infinity'::DOUBLE, 4.0::DOUBLE, 0.04::DOUBLE,
             4.0::DOUBLE, 1.0::DOUBLE, 1.0::DOUBLE, 0.0::DOUBLE, 1::BIGINT, 130.0::DOUBLE),
            ('2024-01-02'::DATE, 'E'::VARCHAR, '-Infinity'::DOUBLE, 5.0::DOUBLE, 0.05::DOUBLE,
             5.0::DOUBLE, 1.0::DOUBLE, 2.0::DOUBLE, 1.0::DOUBLE, 1::BIGINT, 140.0::DOUBLE),
            ('2024-01-03'::DATE, 'A'::VARCHAR, NULL::DOUBLE, 6.0::DOUBLE, 0.06::DOUBLE,
             0.0::DOUBLE, 0.0::DOUBLE, 0.0::DOUBLE, 0.0::DOUBLE, 2::BIGINT, 0.0::DOUBLE),
            ('2024-01-03'::DATE, 'B'::VARCHAR, NULL::DOUBLE, 7.0::DOUBLE, 0.07::DOUBLE,
             0.0::DOUBLE, 0.0::DOUBLE, 0.0::DOUBLE, 0.0::DOUBLE, 2::BIGINT, 0.0::DOUBLE),
            ('2024-01-03'::DATE, 'C'::VARCHAR, NULL::DOUBLE, 8.0::DOUBLE, 0.08::DOUBLE,
             0.0::DOUBLE, 0.0::DOUBLE, 0.0::DOUBLE, 0.0::DOUBLE, 2::BIGINT, 0.0::DOUBLE)
          ) AS t(TradeDate, Symbol, Close, Open, Ret, Numer, Denom, Neg, Flag, Grp, Volume)
        ) TO '{out}' (FORMAT PARQUET)
        """
    )
    con.close()


def _sql_double(val) -> str:
    import pandas as pd

    if pd.isna(val):
        return "'NaN'::DOUBLE"
    return f"{float(val)}::DOUBLE"


def _sql_bigint(val) -> str:
    import pandas as pd

    if pd.isna(val):
        return "NULL::BIGINT"
    return f"{int(val)}::BIGINT"


def seed_duckdb_parquet_from_memory(
    root: Path,
    mem: "InMemorySeriesSource",
    *,
    mapping: dict[str, str],
    anchor: str = "close",
    int_columns: frozenset[str] | None = None,
) -> None:
    """DuckDB COPY 写 parquet，保留 IEEE NaN（pandas ``to_parquet`` 会把 NaN 写成 NULL）。"""
    import duckdb

    int_columns = int_columns or frozenset()
    root.mkdir(parents=True, exist_ok=True)
    out = (root / "panel.parquet").as_posix()
    cols = ["TradeDate", "Symbol"] + [mapping[k] for k in mapping]
    values_sql = []
    for (ts, sym) in mem.data[anchor].index:
        parts = [f"'{ts.date()}'::DATE", f"'{sym}'::VARCHAR"]
        for src in mapping:
            val = mem.data[src].loc[(ts, sym)]
            if src in int_columns:
                parts.append(_sql_bigint(val))
            else:
                parts.append(_sql_double(val))
        values_sql.append(f"({', '.join(parts)})")
    col_sql = ", ".join(cols)
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            {', '.join(values_sql)}
          ) AS t({col_sql})
        ) TO '{out}' (FORMAT PARQUET)
        """
    )
    con.close()

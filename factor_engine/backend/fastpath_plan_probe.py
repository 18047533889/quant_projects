# -*- coding: utf-8
"""Production fastpath full-plan 探测：typed schema + DuckDB 真实执行。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProbeDtype = Literal["float64", "utf8", "boolean", "int64", "datetime"]


@dataclass(frozen=True)
class ProbeColumnSpec:
    """单探测列类型。"""

    name: str
    dtype: ProbeDtype


class ProbeSchemaBuilder:
    """根据 plan 引用列推断探测 schema（非全 Float64）。"""

    _UTF8_HINTS: frozenset[str] = frozenset(
        {"group", "industry", "sector", "symbol", "instrument", "country", "exchange"}
    )
    _BOOL_HINTS: frozenset[str] = frozenset({"flag", "mask", "condition", "is_valid", "is_tradable"})
    _INT_HINTS: frozenset[str] = frozenset(
        {"fiscal_quarter", "fiscal_period", "period", "window", "count", "rank_int"}
    )
    _DATETIME_HINTS: frozenset[str] = frozenset({"ts", "timestamp", "trade_date", "date", "datetime"})

    @classmethod
    def infer_column_dtype(cls, name: str) -> ProbeDtype:
        """按列名启发式推断探测 dtype。"""
        key = str(name or "").strip().lower()
        if key in cls._DATETIME_HINTS:
            return "datetime"
        if any(h in key for h in cls._BOOL_HINTS) or key.startswith("is_"):
            return "boolean"
        if any(h in key for h in cls._INT_HINTS):
            return "int64"
        if any(h in key for h in cls._UTF8_HINTS):
            return "utf8"
        return "float64"

    @classmethod
    def from_plan(cls, plan: Any, *, time_column: str = "ts", instrument_column: str = "inst") -> list[ProbeColumnSpec]:
        """收集 plan 引用列并推断类型。"""
        from backend.polars_expr_emitter import collect_columns

        cols = sorted(collect_columns(plan))
        specs: list[ProbeColumnSpec] = []
        for name in cols:
            if name in {time_column, instrument_column}:
                continue
            specs.append(ProbeColumnSpec(name=name, dtype=cls.infer_column_dtype(name)))
        return specs

    @classmethod
    def build_polars_probe_frame(
        cls,
        plan: Any,
        *,
        time_column: str = "ts",
        instrument_column: str = "inst",
        n_rows: int = 4,
    ) -> Any:
        """构造 typed Polars LazyFrame 供 full-plan compile/collect 探测。"""
        import polars as pl

        specs = cls.from_plan(plan, time_column=time_column, instrument_column=instrument_column)
        ts_values = [f"2024-01-{d + 2:02d}T00:00:00" for d in range(n_rows)]
        data: dict[str, pl.Series] = {
            time_column: pl.Series(ts_values, dtype=pl.Datetime(time_unit="ns")),
            instrument_column: pl.Series(["A"] * n_rows, dtype=pl.Utf8),
        }
        for spec in specs:
            if spec.dtype == "utf8":
                data[spec.name] = pl.Series(["G1"] * n_rows, dtype=pl.Utf8)
            elif spec.dtype == "boolean":
                data[spec.name] = pl.Series([True, False, True, False][:n_rows], dtype=pl.Boolean)
            elif spec.dtype == "int64":
                data[spec.name] = pl.Series([1, 2, 3, 4][:n_rows], dtype=pl.Int64)
            elif spec.dtype == "datetime":
                data[spec.name] = pl.Series(ts_values, dtype=pl.Datetime(time_unit="ns"))
            else:
                data[spec.name] = pl.Series([1.0, 2.0, 3.0, 4.0][:n_rows], dtype=pl.Float64)
        return pl.LazyFrame(data)

    @classmethod
    def build_duckdb_probe_table(
        cls,
        plan: Any,
        *,
        time_column: str = "ts",
        instrument_column: str = "inst",
        n_rows: int = 4,
    ) -> Any:
        """构造 DuckDB 内存探测表（Arrow/pandas）。"""
        import pandas as pd

        specs = cls.from_plan(plan, time_column=time_column, instrument_column=instrument_column)
        rows: list[dict[str, Any]] = []
        for i in range(n_rows):
            row: dict[str, Any] = {
                time_column: pd.Timestamp(f"2024-01-{i + 2:02d}"),
                instrument_column: "A",
            }
            for spec in specs:
                if spec.dtype == "utf8":
                    row[spec.name] = "G1"
                elif spec.dtype == "boolean":
                    row[spec.name] = bool(i % 2 == 0)
                elif spec.dtype == "int64":
                    row[spec.name] = i + 1
                elif spec.dtype == "datetime":
                    row[spec.name] = pd.Timestamp(f"2024-01-{i + 2:02d}")
                else:
                    row[spec.name] = float(i + 1)
            rows.append(row)
        return pd.DataFrame(rows)


def execute_duckdb_probe_sql(
    compiled: Any,
    probe_table: Any,
    *,
    dataset: str = "__fastpath_probe__",
) -> None:
    """在内存 DuckDB 上执行完整 compiled SQL（非仅 emit）。"""
    import duckdb

    con = duckdb.connect()
    try:
        con.register(dataset, probe_table)
        sql = str(compiled.query or "").replace(f"{{{{{dataset}}}}}", dataset)
        table = con.execute(sql).to_arrow_table()
        if table.num_rows <= 0:
            raise RuntimeError("duckdb probe returned zero rows")
        if "value" not in table.column_names:
            raise RuntimeError(f"duckdb probe missing value column: {table.column_names}")
    finally:
        con.close()


def probe_polars_full_plan(plan: Any) -> None:
    """PolarsLong full-plan typed compile + collect 探测。"""
    from backend.polars_expr_emitter import compile_plan_to_polars, plan_is_polars_long_capable

    if not plan_is_polars_long_capable(plan):
        raise RuntimeError("polars_long plan not capable")
    probe = ProbeSchemaBuilder.build_polars_probe_frame(plan)
    compiled = compile_plan_to_polars(plan, probe, ctx=None)
    if compiled is None:
        raise RuntimeError("polars_long compile returned None")
    compiled.frame.collect_schema()
    out = compiled.frame.collect()
    if out.height <= 0:
        raise RuntimeError("polars_long probe returned zero rows")


def probe_duckdb_full_plan(
    plan: Any,
    *,
    dataset: str = "__fastpath_probe__",
    time_column: str = "ts",
    instrument_column: str = "inst",
) -> None:
    """DuckDB full-plan compile + 内存执行探测。"""
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql, plan_is_sql_capable

    if not plan_is_sql_capable(plan):
        raise RuntimeError("duckdb plan not sql capable")
    compiled = compile_plan_to_sql(
        plan,
        dataset=dataset,
        time_column=time_column,
        instrument_column=instrument_column,
        dialect=SqlDialect.DUCKDB,
    )
    if compiled is None or not str(compiled.query or "").strip():
        raise RuntimeError("duckdb compile returned empty")
    probe_table = ProbeSchemaBuilder.build_duckdb_probe_table(
        plan,
        time_column=time_column,
        instrument_column=instrument_column,
    )
    execute_duckdb_probe_sql(compiled, probe_table, dataset=dataset)

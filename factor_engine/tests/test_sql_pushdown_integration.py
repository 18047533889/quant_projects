# -*- coding: utf-8
"""SQL 下推端到端：DuckDB 内算 vs Pandas 对齐。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.factory import build_data_source


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.delenv("DATA_ACCESS_CONFIG", raising=False)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    yield
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass


def _write_registry(tmp_path: Path, root: Path) -> Path:
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
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_data(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=5, freq="D"):
        for sym, base in [("A", 10.0), ("B", 20.0)]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": base + d.day,
                    "Open": base + d.day - 0.5,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _write_group_registry(tmp_path: Path, root: Path) -> Path:
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
    Grp: int64
"""
    path = tmp_path / "datasets_group.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_group_data(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=5, freq="D"):
        for sym, base, grp in [("A", 10.0, 1), ("B", 20.0, 1), ("C", 30.0, 2)]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": base + d.day,
                    "Grp": grp,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet")


def test_duckdb_sql_pushdown_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    expr = rank(ts_mean(col("Close"), 2))
    factor = Factor(name="t", expr=expr)

    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)


def test_hybrid_backend_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")

    source = build_data_source({"type": "data_access", "dataset": "test_daily"})
    expr = rank(ts_mean(col("Close"), 2))
    factor = Factor(name="t", expr=expr)

    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_h = FactorEngine(backend=build_backend("auto"), data_source=source)

    pd.testing.assert_series_equal(
        eng_pd.run(factor)["result"],
        eng_h.run(factor)["result"],
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_ts_corr_pushdown_matches_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=6, freq="D"):
        for sym in ["A", "B"]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": float(d.day + (1 if sym == "A" else 2)),
                    "Open": float(d.day),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")

    from api import ts_corr
    from api.columns import col

    source = build_data_source({"type": "data_access", "dataset": "test_daily"})
    expr = ts_corr(col("Close"), col("Open"), 3)
    factor = Factor(name="t", expr=expr)

    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)

    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_tier2_ffill_and_decay_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    # 注入 NaN 以验证 ffill
    import pyarrow.parquet as pq

    table = pq.read_table(root / "panel.parquet")
    df = table.to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import ts_decay_linear, ffill, bfill

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )

    for expr, name in (
        (ts_decay_linear(col("Close"), 3), "decay"),
        (ffill(col("Close")), "ffill"),
        (bfill(col("Close")), "bfill"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_rank_ffill_with_nulls_match_pandas(tmp_path, monkeypatch):
    """rank 在 SQL 中应跳过 NaN（对齐 pandas 截面 rank）。"""
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import rank, ffill

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    factor = Factor(name="rank_ffill", expr=rank(ffill(col("Close"))))
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_group_rank_and_zscore_match_pandas(tmp_path, monkeypatch):
    """group_rank / group_zscore 在含 NaN 与单元素组时应与 pandas 对齐。"""
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG",
        str(_write_group_registry(tmp_path, tmp_path / "data")),
    )
    _seed_group_data(tmp_path / "data")

    from api import group_rank, group_zscore, group_normalize, group_percentile, group_decay_linear

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    for expr, name in (
        (group_rank(col("Close"), col("Grp")), "group_rank"),
        (group_zscore(col("Close"), col("Grp")), "group_zscore"),
        (group_normalize(col("Close"), col("Grp")), "group_normalize"),
        (group_percentile(col("Close"), col("Grp"), 0.5), "group_percentile"),
        (group_decay_linear(col("Close"), col("Grp"), 5), "group_decay_linear"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(
            a, b, check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6
        )


def test_sql_coalesce_ffill_open_match_pandas(tmp_path, monkeypatch):
    """coalesce 应优先取 ffill(Close)，缺失时回退 Open。"""
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import coalesce, ffill, rank

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    factor = Factor(
        name="coalesce_rank",
        expr=rank(coalesce(ffill(col("Close")), col("Open"))),
    )
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_winsorize_and_protected_div_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.loc[(df["Symbol"] == "B") & (df["TradeDate"] == df["TradeDate"].iloc[0]), "Open"] = 0.0
    df.to_parquet(root / "panel.parquet", index=False)

    from api import protected_div, winsorize

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    for expr, name in (
        (winsorize(col("Close"), 0.05), "winsorize"),
        (protected_div(col("Close"), col("Open")), "protected_div"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_clip_and_protected_math_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[0]), "Close"] = float("nan")
    df.loc[(df["Symbol"] == "B") & (df["TradeDate"] == df["TradeDate"].iloc[0]), "Close"] = -1.0
    df.to_parquet(root / "panel.parquet", index=False)

    from api import clip, protected_log, protected_sqrt

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    for expr, name in (
        (clip(col("Close"), -2, 2), "clip"),
        (protected_log(col("Close")), "protected_log"),
        (protected_sqrt(col("Close")), "protected_sqrt"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_nan_to_num_rank_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import nan_to_num, rank

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    factor = Factor(name="nan_rank", expr=rank(nan_to_num(col("Close"), 0)))
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)


def test_sql_fillna_is_nan_scale_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import fillna, is_nan, is_finite, scale, rank

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    for expr, name in (
        (fillna(col("Close"), 0), "fillna0"),
        (fillna(col("Close"), "zero"), "fillna_zero"),
        (is_nan(col("Close")), "is_nan"),
        (is_finite(col("Close")), "is_finite"),
        (rank(scale(col("Close"))), "scale_rank"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(
            a, b, check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6
        )


def test_sql_where_is_finite_and_cs_demean_pipeline(tmp_path, monkeypatch):
    """where(is_finite) + cs_demean + rank 全 SQL 链路对齐 pandas。"""
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import where, is_finite, cs_demean, rank

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    zero = col("Close") * 0
    for expr, name in (
        (where(is_finite(col("Close")), col("Close"), zero), "where_finite"),
        (cs_demean(col("Close")), "cs_demean"),
        (rank(cs_demean(col("Close"))), "rank_demean"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(
            a, b, check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6
        )


def test_sql_normalize_and_standardize_match_pandas(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    root = tmp_path / "data"
    _seed_data(root)
    import pyarrow.parquet as pq

    df = pq.read_table(root / "panel.parquet").to_pandas()
    df.loc[(df["Symbol"] == "A") & (df["TradeDate"] == df["TradeDate"].iloc[1]), "Close"] = float("nan")
    df.to_parquet(root / "panel.parquet", index=False)

    from api import normalize, standardize, if_else, is_finite, rank

    source = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "start_date": "2024-01-02",
            "end_date": "2024-01-06",
        }
    )
    zero = col("Close") * 0
    for expr, name in (
        (normalize(col("Close")), "normalize"),
        (standardize(col("Close")), "standardize"),
        (if_else(is_finite(col("Close")), col("Close"), zero), "if_else_finite"),
        (rank(normalize(col("Close"))), "rank_normalize"),
    ):
        factor = Factor(name=name, expr=expr)
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
        a = eng_pd.run(factor)["result"]
        b = eng_sql.run(factor)["result"]
        pd.testing.assert_series_equal(
            a, b, check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6
        )


def test_clickhouse_dialect_emitter():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
    from planner.logical_plan import PlanNode

    plan = PlanNode(op="ts_mean", inputs=[PlanNode(op="column", attrs={"name": "close"})], attrs={"d": 3})
    compiled = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert compiled is not None
    assert "panel_daily" in compiled.query

    from backend.sql_pushdown.emitter import (
        SqlDialect,
        SqlPushdownFilter,
        compile_plan_to_sql,
    )
    from planner.logical_plan import PlanNode

    plan = PlanNode(op="column", attrs={"name": "Close"})
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_daily",
        time_column="TradeDate",
        instrument_column="Symbol",
        filt=SqlPushdownFilter(
            time_column="TradeDate",
            start="2024-01-01",
            end="2024-12-31",
            instrument_column="Symbol",
            instruments=("A", "B"),
        ),
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert '"TradeDate" >=' in compiled.query
    assert '"Symbol" IN' in compiled.query

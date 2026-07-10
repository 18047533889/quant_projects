# -*- coding: utf-8
"""P0 语义 edge parity：ties rank、定义域、NaN/Inf、逻辑 NULL、三后端 DuckDB。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.numeric_semantics import (
    normalize_constant_cross_section_fill,
    normalize_single_valid_is_null,
    nan_to_num_replaces_infinite,
)
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def edge_source():
    load_all()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    insts = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    close = pd.Series(
        [
            1.0,
            1.0,
            2.0,
            np.nan,
            -1.0,
            0.0,
            np.nan,
            4.0,
        ],
        index=idx,
    )
    exp = pd.Series([2.0, 2.5, -1.0, 0.5, 1.0, 1.0, 3.0, 0.5], index=idx)
    grp = pd.Series([1, 1, 2, 2] * len(dates), index=idx, dtype=float)
    flag = pd.Series([1.0, np.nan, 0.0, 1.0, 0.0, np.nan, 1.0, 0.0], index=idx)
    inf_val = pd.Series([np.inf, -np.inf, 1.0, np.nan] * len(dates), index=idx)
    return InMemorySeriesSource(
        data={"close": close, "exp": exp, "grp": grp, "flag": flag, "inf_val": inf_val}
    )


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
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
    Exp: double
    Grp: int64
    Flag: double
    InfVal: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "close": "Close",
        "exp": "Exp",
        "grp": "Grp",
        "flag": "Flag",
        "inf_val": "InfVal",
    }
    rows = []
    for (ts, sym) in mem.data["close"].index:
        row = {"TradeDate": ts.date(), "Symbol": sym}
        for src, dst in mapping.items():
            val = mem.data[src].loc[(ts, sym)]
            row[dst] = float(val) if pd.notna(val) else None
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture(scope="module")
def duckdb_edge_source(tmp_path_factory, edge_source):
    tmp = tmp_path_factory.mktemp("duckdb_semantic_edges")
    import os

    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
    reg = _write_duckdb_registry(tmp, tmp / "data")
    os.environ["DATA_ACCESS_CONFIG"] = str(reg)
    _seed_duckdb(tmp / "data", edge_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _col(name: str):
    return col(
        {
            "close": "Close",
            "exp": "Exp",
            "grp": "Grp",
            "flag": "Flag",
            "inf_val": "InfVal",
        }.get(name, name)
    )


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


F = make_cleaned_call_factory


def _assert_triple_backends(
    mem_source,
    duckdb_source,
    mem_expr_builder,
    duckdb_expr_builder=None,
    *,
    rtol=1e-5,
    atol=1e-5,
):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    mem_expr = mem_expr_builder()
    duck_expr = (duckdb_expr_builder or mem_expr_builder)()
    pd_out = _result_series(_run(mem_source, mem_expr, "pandas"))
    long_out = _result_series(_run(mem_source, mem_expr, "polars_long"))
    sql_run = _run(duckdb_source, duck_expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=rtol, atol=atol)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=rtol, atol=atol)


def test_rank_pct_tie_average(edge_source, duckdb_edge_source):
    out = _result_series(_run(edge_source, F("rank_pct")(col("close")), "pandas"))
    day1 = out.loc[pd.Timestamp("2024-01-02")]
    assert day1.loc["A"] == pytest.approx(0.5)
    assert day1.loc["B"] == pytest.approx(0.5)
    assert day1.loc["C"] == pytest.approx(1.0)
    assert math.isnan(day1.loc["D"])
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("rank_pct")(col("close")),
        lambda: F("rank_pct")(_col("close")),
    )


def test_group_rank_tie_average(edge_source, duckdb_edge_source):
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("group_rank")(col("close"), col("grp")),
        lambda: F("group_rank")(_col("close"), _col("grp")),
    )


def test_log_sqrt_power_domain(edge_source, duckdb_edge_source):
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("log")(col("close")),
        lambda: F("log")(_col("close")),
    )
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("sqrt")(col("close")),
        lambda: F("sqrt")(_col("close")),
    )
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("power")(col("close"), col("exp")),
        lambda: F("power")(_col("close"), _col("exp")),
    )


def test_is_nan_is_finite_nan_to_num(edge_source, duckdb_edge_source):
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("is_nan")(col("close")),
        lambda: F("is_nan")(_col("close")),
    )
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("is_finite")(col("close")),
        lambda: F("is_finite")(_col("close")),
    )
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("nan_to_num")(col("close")),
        lambda: F("nan_to_num")(_col("close")),
    )


def test_nan_to_num_replaces_infinite(edge_source, duckdb_edge_source):
    assert nan_to_num_replaces_infinite()
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("nan_to_num")(col("inf_val"), 0.0),
        lambda: F("nan_to_num")(_col("inf_val"), 0.0),
    )


def test_and_or_null_as_false(edge_source, duckdb_edge_source):
    """NULL 在 and_/or_ 中视为 false（非 SQL 三值 NULL）。"""
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("and_")(col("flag"), col("close")),
        lambda: F("and_")(_col("flag"), _col("close")),
    )
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("or_")(col("flag"), col("close")),
        lambda: F("or_")(_col("flag"), _col("close")),
    )
    pd_out = _result_series(
        _run(edge_source, F("and_")(col("flag"), col("close")), "pandas")
    )
    assert pd_out.loc[(pd.Timestamp("2024-01-02"), "B")] == 0.0


def test_normalize_constant_and_single_valid(edge_source, duckdb_edge_source, monkeypatch, tmp_path):
    assert normalize_single_valid_is_null()
    assert normalize_constant_cross_section_fill() == 0.5
    _assert_triple_backends(
        edge_source,
        duckdb_edge_source,
        lambda: F("normalize")(col("grp")),
        lambda: F("normalize")(_col("grp")),
    )
    idx = edge_source.data["close"].index
    single_ts = idx[0][0]
    mask = idx.get_level_values("timestamp") == single_ts
    sub_idx = idx[mask][:1]
    one = pd.Series([42.0], index=sub_idx)
    mem_src = InMemorySeriesSource(data={"close": one})
    expr = F("normalize")(col("close"))
    pd_out = _result_series(_run(mem_src, expr, "pandas")).iloc[0]
    long_out = _result_series(_run(mem_src, expr, "polars_long")).iloc[0]
    assert math.isnan(pd_out)
    assert math.isnan(long_out)
    # DuckDB 单有效值截面
    root = tmp_path / "single_data"
    root.mkdir()
    pd.DataFrame(
        [{"TradeDate": single_ts.date(), "Symbol": sub_idx[0][1], "Close": 42.0}]
    ).to_parquet(root / "single.parquet")
    reg = tmp_path / "single.yaml"
    reg.write_text(
        f"""
test_single:
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
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(reg))
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    duck_src = build_data_source({"type": "data_access", "dataset": "test_single"})
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    sql_run = _run(duck_src, F("normalize")(col("Close")), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run).iloc[0]
    assert math.isnan(sql_out)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass


def test_truthy_null_semantics_contract():
    from backend.numeric_semantics import truthy_null_is_false, log_zero_returns_negative_infinity

    assert truthy_null_is_false()
    assert log_zero_returns_negative_infinity()


def test_bfill_polars_long_raises(edge_source):
    from backend.polars_long_policy import UnsupportedCausalOperatorError

    expr = F("bfill")(col("close"))
    with pytest.raises(UnsupportedCausalOperatorError):
        FactorEngine(backend=build_backend("polars_long"), data_source=edge_source).run(
            Factor(name="t", expr=expr)
        )


def test_bfill_duckdb_raises(edge_source):
    from backend.polars_long_policy import UnsupportedCausalOperatorError

    expr = F("bfill")(col("close"))
    with pytest.raises(UnsupportedCausalOperatorError):
        FactorEngine(backend=build_backend("duckdb_sql"), data_source=edge_source).run(
            Factor(name="t", expr=expr)
        )

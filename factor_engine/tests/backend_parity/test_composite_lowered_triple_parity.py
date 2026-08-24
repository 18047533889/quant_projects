# -*- coding: utf-8
"""Composite lowering 后三后端 parity（Pandas / PolarsLong native / DuckDB SQL）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory

# CI evidence contract 同步源（须与 evidence/composite_verified.json 一致）
COMPOSITE_TRIPLE_PARITY_CASES: tuple[dict, ...] = (
    {"canon": "MOM", "mem": lambda: F("MOM")(_col_mem("close"), 3), "duck": lambda: F("MOM")(_col_duck("close"), 3)},
    {"canon": "ROC", "mem": lambda: F("ROC")(_col_mem("close"), 3), "duck": lambda: F("ROC")(_col_duck("close"), 3)},
    {
        "canon": "BollingerBands",
        "mem": lambda: F("BollingerBands")(_col_mem("close"), 3),
        "duck": lambda: F("BollingerBands")(_col_duck("close"), 3),
    },
    {
        "canon": "BollingerUpper",
        "mem": lambda: F("BollingerUpper")(_col_mem("close"), 3),
        "duck": lambda: F("BollingerUpper")(_col_duck("close"), 3),
    },
    {
        "canon": "BollingerLower",
        "mem": lambda: F("BollingerLower")(_col_mem("close"), 3),
        "duck": lambda: F("BollingerLower")(_col_duck("close"), 3),
    },
    {"canon": "DPO", "mem": lambda: F("DPO")(_col_mem("close"), 4), "duck": lambda: F("DPO")(_col_duck("close"), 4)},
    {
        "canon": "WilliamsR",
        "mem": lambda: F("WilliamsR")(_col_mem("high"), _col_mem("low"), _col_mem("close"), 3),
        "duck": lambda: F("WilliamsR")(_col_duck("high"), _col_duck("low"), _col_duck("close"), 3),
    },
    {
        "canon": "StochasticK",
        "mem": lambda: F("StochasticK")(_col_mem("high"), _col_mem("low"), _col_mem("close"), 3),
        "duck": lambda: F("StochasticK")(_col_duck("high"), _col_duck("low"), _col_duck("close"), 3),
    },
    {
        "canon": "StochasticD",
        "mem": lambda: F("StochasticD")(_col_mem("high"), _col_mem("low"), _col_mem("close"), 3),
        "duck": lambda: F("StochasticD")(_col_duck("high"), _col_duck("low"), _col_duck("close"), 3),
    },
    {
        "canon": "OBV",
        "mem": lambda: F("OBV")(_col_mem("close"), _col_mem("volume")),
        "duck": lambda: F("OBV")(_col_duck("close"), _col_duck("volume")),
    },
    {
        "canon": "operating_margin",
        "mem": lambda: F("operating_margin")(_col_mem("operating_income"), _col_mem("revenue")),
        "duck": lambda: F("operating_margin")(_col_duck("operating_income"), _col_duck("revenue")),
    },
    {
        "canon": "current_ratio",
        "mem": lambda: F("current_ratio")(_col_mem("current_assets"), _col_mem("current_liabilities")),
        "duck": lambda: F("current_ratio")(_col_duck("current_assets"), _col_duck("current_liabilities")),
    },
    {
        "canon": "quick_ratio",
        "mem": lambda: F("quick_ratio")(
            _col_mem("current_assets"), _col_mem("inventory"), _col_mem("current_liabilities")
        ),
        "duck": lambda: F("quick_ratio")(
            _col_duck("current_assets"), _col_duck("inventory"), _col_duck("current_liabilities")
        ),
    },
    {
        "canon": "debt_to_equity",
        "mem": lambda: F("debt_to_equity")(_col_mem("total_debt"), _col_mem("total_equity")),
        "duck": lambda: F("debt_to_equity")(_col_duck("total_debt"), _col_duck("total_equity")),
    },
    {
        "canon": "real_turnover_rate",
        "mem": lambda: F("real_turnover_rate")(_col_mem("volume"), _col_mem("float_shares")),
        "duck": lambda: F("real_turnover_rate")(_col_duck("volume"), _col_duck("float_shares")),
    },
    {
        "canon": "micro_spread",
        "mem": lambda: F("micro_spread")(_col_mem("high"), _col_mem("low"), _col_mem("close")),
        "duck": lambda: F("micro_spread")(_col_duck("high"), _col_duck("low"), _col_duck("close")),
    },
    {
        "canon": "ts_ratio",
        "mem": lambda: F("ts_ratio")(_col_mem("close")),
        "duck": lambda: F("ts_ratio")(_col_duck("close")),
    },
)


def _col_mem(name: str):
    return col(name)


def _col_duck(name: str):
    return col(
        {
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
            "operating_income": "OperatingIncome",
            "revenue": "Revenue",
            "current_assets": "CurrentAssets",
            "current_liabilities": "CurrentLiabilities",
            "inventory": "Inventory",
            "total_debt": "TotalDebt",
            "total_equity": "TotalEquity",
            "float_shares": "FloatShares",
        }.get(name, name)
    )


@pytest.fixture(scope="module")
def composite_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 19.0, 21.0, 22.0], index=idx)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 210.0, 190.0, 220.0], index=idx)
    return InMemorySeriesSource(
        data={
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "operating_income": pd.Series([100.0, 110.0, 0.0, 120.0, 200.0, 180.0, 210.0, 220.0], index=idx),
            "revenue": pd.Series([200.0, 220.0, 110.0, 240.0, 400.0, 0.0, 420.0, 440.0], index=idx),
            "current_assets": pd.Series([50.0, 55.0, 52.0, 60.0, 80.0, 78.0, 82.0, 85.0], index=idx),
            "current_liabilities": pd.Series([25.0, 0.0, 26.0, 30.0, 40.0, 41.0, 0.0, 42.0], index=idx),
            "inventory": pd.Series([5.0, 5.5, 5.2, 6.0, 8.0, 7.8, 8.2, 8.5], index=idx),
            "total_debt": pd.Series([30.0, 31.0, 32.0, 33.0, 60.0, 61.0, 62.0, 63.0], index=idx),
            "total_equity": pd.Series([70.0, 0.0, 72.0, 73.0, 140.0, 141.0, 142.0, 143.0], index=idx),
            "float_shares": pd.Series([1e6, 1e6, 0.0, 1e6, 2e6, 2e6, 2e6, 2e6], index=idx),
        }
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
    High: double
    Low: double
    Volume: double
    OperatingIncome: double
    Revenue: double
    CurrentAssets: double
    CurrentLiabilities: double
    Inventory: double
    TotalDebt: double
    TotalEquity: double
    FloatShares: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "operating_income": "OperatingIncome",
        "revenue": "Revenue",
        "current_assets": "CurrentAssets",
        "current_liabilities": "CurrentLiabilities",
        "inventory": "Inventory",
        "total_debt": "TotalDebt",
        "total_equity": "TotalEquity",
        "float_shares": "FloatShares",
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
def duckdb_composite_source(tmp_path_factory, composite_source):
    tmp = tmp_path_factory.mktemp("duckdb_composite")
    import os

    old_config = os.environ.get("DATA_ACCESS_CONFIG")
    old_skip = os.environ.get("DATA_ACCESS_SKIP_COS_MIRROR")
    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
    reg = _write_duckdb_registry(tmp, tmp / "data")
    os.environ["DATA_ACCESS_CONFIG"] = str(reg)
    _seed_duckdb(tmp / "data", composite_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    yield build_data_source({"type": "data_access", "dataset": "test_daily"})
    if old_config is None:
        os.environ.pop("DATA_ACCESS_CONFIG", None)
    else:
        os.environ["DATA_ACCESS_CONFIG"] = old_config
    if old_skip is None:
        os.environ.pop("DATA_ACCESS_SKIP_COS_MIRROR", None)
    else:
        os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = old_skip


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _assert_lowered_triple(
    mem_source,
    duckdb_source,
    formula_mem,
    formula_duck=None,
    *,
    rtol=1e-5,
    atol=1e-5,
):
    duck_formula = formula_duck or formula_mem
    pd_out = _run(mem_source, formula_mem, "pandas")["result"].sort_index()
    long_run = _run(mem_source, formula_mem, "polars_long")
    long_out = long_run["result"].sort_index()
    duck_run = _run(duckdb_source, duck_formula, "duckdb_sql")
    duck_out = duck_run["result"].sort_index()

    assert long_run.get("used_polars_long_native") is True, long_run
    assert long_run.get("used_polars_long_registry") is not True, long_run
    assert_duckdb_real_sql_execution(duck_run)

    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=rtol, atol=atol)
    pd.testing.assert_series_equal(pd_out, duck_out, check_names=False, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", COMPOSITE_TRIPLE_PARITY_CASES, ids=lambda c: c["canon"])
def test_composite_lowered_triple_parity(composite_source, duckdb_composite_source, case):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if case["canon"] not in OperatorRegistry._operators:
        pytest.skip("migrated to recipe or research layer")
    _assert_lowered_triple(
        composite_source,
        duckdb_composite_source,
        case["mem"](),
        case["duck"](),
    )

# -*- coding: utf-8 -*-
"""EWMA/Wilder 族 SQL 精确 parity：真实 DuckDB 执行 vs pandas 参考 (NaN 缺口)。

覆盖重新启用的 27 个 canonical：RSI_WILDER / ATR_WILDER / DMI_plus / DMI_minus /
DX / ADX / MACD_line / MACD_signal / MACD_hist / DEMA / TEMA / PPO / PPO_signal /
PPO_hist / PVO / PVO_signal / PVO_hist / TSI / TSI_signal / KeltnerMid /
KeltnerUpper / KeltnerLower / KeltnerPosition / ADL / ChaikinOscillator / CMF /
ForceIndex。每个算子用 3 个 instrument（含 NaN 缺口、leading NaN、常量序列），
以 ``duckdb_sql`` backend 真实执行，要求 ``used_sql_pushdown`` 且逐点 == pandas。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")
pytest.importorskip("duckdb")

from pathlib import Path

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from factor_engine.cleaned_operators.registry import OperatorRegistry
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource

# Ordered instruments so MultiIndex is aligned with the per-instrument input.
INSTRUMENTS = ["A", "B", "C"]

INPUTS = {
    "A": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 6.0],
    "B": [np.nan, 1.0, 2.0, 4.0, np.nan, 8.0, 16.0, np.nan],
    "C": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
}

DATE0 = "2024-01-02"
N = 8


def _panel_source():
    load_all()
    dates = pd.date_range(DATE0, periods=N, freq="B")
    idx = pd.MultiIndex.from_product([dates, INSTRUMENTS], names=["timestamp", "instrument"])
    close = pd.Series(
        [INPUTS[s][t] for t in range(N) for s in INSTRUMENTS], index=idx, dtype=float
    )
    high = close + 0.3
    low = close - 0.3
    volume = pd.Series(
        [100.0 + t * 10 + (j + 1) * 7 for t in range(N) for j in range(3)], index=idx
    )
    return InMemorySeriesSource(
        data={"close": close, "high": high, "low": low, "volume": volume}
    )


@pytest.fixture(scope="module")
def panel():
    return _panel_source()


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
wilder_test:
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
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb_panel(root: Path, panel: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    close = panel.data["close"]
    high = panel.data["high"]
    low = panel.data["low"]
    volume = panel.data["volume"]
    for (ts, sym) in close.index:
        rows.append({
            "TradeDate": ts.date(),
            "Symbol": sym,
            "Close": None if pd.isna(close.loc[(ts, sym)]) else float(close.loc[(ts, sym)]),
            "High": None if pd.isna(high.loc[(ts, sym)]) else float(high.loc[(ts, sym)]),
            "Low": None if pd.isna(low.loc[(ts, sym)]) else float(low.loc[(ts, sym)]),
            "Volume": None if pd.isna(volume.loc[(ts, sym)]) else float(volume.loc[(ts, sym)]),
        })
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, panel):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data"))
    )
    _seed_duckdb_panel(tmp_path / "data", panel)
    try:
        from data_access import reset_store
        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "wilder_test"})


def _cols():
    return {"close": "Close", "high": "High", "low": "Low", "volume": "Volume"}


def _C(name: str):
    """Map memory column names to DuckDB schema (Close/High/Low/Volume)."""
    return col(_cols()[name])


def _PC(name: str):
    return col(name)


def _run(source, expr, backend_name: str) -> dict:
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="wilder", expr=expr))


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _input_series(name: str) -> pd.Series:
    dates = pd.date_range(DATE0, periods=N, freq="B")
    idx = pd.MultiIndex.from_product([dates, INSTRUMENTS], names=["timestamp", "instrument"])
    return pd.Series([INPUTS[s][t] for t in range(N) for s in INSTRUMENTS], index=idx, dtype=float)


def _param(args):
    """Return a ''-joined signature name and the expr builder."""
    return args


_OPERATORS = [
    ("RSI_WILDER", lambda w: make_cleaned_call_factory("RSI_WILDER")(_C("close"), w)),
    ("ATR_WILDER", lambda w: make_cleaned_call_factory("ATR_WILDER")(_C("high"), _C("low"), _C("close"), w)),
    ("DMI_plus", lambda w: make_cleaned_call_factory("DMI_plus")(_C("high"), _C("low"), _C("close"), w)),
    ("DMI_minus", lambda w: make_cleaned_call_factory("DMI_minus")(_C("high"), _C("low"), _C("close"), w)),
    ("DX", lambda w: make_cleaned_call_factory("DX")(_C("high"), _C("low"), _C("close"), w)),
    ("ADX", lambda w: make_cleaned_call_factory("ADX")(_C("high"), _C("low"), _C("close"), w)),
    ("MACD_line", lambda w: make_cleaned_call_factory("MACD_line")(_C("close"), 5, 8)),
    ("MACD_signal", lambda w: make_cleaned_call_factory("MACD_signal")(_C("close"), 5, 8, 3)),
    ("MACD_hist", lambda w: make_cleaned_call_factory("MACD_hist")(_C("close"), 5, 8, 3)),
    ("DEMA", lambda w: make_cleaned_call_factory("DEMA")(_C("close"), w)),
    ("TEMA", lambda w: make_cleaned_call_factory("TEMA")(_C("close"), w)),
    ("PPO", lambda w: make_cleaned_call_factory("PPO")(_C("close"), 5, 8)),
    ("PPO_signal", lambda w: make_cleaned_call_factory("PPO_signal")(_C("close"), 5, 8, 3)),
    ("PPO_hist", lambda w: make_cleaned_call_factory("PPO_hist")(_C("close"), 5, 8, 3)),
    ("PVO", lambda w: make_cleaned_call_factory("PVO")(_C("volume"), 5, 8)),
    ("PVO_signal", lambda w: make_cleaned_call_factory("PVO_signal")(_C("volume"), 5, 8, 3)),
    ("PVO_hist", lambda w: make_cleaned_call_factory("PVO_hist")(_C("volume"), 5, 8, 3)),
    ("TSI", lambda w: make_cleaned_call_factory("TSI")(_C("close"), 4, 3)),
    ("TSI_signal", lambda w: make_cleaned_call_factory("TSI_signal")(_C("close"), 4, 3, 2)),
    ("KeltnerMid", lambda w: make_cleaned_call_factory("KeltnerMid")(_C("close"), 3)),
    ("KeltnerUpper", lambda w: make_cleaned_call_factory("KeltnerUpper")(_C("high"), _C("low"), _C("close"), 3, 3, 1.5)),
    ("KeltnerLower", lambda w: make_cleaned_call_factory("KeltnerLower")(_C("high"), _C("low"), _C("close"), 3, 3, 1.5)),
    ("KeltnerPosition", lambda w: make_cleaned_call_factory("KeltnerPosition")(_C("high"), _C("low"), _C("close"), 3, 3, 1.5)),
    ("ADL", lambda w: make_cleaned_call_factory("ADL")(_C("high"), _C("low"), _C("close"), _C("volume"), w)),
    ("ChaikinOscillator", lambda w: make_cleaned_call_factory("ChaikinOscillator")(_C("high"), _C("low"), _C("close"), _C("volume"), 2, 3, w)),
    ("CMF", lambda w: make_cleaned_call_factory("CMF")(_C("high"), _C("low"), _C("close"), _C("volume"), w)),
    ("ForceIndex", lambda w: make_cleaned_call_factory("ForceIndex")(_C("close"), _C("volume"), w)),
]

_WINDOWS = {"RSI_WILDER": 4, "ATR_WILDER": 4, "DMI_plus": 4, "DMI_minus": 4, "DX": 4,
            "ADX": 4, "DEMA": 3, "TEMA": 3, "ADL": 3, "ChaikinOscillator": 3,
            "CMF": 3, "ForceIndex": 3}


@pytest.mark.parametrize("op,window", [(op, _WINDOWS.get(op, 4)) for op, _ in _OPERATORS])
def test_wilder_ewm_duckdb_parity(duckdb_source, panel, op, window):
    expr = _OPERATORS[[o for o, _ in _OPERATORS].index(op)][1](window)
    if OperatorRegistry._aliases.get(op, op) not in OperatorRegistry._operators:
        pytest.skip(f"{op} not registered")
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))
    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)
    pd.testing.assert_series_equal(
        pd_out, sql_out, check_names=False, check_index=False,
        check_dtype=False, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("op,window", [(o, _WINDOWS.get(o, 4)) for o, _ in _OPERATORS])
def test_wilder_ewm_pandas_polars_parity(duckdb_source, op, window):
    """Polars-long reference (independent of SQL) — ensures the pandas reference itself is sound."""
    expr = _OPERATORS[[o for o, _ in _OPERATORS].index(op)][1](window)
    if OperatorRegistry._aliases.get(op, op) not in OperatorRegistry._operators:
        pytest.skip(f"{op} not registered")
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))
    pl_run = _run(duckdb_source, expr, "polars_long")
    pl_out = _result_series(pl_run)
    pd.testing.assert_series_equal(
        pd_out, pl_out, check_names=False, check_index=False,
        check_dtype=False, rtol=1e-6, atol=1e-6
    )

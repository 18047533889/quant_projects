"""Bounded production parity checks for pandas, native Polars, and DuckDB SQL."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource


F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def panel() -> InMemorySeriesSource:
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])
    close = pd.Series(
        [10.0, 20.0, 30.0, 11.0, 19.0, 31.0, 12.0, np.nan, 29.0,
         13.0, 18.0, 28.0, 14.0, 17.0, 27.0, 15.0, 16.0, 26.0,
         16.0, 15.0, 25.0, 17.0, 14.0, 24.0],
        index=index,
    )
    open_ = close - 0.5
    volume = pd.Series(np.tile([100.0, 200.0, 300.0], len(dates)), index=index)
    group_id = pd.Series(np.tile([1.0, 1.0, 2.0], len(dates)), index=index)
    return InMemorySeriesSource(
        data={"close": close, "open": open_, "volume": volume, "group_id": group_id}
    )


def _write_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""bounded_daily:
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
    Volume: double
    group_id: int64
""",
        encoding="utf-8",
    )


def _seed(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, instrument in source.data["close"].index:
        def value(name: str):
            item = source.data[name].loc[(timestamp, instrument)]
            return None if pd.isna(item) else float(item)

        rows.append(
            {
                "TradeDate": timestamp.date(),
                "Symbol": instrument,
                "Close": value("close"),
                "Open": value("open"),
                "Volume": value("volume"),
                "group_id": int(source.data["group_id"].loc[(timestamp, instrument)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel: InMemorySeriesSource):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    root = tmp_path / "data"
    _write_registry(tmp_path / "datasets.yaml", root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(tmp_path / "datasets.yaml"))
    _seed(root, panel)
    from data_access import reset_store

    reset_store()
    return build_data_source({"type": "data_access", "dataset": "bounded_daily"})


def _run(source, expr, backend: str) -> dict:
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="bounded_parity", expr=expr))


def _assert_same(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


CASES = [
    ("add", lambda c: F("add")(c("close"), c("open"))),
    ("ts_mean", lambda c: F("ts_mean")(c("close"), 3)),
    ("cs_rank", lambda c: F("cs_rank")(c("close"))),
    ("group_mean", lambda c: F("group_mean")(c("close"), c("group_id"))),
]


def _memory_col(name: str):
    return col(name)


def _sql_col(name: str):
    return col({"close": "Close", "open": "Open", "volume": "Volume"}.get(name, name))


def test_polars_source_is_true_lazy_frame(panel: InMemorySeriesSource):
    frame = panel.scan_polars_long(["close", "open"])
    assert isinstance(frame, pl.LazyFrame)
    assert {"ts", "inst", "close", "open"}.issubset(frame.collect_schema().names())


@pytest.mark.parametrize("name,builder", CASES)
def test_bounded_triple_backend_parity(panel, duckdb_source, name, builder):
    pandas_out = _run(panel, builder(_memory_col), "pandas")
    polars_out = _run(panel, builder(_memory_col), "polars_long")
    sql_out = _run(duckdb_source, builder(_sql_col), "duckdb_sql")

    reference = pandas_out["result"]
    _assert_same(reference, polars_out["result"])
    assert_duckdb_real_sql_execution(sql_out)
    _assert_same(reference, sql_out["result"])

    assert pandas_out["result"].index.names == ["timestamp", "instrument"]
    assert polars_out["result"].index.names == ["timestamp", "instrument"]
    assert sql_out["result"].index.names == ["timestamp", "instrument"]

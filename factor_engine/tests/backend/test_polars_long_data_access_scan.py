# -*- coding: utf-8
"""polars_long 端到端走 DataAccessSource.scan_polars_long（真实 parquet scan）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
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


def _registry(tmp_path: Path, root: Path) -> Path:
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
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    for d in pd.date_range("2024-01-02", periods=5, freq="D"):
        for sym in ["A", "B"]:
            rows.append(
                {
                    "TradeDate": d.date(),
                    "Symbol": sym,
                    "Close": float(d.day + (1 if sym == "A" else 2)),
                }
            )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _normalize_series_index(series: pd.Series) -> pd.Series:
    out = series.sort_index()
    ts = pd.to_datetime(out.index.get_level_values(0)).astype("datetime64[ns]")
    inst = out.index.get_level_values(1).astype(str)
    out.index = pd.MultiIndex.from_arrays([ts, inst], names=out.index.names)
    return out


def test_polars_long_data_access_scan_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_registry(tmp_path, tmp_path / "data")))
    _seed(tmp_path / "data")

    source = build_data_source({"type": "data_access", "dataset": "test_daily", "long_table": True})
    expr = make_cleaned_call_factory("ts_mean")(col("Close"), 3)
    factor = Factor(name="t", expr=expr)

    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=source).run(factor)
    long_out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(factor)

    assert long_out.get("used_polars_long_path") is True
    assert long_out.get("used_polars_long_native") is True
    assert not long_out.get("polars_long_fallback_reason")
    pd.testing.assert_series_equal(
        _normalize_series_index(pd_out["result"]),
        _normalize_series_index(long_out["result"]),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_scan_index_long_align_without_load_column(tmp_path, monkeypatch):
    """``FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE=1`` 时按 scan_index_long 对齐，不 load_column。"""
    import os

    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_registry(tmp_path, tmp_path / "data")))
    _seed(tmp_path / "data")
    os.environ["FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE"] = "1"

    load_calls: list[str] = []

    source = build_data_source({"type": "data_access", "dataset": "test_daily", "long_table": True})
    orig_load = source.load_column

    def _track_load(name: str):
        load_calls.append(name)
        return orig_load(name)

    source.load_column = _track_load  # type: ignore[method-assign]

    try:
        expr = make_cleaned_call_factory("ts_mean")(col("Close"), 3)
        long_out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(
            Factor(name="t", expr=expr)
        )
        assert long_out.get("used_polars_long_path") is True
        assert load_calls == []
        idx = long_out["result"].sort_index().index
        assert len(idx) == 10
    finally:
        os.environ.pop("FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE", None)


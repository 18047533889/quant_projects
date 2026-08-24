# -*- coding: utf-8
"""Round-7 gap-closure SQL ops: DuckDB 端到端 parity vs pandas.

Covers the six operators added to ``SQL_IMPLEMENTED_CANONICALS`` in the
polars/duckdb gap-closure pass (windowed sign ratios, central moment, group
reducers).  Runs each factor through the real ``duckdb_sql`` backend against a
parquet panel and compares to the ``pandas`` reference.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source


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
    Ret: double
    Group: integer
    W: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_data(root: Path, n_days: int = 160) -> None:
    root.mkdir(parents=True, exist_ok=True)
    import numpy as np

    rng = np.random.default_rng(20260809)
    rows = []
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    for sym, base, grp in [("A", 10.0, 0), ("B", 20.0, 0), ("C", 15.0, 1)]:
        prev = base
        for i, d in enumerate(idx):
            shock = rng.normal(0, 0.02)
            close = base + i * 0.01 + float(shock) * base
            r = close / prev - 1.0 if i else 0.0
            prev = close
            w = float(rng.uniform(0.1, 3.0))
            if rng.random() < 0.05:
                rows.append({"TradeDate": d.date(), "Symbol": sym, "Ret": None, "Group": grp, "W": w})
            else:
                rows.append({"TradeDate": d.date(), "Symbol": sym, "Ret": r, "Group": grp, "W": w})
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _run_pair(tmp_path, monkeypatch, expr, name, *, rtol=1e-4, atol=1e-6):
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_registry(tmp_path, tmp_path / "data")))
    _seed_data(tmp_path / "data")
    source = build_data_source(
        {"type": "data_access", "dataset": "test_daily", "start_date": "2023-01-02", "end_date": "2023-08-31"}
    )
    factor = Factor(name=name, expr=expr)
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
    eng_sql = FactorEngine(backend=build_backend("duckdb_sql"), data_source=source)
    a = eng_pd.run(factor)["result"]
    b = eng_sql.run(factor)["result"]
    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
    if aligned.empty:
        return
    col_a, col_b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    diff = (col_a - col_b).abs()
    tol = max(atol, rtol * col_a.abs().max())
    assert diff.max() <= tol, f"{name}: max diff {diff.max():.6g} > tol {tol:.6g} (n={len(aligned)})"


@pytest.fixture(scope="module", autouse=True)
def _load():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()


def _c(name):
    return make_cleaned_call_factory(name)


def test_gap_closure_window_sign_ratios(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("ts_positive_ratio")(col("Ret"), 20, 0.0, 1), "pos_ratio")
    _run_pair(tmp_path, monkeypatch, _c("ts_negative_ratio")(col("Ret"), 20, 0.0, 1), "neg_ratio")
    _run_pair(tmp_path, monkeypatch, _c("ts_zero_ratio")(col("Ret"), 20, 0.001, 1), "zero_ratio")


def test_gap_closure_central_moment(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("ts_moment")(col("Ret"), 20, 3), "moment3")
    _run_pair(tmp_path, monkeypatch, _c("ts_moment")(col("Ret"), 20, 2), "moment2")


def test_gap_closure_group_reducers(tmp_path, monkeypatch):
    _run_pair(tmp_path, monkeypatch, _c("group_valid_count")(col("Ret"), col("Group")), "gv_count")
    _run_pair(tmp_path, monkeypatch, _c("group_weighted_mean")(col("Ret"), col("Group"), col("W")), "gw_mean")

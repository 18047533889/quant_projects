# -*- coding: utf-8
"""run_many batch 路径 production fastpath runtime audit。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.columns import col
from api.factor import Factor
from api import ts_mean
from backend.factory import build_backend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


@pytest.fixture
def source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0], index=idx)
    open_ = pd.Series([9.0, 10.0, 19.0, 20.0, 9.5, 11.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "open": open_})


def test_run_many_calls_fastpath_runtime_audit(_loaded, source, monkeypatch):
    called: list[str] = []

    def _audit(ctx, *, mode, context):
        called.append(str(context))

    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "1")
    monkeypatch.setattr(
        "runtime.batch_service.assert_production_fastpath_runtime",
        _audit,
    )
    engine = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    factors = [
        Factor(name="f1", expr=ts_mean(col("close"), 3)),
        Factor(name="f2", expr=ts_mean(col("open"), 3)),
    ]
    engine.run_many(factors, enable_cse=True)
    assert any("run_many" in c for c in called)

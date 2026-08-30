# -*- coding: utf-8
"""run_many batch 路径 production fastpath runtime audit。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import ts_mean
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def _loaded():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.registry import _BOOTSTRAP_TOKEN
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    # P0-14: register_sql_backends patches registry catalog metadata for the
    # SQL canonical set.  After load_all() the registry is FROZEN, so the
    # marker pass must run inside the bootstrap thaw window and the registry
    # must be re-frozen afterwards (same pattern as the hardening suite).
    OperatorRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
    try:
        register_sql_backends()
    finally:
        OperatorRegistry.finalize()
        OperatorRegistry.freeze()


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
        "factor_engine.runtime.batch_service.assert_production_fastpath_runtime",
        _audit,
    )
    engine = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    factors = [
        Factor(name="f1", expr=ts_mean(col("close"), 3)),
        Factor(name="f2", expr=ts_mean(col("open"), 3)),
    ]
    engine.run_many(factors, enable_cse=True)
    assert any("run_many" in c for c in called)

"""ExecutionCacheSession 与 PolarsBackend 增强测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cache.layers import CacheLayer
from factor_engine.cache.session import ExecutionCacheSession
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.cache import CacheManager
from tests.helpers import InMemorySeriesSource


def _panel():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(np.arange(4, dtype=float) + 1.0, index=idx)


def test_execution_cache_session_tracks_subplan_hits():
    plan_cache = CacheManager()
    session = ExecutionCacheSession(plan_cache=plan_cache)
    assert session.get_subplan("missing") is None
    assert session.stats.misses.get(CacheLayer.L2_SUBPLAN.value) == 1

    plan_cache.set("k", _panel())
    session2 = ExecutionCacheSession(plan_cache=plan_cache)
    hit = session2.get_subplan("k")
    assert hit is not None
    assert session2.stats.hits.get(CacheLayer.L2_SUBPLAN.value) == 1


def test_polars_backend_prefers_auto():
    pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all

    load_all()
    backend = build_backend("polars")
    src = InMemorySeriesSource(data={"close": _panel()})
    eng = FactorEngine(backend=backend, data_source=src, cache=CacheManager())
    ctx = eng._make_context()
    assert ctx.perf.operator_backend == "auto"


def test_polars_backend_run_many_with_cache():
    pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all

    load_all()
    backend = build_backend("polars")
    src = InMemorySeriesSource(data={"close": _panel()})
    eng = FactorEngine(backend=backend, data_source=src, cache=CacheManager())
    sub = ts_mean(col("close"), 2)
    out = eng.run_many(
        [Factor(name="a", expr=sub), Factor(name="b", expr=rank(sub))],
    )
    assert len(out["results"]) == 2


def test_polars_backend_records_runtime_stats():
    pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all

    load_all()
    eng = FactorEngine(
        backend=build_backend("polars"),
        data_source=InMemorySeriesSource(data={"close": _panel()}),
        cache=CacheManager(),
    )
    eng.run(Factor(name="t", expr=ts_mean(col("close"), 2)))
    ctx = eng._make_context()
    wrapped = ExecutionCacheSession(plan_cache=eng.cache).wrap_context(ctx)
    assert wrapped.runtime_stats is not None

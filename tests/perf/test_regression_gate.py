"""轻量性能回归门禁（CI smoke，阈值宽松）。"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.cache import CacheManager
from tests.helpers import InMemorySeriesSource

_MAX_COMPILE_MANY_SEC = 5.0
_MAX_RUN_MANY_SEC = 10.0


def _large_panel(n_days: int = 60, n_inst: int = 20):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["timestamp", "instrument"])
    rng = np.random.default_rng(42)
    return {
        "close": pd.Series(rng.normal(size=len(idx)), index=idx),
        "open": pd.Series(rng.normal(size=len(idx)), index=idx),
    }


@pytest.mark.perf
def test_compile_many_under_threshold():
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        src = InMemorySeriesSource(data=_large_panel())
        eng = FactorEngine(backend=build_backend("pandas"), data_source=src)
        factors = [
            Factor(name=f"f{i}", expr=ts_mean(col("close"), 5 + i))
            for i in range(8)
        ]
        t0 = time.perf_counter()
        out = eng.analyze_batch(factors)
        elapsed = time.perf_counter() - t0
        assert "batch_graph" in out
        assert len(out["batch_graph"]["parallel_layers"]) >= 1
        assert elapsed < _MAX_COMPILE_MANY_SEC, f"compile_many 过慢: {elapsed:.2f}s"
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)


@pytest.mark.perf
def test_run_many_with_cache_second_run_faster():
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        src = InMemorySeriesSource(data=_large_panel(n_days=30, n_inst=10))
        eng = FactorEngine(
            backend=build_backend("pandas"),
            data_source=src,
            cache=CacheManager(),
        )
        factors = [
            Factor(name="a", expr=ts_mean(col("close"), 10)),
            Factor(name="b", expr=rank(ts_mean(col("close"), 10))),
            Factor(name="c", expr=ts_mean(col("open"), 5)),
        ]
        t0 = time.perf_counter()
        eng.run_many(factors)
        first = time.perf_counter() - t0

        t0 = time.perf_counter()
        eng.run_many(factors)
        second = time.perf_counter() - t0

        assert first < _MAX_RUN_MANY_SEC
        assert second <= first * 1.05 + 0.01
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)

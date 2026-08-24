"""性能基准阈值门禁（PR smoke + nightly slow）。"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

_THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


def _load_thresholds() -> dict:
    if not _THRESHOLDS_PATH.exists():
        return {}
    return yaml.safe_load(_THRESHOLDS_PATH.read_text(encoding="utf-8")) or {}


def _synthetic_panel(n_days: int, n_inst: int) -> dict:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    inst = [f"S{i:04d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["timestamp", "instrument"])
    rng = np.random.default_rng(42)
    return {
        "close": pd.Series(rng.normal(size=len(idx)), index=idx),
        "volume": pd.Series(rng.normal(size=len(idx)), index=idx),
    }


@pytest.mark.perf
def test_compile_many_threshold():
    th = _load_thresholds().get("compile_many", {})
    max_sec = float(th.get("max_sec", 8.0))
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        src = InMemorySeriesSource(data=_synthetic_panel(60, 20))
        eng = FactorEngine(backend=build_backend("pandas"), data_source=src)
        factors = [Factor(name=f"f{i}", expr=ts_mean(col("close"), 5 + i)) for i in range(8)]
        t0 = time.perf_counter()
        out = eng.analyze_batch(factors)
        elapsed = time.perf_counter() - t0
        assert "batch_graph" in out
        assert elapsed < max_sec, f"compile_many 过慢: {elapsed:.2f}s > {max_sec}s"
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)


@pytest.mark.perf
def test_run_many_smoke_threshold():
    th = _load_thresholds().get("run_many_smoke", {})
    n_days = int(th.get("n_days", 120))
    n_inst = int(th.get("n_inst", 50))
    n_factors = int(th.get("n_factors", 8))
    max_sec = float(th.get("max_sec", 20.0))
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        src = InMemorySeriesSource(data=_synthetic_panel(n_days, n_inst))
        eng = FactorEngine(backend=build_backend("pandas"), data_source=src)
        factors = [
            Factor(name=f"f{i}", expr=rank(ts_mean(col("close"), 5 + (i % 10))))
            for i in range(n_factors)
        ]
        t0 = time.perf_counter()
        out = eng.run_many(factors)
        elapsed = time.perf_counter() - t0
        assert len(out["results"]) == n_factors
        assert elapsed < max_sec, f"run_many_smoke 过慢: {elapsed:.2f}s > {max_sec}s"
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)


@pytest.mark.perf
@pytest.mark.slow
def test_run_many_1m_threshold():
    th = _load_thresholds().get("run_many_1m", {})
    n_days = int(th.get("n_days", 250))
    n_inst = int(th.get("n_inst", 4000))
    n_factors = int(th.get("n_factors", 6))
    max_sec = float(th.get("max_sec", 90.0))
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        src = InMemorySeriesSource(data=_synthetic_panel(n_days, n_inst))
        eng = FactorEngine(backend=build_backend("pandas"), data_source=src)
        factors = [
            Factor(name=f"f{i}", expr=ts_mean(col("close"), 5 + i))
            for i in range(n_factors)
        ]
        t0 = time.perf_counter()
        eng.run_many(factors)
        elapsed = time.perf_counter() - t0
        assert elapsed < max_sec, f"run_many_1m 过慢: {elapsed:.2f}s > {max_sec}s"
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)


@pytest.mark.perf
@pytest.mark.slow
def test_run_many_10m_threshold():
    th = _load_thresholds().get("run_many_10m", {})
    n_days = int(th.get("n_days", 500))
    n_inst = int(th.get("n_inst", 20000))
    n_factors = int(th.get("n_factors", 4))
    max_sec = float(th.get("max_sec", 180.0))
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        src = InMemorySeriesSource(data=_synthetic_panel(n_days, n_inst))
        eng = FactorEngine(backend=build_backend("pandas"), data_source=src)
        factors = [
            Factor(name=f"f{i}", expr=ts_mean(col("close"), 5 + i))
            for i in range(n_factors)
        ]
        t0 = time.perf_counter()
        eng.run_many(factors)
        elapsed = time.perf_counter() - t0
        assert elapsed < max_sec, f"run_many_10m 过慢: {elapsed:.2f}s > {max_sec}s"
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)

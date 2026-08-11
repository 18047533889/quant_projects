"""Phase 21：ReadSession / production policy / rolling CSE / factor_matrix / sharded。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean, zscore
from api.cleaned_ops import make_cleaned_call_factory
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from runtime.production_policy import (
    ProductionPolicyViolation,
    assert_columns_explicit,
    assert_production_run_flags,
)
from storage.read_session import DataSourceReadSession
from tests.helpers import InMemorySeriesSource

ts_delta = make_cleaned_call_factory("ts_delta")


def _data():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {
        "close": pd.Series(np.arange(6, dtype=float) + 1.0, index=idx),
    }


@dataclass
class _CountingColumnSource(InMemorySeriesSource):
    load_columns_calls: int = field(default=0, init=False)
    _column_cache: dict[str, pd.Series] = field(default_factory=dict, init=False)

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        needed = [n for n in names if n not in self._column_cache]
        if needed:
            self.load_columns_calls += 1
            batch = super().load_columns(needed)
            self._column_cache.update(batch)
        return {n: self._column_cache[n] for n in names}

    def load_column(self, name: str) -> pd.Series:
        if name in self._column_cache:
            return self._column_cache[name]
        return self.load_columns([name])[name]

    def prefetch_columns(self, names: list[str]) -> None:
        self.load_columns(names)

    def column_cache_stats(self) -> dict[str, int]:
        return {"cached_columns": len(self._column_cache), "cached_panels": 0}


def test_run_many_shared_ts_mean_across_rank_zscore_delta():
    data = _data()
    eng = FactorEngine(backend=PandasBackend(), data_source=InMemorySeriesSource(data=data))
    sub = ts_mean(col("close"), 2)
    factors = [
        Factor(name="a", expr=rank(sub)),
        Factor(name="b", expr=zscore(sub)),
        Factor(name="c", expr=ts_delta(sub, 1)),
    ]
    out = eng.run_many(factors)
    assert len(out["dag"].shared_nodes) >= 1
    assert out["rolling_cache"]["rolling_shared_count"] >= 1
    for i, name in enumerate(("a", "b", "c")):
        solo = eng.run(factors[i])["result"]
        pd.testing.assert_series_equal(
            out["results"][name], solo, check_names=False
        )


def test_run_single_path_uses_read_session_once():
    data = _data()
    src = _CountingColumnSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    f = Factor(name="x", expr=ts_mean(col("close"), 2))
    eng.run(f, input_dq_check=True)
    assert src.load_columns_calls == 1


def test_production_policy_requires_flags():
    with pytest.raises(ProductionPolicyViolation):
        assert_production_run_flags(
            mode="production",
            input_dq_check=False,
            auto_warmup=True,
            pit_enforce=True,
        )


def test_production_columns_explicit():
    with pytest.raises(ProductionPolicyViolation):
        assert_columns_explicit(["*"], mode="production")


def test_factor_matrix_materializer_roundtrip(tmp_path):
    data = _data()
    eng = FactorEngine(backend=PandasBackend(), data_source=InMemorySeriesSource(data=data))
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=rank(col("close")))
    summary = eng.materialize_matrix(
        [f1, f2],
        factor_ids=["alpha_001", "alpha_002"],
        universe="test_univ",
        frequency="1d",
        matrix_root=tmp_path / "matrix",
    )
    assert summary["rows_written"] == 6
    assert set(summary["factor_ids"]) == {"alpha_001", "alpha_002"}

    from storage.factor_matrix_materializer import FactorMatrixMaterializer

    loaded = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix",
        universe="test_univ",
        frequency="1d",
        factor_ids=["alpha_001", "alpha_002"],
    )
    assert {"alpha_001", "alpha_002"}.issubset(set(loaded.columns))
    assert len(loaded) == 6


def test_materialize_sharded_selects_subset(tmp_path, monkeypatch):
    data = _data()
    eng = FactorEngine(backend=PandasBackend(), data_source=InMemorySeriesSource(data=data))
    factors = [
        Factor(name=f"f{i}", expr=ts_mean(col("close"), 2))
        for i in range(4)
    ]
    ids = [f"factor_{i:03d}" for i in range(4)]

    written: list[str] = []

    # R39-PERF-068：materialize_sharded 走 materialize_many_fast 流式 sink，
    # 写路径是 execute_materialize_batch（不再逐因子 execute_materialize）。
    # 测试 patch 批量写入口，记录本 shard 实际写到的 factor_ids。
    def _fake_batch(engine, items, generation=None, shared_options=None):
        for it in items:
            written.append(it.factor_id)
        return {
            "counters": {"batch_write_transaction_count": 1},
            "materializations": {
                it.factor.name: {"factor_id": it.factor_id, "rows_written": 6}
                for it in items
            },
        }

    monkeypatch.setattr(
        "runtime.materialize_batch.execute_materialize_batch",
        _fake_batch,
    )
    out = eng.materialize_sharded(
        factors,
        factor_ids=ids,
        shard_index=0,
        shard_count=2,
        lake_root=tmp_path / "lake",
    )
    assert out["shard_count"] == 2
    assert len(written) >= 1
    assert len(written) <= 4


def test_read_session_load_columns_once():
    data = _data()
    src = _CountingColumnSource(data=data)
    session = DataSourceReadSession(src)
    session.load_columns_once(["close"])
    session.load_columns_once(["close"])
    assert src.load_columns_calls == 1
    assert session.last_scope_key is not None

"""run_many 生产路径：batch input_dq + CSE + 列缓存去重。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import rank, ts_mean
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.read_session import DataSourceReadSession
from tests.helpers import InMemorySeriesSource


def _data():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {
        "close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx),
        "open": pd.Series(np.arange(4, dtype=float) + 0.5, index=idx),
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


def test_run_many_input_dq_uses_batch_path_and_cse():
    data = _data()
    src = _CountingColumnSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    out = eng.run_many([f1, f2], input_dq_check=True)

    assert src.load_columns_calls == 1
    assert out["input_dq"]["passed"] is True
    assert len(out["dag"].shared_nodes) >= 1
    r1 = eng.run(f1, input_dq_check=False)["result"]
    r2 = eng.run(f2, input_dq_check=False)["result"]
    pd.testing.assert_series_equal(out["results"]["a"], r1, check_names=False)
    pd.testing.assert_series_equal(out["results"]["b"], r2, check_names=False)


def test_run_many_parallel_input_dq_single_load():
    pytest.importorskip("joblib")
    data = _data()
    src = _CountingColumnSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=ts_mean(col("open"), 2))
    out = eng.run_many_parallel([f1, f2], n_jobs=2, input_dq_check=True)

    assert src.load_columns_calls == 1
    assert out["input_dq"]["passed"] is True


def test_read_session_wraps_cache():
    data = _data()
    src = _CountingColumnSource(data=data)
    session = DataSourceReadSession(src)
    session.assert_input_dq(["close", "open"])
    session.prefetch(["close", "open"])
    assert src.load_columns_calls == 1
    assert session.cache_stats()["cached_columns"] == 2
    assert session.cached_columns() == frozenset({"close", "open"})

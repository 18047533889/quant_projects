from types import SimpleNamespace

import pandas as pd

from factor_engine.api import col, ts_mean
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.storage.sources.data_access_source import DataAccessSource


class _Lease:
    def release(self):
        return None


class _Broker:
    def acquire_memory(self, *args, **kwargs):
        return _Lease()


class _Handle:
    def __init__(self, table):
        self._table = table
        self.snapshot = SimpleNamespace(snapshot_id="snap-1")
        self.read_identity = None

    def to_arrow(self):
        return self._table

    def close(self):
        return None


class _Store:
    def __init__(self, table):
        self.table = table
        self.read_calls = 0
        self.partition_version = "pv-1"

    def manifest_version(self, dataset, **params):
        return {
            "dataset": dataset,
            "has_manifest": True,
            "fresh": True,
            "dataset_version": "dv-1",
            "partition_version": self.partition_version,
            "file_count": 1,
        }

    def get_dataset(self, dataset, **params):
        return SimpleNamespace(time_column="ts", instrument_column="inst")

    def read(self, dataset, **kwargs):
        self.read_calls += 1
        return _Handle(self.table)


def test_default_oversize_sink_reuses_snapshot_bound_dataaccess_cache_across_waves(
    monkeypatch,
):
    import pyarrow as pa

    from factor_engine.planner import physical_lowerer
    from factor_engine.storage.sources import data_access_source as source_module

    monkeypatch.setattr(physical_lowerer, "_get_adaptive_dag_width_limit", lambda: 2)
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_chunk_size", lambda: 2)
    dates = pd.date_range("2024-01-01", periods=4)
    values = pd.Series(
        range(4),
        index=pd.MultiIndex.from_arrays(
            [dates, ["A"] * 4], names=["timestamp", "instrument"],
        ),
        dtype=float,
    )
    store = _Store(pa.table({"ts": dates, "inst": ["A"] * 4, "close": values.array}))
    monkeypatch.setattr(source_module, "_get_store", lambda: store)
    source = DataAccessSource(dataset="demo", fields={"close": "close"})
    source._cache_broker = _Broker()
    source._max_cache_bytes = 1024 * 1024
    engine = FactorEngine(PandasBackend(), source)
    shared = ts_mean(col("close"), 2)
    observed = {}

    result = engine.run_many(
        [Factor(f"f{i}", shared + i) for i in range(5)],
        result_policy="sink",
        sink=lambda name, value: observed.__setitem__(name, value),
        perf=PerfConfig(
            max_workers=1, native_fusion=False, result_budget_bytes=1024 * 1024,
        ),
    )

    assert result["executor"] == "streaming_waves"
    assert result["completed_waves"] == 3
    assert result["cse_scope"] == "per_wave"
    assert store.read_calls == 1
    assert source.column_cache_stats()["cached_columns"] == 1
    expected = values.groupby(level="instrument").rolling(2, min_periods=1).mean()
    expected.index = expected.index.droplevel(0)
    expected = expected.reorder_levels(values.index.names).sort_index()
    for i in range(5):
        pd.testing.assert_series_equal(observed[f"f{i}"], expected + i, check_names=False)


def test_dataaccess_manifest_change_invalidates_cached_wave_field(monkeypatch):
    import pyarrow as pa

    from factor_engine.storage.sources import data_access_source as source_module

    dates = pd.date_range("2024-01-01", periods=2)
    store = _Store(
        pa.table({"ts": dates, "inst": ["A", "A"], "close": [1.0, 2.0]}),
    )
    monkeypatch.setattr(source_module, "_get_store", lambda: store)
    source = DataAccessSource(dataset="demo", fields={"close": "close"})
    source._cache_broker = _Broker()
    source._max_cache_bytes = 1024 * 1024

    first = source.load_columns(["close"])["close"]
    assert store.read_calls == 1
    assert source.column_cache_stats()["cached_columns"] == 1

    store.partition_version = "pv-2"
    source.refresh_snapshot(force=True)
    assert source.column_cache_stats()["cached_columns"] == 0

    second = source.load_columns(["close"])["close"]
    assert store.read_calls == 2
    pd.testing.assert_series_equal(second, first)

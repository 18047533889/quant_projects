"""Phase B：load_matrix、stats null_ratio、parallel_layers、事件队列。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from backend.routing_env import routing_execution_scope
from data_access.stats import DatasetStatsSnapshot, estimate_column_null_ratios
from runtime.engine import FactorEngine
from runtime.perf_config import PerfConfig
from runtime.quality.input_dq import (
    InputDQThresholds,
    adjust_input_dq_thresholds_from_stats,
)
from runtime.task_queue import FileTaskQueue, JOB_TYPE_DATA_EVENT, enqueue_data_event
from storage.materialize.factor_matrix_materializer import FactorMatrixMaterializer
from storage.result_store import PandasResultStore
from tests.helpers import InMemorySeriesSource


def _panel(n: int = 6) -> dict:
    dates = pd.bdate_range("2024-01-02", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "instrument"])
    return {
        "close": pd.Series([float(i) for i in range(len(idx))], index=idx),
        "volume": pd.Series([float(i * 2) for i in range(len(idx))], index=idx),
    }


def test_load_matrix_roundtrip(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=3)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    s1 = pd.Series([1.0, 2.0, 3.0], index=idx)
    s2 = pd.Series([4.0, 5.0, 6.0], index=idx)
    mat = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    mat.materialize({"f_a": s1, "f_b": s2}, universe="test_u")

    store = PandasResultStore(lake_root=tmp_path / "lake")
    frame = store.load_matrix(["f_a", "f_b"], universe="test_u", matrix_root=tmp_path / "matrix")
    assert set(frame.columns) >= {"datetime", "asset", "f_a", "f_b"}
    assert len(frame) == 3


def test_estimate_column_null_ratios(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"a": [1, None, 3], "b": [1.0, 2.0, 3.0]})
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    ratios = estimate_column_null_ratios([path])
    assert ratios["a"] == pytest.approx(2 / 3)
    assert ratios["b"] == pytest.approx(1.0)


def test_adjust_input_dq_thresholds_from_stats():
    stats = DatasetStatsSnapshot(
        dataset="ds",
        num_rows=100,
        num_files=1,
        partition_columns=("year",),
        column_null_ratio={"close": 0.8},
    )
    th = adjust_input_dq_thresholds_from_stats(
        InputDQThresholds(min_non_null_ratio=0.01),
        stats,
        ["close"],
    )
    assert th.min_non_null_ratio >= 0.8 * 0.95


def test_run_many_parallel_layers_metadata():
    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=_panel()),
    )
    f1 = Factor(name="f1", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="f2", expr=rank(col("volume")))
    out = eng.run_many([f1, f2], input_dq_check=False)
    graph = out.get("batch_graph") or {}
    layers = graph.get("parallel_layers") or []
    assert len(layers) >= 1
    assert set(out["results"]) == {"f1", "f2"}


def test_enqueue_data_event_job_type(tmp_path):
    queue = FileTaskQueue(tmp_path / "queue")
    job = enqueue_data_event(
        queue,
        dataset="daily",
        column="close",
        updated_date="2024-06-01",
    )
    assert job.payload.get("job_type") == JOB_TYPE_DATA_EVENT
    pending = list((tmp_path / "queue" / "pending").glob("*.json"))
    payload = json.loads(pending[0].read_text(encoding="utf-8"))
    assert payload["payload"]["dataset"] == "daily"


def test_routing_execution_scope_sets_numba_env():
    perf = PerfConfig(use_numba_rolling=True)
    import os

    prev = os.environ.pop("FACTOR_ENGINE_USE_NUMBA", None)
    try:
        with routing_execution_scope(perf):
            assert os.environ.get("FACTOR_ENGINE_USE_NUMBA") == "1"
        assert "FACTOR_ENGINE_USE_NUMBA" not in os.environ
    finally:
        if prev is not None:
            os.environ["FACTOR_ENGINE_USE_NUMBA"] = prev

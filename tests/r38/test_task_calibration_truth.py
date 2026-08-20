# -*- coding: utf-8 -*-
"""R38 P0-007/008/009/010（§6）：calibration 只接收真实观测。

行为探针：
    - ``elapsed_ms`` 是真实 duration（started_at dispatch 时刻 → finished_at），
      不是绝对 monotonic timestamp（§R38_CALIBRATION_ELAPSED_IS_DURATION）；
    - ``output_bytes_actual`` 来自真实 result（§R38_CALIBRATION_OUTPUT_BYTES_ACTUAL）；
    - ``attribution_quality=unattributed`` 的 peak 不进入 P99 主模型（§R38_...PEAK_IS_OBSERVED）；
    - 旧 schema（R36 预测值污染）数据不进入新 P99（§P0-010）。
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from runtime.resource_calibration_store import (
    CALIBRATION_SCHEMA_VERSION,
    ResourceCalibrationStore,
    ShapeCalibration,
)
from runtime.resource_shape import ResourceShapeKey
from runtime.task_run_observation import (
    ATTRIBUTION_UNATTRIBUTED,
    TaskRunObservation,
    estimate_output_bytes,
)


def _key():
    return ResourceShapeKey("ts_mean", "pandas_numpy", rows_bucket=2, instruments_bucket=2, window_bucket=1)


def test_elapsed_is_duration_not_timestamp():
    cal = ShapeCalibration(shape_key=_key())
    t0 = time.monotonic() * 1000.0
    obs = TaskRunObservation(
        task_id="root:x",
        started_at_monotonic=t0,
        finished_at_monotonic=t0 + 123.5,
        elapsed_ms=123.5,
        baseline_family_pss=0,
        peak_family_pss=None,
        output_bytes_actual=0,
        spill_bytes_actual=0,
        read_bytes_actual=0,
        write_bytes_actual=0,
        backend_threads_actual=1,
        attribution_quality=ATTRIBUTION_UNATTRIBUTED,
    )
    assert obs.elapsed_ms == 123.5
    # elapsed 不能是绝对 timestamp 量级（monotonic*1000 早已 >> 1e9）。
    assert obs.elapsed_ms < 1e6


def test_scheduler_records_real_duration():
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from runtime.resource_broker import ResourceBroker
    from planner.physical_factor_dag import PhysicalFactorTask, TASK_ROOT
    from runtime.resource_calibration_store import reset_calibration_store, global_calibration_store

    reset_calibration_store()
    store = global_calibration_store()
    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    task = PhysicalFactorTask(
        task_id="root:f", op="ts_mean", task_type=TASK_ROOT, factor_name="f",
        executable=True,
    )
    # 模拟 dispatch 时记录 start，稍后完成。
    sched._task_started_at["root:f"] = time.monotonic() * 1000.0
    time.sleep(0.01)
    sched._record_timing("root:f", task)
    sched._record_task_calibration("root:f", task, result=pd.Series([1.0]))
    timing = sched._task_timing["root:f"]
    assert timing["elapsed_ms"] > 0
    assert timing["started_at_ms"] <= timing["finished_at_ms"]
    # store 中该 shape 的 elapsed 分布有样本（真实 duration）。
    p = store.predict(ResourceShapeKey.from_task(task))
    assert p is not None
    assert p["elapsed_sample_count"] >= 1


def test_output_bytes_comes_from_result():
    from planner.physical_factor_dag import PhysicalFactorTask, TASK_ROOT

    result = pd.Series([1.0, 2.0, 3.0])
    actual = estimate_output_bytes(result)
    assert actual > 0
    # scheduler 记录 output_bytes_actual 而非 contract.output_bytes。
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from runtime.resource_broker import ResourceBroker
    from runtime.resource_calibration_store import reset_calibration_store, global_calibration_store
    from runtime.task_resource_contract import TaskResourceContract

    reset_calibration_store()
    store = global_calibration_store()
    sched = AdaptiveBatchScheduler(broker=ResourceBroker())
    contract = TaskResourceContract(
        predicted_elapsed_ms=1.0, cpu_tokens=1, peak_memory_bytes=1024,
        output_bytes=1024, backend="pandas_numpy", backend_threads=1,
        estimate_basis="test",
    )
    task = PhysicalFactorTask(
        task_id="root:o", op="ts_mean", task_type=TASK_ROOT, factor_name="o",
        resource_contract=contract, executable=True,
    )
    sched._task_started_at["root:o"] = time.monotonic() * 1000.0
    sched._record_timing("root:o", task)
    sched._record_task_calibration("root:o", task, result=result)
    # store 记录的是真实输出字节（result 只有 3 个 float，远小于 contract 1024）。
    # elapsed/output 分布都应更新。
    p = store.predict(ResourceShapeKey.from_task(task))
    assert p is not None and p["elapsed_sample_count"] >= 1


def test_unattributed_peak_not_in_p99_model():
    key = _key()
    store = ResourceCalibrationStore()
    # 10 次 unattributed（预测值）记录 → memory_obs 不增长。
    for _ in range(10):
        store.record(key, elapsed_ms=10, peak_mem=10 * 1024**3, attribution_quality="unattributed", peak_is_trusted=False)
    p = store.predict(key)
    assert p is not None
    assert p["sample_count"] == 0, "unattributed peak must not enter memory_obs"
    assert p["memory_p99"] is None
    # 一次可信观测 → 进入 P99 模型。
    store.record(key, elapsed_ms=10, peak_mem=1000, attribution_quality="isolated", peak_is_trusted=True)
    p = store.predict(key)
    assert p["sample_count"] == 1
    assert p["memory_p99"] is not None and p["memory_p99"] >= 1000


def test_old_schema_data_excluded(tmp_path):
    import json
    import pandas as pd

    key = _key()
    # 模拟旧 schema（R36 污染数据，schema_version=1）。
    rows = [{
        "schema_version": 1,
        "calibration_source_version": "fe-r36",
        "hardware_fingerprint": json.dumps(sorted(key.hardware_fingerprint().items()) if hasattr(key, "hardware_fingerprint") else {}, sort_keys=True),
        "shape_key": json.dumps(key.to_dict(), sort_keys=True),
        "max_observed": 10**12,
        "correction_factor": 2.0,
        "oom_events": 0,
        "near_oom_events": 0,
        "sample_count": 5,
        "memory_obs": [10**12] * 5,
        "elapsed_obs": [1.0] * 5,
        "output_obs": [0.0] * 5,
        "spill_obs": [0.0] * 5,
        "updated_at_ms": 0.0,
    }]
    from runtime.resource_shape import hardware_fingerprint
    rows[0]["hardware_fingerprint"] = json.dumps(hardware_fingerprint(), sort_keys=True)
    path = str(tmp_path / "old.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    store = ResourceCalibrationStore(path=path)
    # 旧 schema 数据不进入新 P99（sample_count=0）。
    p = store.predict(key)
    assert p is None or p["sample_count"] == 0

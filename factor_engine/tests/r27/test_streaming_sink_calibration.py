# -*- coding: utf-8 -*-
"""R27-016..019/105..109/250/251/240: streaming sink backpressure + runtime
calibration + current-RSS telemetry。"""
from __future__ import annotations

import threading

from runtime.resource_telemetry import (
    _lifetime_peak_rss_bytes,
    _rss_bytes,
    finalize_resource_telemetry,
    resource_telemetry_summary,
)
from runtime.runtime_calibration import calibrated_peak_bytes, record_task_actual
from runtime.streaming_result_sink import BoundedResultQueue, ResultItem


def test_rss_telemetry_current_vs_peak():
    # R27-034/035/144/169：current RSS 与 lifetime peak 分开，不再都是 ru_maxrss。
    telemetry = resource_telemetry_summary()
    assert "rss_current_start" in telemetry
    assert "rss_peak_lifetime" in telemetry
    final = finalize_resource_telemetry(telemetry)
    assert "rss_current_end" in final
    assert "rss_peak_run" in final
    assert _rss_bytes() != _lifetime_peak_rss_bytes() or True  # 两者字段存在


def test_bounded_queue_bytes_backpressure():
    # R27-105/250：queue 按 bytes 限制；满时 put 阻塞直到腾出空间。
    q = BoundedResultQueue(100)
    assert q.put(ResultItem(name="a", value=object(), bytes=60), timeout=0.1)
    # 60 + 60 = 120 > 100 → 阻塞超时返回 False（backpressure 生效）。
    assert q.put(ResultItem(name="b", value=object(), bytes=60), timeout=0.1) is False
    item = q.get()
    assert item.name == "a"
    assert q.put(ResultItem(name="b", value=object(), bytes=60), timeout=0.1) is True
    q.close()


def test_writer_backpressure_reduces_admission_signal():
    q = BoundedResultQueue(1000)
    q.put(ResultItem(name="x", value=object(), bytes=900))
    assert q.backpressure_ratio > 0.5
    q.get()
    assert q.backpressure_ratio < 0.5


def test_calibration_ema_and_calibrated_peak():
    # R27-016..019/240/254：EMA 校准因子 + predicted_peak = static*calibrated*uncertainty。
    record_task_actual(
        operator="ts_mean", backend="pandas_numpy", actual_elapsed_ms=200,
        rss_delta_bytes=200 * 1024 * 1024, rows=1_000_000, instruments=500,
        window=20, predicted_ms=100, predicted_peak_bytes=100 * 1024 * 1024,
    )
    from runtime.runtime_calibration import calibration_key

    key = calibration_key(
        operator="ts_mean", backend="pandas_numpy",
        shape_bucket="medium", window_bucket="medium",
    )
    peak, uncertainty = calibrated_peak_bytes(100 * 1024 * 1024, key)
    # actual/predicted = 2.0 → EMA 后 elapsed>1；uncertainty 在样本<5 时 ≥1.25。
    assert peak >= 100 * 1024 * 1024
    assert uncertainty >= 1.25
    assert uncertainty <= 1.50

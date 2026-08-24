# -*- coding: utf-8 -*-
"""R42-051..055/059 focused streaming sink and cache reconciliation gates."""
from __future__ import annotations

import threading
import time

import pytest

from factor_engine.runtime.streaming_result_sink import (
    BoundedResultQueue,
    ResultItem,
    StreamingResultSink,
    _classify_write_error,
)


def test_queue_budget_is_total_and_remainder_is_deterministic():
    sink = StreamingResultSink(
        writer=lambda batch: None,
        queue_bytes=10,
        writer_threads=3,
        partition_key=lambda item: item.name,
    )

    capacities = [queue.max_bytes for queue in sink._worker_queues]
    assert capacities == [4, 3, 3]
    assert sum(capacities) == 10
    assert sink.summary()["total_queue_capacity_bytes"] == 10


def test_rejected_submit_does_not_increment_accepted():
    sink = StreamingResultSink(writer=lambda batch: None, queue_bytes=8)
    sink.queue.close()

    assert sink.submit("closed", b"x", bytes=1) is False
    assert sink.summary()["accepted"] == 0


def test_put_timeout_is_one_absolute_budget_across_notifications():
    queue = BoundedResultQueue(10)
    assert queue.put(ResultItem("resident", b"x" * 10, bytes=10))
    stop = threading.Event()

    def notifier() -> None:
        while not stop.wait(0.01):
            with queue._lock:
                queue._lock.notify_all()

    thread = threading.Thread(target=notifier)
    thread.start()
    started = time.monotonic()
    try:
        assert queue.put(ResultItem("blocked", b"y", bytes=1), timeout=0.12) is False
    finally:
        stop.set()
        thread.join()

    assert time.monotonic() - started < 0.35


def test_put_returns_false_when_queue_closes_while_blocked():
    queue = BoundedResultQueue(1)
    assert queue.put(ResultItem("resident", b"x", bytes=1))
    result = {}

    def blocked_put() -> None:
        result["ok"] = queue.put(ResultItem("blocked", b"y", bytes=1), timeout=1.0)

    thread = threading.Thread(target=blocked_put)
    thread.start()
    time.sleep(0.02)
    queue.close()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert result["ok"] is False
    assert queue.queued_count == 1


    queue = BoundedResultQueue(10)
    oversized = ResultItem("large", b"x" * 25, bytes=25)

    assert queue.put(oversized, timeout=0.01) is True
    assert queue.current_bytes == 25
    assert queue.backpressure_ratio == 2.5
    assert queue.put(ResultItem("next", b"y", bytes=1), timeout=0.05) is False
    assert queue.get(timeout=0.01) is oversized
    assert queue.current_bytes == 0


def test_unknown_writer_error_is_permanent_without_retry():
    calls = 0

    def writer(batch):
        nonlocal calls
        calls += 1
        raise TypeError("unexpected writer contract")

    assert _classify_write_error(TypeError("logic bug")) == "permanent_unknown"
    sink = StreamingResultSink(writer=writer, queue_bytes=16, batch_size=1)
    sink.start()
    assert sink.submit("bad", b"x", bytes=1)
    with pytest.raises(RuntimeError, match="writer fatal"):
        sink.finish()
    assert calls == 1
    assert sink.summary()["retried"] == 1


def test_cache_release_reconciles_before_clear_and_records_outcome(monkeypatch):
    from factor_engine.cache.session import ExecutionCacheSession
    from factor_engine.runtime.resource_governor import global_memory_governor

    session = ExecutionCacheSession(execution_id="r42-release", strict=False)
    session.buffer_store.put("l0", b"a" * 11, bytes_=11)
    session.set_panel("l1", b"b" * 7)
    gov = global_memory_governor()
    gov.reserve_accounting(session._l0_layer, 11)
    gov.reserve_accounting(session._l1_layer, 7)

    seen = {}
    monkeypatch.setattr(
        session.buffer_store,
        "reconciliation",
        lambda *args, **kwargs: (
            seen.setdefault("l0_present_before_clear", "l0" in session.shared_result_cache),
            {
                "accounted_bytes": 11,
                "sampled_actual_bytes": 11,
                "sampled_keys": 1,
                "total_keys": 1,
                "extrapolated": False,
                "drift_bytes": 0,
                "reconciliations": 1,
            },
        )[1],
    )
    monkeypatch.setattr(
        session,
        "_estimate_cache_values",
        lambda values: sum(len(value) for value in values.values()),
    )

    session.release()

    assert seen["l0_present_before_clear"] is True
    assert session.shared_result_cache == {}
    assert session.panel_cache == {}
    assert session.panel_cache_store.store == {}
    report = session.to_dict()["release_reconciliation"]
    assert report == {
        "released_bytes": 18,
        "orphan_bytes": 0,
        "governor_declared_bytes": 18,
        "before_actual_bytes": 18,
        "after_actual_bytes": 0,
    }
    assert session._l0_layer not in gov._usage
    assert session._l1_layer not in gov._usage

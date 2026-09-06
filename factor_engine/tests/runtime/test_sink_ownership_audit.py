import errno
import threading
import gc
import weakref

import pytest

from factor_engine.runtime.streaming_result_sink import (
    BoundedResultQueue, ResultItem, StreamingResultSink, _classify_write_error,
)
from factor_engine.runtime.adaptive_batch_scheduler import classify_error


def test_oversized_rejected_promptly():
    q = BoundedResultQueue(10)
    with pytest.raises(ValueError, match="budget"):
        q.put(ResultItem("x", object(), bytes=11), timeout=0.01)
    assert q.current_bytes == 0


def test_dequeue_keeps_ownership_until_release():
    q = BoundedResultQueue(10)
    item = ResultItem("x", object(), bytes=10)
    assert q.put(item)
    assert q.get() is item
    assert q.current_bytes == 10
    assert not q.put(ResultItem("y", object(), bytes=1), timeout=0.01)
    q.release(item)
    assert q.current_bytes == 0
    assert q.put(ResultItem("y", object(), bytes=1), timeout=0.01)


def test_unknown_size_fails_closed(monkeypatch):
    monkeypatch.setattr("factor_engine.runtime.resource_governor.estimate_object_bytes", lambda _: 0)
    with pytest.raises(ValueError, match="size"):
        BoundedResultQueue(10).put(ResultItem("x", object()))


def test_multiple_writers_need_partition():
    with pytest.raises(ValueError, match="partition"):
        StreamingResultSink(writer=lambda _: None, queue_bytes=10, writer_threads=2)


@pytest.mark.parametrize("code,expected", [(errno.ENOMEM, "oom"), (errno.ENOSPC, "permanent"), (errno.EDQUOT, "permanent"), (errno.EACCES, "permanent"), (errno.ECONNRESET, "transient")])
def test_errno_precedes_message(code, expected):
    exc = OSError(code, "remote retryable transient")
    assert classify_error(exc) == expected
    assert _classify_write_error(exc) == expected


def test_slow_writer_stays_charged(monkeypatch):
    monkeypatch.setattr("factor_engine.runtime.streaming_result_sink._bytes_of", lambda _: 10)
    entered, done = threading.Event(), threading.Event()
    def writer(batch):
        entered.set()
        assert done.wait(3)
    sink = StreamingResultSink(writer=writer, queue_bytes=10)
    sink.start()
    try:
        assert sink.submit("x", object())
        assert entered.wait(2)
        assert sink.queue.current_bytes == 10
        assert sink.backpressure_ratio == 1
        assert not sink.queue.put(ResultItem("y", object(), bytes=1), timeout=0.01)
    finally:
        done.set()
        sink.finish()
    assert sink.queue.current_bytes == 0


def test_shrink_rejects_impossible_waiter():
    q = BoundedResultQueue(10)
    first = ResultItem("first", object(), bytes=10)
    q.put(first)
    errors = []
    def put():
        try:
            q.put(ResultItem("second", object(), bytes=8), timeout=2)
        except ValueError as exc:
            errors.append(exc)
    producer = threading.Thread(target=put)
    producer.start()
    q.set_target_bytes(5)
    producer.join(1)
    assert not producer.is_alive()
    assert len(errors) == 1
    assert q.current_bytes == 10  # existing ownership survives shrink
    assert q.get() is first
    q.release(first)
    assert q.current_bytes == 0


def test_partition_budget_and_zero_capacity_are_not_overallocated():
    sink = StreamingResultSink(writer=lambda _: None, queue_bytes=1,
                               writer_threads=2, partition_key=lambda x: x.name)
    assert sum(q.max_bytes for q in sink._worker_queues) == 1
    with pytest.raises(ValueError, match="budget"):
        sink._worker_queues[1].put(ResultItem("x", object(), bytes=1))


def test_drain_transfers_and_release_is_identity_checked():
    q = BoundedResultQueue(10)
    first = ResultItem("x", object(), bytes=10)
    q.put(first)
    with pytest.raises(ValueError, match="identity"):
        q.put(first)
    assert q.drain() == [first]
    assert q.current_bytes == 10
    with pytest.raises(ValueError, match="owned"):
        q.release(ResultItem("x", first.value, bytes=10))
    q.release(first)
    with pytest.raises(ValueError, match="owned"):
        q.release(first)


def test_queue_retains_strong_identity_until_release():
    q = BoundedResultQueue(10)
    item = ResultItem("x", object(), bytes=10)
    reference = weakref.ref(item)
    q.put(item)
    assert q.get() is item
    del item
    gc.collect()
    assert reference() is not None
    held = reference()
    held.bytes = 999  # original reservation must not be changed by mutation
    q.release(held)
    assert q.current_bytes == 0
    del held
    gc.collect()
    assert reference() is None


def test_duplicate_producers_recheck_identity_after_wait(monkeypatch):
    q = BoundedResultQueue(10)
    first = ResultItem("full", object(), bytes=10)
    same = ResultItem("same", object(), bytes=2)
    q.put(first)
    waiting = threading.Event()
    original_wait = q._lock.wait
    arrivals = []
    def observed_wait(timeout=None):
        arrivals.append(threading.get_ident())
        if len(set(arrivals)) == 2:
            waiting.set()
        return original_wait(timeout)
    monkeypatch.setattr(q._lock, "wait", observed_wait)
    results = []
    def submit():
        try:
            results.append(q.put(same, timeout=2))
        except ValueError:
            results.append("duplicate")
    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        assert waiting.wait(1)
        assert q.get() is first
        q.release(first)
    finally:
        for thread in threads:
            thread.join(3)
    assert sorted(map(str, results)) == ["True", "duplicate"]
    assert q.current_bytes == 2
    assert q.get() is same
    q.release(same)


def test_invalid_total_budget_rejected_and_shrink_zero_honored():
    with pytest.raises(ValueError, match="budget"):
        StreamingResultSink(writer=lambda _: None, queue_bytes=0)
    sink = StreamingResultSink(writer=lambda _: None, queue_bytes=10)
    sink.set_target_bytes(0)
    assert sink.queue.max_bytes == 0
    with pytest.raises(ValueError, match="budget"):
        sink.queue.put(ResultItem("x", object(), bytes=1))


@pytest.mark.parametrize("error,retries", [(OSError(errno.ENOSPC, "retryable"), 0),
                                          (MemoryError("retryable"), 0),
                                          (ConnectionError("reset"), 2)])
def test_write_retries_keep_reservation_and_release_terminally(monkeypatch, error, retries):
    monkeypatch.setattr("factor_engine.runtime.streaming_result_sink._bytes_of", lambda _: 10)
    calls = []
    def writer(batch):
        assert sink.queue.current_bytes == 10
        calls.append(len(batch))
        raise error
    sink = StreamingResultSink(writer=writer, queue_bytes=10)
    sink.start()
    assert sink.submit("x", object())
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert len(calls) == retries + 1
    assert sink.summary()["retried"] == retries
    assert sink.queue.current_bytes == 0


def test_timeout_does_not_release_running_owner(monkeypatch):
    monkeypatch.setattr("factor_engine.runtime.streaming_result_sink._bytes_of", lambda _: 10)
    entered, done = threading.Event(), threading.Event()
    def writer(batch):
        entered.set()
        assert done.wait(3)
    sink = StreamingResultSink(writer=writer, queue_bytes=10, join_timeout=0.01)
    sink.start()
    try:
        sink.submit("x", object())
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="alive"):
            sink.finish()
        assert sink.queue.current_bytes == 10
    finally:
        done.set()
        for thread in sink._threads:
            thread.join(2)
    assert sink.queue.current_bytes == 0


def test_partition_distribution_and_same_partition_order(monkeypatch):
    monkeypatch.setattr("factor_engine.runtime.streaming_result_sink._bytes_of", lambda _: 10)
    entered, done = threading.Event(), threading.Event()
    writes = []
    def writer(batch):
        writes.append((threading.current_thread().name, batch[0].name))
        if batch[0].name == "a1":
            entered.set()
            assert done.wait(2)
    sink = StreamingResultSink(writer=writer, queue_bytes=60, writer_threads=2,
                               partition_key=lambda item: item.name[0])
    sink.start()
    try:
        sink.submit("a1", object())
        assert entered.wait(1)
        sink.submit("b1", object())
        sink.submit("a2", object())
    finally:
        done.set()
        sink.finish()
    a = [(thread, name) for thread, name in writes if name.startswith("a")]
    assert [name for _, name in a] == ["a1", "a2"]
    assert a[0][0] == a[1][0]
    assert next(thread for thread, name in writes if name == "b1") != a[0][0]


@pytest.mark.parametrize("error,expected", [(TimeoutError("query complexity"), "permanent"),
                                           (TimeoutError("network"), "transient"),
                                           (RuntimeError("unknown"), "unknown")])
def test_compute_timeout_and_unknown(error, expected):
    assert classify_error(error) == expected
    assert _classify_write_error(error) == ("permanent_unknown" if expected == "unknown" else expected)

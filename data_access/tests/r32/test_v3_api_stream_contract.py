from __future__ import annotations

import threading
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from data_access.read.read_contract import DataSnapshot, ReadLineage, ReadResult, ReadStats
from data_access.security.execution_context import (
    DataAccessExecutionContext,
    current_execution_context,
)
from data_access.service.app import create_app
from data_access.service.config import ServiceSettings
from data_access.service.http_resource_management import (
    ExecutionScopedIterator,
    StreamResourceLifecycle,
)


def test_execution_scope_is_applied_to_each_interleaved_next_and_restored():
    seen: list[tuple[str, str | None]] = []

    def source(label: str):
        for _ in range(2):
            active = current_execution_context()
            seen.append((label, active.request_id if active else None))
            yield label

    a = ExecutionScopedIterator(source("a"), DataAccessExecutionContext(request_id="req-a"))
    b = ExecutionScopedIterator(source("b"), DataAccessExecutionContext(request_id="req-b"))
    assert [next(a), next(b), next(a), next(b)] == ["a", "b", "a", "b"]
    assert seen == [("a", "req-a"), ("b", "req-b"), ("a", "req-a"), ("b", "req-b")]
    assert current_execution_context() is None


def test_interleaved_http_streams_keep_request_context_and_close_once():
    settings = ServiceSettings(api_key="key", max_concurrency=2)
    store = MagicMock()
    barrier = threading.Barrier(2, timeout=5)
    seen: list[tuple[str, str | None]] = []
    closes: dict[str, int] = {"req-a": 0, "req-b": 0}
    seen_lock = threading.Lock()

    class Reader:
        def __init__(self, request_id: str):
            self.request_id = request_id
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == 2:
                raise StopIteration
            barrier.wait()
            active = current_execution_context()
            with seen_lock:
                seen.append((self.request_id, active.request_id if active else None))
            self.index += 1
            return pa.record_batch([[self.index]], names=["value"])

        def close(self):
            with seen_lock:
                closes[self.request_id] += 1

    def open_reader(*args, **kwargs):
        active = current_execution_context()
        assert active is not None
        return Reader(active.request_id)

    store.read_arrow_stream.side_effect = open_reader
    app = create_app(settings)
    statuses: list[int] = []

    def request(request_id: str):
        response = TestClient(app).post(
            "/v1/read/arrow-stream",
            json={"dataset": "ds", "columns": ["value"]},
            headers={"X-API-Key": "key", "X-Request-ID": request_id},
        )
        statuses.append(response.status_code)

    threads = [threading.Thread(target=request, args=(request_id,)) for request_id in closes]
    with patch("data_access.service.app.get_store", return_value=store):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
    assert statuses == [200, 200]
    assert sorted(seen) == sorted(
        [("req-a", "req-a"), ("req-a", "req-a"), ("req-b", "req-b"), ("req-b", "req-b")]
    )
    assert closes == {"req-a": 1, "req-b": 1}


@pytest.mark.parametrize("call_count", [1, 2, 5])
def test_stream_lifecycle_closes_reader_and_slot_exactly_once(call_count: int):
    reader = MagicMock()
    releases = 0
    lock = threading.Lock()

    def release():
        nonlocal releases
        with lock:
            releases += 1

    lifecycle = StreamResourceLifecycle(reader, release)
    threads = [threading.Thread(target=lifecycle.close) for _ in range(call_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert lifecycle.closed
    reader.close.assert_called_once_with()
    assert releases == 1


def test_scoped_iterator_close_waits_for_active_next_before_returning():
    entered = threading.Event()
    allow_next = threading.Event()
    closed = threading.Event()

    class Reader:
        def __iter__(self):
            return self

        def __next__(self):
            entered.set()
            assert allow_next.wait(5)
            return 1

        def close(self):
            closed.set()

    iterator = ExecutionScopedIterator(Reader(), DataAccessExecutionContext(request_id="race"))
    next_thread = threading.Thread(target=lambda: next(iterator))
    close_thread = threading.Thread(target=iterator.close)
    next_thread.start()
    assert entered.wait(2)
    close_thread.start()
    assert not closed.wait(0.05)
    allow_next.set()
    next_thread.join(2)
    close_thread.join(2)
    assert closed.is_set()
    with pytest.raises(StopIteration):
        next(iterator)


def test_scoped_iterator_attempts_source_close_when_distinct_iterator_close_raises():
    source_closed = False

    class ActualIterator:
        def __next__(self):
            raise StopIteration

        def close(self):
            raise OSError("iterator close failed")

    class Source:
        def __iter__(self):
            return ActualIterator()

        def close(self):
            nonlocal source_closed
            source_closed = True

    iterator = ExecutionScopedIterator(Source(), DataAccessExecutionContext(request_id="close"))
    with pytest.raises(StopIteration):
        next(iterator)
    with pytest.raises(OSError, match="iterator close failed"):
        iterator.close()
    assert source_closed


def _result(rows: int) -> ReadResult:
    from datetime import datetime, timezone

    table = pa.table({"value": list(range(rows))})
    snapshot = DataSnapshot(
        snapshot_id="snap", dataset="ds", registry_hash="r", schema_hash="s",
        file_manifest_hash="f", files=(), created_at=datetime.now(timezone.utc),
    )
    return ReadResult(
        table=table,
        snapshot=snapshot,
        stats=ReadStats(rows=rows, bytes=table.nbytes, elapsed_ms=1.0, paths=()),
        lineage=ReadLineage(dataset="ds"),
    )


def test_json_preview_has_independent_pre_materialization_row_limit(monkeypatch):
    settings = ServiceSettings(
        api_key="key", max_rows=1000, max_bytes=1_000_000,
        json_preview_max_rows=2, json_preview_max_bytes=1_000_000,
    )
    store = MagicMock()
    store.read_result.return_value = _result(3)
    with patch("data_access.service.app.get_store", return_value=store):
        response = TestClient(create_app(settings)).post(
            "/v1/read", json={"dataset": "ds", "columns": ["value"], "format": "json"},
            headers={"X-API-Key": "key"},
        )
    assert response.status_code == 413
    assert "JSON preview" in response.json()["detail"]


def test_json_preview_encodes_arrow_scalars_without_nonfinite_json():
    result = _result(1)
    result = ReadResult(
        table=pa.table({
            "timestamp": [datetime(2026, 9, 7, 1, 2, 3)],
            "decimal": pa.array([Decimal("123.4500")], type=pa.decimal128(10, 4)),
            "binary": [b"\x00\xff"],
            "nan": [float("nan")],
            "inf": [float("inf")],
        }),
        snapshot=result.snapshot,
        stats=ReadStats(rows=1, bytes=64, elapsed_ms=1.0, paths=()),
        lineage=result.lineage,
    )
    store = MagicMock()
    store.read_result.return_value = result
    with patch("data_access.service.app.get_store", return_value=store):
        response = TestClient(create_app(ServiceSettings(api_key="key"))).post(
            "/v1/read", json={"dataset": "ds", "columns": ["timestamp"], "format": "json"},
            headers={"X-API-Key": "key"},
        )
    assert response.status_code == 200
    row = response.json()["data"][0]
    assert row == {
        "timestamp": "2026-09-07T01:02:03",
        "decimal": "123.4500",
        "binary": "base64:AP8=",
        "nan": None,
        "inf": None,
    }


@pytest.mark.parametrize("failure", ["iter", "schema"])
def test_stream_setup_failure_closes_reader_and_releases_slot(failure: str):
    settings = ServiceSettings(api_key="key", max_concurrency=1)
    closes = 0

    class BadReader:
        def __iter__(self):
            if failure == "iter":
                raise RuntimeError("iter failed")
            return self

        def __next__(self):
            if failure == "schema":
                return object()
            raise AssertionError("unreachable")

        def close(self):
            nonlocal closes
            closes += 1

    store = MagicMock()
    store.read_arrow_stream.side_effect = lambda *a, **k: BadReader()
    app = create_app(settings)
    with patch("data_access.service.app.get_store", return_value=store):
        with pytest.raises(Exception, match="iter failed|schema"):
            TestClient(app).post(
                "/v1/read/arrow-stream", json={"dataset": "ds", "columns": ["value"]},
                headers={"X-API-Key": "key"},
            )
        # A second call reaches the same setup failure, rather than a 503 slot leak.
        with pytest.raises(Exception, match="iter failed|schema"):
            TestClient(app).post(
                "/v1/read/arrow-stream", json={"dataset": "ds", "columns": ["value"]},
                headers={"X-API-Key": "key"},
            )
    assert closes == 2


def test_arrow_stream_advertises_eos_completion_contract():
    settings = ServiceSettings(api_key="key", max_concurrency=1)
    store = MagicMock()
    store.read_arrow_stream.return_value = iter([pa.record_batch([[1]], names=["value"])])
    with patch("data_access.service.app.get_store", return_value=store):
        response = TestClient(create_app(settings)).post(
            "/v1/read/arrow-stream", json={"dataset": "ds", "columns": ["value"]},
            headers={"X-API-Key": "key"},
        )
    assert response.status_code == 200
    assert response.headers["X-Stream-Completion-Contract"] == "arrow-ipc-eos"
    assert response.content.endswith(b"\xff\xff\xff\xff\x00\x00\x00\x00")


def test_production_rejects_multi_worker_process_local_admission(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_WORKERS", "2")
    settings = ServiceSettings(production_mode=True)
    with pytest.raises(RuntimeError, match="process-local"):
        settings.validate_resource_boundary()


def test_version_reports_honest_process_resource_boundary():
    response = TestClient(create_app(ServiceSettings(max_concurrency=3))).get("/version")
    assert response.status_code == 200
    assert response.json()["resource_boundary"] == {
        "admission_scope": "process",
        "max_concurrent_http_queries_per_process": 3,
        "distributed_admission": False,
    }

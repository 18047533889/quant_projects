"""Synthetic U05 approval/broker tests; no external dataset is read."""
from types import SimpleNamespace

import pyarrow as pa
import pytest

from factor_engine.runtime.default_execution_policy import ExecutionPurpose
from factor_engine.storage.sources.data_access_source import (
    ApprovedSnapshotMismatch, MissingDataDependencyError,
)
from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource


def _logical(*, snapshots=None, digests=None, broker=None):
    source = object.__new__(LQTPLogicalDataSource)
    source.inner = SimpleNamespace(
        _approved_source_snapshot_tokens=dict(snapshots or {}),
        _approved_source_content_digests=dict(digests or {}),
        _cache_broker=broker, instrument_filter=["000001.SZ"],
        start_date="2024-01-01", end_date="2024-01-31",
        run_mode="production", production=True, strict_unknown_fields=True,
        enforce_mining_gate=False, snapshot_now_only=True,
        mining_coverage_threshold=.9, pit_enforce=True,
        semantic_filters={"IndustrySource": "sw_l1"},
        execution_purpose=ExecutionPurpose(),
    )
    source._cache = {}
    return source


def test_managed_dependency_requires_dataset_specific_digest():
    source = _logical(digests={"daily": "a" * 32})
    with pytest.raises(MissingDataDependencyError) as caught:
        source._approved_dependency("stock_income")
    assert caught.value.reason_code == "DATA_SOURCE_MISSING"
    assert "stock_income" in str(caught.value)


def test_child_inherits_approved_identity_scope_and_same_broker(monkeypatch):
    broker = object()
    source = _logical(
        snapshots={"stock_income": "snapshot-7"},
        digests={"stock_income": "a" * 32}, broker=broker,
    )
    child = SimpleNamespace(bound_snapshot=None, bound_digest=None, bound_broker=None,
                            close=lambda: None)
    child.bind_approved_snapshot_token = lambda value: setattr(child, "bound_snapshot", value)
    child.bind_approved_content_digest = lambda value: setattr(child, "bound_digest", value)
    child.bind_resource_broker = lambda value: setattr(child, "bound_broker", value)
    seen = {}

    def make_child(**kwargs):
        seen.update(kwargs)
        return child

    monkeypatch.setattr(
        "factor_engine.storage.sources.lqtp_logical_source.DataAccessSource",
        make_child,
    )
    actual = source._child("stock_income")
    assert actual is child
    assert seen["dataset"] == "stock_income"
    assert seen["instrument_filter"] == ["000001.SZ"]
    assert (seen["start_date"], seen["end_date"]) == ("2024-01-01", "2024-01-31")
    assert seen["run_mode"] == "production" and seen["production"] is True
    assert seen["strict_unknown_fields"] is True and seen["pit_enforce"] is True
    assert seen["semantic_filters"] == {"IndustrySource": "sw_l1"}
    assert child.bound_snapshot == "snapshot-7"
    assert child.bound_digest == "a" * 32
    assert child.bound_broker is broker


def _financial_store(digest, *, execute_error=None):
    counters = {"prepare": 0, "execute": 0, "release": 0}
    reservation = object()
    snapshot = SimpleNamespace(dataset="stock_income", content_digest=digest)
    prepared = SimpleNamespace(resolved_source_snapshot=snapshot,
                               resource_reservation=reservation)

    def release(value):
        assert value is reservation
        counters["release"] += 1

    def prepare(dataset, **kwargs):
        counters["prepare"] += 1
        counters["prepare_args"] = (dataset, kwargs)
        return prepared

    def execute(value):
        assert value is prepared
        counters["execute"] += 1
        try:
            if execute_error is not None:
                raise execute_error
            table = pa.table({
                "Symbol": ["000001.SZ"],
                "ReportPeriodEndDate": ["2023-12-31"],
                "PubDate": ["2024-01-15"], "Revenue": [1.0],
            })
            return SimpleNamespace(table=table, snapshot=SimpleNamespace(snapshot_id="s1"))
        finally:
            release(reservation)

    store = SimpleNamespace(
        get_dataset=lambda dataset: SimpleNamespace(instrument_column="Symbol"),
        prepare_read=prepare, execute_prepared_read=execute,
        _pipeline=SimpleNamespace(release_reservation=release),
    )
    return store, counters


def test_financial_prepare_is_production_exact_and_success_releases_once(monkeypatch):
    digest = "a" * 32
    source = _logical(digests={"stock_income": digest})
    store, counters = _financial_store(digest)
    monkeypatch.setattr(
        "factor_engine.storage.sources.data_access_source._get_store", lambda: store)
    frame = source._financial_raw_multi("stock_income", ["Revenue"])
    assert list(frame["Revenue"]) == [1.0]
    dataset, kwargs = counters["prepare_args"]
    assert dataset == "stock_income"
    assert kwargs["run_mode"] == "production"
    assert kwargs["snapshot_policy"] == "fail_if_changed"
    assert kwargs["mode"] == "event"
    assert kwargs["instrument_filter"] == ["000001.SZ"]
    assert counters["execute"] == counters["release"] == 1


def test_financial_digest_mismatch_releases_without_execute(monkeypatch):
    source = _logical(digests={"stock_income": "a" * 32})
    store, counters = _financial_store("b" * 32)
    monkeypatch.setattr(
        "factor_engine.storage.sources.data_access_source._get_store", lambda: store)
    with pytest.raises(ApprovedSnapshotMismatch):
        source._financial_raw_multi("stock_income", ["Revenue"])
    assert counters["prepare"] == counters["release"] == 1
    assert counters["execute"] == 0


def test_financial_execute_failure_releases_once(monkeypatch):
    digest = "a" * 32
    source = _logical(digests={"stock_income": digest})
    store, counters = _financial_store(digest, execute_error=RuntimeError("read failed"))
    monkeypatch.setattr(
        "factor_engine.storage.sources.data_access_source._get_store", lambda: store)
    with pytest.raises(RuntimeError, match="read failed"):
        source._financial_raw_multi("stock_income", ["Revenue"])
    assert counters["execute"] == counters["release"] == 1

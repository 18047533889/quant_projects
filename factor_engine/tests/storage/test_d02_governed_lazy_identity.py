from types import SimpleNamespace

import pyarrow as pa
import pytest
from datetime import datetime, timezone
import polars as pl

from data_access.read.query_budget import QueryBudget
from data_access.read.read_contract import DataSnapshot, FileVersion, ReadLineage
from data_access.read.scan_handle import ScanHandle
from factor_engine.backend.polars_lazy import (
    LazyColumnBundle,
    _collect_arrow_table,
    scan_dataset_columns,
)
from factor_engine.storage.sources.data_access_source import ApprovedSnapshotMismatch


class _Pipeline:
    def __init__(self):
        self.releases = 0
        self.counters = SimpleNamespace(execute=0)

    def release_reservation(self, _reservation):
        self.releases += 1
        self.counters = SimpleNamespace(execute=0)

    def verify_before(self, *_args, **_kwargs):
        return None

    def verify_after(self, *_args, **_kwargs):
        return None

class _Store:
    def __init__(self):
        self._pipeline = _Pipeline()


class _NeverSelected:
    def select(self, _columns):
        raise AssertionError("identity mismatch must fail before select/collect")


def _scan(digest: str) -> tuple[ScanHandle, _Store]:
    store = _Store()
    prepared = SimpleNamespace(
        resolved_source_snapshot=SimpleNamespace(dataset="demo", content_digest=digest),
        resource_reservation=object(),
    )
    return ScanHandle(
        _lf=_NeverSelected(),
        snapshot=SimpleNamespace(files=()),
        budget=SimpleNamespace(),
        lineage=SimpleNamespace(dataset="demo"),
        _store=store,
        _prepared=prepared,
    ), store


def test_approved_scan_precheck_fails_before_collect_and_releases():
    scan, store = _scan("b" * 32)
    with pytest.raises(ApprovedSnapshotMismatch, match="approved content"):
        _collect_arrow_table(
            scan,
            select_cols=["value"],
            dataset="demo",
            approved_content_digest="a" * 32,
        )
    assert store._pipeline.releases == 1


def test_approved_scan_checks_actual_result_after_collect(monkeypatch):
    scan, store = _scan("a" * 32)
    monkeypatch.setattr(ScanHandle, "select", lambda self, _columns: self, raising=False)

    def collect(self):
        self._prepared.resolved_source_snapshot = SimpleNamespace(
            dataset="demo", content_digest="b" * 32
        )
        self.close()
        return SimpleNamespace(
            table=pa.table({"value": [1]}),
            snapshot=SimpleNamespace(
                dataset="demo", content_digest="b" * 32, snapshot_id="changed"
            ),
        )

    monkeypatch.setattr(ScanHandle, "collect", collect)
    with pytest.raises(ApprovedSnapshotMismatch, match="approved content"):
        _collect_arrow_table(
            scan,
            select_cols=["value"],
            dataset="demo",
            approved_content_digest="a" * 32,
        )
    assert store._pipeline.releases == 1


def test_warm_bundle_rechecks_request_authority_and_releases():
    scan, store = _scan("a" * 32)
    bundle = LazyColumnBundle(
        lf=scan,
        time_column="time",
        instrument_column="instrument",
        physical_columns=("value",),
        output_names={},
        normalize_timestamp=False,
        timestamp_unit=None,
        dataset="demo",
        approved_content_digest="b" * 32,
        _materialized={"value": object()},
    )
    with pytest.raises(ApprovedSnapshotMismatch, match="bundle authority"):
        scan_dataset_columns(
            SimpleNamespace(),
            "demo",
            physical_columns=["value"],
            time_column="time",
            instrument_column="instrument",
            time_range=None,
            instrument_filter=None,
            output_names=None,
            normalize_timestamp=False,
            timestamp_unit=None,
            params=None,
            bundle=bundle,
            approved_content_digest="a" * 32,
        )
    assert store._pipeline.releases == 1


def test_real_tiny_parquet_collect_uses_governed_scan(tmp_path):
    path = tmp_path / "authorized-synthetic.parquet"
    pl.DataFrame(
        {"time": ["2024-01-02"], "instrument": ["A"], "value": [0.01]}
    ).write_parquet(path)
    stat = path.stat()
    digest = "a" * 32
    snapshot = DataSnapshot(
        snapshot_id="synthetic-snapshot",
        dataset="demo",
        registry_hash="synthetic-registry",
        schema_hash="synthetic-schema",
        file_manifest_hash="synthetic-manifest",
        files=(FileVersion(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns),),
        created_at=datetime.now(timezone.utc),
    )
    store = _Store()
    prepared = SimpleNamespace(
        resolved_source_snapshot=SimpleNamespace(dataset="demo", content_digest=digest),
        snapshot_policy="pin",
        resource_reservation=object(),
    )
    scan = ScanHandle(
        _lf=pl.scan_parquet(path), snapshot=snapshot, budget=QueryBudget(),
        lineage=ReadLineage(dataset="demo", columns=("value",)),
        _store=store, _prepared=prepared,
    )
    identity = {}
    table = _collect_arrow_table(
        scan, select_cols=["value"], dataset="demo",
        approved_content_digest=digest, identity_out=identity,
    )
    assert table.column("value").to_pylist() == [0.01]
    assert identity == {"snapshot_id": "synthetic-snapshot", "content_digest": digest}
    assert store._pipeline.counters.execute == 1
    assert store._pipeline.releases == 1

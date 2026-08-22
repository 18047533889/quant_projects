"""R21 DataReadIdentity regression tests.

Covers:
    1. DataReadIdentity digest covers every declared component (uniqueness).
    2. Every DataAccess read records its DataReadIdentity (all paths).
    3. Production SemanticField missing availability/mining/unit/PIT fails closed.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.read.data_read_identity import (
    DataReadIdentity,
    assert_semantic_field_production_ready,
    build_data_read_identity,
)
from data_access.read.data_request import DataRequest
from data_access.read.semantic_catalog import SemanticField
from data_access.registry.loader import StaticDataset, DatasetRegistry
from data_access.runtime.mode_identity import (
    reset_runtime_mode_identity,
    set_runtime_mode_identity,
)
from data_access.store import DataAccessStore, get_shared_engine


def _store(root: str, *, name: str = "demo", schema: dict | None = None, time_col: str | None = None, inst_col: str | None = None) -> DataAccessStore:
    ds = StaticDataset(
        name=name,
        access_mode="published",
        layout="plain",
        time_column=time_col,
        instrument_column=inst_col,
        hive_partitioning=False,
        union_by_name=False,
        root=root,
        glob="*.parquet",
        schema=schema or {"a": "int64", "b": "string"},
    )
    return DataAccessStore(registry=DatasetRegistry({name: ds}), engine=get_shared_engine())


def _table_rows(*, cols: dict[str, list], n: int = 3) -> pa.Table:
    return pa.table({k: (v * n) if isinstance(v, list) and len(v) == 1 else v for k, v in cols.items()})


def _write(root: str) -> None:
    pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), f"{root}/f.parquet")


def _write_daily(root: str) -> None:
    pq.write_table(
        pa.table({"dt": ["2024-01-01", "2024-01-02"], "sym": ["a", "b"], "v": [1, 2]}),
        f"{root}/f.parquet",
    )


# ---------------------------------------------------------------------------
# 1. DataReadIdentity digest covers every declared component.
# ---------------------------------------------------------------------------

def test_read_identity_digest_covers_every_declared_component() -> None:
    base = DataReadIdentity(dataset="daily", revision="r1")
    assert len(base.digest) == 64
    assert all(ch in "0123456789abcdef" for ch in base.digest)

    variants = (
        DataReadIdentity(dataset="daily", revision="r2"),
        DataReadIdentity(dataset="daily", revision="r1", availability="next_trading_day"),
        DataReadIdentity(dataset="daily", revision="r1", calendar_identity="cal-1"),
        DataReadIdentity(dataset="daily", revision="r1", universe_snapshot="uni-1"),
        DataReadIdentity(dataset="daily", revision="r1", source_snapshot="snap-1"),
        DataReadIdentity(dataset="daily", revision="r1", grain="minute"),
        DataReadIdentity(dataset="daily", revision="r1", columns=("a",)),
        DataReadIdentity(dataset="daily", revision="r1", columns=("a", "b")),
        DataReadIdentity(dataset="daily", revision="r1", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 2))),
        DataReadIdentity(dataset="daily", revision="r1", instrument_filter=("600000.SH",)),
    )
    # DA-P0-04：session 只进执行身份，**不进内容 digest**——与 base 相同。
    assert DataReadIdentity(dataset="daily", revision="r1", session="sess-1").digest == base.digest
    assert all(v.digest != base.digest for v in variants)
    # 单列 ('a',) 与多列 ('a','b') 的 digest 不同。
    assert DataReadIdentity(dataset="daily", revision="r1", columns=("a", "b")).digest != DataReadIdentity(
        dataset="daily", revision="r1", columns=("a",)
    ).digest


def test_read_identity_is_immutable_and_hashable() -> None:
    rid = DataReadIdentity(dataset="daily")
    with pytest.raises(Exception):
        rid.dataset = "other"  # frozen dataclass → FrozenInstanceError
    assert rid in {rid}  # hashable


# ---------------------------------------------------------------------------
# 2. Every DataAccess read records its DataReadIdentity.
# ---------------------------------------------------------------------------

def test_store_read_records_data_read_identity() -> None:
    root = tempfile.mkdtemp()
    _write(root)
    store = _store(root)
    handle = store.read("demo", columns=["a"])
    rid = handle.read_identity
    assert rid is not None
    assert isinstance(rid, DataReadIdentity)
    assert rid.dataset == "demo"
    assert rid.source_snapshot  # exact physical source snapshot digest
    assert rid.calendar_identity  # frozen calendar world


def test_read_result_records_identity_with_prepared_source_snapshot() -> None:
    root = tempfile.mkdtemp()
    _write(root)
    store = _store(root)
    rr = store.read_result("demo", columns=["a"])
    assert rr.read_identity is not None
    assert rr.read_identity.dataset == "demo"
    assert rr.read_identity.source_snapshot


def test_plan_execute_read_records_identity() -> None:
    root = tempfile.mkdtemp()
    _write_daily(root)
    store = _store(
        root,
        name="daily",
        schema={"dt": "string", "sym": "string", "v": "int64"},
        time_col="dt",
        inst_col="sym",
    )
    plan = store.plan(DataRequest(fields=["v"], anchor="daily"))
    handle = plan.execute()
    assert handle.read_identity is not None
    assert handle.read_identity.dataset == "daily"


def test_read_joined_records_identity() -> None:
    root = tempfile.mkdtemp()
    _write_daily(root)
    store = _store(
        root,
        name="daily",
        schema={"dt": "string", "sym": "string", "v": "int64"},
        time_col="dt",
        inst_col="sym",
    )
    handle = store.read_joined("daily", fields={"daily": ["v"]})
    assert handle.read_identity is not None
    assert handle.read_identity.dataset == "daily"


def test_stream_read_records_identity() -> None:
    root = tempfile.mkdtemp()
    _write(root)
    store = _store(root)
    handle = store.read("demo", columns=["a"], result="stream")
    assert handle.read_identity is not None
    assert handle.read_identity.dataset == "demo"


# ---------------------------------------------------------------------------
# 3. Production SemanticField missing availability/mining/unit/PIT fails closed.
# ---------------------------------------------------------------------------

def test_production_semantic_field_missing_semantics_fails_closed() -> None:
    token = set_runtime_mode_identity("production")
    try:
        f = SemanticField(logical_name="v", dataset="daily", physical_name="v")
        with pytest.raises(Exception, match="pit_fidelity"):
            assert_semantic_field_production_ready(f)
    finally:
        reset_runtime_mode_identity(token)


def test_production_semantic_field_complete_passes() -> None:
    token = set_runtime_mode_identity("production")
    try:
        f = SemanticField(
            logical_name="close",
            dataset="daily",
            physical_name="Close",
            availability="same_day",
            mining_allowed=True,
            source_unit="percent",
            canonical_unit="ratio",
            pit_fidelity="knowledge_date_pit",
        )
        assert_semantic_field_production_ready(f)  # no raise
    finally:
        reset_runtime_mode_identity(token)


def test_research_semantic_field_missing_semantics_does_not_fail() -> None:
    token = set_runtime_mode_identity("interactive_research")
    try:
        f = SemanticField(logical_name="v", dataset="daily", physical_name="v")
        assert_semantic_field_production_ready(f)  # research: no raise
    finally:
        reset_runtime_mode_identity(token)


def test_build_data_read_identity_from_store_and_prepared() -> None:
    root = tempfile.mkdtemp()
    _write(root)
    store = _store(root)
    prepared = store.prepare_read("demo", columns=["a"])
    rid = build_data_read_identity(
        store,
        dataset="demo",
        columns=["a"],
        time_range=None,
        instrument_filter=None,
        universe=None,
        prepared=prepared,
    )
    assert rid.dataset == "demo"
    assert rid.source_snapshot == prepared.resolved_source_snapshot.content_digest
    assert rid.calendar_identity == store.calendar_snapshot_id()


# ---------------------------------------------------------------------------
# R22 P0: columns / time_range / instrument_filter are bound into the digest.
# ---------------------------------------------------------------------------

def test_columns_affect_identity_digest() -> None:
    base = DataReadIdentity(dataset="daily", revision="r1")
    same = DataReadIdentity(dataset="daily", revision="r1", columns=("a",))
    different = DataReadIdentity(dataset="daily", revision="r1", columns=("a", "b"))
    assert same.digest == same.digest
    assert same.digest != base.digest
    assert different.digest != same.digest

    # list 输入被折叠成 tuple（frozen dataclass 中允许）。
    listed = DataReadIdentity(dataset="daily", revision="r1", columns=["a"])
    assert listed.columns == ("a",)
    assert listed.digest == same.digest

    # time_range 端点稳定化（datetime → isoformat）；不同范围 → 不同 digest。
    rng_a = DataReadIdentity(
        dataset="daily", revision="r1", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 2))
    )
    rng_b = DataReadIdentity(
        dataset="daily", revision="r1", time_range=(datetime(2024, 1, 1), datetime(2024, 2, 1))
    )
    assert rng_a.digest != base.digest
    assert rng_a.digest != rng_b.digest
    assert rng_a.time_range == ("2024-01-01T00:00:00", "2024-01-02T00:00:00")

    # instrument_filter 进 digest。
    inst = DataReadIdentity(dataset="daily", revision="r1", instrument_filter=["600000.SH"])
    assert inst.digest != base.digest
    assert inst.instrument_filter == ("600000.SH",)
    assert inst.digest == DataReadIdentity(
        dataset="daily", revision="r1", instrument_filter=("600000.SH",)
    ).digest

    # to_dict 携带新字段。
    d = same.to_dict()
    assert d["columns"] == ("a",)
    assert "time_range" in d and "instrument_filter" in d

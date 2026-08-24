"""R21 DA-P0-01..04 DataReadIdentity red-team fix regression tests.

Covers:
    1. DA-P0-01: multi-field mixed-availability read retains all per-field
       semantics; identity hash differs from a collapsed one; order-independent.
    2. DA-P0-02: caller-provided digest is ignored/overridden by internal
       derivation; deserialization recomputes and validates.
    3. DA-P0-03: production/PIT-strict unavailable revision/calendar/universe/
       source-snapshot → fail closed; research → explicit UNKNOWN status.
    4. DA-P0-04: identical content under different request_id/session → SAME
       content identity hash; execution identity recorded separately.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from datetime import datetime

import pytest

from data_access.read.data_read_identity import (
    DataReadContentIdentity,
    DataReadIdentity,
    ReadExecutionIdentity,
    ResolvedFieldIdentity,
    build_data_read_identity,
)
from data_access.read.semantic_catalog import SemanticField
from data_access.runtime.mode_identity import (
    reset_runtime_mode_identity,
    set_runtime_mode_identity,
)


# ---------------------------------------------------------------------------
# DA-P0-01：多字段混合 availability 语义保留 + 顺序无关。
# ---------------------------------------------------------------------------

def _price_field() -> SemanticField:
    return SemanticField(
        logical_name="price",
        dataset="daily",
        physical_name="Close",
        market="ashare",
        availability="same_day",
        temporal_model="panel",
        grain="instrument",
        source_unit="yuan",
        canonical_unit="yuan",
        scale=1.0,
        pit_fidelity="knowledge_date_pit",
        knowledge_time="TradeDate",
    )


def _roe_field() -> SemanticField:
    return SemanticField(
        logical_name="roe",
        dataset="fundamental",
        physical_name="ROE",
        market="ashare",
        availability="next_trading_day",
        temporal_model="financial_event",
        grain="instrument",
        source_unit="percent",
        canonical_unit="ratio",
        scale=0.01,
        pit_fidelity="vintage_pit",
        knowledge_time="PubDate",
    )


def _announcement_field() -> SemanticField:
    return SemanticField(
        logical_name="announcement",
        dataset="events",
        physical_name="AnnDate",
        market="ashare",
        availability="knowledge_time",
        temporal_model="event",
        grain="snapshot",
        source_unit=None,
        canonical_unit=None,
        scale=None,
        pit_fidelity="effective_only",
        knowledge_time="AnnDate",
    )


def test_mixed_availability_all_fields_retained() -> None:
    """三字段（same_day / next_trading_day / knowledge_time）全量保留。"""
    identity = DataReadIdentity(
        dataset="daily",
        revision="r1",
        fields=(_price_field(), _roe_field(), _announcement_field()),
    )
    # 3 个 ResolvedFieldIdentity 全在（按逻辑名排序）。
    logicals = tuple(f.logical_name for f in identity.fields)
    assert logicals == ("announcement", "price", "roe")
    avails = {f.logical_name: f.availability for f in identity.fields}
    assert avails["price"] == "same_day"
    assert avails["roe"] == "next_trading_day"
    assert avails["announcement"] == "knowledge_time"


def test_mixed_availability_hash_differs_from_collapsed() -> None:
    """混合 availability 的身份 ≠ 塌缩成单一 availability 的身份。"""
    mixed = DataReadIdentity(
        dataset="daily",
        revision="r1",
        fields=(_price_field(), _roe_field(), _announcement_field()),
    )
    collapsed = DataReadIdentity(
        dataset="daily",
        revision="r1",
        availability="same_day",
        fields=(_price_field(),),
    )
    assert mixed.digest != collapsed.digest
    # 只保留 price 与 只保留 roe 也各不相同。
    only_price = DataReadIdentity(
        dataset="daily", revision="r1", fields=(_price_field(),)
    )
    only_roe = DataReadIdentity(
        dataset="daily", revision="r1", fields=(_roe_field(),)
    )
    assert only_price.digest != only_roe.digest
    assert mixed.digest != only_price.digest


def test_mixed_availability_field_order_independent() -> None:
    """字段顺序无关：不同声明顺序 → 相同身份 hash。"""
    a = DataReadIdentity(
        dataset="daily",
        revision="r1",
        fields=(_price_field(), _roe_field(), _announcement_field()),
    )
    b = DataReadIdentity(
        dataset="daily",
        revision="r1",
        fields=(_announcement_field(), _price_field(), _roe_field()),
    )
    assert a.digest == b.digest
    assert a.fields == b.fields


# ---------------------------------------------------------------------------
# DA-P0-02：digest 始终内部推导。
# ---------------------------------------------------------------------------

def test_caller_supplied_digest_is_ignored() -> None:
    """caller 传入任何 digest 都被忽略/覆盖为内部推导值。"""
    correct = DataReadIdentity(dataset="daily", revision="r1").digest
    # 显式传 digest kwarg：init=False 拒绝 caller digest（DA-P0-02 预期形态）。
    with pytest.raises(TypeError):
        DataReadIdentity(dataset="daily", revision="r1", digest="deadbeef" * 8)
    # 通过 from_dict 传 bogus digest → 重算校验失败。
    d = DataReadIdentity(dataset="daily", revision="r1").to_dict()
    bad = dict(d)
    bad["digest"] = "deadbeef" * 8
    with pytest.raises(Exception, match="digest"):
        DataReadIdentity.from_dict(bad)
    assert DataReadIdentity(dataset="daily", revision="r1").digest == correct


def test_deserialization_recomputes_and_validates_digest() -> None:
    """from_dict：重算 digest 校验；供给不一致 → raise。"""
    obj = DataReadIdentity(
        dataset="daily",
        revision="r1",
        fields=(_price_field(), _roe_field()),
    )
    d = obj.to_dict()
    # 相同 dict → 重算一致。
    assert DataReadIdentity.from_dict(d).digest == obj.digest
    # 篡改 digest → fail closed。
    bad = dict(d)
    bad["digest"] = "0" * 64
    with pytest.raises(Exception, match="digest"):
        DataReadIdentity.from_dict(bad)


def test_content_identity_deserialization_validates_digest() -> None:
    """DataReadContentIdentity.from_dict：供给 digest 不一致 → raise。"""
    c = DataReadContentIdentity(
        dataset="daily",
        revision="r1",
        fields=(_price_field(), _roe_field()),
    )
    d = c.to_dict()
    assert DataReadContentIdentity.from_dict(d).digest == c.digest
    bad = dict(d)
    bad["digest"] = "f" * 64
    with pytest.raises(Exception, match="digest"):
        DataReadContentIdentity.from_dict(bad)


# ---------------------------------------------------------------------------
# DA-P0-03：production fail-closed / research UNKNOWN。
# ---------------------------------------------------------------------------

def _broken_store() -> object:
    """一个什么 provenance 都解析不了的 store（fail-open 旧行为返回 None）。"""

    class _Broken:
        def manifest_version(self, dataset, **params):
            raise RuntimeError("no manifest backend")

        def calendar_snapshot_id(self):
            raise RuntimeError("no calendar")

        def _resolve_universe_instruments(self, universe, time_range=None, instruments=None):
            raise RuntimeError("no universe")

    return _Broken()


def _no_source_snapshot_store() -> object:
    """revision/calendar 可解析，但物理 source snapshot 缺失的 store。"""

    class _Store:
        def manifest_version(self, dataset, **params):
            return {"has_manifest": True, "manifest_generation_id": "g7"}

        def calendar_snapshot_id(self):
            return "cal-1"

        def _resolve_universe_instruments(self, universe, time_range=None, instruments=None):
            return ("600000.SH",)

    return _Store()


def _universe_broken_store() -> object:
    """revision/calendar 可解析，但 universe 解析失败的 store。"""

    class _Store:
        def manifest_version(self, dataset, **params):
            return {"has_manifest": True, "manifest_generation_id": "g7"}

        def calendar_snapshot_id(self):
            return "cal-1"

        def _resolve_universe_instruments(self, universe, time_range=None, instruments=None):
            raise RuntimeError("no universe backend")

    return _Store()


def test_production_missing_provenance_fails_closed() -> None:
    """production/PIT-strict：revision / calendar / source snapshot 任一不可得 → raise。"""
    token = set_runtime_mode_identity("production")
    try:
        with pytest.raises(Exception, match="revision"):
            build_data_read_identity(
                _broken_store(),
                dataset="daily",
                columns=["a"],
                strict=True,
            )
    finally:
        reset_runtime_mode_identity(token)


def test_production_missing_universe_fails_closed() -> None:
    """production：required universe 不可得 → raise。"""
    token = set_runtime_mode_identity("production")
    try:
        with pytest.raises(Exception, match="universe"):
            build_data_read_identity(
                _universe_broken_store(),
                dataset="daily",
                columns=["a"],
                universe="hs300",
                strict=True,
            )
    finally:
        reset_runtime_mode_identity(token)


def test_production_missing_source_snapshot_fails_closed() -> None:
    """production：物理 source snapshot 缺失 → raise。"""
    token = set_runtime_mode_identity("production")
    try:
        with pytest.raises(Exception, match="source snapshot"):
            build_data_read_identity(
                _no_source_snapshot_store(),
                dataset="daily",
                columns=["a"],
                strict=True,
            )
    finally:
        reset_runtime_mode_identity(token)


def test_research_missing_provenance_is_explicit_unknown() -> None:
    """research：不可得项 → 不 raise，provenance_status 显式 UNKNOWN。"""
    token = set_runtime_mode_identity("interactive_research")
    try:
        identity = build_data_read_identity(
            _broken_store(),
            dataset="daily",
            columns=["a"],
            strict=False,
        )
    finally:
        reset_runtime_mode_identity(token)
    assert identity.provenance_status == "unknown"
    assert identity.revision is None
    assert identity.calendar_identity is None
    assert identity.source_snapshot is None
    assert identity.provenance_notes  # 每条缺失都有显式 note


def test_research_without_universe_ok() -> None:
    """research：universe 未指定不要求，provenance 正常可用。"""
    token = set_runtime_mode_identity("interactive_research")
    try:
        identity = build_data_read_identity(
            _broken_store(),
            dataset="daily",
            columns=["a"],
            strict=False,
        )
    finally:
        reset_runtime_mode_identity(token)
    assert identity.provenance_status == "unknown"


# ---------------------------------------------------------------------------
# DA-P0-04：内容身份不含执行身份。
# ---------------------------------------------------------------------------

def test_content_hash_identical_across_request_ids() -> None:
    """相同内容、不同 request_id/session → 相同内容 hash；执行身份独立记录。"""
    content_args = dict(
        dataset="daily",
        revision="r1",
        columns=("a", "b"),
        time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
        instrument_filter=("600000.SH", "000001.SZ"),
        universe_snapshot="uni-1",
        calendar_identity="cal-1",
        source_snapshot="snap-1",
    )
    rid_a = DataReadIdentity(**content_args, session="req-AAA", decision_clock="2024-01-31")
    rid_b = DataReadIdentity(**content_args, session="req-BBB", decision_clock="2024-02-01")
    # 内容身份相同（request_id 不进内容 hash）。
    assert rid_a.digest == rid_b.digest
    assert rid_a.content == rid_b.content
    # 执行身份独立记录且可区分。
    assert rid_a.execution is not None and rid_b.execution is not None
    assert rid_a.execution.request_id == "req-AAA"
    assert rid_b.execution.request_id == "req-BBB"


def test_execution_identity_fields_recorded_separately() -> None:
    """ReadExecutionIdentity 记录 request/session/executor/timestamp/trace_id。"""
    ex = ReadExecutionIdentity(
        request_id="req-1",
        session="sess-1",
        executor="algo",
        timestamp="2024-01-31T00:00:00",
        trace_id="trace-1",
    )
    d = ex.to_dict()
    assert d["request_id"] == "req-1"
    assert d["session"] == "sess-1"
    assert d["executor"] == "algo"
    assert d["timestamp"] == "2024-01-31T00:00:00"
    assert d["trace_id"] == "trace-1"


def test_session_does_not_change_content_digest() -> None:
    """session 字段不再改变内容 digest（DA-P0-04）。"""
    base = DataReadIdentity(dataset="daily", revision="r1")
    with_session = DataReadIdentity(dataset="daily", revision="r1", session="sess-1")
    assert base.digest == with_session.digest

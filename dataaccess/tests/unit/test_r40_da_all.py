# -*- coding: utf-8 -*-
"""R40 DA 批次 #52-#60 测试（dataaccess/tests/unit，无并行）。"""
from __future__ import annotations

import datetime
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# #52：SnapshotFidelity 枚举 + production 拒绝低保真度
# ---------------------------------------------------------------------------


def test_snapshot_fidelity_production_rejects_low_fidelity():
    from data_access.core.exceptions import SnapshotBuildError
    from data_access.snapshot.fidelity import (
        SnapshotFidelity,
        assert_production_snapshot_fidelity,
        production_snapshot_fidelity_ok,
    )
    from data_access.snapshot.source_snapshot import ResolvedSourceSnapshot

    # 保真度层级
    assert production_snapshot_fidelity_ok(SnapshotFidelity.PUBLISHER_MANIFEST)
    assert production_snapshot_fidelity_ok(SnapshotFidelity.REMOTE_VERSION_ID)
    assert not production_snapshot_fidelity_ok(SnapshotFidelity.CONTENT_HASH)
    assert not production_snapshot_fidelity_ok(SnapshotFidelity.LOCAL_STAT)
    assert not production_snapshot_fidelity_ok(SnapshotFidelity.FALLBACK)
    assert not production_snapshot_fidelity_ok(SnapshotFidelity.UNKNOWN)

    snap = ResolvedSourceSnapshot(dataset="d", fidelity=SnapshotFidelity.LOCAL_STAT)
    with pytest.raises(SnapshotBuildError):
        assert_production_snapshot_fidelity(snap.fidelity, dataset="d", production=True)
    # research 不拒绝
    assert_production_snapshot_fidelity(SnapshotFidelity.FALLBACK, production=False)
    # production 下高保真 OK
    assert_production_snapshot_fidelity(SnapshotFidelity.REMOTE_VERSION_ID, production=True)


def test_snapshot_fidelity_default_unknown():
    from data_access.snapshot.fidelity import SnapshotFidelity
    from data_access.snapshot.source_snapshot import ResolvedSourceSnapshot

    snap = ResolvedSourceSnapshot(dataset="d")
    assert snap.fidelity is SnapshotFidelity.UNKNOWN


# ---------------------------------------------------------------------------
# #53：COST_UNKNOWN_CONSERVATIVE sentinel + production 拒绝
# ---------------------------------------------------------------------------


def test_scan_cost_unknown_sentinel_blocks_production():
    from data_access.core.exceptions import ResourceAdmissionError
    from data_access.read.scan_cost import (
        COST_UNKNOWN_CONSERVATIVE,
        ScanCost,
        assert_scan_cost_usable,
        conservative_scan_cost,
    )

    # 真未知（无法保守估计）→ production reject
    unknown = ScanCost(
        dataset="d", file_count=0, total_bytes=None, estimated_rows=0,
        projected_columns=0, total_columns=None, remote=False, cost_basis="unknown",
    )
    with pytest.raises(ResourceAdmissionError):
        assert_scan_cost_usable(unknown, production=True)

    # None → reject
    with pytest.raises(ResourceAdmissionError):
        assert_scan_cost_usable(None, production=True)

    # 保守估计 → 可继续
    cons = conservative_scan_cost("d")
    assert cons.cost_basis == COST_UNKNOWN_CONSERVATIVE
    assert assert_scan_cost_usable(cons, production=True) is True

    # manifest basis → OK
    ok = ScanCost(
        dataset="d", file_count=1, total_bytes=10, estimated_rows=5,
        projected_columns=1, total_columns=1, remote=False, cost_basis="manifest",
    )
    assert assert_scan_cost_usable(ok, production=True) is True

    # research 不拒绝 unknown
    assert assert_scan_cost_usable(unknown, production=False) is False


# ---------------------------------------------------------------------------
# #54：snapshot pin 只核对实际 time_range/instrument scope 覆盖的 partition
# ---------------------------------------------------------------------------


def test_snapshot_pin_only_pins_relevant_partitions(monkeypatch):
    from data_access.read.data_request import _scope_file_manifests
    from data_access.read.manifest import DatasetManifest, ManifestFile
    from data_access.read.read_contract import FileVersion

    manifest = DatasetManifest(
        dataset="d", time_column="ts", instrument_column="inst",
        files=(
            ManifestFile(path="/d/a.parquet", min_time="2024-01-01", max_time="2024-01-10"),
            ManifestFile(path="/d/b.parquet", min_time="2024-02-01", max_time="2024-02-10"),
            ManifestFile(path="/d/c.parquet", min_time="2024-03-01", max_time="2024-03-10"),
        ),
    )
    import data_access.read.manifest as m

    monkeypatch.setattr(
        m, "load_manifest_for_dataset", lambda store, ds, **kw: manifest
    )

    pinned = [
        FileVersion(path="/d/a.parquet", size=1, mtime_ns=1),
        FileVersion(path="/d/b.parquet", size=1, mtime_ns=1),
        FileVersion(path="/d/c.parquet", size=1, mtime_ns=1),
    ]
    current = list(pinned)
    pinned_s, current_s = _scope_file_manifests(
        None, "d", pinned, current,
        time_range=("2024-01-05", "2024-01-07"), instruments=None, params={},
    )
    assert [f.path for f in pinned_s] == ["/d/a.parquet"]
    assert [f.path for f in current_s] == ["/d/a.parquet"]

    # 无 time/instrument scope → 不裁剪（保守全量）
    pinned_full, current_full = _scope_file_manifests(
        None, "d", pinned, current, time_range=None, instruments=None, params={},
    )
    assert len(pinned_full) == 3 and len(current_full) == 3

    # instrument scope 裁剪
    manifest2 = DatasetManifest(
        dataset="d", time_column="ts", instrument_column="inst",
        files=(
            ManifestFile(path="/d/a.parquet", min_instrument="AAA", max_instrument="AAB"),
            ManifestFile(path="/d/b.parquet", min_instrument="BBB", max_instrument="BBC"),
        ),
    )
    monkeypatch.setattr(
        m, "load_manifest_for_dataset", lambda store, ds, **kw: manifest2
    )
    pinned_s2, _ = _scope_file_manifests(
        None, "d", pinned, current,
        time_range=None, instruments=["AAB"], params={},
    )
    assert [f.path for f in pinned_s2] == ["/d/a.parquet"]


# ---------------------------------------------------------------------------
# #55：ManifestFetchResult typed 降级层级
# ---------------------------------------------------------------------------


def test_manifest_degradation_typed_states():
    from data_access.snapshot.manifest import (
        ManifestFetchResult,
        classify_manifest_fetch,
        fetch_manifest_typed,
    )

    result, val = fetch_manifest_typed(lambda: {"token": 1})
    assert result is ManifestFetchResult.OK and val == {"token": 1}

    result, _ = fetch_manifest_typed(lambda: None)
    assert result is ManifestFetchResult.ABSENT

    result, _ = fetch_manifest_typed(
        lambda: (_ for _ in ()).throw(PermissionError("denied"))
    )
    assert result is ManifestFetchResult.PERMISSION_DENIED

    result, _ = fetch_manifest_typed(
        lambda: (_ for _ in ()).throw(TimeoutError("timed out"))
    )
    assert result is ManifestFetchResult.DEADLINE_EXCEEDED

    result, _ = fetch_manifest_typed(
        lambda: (_ for _ in ()).throw(OSError("broken parquet"))
    )
    assert result is ManifestFetchResult.LOOKUP_FAILED

    assert classify_manifest_fetch(None) is ManifestFetchResult.ABSENT
    assert result.degraded is True
    assert ManifestFetchResult.OK.ok is True


# ---------------------------------------------------------------------------
# #56：resolution cache key 含 PhysicalResolutionContext 维度
# ---------------------------------------------------------------------------


class _FakeStore:
    class _registry:
        @staticmethod
        def get(ds):
            return type(
                "DS", (),
                {"namespace": "ns1", "storage": "remote", "format": "parquet",
                 "base_path": "/x"},
            )()

    def _contract_digest_for(self, ds):
        return "contract-X"

    def manifest_version(self, ds):
        return {"manifest_generation_id": "gen-9", "source_epoch": "7"}


def test_resolution_cache_identity_includes_contract_digest():
    from data_access.read.read_session import (
        PhysicalResolutionContext,
        _ResolutionCache,
    )

    cache_a = _ResolutionCache(
        context_provider=lambda ds: PhysicalResolutionContext(contract_digest="digA")
    )
    cache_a[("d", "params", "tr", "inst")] = "valueA"

    cache_b = _ResolutionCache(
        context_provider=lambda ds: PhysicalResolutionContext(contract_digest="digB")
    )
    cache_b[("d", "params", "tr", "inst")] = "valueB"

    # 同一 context → 命中
    assert cache_a.get(("d", "params", "tr", "inst")) == "valueA"
    # 不同 contract_digest → 不串
    assert cache_b.get(("d", "params", "tr", "inst")) == "valueB"

    # from_store 派生各维度
    ctx = PhysicalResolutionContext.from_store(_FakeStore(), "d")
    assert ctx.contract_digest == "contract-X"
    assert ctx.mutation_generation == "gen-9"
    assert ctx.namespace == "ns1"

    # 基键相同 + context 不同 → 扩键后不冲突
    assert cache_a._enrich(("d", "p", "r", "i")) != cache_b._enrich(("d", "p", "r", "i"))


def test_data_read_session_uses_enriched_cache():
    from data_access.read.read_session import (
        PhysicalResolutionContext,
        _ResolutionCache,
        DataReadSession,
    )

    store = MagicMock()
    store._registry.get.return_value = type(
        "DS", (), {"namespace": "ns", "storage": "local", "format": "parquet",
                   "base_path": "/x"})()

    session = DataReadSession(store)
    assert isinstance(session._resolution_cache, _ResolutionCache)
    # 相同基键 + 相同 context → 命中同一条目
    session._resolution_cache[("d", "p", "r", "i")] = "v"
    assert session._resolution_cache.get(("d", "p", "r", "i")) == "v"


# ---------------------------------------------------------------------------
# #57：两相租约——discovery 相 → execution 相
# ---------------------------------------------------------------------------


def test_two_phase_resolution_then_execution_lease():
    from data_access.r30.execution_lease import ExecutionLease
    from data_access.r30.resolution_lease import ResolutionLease, ResolutionPhase

    lease = ResolutionLease(lease_id="test", max_resolution_slots=2)
    # 发现相
    assert lease.acquire_resolution()
    assert lease.acquire_resolution(slots=2) is False  # 并发超限
    lease.release_resolution()
    # 发现成功后进入执行相
    exec_lease = lease.transition_to_execution()
    assert isinstance(exec_lease, ExecutionLease)
    assert lease.phase is ResolutionPhase.EXECUTION_ACTIVE
    # 执行相后不能再 discovery
    with pytest.raises(RuntimeError):
        lease.acquire_resolution()
    lease.release()
    assert lease.phase is ResolutionPhase.RELEASED


def test_resolution_lease_requires_discovery_before_execution():
    from data_access.r30.resolution_lease import ResolutionLease

    lease = ResolutionLease(lease_id="x")
    with pytest.raises(RuntimeError):
        lease.transition_to_execution()


def test_two_phase_convenience_runner():
    from data_access.r30.resolution_lease import (
        ResolutionLease,
        two_phase_resolution_then_execution,
    )

    out = two_phase_resolution_then_execution(
        resolution_work=lambda lease: lease.acquire_resolution(),
        execution_work=lambda exec_lease: "done",
    )
    assert out == "done"


# ---------------------------------------------------------------------------
# #58：remote discovery slot 独立治理
# ---------------------------------------------------------------------------


def test_remote_discovery_governed_by_lease():
    from data_access.runtime.resource_governor import GlobalResourceGovernor
    from data_access.r30.resolution_lease import ResolutionLease

    gov = GlobalResourceGovernor(max_remote_concurrency=8)
    max_disc = gov.to_dict()["max_remote_discovery"]
    assert max_disc == 4  # max(4, 8//2)

    acquired = 0
    while gov.acquire_remote_discovery_slot():
        acquired += 1
    assert acquired == max_disc
    # 超出 → False
    assert gov.acquire_remote_discovery_slot() is False
    # 释放后可再取
    gov.release_remote_discovery_slot()
    assert gov.acquire_remote_discovery_slot() is True

    # ResolutionLease 消费 governor discovery slot
    for _ in range(max_disc):
        gov.release_remote_discovery_slot()
    lease = ResolutionLease(lease_id="g", governor=gov, max_resolution_slots=2)
    assert lease.acquire_resolution()
    assert gov.remote_discovery_inflight() == 1
    lease.release()
    assert gov.remote_discovery_inflight() == 0


# ---------------------------------------------------------------------------
# #59：StartupCertificate —— production store 要求未过期证书
# ---------------------------------------------------------------------------


def test_startup_certificate_required_for_production_store():
    from data_access.runtime.startup_gate import (
        StartupCertificate,
        build_startup_certificate,
        require_startup_certificate,
        startup_certificate_expired,
    )

    store = MagicMock()

    def _clean(store=None):
        return []

    cert = require_startup_certificate(store, production=True, checks=[_clean])
    assert isinstance(cert, StartupCertificate)
    assert cert.passed is True
    assert cert.evidence_hash
    assert store._startup_certificate is cert

    # 再次 require → 复用未过期证书（不重跑 gate）
    calls = {"n": 0}

    def _counting(store=None):
        calls["n"] += 1
        return []

    require_startup_certificate(store, production=True, checks=[_counting])
    assert calls["n"] == 0

    # production gate 失败 → raise（无证书可用）
    store2 = MagicMock()

    def _blocking(store=None):
        return ["critical failure"]

    with pytest.raises(RuntimeError):
        require_startup_certificate(store2, production=True, checks=[_blocking])

    # 过期检测
    assert startup_certificate_expired(None) is True
    assert startup_certificate_expired(build_startup_certificate(True, [])) is False
    # StartupCertificate 是 frozen dataclass（R32-P0-013），不能原地改 passed；
    # 用 dataclasses.replace 构造一个未通过证书验证过期检测。
    import dataclasses

    bad = dataclasses.replace(build_startup_certificate(True, []), passed=False)
    assert startup_certificate_expired(bad) is True


def test_run_startup_gate_returns_certificate():
    from data_access.runtime.startup_gate import (
        StartupCertificate,
        run_startup_gate,
    )

    store = MagicMock()

    def _clean(store=None):
        return []

    result = run_startup_gate(store, production=True, checks=[_clean])
    assert isinstance(result, StartupCertificate)
    assert result.passed is True
    assert list(result.problems) == []


# ---------------------------------------------------------------------------
# #60：calendar digest + 覆盖窗口校验（不再用 date.today() 算上一个周六）
# ---------------------------------------------------------------------------


def test_startup_calendar_checks_digest_not_date_today():
    import pandas as pd

    from data_access.read.session_calendar import MarketCalendar
    from data_access.runtime.startup_gate import (
        calendar_coverage_check,
        calendar_digest,
    )

    today = date.today()
    bdays = pd.bdate_range(today - timedelta(days=120), today)
    days = [d.date() for d in bdays]

    cal = MarketCalendar("ashare", trading_days=days, source="explicit")
    assert calendar_coverage_check(cal, market="ashare", expected_digest=None) == []
    # digest 匹配 → 无问题
    dg = calendar_digest(cal)
    assert calendar_coverage_check(cal, market="ashare", expected_digest=dg) == []
    # fallback 来源 → 非权威
    cal_fb = MarketCalendar("ashare", trading_days=days, source="fallback")
    assert any("fallback" in p for p in calendar_coverage_check(cal_fb, market="ashare"))
    # digest 不匹配
    cal_other = MarketCalendar("ashare", trading_days=days[:-5], source="explicit")
    assert any("digest" in p for p in calendar_coverage_check(
        cal_other, market="ashare", expected_digest=dg))
    # 覆盖不足（被截断到远古）
    cal_trunc = MarketCalendar(
        "ashare",
        trading_days=[today - timedelta(days=300) + timedelta(days=i) for i in range(5)],
        source="explicit",
    )
    assert any(
        "覆盖不足" in p
        for p in calendar_coverage_check(
            cal_trunc, market="ashare", required_window_days=30, lookback_days=120)
    )

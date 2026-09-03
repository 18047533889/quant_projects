# -*- coding: utf-8 -*-
"""R58 #6 —— governor 只接受明确整数 scan-bytes；未知成本 fail-closed。

AlphaFlow review #3/#6：remote 对象已可 LIST/HEAD 但拿不到 size 时，绝不能
静默按未知成本参与准入（旧语义 ``_coerce_bound(None)`` → 保守上界 2^62-1 →
必然超限拒绝，报错面目全非：真实原因是元数据缺失，表面却是 "scan bytes 超限"）。

新语义：
    - ``estimated_scan_bytes=None``（未知）→ strict/production 抛
      ``RemoteMetadataUnavailable``（fail-closed，绝不静默当 0 或按上界拒绝）；
    - research 可显式豁免：``DATA_ACCESS_ALLOW_UNKNOWN_REMOTE_SCAN_COST=1`` 或
      ``DATA_ACCESS_UNKNOWN_REMOTE_SCAN_BYTES=<int>``（后者提供明确整数上界）；
    - 内存维度（``estimated_memory``）仍走保守上界（R32-P0-040 未知≠0）。
"""
from __future__ import annotations

import pytest

from data_access.core.exceptions import RemoteMetadataUnavailable
from data_access.runtime.resource_governor import (
    GlobalResourceGovernor,
    ResourceReservation,
    _coerce_scan_bytes,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_ALLOW_UNKNOWN_REMOTE_SCAN_COST", raising=False)
    monkeypatch.delenv("DATA_ACCESS_UNKNOWN_REMOTE_SCAN_BYTES", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("DATA_ACCESS_RUN_MODE", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    yield


class TestCoerceScanBytes:
    def test_explicit_int_passthrough(self):
        assert _coerce_scan_bytes(0) == 0
        assert _coerce_scan_bytes(100) == 100
        assert _coerce_scan_bytes(-5) == 0  # 负成本 clamp 到 0

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            _coerce_scan_bytes(True)

    def test_non_int_rejected(self):
        with pytest.raises(TypeError):
            _coerce_scan_bytes(1.5)
        with pytest.raises(TypeError):
            _coerce_scan_bytes("100")

    def test_none_strict_raises_typed(self, monkeypatch):
        """strict/production 下 None → RemoteMetadataUnavailable（不是 ResourceAdmissionError）。"""
        monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
        with pytest.raises(RemoteMetadataUnavailable) as ei:
            _coerce_scan_bytes(None)
        assert ei.value.retryable is True

    def test_none_production_raises_typed(self, monkeypatch):
        monkeypatch.setenv("DATA_ACCESS_RUN_MODE", "production")
        with pytest.raises(RemoteMetadataUnavailable):
            _coerce_scan_bytes(None)

    def test_none_research_opt_in_allow(self, monkeypatch):
        """research + DATA_ACCESS_ALLOW_UNKNOWN_REMOTE_SCAN_COST=1 → 不抛，按保守上界。"""
        monkeypatch.setenv("DATA_ACCESS_ALLOW_UNKNOWN_REMOTE_SCAN_COST", "1")
        assert _coerce_scan_bytes(None) == (1 << 62) - 1

    def test_none_research_opt_in_explicit_bytes(self, monkeypatch):
        """research + DATA_ACCESS_UNKNOWN_REMOTE_SCAN_BYTES=<int> → 用显式整数。"""
        monkeypatch.setenv("DATA_ACCESS_UNKNOWN_REMOTE_SCAN_BYTES", "5000")
        assert _coerce_scan_bytes(None) == 5000

    def test_none_research_default_conservative(self):
        """research 无豁免 → 仍按保守上界（未知≠0，不静默当 0）。"""
        assert _coerce_scan_bytes(None) == (1 << 62) - 1


class TestReservationScanBytes:
    def test_reservation_none_strict_raises(self, monkeypatch):
        monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
        with pytest.raises(RemoteMetadataUnavailable):
            ResourceReservation(
                query_id="q1", principal_id="p1", estimated_scan_bytes=None
            )

    def test_reservation_none_research_opt_in(self, monkeypatch):
        monkeypatch.setenv("DATA_ACCESS_UNKNOWN_REMOTE_SCAN_BYTES", "1234")
        r = ResourceReservation(
            query_id="q1", principal_id="p1", estimated_scan_bytes=None
        )
        assert r.estimated_scan_bytes == 1234

    def test_reservation_memory_none_still_conservative(self, monkeypatch):
        """内存维度未知仍走保守上界（R32-P0-040），不受 scan-bytes 语义影响。"""
        monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
        r = ResourceReservation(
            query_id="q1", principal_id="p1",
            estimated_scan_bytes=100, estimated_memory=None,
        )
        assert r.estimated_scan_bytes == 100
        assert r.estimated_memory == (1 << 62) - 1


class TestGovernorAdmit:
    def test_admit_unknown_scan_strict_rejected(self, monkeypatch):
        """strict 下 governor.admit 对未知 scan-bytes 抛 RemoteMetadataUnavailable。"""
        monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
        gov = GlobalResourceGovernor()
        with pytest.raises(RemoteMetadataUnavailable):
            gov.admit(
                ResourceReservation(
                    query_id="q1", principal_id="p1", estimated_scan_bytes=None
                )
            )

    def test_admit_unknown_scan_research_opt_in(self, monkeypatch):
        """research + 显式豁免 → 准入成功（不抛 ResourceAdmissionError）。"""
        monkeypatch.setenv("DATA_ACCESS_UNKNOWN_REMOTE_SCAN_BYTES", "999")
        gov = GlobalResourceGovernor()
        res = gov.admit(
            ResourceReservation(
                query_id="q1", principal_id="p1", estimated_scan_bytes=None
            )
        )
        assert res.estimated_scan_bytes == 999
        gov.release("q1")

# -*- coding: utf-8 -*-
"""R25 T-SNAP-001..005 —— source snapshot 测试（P0-009/010/012/013）。

    T-SNAP-001  remote wildcard unresolved → reject（production）
    T-SNAP-002  exact object set + ETags → stable snapshot
    T-SNAP-003  same key overwrite → ETag 变化 → snapshot changes
    T-SNAP-004  local stale + remote new → auto 不能混
    T-SNAP-005  exact same generation local+remote → hybrid allowed（research）
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from data_access.core.exceptions import SourceSnapshotChanged, SourceSnapshotUnavailable
from data_access.read.query_budget import QueryBudget, enforce_scan_object_budget
from data_access.snapshot.source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)
from data_access.snapshot.resolver import SourceSnapshotResolver, parse_source_manifest
from data_access.snapshot.verifier import SnapshotVerifier, verify_snapshot_before_execute


def _obj(uri, etag=None, size=100, vid=None):
    return ResolvedObject(uri=uri, etag=etag, content_length=size, version_id=vid)


def test_tsnap001_wildcard_unresolved_rejected():
    """P0-009：production 下 unresolved wildcard → SourceSnapshotUnavailable。"""
    snap = ResolvedSourceSnapshot(
        dataset="t", objects=(_obj("s3://b/t/*.parquet"),)
    )
    assert snap.has_wildcard
    with pytest.raises(SourceSnapshotUnavailable):
        verify_snapshot_before_execute(snap)


def test_tsnap002_exact_objects_stable():
    """P0-010：exact object set + ETags → 稳定 snapshot。"""
    objs = tuple(_obj(f"s3://b/t/{d}.parquet", etag=f"e{d}") for d in range(3))
    snap = ResolvedSourceSnapshot(dataset="t", objects=objs)
    verify_snapshot_before_execute(snap)
    # budget 按真实对象数卡
    enforce_scan_object_budget(QueryBudget(max_scan_objects=5), object_count=3)


def test_tsnap003_overwrite_changes_snapshot():
    """P0-012：same key overwrite（ETag 变化）→ content_digest 变化。"""
    a = content_digest_of_objects((_obj("s3://b/t/f.parquet", etag="v1"),))
    b = content_digest_of_objects((_obj("s3://b/t/f.parquet", etag="v2"),))
    assert a != b


def test_tsnap003b_verifier_detects_etag_change():
    """SnapshotVerifier 执行前检测 ETag 变化 → SourceSnapshotChanged。"""
    snap = ResolvedSourceSnapshot(
        dataset="t",
        objects=(_obj("s3://b/t/f.parquet", etag="expected-v1"),),
    )
    v = SnapshotVerifier(
        remote_meta_fn=lambda uri: {"etag": "actual-v2", "content_length": 100},
        strict=True,
    )
    with pytest.raises(SourceSnapshotChanged):
        v.verify_before_execute(snap)


def _patch_remote_s3_auth():
    return patch("data_access.cos.remote.authorize_s3_path", return_value=None)


def test_tsnap004_auto_hybrid_strict_no_mix():
    """P0-013：strict 下 auto hybrid 禁止 local/remote 混 source epoch。"""
    from data_access.cos.mirror import MirrorSpec
    from data_access.cos.remote import hybrid_cos_read_paths
    from data_access.registry import load_registry

    registry = load_registry()
    ds = registry.get("ashare_stock_daily")
    spec = MirrorSpec(
        cos_prefix="cos://t",
        local_root=Path("/tmp/r25_hybrid"),
        table="StockDailyBar",
        layout="daily_parquet",
    )
    with patch(
        "data_access.cos.remote.mirror_spec_for_dataset",
        return_value=spec,
    ), patch("data_access.cos.remote.cos_read_mode", return_value="auto"), patch(
        "data_access.cos.remote.cos_remote_backend", return_value="httpfs"
    ), patch("data_access.cos.remote._strict_mode", return_value=True), patch(
        "data_access.read.query_budget.is_strict_semantics", return_value=True
    ), _patch_remote_s3_auth():
        # strict：不混，返回 None → 调用方走 all-or-nothing remote。
        out = hybrid_cos_read_paths(ds, time_range=("2024-01-02", "2024-01-03"))
        assert out is None


def test_tsnap005_research_hybrid_allowed():
    """P0-013：research 下（非 strict）auto hybrid 可混（需 lineage 标记 degraded）。"""
    from data_access.cos.mirror import MirrorSpec
    from data_access.cos.remote import hybrid_cos_read_paths
    from data_access.registry import load_registry

    registry = load_registry()
    ds = registry.get("ashare_stock_daily")
    spec = MirrorSpec(
        cos_prefix="cos://t",
        local_root=Path("/tmp/r25_hybrid_research"),
        table="StockDailyBar",
        layout="daily_parquet",
    )
    with patch(
        "data_access.cos.remote.mirror_spec_for_dataset",
        return_value=spec,
    ), patch("data_access.cos.remote.cos_read_mode", return_value="auto"), patch(
        "data_access.cos.remote.cos_remote_backend", return_value="httpfs"
    ), patch("data_access.cos.remote._strict_mode", return_value=False), patch(
        "data_access.cos.remote.local_mirror_complete_for_range", return_value=False
    ), _patch_remote_s3_auth():
        out = hybrid_cos_read_paths(ds, time_range=("2024-01-02", "2024-01-03"))
        # research：可混（返回 merged 列表，非 None）
        assert out is not None


def _digest(objs):
    from data_access.snapshot.source_snapshot import content_digest_of_objects

    return content_digest_of_objects(objs)


def test_source_manifest_parse():
    """R25 §13 / R26-P0-015 / R28-5：publisher source manifest 解析（generation + exact objects）。

    manifest_version / complete / object_count 为 R26 必填；R28-5 起 dataset /
    content_digest / prefix / published_at 也是 strict 必填，且 content_digest
    必须等于重算值。
    """
    objs = [
        {"key": "s3://b/t/2024-01-02.parquet", "etag": "e1", "size": 10},
        {"key": "s3://b/t/2024-01-03.parquet", "etag": "e2", "size": 20},
    ]
    from data_access.snapshot.source_snapshot import ResolvedObject

    digest = _digest(
        [ResolvedObject(uri=o["key"], etag=o["etag"], content_length=o["size"]) for o in objs]
    )
    m = parse_source_manifest(
        {
            "manifest_version": "1",
            "source_generation": "20260810T153000Z-abc",
            "complete": True,
            "dataset": "t",
            "prefix": "s3://b/t",
            "content_digest": digest,
            "published_at": "2026-08-10T00:00:00Z",
            "object_count": 2,
            "objects": objs,
        }
    )
    assert m.source_generation == "20260810T153000Z-abc"
    assert len(m.objects) == 2
    assert m.objects[0].etag == "e1"


def test_source_manifest_parse_digest_mismatch():
    """R28-5：strict 下 content_digest 必须等于重算值（对象被改 → 拒）。"""
    from data_access.snapshot.source_snapshot import ResolvedObject

    objs = [
        {"key": "s3://b/t/2024-01-02.parquet", "etag": "e1", "size": 10},
        {"key": "s3://b/t/2024-01-03.parquet", "etag": "e2", "size": 20},
    ]
    digest = _digest(
        [ResolvedObject(uri=o["key"], etag=o["etag"], content_length=o["size"]) for o in objs]
    )
    with pytest.raises(Exception, match="content_digest"):
        parse_source_manifest(
            {
                "manifest_version": "1",
                "source_generation": "G1",
                "complete": True,
                "dataset": "t",
                "prefix": "s3://b/t",
                "content_digest": "tampered",
                "published_at": "2026-08-10T00:00:00Z",
                "object_count": 2,
                "objects": objs,
            }
        )


def test_resolver_via_manifest():
    """SourceSnapshotResolver 优先 source manifest。"""
    from data_access.snapshot.source_snapshot import ResolvedObject

    obj = {"key": "s3://b/t/2024-01-02.parquet", "etag": "e1", "size": 10}
    digest = _digest([ResolvedObject(uri=obj["key"], etag=obj["etag"], content_length=obj["size"])])
    manifest = {
        "manifest_version": "1",
        "source_generation": "G1",
        "complete": True,
        "dataset": "t",
        "prefix": "s3://b/t",
        "content_digest": digest,
        "published_at": "2026-08-10T00:00:00Z",
        "object_count": 1,
        "objects": [obj],
    }
    r = SourceSnapshotResolver(
        source_manifest_fn=lambda ds: manifest, strict=True
    )
    snap = r.resolve("t", paths=["s3://b/t/*.parquet"])
    assert snap.source_generation == "G1"
    assert snap.object_count == 1
    assert not snap.has_wildcard

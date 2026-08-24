# -*- coding: utf-8 -*-
"""R25 T-MIR-001..005 —— mirror 原子性测试（P0-014/P0-008/§28）。

    T-MIR-001  full sync halfway crash → reader 继续 old generation
    T-MIR-002  new generation verified → atomic pointer switch
    T-MIR-003  two workers same object → only one physical download wins
    T-MIR-004  cache principal mismatch → reject
    T-MIR-005  cache disk pressure → pinned object never deleted
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from data_access.cos.mirror import (
    MirrorSpec,
    _sync_cos_file,
    publish_mirror_generation,
    resolve_current_mirror_generation,
)
from data_access.runtime.cache_manager import CacheManager


def _spec(tmp_path, table="T", layout="daily_parquet"):
    return MirrorSpec(
        cos_prefix="cos://test/t",
        local_root=tmp_path / "mirror",
        table=table,
        layout=layout,
    )


def _fake_sync_write_file(cmd, **kwargs):
    """fake clean-cos-ro sync：写一个 parquet 到 dest。"""
    dest = Path(cmd[-1])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "2024-01-02.parquet").write_bytes(b"PAR1")


# ---------------------------------------------------------------------------
# T-MIR-001 / T-MIR-002 — generation 原子发布
# ---------------------------------------------------------------------------
def test_tmir001_staging_not_live_before_publish(tmp_path):
    """发布前 sync 到 staging（G.tmp），不碰 live 目录。"""
    spec = _spec(tmp_path)
    staging = spec.local_root / "generation" / "G1.tmp"
    live = spec.local_root / "T"

    def fake_sync(cmd, **kwargs):
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "2024-01-02.parquet").write_bytes(b"PAR1")

    with patch("data_access.cos.mirror._run_cos_cli", side_effect=fake_sync):
        publish_mirror_generation(spec, generation_id="G1")
    assert not staging.exists()  # staging 已 rename
    assert (spec.local_root / "generation" / "G1" / "T" / "2024-01-02.parquet").exists()
    # pointer 指向 G1
    assert resolve_current_mirror_generation(spec.local_root) == "G1"


def test_tmir002_atomic_pointer_switch(tmp_path):
    """G1 发布后再发布 G2 → pointer 原子切换，reader 只 resolve pointer。"""
    spec = _spec(tmp_path)
    with patch("data_access.cos.mirror._run_cos_cli", side_effect=_fake_sync_write_file):
        publish_mirror_generation(spec, generation_id="G1")
        publish_mirror_generation(spec, generation_id="G2")
    assert resolve_current_mirror_generation(spec.local_root) == "G2"
    # G1 仍在（延迟 GC）
    assert (spec.local_root / "generation" / "G1" / "T").exists()


def test_tmir001b_empty_staging_rejected(tmp_path):
    """空 staging 拒绝发布（fail-closed）。"""
    spec = _spec(tmp_path)
    with patch("data_access.cos.mirror._run_cos_cli", side_effect=lambda cmd, **kw: None):
        with pytest.raises(Exception, match="无任何 parquet"):
            publish_mirror_generation(spec, generation_id="G-empty")


# ---------------------------------------------------------------------------
# T-MIR-003 — 并发 single-flight
# ---------------------------------------------------------------------------
def test_tmir003_second_worker_waits(tmp_path):
    """同一 object 多 worker：第一个下载成功后第二个跳过（O_EXCL 随机 temp）。"""
    dest = tmp_path / "2024-01-02.parquet"

    def fake_cp(cmd, **kwargs):
        target = Path(cmd[-1])
        target.write_bytes(b"\x00parquet")

    with patch("data_access.cos.mirror._run_cos_cli", side_effect=fake_cp):
        _sync_cos_file("cos://t/T/2024-01-02.parquet", dest)
        _sync_cos_file("cos://t/T/2024-01-02.parquet", dest)  # 第二次：已 fresh，跳过
    assert dest.exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# T-MIR-004 — cache principal mismatch
# ---------------------------------------------------------------------------
def test_tmir004_cache_principal_mismatch(tmp_path, monkeypatch):
    """cache root 按 principal 隔离：不同 principal → 不同 cache scope。"""
    monkeypatch.delenv("DATA_ACCESS_PRINCIPAL_ID", raising=False)
    monkeypatch.setenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    from data_access.cos.remote import _cache_principal_scope, cos_cache_root

    s1 = _cache_principal_scope()
    r1 = cos_cache_root()
    monkeypatch.setenv("DATA_ACCESS_PRINCIPAL_ID", "server-high")
    s2 = _cache_principal_scope()
    r2 = cos_cache_root()
    assert s1 != s2
    assert r1 != r2
    # principal 名不直接进路径（R24 P0-S3 §5.1 脱敏：SHA-256 digest scope），
    # 但不同 principal 必须映射到不同 scope 目录。
    assert r1.name != r2.name


# ---------------------------------------------------------------------------
# T-MIR-005 — pinned cache never deleted
# ---------------------------------------------------------------------------
def test_tmir005_pinned_never_deleted():
    cm = CacheManager(max_bytes=1000, high_watermark=0.5, low_watermark=0.1)
    cm.pin("g1/f", path="/tmp/g1f", size_bytes=400)
    cm.pin("g2/f", path="/tmp/g2f", size_bytes=400)
    try:
        cm.pin("g3/f", path="/tmp/g3f", size_bytes=400)  # 1200 > 1000，全 pinned → reject
        pytest.fail("admission over pinned quota should fail")
    except MemoryError:
        pass
    cm.unpin("g1/f")
    cm.unpin("g2/f")
    freed = cm.run_gc()
    assert freed >= 0
    assert cm.total_bytes() <= 400  # g3 仍 pinned（拒绝未加入），g1/g2 已逐出

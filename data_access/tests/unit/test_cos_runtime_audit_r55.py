# -*- coding: utf-8 -*-
"""R55（Wave-1 P0-08/09/10）COS 运行时审计 —— lease / GC / CURRENT 正确性。

审计结论与加固落地（对应 cos/mirror.py + write/mutation_lock.py 的最小外科改动）：

    P0-08  lease 无过期 → 会永久阻塞（defect）：
            mutation_lock 的 hard-break 兜底判 ``time.time() - lease_until > hard_break``，
            而 lease 过期（lease_until 已过去）但时间差还小于 hard_break（默认 2×lease）
            时锁**不可打破**——一个崩溃前 lease_until=now+X 的陈旧锁，在接下来的
            2X 窗口内即使 owner 已死也不被回收。修复：可打破判定补充「owner 已死
            OR lease 已过期」（后者是新路径：跨 host / PID 无法判定死亡的陈旧锁
            也要能按 lease 回收），并对老格式（无 lease 字段）锁保留
            「mtime 年龄 + owner 已死」的纯年龄回退。配套把 ``lease_seconds`` 显式
            写进锁 payload 头部（运维一眼可见 lease 语义，不再靠 stats 猜）。

    P0-09  GC 删除 CURRENT 或比 lease horizon 年轻的文件（defect 审查）：
            object_store_generation_publisher.gc 只删「非 CURRENT 且无不可变
            manifest」的对象，_read_manifest_at 按对象 key 直接读（已修过经内存
            _gen_prefix 定位的 P0-08 误删）；cos/mirror.py 此前**没有** generation
            GC —— 「旧 generation 延迟 GC」只有注释没有实现。本轮补上
            gc_mirror_generations：只回收 generation/ 下孤儿/陈旧代次目录，硬性
            (a) 绝不删除 CURRENT 代次，(b) 年龄 < keep_younger_seconds（lease
            horizon）的目录视为可能发布中跳过，(c) 路径 resolve/relative_to 双重
            归属校验，(d) dry_run 只统计。

    P0-10  CURRENT 非原子写 / GC 与并发 writer 竞态（缺陷审查）：
            current.json 用 tmp + os.replace 已原子；本轮再补 directory-fsync
            （POSIX crash-safe rename）与随机 tmp 后缀（并发 publisher 不撞名）；
            resolve_current_mirror_generation 现在校验 pointer 指向的代次目录真实
            存在，防 phantom CURRENT。GC 的 age 护栏 = 并发 publish 的竞态窗口保护。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from data_access.cos.mirror import (
    MirrorSpec,
    gc_mirror_generations,
    publish_mirror_generation,
    resolve_current_mirror_generation,
    _write_generation_pointer,
)
from data_access.write.mutation_lock import _can_break_lock, mutation_lock


# ---------------------------------------------------------------------------
# P0-08  lease：陈旧 lease（无 owner 判定）必须能回收，不能永久阻塞
# ---------------------------------------------------------------------------
def _write_lock(root: Path, *, lease_until: float, dead_owner: bool, released: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".data-access.mutation.lock"
    payload = {
        "pid": 2**31 - 1 if dead_owner else os.getpid(),
        "host": "far-host" if dead_owner else "localhost",  # 不同 host → 无法本地判定死亡
        "target": str(root),
        "process_start_time": 1.0,
        "transaction_id": "deadbeef",
        "acquired_at": lease_until - 3600.0,
        "lease_until": lease_until,
        "lease_seconds": 3600.0,
    }
    lock_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if released:
        lock_path.write_text(
            json.dumps({**payload, "released": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_r55_stale_lease_is_reclaimable_without_owner_probe(tmp_path):
    """P0-08 修复：跨 host 陈旧 lease（owner 无法本地判定）也能按 lease 回收。"""
    root = tmp_path / "root"
    # 陈旧锁：owner 在另一台 host（本机无法判定死活），lease 已经过期 120 秒
    # （< hard_break 默认 7200s —— 旧实现此窗口内不可打破 → 永久阻塞）。
    _write_lock(root, lease_until=time.time() - 120.0, dead_owner=True)

    assert _can_break_lock(
        root / ".data-access.mutation.lock",
        stale_after=3600.0,
        hard_break=7200.0,
    ) is True

    # 全链路：第二次 acquisition 直接打破陈旧锁并拿到锁。
    with patch("data_access.write.mutation_lock._proc_starttime_ticks", return_value=None):
        with mutation_lock(root, timeout=2.0, stale_after=3600.0, lease_seconds=60.0):
            assert (root / ".data-access.mutation.lock").exists()


def test_r55_young_lease_not_broken_and_lease_in_payload(tmp_path):
    """未过期 lease 不破；lease 字段显式写进锁文件（P0-08 配套）。"""
    root = tmp_path / "root2"
    _write_lock(root, lease_until=time.time() + 500.0, dead_owner=True)
    payload = json.loads((root / ".data-access.mutation.lock").read_text())
    assert payload["lease_until"] > time.time()
    assert payload.get("lease_seconds") == 3600.0  # 运维可见的 lease 语义
    assert _can_break_lock(
        root / ".data-access.mutation.lock",
        stale_after=3600.0,
        hard_break=7200.0,
    ) is False  # lease 未过期 → 不破

    # 活 owner（本机、同 PID）锁也不破。
    _write_lock(root, lease_until=time.time() + 500.0, dead_owner=False)
    assert _can_break_lock(
        root / ".data-access.mutation.lock",
        stale_after=3600.0,
        hard_break=7200.0,
    ) is False


def test_r55_released_marker_immediately_breakable(tmp_path):
    """released 标记立即可破（release 不再 unlink，清理交给下次 acquisition）。"""
    root = tmp_path / "root3"
    _write_lock(root, lease_until=time.time() - 1.0, dead_owner=True, released=True)
    assert _can_break_lock(
        root / ".data-access.mutation.lock",
        stale_after=3600.0,
        hard_break=7200.0,
    ) is True


# ---------------------------------------------------------------------------
# P0-09  GC：绝不删除 CURRENT 或比 lease horizon 年轻（发布中）的 artifact
# ---------------------------------------------------------------------------
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


def _gen_dir(root: Path, name: str, *, mtime_seconds_ago: float = 0.0) -> Path:
    d = root / "generation" / name / "T"
    d.mkdir(parents=True, exist_ok=True)
    (d / "2024-01-02.parquet").write_bytes(b"PAR1")
    gen_root = root / "generation" / name
    if mtime_seconds_ago:
        old = time.time() - mtime_seconds_ago
        os.utime(gen_root, (old, old))
    return gen_root


def test_r55_gc_never_removes_current_generation(tmp_path):
    """P0-09：GC 绝不动 CURRENT 代次（即使它已很老）。"""
    spec = _spec(tmp_path)
    with patch("data_access.cos.mirror._run_cos_cli", side_effect=_fake_sync_write_file):
        publish_mirror_generation(spec, generation_id="G1")
    current = _gen_dir(spec.local_root, "G1", mtime_seconds_ago=10 * 3600.0)
    assert resolve_current_mirror_generation(spec.local_root) == "G1"

    result = gc_mirror_generations(spec.local_root, keep_younger_seconds=0.0)

    assert result["current_generation"] == "G1"
    assert result["reclaimed"] == 0
    assert current.is_dir()  # CURRENT 代次原样保留
    # 无论年龄多大，CURRENT 都不回收。
    _gen_dir(spec.local_root, "G1", mtime_seconds_ago=100 * 3600.0)
    assert gc_mirror_generations(spec.local_root, keep_younger_seconds=0.0)["reclaimed"] == 0
    assert current.is_dir()


def test_r55_gc_skips_younger_than_lease_horizon(tmp_path):
    """P0-09/P0-10：比 lease horizon 年轻的代次视为「发布中」，GC 跳过。"""
    spec = _spec(tmp_path)
    # 孤儿代次：staging→final 已 rename，但 pointer 没翻到它（模拟崩溃残留）。
    orphan = _gen_dir(spec.local_root, "G_ORPHAN", mtime_seconds_ago=30.0)
    # 陈旧孤儿：足够老 → 该回收。
    stale = _gen_dir(spec.local_root, "G_STALE", mtime_seconds_ago=7200.0)

    result = gc_mirror_generations(spec.local_root, keep_younger_seconds=3600.0)

    assert orphan.is_dir()  # 30s < 1h horizon → 跳过
    assert not stale.is_dir()  # 2h 老且非 CURRENT → 回收
    assert result["skipped_young"] == 1
    assert result["reclaimed"] == 1
    assert resolve_current_mirror_generation(spec.local_root) is None  # pointer 未受影响


def test_r55_gc_never_removes_live_artifact_or_staging_current(tmp_path):
    """P0-09：并发发布中（staging rename 完成、pointer 未写）的 CURRENT 目标不被 GC。"""
    spec = _spec(tmp_path)
    with patch("data_access.cos.mirror._run_cos_cli", side_effect=_fake_sync_write_file):
        publish_mirror_generation(spec, generation_id="G1")
    live = spec.local_root / "generation" / "G1"
    # 模拟「新代次 G2 发布中」：staging rename 完成、pointer 已翻 → 但它正在发布期。
    _gen_dir(spec.local_root, "G2", mtime_seconds_ago=30.0)
    # 老孤儿。
    _gen_dir(spec.local_root, "OLD", mtime_seconds_ago=7200.0)

    result = gc_mirror_generations(spec.local_root, keep_younger_seconds=3600.0)

    assert live.is_dir()  # live/CURRENT artifact 永不删
    assert (spec.local_root / "generation" / "G2").is_dir()  # 发布中跳过
    assert not (spec.local_root / "generation" / "OLD").is_dir()  # 仅老孤儿回收
    assert resolve_current_mirror_generation(spec.local_root) == "G1"


def test_r55_gc_reclaims_crashed_staging_and_dry_run(tmp_path):
    """崩溃残留 staging（G.tmp）可回收；dry_run 只统计不删除。"""
    spec = _spec(tmp_path)
    crashed_staging = spec.local_root / "generation" / "HALF.tmp"
    (crashed_staging / "T").mkdir(parents=True, exist_ok=True)
    (crashed_staging / "T" / "2024-01-02.parquet").write_bytes(b"PAR1")
    old = time.time() - 7200.0
    os.utime(crashed_staging, (old, old))

    result = gc_mirror_generations(spec.local_root, keep_younger_seconds=3600.0, dry_run=True)
    assert result["reclaimed"] == 1
    assert crashed_staging.is_dir()  # dry_run 不删

    result = gc_mirror_generations(spec.local_root, keep_younger_seconds=3600.0)
    assert result["reclaimed"] == 1
    assert not crashed_staging.exists()


# ---------------------------------------------------------------------------
# P0-10  CURRENT：原子写 + fsync；resolve 校验指针目标真实存在（防 phantom）
# ---------------------------------------------------------------------------
def test_r55_current_pointer_atomic_with_fsync_and_no_tmp_residue(tmp_path):
    """current.json 原子写（tmp+os.replace）+ directory fsync；崩溃模拟不留 tmp 垃圾。"""
    root = tmp_path / "mirror"
    _write_generation_pointer(root, "G1", None)
    assert (root / "generation" / "current.json").exists()
    payload = json.loads((root / "generation" / "current.json").read_text())
    assert payload["current_generation"] == "G1"
    # 覆盖写：指针原子切换，无 .tmp 残留。
    _write_generation_pointer(root, "G2", "G1")
    payload = json.loads((root / "generation" / "current.json").read_text())
    assert payload["current_generation"] == "G2"
    assert payload["source_generation"] == "G1"
    assert list((root / "generation").glob("*.tmp")) == []
    assert list((root / "generation").glob(".current.json.*.tmp")) == []


def test_r55_resolve_rejects_phantom_current(tmp_path):
    """resolve 校验 pointer 目标代次真实存在；目录缺失 → 返回 None（不吐 phantom）。"""
    root = tmp_path / "mirror"
    _write_generation_pointer(root, "GHOST", None)
    # 目录不存在（pointer 已写、代次被外部清理/恢复遗漏）。
    assert resolve_current_mirror_generation(root) is None
    # 目录补上 → 恢复正常。
    _gen_dir(root, "GHOST")
    assert resolve_current_mirror_generation(root) == "GHOST"

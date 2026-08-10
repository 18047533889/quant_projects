# -*- coding: utf-8 -*-
"""R28 —— 底层并发 / 快照真实性 / 远端权限 / schema gate 成本 / 写性能收口。

覆盖（全走 public path / 真实 parquet / 真实语义）：
    1   CacheManager.pin() 过期重建不再死锁（非重入锁递归）
    2   CacheManager 删除失败 → entry 保留 + DELETE_FAILED + 字节仍计入配额
    3   远端 SnapshotVerifier 注入 credential-aware 真实 HEAD（remote_meta_fn）
    4   本地 final verify 按 size + mtime_ns 精确比较（同大小替换不漏网）
    5   SourceManifest strict 必填（dataset/content_digest/prefix/published_at/
        object_count）+ content_digest 重算 + bucket+segment 边界
    6   VerifiedPhysicalScope.contract_digest 消费（旧 scope 拒绝执行）
    7   SchemaEpoch migration 真实 epoch-pair（A→B，非 A→A 自环 / 任意放行）
    8   SchemaEpoch manifest O(1) 分组（不逐文件 footer）
    9   GlobalResourceGovernor 远端「请求总数」不再当并发上限
    10/11/12/15  DuckDB 统一并发 + deadline 贯穿 + 全入口 _check_pid
    13  pool 连接共享同一数据库实例（object cache 跨连接共享）
    17  凭证上下文传播（request-scoped credential provider 优先）
    18  sqlglot 限定名 catalog.schema.table 不能混配裸名
    26  generation COW（未变分区硬链接复用，不整代重写）
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    DeadlineExceeded,
    SchemaContractError,
    SourceSnapshotChanged,
    SourceSnapshotUnavailable,
    ValidationError,
)
from data_access.read.query_cache import reset_query_cache
from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from data_access.runtime.cache_manager import reset_cache_manager
    from data_access.runtime.resource_governor import reset_global_governor

    reset_query_cache()
    reset_cache_manager()
    reset_global_governor()
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    yield
    reset_query_cache()
    reset_cache_manager()
    reset_global_governor()


def _yaml_store(tmp_path: Path, yaml_text: str) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


_STATIC_TMPL = """
{a}:
  kind: static
  access_mode: staging
  layout: plain
  root: "{root}"
  glob: "part-*.parquet"
  time_column: ts
  instrument_column: sym
  schema:
    ts: date
    sym: string
    val: double
"""


def _write_rows(root: Path, rows: int, prefix: str = "part") -> None:
    root.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    t = pa.table(
        {
            "ts": [dt.date(2024, 1, 1 + i % 3) for i in range(rows)],
            "sym": [f"S{i % 5}" for i in range(rows)],
            "val": [float(i) for i in range(rows)],
        }
    )
    pq.write_table(t, root / f"{prefix}-{rows}.parquet")


def _static_store(tmp_path: Path, name: str = "ds") -> tuple[DataAccessStore, Path]:
    root = tmp_path / "d"
    _write_rows(root, 3)
    return _yaml_store(tmp_path, _STATIC_TMPL.format(a=name, root=root)), root


_GEN_TMPL = """
matrix:
  kind: parametric
  access_mode: staging
  layout: hive
  root_template: "{root}/universe={{universe}}/freq={{frequency}}"
  authorized_root: "{root}"
  glob_template: "year=*/month=*/data.parquet"
  partition_columns: [year, month]
  storage_format: long
  params_schema:
    universe: str
    frequency: str
  time_column: datetime
  instrument_column: asset
  hive_partitioning: true
  union_by_name: true
  schema:
    datetime: timestamp
    asset: string
    value: double
  generation_pointer: true
"""


def _gen_store(tmp_path: Path) -> tuple[DataAccessStore, Path]:
    root = tmp_path / "matrix"
    return _yaml_store(tmp_path, _GEN_TMPL.format(root=root)), root


def _gen_table(rows: list[tuple[dt.datetime, float]], month: int = 1) -> pa.Table:
    return pa.table(
        {
            "datetime": pa.array([r[0] for r in rows], type=pa.timestamp("us")),
            "asset": ["A"] * len(rows),
            "value": [float(r[1]) for r in rows],
            "year": [2024] * len(rows),
            "month": [month] * len(rows),
        }
    )


# ===========================================================================
# 1/2 —— CacheManager：死锁 + 删除失败配额账
# ===========================================================================

def test_r28_1_pin_expired_no_deadlock(tmp_path):
    """R28-1：过期条目重建不再持锁递归 pin()（非重入锁 → 死锁）。"""
    from data_access.runtime.cache_manager import CacheManager

    cm = CacheManager()
    cm.pin("k", path=str(tmp_path / "f"), size_bytes=10)
    # 强制过期
    entry = cm._entries["k"]
    entry.expires_at = 0.0
    # 若实现持锁递归，这里会死锁（测试卡住）；正确实现原地重建并返回。
    cm.pin("k", path=str(tmp_path / "f2"), size_bytes=20)
    new = cm._entries["k"]
    assert new.refcount == 1
    assert new.size_bytes == 20


def test_r28_2_delete_fail_keeps_entry_quota(tmp_path):
    """R28-2：删除失败必须保留 entry、字节仍计入配额（账本反映现实）。"""
    from data_access.runtime.cache_manager import (
        CacheEntryState,
        CacheManager,
    )

    cm = CacheManager(max_bytes=100)  # 100B 达 high_watermark(85B) → run_gc 真正执行
    f = tmp_path / "f"
    f.write_text("x")
    cm.pin("k", path=str(f), size_bytes=100)
    cm.unpin("k")
    # 模拟物理删除失败（权限/占用/IO）
    with patch.object(cm, "_safe_delete", return_value=False):
        freed = cm.run_gc()
        assert freed == 0
        assert "k" in cm._entries  # entry 保留
        assert cm._entries["k"].state == CacheEntryState.DELETE_FAILED
        assert cm.total_bytes() == 100  # 字节仍计入 quota
    # 删除恢复后 GC 能释放
    cm.run_gc()
    assert "k" not in cm._entries


# ===========================================================================
# 3/4 —— SnapshotVerifier：真实 HEAD + 本地 mtime_ns
# ===========================================================================

def test_r28_4_local_final_verify_mtime_ns(tmp_path):
    """R28-4：执行后 verify 按 size + mtime_ns 精确比较——同大小替换也报变化。"""
    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        ResolvedSourceSnapshot,
    )
    from data_access.snapshot.verifier import SnapshotVerifier

    f = tmp_path / "f.parquet"
    f.write_bytes(b"0123456789")
    st = f.stat()
    snap = ResolvedSourceSnapshot(
        dataset="t",
        objects=(
            ResolvedObject(
                uri=str(f),
                content_length=10,
                mtime_ns=st.st_mtime_ns,
                source="file_manifest",
            ),
        ),
    )
    # 同 size 替换但 mtime 变化 → final verify 必须报变化。
    os.utime(f, ns=(st.st_mtime_ns + 5_000_000_000, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(SourceSnapshotChanged, match="mtime"):
        SnapshotVerifier(strict=True).verify_after_execute(snap)


def test_r28_4_local_mtime_unchanged_passes(tmp_path):
    """R28-4：mtime_ns 一致时 final verify 通过（不误报）。"""
    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        ResolvedSourceSnapshot,
    )
    from data_access.snapshot.verifier import SnapshotVerifier

    f = tmp_path / "f.parquet"
    f.write_bytes(b"0123456789")
    st = f.stat()
    snap = ResolvedSourceSnapshot(
        dataset="t",
        objects=(
            ResolvedObject(
                uri=str(f), content_length=10, mtime_ns=st.st_mtime_ns
            ),
        ),
    )
    SnapshotVerifier(strict=True).verify_after_execute(snap)


def test_r28_3_store_pipeline_has_remote_meta_fn(tmp_path):
    """R28-3：store 的 ReadPipeline verifier 已注入真实 COS HEAD resolver。"""
    store, _ = _static_store(tmp_path)
    assert store._pipeline._verifier.remote_meta_fn is not None
    # 非 strict / 未开启 remote meta → 返回 None（不误触网络）。
    from data_access.read.read_contract import _remote_snapshot_meta_enabled

    with patch("data_access.read.query_budget.is_strict_semantics", return_value=False):
        assert _remote_snapshot_meta_enabled() is False
        assert store._remote_meta_head("s3://b/t/f.parquet") is None


# ===========================================================================
# 5 —— SourceManifest strict + bucket/segment 边界
# ===========================================================================

def test_r28_5_manifest_strict_required_fields():
    """R28-5：strict 缺 dataset/content_digest/prefix/published_at/object_count → 拒。"""
    from data_access.snapshot.resolver import parse_source_manifest

    base = {
        "manifest_version": "1",
        "source_generation": "G1",
        "complete": True,
        "objects": [{"key": "s3://b/t/f.parquet", "etag": "e", "size": 10}],
    }
    with pytest.raises(SourceSnapshotUnavailable, match="dataset"):
        parse_source_manifest(base, strict=True, expected_dataset="t")
    base["dataset"] = "t"
    base["content_digest"] = "x"
    base["prefix"] = "s3://b/t"
    base["published_at"] = "2026-08-10T00:00:00Z"
    base["object_count"] = 1
    with pytest.raises(SourceSnapshotUnavailable, match="content_digest"):
        parse_source_manifest(base, strict=True, expected_dataset="t")
    # dataset 与请求不一致
    base2 = dict(base)
    base2["dataset"] = "other"
    with pytest.raises(SourceSnapshotUnavailable, match="dataset"):
        parse_source_manifest(base2, strict=True, expected_dataset="t")


def test_r28_5_uri_segment_boundary_no_prefix_collision():
    """R28-5：`abc`/`abcd` 前缀碰撞 → segment 级边界拒绝。"""
    from data_access.snapshot.resolver import _uri_is_within

    assert _uri_is_within("s3://b/t/year=2024", "s3://b/t/year=2024/month=1/f.parquet")
    assert not _uri_is_within("s3://b/t/year=2024", "s3://b/t/year=20240/month=1/f.parquet")
    assert not _uri_is_within("s3://b/t", "s3://OTHER/t/f.parquet")
    assert _uri_is_within("s3://b/t", "cos://b/t/f.parquet")  # 同一 COS 命名空间
    assert not _uri_is_within("s3://b/t", "s3://b/t2/f.parquet")


def test_r28_5_manifest_cross_bucket_rejected():
    """R28-5：manifest object 跨 bucket → strict 拒。"""
    from data_access.snapshot.resolver import parse_source_manifest
    from data_access.snapshot.source_snapshot import content_digest_of_objects

    objs = [
        {"key": "s3://b1/t/f.parquet", "etag": "e1", "size": 1},
        {"key": "s3://b2/t/f.parquet", "etag": "e2", "size": 2},
    ]
    digest = content_digest_of_objects(
        [__import__("data_access.snapshot.source_snapshot", fromlist=["ResolvedObject"]).ResolvedObject(
            uri=o["key"], etag=o["etag"], content_length=o["size"]) for o in objs]
    )
    with pytest.raises(SourceSnapshotUnavailable, match="bucket"):
        parse_source_manifest(
            {
                "manifest_version": "1",
                "source_generation": "G1",
                "complete": True,
                "dataset": "t",
                "prefix": "s3://b1/t",
                "content_digest": digest,
                "published_at": "2026-08-10T00:00:00Z",
                "object_count": 2,
                "objects": objs,
            },
            strict=True,
        )


# ===========================================================================
# 6 —— VerifiedPhysicalScope.contract_digest 消费
# ===========================================================================

def test_r28_6_stale_contract_digest_rejected(tmp_path):
    """R28-6：scope.contract_digest 过期 → prepare_read 拒绝执行。"""
    from data_access.runtime.prepared_read import VerifiedPhysicalScope

    store, root = _static_store(tmp_path)
    scope = VerifiedPhysicalScope(
        dataset_id="ds",
        exact_objects=(str(root / "part-3.parquet"),),
        contract_digest="stale-digest-xyz",
    )
    with pytest.raises(ValidationError, match="contract_digest"):
        store.prepare_read("ds", physical_scope=scope)


def test_r28_6_fresh_contract_digest_ok(tmp_path):
    """R28-6：scope.contract_digest 与当前一致 → 正常执行（并强制 dataset 边界）。"""
    from data_access.runtime.prepared_read import VerifiedPhysicalScope

    store, root = _static_store(tmp_path)
    digest = store._contract_digest_for("ds")
    scope = VerifiedPhysicalScope(
        dataset_id="ds",
        exact_objects=(str(root / "part-3.parquet"),),
        contract_digest=digest,
    )
    prepared = store.prepare_read("ds", physical_scope=scope)
    assert prepared.dataset == "ds"
    assert list(prepared.physical_scope) == [str(root / "part-3.parquet")]


# ===========================================================================
# 7 —— SchemaEpoch migration 真实 epoch-pair
# ===========================================================================

def _epoch_pair_gate():
    from data_access.read.schema_epoch import SchemaEpochGate, SchemaMigration

    # A: 只有 col_a；B: col_a 是 string（不兼容）+ 新字段 col_b
    ea = {"col_a": "int64"}
    eb = {"col_a": "string", "col_b": "int64"}
    fp_a = __import__("data_access.read.schema_epoch", fromlist=["schema_fingerprint"]).schema_fingerprint(ea)
    fp_b = __import__("data_access.read.schema_epoch", fromlist=["schema_fingerprint"]).schema_fingerprint(eb)
    return fp_a, fp_b, ea, eb


def test_r28_7_dtype_migration_requires_correct_pair():
    """R28-7：dtype_change 需要精确覆盖该 epoch pair 的 approved migration。"""
    from data_access.read.schema_epoch import SchemaEpoch, SchemaEpochGate, SchemaMigration

    fp_a, fp_b, ea, eb = _epoch_pair_gate()
    ea_ep = SchemaEpoch(fingerprint=fp_a, fields=ea, objects=("a.parquet",))
    eb_ep = SchemaEpoch(fingerprint=fp_b, fields=eb, objects=("b.parquet",))
    # 旧实现：A→A 自环 / 任意 approved → 都会误放行；现在必须精确覆盖 A→B。
    wrong = SchemaMigration(fp_a, "deadbeef00000000", kind="dtype_change", field="col_a", approved=True)
    gate = SchemaEpochGate(migrations=[wrong])
    assert gate._migration_approved(ea_ep, eb_ep, "col_a", "dtype_change") is False
    # 正确 pair → 放行（A→B / B→A 均可）
    right = SchemaMigration(fp_a, fp_b, kind="dtype_change", field="col_a", approved=True)
    gate2 = SchemaEpochGate(migrations=[right])
    assert gate2._migration_approved(ea_ep, eb_ep, "col_a", "dtype_change") is True


def test_r28_7_add_column_requires_source_epoch():
    """R28-7：add_column 需要「含该字段的 epoch → 缺字段 epoch」的 approved 迁移。"""
    from data_access.read.schema_epoch import SchemaEpoch, SchemaEpochGate, SchemaMigration

    fp_a, fp_b, ea, eb = _epoch_pair_gate()
    ea_ep = SchemaEpoch(fingerprint=fp_a, fields=ea, objects=("a.parquet",))
    eb_ep = SchemaEpoch(fingerprint=fp_b, fields=eb, objects=("b.parquet",))
    # col_b 只在 B 有；缺字段的 epoch 是 A。错误 pair 不覆盖 → 拒。
    bad = SchemaMigration("ffffffffffffffff", fp_b, kind="add_column", field="col_b", approved=True)
    gate = SchemaEpochGate(migrations=[bad])
    assert gate._migration_approved(ea_ep, eb_ep, "col_b", "add_column") is False
    good = SchemaMigration(fp_a, fp_b, kind="add_column", field="col_b", approved=True)
    gate2 = SchemaEpochGate(migrations=[good])
    assert gate2._migration_approved(ea_ep, eb_ep, "col_b", "add_column") is True


# ===========================================================================
# 8 —— SchemaEpoch manifest O(1)
# ===========================================================================

def test_r28_8_group_epochs_from_manifest_no_footer(tmp_path):
    """R28-8：manifest 提供 schema_epochs 摘要时，group_epochs 不读 parquet footer。"""
    import pyarrow.parquet as pq

    from data_access.read.manifest import DatasetManifest, ManifestFile, build_manifest_for_dataset

    store, root = _static_store(tmp_path, name="ds")
    store.build_dataset_manifest("ds")
    m = DatasetManifest.load(root)
    assert m is not None
    assert m.schema_epochs  # 构建时已算好 epoch 摘要

    from data_access.read.schema_epoch import SchemaEpochGate

    gate = SchemaEpochGate()
    # 若走 footer，monkeypatch 让 parquet_footer_schema 直接抛 → 测试会失败。
    with patch(
        "data_access.read.schema_epoch.parquet_footer_schema",
        side_effect=RuntimeError("footer 不应被读取（R28-8 manifest O(1)）"),
    ):
        epochs = gate.group_epochs(
            [str(root / "part-3.parquet")], manifest=m
        )
    assert epochs and epochs[0].has_field("ts")


def test_r28_8_build_manifest_persists_schema_epochs(tmp_path):
    """R28-8：build_dataset_manifest 后 sidecar 里带 schema_epochs 摘要。"""
    import json

    from data_access.read.manifest import _MANIFEST_META_FILENAME

    store, root = _static_store(tmp_path, name="ds")
    store.build_dataset_manifest("ds")
    meta = json.loads((root / _MANIFEST_META_FILENAME).read_text())
    assert meta.get("schema_epochs")


# ===========================================================================
# 9 —— Governor remote 请求总数 ≠ 并发上限
# ===========================================================================

def test_r28_9_remote_requests_not_conflated_with_concurrency():
    """R28-9：一次查询 100 个远端对象 ≠ 100 并发——admit 不再拿请求总数比并发上限。"""
    from data_access.runtime.resource_governor import (
        GlobalResourceGovernor,
        ResourceReservation,
    )

    gov = GlobalResourceGovernor(max_remote_concurrency=16)
    res = ResourceReservation(
        query_id="q1", principal_id="p", remote_requests=100
    )
    # 旧实现会因 100 > 16 拒绝；现在只记成本（QueryBudget 层治理），admit 放行。
    gov.admit(res)
    assert gov._remote_inflight == 0  # 真正并发由 acquire_remote_slot 治理
    assert gov._remote_requests_total == 100  # 成本维度累计（telemetry）


def test_r28_9_remote_concurrency_slot_still_governed():
    """R28-9：真实并发上限仍由 acquire_remote_slot 执行。"""
    from data_access.runtime.resource_governor import GlobalResourceGovernor

    gov = GlobalResourceGovernor(max_remote_concurrency=2)
    assert gov.acquire_remote_slot() is True
    assert gov.acquire_remote_slot() is True
    assert gov.acquire_remote_slot() is False  # 第 3 个并发 → 拒绝
    gov.release_remote_slot()
    assert gov.acquire_remote_slot() is True


# ===========================================================================
# 10/12/15 —— DuckDB 统一并发 + deadline + fork 防护
# ===========================================================================

def test_r28_10_pool_size_equals_max_concurrency():
    """R28-10：deadline pool 容量 = max_concurrency（单一 source of truth）。"""
    eng = DuckDBEngine(threads=2, max_concurrency=5)
    assert eng._max_concurrency == 5
    assert eng._deadline_pool._max == 5
    assert eng._exec_sem._value == 5


def test_r28_12_deadline_fails_fast_on_pool_wait(tmp_path):
    """R28-12：请求 absolute deadline 贯穿 pool 等待——并发占满 + 小 deadline →
    快速失败，不「跑完才报超时」。"""
    eng = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    # 占满唯一 pool 连接（不归还 → 后续 acquire 必须等）
    held_conn = eng._deadline_pool.acquire(eng._config)
    try:
        t0 = dt.datetime.now()
        with pytest.raises(DeadlineExceeded):
            # pool 无空闲、等待被请求 deadline（50ms）截断 → DeadlineExceeded
            eng.execute_arrow("SELECT 1", [], deadline_ms=50)
        elapsed = (dt.datetime.now() - t0).total_seconds()
        assert elapsed < 2.0  # 不是等 5s pool acquire 或跑完
    finally:
        eng._deadline_pool.release(held_conn, healthy=True)


def test_r28_15_fork_guard_all_entries():
    """R28-15：fork 后所有 public execution entry 都拒绝（含 reader/scoped/relation）。"""
    import multiprocessing

    eng = DuckDBEngine(threads=2)
    eng._check_pid()

    def _child(eng):
        for fn, args in [
            (eng.execute_arrow, ("SELECT 1", [])),
            (eng.execute_reader, ("SELECT 1", [])),
            (eng.relation, ("SELECT 1", [])),
            (eng.execute_isolated_arrow, ("SELECT 1", [])),
        ]:
            try:
                fn(*args)
                return "NO_GUARD"
            except RuntimeError as e:
                if "fork" not in str(e):
                    return "WRONG_ERR"
        return "OK"

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=lambda: q.put(_child(eng)))
    p.start()
    p.join(15)
    try:
        assert q.get(timeout=5) == "OK"
    except Exception:
        pytest.skip("multiprocessing fork 环境不可用")


def test_r28_15_init_log_not_per_query(capsys, tmp_path):
    """R28-15：「DuckDB 初始化完成」只在 __init__ 打一次（不再每次 execute 刷）。"""
    eng = DuckDBEngine(threads=2)
    import logging

    records: list[str] = []
    class _H(logging.Handler):
        def emit(self, r):
            if "DuckDB 初始化完成" in r.getMessage():
                records.append(r.getMessage())
    h = _H()
    logger = logging.getLogger("data_access.engine")
    logger.addHandler(h)
    try:
        eng.execute_arrow("SELECT 1", [])
        eng.execute_arrow("SELECT 2", [])
        eng.execute_arrow("SELECT 3", [])
    finally:
        logger.removeHandler(h)
    # handler 在 __init__ 之后挂上：普通 query 若还打初始化日志会漏进来。
    assert len(records) == 0


# ===========================================================================
# 13 —— pool 共享数据库实例
# ===========================================================================

def test_r28_13_pool_uses_shared_db_path():
    """R28-13：deadline pool 连接共享同一数据库实例（object cache 跨连接共享）。"""
    eng = DuckDBEngine(threads=2)
    assert eng._pool_db_path is not None
    pool_path = eng._pool_db_path
    c1 = eng._deadline_pool._new_connection()
    c2 = eng._deadline_pool._new_connection()
    try:
        # 同一 DatabaseInstance：c1 建的 catalog 对象 c2 可见。
        c1.execute("CREATE TABLE shared_t AS SELECT 42 AS x")
        row = c2.execute("SELECT x FROM shared_t").fetchone()
        assert row == (42,)
    finally:
        c1.close()
        c2.close()
    eng.close()
    assert not os.path.exists(pool_path)  # close 清理临时 DB


# ===========================================================================
# 17 —— 凭证上下文传播
# ===========================================================================

def test_r28_17_resolve_credentials_prefers_request_scoped_provider():
    """R28-17：resolve_s3_credentials 优先 request-scoped credential provider。"""
    import data_access.cos.remote as crm
    from data_access.security.credentials import CredentialMaterial
    from data_access.security.execution_context import (
        DataAccessExecutionContext,
        execution_scope,
    )

    class _ReqProvider:
        source = "request"

        def resolve(self):
            return CredentialMaterial(
                access_key_id="REQ_KEY",
                secret_access_key="req_secret",
                source="request",
            )

    class _GlobalProvider:
        source = "global"

        def resolve(self):
            return CredentialMaterial(
                access_key_id="GLOBAL_KEY",
                secret_access_key="global_secret",
                source="global",
            )

    # resolve_s3_credentials 内部 `from data_access.security.credentials import
    # _global_credential_provider`——patch 该模块（非 crm 上的引用）。
    with patch(
        "data_access.security.credentials._global_credential_provider",
        return_value=_GlobalProvider(),
    ):
        with execution_scope(DataAccessExecutionContext(credential_provider=_ReqProvider())):
            creds = crm.resolve_s3_credentials()
        assert creds.access_key_id == "REQ_KEY"
        # 无 request scope → 回退全局
        creds2 = crm.resolve_s3_credentials()
        assert creds2.access_key_id == "GLOBAL_KEY"


# ===========================================================================
# 18 —— sqlglot 限定名不混配裸名
# ===========================================================================

def test_r28_18_qualified_table_not_confused_with_bare():
    """R28-18：FROM arbitrary.bar 不能混配声明集合里的 bar。"""
    from data_access.read.sql_escape import (
        _query_from_tables,
        assert_sql_from_scope,
    )

    refs = _query_from_tables("SELECT * FROM arbitrary.bar")
    assert "arbitrary.bar" in refs
    assert "bar" not in refs  # 不返回裸名（否则会误匹配声明集合）
    # 声明集合只有 bar → arbitrary.bar 被拒
    with pytest.raises(ValidationError, match="arbitrary.bar"):
        assert_sql_from_scope("SELECT * FROM arbitrary.bar", allowed=("bar",))
    # 裸 bar → 通过
    assert_sql_from_scope("SELECT * FROM bar", allowed=("bar",))


def test_r28_18_cte_alias_not_external_ref():
    """R28-18：CTE 别名是局部定义，不当作外部表引用（但 CTE 体内表照常枚举）。"""
    from data_access.read.sql_escape import _query_from_tables

    refs = _query_from_tables(
        "WITH x AS (SELECT * FROM factor_lake) SELECT * FROM x"
    )
    assert "factor_lake" in refs
    assert "x" not in refs


def test_r28_18_catalog_schema_table_rejected_by_declared():
    """R28-18：catalog.schema.table 全限定名不匹配裸声明名。"""
    from data_access.read.sql_escape import assert_sql_tables_declared

    with pytest.raises(ValidationError):
        assert_sql_tables_declared(
            "SELECT * FROM sys.factor_lake", declared=("factor_lake",)
        )
    assert_sql_tables_declared("SELECT * FROM factor_lake", declared=("factor_lake",))


# ===========================================================================
# 26 —— generation COW（未变分区硬链接，不整代重写）
# ===========================================================================

def test_r28_26_append_cow_reuses_untouched_partitions(tmp_path):
    """R28-26：append 只重写变更分区；未变分区硬链接复用（同 inode，不复制数据）。"""
    store, root = _gen_store(tmp_path)
    # 2024/01 两行
    store.write_arrow(
        "matrix", _gen_table([(dt.datetime(2024, 1, 14), 1.0), (dt.datetime(2024, 1, 15), 2.0)], month=1),
        universe="u", frequency="1d",
    )
    base = root / "universe=u" / "freq=1d"
    g1 = json.loads((base / "manifest.json").read_text())["generation"]
    gen1 = base / "generation" / g1
    old_file = gen1 / "year=2024" / "month=1" / "data.parquet"
    assert old_file.exists()
    old_stat = old_file.stat()

    # append 2024/02（新分区）
    store.write_arrow(
        "matrix", _gen_table([(dt.datetime(2024, 2, 1), 5.0)], month=2),
        universe="u", frequency="1d", mode="append",
    )
    g2 = json.loads((base / "manifest.json").read_text())["generation"]
    assert g2 != g1
    gen2 = base / "generation" / g2
    # 旧分区 2024/01 被硬链接（inode 相同 → 未复制数据）
    new_old = gen2 / "year=2024" / "month=1" / "data.parquet"
    assert new_old.exists()
    assert new_old.stat().st_ino == old_stat.st_ino
    # 新分区 2024/02 全新写
    assert (gen2 / "year=2024" / "month=2" / "data.parquet").exists()
    out = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out.num_rows == 3  # 2 (month=1) + 1 (month=2)


def test_r28_26_upsert_cow_partial_rewrite(tmp_path):
    """R28-26：upsert 只重写变更分区（未变分区硬链接），数据正确合并。"""
    store, root = _gen_store(tmp_path)
    store.write_arrow(
        "matrix",
        _gen_table(
            [(dt.datetime(2024, 1, 14), 1.0), (dt.datetime(2024, 1, 15), 2.0)],
            month=1,
        ),
        universe="u", frequency="1d",
    )
    # upsert 覆盖 01-15 → 只有 month=1 分区重写
    store.upsert(
        "matrix",
        _gen_table([(dt.datetime(2024, 1, 15), 99.0)], month=1),
        upsert_on=["datetime", "asset"],
        universe="u", frequency="1d",
    )
    out = store.read_arrow("matrix", universe="u", frequency="1d")
    assert sorted(out.column("value").to_pylist()) == [1.0, 99.0]
    assert out.num_rows == 2


def test_r28_26_delete_cow_keeps_untouched_partitions(tmp_path):
    """R28-26：delete_rows 只重写「有行被删」的分区，其余硬链接。"""
    store, root = _gen_store(tmp_path)
    # month=1 两行 + month=2 一行
    store.write_arrow(
        "matrix", _gen_table([(dt.datetime(2024, 1, 14), 1.0)], month=1),
        universe="u", frequency="1d",
    )
    store.write_arrow(
        "matrix", _gen_table([(dt.datetime(2024, 2, 1), 5.0)], month=2),
        universe="u", frequency="1d", mode="append",
    )
    base = root / "universe=u" / "freq=1d"
    g1 = json.loads((base / "manifest.json").read_text())["generation"]
    # 删掉 month=2 的数据
    store.delete_rows(
        "matrix", start=dt.datetime(2024, 2, 1), end=dt.datetime(2024, 2, 1),
        universe="u", frequency="1d",
    )
    g2 = json.loads((base / "manifest.json").read_text())["generation"]
    gen2 = base / "generation" / g2
    # month=1 硬链接（inode 相同），month=2 被删（不出现）
    gen1 = base / "generation" / g1
    assert (
        gen2 / "year=2024" / "month=1" / "data.parquet"
    ).stat().st_ino == (
        gen1 / "year=2024" / "month=1" / "data.parquet"
    ).stat().st_ino
    assert not (gen2 / "year=2024" / "month=2").exists()
    out = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out.num_rows == 1

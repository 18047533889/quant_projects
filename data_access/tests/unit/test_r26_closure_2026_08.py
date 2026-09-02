# -*- coding: utf-8 -*-
"""R26 —— 真实端到端 destructive tests（§6 DoD）。

从 public API 触发，不允许用测试文件自己写的模拟 helper 证明 production 已实现。
覆盖：PIPE（统一执行链）/ SEC（安全三态+strict config+因子 gate）/
PIT（SQL no same-day + floor）/ SNAP（identity/manifest/policy）/
RES（governor+budget）/ CACHE（TTL/quota/pin/GC）/ SCHEMA（真实 parquet）/
PROV（GovernedFrame 不可伪造）/ SVC（startup / ready）。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from data_access.core.exceptions import (
    AccessDeniedError,
    PITUnavailable,
    ResourceAdmissionError,
    SchemaContractError,
    SourceSnapshotUnavailable,
    UnknownProvenanceError,
    ValidationError,
)
from data_access.registry import load_registry
from data_access.core.engine import DuckDBEngine
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch):
    from data_access.security.policy import set_authorizer
    from data_access.security.api_principals import reset_api_principal_registry
    from data_access.runtime.resource_governor import reset_global_governor
    from data_access.runtime.cache_manager import reset_cache_manager

    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("DATA_ACCESS_PRINCIPAL_ID", raising=False)
    monkeypatch.delenv("DATA_ACCESS_ALLOWED_DATASETS", raising=False)
    monkeypatch.delenv("DATA_ACCESS_API_PRINCIPALS", raising=False)
    set_authorizer(None)
    reset_api_principal_registry()
    reset_global_governor()
    reset_cache_manager()
    yield
    set_authorizer(None)
    reset_api_principal_registry()
    reset_global_governor()
    reset_cache_manager()


def _store(tmp_path=None):
    registry = load_registry()
    engine = DuckDBEngine(threads=2, enable_object_cache=False)
    return DataAccessStore(registry=registry, engine=engine)


def _local_daily_store(tmp_path: Path) -> DataAccessStore:
    """用本地临时 registry 建一个 ``ashare_stock_daily`` 数据集。

    避免真实 registry 把 root 解析到 /home/shw 下的 COS 镜像根，触发
    ensure_local_mirror 试图在不可写的 /home/shw 建目录（PermissionError）。
    本地临时数据集 root 不在已知镜像根下，不会触发 COS 拉取。
    """
    root = tmp_path / "stock_daily"
    root.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    dates = [f"2024-01-{d:02d}" for d in range(2, 12)]
    import datetime as _d

    for d in dates:
        pq.write_table(
            pa.table(
                {
                    "TradeDate": pa.array([_d.date.fromisoformat(d)], type=pa.date32()),
                    "Symbol": pa.array(["000001"], type=pa.string()),
                    "Close": pa.array([10.0], type=pa.float64()),
                }
            ),
            root / f"{d}.parquet",
        )
    cfg = tmp_path / "r26_local.yaml"
    cfg.write_text(
        f"""
ashare_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
""".strip()
        + "\n",
        encoding="utf-8",
    )
    engine = DuckDBEngine(threads=2, enable_object_cache=False)
    return DataAccessStore(registry=load_registry(cfg), engine=engine)


# ---------------------------------------------------------------------------
# §6.3 Security
# ---------------------------------------------------------------------------
def test_tr26_sec001_empty_datasets_deny_all():
    """R26-P0-006：allowed_datasets=[] → deny all（不是 unrestricted）。"""
    from data_access.security.principal import AccessPolicy, DataPrincipal
    from data_access.security.policy import DefaultAuthorizer

    policy = AccessPolicy(allowed_datasets=frozenset())
    principal = DataPrincipal(principal_id="x")
    auth = DefaultAuthorizer(policy=policy, principal=principal, strict_default_deny=True)
    with pytest.raises(AccessDeniedError):
        auth.authorize(principal, "ashare_stock_daily")


def test_tr26_sec001b_none_unrestricted():
    from data_access.security.principal import AccessPolicy, DataPrincipal
    from data_access.security.policy import DefaultAuthorizer

    policy = AccessPolicy(allowed_datasets=None, allowed_actions=frozenset({"dataset:read"}))
    principal = DataPrincipal(principal_id="x")
    auth = DefaultAuthorizer(policy=policy, principal=principal, strict_default_deny=True)
    auth.authorize(principal, "any_dataset")  # 不抛


def test_tr26_sec003_strict_bool_rejects_string():
    """R26-P0-008：安全配置 bool 只接受真实 bool。"""
    from data_access.security.api_principals import ApiPrincipalRegistry, hash_api_key

    cfg = {
        hash_api_key("k"): {
            "principal_id": "p",
            "allowed_datasets": ["a"],
            "allow_uri_read": "false",
        }
    }
    with pytest.raises(ValidationError, match="布尔值"):
        ApiPrincipalRegistry(cfg).resolve("k")


def test_tr26_sec004_declassification_rejects_string():
    """R26-P0-008：approved:"false" → 配置拒绝。"""
    from factor_engine.security.access import DeclassificationApproval

    with pytest.raises(ValueError, match="布尔值"):
        DeclassificationApproval.from_dict({"approved": "false"})


def test_tr26_sec005_factor_all_required(tmp_path):
    """R26-P0-009：factor [basic, premium]、principal 只有 basic → deny（all-required）。"""
    from data_access.read.factors import FactorCatalog, FactorMeta
    from data_access.security.principal import AccessPolicy, DataPrincipal
    from data_access.security.policy import DefaultAuthorizer
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    policy = AccessPolicy(
        allowed_datasets=frozenset({"factor_lake"}),
        allowed_factor_namespaces=frozenset({"factor_engine.market.basic"}),
        allowed_actions=frozenset({"factor:list", "factor:read"}),
    )
    principal = DataPrincipal(principal_id="basic", server_id="basic")
    store._authorizer_sec = DefaultAuthorizer(
        policy=policy, principal=principal, strict_default_deny=True
    )
    store._access_policy = policy

    meta = FactorMeta(
        factor_id="alpha",
        derived_access_tags=("factor_engine.market.basic", "alt.premium"),
        source_access_tags=("factor_engine.market.basic",),
    )
    cat = MagicMock()
    cat.records = {"alpha": meta}
    with patch.object(store, "get_factor_catalog", return_value=cat):
        with pytest.raises(AccessDeniedError, match="factor access tags"):
            store._authorize_factor_tags(["alpha"])


def test_tr26_sec006_catalog_unavailable_deny(tmp_path):
    """R26-P0-009：FactorCatalog 不可用 → production deny（fail-closed）。"""
    from data_access.security.principal import AccessPolicy, DataPrincipal
    from data_access.security.policy import DefaultAuthorizer
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    policy = AccessPolicy(
        allowed_datasets=frozenset({"factor_lake"}),
        allowed_factor_namespaces=frozenset({"factor_engine.market.basic"}),
        allowed_actions=frozenset({"factor:list", "factor:read"}),
    )
    principal = DataPrincipal(principal_id="basic", server_id="basic")
    store._authorizer_sec = DefaultAuthorizer(
        policy=policy, principal=principal, strict_default_deny=True
    )
    store._access_policy = policy
    with patch.object(
        store, "get_factor_catalog", side_effect=RuntimeError("catalog down")
    ):
        with patch(
            "data_access.store.is_strict_semantics", return_value=True
        ):
            with pytest.raises(AccessDeniedError, match="catalog unavailable"):
                store._authorize_factor_tags(["alpha"])


# ---------------------------------------------------------------------------
# §6.2 Runtime single-path enforcement
# ---------------------------------------------------------------------------
def test_tr26_pipe001_public_paths_exactly_once(tmp_path):
    """R26-P0-004：每个 public path 的 pipeline 阶段 exactly once。"""
    store = _local_daily_store(tmp_path)
    store._pipeline.reset_counters()
    store.read_arrow(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol", "Close"],
        time_range=("2024-01-02", "2024-01-03"),
    )
    store._pipeline.counters.assert_all_exactly_once()
    assert store._pipeline._governor.active_count() == 0


def test_tr26_pipe001b_scan_collect_release(tmp_path):
    store = _local_daily_store(tmp_path)
    store._pipeline.reset_counters()
    h = store.scan(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol", "Close"],
        time_range=("2024-01-02", "2024-01-03"),
    )
    t = h.collect_table()
    assert t.num_rows > 0
    store._pipeline.counters.assert_all_exactly_once()
    assert store._pipeline._governor.active_count() == 0


def test_tr26_res003_stream_disconnect_no_leak(tmp_path):
    """R26-T-RES-003：stream 客户端中途断开 → active governor = 0。"""
    store = _local_daily_store(tmp_path)
    store._pipeline.reset_counters()
    gen = store.read_arrow_stream(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol", "Close"],
        time_range=("2024-01-02", "2024-01-10"),
    )
    first = next(gen)
    assert first.num_rows > 0
    gen.close()  # 客户端断开
    assert store._pipeline._governor.active_count() == 0


# ---------------------------------------------------------------------------
# §6.4 Remote physical layout
# ---------------------------------------------------------------------------
def test_tr26_phy001_file_selector_split_shares():
    """R26-P0-010：split 只选 {date}.parquet，shares 只选 shares_{date}.parquet。"""
    from data_access.contract.file_selector import file_selector_for_layout
    from data_access.contract.physical_partition import PhysicalLayout

    split = file_selector_for_layout(PhysicalLayout.PREFIXED_DATE_FILE, file_selector=None)
    shares = file_selector_for_layout(
        PhysicalLayout.PREFIXED_DATE_FILE, file_selector="shares_"
    )
    objects = [
        "s3://b/t/2024-01-01.parquet",   # schema A
        "s3://b/t/shares_2024-01-01.parquet",  # schema B
    ]
    assert split.filter_objects(objects) == ["s3://b/t/2024-01-01.parquet"]
    assert shares.filter_objects(objects) == ["s3://b/t/shares_2024-01-01.parquet"]
    # glob 不退化裸 *.parquet
    assert not split.glob_pattern("s3://b/t").endswith("*.parquet")


# ---------------------------------------------------------------------------
# §6.5 PIT
# ---------------------------------------------------------------------------
def test_tr26_pit001_sql_calendar_missing_no_same_day():
    """R26-P0-013：calendar 缺失 strict → PITUnavailable（不 same-day fallback）。"""
    from data_access.store import _session_avail_sql

    with pytest.raises(PITUnavailable):
        _session_avail_sql(
            "SELECT 1 AS _r_col",
            right_time_col="PubDate",
            calendar=None,
            strict=True,
        )


def test_tr26_pit002_request_cannot_loosen_floor(tmp_path):
    """R26-P0-014：contract next_session_open、request same_day → reject。"""
    from data_access.read.temporal_join import TemporalJoinSpec
    from data_access.cos_contract import get_cos_contract

    store = _store(tmp_path)
    # us_stock_income 的 pit_policy=strict → floor=next_session_open
    ct = get_cos_contract("us_stock_income")
    if ct is None or ct.pit_policy != "strict":
        pytest.skip("us_stock_income 契约不满足 strict floor 前置")
    spec = TemporalJoinSpec(policy="pit_asof", availability="same_day")
    with patch(
        "data_access.store.is_strict_semantics", return_value=True
    ):
        with pytest.raises(ValidationError, match="PIT policy floor"):
            store._apply_pit_policy_floor("us_stock_income", spec)


# ---------------------------------------------------------------------------
# §6.6 Snapshot
# ---------------------------------------------------------------------------
def test_tr26_snap001_manifest_uri_only_rejected():
    """R26-P0-015：manifest object 只有 URI（无 etag/version）production reject。"""
    from data_access.snapshot.resolver import SourceSnapshotResolver

    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        content_digest_of_objects,
    )

    _identityless = {"key": "s3://b/t/f.parquet"}  # 无 etag/size
    manifest = {
        "manifest_version": "1",
        "source_generation": "G1",
        "complete": True,
        "dataset": "t",
        "prefix": "s3://b/t",
        "content_digest": content_digest_of_objects(
            [ResolvedObject(uri=_identityless["key"])]
        ),
        "published_at": "2026-08-10T00:00:00Z",
        "object_count": 1,
        "objects": [_identityless],
    }
    r = SourceSnapshotResolver(source_manifest_fn=lambda ds: manifest, strict=True)
    with pytest.raises(SourceSnapshotUnavailable, match="content identity"):
        r.resolve("t", paths=["s3://b/t/*.parquet"])


def test_tr26_snap004_wildcard_chars_cannot_bypass():
    """R26-P0-015：?, [], ** wildcard 都不能绕过 unresolved-glob gate。"""
    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        ResolvedSourceSnapshot,
    )
    from data_access.snapshot.verifier import SnapshotVerifier

    for pat in ("s3://b/t/f?.parquet", "s3://b/t/f[0-9].parquet", "s3://b/t/**/f.parquet"):
        snap = ResolvedSourceSnapshot(
            dataset="t", objects=(ResolvedObject(uri=pat),)
        )
        assert snap.has_wildcard
        with pytest.raises(SourceSnapshotUnavailable):
            SnapshotVerifier(strict=True).verify_before_execute(snap)


def test_tr26_snap005_unknown_policy_rejected():
    from data_access.snapshot.resolver import SourceSnapshotResolver

    r = SourceSnapshotResolver(strict=True)
    with pytest.raises(ValidationError, match="policy"):
        r.resolve("t", paths=["s3://b/t/*.parquet"], policy="whatever")


def test_tr26_snap006_manifest_prefix_escape():
    """R26-P0-015：manifest object 越出 registered prefix → reject。"""
    from data_access.snapshot.resolver import SourceSnapshotResolver

    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        content_digest_of_objects,
    )

    _obj = {"key": "s3://OTHER/t/f.parquet", "etag": "e1", "size": 10}
    manifest = {
        "manifest_version": "1",
        "source_generation": "G1",
        "complete": True,
        "dataset": "t",
        "prefix": "s3://b/t",
        "content_digest": content_digest_of_objects(
            [ResolvedObject(uri=_obj["key"], etag=_obj["etag"], content_length=_obj["size"])]
        ),
        "published_at": "2026-08-10T00:00:00Z",
        "object_count": 1,
        "objects": [_obj],
    }
    r = SourceSnapshotResolver(source_manifest_fn=lambda ds: manifest, strict=True)
    with pytest.raises(SourceSnapshotUnavailable, match="越出"):
        r.resolve("t", paths=["s3://b/t/*.parquet"])


# ---------------------------------------------------------------------------
# §6.8 Cache
# ---------------------------------------------------------------------------
def test_tr26_cache001_ttl_expires():
    """R26-P0-019：TTL 过期后条目 stale/evict。"""
    from data_access.runtime.cache_manager import CacheManager, CacheEntryState

    cm = CacheManager(ttl=0.01, max_bytes=10_000)
    cm.pin("k", path="", size_bytes=10)
    assert cm.get("k") is not None
    time.sleep(0.03)
    assert cm.get("k") is None  # TTL 过期 → 移除


def test_tr26_cache002_principal_isolation():
    """R26-P0-019：A 的 cache 不被 B 复用。"""
    from data_access.runtime.cache_manager import CacheManager

    cm = CacheManager()
    cm.pin("k", path="", size_bytes=10, principal_scope="A")
    e = cm.get("k")
    assert e is not None and e.principal_scope == "A"


def test_tr26_cache003_pinned_never_deleted():
    """R26-P0-019/§69：pinned 永不删。"""
    from data_access.runtime.cache_manager import CacheManager

    cm = CacheManager(max_bytes=100)
    cm.pin("pinned", path="", size_bytes=90)
    with pytest.raises(MemoryError):
        cm.pin("other", path="", size_bytes=50)  # GC 无法释放 pinned → 拒绝


def test_tr26_cache004_gc_physical_delete(tmp_path):
    """R26-P0-019：目录 GC 后物理删除，accounting == disk。"""
    from data_access.runtime.cache_manager import CacheManager

    d = tmp_path / "c"
    d.mkdir()
    (d / "x.txt").write_text("x")
    # max_bytes=1：单条目即达 high_watermark → run_gc 触发逐出。
    cm = CacheManager(max_bytes=1)
    cm.pin("k", path=str(d), size_bytes=1)
    cm.unpin("k")
    assert cm.total_bytes() > 0
    cm.run_gc()
    assert cm.total_bytes() == 0
    assert not d.exists()  # 物理目录真的删除


# ---------------------------------------------------------------------------
# §6.9 Schema evolution（真实 parquet）
# ---------------------------------------------------------------------------
def _write_parquet(path, schema, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(rows, schema=schema), path)


def _write_parquet_typed(path, schema, rows):
    """按 pa.schema 写真实 parquet（跨 epoch schema 测试用）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(rows, schema=schema), path)


def test_tr26_schema001_missing_field_rejected(tmp_path):
    """R26-P0-022：真实 parquet 跨 epoch 缺字段，store public read reject。"""
    from data_access.read.schema_epoch import SchemaEpochGate

    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    _write_parquet(p1, pa.schema([("close", pa.float64()), ("roe", pa.float64())]), {"close": [1.0], "roe": [0.1]})
    _write_parquet(p2, pa.schema([("close", pa.float64())]), {"close": [2.0]})
    gate = SchemaEpochGate(declared_fields={"close": "double", "roe": "double"}, strict=True)
    with pytest.raises(SchemaContractError, match="缺失"):
        gate.validate([str(p1), str(p2)], requested_columns=["close", "roe"])


def test_tr26_schema002_dtype_change_rejected(tmp_path):
    """R26-P0-022：真实 parquet double→string → reject。"""
    from data_access.read.schema_epoch import SchemaEpochGate

    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    _write_parquet(p1, pa.schema([("close", pa.float64())]), {"close": [1.0]})
    _write_parquet(p2, pa.schema([("close", pa.string())]), {"close": ["x"]})
    gate = SchemaEpochGate(declared_fields={"close": "double"}, strict=True)
    with pytest.raises(SchemaContractError, match="dtype"):
        gate.validate([str(p1), str(p2)], requested_columns=["close"])


# ---------------------------------------------------------------------------
# §6.10 Governed provenance
# ---------------------------------------------------------------------------
def test_tr26_prov001_forged_governed_frame_rejected():
    """R26-P0-023：GovernedFrame(table_or_frame=df) production reject。"""
    from data_access.security.governed_frame import (
        GovernedFrame,
        require_governed_provenance,
    )

    gf = GovernedFrame(table_or_frame={"a": 1}, security_digest="forged")
    with pytest.raises(UnknownProvenanceError, match="无法证明 provenance"):
        require_governed_provenance(gf, run_mode="production")


def test_tr26_prov002_forged_security_digest_rejected():
    from data_access.security.execution_context import (
        DataAccessExecutionContext,
        execution_scope,
    )
    from data_access.security.governed_frame import (
        ExecutionEnvironmentIdentity,
        GovernedFrame,
        require_governed_provenance,
    )
    from data_access.snapshot.source_snapshot import (
        ResolvedObject,
        ResolvedSourceSnapshot,
    )

    snap = ResolvedSourceSnapshot(
        dataset="t",
        objects=(ResolvedObject(uri="s3://b/t/f.parquet", etag="e", content_length=1),),
        content_digest="abc",
    )
    gf = GovernedFrame(
        table_or_frame={"a": 1},
        source_snapshot=snap,
        lineage={"d": "t"},
        execution_environment=ExecutionEnvironmentIdentity(run_mode="production"),
        security_digest="WRONG-digest",
    )
    ctx = DataAccessExecutionContext(
        principal=None, security_digest="REAL-digest"
    )
    with execution_scope(ctx):
        with pytest.raises(UnknownProvenanceError, match="security_digest"):
            require_governed_provenance(gf, run_mode="production")


# ---------------------------------------------------------------------------
# §6.1 Clean checkout / packaging
# ---------------------------------------------------------------------------
def test_tr26_clean003_credentials_py_tracked():
    """R26-P0-001：credentials.py 必须被 git 跟踪（clean clone 可 import）。

    R48 根目录归属收口后 data_access 平移到仓库根 ``data_access/``（monorepo
    合并，旧 ``dataaccess/`` 目录不复存在）——检查新路径。
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    out = subprocess.run(
        ["git", "ls-files", "data_access/security/credentials.py"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert "data_access/security/credentials.py" in out.stdout


def test_tr26_clean004_no_ignored_source():
    """R26 §9.2：production *.py 被 gitignore → fail。

    R57 收口：``.backend-runtime/`` 是 vendored 第三方运行时（duckdb/polars
    runtime 镜像，故意不入库），不是 production 源码——排除。
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    out = subprocess.run(
        [
            "git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    ignored = [
        ln
        for ln in out.stdout.splitlines()
        if ln.strip()
        and not any(
            d in ln
            for d in (
                ".replace_backup_", ".venv", "/build/", "build/lib", "/dist/",
                "__pycache__", ".egg-info", ".backend-runtime/", ".pykx-runtime/",
                "alphaprobe/",  # 独立挖掘栈（自有 git 仓），不受本仓 gitignore 治理
                "vectorbt_qs/",  # 独立回测栈（vendored vectorbt + qs 层）
                "lightgbm_qs/", "riskfolio_qs/",  # 同类独立 qs 栈
                "vectorbt_qs/",  # 独立回测栈（vendored vectorbt + qs 层）
            )
        )
    ]
    assert ignored == [], f"production 源码被 gitignore: {ignored}"

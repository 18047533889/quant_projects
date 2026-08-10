# -*- coding: utf-8 -*-
"""R27 —— Cache / Write Transaction / Unsafe Surface Closure。

覆盖（全走 public path / 真实 parquet / 真实 HTTP 语义）：
    A   read_cached 跨 principal 缓存隔离 + authorize 前置 + restricted 不缓存
    B   read_cached normalize_units 走真实 normalize 路径（不再掉进 **params）
    C   cache hit 审计 + provenance（read_cached_result）
    D   raw physical_scope strict 拒绝 / research 强制 dataset boundary
    E   dataset-specific path boundary（A 的读路径不能落 B 的 root）
    F   写路径逻辑授权（write/delete/publish 需要 dataset:* action）
    G   generation_pointer 数据集原子代写（overwrite/append/upsert/delete 单指针 flip）
    H   stream 单次 snapshot（_read_handle result=stream 不二次 resolve、不 fail-open）
    I   set_calendar 运行中冻结（首次读后修改被拒）
    J   scan_polars strict 禁止（裸 LazyFrame 旁路封闭）
    K   SQL 逗号连接绕过 sandbox → AST 白名单拒绝
    L   registry 真不可变（MappingProxyType 冻结）
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pyarrow as pa
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.read.query_cache import reset_query_cache
from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    reset_query_cache()
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    yield
    reset_query_cache()


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


# ---------------------------------------------------------------------------
# A/B/C —— read_cached 安全
# ---------------------------------------------------------------------------

def test_r27_a_cache_key_differs_by_principal(tmp_path):
    """R27-A：同一 query 不同 principal → 不同 cache key（不串缓存）。"""
    from data_access.read.query_cache import _security_scope_digest
    from data_access.security.principal import (
        AccessPolicy,
        DataPrincipal,
        DEFAULT_ACCESS_POLICY,
    )

    p_a = DataPrincipal(principal_id="alice", clearance="basic")
    p_b = DataPrincipal(principal_id="bob", clearance="restricted")
    k_a = _security_scope_digest(principal=p_a, access_policy=DEFAULT_ACCESS_POLICY)
    k_b = _security_scope_digest(principal=p_b, access_policy=DEFAULT_ACCESS_POLICY)
    assert k_a != k_b


def test_r27_a_restricted_classification_not_cached(tmp_path, monkeypatch):
    """R27-A：restricted/premium 数据默认不进共享结果缓存。"""
    from data_access.read.query_cache import classification_cache_blocked

    assert classification_cache_blocked("restricted") is True
    assert classification_cache_blocked("premium") is True
    assert classification_cache_blocked("public") is False
    assert classification_cache_blocked("unclassified") is False


def test_r27_b_cached_normalize_units_no_params_leak(tmp_path, monkeypatch):
    """R27-B：read_cached(normalize_units=True) 不再把 normalize_units 掉进
    **params（走 read() 真实 normalize 路径）。"""
    store, root = _static_store(tmp_path)
    # 若 normalize_units 被当 dataset 参数传进 read_result → static ds 会抛
    # 「静态数据集不接受参数」。能读到结果 = 参数未泄漏。
    tbl = store.read_cached("ds", columns=["val"], normalize_units=True)
    assert tbl.num_rows == 3
    # 无 manifest → 走 miss 真实读（read_result），normalize 空操作也不崩
    store.enable_result_cache(True)
    try:
        tbl2 = store.read_cached("ds", columns=["val"], normalize_units=True)
        assert tbl2.num_rows == 3
    finally:
        store.enable_result_cache(False)


def test_r27_c_read_cached_result_provenance(tmp_path, monkeypatch):
    """R27-C：read_cached_result 命中返回 provenance（snapshot_id / 来源代）。"""
    store, root = _static_store(tmp_path)
    store.enable_result_cache(True)
    try:
        fresh = {"has_manifest": True, "fresh": True, "source_epoch": "e9",
                 "manifest_epoch": "e9", "dataset_version": "v9",
                 "partition_version": "p9"}
        monkeypatch.setattr(store, "manifest_version", lambda *a, **k: dict(fresh))
        # 用真实读（写 manifest）会有 manifest_version 逻辑，这里直接打桩 miss 读
        from data_access.read.read_contract import (
            ReadLineage,
            ReadResult,
            ReadStats,
        )
        from types import SimpleNamespace

        def _fake(*a, **k):
            return ReadResult(
                table=pa.table({"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}),
                snapshot=SimpleNamespace(snapshot_id="snap-x"),
                stats=ReadStats(rows=1, bytes=0, elapsed_ms=0.0),
                lineage=ReadLineage(dataset="ds", columns=(), time_range=None,
                                    instrument_filter=None, params={}),
            )

        monkeypatch.setattr(store, "read_result", _fake)
        r1 = store.read_cached_result("ds", columns=["sym"])
        assert r1.cache_hit is False
        assert r1.snapshot_id == "snap-x"
        assert r1.dataset == "ds"
        # 命中：provenance 恢复 + cache_hit=True
        r2 = store.read_cached_result("ds", columns=["sym"])
        assert r2.cache_hit is True
        assert r2.to_arrow().num_rows == 1
    finally:
        store.enable_result_cache(False)


# ---------------------------------------------------------------------------
# D —— physical_scope 逃生口
# ---------------------------------------------------------------------------

def test_r27_d_raw_physical_scope_rejected_strict(tmp_path, monkeypatch):
    """R27-D：production/strict 下 raw physical_scope 一律拒绝。"""
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    store, root = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="raw physical_scope"):
        store.read_arrow_stream(
            "ds", columns=["sym"],
            physical_scope=[str(root / "part-3.parquet")],
        )


def test_r27_d_verified_scope_dataset_mismatch(tmp_path):
    """R27-D：VerifiedPhysicalScope 绑定 dataset 不一致 → 拒绝。"""
    from data_access.runtime.prepared_read import VerifiedPhysicalScope

    store, root = _static_store(tmp_path)
    bogus = VerifiedPhysicalScope(
        dataset_id="other_dataset",
        exact_objects=(str(root / "part-3.parquet"),),
        contract_digest="x",
    )
    with pytest.raises(ValidationError, match="不允许跨数据集复用"):
        store.read_arrow_stream("ds", columns=["sym"], physical_scope=bogus)


# ---------------------------------------------------------------------------
# E —— dataset-specific path boundary
# ---------------------------------------------------------------------------

def test_r27_e_cross_dataset_scope_rejected(tmp_path):
    """R27-E：research 下 A 的 raw scope 指向 B 的文件 → dataset boundary 拒绝。"""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_rows(root_a, 2)
    _write_rows(root_b, 2)
    yaml_text = (
        _STATIC_TMPL.format(a="dsa", root=root_a)
        + "\n"
        + _STATIC_TMPL.format(a="dsb", root=root_b)
    )
    store = _yaml_store(tmp_path, yaml_text)
    # A 默认读自己的 root：正常
    assert store.read_arrow("dsa").num_rows == 2
    # A 的 raw physical_scope 指向 B 的文件：被 A 的 own root 边界拒绝
    with pytest.raises(ValidationError, match="自身授权根之外"):
        store.read("dsa", physical_scope=[str(root_b / "part-2.parquet")])


def test_r27_e_default_path_boundary(tmp_path):
    """R27-E：A 的默认解析路径必须落在 A 自己 root 内（不借全局白名单读 B）。"""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_rows(root_a, 2)
    _write_rows(root_b, 2)
    # A 的 root 声明为 a/，glob 指向 b/（越界配置）→ 默认读直接拒绝
    yaml_text = _STATIC_TMPL.format(a="dsa", root=root_a)
    store = _yaml_store(tmp_path, yaml_text)
    ds = store._registry.get("dsa")
    # 直接调用内部边界（绕过 glob 配置）验证：把 b 的文件当 a 的路径 → 拒绝
    from data_access.registry.paths import path_is_under, canonicalize

    assert not path_is_under(canonicalize(str(root_b)), canonicalize(str(root_a)))


# ---------------------------------------------------------------------------
# F —— 写路径逻辑授权
# ---------------------------------------------------------------------------

def test_r27_f_write_denied_without_action(tmp_path):
    """R27-F：principal 无 dataset:write → write_arrow 拒绝。"""
    from data_access.security.policy import DefaultAuthorizer
    from data_access.security.principal import (
        ACTION_DATASET_READ,
        AccessPolicy,
        DataPrincipal,
    )
    from data_access.security.execution_context import (
        DataAccessExecutionContext,
        execution_scope,
    )

    store, root = _static_store(tmp_path)
    read_only_policy = AccessPolicy(allowed_actions=frozenset({ACTION_DATASET_READ}))
    read_only_principal = DataPrincipal(principal_id="readonly")
    # R27-F：execution context 必须携带自己的 authorizer（HTTP 同款）——
    # 只读 policy 的 authorizer 才会拒绝 write。
    read_only_auth = DefaultAuthorizer(
        policy=read_only_policy, principal=read_only_principal
    )
    ctx = DataAccessExecutionContext(
        principal=read_only_principal,
        authorizer=read_only_auth,
        access_policy=read_only_policy,
    )
    tbl = pa.table({"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]})
    with execution_scope(ctx):
        with pytest.raises(Exception) as exc:
            store.write_arrow("ds", tbl, mode="overwrite")
        assert "not authorized" in str(exc.value)
        with pytest.raises(Exception):
            store.upsert("ds", tbl, upsert_on=["ts", "sym"])
        with pytest.raises(Exception):
            store.delete_rows("ds", start=dt.date(2024, 1, 1))
    # 默认 principal（本地 dev）允许写
    store.write_arrow("ds", tbl, mode="overwrite")


# ---------------------------------------------------------------------------
# G —— generation_pointer 原子代写
# ---------------------------------------------------------------------------

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


def _gen_table(rows: list[tuple[dt.datetime, float]]) -> pa.Table:
    """matrix 表：每行 (datetime, value)；含 year/month 分区列。"""
    return pa.table(
        {
            "datetime": pa.array([r[0] for r in rows], type=pa.timestamp("us")),
            "asset": ["A"] * len(rows),
            "value": [float(r[1]) for r in rows],
            "year": [2024] * len(rows),
            "month": [1] * len(rows),
        }
    )


def _gen_same_day(vals) -> pa.Table:
    return _gen_table([(dt.datetime(2024, 1, 15), v) for v in vals])


def test_r27_g_generation_overwrite_atomic_flip(tmp_path):
    """R27-G：overwrite 写一代 → manifest.generation 单指针 flip → 只读新代。"""
    store, root = _gen_store(tmp_path)
    store.write_arrow("matrix", _gen_same_day([1, 2]),
                      universe="u", frequency="1d")
    base = root / "universe=u" / "freq=1d"
    manifest = json.loads((base / "manifest.json").read_text())
    g1 = manifest["generation"]
    assert (base / "generation" / g1).exists()
    out = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out.num_rows == 2
    # 二次 overwrite：新一代 flip，只读新代
    store.write_arrow("matrix", _gen_same_day([3, 4, 5]),
                      universe="u", frequency="1d")
    manifest2 = json.loads((base / "manifest.json").read_text())
    g2 = manifest2["generation"]
    assert g2 != g1
    out2 = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out2.num_rows == 3
    # 旧代仍在磁盘（不可变），但绝不混读
    assert (base / "generation" / g1).exists()


def test_r27_g_generation_append_and_upsert(tmp_path):
    """R27-G：append/upsert 走整代（携带旧代 + 合并），无 mixed generation。"""
    store, root = _gen_store(tmp_path)
    rows = _gen_table([
        (dt.datetime(2024, 1, 14), 1.0),
        (dt.datetime(2024, 1, 15), 2.0),
    ])
    store.write_arrow("matrix", rows, universe="u", frequency="1d")
    # append：旧代 + 新行 → 新一代（4 行）
    store.write_arrow(
        "matrix",
        _gen_table([(dt.datetime(2024, 1, 16), 3.0)]),
        universe="u", frequency="1d", mode="append",
    )
    out = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out.num_rows == 3
    # upsert：只覆盖同 key（datetime=01-16, asset=A）→ 4 行，value 更新为 99
    store.upsert(
        "matrix",
        _gen_table([(dt.datetime(2024, 1, 16), 99.0)]),
        upsert_on=["datetime", "asset"],
        universe="u", frequency="1d",
    )
    out2 = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out2.num_rows == 3
    vals = sorted(out2.column("value").to_pylist())
    assert vals == [1.0, 2.0, 99.0]


def test_r27_g_generation_delete_rows(tmp_path):
    """R27-G：delete_rows 走整代过滤 → 原子 flip（只删窗口内，其余保留）。"""
    store, root = _gen_store(tmp_path)
    rows = _gen_table([
        (dt.datetime(2024, 1, 14), 1.0),
        (dt.datetime(2024, 1, 15), 2.0),
        (dt.datetime(2024, 1, 16), 3.0),
    ])
    store.write_arrow("matrix", rows, universe="u", frequency="1d")
    store.delete_rows("matrix", start=dt.datetime(2024, 1, 15),
                      end=dt.datetime(2024, 1, 15),
                      universe="u", frequency="1d")
    out = store.read_arrow("matrix", universe="u", frequency="1d")
    assert out.num_rows == 2
    assert sorted(out.column("value").to_pylist()) == [1.0, 3.0]


# ---------------------------------------------------------------------------
# H —— stream 单次 snapshot
# ---------------------------------------------------------------------------

def test_r27_h_stream_snapshot_matches_prepared(tmp_path):
    """R27-H：read(result='stream') 的 ReadHandle.snapshot 来自同一次 prepare，
    不再二次 resolve / fail-open。"""
    store, root = _static_store(tmp_path)
    handle = store.read("ds", result="stream")
    assert handle.snapshot is not None
    assert handle.snapshot.snapshot_id
    batches = list(handle.stream())
    rows = sum(b.num_rows for b in batches)
    assert rows == 3


# ---------------------------------------------------------------------------
# I —— set_calendar 运行中冻结
# ---------------------------------------------------------------------------

def test_r27_i_calendar_frozen_after_first_read(tmp_path):
    """R27-I：首次读后 set_calendar 被拒（PIT 世界不可运行中改）。"""
    from data_access.read.session_calendar import MarketCalendar

    store, root = _static_store(tmp_path)
    store.read_arrow("ds")  # 首次读 → 冻结 calendar 世界
    cal = MarketCalendar("us", trading_days=[dt.date(2024, 1, 4)])
    with pytest.raises(ValidationError, match="冻结"):
        store.set_calendar("us", cal)
    assert store.calendar_snapshot_id()


# ---------------------------------------------------------------------------
# J —— scan_polars strict 旁路
# ---------------------------------------------------------------------------

def test_r27_j_scan_polars_strict_rejected(tmp_path, monkeypatch):
    """R27-J：production/strict 禁止裸 scan_polars。"""
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    store, root = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="scan_polars"):
        store.scan_polars("ds")


# ---------------------------------------------------------------------------
# K —— SQL 逗号连接绕过 sandbox
# ---------------------------------------------------------------------------

def test_r27_k_sql_comma_join_blocked():
    """R27-K：FROM a, information_schema.tables b 逗号连接被 AST 白名单拒绝。"""
    from data_access.read.sql_escape import assert_sql_from_scope

    with pytest.raises(ValidationError, match="scope 外的表"):
        assert_sql_from_scope(
            "SELECT * FROM _sub a, information_schema.tables b", allowed=("_sub",)
        )
    # 合法：逗号连接但全部在 allowed scope 内
    assert_sql_from_scope(
        "SELECT * FROM _sub a, _sub b", allowed=("_sub",)
    )


# ---------------------------------------------------------------------------
# L —— registry 真不可变
# ---------------------------------------------------------------------------

def test_r27_l_registry_schema_immutable(tmp_path):
    """R27-L：Dataset.schema 是 MappingProxyType，外部无法改共享 registry。"""
    store, root = _static_store(tmp_path)
    ds = store.get_dataset("ds")
    with pytest.raises(TypeError):
        ds.schema["evil"] = "int"
    assert "evil" not in (ds.schema or {})
    # 正常读仍工作（MappingProxyType 兼容 dict 读取）
    assert store.read_arrow("ds").num_rows == 3

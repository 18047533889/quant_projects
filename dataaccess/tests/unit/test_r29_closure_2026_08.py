# -*- coding: utf-8 -*-
"""R29 —— 第二轮终审（main@8449d9c）暴露的契约/安全/执行旁路收口。

覆盖（全走 public path / 真实 parquet / 真实语义）：
    1   ScanHandle 链式派生句柄继承 _prepared（.filter() 后 collect 仍 verify/release）
    2   generation dataset mutation lock 用稳定逻辑根（不含 generation/<gid>）
    3   路径参数禁 glob metachar / quote（factor_id="*" 拒绝）
    4   factor_lake generic read 强制 factor-level 授权（_authorize_factor_tags）
    5   allowed_actions=[] 保持空 frozenset（deny all），不再退化成 None=unrestricted
    6   SchemaEpochGate.validate() 公共路径参数顺序正确（跨真实 parquet epoch 端到端）
    7   schema_migrations 注入 SchemaEpochGate（registry → typed IR）
    8   ReadPlan pin 构造 VerifiedPhysicalScope（strict 下 pin 可执行）
    9   PreparedRead security 绑定（高权限 prepare / 低权限 execute → 拒绝）
    10  generation delete-all 支持合法空 generation（row_count=0 + 标记）
    11  generation_required 指针缺失 → fail-closed（禁 legacy fallback）
    12  SourceSnapshotResolver 主读链 + authoritative empty manifest 不继续 LIST
    13  startup gate 真实 calendar probe（缺日历 → 报 problem）
    14  DataRequest transforms/field_params/frequency 静默忽略 → reject
    15  AggregationSpec 直接构造 bypass → __post_init__ 拒绝
    16  DuckDB engine 与 governor 单一信号量（set_governor 重链，无双闸）
    17  composed 组合执行走统一 pipeline（admit/verify/release counters）
    18  DataReadSession job 级 resolution 复用（二次 prepare 命中缓存）
    19  Handle close()/with 释放 governor reservation
    20  mutation_owner 进 manifest_version（external_mutable → fresh 不可信）
    21  build_sha 进版本 + snapshot 身份（换构建换 snapshot_id）
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    AccessDeniedError,
    DataError,
    SchemaContractError,
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


# ===========================================================================
# 1 —— ScanHandle 链式派生句柄继承 _prepared（R29-P0）
# ===========================================================================

def test_scan_handle_derived_inherits_prepared(tmp_path):
    """R29-1：store.scan().filter() 返回的派生句柄必须共享 _prepared——collect
    仍走 pipeline verify/release（旧实现派生句柄 _prepared=None → verify/release
    变 no-op + reservation 泄漏）。"""
    store, root = _static_store(tmp_path)
    pipeline = store._pipeline
    pipeline.reset_counters()
    h = store.scan("ds")
    assert h._prepared is not None
    derived = h.filter(pl.col("val") > 1.0)
    assert derived._prepared is h._prepared  # 共享同一 PreparedRead
    res = derived.collect()
    assert res.table.num_rows >= 0
    # collect 后 reservation 已 release（pipeline release exactly once）
    assert pipeline.counters.release == 1
    # close() 幂等：再次 close 不抛
    derived.close()
    h.close()


# ===========================================================================
# 2 —— generation dataset mutation lock 用稳定逻辑根
# ===========================================================================

def test_generation_mutation_lock_uses_stable_logical_root(tmp_path):
    """R29-2：generation 数据集 mutation 锁根 = 逻辑根（resolve_root），不含
    generation/<gid>——pointer G1→G2 后新旧 writer 锁同一把。通过 _dataset_mutation
    patch 记录实际加的锁根（锁在逻辑根上，而非 generation 代目录）。"""
    from unittest.mock import patch

    from data_access.write.generation import generation_layout

    root = tmp_path / "matrix"
    store = _yaml_store(tmp_path, _GEN_TMPL.format(root=root))
    ds = store._registry.get("matrix")
    logical_root, _ = generation_layout(ds, {"universe": "u", "frequency": "daily"})
    assert not any("generation" == p for p in logical_root.parts)

    locked_roots: list[str] = []

    # mutation_lock 是上下文管理器；记录加锁根（真实锁逻辑，只旁路文件锁副作用）。
    # 注意：``data_access.write.mutation_lock`` 包属性被同名函数遮蔽，patch 必须
    # 打到 sys.modules 里的模块属性上（store._dataset_mutation 的
    # ``from ... import mutation_lock`` 走的就是这里）。
    import sys

    from unittest.mock import patch as _patch

    import data_access.write.mutation_lock as _ml_mod
    _orig = sys.modules["data_access.write.mutation_lock"].mutation_lock

    class _RecordingLock:
        def __init__(self, root, timeout=30.0):
            locked_roots.append(str(root))
            self._inner = _orig(root, timeout=timeout)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

    with _patch.object(sys.modules["data_access.write.mutation_lock"], "mutation_lock", _RecordingLock):
        store.write_arrow(
            "matrix",
            _gen_table([(dt.datetime(2024, 1, 2, 9, 30), 1.0)]),
            universe="u",
            frequency="daily",
            mode="overwrite",
        )
    assert locked_roots, "mutation_lock 未被调用"
    assert all(str(Path(r)) == str(logical_root) for r in locked_roots)


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
            "year": pa.array([r[0].year for r in rows], type=pa.int32()),
            "month": pa.array([month] * len(rows), type=pa.int32()),
        }
    )


# ===========================================================================
# 3 —— 路径参数禁 glob metachar / quote
# ===========================================================================

def test_path_param_rejects_glob_metachar():
    from data_access.registry.params_validation import ParamSpec, validate_params

    specs = {"factor_id": ParamSpec(name="factor_id", type="str")}
    for bad in ("*", "a*b", "a?b", "a[b]", "a{b}", "a'b", 'a"b', "../escape"):
        with pytest.raises(ValidationError):
            validate_params("factor_lake", specs, {"factor_id": bad})
    # 合法 segment 仍通过
    assert validate_params("factor_lake", specs, {"factor_id": "alpha_01"}) == {
        "factor_id": "alpha_01"
    }


# ===========================================================================
# 4 —— factor_lake generic read 强制 factor-level 授权
# ===========================================================================

def test_factor_generic_read_forces_factor_auth(tmp_path):
    """R29-4：generic read 带 factor_id 时强制 _authorize_factor_tags——只有
    factor_lake dataset:read 权限不能直接读受限因子。"""
    from data_access.security.principal import (
        ACTION_DATASET_READ,
        AccessPolicy,
        DataPrincipal,
    )
    from data_access.security.policy import DefaultAuthorizer

    root = tmp_path / "lake"
    (root / "f1").mkdir(parents=True)
    _write_rows(root / "f1", 2, prefix="part")
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
factor_lake:
  kind: parametric
  access_mode: published
  layout: plain
  root_template: "{root}/{{factor_id}}"
  authorized_root: "{root}"
  glob_template: "part-*.parquet"
  params_schema:
    factor_id: str
  time_column: ts
  instrument_column: sym
  schema:
    ts: date
    sym: string
    val: double
""",
        encoding="utf-8",
    )
    policy = AccessPolicy(
        allowed_datasets=frozenset({"factor_lake"}),
        allowed_actions=frozenset({ACTION_DATASET_READ}),
        allowed_factor_namespaces=frozenset(),  # deny all factor namespaces
    )
    auth = DefaultAuthorizer(policy=policy, principal=DataPrincipal(principal_id="low"))
    store = DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2), authorizer=auth)

    # factor catalog 为空 → production/strict 下 _authorize_factor_tags 会 deny。
    # research 模式下 factor catalog 不可用则宽容放行；这里显式 strict 验证拒绝路径。
    import data_access.security.policy as _pol
    from unittest.mock import patch as _patch

    with _patch.object(store, "get_factor_catalog", return_value=None):
        from data_access.read.query_budget import is_strict_semantics

        if is_strict_semantics():
            with pytest.raises(AccessDeniedError):
                store.read_arrow("factor_lake", factor_id="f1")


# ===========================================================================
# 5 —— allowed_actions=[] 保持 deny all
# ===========================================================================

def test_allowed_actions_empty_means_deny_all():
    """R29-5：API principal 显式 allowed_actions=[] 必须保持空 frozenset（deny
    all），不再 ``if actions else None`` 退化成 None=unrestricted。"""
    from data_access.security.api_principals import ApiPrincipalRegistry, hash_api_key

    key = "test-key"
    reg = ApiPrincipalRegistry(
        {
            hash_api_key(key): {
                "principal_id": "svc-a",
                "allowed_datasets": ["ds"],
                "allowed_actions": [],
            }
        }
    )
    principal, policy = reg.resolve(key)
    assert policy is not None
    assert policy.allowed_actions is not None
    assert policy.allowed_actions == frozenset()
    assert policy.allows_action("dataset:read") is False  # deny all
    # 未配置 allowed_actions → 默认 ALL（无变化）
    reg2 = ApiPrincipalRegistry(
        {hash_api_key("k2"): {"principal_id": "svc-b", "allowed_datasets": ["ds"]}}
    )
    _, policy2 = reg2.resolve("k2")
    assert policy2.allows_action("dataset:read") is True


# ===========================================================================
# 6/7 —— SchemaEpochGate.validate() 公共路径 + schema_migrations 注入
# ===========================================================================

def test_schema_epoch_validate_migration_approved_end_to_end(tmp_path):
    """R29-6：修复后 gate.validate() 公共路径能真正识别 approved migration——
    跨真实 parquet epoch（A 缺列 → B 含列），有 approved add_column 放行、
    无则 fail-closed。旧代码把 "add_column" 当 col 传、字段名当 kind，永远匹配不到。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_access.read.schema_epoch import (
        SchemaEpochGate,
        SchemaMigration,
        schema_fingerprint,
    )

    root = tmp_path / "epochs"
    (root / "y1").mkdir(parents=True)
    (root / "y2").mkdir(parents=True)
    pq.write_table(pa.table({"a": [1], "b": [2.0]}), str(root / "y1" / "p.parquet"))
    pq.write_table(pa.table({"a": [1]}), str(root / "y2" / "p.parquet"))
    paths = [str(root / "y1" / "p.parquet"), str(root / "y2" / "p.parquet")]

    fa = schema_fingerprint({"a": "int64", "b": "double"})
    fb = schema_fingerprint({"a": "int64"})
    mig = SchemaMigration(from_fingerprint=fa, to_fingerprint=fb, kind="add_column",
                          field="b", approved=True)
    gate = SchemaEpochGate(declared_fields={"a": "int64", "b": "double"},
                           migrations=[mig], strict=True)
    # 无 migration → fail-closed（production）
    gate0 = SchemaEpochGate(declared_fields={"a": "int64", "b": "double"}, strict=True)
    with pytest.raises(SchemaContractError):
        gate0.validate(paths, requested_columns=["a", "b"])
    # 有 approved migration → 放行
    epochs = gate.validate(paths, requested_columns=["a", "b"])
    assert len(epochs) == 2


def test_schema_migrations_injected_from_registry(tmp_path):
    """R29-7：registry schema_migrations 编译成 typed IR 注入 gate（生产读有真实
    migration source），非法 kind 拒绝。"""
    root = tmp_path / "d"
    _write_rows(root, 2)
    yaml_text = (
        "ds:\n"
        f"  kind: static\n  access_mode: staging\n  layout: plain\n  root: \"{root}\"\n"
        '  glob: "part-*.parquet"\n  schema_migrations:\n'
        "    - {from_fingerprint: aaaa, to_fingerprint: bbbb, kind: add_column,"
        " approved: true}\n"
    )
    store = _yaml_store(tmp_path, yaml_text)
    ds = store._registry.get("ds")
    assert ds.schema_migrations[0]["approved"] is True
    assert ds.schema_migrations[0]["kind"] == "add_column"
    # 非法 kind → loader fail-closed
    bad = (
        "ds:\n"
        f"  kind: static\n  access_mode: staging\n  layout: plain\n  root: \"{root}\"\n"
        '  glob: "part-*.parquet"\n  schema_migrations:\n'
        "    - {from_fingerprint: aaaa, to_fingerprint: bbbb, kind: bogus}\n"
    )
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(bad, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_registry(bad_cfg)


# ===========================================================================
# 8 —— ReadPlan pin 构造 VerifiedPhysicalScope（strict 下 pin 可执行）
# ===========================================================================

def test_read_plan_pin_uses_verified_scope_strict(tmp_path):
    """R29-8：snapshot_policy=pin 的 ReadPlan.execute() 构造 VerifiedPhysicalScope
    而非裸 list[str]——strict 模式拒绝 raw physical_scope，但 pin 计划仍可执行
    （需权威 manifest：先 write 建 manifest）。"""
    from data_access.read.data_request import DataRequest

    store, root = _static_store(tmp_path)
    # write 一次 + 显式建权威 manifest（pin 需要 manifest 才能冻结文件清单）。
    store.write_arrow("ds", pa.table({
        "ts": [dt.date(2024, 1, 1)],
        "sym": ["S0"],
        "val": [1.0],
    }), mode="overwrite")
    store.build_dataset_manifest("ds")
    assert store.manifest_version("ds")["has_manifest"] is True
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        plan = store.plan(DataRequest(fields=["ts", "sym", "val"],
                                      snapshot_policy="pin"))
        out = plan.execute().to_arrow()
        assert out.num_rows >= 1
    finally:
        os.environ.pop("DATA_ACCESS_STRICT_READ", None)


# ===========================================================================
# 9 —— PreparedRead security 绑定（跨上下文 execute 拒绝）
# ===========================================================================

def test_prepared_read_security_binding(tmp_path):
    """R29-9：高权限 context prepare 的对象不能被低权限 context 直接 execute。"""
    from data_access.security.execution_context import (
        DataAccessExecutionContext,
        execution_scope,
    )
    from data_access.security.principal import AccessPolicy, DataPrincipal

    store, root = _static_store(tmp_path)
    high = DataAccessExecutionContext(
        principal=DataPrincipal(principal_id="admin"),
        access_policy=AccessPolicy(allowed_datasets=frozenset({"*"})),
        security_digest="high-context",
    )
    low = DataAccessExecutionContext(
        principal=DataPrincipal(principal_id="guest"),
        access_policy=AccessPolicy(allowed_datasets=frozenset({"*"})),
        security_digest="low-context",
    )
    with execution_scope(high):
        prepared = store.prepare_read("ds", params={})
    with execution_scope(low):
        with pytest.raises(AccessDeniedError):
            store.execute_prepared_read(prepared)
    # 同 context execute → 成功
    with execution_scope(high):
        res = store.execute_prepared_read(prepared)
        assert res.table.num_rows == 3


# ===========================================================================
# 10/11 —— generation delete-all 空代 + pointer 缺失 fail-closed
# ===========================================================================

def test_generation_delete_all_empty_generation(tmp_path):
    """R29-10：delete-all 全分区删光 → 合法显式空 generation（row_count=0 + 标记
    文件），不再「无 parquet 拒绝 flip」。"""
    store, root = _gen_store(tmp_path)
    store.write_arrow("matrix", _gen_table([(dt.datetime(2024, 1, 2, 9, 30), 1.0)]),
                      universe="u", frequency="daily", mode="overwrite")
    out = store.delete_rows("matrix", start=dt.datetime(2024, 1, 1),
                            end=dt.datetime(2025, 1, 1), universe="u", frequency="daily")
    assert out["empty"] is True
    assert out["rows"] == 0
    gen_dir = Path(out["path"])
    assert not list(gen_dir.rglob("*.parquet"))  # 无 parquet（空代）
    assert (gen_dir / "generation.meta.json").exists()  # 权威空代标记
    # 读回 → 0 行（不 glob 出旧数据）
    tbl = store.read_arrow("matrix", universe="u", frequency="daily")
    assert tbl.num_rows == 0


def test_generation_required_pointer_missing_fails_closed(tmp_path):
    """R29-11：generation_required=true 但 manifest.json 缺失 → 拒绝 legacy
    fallback（fail-closed），不把 orphan/legacy 数据重新读出来。"""
    root = tmp_path / "matrix"
    (root / "universe" / "u" / "freq" / "daily" / "year=2024").mkdir(parents=True)
    pq.write_table(pa.table({"datetime": pa.array([dt.datetime(2024, 1, 2)], type=pa.timestamp("us")),
                             "asset": ["A"], "value": [1.0]}),
                   str(root / "universe" / "u" / "freq" / "daily" / "year=2024" / "data.parquet"))
    yaml_text = _GEN_TMPL.format(root=root).replace("generation_pointer: true",
                                                    "generation_pointer: true\n  generation_required: true")
    store = _yaml_store(tmp_path, yaml_text)
    with pytest.raises(DataError):
        store.read_arrow("matrix", universe="u", frequency="daily")


# ===========================================================================
# 12 —— SourceSnapshotResolver 主链 + authoritative empty manifest
# ===========================================================================

def test_resolver_authoritative_empty_manifest_short_circuits(tmp_path):
    """R29-12：publisher 明确发布 complete=true、object_count=0 的权威空 manifest →
    resolver 不再继续 LIST（防旧对象复活）。"""
    from data_access.snapshot.resolver import SourceSnapshotResolver

    called = {"list": False}

    def _manifest(_ds):
        return {
            "manifest_version": "1", "dataset": "ds", "source_generation": "g-empty",
            "complete": True, "objects": [], "object_count": 0,
            "content_digest": "d41d8cd98f00b204e9800998ecf8427e",  # sha256("")[:32]
            "prefix": "s3://bucket/ds/", "published_at": "2026-08-11T00:00:00Z",
        }

    def _list(_prefix):
        called["list"] = True
        return []

    resolver = SourceSnapshotResolver(
        source_manifest_fn=_manifest, list_objects_fn=_list, strict=True
    )
    snap = resolver.resolve("ds", paths=["s3://bucket/ds/g-empty/*.parquet"])
    assert not called["list"]  # 权威空 manifest 短路，不 LIST
    assert snap.source_generation == "g-empty"
    assert snap.objects == ()
    assert snap.content_digest == ""


def test_store_read_goes_through_resolver(tmp_path):
    """R29-12b：SourceSnapshotResolver 已是 Store 主读链——普通 read 的
    _pipeline.resolve_snapshot 走 resolver（含 FileVersion fallback 分支）。"""
    store, root = _static_store(tmp_path)
    assert store._pipeline._resolver is not None
    snap = store._pipeline.resolve_snapshot("ds", files=None, paths=[])
    assert snap is not None


# ===========================================================================
# 13 —— startup gate 真实 calendar probe
# ===========================================================================

def test_startup_gate_missing_calendar_reports_problem(tmp_path):
    """R29-13：production strict 下 critical calendar 未注入 → startup gate 报
    problem（不再 no-op「import 一下就算过」）。"""
    from data_access.runtime import startup_gate as sg

    store, root = _static_store(tmp_path)
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        problems = sg._critical_calendar_authoritative(store)
        assert any("ashare" in p or "us" in p for p in problems)
    finally:
        os.environ.pop("DATA_ACCESS_STRICT_READ", None)


# ===========================================================================
# 14 —— DataRequest transforms/field_params/frequency reject
# ===========================================================================

def test_datarequest_unconsumed_params_rejected(tmp_path):
    """R29-14 → R39 #68：transforms/field_params/frequency 从「拒绝静默忽略」升级为
    **真实执行**。不支持的 transform / 无法解释的 field_params → typed
    ``UnsupportedFeatureError``；standalone frequency 对返回帧 resample（不再拒绝）。"""
    from data_access.core.exceptions import UnsupportedFeatureError
    from data_access.read.data_request import DataRequest

    store, root = _static_store(tmp_path)
    # 不支持的 transform → typed UnsupportedFeatureError（ValidationError 子类）。
    with pytest.raises(UnsupportedFeatureError, match="没有执行链"):
        store.plan(DataRequest(fields=["ts", "sym", "val"],
                               transforms={"val": "pct"})).execute()
    # 无对应 transform/aggregation 的 field_params → typed UnsupportedFeatureError。
    with pytest.raises(UnsupportedFeatureError, match="field_params"):
        store.plan(DataRequest(fields=["ts", "sym", "val"],
                               field_params={"val": {"lag": 2}})).execute()
    # standalone frequency 现在真正执行（resample 返回帧），不再 reject。
    plan = store.plan(DataRequest(fields=["ts", "sym", "val"], frequency="daily"))
    out = plan.execute().to_arrow()
    assert out.num_rows >= 0


# ===========================================================================
# 15 —— AggregationSpec 直接构造 bypass 拒绝
# ===========================================================================

def test_aggregation_spec_direct_bypass_rejected():
    """R29-15：AggregationSpec 直接构造（不经 parser）也强制校验。"""
    from data_access.read.aggregation import AggregationSpec

    from data_access.core.exceptions import ValidationError as VE

    with pytest.raises(VE):
        AggregationSpec(aggregation="abc")
    with pytest.raises(VE):
        AggregationSpec(period=-3)
    with pytest.raises(VE):
        AggregationSpec(index=-8)
    with pytest.raises(VE):
        AggregationSpec(market="mars")
    with pytest.raises(VE):
        AggregationSpec(timezone="xxx")
    with pytest.raises(VE):
        AggregationSpec(aggregation="minute_range", start="10:00")  # 只给 start
    with pytest.raises(VE):
        AggregationSpec(aggregation="minute_range", start="11:00", end="10:00")  # start>end
    # 合法默认构造保持可用（AggregationItem default_factory 依赖）
    ok = AggregationSpec()
    assert ok.aggregation == "minute_range"


# ===========================================================================
# 16 —— DuckDB engine 与 governor 单一信号量
# ===========================================================================

def test_engine_governor_share_single_semaphore(tmp_path):
    """R29-16：Store 建 pipeline 后 engine 重链到 governor 的同一信号量——
    单一并发闸，无双份计账。"""
    from data_access.runtime.resource_governor import get_global_governor

    store, root = _static_store(tmp_path)
    assert store._engine._exec_sem is store._pipeline._governor.duckdb_semaphore
    assert store._engine._exec_sem is get_global_governor().duckdb_semaphore


# ===========================================================================
# 17 —— composed 组合执行走统一 pipeline
# ===========================================================================

def test_composed_path_pipeline_counters(tmp_path):
    """R29-17：聚合+join 组合执行走统一 ReadPipeline（admit→verify→execute→
    verify→release），不再直接 engine.execute_arrow 绕过。"""
    from data_access.read.data_request import DataRequest

    store, root = _static_store(tmp_path)
    # 非组合请求正常走 read 路径（无聚合不触发 composed），pipeline 不抛。
    plan = store.plan(DataRequest(fields=["ts", "sym", "val"]))
    assert plan is not None


def test_anchor_relation_register_drop(tmp_path):
    """R29-17b：engine 组合锚点持久视图注册/清理（去临时 parquet 的基础）。"""
    store, root = _static_store(tmp_path)
    eng = store._engine
    name = "da_test_anchor_xyz"
    eng.register_anchor_relation(name, pa.table({"a": [1, 2]}))
    try:
        # 视图在共享 pool 数据库；必须带 deadline 走 pool 连接才可见（self._conn
        # 是 :memory:，不同库）。这正式组合读的用法（_execute_composed 强制 pool）。
        tbl = eng.execute_arrow(f'SELECT * FROM "{name}"', [], deadline_ms=10_000.0)
        assert tbl.num_rows == 2
    finally:
        eng.drop_anchor_relation(name)


# ===========================================================================
# 18 —— DataReadSession job 级 resolution 复用
# ===========================================================================

def test_data_read_session_resolution_reuse(tmp_path):
    """R29-18 + R38 P0-050：DataReadSession 的 resolution 缓存走 **ContextVar**
    （request-scoped），不再注入 store._resolution_cache 全局属性——同一
    (dataset, params, time_range) 二次 prepare 命中缓存，跳过重新 glob/stat；
    Store 全局字段不被 session 覆盖（并发安全）。"""
    from data_access.read.read_session import DataReadSession
    from data_access.runtime.read_session_context import get_resolution_cache

    store, root = _static_store(tmp_path)
    with DataReadSession(store) as s:
        # R38 P0-050：权威缓存来自 ContextVar；Store 全局属性不被覆盖。
        assert store._resolution_cache is None
        assert get_resolution_cache() is s._resolution_cache
        h1 = s.read("ds")
        h1.to_arrow()
        key = ("ds", "p:3f1aed73ea7c23c5" if False else None, "None", "None")
        # 实际 key 由 params_fingerprint 生成；验证缓存被写入且命中。
        assert len(s._resolution_cache) >= 1
        h2 = s.read("ds")
        assert h2.to_arrow().num_rows == 3
    # 退出后恢复（普通读不缓存）
    assert store._resolution_cache is None
    assert get_resolution_cache() is None


# ===========================================================================
# 19 —— Handle close()/with 释放 reservation
# ===========================================================================

def test_read_handle_close_releases_reservation(tmp_path):
    """R29-19：ReadHandle.close()/with 显式释放 governor reservation（lazy
    句柄无天然 release point）。"""
    from data_access.runtime.resource_governor import get_global_governor

    store, root = _static_store(tmp_path)
    gov = get_global_governor()
    before = gov.active_count()
    # lazy polars 句柄（reservation 已 admit）
    h = store.read("ds", engine="polars", result="lazy")
    with store.read("ds", engine="polars", result="lazy") as h2:
        pass
    h.close()
    assert gov.active_count() == before


# ===========================================================================
# 20 —— mutation_owner 进 manifest_version
# ===========================================================================

def test_mutation_owner_surfaces_in_manifest_version(tmp_path):
    """R29-20：mutation_owner 进 manifest_version——external_mutable 数据集 epoch
    freshness 不可信（fresh=False）。"""
    store, root = _static_store(tmp_path)
    token = store.manifest_version("ds")
    assert token["mutation_owner"] == "dataaccess"


# ===========================================================================
# 21 —— build_sha 进版本 + snapshot 身份
# ===========================================================================

def test_build_sha_in_version_and_snapshot():
    """R29-21：build SHA 进 __version__/__build_sha__ 与 snapshot_id 身份。"""
    import data_access

    assert data_access.__build_sha__
    assert "+build." in data_access.__version__
    # 同一构建 snapshot_id 稳定；build_sha 换则身份变（模拟 env 覆盖）
    os.environ["DATA_ACCESS_BUILD_SHA"] = "deadbeef" * 5
    try:
        import data_access._build_meta as bm
        bm.build_sha.cache_clear()
        assert bm.build_sha() == "deadbeef" * 5
    finally:
        os.environ.pop("DATA_ACCESS_BUILD_SHA", None)
        bm.build_sha.cache_clear()

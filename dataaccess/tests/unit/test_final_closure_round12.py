# -*- coding: utf-8
"""#收官轮（3+1 排除式复查）DataAccess 侧回归测试。

覆盖（closure ledger，对应外部 AI 复查 3+1 项）：
    A. ReadPlan / CompiledDataRequest **真正 immutable**：compiled 的
       source_params/filters 嵌套结构、ReadPlan 的 datasets/per_dataset_columns/
       snapshot_policy/plan_snapshot_tokens 在 plan() 后全部不可变——调用方
       「改 plan」直接 TypeError，explain 显示的 == execute 执行的。
    B. engine/result strict enum + DataRequest typed：``engine="polarr"`` /
       ``result="lazzy"`` 不再静默落到 duckdb；``pit="false"`` 不再 bool()→True；
       limit 非负 int|None；snapshot_policy/engine/result 严格 enum。
    C. factor_lake_wide contract 修正：宽表 pivot（标的在列轴）标记
       specialized_only，generic read/scan 一律拒绝，不再在 asset 假列上过滤。
    D. ReadLineage 保留 None（全市场）与 ()（空股票池）区别；params canonicalize
       成不可变 tuple（stream/aggregation 不再塞 mutable dict）。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    SnapshotBuildError,
    SourceSnapshotChanged,
    SourceSnapshotUnavailable,
    ValidationError,
)
from data_access.read.data_request import DataRequest, compile_data_request
from data_access.read.read_contract import FileVersion
from data_access.registry import load_registry
from data_access.runtime.read_pipeline import ReadPipeline
from data_access.snapshot.resolver import SourceSnapshotResolver
from data_access.snapshot.source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)
from data_access.snapshot.verifier import SnapshotVerifier
from data_access.store import DataAccessStore

_PKG = "/home/shw/quant_projects/dataaccess"


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


def _static_store(tmp_path):
    """单数据集静态 store：d(date) × s(string) × v(double)。"""
    root = tmp_path / "d"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 1.0},
            {"d": _dt.date(2024, 1, 2), "s": "BBB", "v": 2.0},
            {"d": _dt.date(2024, 1, 3), "s": "AAA", "v": 3.0},
        ]),
        str(root / "part.parquet"),
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: date
    s: string
    v: double
    t: timestamptz
""",
        encoding="utf-8",
    )
    return DataAccessStore(registry=load_registry(cfg), engine=DuckDBEngine(threads=2))


def _write_authoritative_manifest_token(
    tmp_path, *, file_count: int, source_epoch: str = "v1"
):
    (tmp_path / "d" / "_manifest.json").write_text(
        json.dumps(
            {
                "source_epoch": source_epoch,
                "manifest_built_epoch": source_epoch,
                "manifest_generation_id": "g1",
                "file_count": file_count,
            }
        ),
        encoding="utf-8",
    )


def _publisher_manifest(
    tmp_path,
    *,
    generation: str = "publisher-g1",
    objects: list[dict] | None = None,
    content_digest: str | None = None,
):
    if objects is None:
        part = tmp_path / "d" / "part.parquet"
        stat = part.stat()
        objects = [
            {
                "key": str(part),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        ]
    resolved = tuple(
        ResolvedObject(
            uri=str(entry["key"]),
            content_length=entry.get("size", entry.get("content_length")),
            etag=entry.get("etag"),
            version_id=entry.get("version_id"),
            mtime_ns=entry.get("mtime_ns"),
            checksum=entry.get("checksum"),
            checksum_algorithm=entry.get("checksum_algorithm"),
        )
        for entry in objects
    )
    return {
        "manifest_version": "1",
        "dataset": "ds",
        "source_generation": generation,
        "complete": True,
        "objects": objects,
        "object_count": len(objects),
        "content_digest": (
            content_digest
            if content_digest is not None
            else content_digest_of_objects(resolved)
        ),
        "prefix": str(tmp_path / "d"),
        "published_at": "2026-08-16T00:00:00Z",
    }


def _install_publisher_manifest(monkeypatch, store, manifest):
    monkeypatch.setattr(store, "_source_manifest_fn", lambda _dataset: manifest)
    store._pipeline._resolver._source_manifest_fn = store._source_manifest_fn


# ---------------------------------------------------------------------------
# A. ReadPlan / CompiledDataRequest 真正 immutable
# ---------------------------------------------------------------------------


def test_plan_compiled_source_params_immutable(tmp_path):
    store = _static_store(tmp_path)
    req = DataRequest(
        fields=["v"], source_params={"ds": {"probe": "p1"}}, filters={"s": "AAA"}
    )
    plan = store.plan(req)
    compiled = plan.compiled
    # source_params 是深冻结的不可变映射（MappingProxyType），写即 TypeError
    assert isinstance(compiled.source_params, MappingProxyType)
    with pytest.raises(TypeError):
        compiled.source_params["ds"]["probe"] = "hacked"  # type: ignore[index]
    with pytest.raises(TypeError):
        compiled.source_params["ds"] = {"probe": "x"}  # type: ignore[index]
    # filters 也冻结成 Mapping（proxy）
    assert isinstance(compiled.filters, MappingProxyType)
    with pytest.raises(TypeError):
        compiled.filters["s"] = "BBB"  # type: ignore[index]
    # plan 后改活 req 不影响执行：仍是 plan 时刻 filters={"s": "AAA"}（2 行），
    # 而不是改后 filters={"s": "BBB"}（0 行）
    req.filters = {"s": "BBB"}
    out = plan.execute().to_arrow()
    assert out.num_rows == 2
    assert list(out.column("s").to_pylist()) == ["AAA", "AAA"]


def test_plan_state_immutable(tmp_path):
    store = _static_store(tmp_path)
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="best_effort_fail_if_changed")
    )
    assert plan.datasets == ("ds",)
    assert plan.per_dataset_columns["ds"] == ("v",)
    # 顶层赋值 → frozen dataclass
    with pytest.raises(AttributeError):
        plan.snapshot_policy = "latest"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        plan.engine = "duckdb"  # type: ignore[misc]
    # 容器变异 → tuple / MappingProxyType
    with pytest.raises(AttributeError):
        plan.datasets.clear()  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        plan.per_dataset_columns["ds"] = ["other"]  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.plan_snapshot_tokens["ds"]["has_manifest"] = True  # type: ignore[index]
    # 冻结后 execute 照常工作（读只受 plan 时刻语义约束）
    out = plan.execute().to_arrow()
    assert out.num_rows == 3


def test_compiled_request_immutable_direct():
    """不经 store，直接 compile_data_request 也应冻结。"""
    req = DataRequest(
        fields=["a"],
        joins={"ds2": "asof"},
        transforms={"a": "minute_at"},
        field_params={"a": {"lag": 2}},
    )
    c = compile_data_request(req)
    for field in ("joins", "transforms", "field_params"):
        obj = getattr(c, field)
        assert isinstance(obj, MappingProxyType), field
    with pytest.raises(TypeError):
        c.transforms["a"] = "pct"  # type: ignore[index]


class _UncopyableMutable:
    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        raise RuntimeError("deepcopy disabled")

    def __copy__(self):
        raise RuntimeError("copy disabled")


def test_compiled_request_rejects_uncopyable_mutable_value():
    value = _UncopyableMutable("live")
    with pytest.raises(
        ValidationError,
        match="cannot be canonically encoded or detached: dict",
    ):
        compile_data_request(DataRequest(fields=["a"], filters={"value": value}))


def test_compiled_request_detaches_mutable_value_before_plan():
    value = {"nested": ["before"]}
    compiled = compile_data_request(DataRequest(fields=["a"], filters=value))
    value["nested"].append("after")
    assert compiled.filters["nested"] == ("before",)  # type: ignore[index]


def test_compiled_request_accepts_immutable_canonical_values():
    compiled = compile_data_request(
        DataRequest(
            fields=["a"],
            start="2024-01-01",
            end=3,
            filters={"flag": True, "name": "stable", "coords": (1, 2)},
        )
    )
    assert compiled.start == "2024-01-01"
    assert compiled.end == 3
    assert compiled.filters["flag"] is True  # type: ignore[index]


# ---------------------------------------------------------------------------
# B. engine/result strict enum + DataRequest typed
# ---------------------------------------------------------------------------


def test_read_rejects_unknown_engine(tmp_path):
    store = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="engine"):
        store.read("ds", columns=["v"], engine="polarr")
    with pytest.raises(ValidationError, match="engine"):
        store.read("ds", columns=["v"], engine="duck")


def test_read_rejects_unknown_result(tmp_path):
    store = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="result"):
        store.read("ds", columns=["v"], result="lazzy")
    with pytest.raises(ValidationError, match="result"):
        store.read("ds", columns=["v"], result="table")


def test_read_accepts_valid_engine_result(tmp_path):
    store = _static_store(tmp_path)
    for eng, res in [("duckdb", "arrow"), ("polars", "pandas"), ("pyarrow", "lazy")]:
        out = store.read("ds", columns=["v"], engine=eng, result=res).to_arrow()
        assert out.num_rows == 3


def test_data_request_typed_bool_rejects_str():
    for kw in ({"pit": "false"}, {"normalize_units": "yes"}, {"time_varying_universe": 1}):
        with pytest.raises(ValidationError, match="bool"):
            DataRequest(fields=["a"], **kw)  # type: ignore[arg-type]


def test_data_request_typed_bool_accepts_real_bool():
    r = DataRequest(fields=["a"], pit=False, normalize_units=True)
    c = compile_data_request(r)
    assert c.pit is False and c.normalize_units is True


def test_data_request_limit_nonnegative_int():
    for bad in (-1, True, 1.5, "3"):
        with pytest.raises(ValidationError, match="limit"):
            DataRequest(fields=["a"], limit=bad)  # type: ignore[arg-type]
    r = DataRequest(fields=["a"], limit=0)
    assert r.limit == 0


def test_data_request_enum_rejects_and_normalizes():
    with pytest.raises(ValidationError, match="snapshot_policy"):
        DataRequest(fields=["a"], snapshot_policy="bogus")
    with pytest.raises(ValidationError, match="engine"):
        DataRequest(fields=["a"], engine="DuckDbX")
    # 大小写不敏感：normalize 成小写规范值
    r = DataRequest(fields=["a"], engine="DuckDB", result="Arrow")
    assert r.engine == "duckdb" and r.result == "arrow"


def test_snapshot_policy_taxonomy_is_explicit():
    for policy in (
        "latest",
        "best_effort_fail_if_changed",
        "verified_fail_if_changed",
        "fail_if_changed",
        "pin",
    ):
        assert DataRequest(fields=["a"], snapshot_policy=policy).snapshot_policy == policy


def test_verified_snapshot_policy_rejects_missing_plan_manifest(tmp_path):
    store = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="权威 manifest"):
        store.plan(
            DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
        )


def test_verified_snapshot_policy_rejects_manifest_without_version_identity(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    monkeypatch.setattr(
        store, "manifest_version", lambda *_args, **_kwargs: {"has_manifest": True}
    )
    with pytest.raises(ValidationError, match="版本 token"):
        store.plan(
            DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
        )


def test_best_effort_snapshot_policy_allows_missing_manifest(tmp_path):
    store = _static_store(tmp_path)
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="best_effort_fail_if_changed")
    )
    assert plan.execute().to_arrow().num_rows == 3


def test_verified_snapshot_policy_rejects_execution_manifest_lookup_failure(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, _publisher_manifest(tmp_path))
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )

    def _fail_manifest(*_args, **_kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(store, "manifest_version", _fail_manifest)
    with pytest.raises(SnapshotBuildError, match="无法证明"):
        plan.execute()


def test_verified_snapshot_policy_rejects_execution_manifest_disappearance(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, _publisher_manifest(tmp_path))
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    monkeypatch.setattr(
        store, "manifest_version", lambda *_args, **_kwargs: {"has_manifest": False}
    )
    with pytest.raises(SnapshotBuildError, match="无法证明"):
        plan.execute()


def test_verified_snapshot_policy_rejects_execution_version_change(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    planned = {
        "has_manifest": True,
        "fresh": True,
        "dataset_version": "v1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(planned))
    _install_publisher_manifest(monkeypatch, store, _publisher_manifest(tmp_path))
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    monkeypatch.setattr(
        store,
        "manifest_version",
        lambda *_args, **_kwargs: {
            "has_manifest": True,
            "fresh": True,
            "dataset_version": "v2",
        },
    )
    with pytest.raises(SnapshotBuildError, match="版本已变化"):
        plan.execute()


def test_pin_rejects_unprovable_plan_file_identity(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    monkeypatch.setattr(
        "data_access.read.read_contract.build_file_manifest",
        lambda _paths: (FileVersion(path="/unstatable/file.parquet"),),
    )
    with pytest.raises(ValidationError, match="精确物理文件身份"):
        store.plan(DataRequest(fields=["v"], snapshot_policy="pin"))


def test_pin_rejects_unprovable_execution_file_identity(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="pin"))
    monkeypatch.setattr(
        "data_access.read.read_contract.build_file_manifest",
        lambda _paths: (FileVersion(path="/unstatable/file.parquet"),),
    )
    with pytest.raises(SnapshotBuildError, match="物理文件身份不完整"):
        plan.execute()


def test_legacy_fail_if_changed_remains_best_effort(tmp_path):
    store = _static_store(tmp_path)
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="fail_if_changed"))
    assert plan.execute().to_arrow().num_rows == 3


def test_pin_accepts_authoritative_empty_physical_set(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    _write_authoritative_manifest_token(tmp_path, file_count=0)
    token = store.manifest_version("ds")
    assert token["file_count"] == 0
    assert token["fresh"] is True
    monkeypatch.setattr(
        "data_access.read.read_contract.build_file_manifest", lambda _paths: ()
    )
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="pin"))
    assert plan.plan_snapshot_tokens["ds"]["file_count"] == 0
    assert plan.plan_pinned_files["ds"] == ()
    assert plan.execute().to_arrow().num_rows == 0


def test_pin_rejects_file_appearing_before_prepare_for_authoritative_empty(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    _write_authoritative_manifest_token(tmp_path, file_count=0)
    monkeypatch.setattr(
        "data_access.read.read_contract.build_file_manifest", lambda _paths: ()
    )
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="pin"))
    monkeypatch.undo()
    monkeypatch.setattr(type(plan), "_verify_snapshot_pin", lambda _self, _store: None)
    original_prepare = store.prepare_read

    def _appear_then_prepare(*args, **kwargs):
        scope = kwargs["physical_scope"]
        object.__setattr__(
            scope,
            "exact_objects",
            (str(tmp_path / "d" / "part.parquet"),),
        )
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(store, "prepare_read", _appear_then_prepare)
    with pytest.raises(ValidationError, match="terminal snapshot"):
        plan.execute()


def test_pin_rejects_replacement_between_plan_verification_and_prepare(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "file_count": 1,
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="pin"))
    original_prepare = store.prepare_read

    def _replace_then_prepare(*args, **kwargs):
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 99.0},
                    {"d": _dt.date(2024, 1, 4), "s": "CCC", "v": 100.0},
                ]
            ),
            str(tmp_path / "d" / "part.parquet"),
        )
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(store, "prepare_read", _replace_then_prepare)
    with pytest.raises(ValidationError, match="terminal snapshot"):
        plan.execute()


def test_pin_rejects_unbound_aggregation_execution(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    plan = store.plan(
        DataRequest(
            fields=["v"],
            snapshot_policy="pin",
            aggregations=[{"field": "v", "spec": "sum"}],
        )
    )
    with pytest.raises(SnapshotBuildError, match="aggregation"):
        plan.execute()


def test_pin_rejects_unbound_multi_dataset_execution(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="latest"))
    object.__setattr__(plan, "snapshot_policy", "pin")
    object.__setattr__(plan, "datasets", ("ds", "other"))
    object.__setattr__(
        plan,
        "plan_pinned_files",
        MappingProxyType({"ds": (), "other": ()}),
    )
    monkeypatch.setattr(type(plan), "_verify_snapshot_pin", lambda _self, _store: None)
    with pytest.raises(SnapshotBuildError, match="多数据集 join"):
        plan.execute()


def test_pin_rejects_unbound_time_varying_universe_execution(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    plan = store.plan(
        DataRequest(
            fields=["v"],
            snapshot_policy="pin",
            universe="ds",
            time_varying_universe=True,
        )
    )
    with pytest.raises(SnapshotBuildError, match="时变 universe"):
        plan.execute()


def test_direct_verified_resolver_requires_publisher_manifest():
    resolver = SourceSnapshotResolver(
        list_objects_fn=lambda _prefix: (),
        fallback_fn=lambda *_args: None,
        strict=False,
    )
    with pytest.raises(SourceSnapshotUnavailable, match="publisher source manifest"):
        resolver.resolve(
            "ds",
            policy="verified_fail_if_changed",
            paths=["s3://bucket/ds/*.parquet"],
        )


def test_direct_verified_resolver_rejects_identityless_manifest():
    resolver = SourceSnapshotResolver(
        source_manifest_fn=lambda _dataset: {"source_generation": "g1"},
        strict=False,
    )
    with pytest.raises(SourceSnapshotUnavailable, match="source manifest 解析失败"):
        resolver.resolve("ds", policy="verified_fail_if_changed")


def test_direct_verified_resolver_preserves_authoritative_empty_digest(tmp_path):
    manifest = _publisher_manifest(tmp_path, objects=[])
    resolver = SourceSnapshotResolver(
        source_manifest_fn=lambda _dataset: manifest,
        strict=False,
    )
    snapshot = resolver.resolve("ds", policy="verified_fail_if_changed")
    assert snapshot.source_generation == "publisher-g1"
    assert snapshot.objects == ()
    assert snapshot.content_digest == content_digest_of_objects(())
    assert snapshot.content_digest


def test_direct_verified_resolver_rejects_bad_empty_digest(tmp_path):
    manifest = _publisher_manifest(tmp_path, objects=[], content_digest="bad-digest")
    resolver = SourceSnapshotResolver(
        source_manifest_fn=lambda _dataset: manifest,
        strict=False,
    )
    with pytest.raises(SourceSnapshotUnavailable, match="content_digest"):
        resolver.resolve("ds", policy="verified_fail_if_changed")


def test_direct_verified_resolver_rejects_nonempty_terminal_for_empty_publisher(
    tmp_path,
):
    manifest = _publisher_manifest(tmp_path, objects=[])
    resolver = SourceSnapshotResolver(
        source_manifest_fn=lambda _dataset: manifest,
        strict=False,
    )
    with pytest.raises(SourceSnapshotUnavailable, match="object set"):
        resolver.resolve(
            "ds",
            policy="verified_fail_if_changed",
            paths=[str(tmp_path / "d" / "unexpected.parquet")],
        )


def test_direct_verified_resolver_rejects_terminal_object_set_mismatch(tmp_path):
    _static_store(tmp_path)
    manifest = _publisher_manifest(tmp_path)
    resolver = SourceSnapshotResolver(
        source_manifest_fn=lambda _dataset: manifest,
        strict=True,
    )
    part = tmp_path / "d" / "part.parquet"
    extra = tmp_path / "d" / "extra.parquet"
    for paths in ([str(extra)], [str(part), str(extra)], []):
        with pytest.raises(SourceSnapshotUnavailable, match="object set"):
            resolver.resolve(
                "ds",
                policy="verified_fail_if_changed",
                paths=paths,
            )


def test_direct_verified_resolver_rejects_local_manifest_without_strong_identity(
    tmp_path,
):
    _static_store(tmp_path)
    part = tmp_path / "d" / "part.parquet"
    manifest = _publisher_manifest(
        tmp_path,
        objects=[{"key": str(part), "size": part.stat().st_size}],
    )
    resolver = SourceSnapshotResolver(
        source_manifest_fn=lambda _dataset: manifest,
        strict=False,
    )
    with pytest.raises(SourceSnapshotUnavailable, match=r"size\+mtime_ns|checksum"):
        resolver.resolve(
            "ds",
            policy="verified_fail_if_changed",
            paths=[str(part)],
        )


def test_verified_local_checksum_is_bound_and_checked(tmp_path):
    part = tmp_path / "part.bin"
    part.write_bytes(b"publisher-authorized")
    checksum = hashlib.sha256(part.read_bytes()).hexdigest()
    obj = ResolvedObject(
        uri=str(part),
        content_length=part.stat().st_size,
        checksum=checksum,
        checksum_algorithm="sha256",
    )
    snapshot = ResolvedSourceSnapshot(
        dataset="ds",
        objects=(obj,),
        content_digest=content_digest_of_objects((obj,)),
    )
    verifier = SnapshotVerifier(strict=True)
    verifier.verify_before_execute(snapshot)
    part.write_bytes(b"publisher-tampered!!")
    assert part.stat().st_size == obj.content_length
    with pytest.raises(SourceSnapshotChanged, match="checksum"):
        verifier.verify_after_execute(snapshot)


def test_checksum_changes_snapshot_content_digest():
    first = ResolvedObject(
        uri="/tmp/checksum-object",
        content_length=4,
        checksum="a" * 64,
        checksum_algorithm="sha256",
    )
    second = ResolvedObject(
        uri="/tmp/checksum-object",
        content_length=4,
        checksum="b" * 64,
        checksum_algorithm="sha256",
    )
    assert content_digest_of_objects((first,)) != content_digest_of_objects((second,))


def test_verified_snapshot_policy_freezes_publisher_identity(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    manifest = _publisher_manifest(tmp_path)
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, manifest)
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    assert dict(plan.plan_publisher_snapshots["ds"]) == {
        "source_generation": manifest["source_generation"],
        "content_digest": manifest["content_digest"],
        "exact_objects": tuple(obj["key"] for obj in manifest["objects"]),
    }


def test_verified_snapshot_policy_pyarrow_scans_frozen_publisher_objects(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    manifest = _publisher_manifest(tmp_path)
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, manifest)
    calls = []
    original_before = store._pipeline.verify_before
    original_after = store._pipeline.verify_after

    def _before(snapshot, **kwargs):
        calls.append(
            (
                "before",
                tuple(str(obj.uri) for obj in snapshot.objects),
                kwargs.get("snapshot_policy"),
            )
        )
        return original_before(snapshot, **kwargs)

    def _after(snapshot, **kwargs):
        calls.append(
            (
                "after",
                tuple(str(obj.uri) for obj in snapshot.objects),
                kwargs.get("snapshot_policy"),
            )
        )
        return original_after(snapshot, **kwargs)

    monkeypatch.setattr(store._pipeline, "verify_before", _before)
    monkeypatch.setattr(store._pipeline, "verify_after", _after)
    plan = store.plan(
        DataRequest(
            fields=["v"],
            engine="pyarrow",
            snapshot_policy="verified_fail_if_changed",
        )
    )
    assert plan.execute().to_arrow().num_rows == 3
    expected = tuple(obj["key"] for obj in manifest["objects"])
    assert calls == [
        ("before", expected, "verified_fail_if_changed"),
        ("after", expected, "verified_fail_if_changed"),
    ]


def test_verified_policy_forces_final_verification_when_runtime_is_not_strict(
    tmp_path,
):
    part = tmp_path / "part.bin"
    part.write_bytes(b"publisher-authorized")
    stat = part.stat()
    obj = ResolvedObject(
        uri=str(part),
        content_length=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )
    snapshot = ResolvedSourceSnapshot(
        dataset="ds",
        objects=(obj,),
        content_digest=content_digest_of_objects((obj,)),
    )
    from data_access.runtime.read_pipeline import ReadPipeline

    pipeline = ReadPipeline(verifier=SnapshotVerifier(strict=False))
    pipeline.verify_before(
        snapshot, snapshot_policy="verified_fail_if_changed"
    )
    part.write_bytes(b"publisher-tampered!!")
    os.utime(
        part,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )
    assert part.stat().st_size == obj.content_length
    with pytest.raises(SourceSnapshotChanged, match="mtime"):
        pipeline.verify_after(
            snapshot, snapshot_policy="verified_fail_if_changed"
        )


def test_verified_policy_forces_remote_head_failure_closed_when_not_strict():
    obj = ResolvedObject(
        uri="s3://bucket/ds/part.parquet",
        etag="publisher-etag",
        content_length=10,
    )
    snapshot = ResolvedSourceSnapshot(
        dataset="ds",
        objects=(obj,),
        content_digest=content_digest_of_objects((obj,)),
    )
    verifier = SnapshotVerifier(
        remote_meta_fn=lambda _uri: (_ for _ in ()).throw(OSError("HEAD failed")),
        strict=False,
    )
    with pytest.raises(SourceSnapshotUnavailable, match="remote HEAD"):
        verifier.verify_before_execute(snapshot, force_strict=True)


def test_verified_policy_rejects_missing_remote_head_provider_when_not_strict():
    obj = ResolvedObject(
        uri="s3://bucket/ds/part.parquet",
        etag="publisher-etag",
        content_length=10,
    )
    snapshot = ResolvedSourceSnapshot(
        dataset="ds",
        objects=(obj,),
        content_digest=content_digest_of_objects((obj,)),
    )
    pipeline = ReadPipeline(verifier=SnapshotVerifier(strict=False))
    with pytest.raises(SourceSnapshotUnavailable, match="HEAD provider"):
        pipeline.verify_before(
            snapshot, snapshot_policy="verified_fail_if_changed"
        )
    with pytest.raises(SourceSnapshotUnavailable, match="HEAD provider"):
        pipeline.verify_after(
            snapshot, snapshot_policy="verified_fail_if_changed"
        )


def test_pyarrow_pin_rejects_terminal_file_version_change(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    _write_authoritative_manifest_token(tmp_path, file_count=1)
    plan = store.plan(DataRequest(fields=["v"], engine="pyarrow", snapshot_policy="pin"))
    part = tmp_path / "d" / "part.parquet"
    before = part.stat()
    monkeypatch.setattr(type(plan), "_verify_snapshot_pin", lambda _self, _store: None)
    os.utime(
        part,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    assert part.stat().st_size == before.st_size
    with pytest.raises(ValidationError, match="PyArrow terminal scan"):
        plan.execute()


def test_verified_authoritative_empty_pyarrow_preserves_requested_schema(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
        "file_count": 0,
    }
    manifest = _publisher_manifest(tmp_path, objects=[])
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, manifest)
    table = store.plan(
        DataRequest(
            fields=["d", "s", "v", "t"],
            engine="pyarrow",
            snapshot_policy="verified_fail_if_changed",
        )
    ).execute().to_arrow()
    assert table.num_rows == 0
    assert table.schema.names == ["d", "s", "v", "t"]
    assert table.schema.field("d").type == pa.date32()
    assert table.schema.field("s").type == pa.string()
    assert table.schema.field("v").type == pa.float64()
    assert table.schema.field("t").type == pa.timestamp("ms", tz="UTC")


def test_verified_snapshot_policy_rejects_same_size_local_replacement(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    manifest = _publisher_manifest(tmp_path)
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, manifest)
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    part = tmp_path / "d" / "part.parquet"
    before = part.stat()
    os.utime(
        part,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    assert part.stat().st_size == before.st_size
    assert part.stat().st_mtime_ns != before.st_mtime_ns
    with pytest.raises(SourceSnapshotChanged, match="mtime|snapshot"):
        plan.execute()


@pytest.mark.parametrize("changed_field", ["source_generation", "content_digest"])
def test_verified_snapshot_policy_rejects_publisher_identity_change(
    tmp_path, monkeypatch, changed_field
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    manifest = _publisher_manifest(tmp_path)
    current = {"value": manifest}
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    _install_publisher_manifest(monkeypatch, store, current["value"])
    store._pipeline._resolver._source_manifest_fn = lambda _dataset: current["value"]
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    if changed_field == "source_generation":
        current["value"] = _publisher_manifest(tmp_path, generation="publisher-g2")
    else:
        part = tmp_path / "d" / "part.parquet"
        current["value"] = _publisher_manifest(
            tmp_path,
            objects=[
                {
                    "key": str(part),
                    "size": part.stat().st_size,
                    "mtime_ns": part.stat().st_mtime_ns,
                    "etag": "local-publisher-v2",
                }
            ],
        )
    with pytest.raises(SnapshotBuildError, match="publisher|版本已变化"):
        plan.execute()


def test_verified_snapshot_policy_terminal_rejects_same_generation_digest_change(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    token = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_generation_id": "g1",
    }
    current = {"value": _publisher_manifest(tmp_path)}
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(token))
    store._pipeline._resolver._source_manifest_fn = lambda _dataset: current["value"]
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    part = tmp_path / "d" / "part.parquet"
    current["value"] = _publisher_manifest(
        tmp_path,
        generation="publisher-g1",
        objects=[
            {
                "key": str(part),
                "size": part.stat().st_size,
                "mtime_ns": part.stat().st_mtime_ns,
                "etag": "local-publisher-v2",
            }
        ],
    )
    monkeypatch.setattr(type(plan), "_verify_snapshot_pin", lambda _self, _store: None)
    with pytest.raises(ValidationError, match="digest"):
        plan.execute()


def test_verified_snapshot_policy_detects_independent_manifest_epoch_change(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    current = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "v1",
        "manifest_epoch": "m1",
        "manifest_generation_id": "g1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(current))
    _install_publisher_manifest(monkeypatch, store, _publisher_manifest(tmp_path))
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    current["manifest_epoch"] = "m2"
    with pytest.raises(SnapshotBuildError, match="版本已变化"):
        plan.execute()


def test_verified_snapshot_policy_detects_missing_and_falsy_identity(
    tmp_path, monkeypatch
):
    store = _static_store(tmp_path)
    current = {
        "has_manifest": True,
        "fresh": True,
        "source_epoch": "",
        "manifest_generation_id": "g1",
    }
    monkeypatch.setattr(store, "manifest_version", lambda *_args, **_kwargs: dict(current))
    _install_publisher_manifest(monkeypatch, store, _publisher_manifest(tmp_path))
    plan = store.plan(
        DataRequest(fields=["v"], snapshot_policy="verified_fail_if_changed")
    )
    current.pop("source_epoch")
    with pytest.raises(SnapshotBuildError, match="版本已变化"):
        plan.execute()


def test_pin_rejects_contradictory_empty_counts(tmp_path, monkeypatch):
    store = _static_store(tmp_path)
    monkeypatch.setattr(
        store,
        "manifest_version",
        lambda *_args, **_kwargs: {
            "has_manifest": True,
            "fresh": True,
            "source_epoch": "v1",
            "file_count": 0,
            "object_count": 1,
        },
    )
    monkeypatch.setattr(
        "data_access.read.read_contract.build_file_manifest", lambda _paths: ()
    )
    with pytest.raises(ValidationError, match="精确物理文件身份"):
        store.plan(DataRequest(fields=["v"], snapshot_policy="pin"))


# ---------------------------------------------------------------------------
# C. factor_lake_wide contract：specialized_only，generic read 拒绝
# ---------------------------------------------------------------------------


def test_factor_lake_wide_specialized_only_registry():
    reg = load_registry()
    ds = reg.get("factor_lake_wide")
    assert ds.specialized_only is True
    assert ds.specialized_only_reason
    # asset 是列轴不是物理列 → instrument_column 不该声明
    assert ds.instrument_column is None


def test_factor_lake_wide_generic_read_rejected(tmp_path):
    reg = load_registry()
    store = DataAccessStore(registry=reg, engine=DuckDBEngine(threads=1))
    for call in (
        lambda: store.read("factor_lake_wide", columns=["datetime"], params={"factor_id": "f"}),
        lambda: store.read_arrow("factor_lake_wide", params={"factor_id": "f"}),
        lambda: store.scan_polars("factor_lake_wide", params={"factor_id": "f"}),
        lambda: store.read_result("factor_lake_wide", params={"factor_id": "f"}),
    ):
        with pytest.raises(ValidationError, match="specialized_only"):
            call()


def test_specialized_only_requires_reason(tmp_path):
    """specialized_only=true 必须给 reason（防止配置裸标记无人能读）。"""
    root = tmp_path / "w"
    root.mkdir(parents=True, exist_ok=True)
    from data_access.registry.loader import _parse_dataset

    with pytest.raises(ValidationError, match="specialized_only_reason"):
        _parse_dataset("w", {"kind": "static", "access_mode": "published", "layout": "plain",
                             "root": str(root), "glob": "*.parquet",
                             "time_column": "d", "instrument_column": "s",
                             "specialized_only": True})


# ---------------------------------------------------------------------------
# D. ReadLineage None（全市场）vs ()（空股票池）+ params 不可变
# ---------------------------------------------------------------------------


def test_read_lineage_preserves_none_vs_empty(tmp_path):
    store = _static_store(tmp_path)
    # None → lineage.instrument_filter is None（全市场，不限制）
    h_none = store.read("ds", columns=["v"])
    assert h_none.lineage.instrument_filter is None
    # [] → 空股票池 0 行，lineage 记为 ()
    h_empty = store.read("ds", columns=["v"], instrument_filter=[])
    assert h_empty.lineage.instrument_filter == ()
    assert h_empty.lineage.instrument_filter is not None
    assert h_empty.to_arrow().num_rows == 0
    # 具体标的 → tuple
    h_aaa = store.read("ds", columns=["v"], instrument_filter=["AAA"])
    assert h_aaa.lineage.instrument_filter == ("AAA",)
    assert h_aaa.to_arrow().num_rows == 2


def test_read_lineage_params_immutable(tmp_path):
    store = _static_store(tmp_path)
    h = store.read("ds", columns=["v"])
    params = h.lineage.params
    assert isinstance(params, tuple)
    assert all(isinstance(pair, tuple) for pair in params)
    # 值全部冻结：没有可变 dict/list 泄漏
    for _k, v in params:
        assert not isinstance(v, (dict, list))


def test_stream_lineage_params_immutable(tmp_path):
    store = _static_store(tmp_path)
    handle = store.read("ds", columns=["v"], result="stream")
    params = handle.lineage.params
    assert isinstance(params, tuple)
    assert all(isinstance(pair, tuple) for pair in params)
    for _k, v in params:
        assert not isinstance(v, (dict, list))

"""Phase 6 加固回归：本轮新 invariants 的聚焦测试。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_access.core.exceptions import (
    DataError,
    ResourceBudgetExceeded,
    ValidationError,
)
from data_access.core.engine import _DeadlineConnectionPool
from data_access.cos.remote import _resolve_storage_paths, cos_uri_to_s3_uri
from data_access.cos.s3_duckdb import S3ConfigState
from data_access.read.query_budget import QueryBudget, resolve_query_budget
from data_access.registry.loader import DatasetRegistry, load_registry
from data_access.registry.params_validation import ParamSpec
from data_access.registry.yaml_loader import strict_yaml_load

_DATAACCESS_ROOT = Path(__file__).resolve().parent.parent.parent
_SEMANTIC_YAML = _DATAACCESS_ROOT / "config" / "semantic_fields.yaml"


# ---- #P0-3 params format 错误 fail-closed（不再 **/*.parquet 降级） ----


def _mk_parametric(tmp_path, *, glob_template):
    from data_access.registry.loader import ParametricDataset

    return ParametricDataset(
        name="lake",
        access_mode="namespaced",
        layout="plain",
        time_column=None,
        instrument_column=None,
        hive_partitioning=False,
        union_by_name=False,
        root_template=str(tmp_path) + "/{factor_id}",
        glob_template=glob_template,
        params_schema={"factor_id": "str"},
        param_specs={"factor_id": ParamSpec(name="factor_id", type="str")},
        static_root=tmp_path,
        storage={"source": {"type": "cos", "uri": "cos://bucket/lake"}},
    )


def test_remote_paths_from_storage_params_missing_fails_closed(tmp_path, monkeypatch):
    from data_access.cos.remote import _remote_paths_from_storage

    ds = _mk_parametric(tmp_path, glob_template="{factor_id}/**/*.parquet")
    # 缺参数 → 必须 ValidationError，禁止 fallback 全库 glob
    with pytest.raises(ValidationError):
        _remote_paths_from_storage(ds, time_range=None, params={})


def test_remote_paths_from_storage_glob_format_error_fails_closed(tmp_path):
    from data_access.cos.remote import _remote_paths_from_storage

    ds = _mk_parametric(
        tmp_path, glob_template="{missing}/**/*.parquet"  # .format(**validated) 会 KeyError
    )
    with pytest.raises(ValidationError):
        _remote_paths_from_storage(ds, time_range=None, params={"factor_id": "f1"})


# ---- #P0-2 / #P0-4 canonical s3:// execution URI + authorize ----


def test_storage_paths_normalize_cos_to_s3(monkeypatch):
    monkeypatch.setattr(
        "data_access.cos.remote.allowed_s3_prefixes",
        lambda: ("s3://bucket/",),
    )
    out = _resolve_storage_paths(
        ["cos://bucket/lake/2024-01-01.parquet"],
        authorized_prefix="s3://bucket/lake",
    )
    assert out == ["s3://bucket/lake/2024-01-01.parquet"]
    assert cos_uri_to_s3_uri("cos://a/b") == "s3://a/b"


def test_storage_paths_outside_prefix_authorize_fails(monkeypatch):
    monkeypatch.setattr(
        "data_access.cos.remote.allowed_s3_prefixes",
        lambda: ("s3://bucket/",),
    )
    with pytest.raises(ValidationError):
        _resolve_storage_paths(
            ["s3://other/nope.parquet"],
            authorized_prefix="s3://bucket/lake",
        )


# ---- #P0-5 S3 credential state per-connection ----


class _FakeConn:
    def __init__(self, name):
        self.name = name
        self.configured = 0

    def execute(self, sql):
        self.configured += 1


def test_s3_state_is_per_connection(monkeypatch):
    state = S3ConfigState()
    # fake creds 必须带齐 _creds_fingerprint 读的非 secret 身份字段
    # （principal_id/credential_scope_id/credential_generation_id/expires_at，
    # R24 P0-S1 §3.4：身份不含 secret）。
    monkeypatch.setattr(
        "data_access.cos.s3_duckdb.resolve_s3_credentials",
        lambda: type("C", (), {
            "access_key_id": "k", "secret_access_key": "s",
            "endpoint": "e", "region": "r", "url_style": "path", "use_ssl": True,
            "principal_id": None, "credential_scope_id": None,
            "credential_generation_id": None, "expires_at": None,
        })(),
    )
    monkeypatch.setattr(
        "data_access.cos.s3_duckdb.apply_s3_credentials",
        lambda conn, creds: conn.execute("CREATE SECRET ..."),
    )
    a, b = _FakeConn("a"), _FakeConn("b")
    state.ensure(a)
    state.configure_fresh(b)  # #P0-5 绝不能影响 a 的状态
    assert a.configured == 1
    assert b.configured == 1
    state.ensure(a)  # a 指纹相同 → 不重配
    assert a.configured == 1
    state.ensure(b)  # b 也已配 → 不重配
    assert b.configured == 1


# ---- #P0-8 deadline pool 真正限并发 ----


def test_deadline_pool_caps_concurrency():
    from data_access.core.duckdb_config import DuckDBConfig

    pool = _DeadlineConnectionPool(size=1)
    cfg = DuckDBConfig(threads=1)
    c1 = pool.acquire(cfg, wait_seconds=0.0)
    assert c1 is not None
    with pytest.raises(ResourceBudgetExceeded):
        pool.acquire(cfg, wait_seconds=0.0)
    pool.release(c1, healthy=True)
    c2 = pool.acquire(cfg, wait_seconds=0.0)  # 释放后可复用
    pool.release(c2, healthy=True)
    pool.close()


def test_deadline_pool_discards_unhealthy_connection():
    from data_access.core.duckdb_config import DuckDBConfig

    pool = _DeadlineConnectionPool(size=1)
    cfg = DuckDBConfig(threads=1)
    c = pool.acquire(cfg, wait_seconds=0.0)
    pool.release(c, healthy=False)  # interrupt/timeout 连接必须丢弃，不进 idle
    assert pool._idle == []
    c2 = pool.acquire(cfg, wait_seconds=0.0)  # 重新创建
    pool.release(c2, healthy=True)
    assert len(pool._idle) == 1
    pool.close()


# ---- #P0-14 QueryBudget production floor ----


def test_query_budget_production_floor_takes_stricter(monkeypatch):
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    b = resolve_query_budget(
        QueryBudget(max_rows=None, require_columns=False, require_time_range=False)
    )
    # 显式宽松 budget 不能取消 production 默认
    assert b.max_rows == 50_000_000
    assert b.require_columns is True


def test_query_budget_relaxed_not_in_production(monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    b = resolve_query_budget(QueryBudget(max_rows=1000, require_columns=False))
    assert b.max_rows == 1000
    assert b.require_columns is False


# ---- #P0-50 path_segment "false" → False ----


def test_param_spec_path_segment_string_false_is_false():
    spec = ParamSpec.from_yaml_value("p", {"type": "str", "path_segment": "false"})
    assert spec.path_segment is False
    spec2 = ParamSpec.from_yaml_value("p", {"type": "str", "path_segment": "true"})
    assert spec2.path_segment is True


def test_param_spec_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ParamSpec.from_yaml_value("p", {"type": "str", "bogus": 1})


# ---- #P0-46 duplicate YAML key → 硬错误 ----


def test_strict_yaml_duplicate_key_rejected():
    text = "a: 1\na: 2\n"
    with pytest.raises(ValidationError):
        strict_yaml_load(text)


def test_strict_yaml_allows_merge_anchor():
    text = "_base: &b\n  x: 1\nf:\n  <<: *b\n  y: 2\n"
    out = strict_yaml_load(text)
    assert out["f"]["x"] == 1
    assert out["f"]["y"] == 2


# ---- #P0-47 unresolved ${ENV} → 启动失败 ----


def test_registry_unresolved_env_fails(tmp_path, monkeypatch):
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        "lake:\n"
        "  kind: static\n"
        "  access_mode: published\n"
        "  layout: plain\n"
        "  root: ${DEFINITELY_UNSET_VAR_XYZ}/data\n"
        "  glob: '**/*.parquet'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_registry(yaml_path)


# ---- #P0-28 manifest 坏 parquet fail-closed ----


def test_manifest_build_corrupt_parquet_fails(tmp_path):
    from data_access.read.manifest import build_manifest_for_dataset
    from data_access.registry.loader import StaticDataset

    root = tmp_path / "data"
    root.mkdir()
    good = root / "good.parquet"
    pq.write_table(pa.table({"a": [1, 2], "ts": ["2024-01-01", "2024-01-02"]}), str(good))
    bad = root / "bad.parquet"
    bad.write_bytes(b"NOT A PARQUET FILE" * 10)

    ds = StaticDataset(
        name="ds",
        access_mode="published",
        layout="plain",
        time_column="ts",
        instrument_column=None,
        hive_partitioning=False,
        union_by_name=False,
        root=root,
        glob="*.parquet",
        schema={"a": "int", "ts": "string"},
    )
    registry = DatasetRegistry({"ds": ds})

    class _FakeStore:
        _registry = registry

        def _resolve_raw_paths(self, ds_obj, time_range=None, params=None):
            return [str(root / "*.parquet")]

    with pytest.raises(DataError):
        build_manifest_for_dataset(_FakeStore(), "ds")


# ---- #P0-31 manifest_root_for_paths exact file → parent ----


def test_manifest_root_for_paths_exact_file_is_parent(tmp_path):
    from data_access.read.manifest import manifest_root_for_paths

    p = tmp_path / "sub" / "data.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    root = manifest_root_for_paths([str(p)])
    assert root == p.parent


def test_manifest_root_for_paths_glob_is_static_prefix():
    from data_access.read.manifest import manifest_root_for_paths

    root = manifest_root_for_paths(["/data/factors/**/*.parquet"])
    assert str(root) == "/data/factors"


# ---- #P0-37 delete two-phase preflight：max_rows 超限 → 0 行删除 ----


def test_delete_rows_preflight_aborts_before_any_delete(tmp_path, monkeypatch):
    from data_access.registry.loader import StaticDataset
    from data_access.write.upsert import delete_rows_from_dataset

    ds = StaticDataset(
        name="staging",
        access_mode="staging",
        layout="plain",
        time_column="ts",
        instrument_column=None,
        hive_partitioning=False,
        union_by_name=False,
        root=tmp_path,
        glob="*.parquet",
    )
    table = pa.table(
        {
            "ts": pa.array(["2024-01-01"] * 10),
            "v": pa.array(range(10)),
        }
    )
    pq.write_table(table, str(tmp_path / "data.parquet"))

    class _Auth:
        def resolve_and_authorize(self, path):
            return Path(path)

    with pytest.raises(ValidationError):
        delete_rows_from_dataset(
            ds=ds,
            authorizer=_Auth(),
            target_dir=tmp_path,
            time_column="ts",
            start="2024-01-01",
            end="2024-01-01",
            max_rows=5,  # 10 行将删，超过 5 → abort
        )
    # 0 行实际删除
    assert pq.read_table(str(tmp_path / "data.parquet")).num_rows == 10


# ---- #P0-25 RAW_EVENT 禁止 generic latest-asof ----


def test_read_cos_events_asof_rejects_raw_event():
    from data_access.cos_contract import COSDatasetContract

    class _FakeContract:
        name = "us_fact_news"
        pit_policy = "strict"
        is_raw_event = True
        cardinality = "one_to_many"

    class _FakeStore:
        def read_cos_events(self, *a, **k):
            return None

    import pandas as pd

    import data_access.cos_event_runtime as cer

    orig = cer.resolve_event_clock
    orig_filters = cer.validate_event_filters
    cer.resolve_event_clock = lambda dataset, allow_effective_time=False: (
        _FakeContract(),
        "published_utc",
    )
    cer.validate_event_filters = lambda *a, **k: None
    try:
        with pytest.raises(ValidationError):
            cer.read_cos_events_asof(
                _FakeStore(),
                "us_fact_news",
                pd.DataFrame({"decision_timestamp": ["2024-01-01"], "instrument": ["AAA"]}),
            )
    finally:
        # 两个 monkeypatch 都必须恢复——漏掉 validate_event_filters 会让
        # `filters=None` 泄漏到整个 pytest 进程，后续任何 read_cos_events 在
        # `filters.keys()` 处崩（测试顺序敏感的静默失败）。
        cer.resolve_event_clock = orig
        cer.validate_event_filters = orig_filters


# ---- #P0-15 / #P0-23 semantic catalog contracts ----


def test_semantic_catalog_eps_is_financial_pit():
    from data_access.read.semantic_catalog import get_semantic_catalog

    catalog = get_semantic_catalog(_SEMANTIC_YAML)
    eps = catalog.get("eps", market="ashare")
    assert eps.temporal_model == "financial_event"
    assert eps.period_selection == "latest_period"
    assert eps.join_policy == "pit_asof_backward"
    assert eps.knowledge_time == "PubDate"
    assert eps.period_time == "ReportPeriodEndDate"
    assert eps.revision_order == ("UpdateTime",)


def test_semantic_catalog_us_sparse_raw_not_mining_allowed():
    from data_access.read.semantic_catalog import get_semantic_catalog

    catalog = get_semantic_catalog(_SEMANTIC_YAML)
    for name in ("dividend_yield", "return_on_equity", "return_on_assets",
                 "price_to_earnings", "price_to_book", "us_market_cap"):
        f = catalog.get(name, market="us")
        assert f.mining_allowed is False, name


def test_semantic_catalog_derived_market_cap_daily():
    from data_access.read.semantic_catalog import get_semantic_catalog

    catalog = get_semantic_catalog(_SEMANTIC_YAML)
    f = catalog.get("market_cap_daily", market="us")
    assert f.derived_expression is not None
    # #P1-final closure 7：DerivedFieldCompiler 未实现执行链 → mining_allowed=false
    # （fail-closed，Planner 遇到 derived 字段给清晰 ValidationError，不宣称可挖掘）。
    assert f.mining_allowed is False

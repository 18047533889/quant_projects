"""R39 misc P0/P1 —— DA 侧验收测试（#66/#67/#68/#69/#70/#71/#72/#73/#74）。

覆盖：
    #66   duckdb_inflight() 不再读 BoundedSemaphore._value，计数在 acquire/release
          （含 engine 直接 acquire 同一信号量）同步；
    #67   DerivedFieldCompiler：derived 字段 plan() 编译 + execute() 真正算出列；
          表达式不可编译 → typed UnsupportedFeatureError；
    #68   transforms/field_params 降级成分钟→日聚合执行；standalone frequency 对
          返回帧 resample；不支持 transform → typed UnsupportedFeatureError；
    #69   resolve_fields 重复字段名 research/production 一律 AmbiguousFieldError；
    #70   _check_factor_versions 显式 versions/require_same_* → 无法证明 == reject
          在 research（非 strict）也成立；
    #71   QueryResultCache 默认字节上限 fail-closed（env 未设也有 cap）；
    #72   动态 shrink hook + FE autopilot cache 消费者接线；
    #73   cache hit 不豁免结果预算（超 max_result_bytes → ValidationError）；
    #74   统一 cache inventory：DA QueryResultCache 上报字节、reclaim_to 生效。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from data_access import DataRequest
from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    AmbiguousFieldError,
    DataError,
    UnsupportedFeatureError,
    ValidationError,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, datasets: dict[str, dict]) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    lines.append(f"    {sk}: {json.dumps(str(sv))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


def _plain_ds(root: Path, name: str, *, extra_schema: dict[str, str] | None = None) -> dict[str, dict]:
    schema = {"ts": "date", "sym": "string", "val": "double"}
    if extra_schema:
        schema.update(extra_schema)
    return {
        name: {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(root), "glob": "part-*.parquet",
            "time_column": "ts", "instrument_column": "sym",
            "schema": schema,
        }
    }


# ---------------------------------------------------------------------------
# #66 duckdb_inflight 原子计数
# ---------------------------------------------------------------------------


def test_duckdb_inflight_counter_accurate_acquire_release():
    from data_access.runtime.resource_governor import (
        GlobalResourceGovernor,
        reset_global_governor,
    )

    gov = GlobalResourceGovernor(max_duckdb_concurrency=4)
    try:
        assert gov.duckdb_inflight() == 0
        gov.acquire_duckdb_slot()
        gov.acquire_duckdb_slot()
        assert gov.duckdb_inflight() == 2
        # engine 直接 acquire 同一信号量（_exec_sem 是同一对象）也同步计数。
        gov.duckdb_semaphore.acquire()
        assert gov.duckdb_inflight() == 3
        gov.release_duckdb_slot()
        gov.duckdb_semaphore.release()
        gov.release_duckdb_slot()
        assert gov.duckdb_inflight() == 0
    finally:
        reset_global_governor()


# ---------------------------------------------------------------------------
# #67 derived 字段：compile + execute
# ---------------------------------------------------------------------------


_DERIVED_YAML = """
adj_close:
  market: any
  dtype: double
  frequency: daily
  grain: instrument
  time_role: event_time
  temporal_model: panel
  join_policy: exact
  derived_expression: >-
    price.Close * price.Factor
  derived_from: [price.Close, price.Factor]
  mining_allowed: true
"""


def _reset_catalog():
    from data_access.read.semantic_catalog import reset_semantic_catalog

    reset_semantic_catalog()


def test_derived_field_compiles_and_executes(tmp_path, monkeypatch):
    sem = tmp_path / "semantic_fields.yaml"
    sem.write_text(_DERIVED_YAML, encoding="utf-8")
    monkeypatch.setenv("DATA_ACCESS_SEMANTIC_FIELDS", str(sem))
    _reset_catalog()
    try:
        root = tmp_path / "d"
        store = _store(
            tmp_path,
            _plain_ds(root, "price", extra_schema={"Close": "double", "Factor": "double"}),
        )
        store.write_arrow(
            "price",
            pa.table(
                {
                    "ts": [dt.date(2024, 1, 2)] * 2,
                    "sym": ["AAA", "BBB"],
                    "Close": [10.0, 20.0],
                    "Factor": [1.5, 2.0],
                }
            ),
        )
        plan = store.plan(DataRequest(fields=["adj_close"], anchor="price"))
        assert any(
            f.logical_name == "adj_close" for f in plan.derived_fields
        ), "derived field 必须进 plan.derived_fields"
        tbl = plan.execute().to_arrow()
        assert "adj_close" in tbl.column_names
        # Close * Factor
        expected = [10.0 * 1.5, 20.0 * 2.0]
        got = sorted(tbl.column("adj_close").to_pylist())
        assert got == sorted(expected)
    finally:
        _reset_catalog()


def test_derived_field_unparseable_raises_typed(tmp_path, monkeypatch):
    sem = tmp_path / "semantic_fields.yaml"
    sem.write_text(
        _DERIVED_YAML.replace("price.Close * price.Factor", "price.Close @@ price.Factor"),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_SEMANTIC_FIELDS", str(sem))
    _reset_catalog()
    try:
        root = tmp_path / "d"
        store = _store(tmp_path, _plain_ds(root, "price", extra_schema={"Close": "double", "Factor": "double"}))
        with pytest.raises(UnsupportedFeatureError):
            store.plan(DataRequest(fields=["adj_close"], anchor="price"))
    finally:
        _reset_catalog()


def test_derived_parse_semantic_field_allows_mining(tmp_path):
    from data_access.read.semantic_catalog import parse_semantic_field

    f = parse_semantic_field(
        "adj_close",
        {
            "market": "any",
            "derived_expression": "price.Close * price.Factor",
            "derived_from": ["price.Close", "price.Factor"],
            "mining_allowed": True,
        },
    )
    assert f.derived_expression
    assert f.mining_allowed is True


# ---------------------------------------------------------------------------
# #68 transforms / field_params / standalone frequency 真实执行
# ---------------------------------------------------------------------------


def test_transforms_and_field_params_lower_to_aggregation(tmp_path):
    """transforms: {vol: minute_range} + field_params 必须真实执行（不 reject）。"""
    root = tmp_path / "min"
    store = _store(
        tmp_path,
        {
            "minute": {
                "kind": "static", "access_mode": "staging", "layout": "plain",
                "root": str(root), "glob": "part-*.parquet",
                "time_column": "QuoteTime", "instrument_column": "Symbol",
                "schema": {
                    "QuoteTime": "timestamp",
                    "Symbol": "string",
                    "Volume": "double",
                },
            }
        },
    )
    store.write_arrow(
        "minute",
        pa.table(
            {
                "QuoteTime": [
                    dt.datetime(2024, 1, 2, 9, 31),
                    dt.datetime(2024, 1, 2, 9, 32),
                    dt.datetime(2024, 1, 2, 10, 0),
                ],
                "Symbol": ["A", "A", "A"],
                "Volume": [100.0, 200.0, 50.0],
            }
        ),
    )
    plan = store.plan(
        DataRequest(
            fields=["Volume"],
            anchor="minute",
            transforms={"Volume": "minute_range"},
            field_params={"Volume": {"start": "09:30", "end": "10:00", "metric": "sum"}},
            frequency="daily",
        )
    )
    tbl = plan.execute().to_arrow()
    assert tbl.num_rows == 1  # 聚合到单日
    # 输出列名由 AggregationSpec.effective_output_name 生成（volume_0930_1000）。
    vol_cols = [c for c in tbl.column_names if c.lower().startswith("volume")]
    assert vol_cols, tbl.column_names
    # 09:30~10:00 内三根分钟 volume 之和
    assert tbl.column(vol_cols[0]).to_pylist() == [350.0]


def test_transform_unsupported_raises_typed(tmp_path):
    root = tmp_path / "d"
    store = _store(tmp_path, _plain_ds(root, "ds"))
    store.write_arrow("ds", pa.table({"ts": [dt.date(2024, 1, 2)], "sym": ["A"], "val": [1.0]}))
    plan = store.plan(
        DataRequest(
            fields=["val"],
            anchor="ds",
            transforms={"val": "pct_change"},
        )
    )
    with pytest.raises(UnsupportedFeatureError):
        plan.execute()


def test_standalone_frequency_resamples_frame(tmp_path):
    """frequency 单独设置（无 aggregations）→ 对返回帧 resample，行数变少。"""
    root = tmp_path / "min"
    store = _store(
        tmp_path,
        {
            "minute": {
                "kind": "static", "access_mode": "staging", "layout": "plain",
                "root": str(root), "glob": "part-*.parquet",
                "time_column": "QuoteTime", "instrument_column": "Symbol",
                "schema": {
                    "QuoteTime": "timestamp",
                    "Symbol": "string",
                    "Volume": "double",
                },
            }
        },
    )
    store.write_arrow(
        "minute",
        pa.table(
            {
                "QuoteTime": [
                    dt.datetime(2024, 1, 2, 9, 31),
                    dt.datetime(2024, 1, 2, 9, 32),
                    dt.datetime(2024, 1, 2, 9, 33),
                ],
                "Symbol": ["A", "A", "A"],
                "Volume": [1.0, 2.0, 3.0],
            }
        ),
    )
    plan = store.plan(
        DataRequest(fields=["Volume"], anchor="minute", frequency="daily")
    )
    tbl = plan.execute().to_arrow()
    assert tbl.num_rows == 1  # 3 根分钟 → 1 个 daily 桶
    assert set(tbl.column_names) >= {"Volume", "QuoteTime"}


# ---------------------------------------------------------------------------
# #69 resolve_fields 歧义 research 也拒绝
# ---------------------------------------------------------------------------


def test_resolve_fields_ambiguous_rejects_even_research(tmp_path):
    root1 = tmp_path / "d1"
    root2 = tmp_path / "d2"
    store = _store(
        tmp_path,
        {
            **_plain_ds(root1, "dsa"),
            **_plain_ds(root2, "dsb"),
        },
    )
    # dsa 和 dsb 都有同名物理列 val（不在 catalog）→ registry 回退歧义。
    store.write_arrow("dsa", pa.table({"ts": [dt.date(2024, 1, 2)], "sym": ["A"], "val": [1.0]}))
    store.write_arrow("dsb", pa.table({"ts": [dt.date(2024, 1, 2)], "sym": ["B"], "val": [2.0]}))
    # 非 strict（research）也必须抛 AmbiguousFieldError，不再「取第一个」。
    with pytest.raises(AmbiguousFieldError):
        store.resolve_fields(["val"])
    # 限定名/显式 dataset 仍可解析。
    fields = store.resolve_fields(["val"], dataset="dsa")
    assert fields[0].dataset == "dsa"


# ---------------------------------------------------------------------------
# #70 _check_factor_versions 无法证明 == reject（research 也）
# ---------------------------------------------------------------------------


def _factor_lake_cfg(root: Path) -> dict[str, dict]:
    return {
        "factor_lake": {
            "kind": "parametric", "access_mode": "published", "layout": "hive",
            "root_template": str(root) + "/factors/{factor_id}",
            "glob_template": "year=*/panel.parquet",
            "partition_columns": ["year"],
            "time_column": "datetime", "instrument_column": "asset",
            "hive_partitioning": True,
            "params_schema": {"factor_id": "str"},
            "schema": {
                "datetime": "timestamp", "asset": "string", "value": "double",
                "factor_version": "string", "data_snapshot_id": "string",
            },
        },
    }


def _write_factor(root: Path, fid: str, *, version: str, snapshot: str):
    d = root / "factors" / fid / "year=2024"
    d.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "datetime": dt.datetime(2024, 1, 2, 9, 30),
                    "asset": "AAA",
                    "value": 1.0,
                    "factor_version": version,
                    "data_snapshot_id": snapshot,
                }
            ]
        ),
        str(d / "panel.parquet"),
    )


def test_check_factor_versions_cannot_prove_rejects_in_research(tmp_path):
    root = tmp_path / "lake"
    # f1 有数据、f2 目录空（无 panel.parquet → probe 结果为空集合）。
    _write_factor(root, "f1", version="v1", snapshot="s")
    (root / "factors" / "f2" / "year=2024").mkdir(parents=True, exist_ok=True)
    store = _store(tmp_path, _factor_lake_cfg(root))
    # 非 strict（research）：显式传 versions 且 f2 probe 为空 → 仍必须拒绝。
    with pytest.raises(DataError):
        store._check_factor_versions(
            ["f1", "f2"],
            versions={"f1": "v1", "f2": "v1"},
            time_range=("2024-01-01", "2024-01-31"),
        )


def test_check_factor_versions_same_snapshot_empty_probe_research(tmp_path):
    root = tmp_path / "lake"
    _write_factor(root, "f1", version="v1", snapshot="s")
    (root / "factors" / "f2" / "year=2024").mkdir(parents=True, exist_ok=True)
    store = _store(tmp_path, _factor_lake_cfg(root))
    with pytest.raises(DataError):
        store._check_factor_versions(
            ["f1", "f2"],
            require_same_data_snapshot=True,
            time_range=("2024-01-01", "2024-01-31"),
        )


# ---------------------------------------------------------------------------
# #71/#72/#73/#74 QueryResultCache：默认字节上限、shrink、hit 预算、inventory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache():
    from data_access.read.query_cache import reset_query_cache

    reset_query_cache()
    yield
    reset_query_cache()


def test_query_cache_default_byte_cap_fail_closed():
    """#71：env 未设 DATA_ACCESS_QUERY_CACHE_BYTES → max_bytes 也有 fail-closed 默认。"""
    from data_access.read.query_cache import get_query_cache

    cache = get_query_cache()
    assert cache.max_bytes is not None
    assert cache.max_bytes >= 1


def test_query_cache_shrink_hook_lru():
    """#72：shrink_to 逐出 LRU 最旧条目直到字节 ≤ target。"""
    from data_access.read.query_cache import get_query_cache

    cache = get_query_cache()
    cache.set_with_size("k1", pa.table({"a": [1]}), nbytes=100)
    cache.set_with_size("k2", pa.table({"a": [2]}), nbytes=100)
    cache.set_with_size("k3", pa.table({"a": [3]}), nbytes=100)
    assert cache.total_bytes() == 300
    cache.shrink_to(150)
    assert cache.total_bytes() <= 150
    # LRU：k1 最旧，先被逐出。
    assert cache.get("k1") is None
    assert cache.get("k3") is not None


def test_query_cache_hit_not_exempt_from_budget(tmp_path, monkeypatch):
    """#73：cache hit 返回前用当前 dataset/query budget 核对 num_rows/nbytes。

    先一次 miss 把结果写进缓存；再把 ``_resolve_read_budget`` 换成极紧预算，
    第二次读是 hit → 也必须被 budget 拦下（cache hit 不豁免）。
    """
    from data_access.read.query_cache import reset_query_cache
    from data_access.read.read_contract import (
        ReadLineage,
        ReadResult,
        ReadStats,
    )

    reset_query_cache()
    root = tmp_path / "d"
    store = _store(tmp_path, _plain_ds(root, "ds"))
    store.enable_result_cache(True)
    try:
        fresh = {"has_manifest": True, "fresh": True, "source_epoch": "e1",
                 "manifest_epoch": "e1", "dataset_version": "v1",
                 "partition_version": "p1"}
        monkeypatch.setattr(store, "manifest_version", lambda *a, **k: dict(fresh))

        def _fake_read_result(*a, **k):
            big = pa.table({"ts": [dt.date(2024, 1, 2)] * 10, "sym": ["A"] * 10,
                            "val": [1.0] * 10})
            return ReadResult(
                table=big,
                snapshot=SimpleNamespace(snapshot_id="snap1"),
                stats=ReadStats(rows=10, bytes=big.nbytes, elapsed_ms=0.0),
                lineage=ReadLineage(
                    dataset="ds", columns=(), time_range=None,
                    instrument_filter=None, params={},
                ),
            )

        monkeypatch.setattr(store, "read_result", _fake_read_result)
        first = store.read_cached("ds", columns=["val"])
        assert first.num_rows == 10  # miss 已写缓存
        # 收紧结果预算 → 第二次读（cache hit）也必须被拦。
        monkeypatch.setattr(
            store, "_resolve_read_budget", lambda *a, **k: SimpleBudget(max_result_bytes=1)
        )
        with pytest.raises(ValidationError, match="超过预算"):
            store.read_cached("ds", columns=["val"])
    finally:
        store.enable_result_cache(False)
        reset_query_cache()


def test_query_cache_inventory_reports_bytes_and_reclaim():
    """#74：DA QueryResultCache 注册进统一 inventory；reclaim_to 生效。"""
    from data_access.read.query_cache import get_query_cache
    from data_access.runtime.cache_inventory import (
        inventory_summary,
        reclaim_to,
        total_reclaimable_bytes,
    )

    cache = get_query_cache()
    cache.set_with_size("k1", pa.table({"a": [1]}), nbytes=500)
    cache.set_with_size("k2", pa.table({"a": [2]}), nbytes=500)
    summary = inventory_summary()
    names = {o["name"] for o in summary["owners"]}
    assert "da_query_result_cache" in names
    assert total_reclaimable_bytes() >= 1000
    reclaimed = reclaim_to(200)
    assert cache.total_bytes() <= 200
    assert reclaimed.get("da_query_result_cache", 0) >= 0


# ---------------------------------------------------------------------------
# 小工具：给 test_query_cache_hit_not_exempt 用
# ---------------------------------------------------------------------------


class SimpleBudget:
    def __init__(self, max_result_bytes=None):
        self.max_result_bytes = max_result_bytes
        self.max_rows = None
        self.max_elapsed_ms = None
        self.require_columns = False
        self.require_time_range = False

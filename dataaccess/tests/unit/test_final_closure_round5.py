# -*- coding: utf-8
"""第五轮收口回归：二阶边界 bug（Core Freeze blockers 1-7 + P1 收尾 8-12）。

覆盖：
    P0-C1  query_cache_key 严格区分 instrument_filter=[]（空池）与 None（全市场）
    P0-C2  read_cached 缓存命中也必须经过 read gates（semantic/budget 前置）
    P0-C3  COS semantic helper（panel/events）继承 [] = empty（WHERE FALSE）
    P0-C4  ReadHandle one-shot stream 完整消费后第二终点 fail-closed（buffer=True 例外）
    P0-C5  ReadHandle lazy 第一次 terminal collect 缓存 canonical Arrow，不重复执行
    P0-C6  remote snapshot metadata 进程缓存 TTL（失败 None 不永久缓存）
    P0-C7  _remote_object_meta 统一 cos_uri_to_s3_uri 再切 bucket/key
    P1-C8  MetadataPlane PIT 复用 validate_current_source（epoch 不变但 schema 变 → stale）
    P1-C9  RelationHandle.sql/sql_fragment 的 ? 计数 lexical-safe（跳过字面量/注释）
    P1-C10 _stats_sidecar_fresh legacy sidecar 无 source identity → production stale
    P1-C11 read_cos_events_asof 输出列冲突统一 allocate_unique_column_name
    P1-C12 DataRequest/SemanticField 严格 sequence parser（拒绝裸 str/非法元素）
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.read.query_budget import QueryBudget
from data_access.read.read_handle import ReadHandle
from data_access.registry import load_registry
from data_access.store import DataAccessStore


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


def _writable_ds(root: Path, name: str) -> dict[str, dict]:
    return {
        name: {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(root), "glob": "part-*.parquet",
            "time_column": "ts", "instrument_column": "sym",
            "schema": {"ts": "date", "sym": "string", "val": "double"},
        }
    }


def _reset_cache():
    from data_access.read.query_cache import reset_query_cache

    reset_query_cache()


# ---------------------------------------------------------------------------
# P0-C1 cache key：[]（空股票池）≠ None（全市场）
# ---------------------------------------------------------------------------

def test_cache_key_distinguishes_empty_pool_from_all_market():
    from data_access.read.query_cache import query_cache_key

    kw = dict(dataset="d", params={}, time_range=None, columns=None,
              manifest_token=None)
    k_all = query_cache_key(**kw, instruments=None)
    k_empty = query_cache_key(**kw, instruments=[])
    assert k_all != k_empty, "[]（空池）与 None（全市场）必须不同 key——否则串 cache 命中错误结果"
    # 顺序无关（有序集合同一 key），但绝不能与全市场/空池同 key
    k_a = query_cache_key(**kw, instruments=["B", "A"])
    k_b = query_cache_key(**kw, instruments=["A", "B"])
    assert k_a == k_b
    assert k_a != k_all and k_a != k_empty


# ---------------------------------------------------------------------------
# P0-C2 read_cached：缓存命中也要经过 read gates
# ---------------------------------------------------------------------------

def test_read_cached_hit_still_runs_gates(tmp_path, monkeypatch):
    _reset_cache()
    st = _store(tmp_path, _writable_ds(tmp_path / "d", "ds"))
    st.enable_result_cache(True)
    try:
        fresh = {"has_manifest": True, "fresh": True, "source_epoch": "e1",
                 "manifest_epoch": "e1", "dataset_version": "v1",
                 "partition_version": "p1"}
        monkeypatch.setattr(st, "manifest_version", lambda *a, **k: dict(fresh))
        monkeypatch.setattr(
            st, "read_arrow",
            lambda *a, **k: pa.table({"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [2.0]}),
        )
        first = st.read_cached("ds", columns=["sym"])
        assert first.num_rows == 1
        # 缓存已命中；gate 必须仍执行（hit 不能绕过 semantic/budget gate）
        def boom(*a, **k):
            raise ValidationError("gate enforced on cache hit")

        monkeypatch.setattr(st, "_prepare_read_request", boom)
        with pytest.raises(ValidationError, match="gate enforced"):
            st.read_cached("ds", columns=["sym"])
    finally:
        st.enable_result_cache(False)


# ---------------------------------------------------------------------------
# P0-C3 COS helper [] = empty
# ---------------------------------------------------------------------------

class _CaptureSqlStore:
    def __init__(self):
        self.queries: list[tuple[str, list]] = []

    def sql(self, query, *, view_columns=None, **kwargs):
        self.queries.append((query, list(kwargs.get("params") or [])))
        cols = next(iter((view_columns or {}).values()))
        return pa.table({c: pa.array([], type=pa.string()) for c in cols})


def test_cos_panel_empty_instrument_pool_generates_where_false():
    from data_access.cos_panel_runtime import read_cos_panel

    st = _CaptureSqlStore()
    read_cos_panel(
        st, "ashare_stock_industry",
        columns=["TradeDate", "Symbol", "IndustrySource"],
        semantic_filters={"IndustrySource": "sw_l2"},
        instrument_filter=[],
    )
    assert st.queries, "空股票池也应走 SQL 分支"
    query, params = st.queries[-1]
    assert "1 = 0" in query, "instrument_filter=[] 必须生成 WHERE FALSE，不能静默全市场"
    assert "IndustrySource" in query


def test_cos_events_empty_instrument_pool_generates_where_false():
    from data_access.cos_event_runtime import read_cos_events

    st = _CaptureSqlStore()
    read_cos_events(
        st, "us_stock_balance",
        columns=["value"],
        instrument_filter=[],
        event_filters={"timeframe": "quarterly"},
    )
    assert st.queries
    query, params = st.queries[-1]
    assert "1 = 0" in query, "read_cos_events(instrument_filter=[]) 必须 WHERE FALSE"


# ---------------------------------------------------------------------------
# P0-C4 ReadHandle one-shot stream fail-closed
# ---------------------------------------------------------------------------

def _two_batches():
    yield pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["x"])
    yield pa.RecordBatch.from_arrays([pa.array([3, 4])], names=["x"])


def test_read_handle_stream_full_consumption_then_to_arrow_fail_closed():
    h = ReadHandle(stream=_two_batches())
    batches = list(h.stream())
    assert sum(b.num_rows for b in batches) == 4
    # 迭代器已耗尽：to_arrow() 再 list() 只会得空表 → 必须 fail-closed
    with pytest.raises(RuntimeError, match="one-shot|完整消费"):
        h.to_arrow()
    # 第二次 stream() 同样 fail-closed
    with pytest.raises(RuntimeError, match="one-shot|完整消费"):
        list(h.stream())


def test_read_handle_stream_partial_break_then_to_arrow_fail_closed():
    h = ReadHandle(stream=_two_batches())
    gen = h.stream()
    next(gen)
    gen.close()  # 模拟 break / GeneratorExit
    with pytest.raises(RuntimeError, match="one-shot|部分消费"):
        h.to_arrow()


def test_read_handle_stream_buffer_explicit_reuse():
    h = ReadHandle(stream=_two_batches())
    batches = list(h.stream(buffer=True))
    assert sum(b.num_rows for b in batches) == 4
    tbl = h.to_arrow()  # buffer=True 显式固化 → 可复用
    assert tbl.num_rows == 4
    assert sum(b.num_rows for b in h.stream()) == 4


# ---------------------------------------------------------------------------
# P0-C5 ReadHandle lazy canonical materialization（单次执行）
# ---------------------------------------------------------------------------

def test_read_handle_lazy_canonical_single_materialization():
    import polars as pl

    table = pa.table({"x": [1, 2, 3]})

    class _FakeLazy:
        def __init__(self, t):
            self._t = t
            self.collect_calls = 0

        def collect(self):
            self.collect_calls += 1
            return pl.from_arrow(self._t)

        @property
        def columns(self):
            return self._t.column_names

    lazy = _FakeLazy(table)
    h = ReadHandle(lazy=lazy)
    df = h.to_polars()
    assert df["x"].to_list() == [1, 2, 3]
    assert lazy.collect_calls == 1
    tbl = h.to_arrow()
    assert tbl.num_rows == 3
    assert lazy.collect_calls == 1, "canonical materialization 后不得二次执行 LazyFrame"
    assert sum(b.num_rows for b in h.stream(batch_size=2)) == 3
    assert lazy.collect_calls == 1


def test_read_handle_governed_lazy_single_execution(monkeypatch):
    import polars as pl

    import data_access.read.query_budget as qb

    calls = {"n": 0}
    orig = qb.collect_polars_with_budget

    def counted(lf, query_budget=None):
        calls["n"] += 1
        return orig(lf, query_budget=query_budget)

    monkeypatch.setattr(qb, "collect_polars_with_budget", counted)
    h = ReadHandle(lazy=pl.LazyFrame({"x": [1, 2, 3]}), budget=QueryBudget(),
                   govern_lazy=True)
    assert h.to_arrow().num_rows == 3
    assert h.to_polars()["x"].to_list() == [1, 2, 3]
    assert sum(b.num_rows for b in h.stream(batch_size=2)) == 3
    assert calls["n"] == 1, "to_arrow/to_polars/stream 必须共享同一次底层执行"


# ---------------------------------------------------------------------------
# P0-C6 / P0-C7 remote meta：TTL + cos:// URI 归一化
# ---------------------------------------------------------------------------

def _install_fake_boto3(monkeypatch, client_impl):
    fake_cfg = SimpleNamespace(Config=lambda **k: None)
    fake_botocore = SimpleNamespace(config=fake_cfg)
    fake_boto3 = SimpleNamespace(client=client_impl)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_cfg)


def test_remote_object_meta_cos_uri_normalized(monkeypatch):
    """#P0-C7 cos:// 必须先 cos_uri_to_s3_uri 再切 bucket/key。"""
    import data_access.cos.remote as crm
    import data_access.read.read_contract as rc

    rc._remote_meta_cache.clear()
    captured: dict = {}

    class _FakeClient:
        def head_object(self, **kw):
            captured.update(kw)
            return {"ETag": '"abc"', "ContentLength": 5}

    _install_fake_boto3(monkeypatch, lambda *a, **k: _FakeClient())

    class _Creds:
        use_ssl = False
        endpoint = None
        region = "us-east-1"
        access_key_id = "k"
        secret_access_key = "s"

    monkeypatch.setattr(crm, "resolve_s3_credentials", lambda: _Creds())
    meta = rc._remote_object_meta("cos://qs-cold/foo/bar.parquet")
    assert captured.get("Bucket") == "qs-cold"
    assert captured.get("Key") == "foo/bar.parquet"
    assert meta["etag"] == "abc"


def test_remote_meta_cache_ttl_no_permanent_none(monkeypatch):
    """#P0-C6 失败/无凭证的 None 不能永久缓存；TTL 过期后必须重试。"""
    import data_access.cos.remote as crm
    import data_access.read.read_contract as rc

    rc._remote_meta_cache.clear()
    calls = {"n": 0}

    def no_creds():
        calls["n"] += 1
        raise RuntimeError("no creds")

    monkeypatch.setattr(crm, "resolve_s3_credentials", no_creds)
    rc._remote_object_meta("s3://b/x.parquet")
    rc._remote_object_meta("s3://b/x.parquet")
    assert calls["n"] == 1, "fresh TTL 内命中缓存，不重复尝试"
    rc._remote_meta_cache["s3://b/x.parquet"] = (time.monotonic() - 100, None)
    rc._remote_object_meta("s3://b/x.parquet")
    assert calls["n"] == 2, "TTL 过期 → 重新尝试（首次无凭证、之后补凭证能拿到）"


# ---------------------------------------------------------------------------
# P1-C8 validate_current_source 统一 PIT 源身份校验
# ---------------------------------------------------------------------------

def test_validate_current_source_detects_schema_change(tmp_path):
    import data_access.read.pit_event_index as pei

    class _Ds:
        name = "ds"
        schema = {"ticker": "string", "value": "double"}
        instrument_column = "ticker"

    class _Store:
        def __init__(self):
            self.ds = _Ds()

        def manifest_version(self, *a, **k):
            return {"manifest_epoch": "e1"}

        def _prepare_dataset_read(self, *a, **k):
            return ["/nonexistent/x.parquet"]

        @property
        def _registry(self):
            return SimpleNamespace(get=lambda n: self.ds)

    st = _Store()
    meta = pei.PITIndexMetadata(
        complete=True,
        source_file_count=1,
        indexed_file_count=1,
        manifest_epoch="e1",
        schema_hash=pei._schema_hash_of(st.ds),
        source_snapshot=pei._source_snapshot(st, "ds", ["/nonexistent/x.parquet"]),
    )
    assert pei.validate_current_source(st, "ds", meta) is True
    # epoch 不变、schema 变（外部绕过 DataAccess 写入）→ 必须 stale
    st.ds.schema = {"ticker": "string", "value": "double", "extra": "double"}
    assert pei.validate_current_source(st, "ds", meta) is False


def test_metadata_plane_pit_index_delegates_to_validate_current_source(monkeypatch):
    """#P1-C8 MetadataPlane 不再写第二套判断，直接复用统一 validator。"""
    import data_access.read.pit_event_index as pei
    from data_access.read.metadata_plane import DatasetMetadataPlane

    called = {"n": 0}
    orig = pei.validate_current_source

    def spy(store, dataset, meta):
        called["n"] += 1
        return False  # 统一校验说不权威 → plane 必须拒绝

    monkeypatch.setattr(pei, "validate_current_source", spy)
    class _FakeStore:
        def __init__(self):
            self._registry = SimpleNamespace(
                get=lambda n: SimpleNamespace(schema={"a": "double"}),
            )

        def manifest_version(self, *a, **k):
            return {"source_epoch": "e1", "manifest_epoch": "e1"}

        def _resolve_raw_paths(self, *a, **k):
            return []

    fake = _FakeStore()
    monkeypatch.setattr(pei, "_index_path_for", lambda store, ds: Path("/x/_pit_event_index.parquet"))
    monkeypatch.setattr(pei, "load_pit_event_index", lambda p: SimpleNamespace(
        metadata=SimpleNamespace(),
    ))
    result = DatasetMetadataPlane(fake, "ds").pit_event_index()
    assert result is None
    assert called["n"] >= 1, "pit_event_index 必须调用统一的 validate_current_source"


# ---------------------------------------------------------------------------
# P1-C9 RelationHandle sql() lexical-safe placeholder 计数
# ---------------------------------------------------------------------------

def test_sql_placeholder_scanner_lexical():
    from data_access.read.relation_handle import _count_sql_placeholders

    text = "SELECT '?' AS x, ? FROM _sub"
    assert _count_sql_placeholders(text, text.index("FROM")) == 1
    text2 = "-- ? comment\nSELECT ? FROM _sub"
    assert _count_sql_placeholders(text2, text2.index("FROM")) == 1
    text3 = 'SELECT "?" AS c, ? FROM _sub'
    assert _count_sql_placeholders(text3, text3.index("FROM")) == 1
    text4 = "/* ? */ SELECT ? FROM _sub"
    assert _count_sql_placeholders(text4, text4.index("FROM")) == 1
    text5 = "SELECT ? , ? FROM _sub"
    assert _count_sql_placeholders(text5, text5.index("FROM")) == 2


def test_relation_handle_sql_param_alignment_lexical():
    from data_access.read.relation_handle import RelationHandle

    base = RelationHandle(SimpleNamespace(_engine=None), "SELECT 1", params=[])
    out = base.sql("SELECT '?' AS q, ? AS z FROM _sub", params=["OUTER"])
    assert out._params == ["OUTER"], "字符串字面量里的 ? 不能计入参数位置"
    out2 = base.sql_fragment("SELECT '?' AS q, ? AS z FROM {sub}", params=["OUTER"])
    assert out2._params == ["OUTER"]


# ---------------------------------------------------------------------------
# P1-C10 stats sidecar fail-open
# ---------------------------------------------------------------------------

def test_stats_sidecar_fresh_legacy_fail_closed(monkeypatch):
    from data_access.read.read_auto_router import _stats_sidecar_fresh

    class _GoodStore:
        def manifest_version(self, *a, **k):
            return {"source_epoch": "e1", "manifest_epoch": "e1"}

    class _BadStore:
        def manifest_version(self, *a, **k):
            raise RuntimeError("boom")

    legacy = SimpleNamespace(source_epoch=None)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    assert _stats_sidecar_fresh(_GoodStore(), "ds", legacy) is True  # research 兼容
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    assert _stats_sidecar_fresh(_GoodStore(), "ds", legacy) is False  # production → stale
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    # 有 source_epoch 且一致 → fresh；manifest 检查失败 → fail-closed stale
    modern = SimpleNamespace(source_epoch="e1")
    assert _stats_sidecar_fresh(_GoodStore(), "ds", modern) is True
    assert _stats_sidecar_fresh(_BadStore(), "ds", modern) is False


# ---------------------------------------------------------------------------
# P1-C11 read_cos_events_asof 输出列冲突统一分配
# ---------------------------------------------------------------------------

class _AsofStore:
    def __init__(self, tables):
        self.tables = tables

    def sql(self, query, **kwargs):
        dataset = kwargs["read_datasets"][0]
        return pa.Table.from_pandas(self.tables[dataset], preserve_index=False)


def test_cos_events_asof_column_conflict_two_level():
    from data_access.cos_runtime import _read_cos_events_asof

    events = pd.DataFrame({
        "ticker": ["A"], "filing_date": ["2024-05-01"], "period_end": ["2024-03-31"],
        "timeframe": ["quarterly"], "value": [10.0], "x": [99.0],
    })
    decisions = pd.DataFrame({
        "instrument": ["A"], "decision_timestamp": ["2024-06-01"],
        "x": ["keep"], "x_event": ["keep2"],
    })
    result = _read_cos_events_asof(
        _AsofStore({"us_stock_balance": events}),
        "us_stock_balance", decisions,
        columns=["value"], event_filters={"timeframe": "quarterly"},
    )
    assert result.loc[0, "x"] == "keep", "decisions 原 x 列不能被覆盖"
    assert result.loc[0, "x_event"] == "keep2", "decisions 原 x_event 列不能被覆盖"
    assert "x_event_2" in result.columns and result.loc[0, "x_event_2"] == 99.0
    assert "fundamental_staleness_days" in result.columns


def test_cos_events_asof_empty_decisions_staleness_conflict():
    from data_access.cos_runtime import _read_cos_events_asof

    events = pd.DataFrame(columns=["ticker", "filing_date", "period_end",
                                   "timeframe", "value"])
    decisions = pd.DataFrame({
        "instrument": pd.Series(dtype="string"),
        "decision_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        "fundamental_staleness_days": pd.Series(dtype="int64"),
    })
    result = _read_cos_events_asof(
        _AsofStore({"us_stock_balance": events}),
        "us_stock_balance", decisions,
        columns=["value"], event_filters={"timeframe": "quarterly"},
    )
    assert "fundamental_staleness_days" in result.columns, "原列保留"
    assert "fundamental_staleness_days_event" in result.columns, "staleness 列改名不覆盖"


# ---------------------------------------------------------------------------
# P1-C12 严格 sequence parser（DataRequest / SemanticField）
# ---------------------------------------------------------------------------

def test_data_request_rejects_bare_str_sequences():
    from data_access import DataRequest

    with pytest.raises(ValidationError):
        DataRequest(fields="close")
    with pytest.raises(ValidationError):
        DataRequest(fields=["close"], instruments="AAPL")
    with pytest.raises(ValidationError):
        DataRequest(fields=["close"], order_by="date")
    with pytest.raises(ValidationError):
        DataRequest(fields=[1, 2])
    ok = DataRequest(fields=["close", "open"], instruments=["A"], order_by=["date"])
    assert ok.fields == ("close", "open")
    assert ok.instruments == ("A",)


def test_strict_sequence_shared_parser():
    from data_access.read.predicate import strict_sequence

    assert strict_sequence(None, name="x", allow_none=False) == ()
    assert strict_sequence(None, name="x", allow_none=True) is None
    assert strict_sequence(["a", "b"], name="x") == ("a", "b")
    with pytest.raises(ValidationError):
        strict_sequence("abc", name="x")
    with pytest.raises(ValidationError):
        strict_sequence(["a", 1], name="x")
    with pytest.raises(ValidationError):
        strict_sequence(42, name="x")


def test_semantic_field_tuple_of_strict():
    from data_access.read.semantic_catalog import _tuple_of

    assert _tuple_of(None) == ()
    assert _tuple_of(["a", "b"]) == ("a", "b")
    with pytest.raises(ValidationError):
        _tuple_of("bad")
    with pytest.raises(ValidationError):
        _tuple_of([1, 2])
    with pytest.raises(ValidationError):
        _tuple_of(42)

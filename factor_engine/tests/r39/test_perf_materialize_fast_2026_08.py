# -*- coding: utf-8 -*-
"""R39-PERF-038/039/040/041/042/068：批量物化 + 流式 sink 性能整改验收。

覆盖：
    (a) Gate-02：materialize_many_fast writer 热循环 O(1) 查找（静态 + 动态）；
    (b) execute_materialize_batch 与 N×execute_materialize 输出完全一致
        （value/index/dtype/factor_version/catalog 依赖边）；
    (c) Gate-03：batch_write_transaction_count << item_count；
    (d) sink flush 谓词：bytes + age + count 任一触发；
    (e) PERF-042：least-loaded partition ownership（首见分区给最闲 worker，
        同 partition 恒同一 worker）；
    (f) PERF-068：materialize_sharded batch 不再 run_many 攒全量 results dict。
"""
from __future__ import annotations

import inspect
import os
import threading
import time

import pandas as pd
import pytest

from api import rank, ts_mean, ts_std
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _engine(dates: int = 8):
    dts = pd.bdate_range("2024-01-02", periods=dates)
    idx = pd.MultiIndex.from_product(
        [dts, ["A", "B"]], names=["timestamp", "instrument"]
    )
    close = pd.Series([float(i) for i in range(len(idx))], index=idx)
    vol = pd.Series([float(i * 2) for i in range(len(idx))], index=idx)
    return FactorEngine(
        data_source=InMemorySeriesSource({"close": close, "volume": vol}),
        backend=PandasBackend(),
    )


def _factors():
    return [
        Factor(name="f_mean5", expr=ts_mean(col("close"), 5)),
        Factor(name="f_std20", expr=ts_std(col("close"), 3)),
        Factor(name="f_rank", expr=rank(col("close"))),
    ]


@pytest.fixture
def engine():
    return _engine()


def _run_output(engine, factor):
    run = engine.run(factor)
    return {
        "factor": factor,
        "analysis": run["analysis"],
        "result": run["result"],
    }


# ---------------------------------------------------------------------------
# (a) Gate-02：O(1) 查找
# ---------------------------------------------------------------------------


def test_gate02_static_no_oN2_lookup_in_writer():
    """静态：writer 热循环禁止 ``list(ids).index`` / ``next((f for f in factors``。"""
    src = inspect.getsource(FactorEngine.materialize_many_fast)
    assert "list(ids).index" not in src
    assert "next((f for f in factors" not in src
    # dict 预建必须存在
    assert "factor_by_name" in src
    assert "factor_id_by_name" in src


def test_gate02_custom_factor_id_mapping_dynamic(tmp_path, engine, monkeypatch):
    """PERF-038 动态：writer 用 ``factor_id_by_name`` dict 携带真实 factor_id——
    ``factor_id != factor.name`` 时写入侧拿到的必须是自定义 ID。

    （``builtins.list.index`` 是 CPython 不可变 C 类型方法，无法 monkeypatch；
    O(1) 由 ``test_gate02_static_no_oN2_lookup_in_writer`` 静态断言保证。）
    """
    from runtime import materialize_batch

    captured: list[tuple[str, str]] = []

    def _recorder(engine_, items, generation=None, **kwargs):
        captured.extend((i.factor.name, i.factor_id) for i in items)
        return {
            "materializations": {
                i.factor.name: {"rows_written": 1, "factor_id": i.factor_id}
                for i in items
            },
            "counters": {"batch_write_transaction_count": 1},
        }

    monkeypatch.setattr(materialize_batch, "execute_materialize_batch", _recorder)
    custom_ids = ["custom_a", "custom_b", "custom_c"]
    os.environ["FACTOR_ENGINE_HYBRID_FORCE"] = "thread"
    try:
        out = engine.materialize_many_fast(
            _factors(),
            factor_ids=custom_ids,
            write_results=True,
            materialize_kwargs={
                "lake_root": str(tmp_path / "lake"),
                "write_target": "local",
            },
        )
    finally:
        os.environ.pop("FACTOR_ENGINE_HYBRID_FORCE", None)
    assert out["done"] >= 3
    assert dict(captured) == {
        "f_mean5": "custom_a",
        "f_std20": "custom_b",
        "f_rank": "custom_c",
    }, "writer 必须用 dict 携带真实 factor_id（factor_id != factor.name）"


# ---------------------------------------------------------------------------
# (b) execute_materialize_batch == N× execute_materialize
# ---------------------------------------------------------------------------


def _read_factor_long(lake_root, factor_id):
    from security.factor_id import factor_dir_for

    fdir = factor_dir_for(lake_root, factor_id)
    parts = sorted(fdir.rglob("data.parquet"))
    assert parts, f"no data.parquet under {fdir}"
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # 长表列名：datetime / asset（materializer 统一命名，非原始 instrument）。
    df = df.sort_values(["datetime", "asset"]).reset_index(drop=True)
    return df


def _serial_materialize(engine, factor, output, lake_root):
    from runtime.materialize_service import execute_materialize

    return execute_materialize(
        engine,
        factor,
        output,
        target="local",
        lake_root=lake_root,
        staging_dataset="factor_lake_staging",
        factor_id=factor.name,
        author=None,
        frequency=None,
        description=None,
        expression=None,
        dq_check=False,
        dq_strict=True,
        dq_thresholds=None,
        write_metadata=True,
        data_source_config=None,
        resume_materialize=False,
        isolate_partition_failures=True,
        preserve_invalid_rows=False,
        value_dtype="float32",
        clickhouse_table=None,
        ch_ensure_table=True,
        ch_host=None,
        ch_port=None,
        ch_database=None,
        ch_username=None,
        ch_password=None,
        ch_secure=None,
        storage_format="long",
        partition_columns=None,
    )


def test_execute_materialize_batch_parity(tmp_path, engine):
    from runtime.materialize_batch import (
        MaterializeItem,
        execute_materialize_batch,
    )

    factors = _factors()
    serial_lake = tmp_path / "serial_lake"
    batch_lake = tmp_path / "batch_lake"

    # serial reference：逐因子 execute_materialize
    serial_summary = {}
    for f in factors:
        out = _serial_materialize(engine, f, _run_output(engine, f), serial_lake)
        serial_summary[f.name] = out["materialization"]

    # batch：单次 execute_materialize_batch
    items = [
        MaterializeItem(
            factor=f,
            output=_run_output(engine, f),
            factor_id=f.name,
            options={"write_target": "local", "value_dtype": "float32"},
        )
        for f in factors
    ]
    batch_out = execute_materialize_batch(
        engine,
        items,
        shared_options={
            "lake_root": batch_lake,
            "write_target": "local",
            "value_dtype": "float32",
            "storage_format": "long",
        },
    )
    assert set(batch_out["materializations"].keys()) == {f.name for f in factors}

    # 逐因子 read-back parity：values + index（含 dtype）
    for f in factors:
        sdf = _read_factor_long(serial_lake, f.name)
        bdf = _read_factor_long(batch_lake, f.name)
        assert list(sdf.columns) == list(bdf.columns)
        assert list(sdf.dtypes) == list(bdf.dtypes), f"{f.name} dtype mismatch"
        assert len(sdf) == len(bdf), f"{f.name} row count mismatch"
        assert (sdf[["datetime", "asset"]].values == bdf[["datetime", "asset"]].values).all(), (
            f"{f.name} index mismatch"
        )
        a = sdf["value"].fillna(0.0).to_numpy()
        b = bdf["value"].fillna(0.0).to_numpy()
        assert (a == b).all(), f"{f.name} value mismatch"
        assert (sdf["is_valid"].fillna(0).to_numpy() == bdf["is_valid"].fillna(0).to_numpy()).all(), (
            f"{f.name} is_valid mismatch"
        )

    # catalog parity：factor_version + 依赖边
    from runtime.dependency_catalog import DependencyCatalog
    from storage.materializer import ParquetMaterializer

    serial_cat = ParquetMaterializer(lake_root=serial_lake).catalog
    batch_cat = ParquetMaterializer(lake_root=batch_lake).catalog
    for f in factors:
        si = serial_cat.get_factor_info(f.name)
        bi = batch_cat.get_factor_info(f.name)
        assert si is not None and bi is not None
        assert si["factor_version"] == bi["factor_version"], f"{f.name} factor_version mismatch"
        ser_edges = DependencyCatalog(serial_cat).get_factor_dependency_edges(f.name)
        bat_edges = DependencyCatalog(batch_cat).get_factor_dependency_edges(f.name)
        assert sorted(e.dependency_id for e in ser_edges) == sorted(
            e.dependency_id for e in bat_edges
        ), f"{f.name} dependency edges mismatch"


# ---------------------------------------------------------------------------
# (c) Gate-03：batch transaction count < item count
# ---------------------------------------------------------------------------


def test_batch_transaction_count_lt_item_count(tmp_path, engine):
    from runtime.materialize_batch import (
        GenerationTransaction,
        MaterializeItem,
        execute_materialize_batch,
    )

    factors = _factors()
    items = [
        MaterializeItem(
            factor=f,
            output=_run_output(engine, f),
            factor_id=f.name,
            options={"write_target": "local"},
        )
        for f in factors
    ]
    gen = GenerationTransaction(generation_id="gen-batch-txn-test")
    out = execute_materialize_batch(
        engine,
        items,
        generation=gen,
        shared_options={"lake_root": tmp_path / "lake", "write_target": "local"},
    )
    counters = out["counters"]
    assert counters["item_count"] == 3
    assert counters["batch_write_transaction_count"] == 1
    assert counters["batch_write_transaction_count"] < counters["item_count"]
    assert counters["manifest_writes"] == 3
    assert gen.published_items == 3
    # 输出带 generation
    assert out["generation"] == "gen-batch-txn-test"


def test_materialize_many_fast_batch_transaction_count_lt_items(tmp_path):
    """Gate-03 主路径：materialize_many_fast 默认 count-batch 64 → writer 攒批，
    batch_write_transaction_count < factor_count。"""
    eng = _engine(dates=6)
    os.environ["FACTOR_ENGINE_HYBRID_FORCE"] = "thread"
    try:
        out = eng.materialize_many_fast(
            _factors(),
            materialize_kwargs={
                "lake_root": str(tmp_path / "lake"),
                "write_target": "local",
                "value_dtype": "float32",
            },
        )
    finally:
        os.environ.pop("FACTOR_ENGINE_HYBRID_FORCE", None)
    assert out["batch_write_transaction_count"] >= 1
    assert out["batch_write_transaction_count"] < len(_factors())
    assert out["generation"].startswith("batch-")
    assert "materializations" in out


# ---------------------------------------------------------------------------
# (d) flush 谓词：bytes + age + count
# ---------------------------------------------------------------------------


def test_flush_predicate_bytes_age_count():
    from runtime.streaming_result_sink import _should_flush

    # 旧 count-only 默认：target=None/age=None → batch_size 主导
    assert _should_flush(1, 0, 0.0, batch_size=1, target_batch_bytes=None, max_batch_age_s=None) is True
    assert _should_flush(1, 0, 0.0, batch_size=5, target_batch_bytes=None, max_batch_age_s=None) is False
    # bytes 触发
    assert _should_flush(2, 70, 0.0, batch_size=100, target_batch_bytes=64, max_batch_age_s=None) is True
    # age 触发
    assert _should_flush(2, 1, 2.0, batch_size=100, target_batch_bytes=None, max_batch_age_s=1.0) is True
    # count 仍触发
    assert _should_flush(10, 1, 0.0, batch_size=5, target_batch_bytes=9999, max_batch_age_s=999.0) is True
    # 均未触发
    assert _should_flush(2, 5, 0.1, batch_size=5, target_batch_bytes=64, max_batch_age_s=1.0) is False
    # bytes 与 age 组合（任一）
    assert _should_flush(2, 100, 0.1, batch_size=5, target_batch_bytes=64, max_batch_age_s=1.0) is True


def test_sink_flushes_by_bytes():
    from runtime.streaming_result_sink import StreamingResultSink

    batches: list[list[str]] = []
    lock = threading.Lock()

    def writer(batch):
        with lock:
            batches.append([i.name for i in batch])

    sink = StreamingResultSink(
        writer=writer,
        batch_size=100,
        target_batch_bytes=50,
        writer_threads=1,
    )
    sink.start()
    # 连续提交两个 30B item；worker 内层 0.05s 收集窗口内必能拿到 b → 60B>=50B
    # 触发 bytes flush，写成一个 2-item batch。
    sink.submit("a", b"x" * 30, bytes=30)
    sink.submit("b", b"y" * 30, bytes=30)
    sink.finish()
    with lock:
        names = [n for b in batches for n in b]
    assert sorted(names) == ["a", "b"]
    assert any(len(b) >= 2 for b in batches), f"expected byte-triggered 2-item batch, got {batches}"


def test_sink_default_stays_count_only():
    from runtime.streaming_result_sink import StreamingResultSink

    batches: list[int] = []

    def writer(batch):
        batches.append(len(batch))

    # 默认（target_batch_bytes=None, max_batch_age_s=None）→ batch_size=1 逐条写
    sink = StreamingResultSink(writer=writer, batch_size=1, writer_threads=1)
    sink.start()
    sink.submit("a", object())
    sink.submit("b", object())
    sink.finish()
    assert sorted(batches) == [1, 1]


# ---------------------------------------------------------------------------
# (e) PERF-042：least-loaded ownership
# ---------------------------------------------------------------------------


def test_least_loaded_routing_pins_partition():
    from runtime.streaming_result_sink import ResultItem, StreamingResultSink

    sink = StreamingResultSink(
        writer=lambda b: None,
        writer_threads=2,
        partition_key=lambda i: i.meta["part"],
    )
    # 不 start()（无 writer 线程），手动给 worker-0 队列制造负载。
    sink._worker_queues[0].put(ResultItem("x", b"x" * 100, bytes=100))
    # worker-0 100B，worker-1 0B → least loaded = worker 1
    assert sink._least_loaded_writer() == 1
    # 首见分区 p1 → worker 1（最闲）
    assert sink._route_worker(ResultItem("a", object(), meta={"part": "p1"})) == 1
    # p1 已 pin → 恒 worker 1（同 partition 永不并发写）
    assert sink._route_worker(ResultItem("b", object(), meta={"part": "p1"})) == 1
    assert sink._partition_owners["p1"] == 1
    # 新分区 p2 → 仍最闲 worker 1（worker-0 仍 100B）
    assert sink._route_worker(ResultItem("c", object(), meta={"part": "p2"})) == 1
    assert sink._partition_owners["p2"] == 1
    # 已 pin 的 p1 不因负载变化改投其他 worker
    sink._worker_queues[1].put(ResultItem("y", b"y" * 1000, bytes=1000))
    assert sink._route_worker(ResultItem("d", object(), meta={"part": "p1"})) == 1


def test_auto_writer_threads_backcompat():
    from runtime.streaming_result_sink import StreamingResultSink

    # 无 partition_key → 单 worker（保持旧默认）
    sink = StreamingResultSink(writer=lambda b: None)
    assert sink._writer_threads == 1
    # 显式覆盖
    sink2 = StreamingResultSink(writer=lambda b: None, writer_threads=3)
    assert sink2._writer_threads == 3
    # partition-aware → min(2, cpu) >= 1
    sink3 = StreamingResultSink(
        writer=lambda b: None, partition_key=lambda i: i.meta["p"]
    )
    assert 1 <= sink3._writer_threads <= 2


# ---------------------------------------------------------------------------
# (f) PERF-068：materialize_sharded batch 走流式 sink，不 run_many 攒全量
# ---------------------------------------------------------------------------


def test_materialize_sharded_streaming_no_full_results_dict(tmp_path, engine, monkeypatch):
    called: list = []

    def _boom(*a, **k):
        called.append(a)
        raise AssertionError("run_many must not be called in streaming sharded batch path")

    monkeypatch.setattr(FactorEngine, "run_many", _boom)
    os.environ["FACTOR_ENGINE_HYBRID_FORCE"] = "thread"
    try:
        out = engine.materialize_sharded(
            _factors(),
            factor_ids=[f.name for f in _factors()],
            shard_by="factor_id",
            shard_index=0,
            shard_count=1,
            run_many_batch=True,
            lake_root=str(tmp_path / "lake"),
            write_target="local",
        )
    finally:
        os.environ.pop("FACTOR_ENGINE_HYBRID_FORCE", None)
    assert not called, "materialize_sharded 应走 materialize_many_fast 流式 sink，而非 run_many"
    assert set(out["materializations"].keys()) == {"f_mean5", "f_std20", "f_rank"}
    for m in out["materializations"].values():
        assert m.get("shard_index") == 0
        assert m.get("shard_count") == 1
        assert m.get("shard_by") == "factor_id"
        assert m.get("batched_run") is True

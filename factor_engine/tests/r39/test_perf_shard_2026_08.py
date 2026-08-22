# -*- coding: utf-8 -*-
"""R39-P0-PERF-024/025/026 行为测试 —— 真实探针，不是 grep/module-existence。

覆盖：
  - PERF-024：ArrowSpoolRef round-trip == 旧 parquet spool（同数据逐值一致）；
    row_count / schema 在 spool 时记录；Arrow 是 SpooledShard 子类（兼容既有
    isinstance 探针）。
  - PERF-025 (Gate-07)：已证明「每片排序 + 范围不重叠 + merge_order 正确」→
    单次 concat，``shard_concat_sort_count == 1``，输出 == 参考 K-merge；
    重叠 / 未排序 → 回退 ordered merge，输出仍 == 参考；``merge_sorted_streams``
    k-way merge 4 片 == 参考。
  - PERF-026：ShardMergeMode 选择——CONCAT_ONLY-eligible 分组不产生 ORDERED_MERGE；
    DIRECT_DURABLE_APPEND 只提交 manifest（concat_sort_count == 0）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.shard_execution_plan import (
    SHARD_TIME,
    ShardDescriptor,
    ShardExecutionPlan,
    ShardMergeContract,
    ShardMergeMode,
)
from runtime.shard_executor import (
    ArrowSpoolRef,
    ShardDirectWriteManifest,
    ShardExecutor,
    SpooledShard,
    choose_shard_merge_mode,
    merge_sorted_streams,
    verify_shards_sorted_nonoverlapping,
)


def _series(lo: str, hi: str, base: float = 0.0) -> pd.Series:
    """按 (timestamp, instrument) MultiIndex 的时间块 Series（内部有序）。"""
    idx = pd.MultiIndex.from_product(
        [pd.date_range(lo, hi, freq="D"), ["A001", "A002", "A003"]],
        names=["timestamp", "instrument"],
    )
    vals = [base + float(i % 5) for i in range(len(idx))]
    return pd.Series(vals, index=idx)


def _plan(descriptors, merge_policy: ShardMergeContract | None = None) -> ShardExecutionPlan:
    return ShardExecutionPlan(
        original_task_id="perf:root",
        dimension=SHARD_TIME,
        shards=tuple(descriptors),
        merge_policy=merge_policy or ShardMergeContract(),
        merge_task_id="perf:root:merge",
        shard_task_ids=tuple(d.shard_id for d in descriptors),
        per_shard_peak_bytes=1,
        original_peak_bytes=1024,
    )


def _ref_ordered_merge(chunks: list[pd.Series]) -> pd.Series:
    """参考实现：逐片 concat+sort（旧 K-concat+sort 路径）。"""
    acc: pd.Series | None = None
    for c in chunks:
        if acc is None:
            acc = c
        else:
            acc = pd.concat([acc, c]).sort_index()
    return acc


# ---------------------------------------------------------------------------
# PERF-024: Arrow spool round-trip == parquet spool
# ---------------------------------------------------------------------------


def test_arrow_spool_roundtrip_equals_parquet():
    s = _series("2024-01-01", "2024-01-31")
    ex_arrow = ShardExecutor()
    ex_parquet = ShardExecutor()

    ref_arrow = ex_arrow.spool_or_keep(s, spool_threshold=1, spool_format="arrow")
    ref_parquet = ex_parquet.spool_or_keep(s, spool_threshold=1, spool_format="parquet")

    assert isinstance(ref_arrow, ArrowSpoolRef)
    assert isinstance(ref_arrow, SpooledShard), "ArrowSpoolRef must stay a SpooledShard"
    assert isinstance(ref_parquet, SpooledShard)
    assert not isinstance(ref_parquet, ArrowSpoolRef)

    # row_count / schema 在 spool 时记录
    assert ref_arrow.row_count == len(s)
    assert ref_arrow.schema is not None
    assert ref_arrow.n_bytes > 0
    assert ref_arrow.spool_format == "arrow"
    assert ref_parquet.spool_format == "parquet"

    loaded_arrow = ex_arrow.load_partial(ref_arrow)
    loaded_parquet = ex_parquet.load_partial(ref_parquet)

    # Arrow round-trip 与原始一致
    pd.testing.assert_series_equal(loaded_arrow, s, check_names=False)
    # Arrow round-trip 与 parquet round-trip 逐值一致（同数据）
    pd.testing.assert_series_equal(
        loaded_arrow.sort_index(), loaded_parquet.sort_index(), check_names=False
    )


def test_arrow_spool_dataframe_and_singlelevel():
    # 双列 DataFrame → Arrow 往返后仍为 DataFrame（单列会按 spool 约定还原为 Series）
    df = pd.DataFrame(
        {"v": [1.0, 2.0, float("nan"), 4.0], "w": [5.0, 6.0, 7.0, 8.0]},
        index=pd.date_range("2024-02-01", periods=4, name="timestamp"),
    )
    ex = ShardExecutor()
    ref = ex.spool_or_keep(df, spool_threshold=1, spool_format="arrow")
    assert isinstance(ref, ArrowSpoolRef)
    loaded = ex.load_partial(ref)
    assert isinstance(loaded, pd.DataFrame)
    # Arrow spool 不保证保留 index.freq 元数据（parquet 同样不保留）→ check_freq=False
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)
    # NaN 保留
    assert loaded["v"].isna().sum() == 1
    # 单列 spool → Series（与 parquet 同约定）
    s = pd.Series([1.0, 2.0], index=pd.date_range("2024-03-01", periods=2, name="timestamp"))
    ref_s = ex.spool_or_keep(s, spool_threshold=1, spool_format="arrow")
    loaded_s = ex.load_partial(ref_s)
    assert isinstance(loaded_s, pd.Series)


def test_arrow_spool_fallback_to_parquet_on_nonconvertible():
    # numpy ndarray 无标签 → 不可 Arrow 转换（且 parquet 可兜底）
    arr = [1.0, 2.0, 3.0]
    ex = ShardExecutor()
    ref = ex.spool_or_keep(arr, spool_threshold=1, spool_format="auto")
    # 不可转换 → 回退 parquet，仍为 SpooledShard
    assert isinstance(ref, SpooledShard)
    assert not isinstance(ref, ArrowSpoolRef)
    assert ref.spool_format == "parquet"


# ---------------------------------------------------------------------------
# PERF-025 (Gate-07): sorted + non-overlapping → single concat
# ---------------------------------------------------------------------------


def test_sorted_nonoverlap_single_concat_matches_reference():
    chunks = [
        _series("2024-01-01", "2024-01-10"),
        _series("2024-01-11", "2024-01-20"),
        _series("2024-01-21", "2024-01-31"),
    ]
    descs = [
        ShardDescriptor(shard_id="perf:shard:0", dimension=SHARD_TIME,
                        input_slice=("2024-01-01", "2024-01-10"),
                        warmup_slice=("2024-01-01", "2024-01-10"),
                        output_slice=("2024-01-01", "2024-01-10"), merge_order=0),
        ShardDescriptor(shard_id="perf:shard:1", dimension=SHARD_TIME,
                        input_slice=("2024-01-11", "2024-01-20"),
                        warmup_slice=("2024-01-11", "2024-01-20"),
                        output_slice=("2024-01-11", "2024-01-20"), merge_order=1),
        ShardDescriptor(shard_id="perf:shard:2", dimension=SHARD_TIME,
                        input_slice=("2024-01-21", "2024-01-31"),
                        warmup_slice=("2024-01-21", "2024-01-31"),
                        output_slice=("2024-01-21", "2024-01-31"), merge_order=2),
    ]
    plan = _plan(descs)
    partials = {d.shard_id: c for d, c in zip(descs, chunks)}

    # 证明必须通过
    contract = plan.merge_policy
    assert verify_shards_sorted_nonoverlapping(chunks, contract) is True

    ex = ShardExecutor()
    merged = ex.execute_merge(None, None, plan, partials)

    ref = _ref_ordered_merge(chunks)
    # 严格一致（值 / index / 顺序 / dtype）——不排序比较
    pd.testing.assert_series_equal(merged, ref)
    assert ex.summary()["shard_concat_sort_count"] == 1, "proven path must single-concat"


def test_sorted_nonoverlap_concat_only_events():
    descs = [
        ShardDescriptor(shard_id="s0", dimension=SHARD_TIME,
                        input_slice=("2024-03-01", "2024-03-10"),
                        output_slice=("2024-03-01", "2024-03-10"), merge_order=0),
        ShardDescriptor(shard_id="s1", dimension=SHARD_TIME,
                        input_slice=("2024-03-11", "2024-03-20"),
                        output_slice=("2024-03-11", "2024-03-20"), merge_order=1),
    ]
    chunks = [_series("2024-03-01", "2024-03-10"), _series("2024-03-11", "2024-03-20")]
    plan = _plan(descs)
    ex = ShardExecutor()
    ex.execute_merge(None, None, plan, {d.shard_id: c for d, c in zip(descs, chunks)})
    assert any("merge_mode:concat_only" in e for e in ex.summary()["events"])
    assert not any("merge_mode:ordered_merge" in e for e in ex.summary()["events"])


# ---------------------------------------------------------------------------
# PERF-025: overlapping / unsorted → fallback to ordered merge, still correct
# ---------------------------------------------------------------------------


def test_overlapping_shards_fallback_ordered_merge():
    # 每片内部有序但范围交错/重叠（跨片键交错，无重复键）→ 证明失败 → 回退 ordered
    # merge，输出仍 == 参考 K-merge。
    base = pd.date_range("2024-04-01", periods=9, name="timestamp")
    c0 = pd.Series([0.0, 3.0, 6.0], index=base[[0, 3, 6]])
    c1 = pd.Series([1.0, 4.0, 7.0], index=base[[1, 4, 7]])  # 与 c0 交错
    c2 = pd.Series([2.0, 5.0, 8.0], index=base[[2, 5, 8]])  # 与 c0/c1 交错
    chunks = [c0, c1, c2]
    descs = [
        ShardDescriptor(shard_id="o0", dimension=SHARD_TIME,
                        input_slice=("2024-04-01", "2024-04-09"),
                        output_slice=("2024-04-01", "2024-04-09"), merge_order=0),
        ShardDescriptor(shard_id="o1", dimension=SHARD_TIME,
                        input_slice=("2024-04-01", "2024-04-09"),
                        output_slice=("2024-04-01", "2024-04-09"), merge_order=1),
        ShardDescriptor(shard_id="o2", dimension=SHARD_TIME,
                        input_slice=("2024-04-01", "2024-04-09"),
                        output_slice=("2024-04-01", "2024-04-09"), merge_order=2),
    ]
    plan = _plan(descs)
    assert verify_shards_sorted_nonoverlapping(chunks, plan.merge_policy) is False

    ex = ShardExecutor()
    merged = ex.execute_merge(None, None, plan, {d.shard_id: c for d, c in zip(descs, chunks)})
    ref = _ref_ordered_merge(chunks)
    pd.testing.assert_series_equal(merged, ref)
    # 回退路径逐片 concat+sort
    assert ex.summary()["shard_concat_sort_count"] == len(chunks)
    assert any("merge_mode:ordered_merge" in e for e in ex.summary()["events"])


def test_unsorted_shard_fallback_ordered_merge():
    # 构造片内未排序 → 证明失败 → 回退
    idx = pd.date_range("2024-05-01", periods=6, name="timestamp")
    unsorted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx[::-1])  # 倒序
    c1 = pd.Series([1.5, 2.5], index=pd.date_range("2024-06-01", periods=2, name="timestamp"))
    chunks = [unsorted, c1]
    descs = [
        ShardDescriptor(shard_id="u0", dimension=SHARD_TIME,
                        input_slice=("2024-05-01", "2024-05-06"),
                        output_slice=("2024-05-01", "2024-05-06"), merge_order=0),
        ShardDescriptor(shard_id="u1", dimension=SHARD_TIME,
                        input_slice=("2024-06-01", "2024-06-02"),
                        output_slice=("2024-06-01", "2024-06-02"), merge_order=1),
    ]
    plan = _plan(descs)
    assert verify_shards_sorted_nonoverlapping(chunks, plan.merge_policy) is False

    ex = ShardExecutor()
    merged = ex.execute_merge(None, None, plan, {d.shard_id: c for d, c in zip(descs, chunks)})
    ref = _ref_ordered_merge(chunks)
    pd.testing.assert_series_equal(merged, ref)


# ---------------------------------------------------------------------------
# PERF-025: k-way streaming merge
# ---------------------------------------------------------------------------


def test_kway_merge_nonoverlapping_matches_reference():
    chunks = [
        _series("2024-07-01", "2024-07-07"),
        _series("2024-07-08", "2024-07-14"),
        _series("2024-07-15", "2024-07-21"),
        _series("2024-07-22", "2024-07-28"),
    ]
    merged = merge_sorted_streams(chunks)
    ref = pd.concat(chunks).sort_index()
    pd.testing.assert_series_equal(merged, ref)


def test_kway_merge_interleaved_sorted_matches_reference():
    # 3 片各自排序、跨片交错但不重复键 → k-way 归并 == 参考
    base = pd.date_range("2024-08-01", periods=9, name="timestamp")
    c0 = pd.Series([0.0, 3.0, 6.0], index=base[[0, 3, 6]])
    c1 = pd.Series([1.0, 4.0, 7.0], index=base[[1, 4, 7]])
    c2 = pd.Series([2.0, 5.0, 8.0], index=base[[2, 5, 8]])
    chunks = [c0, c1, c2]
    merged = merge_sorted_streams(chunks)
    ref = pd.concat(chunks).sort_index()
    pd.testing.assert_series_equal(merged, ref)


def test_kway_merge_rejects_unsorted_chunk():
    idx = pd.date_range("2024-09-01", periods=3, name="timestamp")
    bad = pd.Series([1.0, 2.0, 3.0], index=idx[::-1])
    good = pd.Series([4.0], index=pd.date_range("2024-09-04", periods=1, name="timestamp"))
    with pytest.raises(ValueError):
        merge_sorted_streams([bad, good])


# ---------------------------------------------------------------------------
# PERF-026: ShardMergeMode selection
# ---------------------------------------------------------------------------


def test_mode_selection_concat_only_for_proven_group():
    chunks = [_series("2024-10-01", "2024-10-10"), _series("2024-10-11", "2024-10-20")]
    contract = ShardMergeContract()
    mode = choose_shard_merge_mode(contract, chunks)
    assert mode is ShardMergeMode.CONCAT_ONLY, "proven group must not produce ORDERED_MERGE"


def test_mode_selection_ordered_for_overlap():
    chunks = [_series("2024-11-01", "2024-11-10"), _series("2024-11-05", "2024-11-15")]
    contract = ShardMergeContract()
    assert choose_shard_merge_mode(contract, chunks) is ShardMergeMode.ORDERED_MERGE


def test_mode_selection_direct_append_requested():
    chunks = [_series("2024-12-01", "2024-12-10")]
    contract = ShardMergeContract(merge_mode="direct_durable_append")
    assert choose_shard_merge_mode(contract, chunks) is ShardMergeMode.DIRECT_DURABLE_APPEND


def test_direct_append_merge_returns_manifest_no_rebuild():
    descs = [
        ShardDescriptor(shard_id="d0", dimension=SHARD_TIME,
                        input_slice=("2024-12-01", "2024-12-10"),
                        output_slice=("2024-12-01", "2024-12-10"), merge_order=0),
    ]
    chunks = [_series("2024-12-01", "2024-12-10")]
    contract = ShardMergeContract(merge_mode="direct_durable_append")
    plan = _plan(descs, contract)
    ex = ShardExecutor()
    out = ex.execute_merge(None, None, plan, {descs[0].shard_id: chunks[0]})
    assert isinstance(out, ShardDirectWriteManifest)
    assert out.fragments == tuple(chunks)
    assert out.shard_ids == ("d0",)
    assert out.plan_metadata["merge_mode"] == "direct_durable_append"
    # 直接落盘 → 不重建内存结果 → concat_sort_count == 0
    assert ex.summary()["shard_concat_sort_count"] == 0


def test_mode_default_is_ordered_merge():
    assert ShardMergeContract().merge_mode == "ordered_merge"


# ---------------------------------------------------------------------------
# 随机 + 对抗输入：merge 输出与参考 K-concat+sort 严格一致
# ---------------------------------------------------------------------------


def test_random_shards_parity_with_reference():
    rng = np.random.default_rng(42)
    pool = pd.date_range("2024-01-01", periods=40, freq="D")
    for _ in range(6):
        n = int(rng.integers(1, 6))
        chunks: list[pd.Series] = []
        for _i in range(n):
            size = int(rng.integers(2, 10))
            keys = sorted(rng.choice(pool, size=size, replace=False).tolist())
            if rng.random() < 0.3:
                keys = keys[::-1]  # 30% 概率片内未排序（对抗）
            vals = rng.normal(size=len(keys))
            chunks.append(pd.Series(vals, index=pd.DatetimeIndex(keys, name="timestamp")))
        descs = [
            ShardDescriptor(
                shard_id=f"r{i}", dimension=SHARD_TIME,
                input_slice=("2024-01-01", "2024-02-09"),
                output_slice=("2024-01-01", "2024-02-09"),
                merge_order=i,
            )
            for i in range(n)
        ]
        plan = _plan(descs)
        partials = {d.shard_id: c for d, c in zip(descs, chunks)}
        ex = ShardExecutor()
        try:
            merged = ex.execute_merge(None, None, plan, partials)
        except RuntimeError:
            # duplicate_key_policy=error：重复键必须真实存在于原始 concat 结果
            raw = pd.concat(chunks).sort_index()
            assert raw.index.duplicated().any(), "raise must be justified by real duplicates"
            continue
        ref = _ref_ordered_merge(chunks)
        pd.testing.assert_series_equal(merged, ref)

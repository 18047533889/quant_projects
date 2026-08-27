# -*- coding: utf-8 -*-
"""P11+P12: FeatureBlock 物理布局 + MaterializationTier + writer/reader 接线。

验证 (a) 1000 因子 → 少量 block（不是 1000 文件）；(b) manifest 映射
FactorID→BlockID→Column 且回读一致；(c) float32 dtype 契约；(d) tier 行为
（TIER0 丢值 / TIER1 留窗 / TIER2 全历史）；(e) 块大小动态 + 尊重 writer 内存上限。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from factor_engine.runtime.feature_block import (
    FACTOR_VALUE_DTYPE,
    MAX_COLUMNS_PER_BLOCK,
    MIN_COLUMNS_PER_BLOCK,
    BlockManifest,
    FeatureBlockReader,
    FeatureBlockWriter,
    block_bytes,
    choose_columns_per_block,
    load_manifest,
    make_sink_writer,
)
from factor_engine.runtime.materialization_tier import (
    DEFAULT_HISTORY_DAYS,
    TIER0,
    TIER1,
    TIER2,
    FactorMaterialization,
    history_days_for,
    resolve_tier,
    tier_from_label,
    trim_to_window,
)


def _write_n(tmp_path: Path, n: int, *, cols_per_block: int | None = None,
             rows: int = 2500, dtype: str = FACTOR_VALUE_DTYPE) -> tuple[FeatureBlockWriter, list[float]]:
    w = FeatureBlockWriter(
        tmp_path / "feat",
        columns_per_block=cols_per_block,
        dtype=dtype,
    )
    truth: list[float] = []
    for i in range(n):
        arr = np.arange(rows, dtype=np.float64) + float(i)  # 非 float32 源，验证 cast
        truth.append(float(i))
        w.add(f"factor_{i:04d}", arr)
    w.finish()
    return w, truth


# ---- (a) 1000 因子 → 少量 block，不是 1000 文件 --------------------------------
def test_1000_factors_group_into_small_block_count(tmp_path):
    n = 1000
    cols_per_block = 256
    w, _ = _write_n(tmp_path, n, cols_per_block=cols_per_block, rows=2500)
    blocks = list((tmp_path / "feat").rglob("block_*.npy"))
    expected_blocks = math.ceil(n / cols_per_block)  # ~4
    assert len(blocks) == expected_blocks, f"got {len(blocks)} files, expected ~{expected_blocks}"
    assert len(blocks) < n  # 绝非 1000 文件
    assert w.block_count() == expected_blocks
    # manifest 只有 1 个（元数据与值分离）
    manifests = list((tmp_path / "feat").rglob("manifest.json"))
    assert len(manifests) == 1


# ---- (b) manifest 映射 FactorID→BlockID→Column + 回读一致 ----------------------
def test_manifest_maps_and_roundtrip(tmp_path):
    n = 300
    cols_per_block = 128
    rows = 1000
    w, truth = _write_n(tmp_path, n, cols_per_block=cols_per_block, rows=rows)

    man = w.manifest
    assert isinstance(man, BlockManifest)
    assert man.factor_count if hasattr(man, "factor_count") else len(man.factor_to_block) == n

    # 每因子都映射到 block + 列
    for i in range(n):
        fid = f"factor_{i:04d}"
        assert fid in man.factor_to_block
        resolved = man.resolve(fid)
        assert len(resolved) == 1
        partition, spec, column = resolved[0]
        assert spec.column_map[fid] == column
        assert spec.factor_ids[column] == fid

    # 回读与写入一致
    r = FeatureBlockReader(tmp_path / "feat")
    assert r.manifest == man or r.manifest.to_dict() == man.to_dict()
    for i in range(n):
        fid = f"factor_{i:04d}"
        vals = r.read_factor_full(fid)
        assert vals.shape == (rows,)
        expected = np.arange(rows, dtype=np.float64) + float(i)
        np.testing.assert_array_equal(vals, expected.astype(np.float32))


# ---- (c) float32 dtype 契约 ---------------------------------------------------
def test_float32_dtype_contract(tmp_path):
    n = 50
    rows = 1000
    w, _ = _write_n(tmp_path, n, cols_per_block=64, rows=rows)
    man = w.manifest
    assert man.dtype == "float32"

    r = FeatureBlockReader(tmp_path / "feat")
    # 每 block 值文件回读都是 float32
    for block_id in man.blocks:
        arr = r.read_block(block_id)
        assert arr.dtype == np.dtype("float32"), arr.dtype

    fid = "factor_0003"
    vals = r.read_factor_full(fid)
    assert vals.dtype == np.dtype("float32")
    # 源是 float64，但落盘/回读必须是 float32
    assert vals.dtype != np.dtype("float64")


# ---- (d) tier 行为 ------------------------------------------------------------
def test_tier_resolve_defaults_to_tier0():
    # agent mining 默认：候选 → TIER0（绝不自动全历史）
    assert resolve_tier(is_candidate=True) == TIER0
    assert resolve_tier(is_candidate=True, is_production=True) == TIER0
    # 非候选 + 生产 → TIER2；研究窗 → TIER1
    assert resolve_tier(is_candidate=False, is_production=True) == TIER2
    assert resolve_tier(is_candidate=False, research_window=250) == TIER1
    # 显式优先
    assert resolve_tier(explicit=TIER2, is_candidate=True) == TIER2
    with pytest.raises(ValueError):
        resolve_tier(explicit="NOPE")


def test_tier0_discards_values(tmp_path):
    w = FeatureBlockWriter(tmp_path / "feat", columns_per_block=128)
    m = FactorMaterialization(factor_id="f", tier=TIER0, fingerprint="fp1")
    # TIER0 永不物理落值
    with pytest.raises(ValueError):
        m.persist(np.arange(100), block_id="b", column=0)
    assert m.keep_values() is False
    assert m.persisted is False
    # TIER0 值丢弃：只留 fingerprint/evaluation/definition
    assert m.fingerprint == "fp1"
    assert m.values is None
    # 元数据仍落 manifest（fingerprint + tier 标签）
    meta = m.to_metadata()
    assert meta["fingerprint"] == "fp1"
    assert tier_from_label(meta["tier"]) == TIER0


def test_tier1_keeps_window(tmp_path):
    rows = 1000
    history_days = 250
    m = FactorMaterialization(factor_id="f", tier=TIER1, history_days=history_days)
    full = np.arange(rows, dtype=np.float32)
    m.persist(full, block_id="b0", column=0)
    assert m.persisted is True
    # TIER1 裁剪到最近 history_days 天
    trimmed = trim_to_window(full, m.history_days)
    assert trimmed.shape == (history_days,)
    np.testing.assert_array_equal(trimmed, full[-history_days:])
    assert history_days_for(TIER1) == DEFAULT_HISTORY_DAYS[TIER1]


def test_tier2_keeps_full_history(tmp_path):
    rows = 5000
    m = FactorMaterialization(factor_id="f", tier=TIER2, history_days=None)
    full = np.arange(rows, dtype=np.float32)
    m.persist(full, block_id="b0", column=0)
    assert m.keep_values() is True
    trimmed = trim_to_window(full, history_days_for(TIER2))
    assert trimmed.shape == (rows,)  # 全历史不裁剪
    assert history_days_for(TIER2) is None
    # TIER2 走 FeatureBlock 物理落盘
    w = FeatureBlockWriter(tmp_path / "feat", columns_per_block=128)
    w.add("f", full, partition="2026-01")
    w.finish()
    r = FeatureBlockReader(tmp_path / "feat")
    assert r.read_factor_full("f").shape == (rows,)


# ---- (e) 块大小动态 + 尊重 writer 内存上限 ------------------------------------
def test_choose_columns_per_block_is_dynamic_and_respects_memory():
    # 行增长 → 更大块（在内存上限以下，读模式偏好整块）
    small = choose_columns_per_block(total_factors=100, row_count=1_000)
    large = choose_columns_per_block(total_factors=100, row_count=100_000)
    assert small < large
    assert MIN_COLUMNS_PER_BLOCK <= small <= MAX_COLUMNS_PER_BLOCK

    # 内存上限：rows 极大时列数被内存压回，block 字节绝不超预算 headroom
    rows = 100_000_000
    budget = 1 << 30  # 1GiB
    cols = choose_columns_per_block(total_factors=1000, row_count=rows, writer_memory_bytes=budget)
    est = block_bytes(rows, cols)
    assert est <= budget  # 尊重 writer 内存上限
    # 内存预算更紧 → 列数更小
    cols_tight = choose_columns_per_block(
        total_factors=1000, row_count=100_000_000, writer_memory_bytes=budget // 8
    )
    assert cols_tight <= cols


def test_writer_uses_dynamic_block_size(tmp_path):
    # 显式不传 columns_per_block → 动态选择；行少 → 每 block 列少，block 数更多
    rows = 10
    n = 200
    w = FeatureBlockWriter(tmp_path / "feat", writer_memory_bytes=1 << 20)
    for i in range(n):
        w.add(f"f{i:04d}", np.arange(rows, dtype=np.float32))
    w.finish()
    dynamic_cols = choose_columns_per_block(total_factors=n, row_count=rows,
                                            writer_memory_bytes=1 << 20)
    assert w.block_count() == math.ceil(n / dynamic_cols)
    assert w.block_count() > 1  # 小行 → 小块 → 多个块
    # 读回全部因子
    r = FeatureBlockReader(tmp_path / "feat")
    assert r.factor_count() == n


# ---- sink 接线：result-complete → hand to writer ------------------------------
def test_sink_writer_integration_groups_factors(tmp_path):
    from factor_engine.runtime.streaming_result_sink import (
        ResultItem,
        StreamingResultSink,
    )

    n = 300
    cols_per_block = 128
    rows = 500
    w = FeatureBlockWriter(tmp_path / "feat", columns_per_block=cols_per_block)
    sink = StreamingResultSink(
        writer=make_sink_writer(w),
        batch_size=64,
        writer_threads=1,
        queue_bytes=1 << 28,
    )
    sink.start()
    for i in range(n):
        fid = f"factor_{i:04d}"
        arr = np.arange(rows, dtype=np.float32) + float(i)
        accepted = sink.submit(name=fid, value=arr, partition="2026-08")
        assert accepted, f"submit rejected for {fid}"
    sink.finish()
    w.finish()

    # 分组块：300 因子 / 128 列 → ~3 个 block，不是 300 文件
    block_files = list((tmp_path / "feat").rglob("block_*.npy"))
    assert len(block_files) == math.ceil(n / cols_per_block)
    assert len(block_files) < n

    r = FeatureBlockReader(tmp_path / "feat")
    for i in range(n):
        fid = f"factor_{i:04d}"
        vals = r.read_factor_full(fid)
        expected = np.arange(rows, dtype=np.float32) + float(i)
        np.testing.assert_array_equal(vals, expected)

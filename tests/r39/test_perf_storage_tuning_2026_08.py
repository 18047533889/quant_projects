# -*- coding: utf-8 -*-
"""R39 PERF-055/056/057/058 —— block lake 布局 + 批量 parquet writer + storage tuning 基准。

覆盖：
  (a) BatchParquetWriter round-trip 正确性 + row-group 字节预算切分
  (b) block lake Layout B（column block）：300 因子 → 2 blocks(256/44)、
      manifest 映射、列裁剪、time_range 分区裁剪、instrument 谓词
  (c) block lake Layout A（long block）factor_id 谓词读取
  (d) LakeWriteLayoutPolicy.choose_layout 布局选择
  (e) storage_tuning_benchmark --quick 输出 JSON（含所需键、all_validated）

这些是独立基础设施测试，不触碰默认 materialize 写路径。
"""

from __future__ import annotations

import glob
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _install_storage_shim_if_needed() -> None:
    """storage/__init__.py 依赖 materializer 等重模块；R39 并发整改期间可能
    处于中间态（SyntaxError）。本测试只依赖 storage.block_lake /
    storage.parquet_batch_writer（独立基础设施），在真实包导入失败时安装
    轻量 shim 包，绕过 __init__，保证本测试不被并发编辑打断。"""
    try:
        import factor_engine.storage  # noqa: F401

        return
    except Exception:
        pass
    _fe_root = Path(__file__).resolve().parents[2]  # factor_engine/
    pkg = types.ModuleType("storage")
    pkg.__path__ = [str(_fe_root / "storage")]
    pkg.__package__ = "storage"
    sys.modules["storage"] = pkg


_install_storage_shim_if_needed()

from factor_engine.storage.block_lake import (  # noqa: E402
    FACTOR_BLOCK_LONG,
    FACTOR_BLOCK_WIDE,
    MATRIX_COLUMN_BLOCK,
    SINGLE_FACTOR_DELTA,
    BLOCK_COLUMN_COUNT,
    FactorBlockLakeReader,
    FactorBlockLakeWriter,
    LakeWriteLayoutPolicy,
)
from factor_engine.storage.parquet_batch_writer import BatchParquetWriter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grid_series(n_dates: int, n_instruments: int, seed: int, offset: float = 0.0):
    """构造 MultiIndex(datetime, instrument) Series 确定性数据。"""
    dates = pd.date_range("2026-08-03", periods=n_dates, freq="B")
    instruments = [f"EQ{i:05d}" for i in range(n_instruments)]
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    vals = rng.normal(size=len(idx)) + offset
    return pd.Series(vals, index=idx, name="value")


def _assert_series_equal(actual: pd.Series, expected: pd.Series) -> None:
    actual = actual.sort_index()
    expected = expected.sort_index()
    assert len(actual) == len(expected), (len(actual), len(expected))
    np.testing.assert_array_equal(actual.values, expected.values)


# ---------------------------------------------------------------------------
# (a) BatchParquetWriter
# ---------------------------------------------------------------------------


def test_batch_parquet_writer_roundtrip_and_row_groups(tmp_path):
    """round-trip 字节级一致 + row_group 数符合行数切分。"""
    n = 10_000
    table = pa.table(
        {
            "datetime": pa.array(np.arange(n), type=pa.timestamp("us")),
            "instrument": [f"EQ{i % 50:05d}" for i in range(n)],
            "value": np.random.default_rng(0).normal(size=n).astype("float64"),
        }
    )
    path = tmp_path / "by_rows.parquet"
    res = BatchParquetWriter.write_pa_table(
        table, path, compression="ZSTD-3", row_group_size=3000
    )
    assert res.num_row_groups == 4  # 3000/3000/3000/1000
    pf = pq.ParquetFile(str(path))
    assert pf.num_row_groups == 4
    assert [pf.metadata.row_group(i).num_rows for i in range(4)] == [3000, 3000, 3000, 1000]
    # 压缩生效
    codec = pf.metadata.row_group(0).column(2).compression
    assert codec == "ZSTD"
    # 值一致
    back = pq.read_table(str(path))
    assert back.num_rows == n
    for name in table.column_names:
        np.testing.assert_array_equal(
            back.column(name).to_numpy(), table.column(name).to_numpy()
        )


def test_batch_parquet_writer_byte_budget_honored(tmp_path):
    """compress_row_group_bytes 字节预算自动切行组（跨 batch 累积）。"""
    n = 20_000
    table = pa.table(
        {
            "datetime": pa.array(np.arange(n), type=pa.timestamp("us")),
            "instrument": [f"EQ{i % 100:05d}" for i in range(n)],
            "value": np.random.default_rng(1).normal(size=n).astype("float64"),
        }
    )
    batches = table.to_batches(max_chunksize=1024)  # 20 个小 batch
    assert len(batches) > 10
    path = tmp_path / "by_bytes.parquet"
    res = BatchParquetWriter.write_batches(
        batches, table.schema, path,
        compression="SNAPPY", compress_row_group_bytes=64 * 1024,
    )
    pf = pq.ParquetFile(str(path))
    # 我的计数与真实文件一致
    assert res.num_row_groups == pf.num_row_groups
    # 字节预算导致切分：>1 行组（证明切分）且 < 输入 batch 数（证明累积）
    assert 1 < res.num_row_groups < len(batches)
    # 行数守恒
    assert res.num_rows == n
    back = pq.read_table(str(path))
    assert back.num_rows == n
    for name in table.column_names:
        np.testing.assert_array_equal(
            back.column(name).to_numpy(), table.column(name).to_numpy()
        )
    # 每个 row group 未压缩字节不超过 budget + 单个 batch 上界（允许多余 ≤1 batch）
    one_batch_bytes = max(
        pa.Table.from_batches([b]).nbytes for b in batches
    )
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i)
        assert rg.nbytes <= 64 * 1024 + one_batch_bytes


def test_batch_parquet_writer_compression_matrix_uncompressed(tmp_path):
    """UNCOMPRESSED 与 dictionary 编码往返正确。"""
    n = 2000
    table = pa.table(
        {
            "instrument": [f"EQ{i % 20:05d}" for i in range(n)],
            "value": np.arange(n, dtype="float64"),
        }
    )
    path = tmp_path / "u.parquet"
    res = BatchParquetWriter.write_pa_table(
        table, path, compression="UNCOMPRESSED", dictionary_encode_cols=("instrument",)
    )
    pf = pq.ParquetFile(str(path))
    assert pf.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
    back = pq.read_table(str(path))
    np.testing.assert_array_equal(
        back.column("value").to_numpy(), table.column("value").to_numpy()
    )
    assert res.num_rows == n


# ---------------------------------------------------------------------------
# (b) block lake Layout B
# ---------------------------------------------------------------------------


def test_block_lake_layout_b_blocks_manifest_and_pushdown(tmp_path):
    """Layout B：300 因子 → 2 blocks(256/44)；manifest 映射；读回正确；谓词裁剪。"""
    factors = {f"f_{i:04d}": _grid_series(10, 3, seed=i, offset=float(i)) for i in range(300)}
    root = tmp_path / "lake_b"
    w = FactorBlockLakeWriter(root, layout=FACTOR_BLOCK_WIDE, generation=1)
    for fid, s in factors.items():
        w.write_factor(fid, s)
    w.close()

    parquet_files = sorted(glob.glob(str(root / "**" / "data.parquet"), recursive=True))
    assert len(parquet_files) == 2  # 256 + 44

    r = FactorBlockLakeReader(root)
    assert len(r.factor_ids()) == 300

    # manifest 映射 factor → block/column
    e0 = r.manifest_entry("f_0000")[0]
    assert e0["block"] == "0001"
    assert e0["column"] == "factor_0001"
    assert e0["layout"] == "WIDE"
    e299 = r.manifest_entry("f_0299")[0]
    assert e299["block"] == "0002"
    assert e299["column"] == "factor_0044"  # 44 个因子（block 内编号 1..44）

    # block 1 有 256 个因子列 + datetime/instrument
    pf1 = pq.ParquetFile(str(root / "date_bucket=2026-08" / "block=0001" / "data.parquet"))
    assert len(pf1.schema_arrow.names) == 2 + BLOCK_COLUMN_COUNT
    pf2 = pq.ParquetFile(str(root / "date_bucket=2026-08" / "block=0002" / "data.parquet"))
    assert len(pf2.schema_arrow.names) == 2 + 44

    # 读回正确性
    _assert_series_equal(r.read_factor("f_0000"), factors["f_0000"])
    _assert_series_equal(r.read_factor("f_0299"), factors["f_0299"])

    # instrument 谓词
    inst = r.read_factor("f_0000", instrument_filter="EQ00000")
    assert set(inst.index.get_level_values("instrument")) == {"EQ00000"}
    assert len(inst) == 10

    # time_range 行级过滤
    tr = (pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04"))
    sub = r.read_factor("f_0000", time_range=tr)
    assert len(sub) == 2 * 3  # 2 天 × 3 标的
    assert sub.index.get_level_values("datetime").min() == pd.Timestamp("2026-08-03")
    assert sub.index.get_level_values("datetime").max() == pd.Timestamp("2026-08-04")


def test_block_lake_layout_b_partition_pruning_multi_month(tmp_path):
    """time_range 分区裁剪：跨月 block 不被误删，纯错月查询被剪掉。"""
    dates = pd.date_range("2026-08-31", periods=6, freq="D")  # 8/31 + 9 月 5 天
    idx = pd.MultiIndex.from_product([dates, ["EQ00000", "EQ00001"]], names=["datetime", "instrument"])
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    root = tmp_path / "lake_mm"
    with FactorBlockLakeWriter(root, layout=FACTOR_BLOCK_WIDE, generation=1) as w:
        w.write_factor("mm", s)

    r = FactorBlockLakeReader(root)
    e = r.manifest_entry("mm")[0]
    assert e["date_bucket"] == "2026-08"
    assert e["date_bucket_end"] == "2026-09"

    aug = r.read_factor("mm", time_range=("2026-08-01", "2026-08-31"))
    sep = r.read_factor("mm", time_range=("2026-09-01", "2026-09-30"))
    oct_ = r.read_factor("mm", time_range=("2026-10-01", "2026-10-31"))
    assert len(aug) == 2  # 1 天 × 2
    assert len(sep) == 10  # 5 天 × 2
    assert len(oct_) == 0  # 纯错月 → 分区裁剪为空
    assert len(set(aug.index) | set(sep.index)) == 12


# ---------------------------------------------------------------------------
# (c) block lake Layout A（long block）
# ---------------------------------------------------------------------------


def test_block_lake_layout_a_long_read_via_factor_id(tmp_path):
    """Layout A：多因子长块；read_factor 用 factor_id 谓词读取。"""
    factors = {f"g_{i:03d}": _grid_series(10, 3, seed=i + 500, offset=float(i)) for i in range(300)}
    root = tmp_path / "lake_a"
    w = FactorBlockLakeWriter(
        root, layout=FACTOR_BLOCK_LONG, generation=7, compression="ZSTD-1"
    )
    for fid, s in factors.items():
        w.write_factor(fid, s)
    w.close()

    parquet_files = sorted(glob.glob(str(root / "**" / "data.parquet"), recursive=True))
    assert len(parquet_files) == 2  # 256 + 44

    r = FactorBlockLakeReader(root)
    e = r.manifest_entry("g_000")[0]
    assert e["layout"] == "LONG"
    assert e["column"] is None

    _assert_series_equal(r.read_factor("g_000"), factors["g_000"])
    _assert_series_equal(r.read_factor("g_299"), factors["g_299"])

    # long block 只含 5 列
    pf = pq.ParquetFile(str(root / "date_bucket=2026-08" / "block=0001" / "data.parquet"))
    assert pf.schema_arrow.names == ["datetime", "instrument", "factor_id", "value", "generation"]
    # 行数 = 256 因子 × 30 行
    assert pf.metadata.num_rows == 256 * 30

    # 时间过滤
    tr = (pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04"))
    sub = r.read_factor("g_000", time_range=tr)
    assert len(sub) == 6


# ---------------------------------------------------------------------------
# (d) LakeWriteLayoutPolicy
# ---------------------------------------------------------------------------


def test_choose_layout_matrix_and_high_factor_count():
    L = LakeWriteLayoutPolicy
    # matrix demand → MATRIX_COLUMN_BLOCK
    assert L.choose_layout(3, matrix_demand="high") == MATRIX_COLUMN_BLOCK
    assert L.choose_layout(50, read_pattern="matrix") == MATRIX_COLUMN_BLOCK
    assert L.choose_layout(3, matrix_demand=True) == MATRIX_COLUMN_BLOCK
    # 中等 matrix demand + 较多因子
    assert L.choose_layout(20, matrix_demand="medium") == MATRIX_COLUMN_BLOCK
    # 高因子数 + 列式/object store → FACTOR_BLOCK_WIDE
    assert L.choose_layout(300, read_pattern="column", storage_class="object") == FACTOR_BLOCK_WIDE
    assert L.choose_layout(64, read_pattern="column", storage_class="local") == FACTOR_BLOCK_WIDE
    # 中高因子数 → FACTOR_BLOCK_LONG
    assert L.choose_layout(100) == FACTOR_BLOCK_LONG
    assert L.choose_layout(10, update_frequency="low") == FACTOR_BLOCK_LONG
    # 单因子 + 低频 → SINGLE_FACTOR_DELTA
    assert L.choose_layout(1, update_frequency="low") == SINGLE_FACTOR_DELTA
    assert L.choose_layout(1, update_frequency=0.1) == SINGLE_FACTOR_DELTA


def test_choose_layout_numeric_frequency():
    L = LakeWriteLayoutPolicy
    assert L.choose_layout(1, update_frequency=0) == SINGLE_FACTOR_DELTA
    assert L.choose_layout(1, update_frequency=0.5) == SINGLE_FACTOR_DELTA
    # 高频更新 → 即使单因子也是 long block
    assert L.choose_layout(1, update_frequency="high") == FACTOR_BLOCK_LONG


# ---------------------------------------------------------------------------
# (e) storage tuning benchmark --quick
# ---------------------------------------------------------------------------


def test_storage_tuning_benchmark_quick_emits_json(tmp_path):
    from scripts.storage_tuning_benchmark import main

    out = tmp_path / "r39_storage_tuning.json"
    workdir = tmp_path / "work"
    rc = main(["--quick", "--out", str(out), "--workdir", str(workdir)])
    assert rc == 0
    assert out.exists()

    report = json.loads(out.read_text(encoding="utf-8"))
    # 顶层键
    for key in ("spec", "generated_at", "quick", "panel", "matrix", "combinations", "summary"):
        assert key in report, key
    assert report["quick"] is True
    assert report["spec"] == "r39-storage-tuning"
    # 每组合所需键
    assert len(report["combinations"]) >= 4  # 4 种压缩
    required = {
        "compression", "row_group_target_bytes", "num_row_groups",
        "num_rows", "bytes_written", "write_cpu_seconds", "write_wall_seconds",
        "read_scan_seconds", "read_scan_cpu_seconds", "validated",
    }
    for combo in report["combinations"]:
        assert required <= set(combo), combo
        assert combo["validated"] is True
        assert combo["num_rows"] == report["panel"]["rows"]
    # summary
    assert report["summary"]["all_validated"] is True

#!/usr/bin/env python
"""R39 PERF-058 —— storage tuning benchmark（写放大 / 压缩 / row group / 读成本）。

针对 ``storage.parquet_batch_writer.BatchParquetWriter`` 做存储参数扫描：

- 压缩矩阵：SNAPPY / ZSTD-1 / ZSTD-3 / UNCOMPRESSED
- row-group 目标：64MB / 128MB / 256MB（未压缩字节预算自动切行组）

每个组合测量：
    write CPU 秒 / write wall 秒 / bytes written / row-group 数 / 行数
    read scan 秒（pyarrow 投影 + 时间过滤）/ read scan CPU 秒

**每条路径都必须校验正确性**：写完后读回整表与原始数据逐值比对，比对不过则
抛错退出（绝不只测时间）。

用法::

    .venv/bin/python factor_engine/scripts/storage_tuning_benchmark.py \\
        --quick                      # CI 小样本
    .venv/bin/python factor_engine/scripts/storage_tuning_benchmark.py \\
        --stocks 300 --days 1200 --factors 4 --out r39_storage_tuning.json

输出: 默认 ``r39_storage_tuning.json``，含 ``combinations`` 逐组合指标。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_storage_shim_if_needed() -> None:
    """storage/__init__.py 依赖 materializer 等重模块；R39 并发整改期间可能
    处于中间态（SyntaxError）。本基准只依赖 storage.parquet_batch_writer
    （独立基础设施），在真实包导入失败时安装轻量 shim 包绕过 __init__。"""
    try:
        import factor_engine.storage  # noqa: F401

        return
    except Exception:
        pass
    import sys
    import types

    _fe_root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("storage")
    pkg.__path__ = [str(_fe_root / "storage")]
    pkg.__package__ = "storage"
    sys.modules["storage"] = pkg


_install_storage_shim_if_needed()

from factor_engine.storage.parquet_batch_writer import (  # noqa: E402
    BatchParquetWriter,
    ROW_GROUP_BYTES_LITERALS,
)

try:
    import duckdb
except ImportError:  # pragma: no cover - optional
    duckdb = None

COMPRESSIONS = ("SNAPPY", "ZSTD-1", "ZSTD-3", "UNCOMPRESSED")
DEFAULT_RG = ("64MB", "128MB", "256MB")
QUICK_RG = ("64MB",)

#: 时间过滤读投影时取的中间时间点比例（读后半段）。
_PROJECTION_FRAC = 0.5


# ---------------------------------------------------------------------------
# 合成面板
# ---------------------------------------------------------------------------


def build_wide_panel(stocks: int, days: int, factors: int, seed: int = 42) -> pa.Table:
    """生成合成宽表面板：datetime × instrument × factor_0001..factor_000N。

    排布：每个 instrument 内 datetime 升序（``np.repeat(instruments, days)`` +
    ``np.tile(dates, stocks)``），值 = 因子权重 + 确定性噪声。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    instruments = np.array([f"EQ{i:05d}" for i in range(stocks)], dtype=object)
    n = stocks * days
    ts = np.tile(dates.values, stocks)
    inst = np.repeat(instruments, days)

    arrays = [pa.array(ts, type=pa.timestamp("us")), pa.array(inst, type=pa.string())]
    fields = [
        pa.field("datetime", pa.timestamp("us")),
        pa.field("instrument", pa.string()),
    ]
    for f in range(factors):
        fid = f"factor_{f + 1:04d}"
        vals = ((f + 1) * 0.5 + rng.normal(size=n) * 0.1).astype("float64")
        arrays.append(pa.array(vals, type=pa.float64()))
        fields.append(pa.field(fid, pa.float64()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


# ---------------------------------------------------------------------------
# 校验 + DuckDB 元数据
# ---------------------------------------------------------------------------


def validate_roundtrip(path: str, original: pa.Table) -> bool:
    """读回整表，逐列与原始数据比对（顺序保持）。失败抛 AssertionError。"""
    back = pq.read_table(path)
    if back.num_rows != original.num_rows:
        raise AssertionError(
            f"行数不一致: 写 {original.num_rows} 读回 {back.num_rows} @ {path}"
        )
    for name in original.column_names:
        a = back.column(name).to_numpy()
        b = original.column(name).to_numpy()
        if not np.array_equal(a, b):
            raise AssertionError(f"列 {name!r} 读回值不一致 @ {path}")
    return True


def duckdb_row_groups(path: str) -> int:
    """用 DuckDB parquet_metadata 数 row group。

    ``parquet_metadata`` 每个 (row_group, column) 一行，需按 row_group_id 去重。
    """
    if duckdb is None:  # pragma: no cover - 可选依赖
        return -1
    return int(
        duckdb.execute(
            "SELECT count(DISTINCT row_group_id) FROM parquet_metadata(?)", [path]
        ).fetchone()[0]
    )


def _scan_projection(path: str, proj_col: str, mid_ts) -> tuple[float, float]:
    """读投影列 + 时间过滤，返回 (wall 秒, cpu 秒)。"""
    filters = [("datetime", ">=", mid_ts.to_pydatetime())]
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    table = pq.read_table(path, columns=["datetime", proj_col], filters=filters)
    # 触达数据，避免死代码被优化掉
    _ = table.num_rows
    return time.perf_counter() - wall0, time.process_time() - cpu0


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def run_benchmark(
    out_path: str,
    *,
    stocks: int,
    days: int,
    factors: int,
    compressions: tuple[str, ...],
    row_group_targets: tuple[int, ...],
    seed: int = 42,
    quick: bool = False,
    workdir: str | None = None,
) -> dict:
    """执行扫描并返回 JSON 报告 dict。"""
    out_path = str(out_path)
    workdir = workdir or os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(workdir, exist_ok=True)
    panel = build_wide_panel(stocks, days, factors, seed=seed)
    n = panel.num_rows
    proj_col = panel.column_names[2]  # factor_0001
    dates = panel.column("datetime").to_pandas()
    mid_ts = pd.Timestamp(dates.min()) + (pd.Timestamp(dates.max()) - pd.Timestamp(dates.min())) * _PROJECTION_FRAC

    # 以较小的 RecordBatch 流式喂入 writer，让 row-group 字节预算能跨 batch 累积
    # （模拟计算引擎逐 batch 下发的真实流式写入）。
    batches = panel.to_batches(max_chunksize=2048)

    combinations = []
    for comp in compressions:
        for rg_bytes in row_group_targets:
            path = os.path.join(workdir, f"panel_{comp}_{rg_bytes}.parquet")
            if os.path.exists(path):
                os.remove(path)

            wall0 = time.perf_counter()
            cpu0 = time.process_time()
            res = BatchParquetWriter.write_batches(
                batches,
                panel.schema,
                path,
                compression=comp,
                compress_row_group_bytes=rg_bytes,
                stats=True,
            )
            write_wall = time.perf_counter() - wall0
            write_cpu = time.process_time() - cpu0

            # 正确性（绝不只测时间）
            validated = validate_roundtrip(path, panel)

            read_wall, read_cpu = _scan_projection(path, proj_col, mid_ts)
            rgs_duck = duckdb_row_groups(path)
            combos = {
                "compression": comp,
                "row_group_target_bytes": rg_bytes,
                "num_row_groups": res.num_row_groups,
                "num_row_groups_duckdb": rgs_duck,
                "num_rows": res.num_rows,
                "num_columns": res.num_columns,
                "bytes_written": res.bytes_written,
                "write_cpu_seconds": round(write_cpu, 6),
                "write_wall_seconds": round(write_wall, 6),
                "read_scan_seconds": round(read_wall, 6),
                "read_scan_cpu_seconds": round(read_cpu, 6),
                "validated": bool(validated),
                "dictionary_columns": list(res.dictionary_encode_cols),
                "compression_level": res.compression_level,
            }
            combinations.append(combos)

    report = {
        "spec": "r39-storage-tuning",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": bool(quick),
        "panel": {
            "stocks": stocks,
            "days": days,
            "factors": factors,
            "rows": n,
            "seed": seed,
        },
        "matrix": {
            "compressions": list(compressions),
            "row_group_target_bytes": list(row_group_targets),
        },
        "combinations": combinations,
        "summary": {
            "min_bytes_written": min(c["bytes_written"] for c in combinations),
            "max_bytes_written": max(c["bytes_written"] for c in combinations),
            "min_write_cpu_seconds": min(c["write_cpu_seconds"] for c in combinations),
            "min_read_scan_seconds": min(c["read_scan_seconds"] for c in combinations),
            "all_validated": all(c["validated"] for c in combinations),
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def _parse_rg(value: str) -> int:
    key = value.strip().upper()
    if key in ROW_GROUP_BYTES_LITERALS:
        return ROW_GROUP_BYTES_LITERALS[key]
    return int(value)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R39 storage tuning benchmark")
    ap.add_argument("--quick", action="store_true", help="CI 小样本")
    ap.add_argument("--out", default="r39_storage_tuning.json", help="JSON 输出路径")
    ap.add_argument("--stocks", type=int, default=300, help="股票数")
    ap.add_argument("--days", type=int, default=1200, help="交易日数")
    ap.add_argument("--factors", type=int, default=4, help="因子列数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compressions", type=str, default=",".join(COMPRESSIONS))
    ap.add_argument(
        "--row-group-sizes",
        type=str,
        default=None,
        help="逗号分隔：64MB/128MB/256MB 或字节数",
    )
    ap.add_argument("--workdir", type=str, default=None, help="中间文件目录")
    args = ap.parse_args(argv)

    if args.quick:
        stocks, days, factors = 20, 50, 3
        row_group_targets = tuple(ROW_GROUP_BYTES_LITERALS[k] for k in QUICK_RG)
    else:
        stocks, days, factors = args.stocks, args.days, args.factors
        if args.row_group_sizes:
            row_group_targets = tuple(_parse_rg(s) for s in args.row_group_sizes.split(","))
        else:
            row_group_targets = tuple(ROW_GROUP_BYTES_LITERALS[k] for k in DEFAULT_RG)

    compressions = tuple(
        c.strip().upper() for c in args.compressions.split(",") if c.strip()
    )

    print(
        f"[storage-tuning] panel={stocks}x{days}x{factors} "
        f"compressions={compressions} row_groups={row_group_targets}"
    )
    t0 = time.perf_counter()
    report = run_benchmark(
        args.out,
        stocks=stocks,
        days=days,
        factors=factors,
        compressions=compressions,
        row_group_targets=row_group_targets,
        seed=args.seed,
        quick=args.quick,
        workdir=args.workdir,
    )
    el = time.perf_counter() - t0
    print(f"[storage-tuning] done in {el:.1f}s -> {args.out}")
    print(
        f"[storage-tuning] combos={len(report['combinations'])} "
        f"all_validated={report['summary']['all_validated']} "
        f"min_bytes={report['summary']['min_bytes_written']}"
    )
    if not report["summary"]["all_validated"]:
        print("[storage-tuning] FAIL: 存在未通过正确性校验的组合", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

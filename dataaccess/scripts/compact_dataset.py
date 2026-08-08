#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小文件 compaction（#37）：把数据集里平均尺寸过小的 parquet 合并成大文件。

用法
    python3 scripts/compact_dataset.py <dataset> \
        [--min-file-bytes 33554432] [--dry-run] [--params factor_id=x]

策略
    - 扫描数据集根下所有 parquet（走 registry + manifest）
    - 平均文件 < ``--min-file-bytes``（默认 32MB，COS 上建议 256MB+）或
      文件数超阈值 → 按 time 顺序合并重写
    - 写完更新 manifest epoch（touch_manifest_epoch），读路径立刻看到新快照
    - ``--dry-run`` 只打印统计不写文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--min-file-bytes", type=int, default=32 * 1024 * 1024)
    p.add_argument("--min-files", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--params", action="append", default=[])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    params: dict[str, str] = {}
    for kv in args.params:
        k, _, v = kv.partition("=")
        params[k] = v

    from data_access import get_store

    store = get_store()
    ds = store._registry.get(args.dataset)
    paths = store._resolve_raw_paths(ds, time_range=None, params=params)
    import glob as glob_mod

    files = sorted(
        p for pattern in paths for p in glob_mod.glob(pattern, recursive=True)
        if not p.endswith(("_manifest.parquet", "_manifest.json", "_manifest_rowgroups.parquet"))
    )
    if not files:
        print(f"{args.dataset}: 没有可枚举文件")
        return 1
    sizes = []
    for fp in files:
        try:
            sizes.append((Path(fp).stat().st_size, fp))
        except OSError:
            pass
    total_bytes = sum(s for s, _ in sizes)
    avg = total_bytes / max(1, len(sizes))
    print(
        f"{args.dataset}: files={len(sizes)} total={total_bytes/1024/1024:.1f}MB "
        f"avg={avg/1024/1024:.2f}MB"
    )
    too_small = avg < args.min_file_bytes or len(sizes) >= args.min_files
    if not too_small:
        print("OK：文件平均尺寸达标，无需 compaction")
        return 0
    if args.dry_run:
        print(f"dry-run：需要 compaction（avg 文件 {avg/1024:.0f}KB，阈值 "
              f"{args.min_file_bytes/1024/1024:.0f}MB）")
        return 0

    # 按日期分组（文件名通常是 date.parquet）合并成更大的按年文件
    import pyarrow as pa
    import pyarrow.parquet as pq

    groups: dict[str, list[str]] = {}
    for _, fp in sizes:
        name = Path(fp).stem
        year = name[:4] if len(name) >= 4 and name[:4].isdigit() else "misc"
        groups.setdefault(year, []).append(fp)

    out_dir = ds.root if isinstance(getattr(ds, "root", None), Path) else Path(ds.root)
    written = 0
    # compaction 也是 mutation：走统一事务（bump source_epoch + 重建 manifest，
    # 合并后 min/max/rows 全部变化，旧 manifest 必须失效重建）。
    with store._dataset_mutation(args.dataset, **params):
        for year, fps in sorted(groups.items()):
            if not fps:
                continue
            chunks = [pq.read_table(f) for f in fps]
            table = pa.concat_tables(chunks, promote_options="permissive")
            out = out_dir / f"{year}.parquet"
            tmp = out_dir / f".{year}.parquet.compact_tmp"
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(out)
            written += 1
            print(f"  compacted {len(fps)} files -> {out.name} ({len(fps)} 合并)")
            for f in fps:
                if Path(f) != out and Path(f).name != out.name:
                    try:
                        Path(f).unlink()
                    except OSError:
                        pass
    print(f"完成：写入 {written} 个大文件并重建 manifest（source_epoch bump）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

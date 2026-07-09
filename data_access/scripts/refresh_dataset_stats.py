#!/usr/bin/env python3
"""刷新 datasets.yaml 静态数据集的 sidecar 统计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _refresh_one(
    store,
    registry,
    dataset: str,
    *,
    with_null_ratio: bool,
    null_ratio_sample_files: int,
) -> tuple[str, int, int]:
    from data_access.registry import StaticDataset
    from data_access.stats import build_dataset_stats_snapshot, write_stats_sidecar

    ds = registry.get(dataset)
    if not isinstance(ds, StaticDataset):
        raise ValueError(f"仅支持 static 数据集，收到 {dataset!r} kind={ds.kind}")
    snapshot = build_dataset_stats_snapshot(
        store,
        dataset,
        with_null_ratio=with_null_ratio,
        null_ratio_sample_files=null_ratio_sample_files,
    )
    out = write_stats_sidecar(ds.root, snapshot)
    return str(out), snapshot.num_rows, snapshot.num_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="刷新 data_access 数据集 sidecar 统计")
    parser.add_argument("--dataset", default=None, help="datasets.yaml 中的数据集名")
    parser.add_argument(
        "--all",
        action="store_true",
        help="刷新 registry 中全部 static 数据集",
    )
    parser.add_argument("--config", default=None, help="datasets.yaml 路径")
    parser.add_argument(
        "--with-null-ratio",
        action="store_true",
        help="采样 parquet 估算列非空率并写入 sidecar",
    )
    parser.add_argument(
        "--null-ratio-sample-files",
        type=int,
        default=3,
        help="null_ratio 采样文件数",
    )
    args = parser.parse_args(argv)

    if not args.all and not args.dataset:
        parser.error("请指定 --dataset 或 --all")
    if args.all and args.dataset:
        parser.error("--dataset 与 --all 不能同时使用")

    root = Path(__file__).resolve().parents[1]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    from data_access.registry import StaticDataset, load_registry
    from data_access.store import DataAccessStore
    from data_access.engine import DuckDBEngine

    registry = load_registry(args.config)
    store = DataAccessStore(registry, DuckDBEngine())

    if args.all:
        names = sorted(
            name
            for name in registry
            if isinstance(registry.get(name), StaticDataset)
        )
        if not names:
            print("未找到 static 数据集", file=sys.stderr)
            return 2
    else:
        names = [args.dataset]

    failures = 0
    for name in names:
        try:
            out, rows, files = _refresh_one(
                store,
                registry,
                name,
                with_null_ratio=args.with_null_ratio,
                null_ratio_sample_files=args.null_ratio_sample_files,
            )
            print(f"[ok] {name}: {out} rows={rows} files={files}")
        except Exception as exc:
            failures += 1
            print(f"[fail] {name}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

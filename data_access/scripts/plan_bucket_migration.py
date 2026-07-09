#!/usr/bin/env python3
"""规划 Hive bucket 物理分区迁移：输出目标路径模板与 instrument→bucket 映射摘要。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="规划 bucket 分区 ETL 迁移")
    parser.add_argument("--dataset", required=True, help="datasets.yaml 中的数据集名")
    parser.add_argument("--config", default=None, help="datasets.yaml 路径")
    parser.add_argument(
        "--sample-instruments",
        default="",
        help="逗号分隔标的样本，用于演示 bucket 剪枝",
    )
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    from data_access.layout_policy import (
        LayoutPolicy,
        bucket_partition_predicate,
        parse_layout_policy,
        stable_bucket,
    )
    from data_access.registry import load_registry

    registry = load_registry(args.config)
    ds = registry.get(args.dataset)
    layout_raw = getattr(ds, "layout_policy", None)
    if isinstance(layout_raw, LayoutPolicy):
        layout = layout_raw
    else:
        layout = parse_layout_policy(layout_raw)
    if layout is None or layout.bucket is None:
        print(f"数据集 {args.dataset!r} 未配置 layout_policy.bucket", file=sys.stderr)
        return 2

    bucket_col = layout.bucket.column
    bucket_count = layout.bucket.count
    partition_cols = list(getattr(ds, "partition_columns", None) or [])
    root_path = str(getattr(ds, "root", ""))
    glob_pattern = str(getattr(ds, "glob", "**/*.parquet"))

    target_glob = glob_pattern
    if bucket_col not in partition_cols:
        partition_cols = [*partition_cols, bucket_col]
    if "year" not in partition_cols:
        partition_cols = ["year", "month", *partition_cols]

    target_template = (
        f"{root_path.rstrip('/')}/"
        f"year={{year}}/month={{month:02d}}/{bucket_col}={{{{bucket}}}}/part-*.parquet"
    )

    instruments = [s.strip() for s in args.sample_instruments.split(",") if s.strip()]
    bucket_map = {inst: stable_bucket(inst, bucket_count) for inst in instruments}
    predicate_sql, bucket_values = bucket_partition_predicate(
        instruments,
        bucket_column=bucket_col,
        bucket_count=bucket_count,
    )

    plan = {
        "dataset": args.dataset,
        "source_root": root_path,
        "source_glob": glob_pattern,
        "layout_policy": {"bucket_column": bucket_col, "bucket_count": bucket_count},
        "recommended_partition_columns": partition_cols,
        "target_path_template": target_template,
        "etl_steps": [
            "读取源 parquet（按日期窗口分批）",
            "为每行计算 bucket = stable_hash(instrument) % count",
            "按 year/month/bucket 写出 hive 分区目录",
            "更新 datasets.yaml：启用 partition_columns 含 bucket",
            "运行 refresh_dataset_stats.py 刷新 sidecar",
        ],
        "sample_instrument_buckets": bucket_map,
        "sample_prune_predicate": predicate_sql,
        "sample_bucket_values": bucket_values,
    }

    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(f"数据集: {args.dataset}")
        print(f"源路径: {root_path} / {glob_pattern}")
        print(f"Bucket: {bucket_col} × {bucket_count}")
        print(f"建议 partition_columns: {partition_cols}")
        print(f"目标模板: {target_template}")
        print("ETL 步骤:")
        for i, step in enumerate(plan["etl_steps"], 1):
            print(f"  {i}. {step}")
        if bucket_map:
            print("样本 instrument → bucket:")
            for inst, b in sorted(bucket_map.items(), key=lambda x: x[1]):
                print(f"  {inst}: {b}")
            if predicate_sql:
                print(f"剪枝谓词: {predicate_sql}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

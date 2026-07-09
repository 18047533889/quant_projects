#!/usr/bin/env python3
"""校验 bucket 分区迁移：对比源与目标行数及分区覆盖。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_bucket_migration(
    *,
    dataset: str,
    config: str | None,
    target_root: str | Path,
    tolerance_ratio: float = 0.001,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    from data_access.registry import StaticDataset, load_registry
    from data_access.stats import estimate_parquet_rows, expand_parquet_paths

    registry = load_registry(config)
    ds = registry.get(dataset)
    if not isinstance(ds, StaticDataset):
        raise ValueError(f"仅支持 static 数据集，收到 kind={getattr(ds, 'kind', '?')}")

    root_path = Path(str(getattr(ds, "root", "")))
    glob_pattern = str(getattr(ds, "glob", "**/*.parquet"))
    pattern = glob_pattern if glob_pattern.startswith("/") else str(root_path / glob_pattern)
    source_paths = expand_parquet_paths([pattern])
    source_rows = estimate_parquet_rows(source_paths)

    target = Path(target_root)
    target_paths = expand_parquet_paths([str(target / "**/*.parquet")])
    target_rows = estimate_parquet_rows(target_paths)

    diff = abs(source_rows - target_rows)
    ratio = diff / source_rows if source_rows else 0.0
    passed = source_rows == 0 or ratio <= tolerance_ratio

    partition_cols = list(getattr(ds, "partition_columns", None) or [])
    layout = getattr(ds, "layout_policy", None)
    bucket_col = None
    if layout is not None and getattr(layout, "bucket", None) is not None:
        bucket_col = layout.bucket.column
        if bucket_col not in partition_cols:
            partition_cols = [*partition_cols, bucket_col]

    target_partitions: set[str] = set()
    for path in target_paths:
        parts: list[str] = []
        for segment in path.parts:
            if "=" in segment and not segment.startswith("."):
                parts.append(segment)
        if parts:
            target_partitions.add("/".join(parts[-len(partition_cols) :] if partition_cols else parts))

    return {
        "dataset": dataset,
        "passed": passed,
        "source_files": len(source_paths),
        "target_files": len(target_paths),
        "source_rows": source_rows,
        "target_rows": target_rows,
        "row_diff": diff,
        "row_diff_ratio": ratio,
        "tolerance_ratio": tolerance_ratio,
        "target_partitions": len(target_partitions),
        "partition_columns": partition_cols,
        "bucket_column": bucket_col,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 bucket ETL 迁移行数一致性")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--target-root", required=True)
    parser.add_argument(
        "--tolerance-ratio",
        type=float,
        default=0.001,
        help="允许行数偏差比例（默认 0.1%）",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = validate_bucket_migration(
            dataset=args.dataset,
            config=args.config,
            target_root=args.target_root,
            tolerance_ratio=args.tolerance_ratio,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['dataset']}")
        print(
            f"源: files={result['source_files']} rows={result['source_rows']}  "
            f"目标: files={result['target_files']} rows={result['target_rows']}"
        )
        print(f"偏差: {result['row_diff']} ({result['row_diff_ratio']:.4%})")
        print(f"目标分区数: {result['target_partitions']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

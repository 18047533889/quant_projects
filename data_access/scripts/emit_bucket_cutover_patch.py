#!/usr/bin/env python3
"""生成 datasets.yaml bucket 切读补丁（partition_columns + root 提示）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_cutover_patch(
    *,
    dataset: str,
    config: str | None,
    target_root: str | Path,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    from data_access.layout_policy import LayoutPolicy, parse_layout_policy
    from data_access.registry import load_registry

    registry = load_registry(config)
    ds = registry.get(dataset)
    layout_raw = getattr(ds, "layout_policy", None)
    if isinstance(layout_raw, LayoutPolicy):
        layout = layout_raw
    else:
        layout = parse_layout_policy(layout_raw)
    bucket_col = layout.bucket.column if layout and layout.bucket else "bucket"
    partition_cols = list(getattr(ds, "partition_columns", None) or [])
    if "year" not in partition_cols:
        partition_cols = ["year", "month", *partition_cols]
    if bucket_col not in partition_cols:
        partition_cols.append(bucket_col)

    env_key = f"DATA_ACCESS_READ_ROOT_{dataset.upper().replace('-', '_')}"
    yaml_snippet = (
        f"# {dataset} bucket 切读补丁（迁移校验通过后应用）\n"
        f"{dataset}:\n"
        f"  root: {target_root}\n"
        f"  partition_columns: [{', '.join(partition_cols)}]\n"
    )
    return {
        "dataset": dataset,
        "target_root": str(target_root),
        "partition_columns": partition_cols,
        "env_cutover": {env_key: str(target_root)},
        "yaml_snippet": yaml_snippet,
        "steps": [
            f"1. validate_bucket_migration.py --dataset {dataset} --target-root {target_root}",
            f"2. 合并 yaml_snippet 到 datasets.yaml（或 export {env_key} 灰度）",
            "3. refresh_dataset_stats.py --with-null-ratio",
            "4. factor_engine read smoke + input_dq",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 bucket 切读 YAML 补丁")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    patch = build_cutover_patch(
        dataset=args.dataset,
        config=args.config,
        target_root=args.target_root,
    )
    if args.json:
        print(json.dumps(patch, ensure_ascii=False, indent=2))
    else:
        print(patch["yaml_snippet"])
        print("Steps:")
        for s in patch["steps"]:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

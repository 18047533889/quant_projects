#!/usr/bin/env python3
"""执行 Hive bucket 物理分区迁移：源 parquet → year/month/bucket 分区。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _partition_path(target_root: Path, row: dict, partition_cols: list[str]) -> Path:
    parts = [target_root]
    for col in partition_cols:
        val = row[col]
        if col in {"month"}:
            parts.append(f"{col}={int(val):02d}")
        else:
            parts.append(f"{col}={val}")
    return Path(*parts)


def migrate_dataset(
    *,
    dataset: str,
    config: str | None,
    target_root: str | Path,
    dry_run: bool = True,
    max_files: int = 0,
    overwrite: bool = False,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    import pandas as pd

    from data_access.layout_policy import LayoutPolicy, parse_layout_policy, stable_bucket
    from data_access.registry import StaticDataset, load_registry
    from data_access.stats import expand_parquet_paths

    registry = load_registry(config)
    ds = registry.get(dataset)
    if not isinstance(ds, StaticDataset):
        raise ValueError(f"仅支持 static 数据集，收到 kind={getattr(ds, 'kind', '?')}")

    layout_raw = getattr(ds, "layout_policy", None)
    if isinstance(layout_raw, LayoutPolicy):
        layout = layout_raw
    else:
        layout = parse_layout_policy(layout_raw)
    if layout is None or layout.bucket is None:
        raise ValueError(f"数据集 {dataset!r} 未配置 layout_policy.bucket")

    bucket_col = layout.bucket.column
    bucket_count = layout.bucket.count
    inst_col = ds.instrument_column
    time_col = ds.time_column

    partition_cols = list(getattr(ds, "partition_columns", None) or [])
    if "year" not in partition_cols:
        partition_cols = ["year", "month", *partition_cols]
    if bucket_col not in partition_cols:
        partition_cols.append(bucket_col)

    root_path = Path(str(getattr(ds, "root", "")))
    glob_pattern = str(getattr(ds, "glob", "**/*.parquet"))
    pattern = glob_pattern if glob_pattern.startswith("/") else str(root_path / glob_pattern)
    paths = expand_parquet_paths([pattern])
    if max_files > 0:
        paths = paths[:max_files]

    target = Path(target_root)
    files_written = 0
    rows_written = 0
    partitions: set[str] = set()

    for src in paths:
        df = pd.read_parquet(src)
        if time_col not in df.columns or inst_col not in df.columns:
            raise ValueError(
                f"源文件缺少列 {time_col!r}/{inst_col!r}: {src}"
            )
        ts = pd.to_datetime(df[time_col])
        df = df.copy()
        df["year"] = ts.dt.year
        df["month"] = ts.dt.month
        df[bucket_col] = (
            df[inst_col].astype(str).map(lambda x: stable_bucket(x, bucket_count))
        )

        if dry_run:
            rows_written += len(df)
            for key, _ in df.groupby(partition_cols, sort=False):
                if isinstance(key, tuple):
                    parts = dict(zip(partition_cols, key))
                else:
                    parts = {partition_cols[0]: key}
                partitions.add("|".join(f"{k}={parts[k]}" for k in partition_cols))
            continue

        for key, group in df.groupby(partition_cols, sort=False):
            if isinstance(key, tuple):
                parts = dict(zip(partition_cols, key))
            else:
                parts = {partition_cols[0]: key}
            out_dir = _partition_path(target, parts, partition_cols)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"part-{src.stem}.parquet"
            if out_path.exists() and not overwrite:
                continue
            drop_cols = [c for c in partition_cols if c in group.columns]
            group.drop(columns=drop_cols).to_parquet(out_path, index=False)
            files_written += 1
            rows_written += len(group)
            partitions.add("|".join(f"{k}={parts[k]}" for k in partition_cols))

    return {
        "dataset": dataset,
        "dry_run": dry_run,
        "source_files": len(paths),
        "rows_processed": rows_written,
        "files_written": files_written,
        "partitions": sorted(partitions),
        "target_root": str(target),
        "partition_columns": partition_cols,
        "bucket_column": bucket_col,
        "bucket_count": bucket_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="执行 bucket 分区 ETL 迁移")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--target-root",
        required=True,
        help="目标 hive 分区根目录",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际写入（默认 dry-run）",
    )
    parser.add_argument("--max-files", type=int, default=0, help="最多处理源文件数；0=全部")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = migrate_dataset(
            dataset=args.dataset,
            config=args.config,
            target_root=args.target_root,
            dry_run=not args.execute,
            max_files=args.max_files,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        mode = "DRY-RUN" if result["dry_run"] else "EXECUTE"
        print(f"[{mode}] 数据集: {result['dataset']}")
        print(f"源文件: {result['source_files']}  行: {result['rows_processed']}")
        print(f"目标: {result['target_root']}")
        print(f"分区列: {result['partition_columns']}")
        if not result["dry_run"]:
            print(f"写出文件: {result['files_written']}  分区数: {len(result['partitions'])}")
        elif result["partitions"]:
            print(f"预计分区数: {len(result['partitions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

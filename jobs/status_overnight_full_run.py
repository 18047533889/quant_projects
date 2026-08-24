#!/usr/bin/env python3
"""Full-history partitioned computation + COS upload, resumable, for overnight runs.

Processes every StockDailyBar date partition once, computes the core feature
table locally, uploads the per-date parquet to a versioned COS staging prefix,
and checkpoints each partition's result/failure atomically. Designed to run as
a detached background process that survives the interactive session.

Usage:
    python3 status_overnight_full_run.py \
        --start 2016-01-04 --end 2026-08-21 \
        --workdir /home/sunhaiwei/quantsociety/runs/full-history-overnight \
        --run-id full-history-overnight --resume
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sunhaiwei/.local/lib/python3.12/site-packages")

from ashare_feature_pipeline import BUCKET, build_daily_features, load_registry, read_table, validate_daily
from status_partition_runner import PartitionRunner, _write_partition_features, cos_cp, list_partition_keys

STAGING_PREFIX = "candidate_pool/hsunbj/ashare/status/staging"


def upload_partition(context, registry, run_id: str):
    """Compute one date's features, write locally, then upload to COS staging."""
    result = dict(_write_partition_features(context, registry))
    if context.date is None:
        raise ValueError(f"partition key has no date: {context.key}")
    key = f"{STAGING_PREFIX}/run_id={run_id}/frequency=1d/{context.date.isoformat()}.parquet"
    cos_cp(result["output"]["local_path"], f"cos://{BUCKET}/{key}")
    result["cos_key"] = key
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    keys = list_partition_keys("StockDailyBar", args.start, args.end)
    print(
        json.dumps(
            {
                "total_partitions": len(keys),
                "range": [args.start.isoformat(), args.end.isoformat()],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def processor(context):
        return upload_partition(context, registry, args.run_id)

    runner = PartitionRunner(
        workdir, run_id=args.run_id, dataset="StockDailyBar", resume=args.resume
    )
    summary = runner.run(keys, processor)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

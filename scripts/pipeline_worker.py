#!/usr/bin/env python3
"""文件任务队列 worker：消费 pending config 任务。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "factor_engine"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from pipeline import run_from_config
from runtime.task_queue import FileTaskQueue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume factor_engine file task queue")
    parser.add_argument("--queue-root", required=True, help="Queue root with pending/running/done")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = FileTaskQueue(args.queue_root)
    processed = 0
    while processed < args.max_jobs:
        job = queue.claim()
        if job is None:
            break
        try:
            payload = dict(job.payload or {})
            materialize = args.materialize
            if materialize is None:
                materialize = payload.pop("materialize", None)
            output = run_from_config(
                job.config_path,
                output_root=payload.pop("output_root", None),
                materialize=materialize,
                preview_rows=int(payload.pop("preview_rows", 5)),
                incremental=bool(payload.pop("incremental", False)),
                dq_check=bool(payload.pop("dq_check", False)),
                dq_strict=bool(payload.pop("dq_strict", True)),
                input_dq_check=bool(payload.pop("input_dq_check", False)),
                input_dq_strict=bool(payload.pop("input_dq_strict", True)),
                max_retries=int(payload.pop("max_retries", 0)),
                resume_materialize=bool(payload.pop("resume_materialize", False)),
                profile=payload.pop("profile", None),
            )
            queue.complete(job.job_id, result=output.get("summary"))
        except Exception as exc:
            queue.fail(job.job_id, error=str(exc))
        processed += 1

    print(json.dumps({"processed": processed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

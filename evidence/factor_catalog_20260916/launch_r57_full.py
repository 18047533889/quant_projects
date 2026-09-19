# -*- coding: utf-8 -*-
"""R57 launcher: bounded-parallel compile of the whole catalog. No reads, no execution.

Memory-bounded the same way as launch_r20_full.py: concurrency is derived from
MemAvailable, never from a hard-coded worker count, and each shard runs under the
existing RSS watchdog.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/home/sunhaiwei/quant_projects")


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--max-slots", type=int, default=6)
    ap.add_argument("--per-shard-rss-mib", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    available = mem_available_bytes()
    slots = min(args.max_slots, max(1, int(available * 0.5) // (2 * 1024**3)))
    env = dict(
        os.environ,
        PYTHONPATH=".",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        POLARS_MAX_THREADS="2",
    )

    def run(shard: int) -> int:
        prefix = outdir / f"full-s{shard}"
        cmd = [
            str(ROOT / ".venv/bin/python"),
            "evidence/r3/test_watchdog.py",
            "--log", str(prefix) + ".log",
            "--max-rss-mib", str(args.per_shard_rss_mib),
            "--timeout", str(args.timeout),
            "--",
            str(ROOT / ".venv/bin/python"),
            "evidence/factor_catalog_20260916/compile_r57_full.py",
            "--source", args.source,
            "--source-sha", args.source_sha,
            "--prefix", str(prefix),
            "--shard", str(shard),
            "--shards", str(args.shards),
        ]
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        print(json.dumps({
            "shard": shard,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-1500:],
        }), flush=True)
        return result.returncode

    print(json.dumps({
        "slots": slots,
        "available_bytes": available,
        "shards": args.shards,
        "outdir": str(outdir),
    }), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        codes = list(pool.map(run, range(args.shards)))
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

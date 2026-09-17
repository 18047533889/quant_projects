"""Warm-process, bounded smoke sweep with independently closed batch evidence.

Use an external sampled watchdog. A killed batch is NOT published, while every
earlier batch and its source cursor remain independently readable and resumable.
This is a research smoke harness, not production factor publication.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time

def sweep(args, *, runner=None):
    if runner is None:
        import smoke_catalog as runner
    base = Path(__file__).resolve().parent
    prefix = args.output_prefix
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("output-prefix must be a plain new filename prefix")
    if min(args.limit, args.chunk, args.max_rss_mib) <= 0 or args.offset < 0:
        raise ValueError("positive limits and non-negative source offset required")
    completed = 0
    cursor = args.offset
    started = time.monotonic()
    ledger_path = base / (prefix + ".ledger.jsonl")
    original_argv = sys.argv
    with ledger_path.open("x", encoding="utf-8") as ledger:
        ledger.write(json.dumps({"event": "start", "config": vars(args),
                                 "scope": "real_data_research_smoke_not_production"}) + "\n")
        ledger.flush()
        try:
            while completed < args.limit:
                take = min(args.chunk, args.limit - completed)
                output = base / f"{prefix}-{cursor:06d}.jsonl.gz"
                summary_path = output.with_suffix("").with_suffix(".summary.json")
                if output.exists() or summary_path.exists():
                    raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
                sys.argv = ["smoke_catalog.py", "--limit", str(take),
                            "--offset", str(cursor), "--prepare-chunk", str(take),
                            "--batch-size", str(take), "--output", str(output),
                            "--max-rss-mib", str(args.max_rss_mib),
                            "--rss-rotate-mib", str(args.rotate_rss_mib),
                            "--timeout-seconds", str(args.timeout_seconds),
                            "--deadline-mode", args.deadline_mode,
                            "--backend", args.backend]
                if getattr(args, "input", None):
                    sys.argv.extend(["--input", str(args.input)])
                if args.daily_only:
                    sys.argv.append("--daily-only")
                runner.main()
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                processed = summary["processed"]
                next_cursor = summary.get("next_offset")
                entry = {"event": "batch_complete", "output": str(output),
                         "summary": str(summary_path), "source_offset": cursor,
                         "processed": processed, "next_offset": next_cursor,
                         "counts": summary["counts"], "seconds": summary["seconds"]}
                ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
                ledger.flush()
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                completed += processed
                if not processed:
                    break
                if next_cursor is None or next_cursor <= cursor:
                    raise ValueError("batch did not provide a strictly advancing source cursor")
                cursor = next_cursor
                gc.collect()
                if summary.get("stopped_reason") or runner.current_rss_mib() > args.rotate_rss_mib:
                    break
        finally:
            sys.argv = original_argv
        result = {"event": "sweep_closed", "processed": completed,
                  "requested": args.limit, "next_offset": cursor,
                  "seconds": time.monotonic() - started,
                  "requested_count_completed": completed == args.limit}
        ledger.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="explicit reviewed catalog JSONL gzip")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--rotate-rss-mib", type=int, default=3000)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--deadline-mode", choices=("external-watchdog", "legacy-cooperative"),
                        default="external-watchdog")
    parser.add_argument("--backend", choices=("pandas", "polars", "auto"), default="pandas")
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    sweep(parser.parse_args())

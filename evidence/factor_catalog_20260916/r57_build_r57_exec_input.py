# -*- coding: utf-8 -*-
"""Build a bounded execution input for the R57 catalog.

Selects COMPILED rows whose every binding lives in a single dataset (default the
adjusted daily bar surface), takes a deterministic stride sample across the whole
catalog so the sample is not biased to one batch/prefix, and writes the record
shape that evidence/factor_catalog_20260915/smoke_catalog.py already consumes.

Read-only w.r.t. the repo; writes only --out.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--dataset", default="ashare_stock_daily_adj")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    # Pass 1: count eligible rows so the stride is uniform over the catalog.
    eligible = 0
    total = 0
    files = sorted(args.manifest_dir.glob("full-s*.jsonl.gz"))
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                total += 1
                if r.get("compile_status") != "COMPILED":
                    continue
                ds = {b.get("dataset") for b in (r.get("bindings") or [])}
                if ds and ds <= {args.dataset}:
                    eligible += 1

    stride = max(1, eligible // args.sample)
    written = 0
    seen = 0
    ids: list[str] = []
    with gzip.open(args.out, "wt", encoding="utf-8") as dst:
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("compile_status") != "COMPILED":
                        continue
                    ds = {b.get("dataset") for b in (r.get("bindings") or [])}
                    if not (ds and ds <= {args.dataset}):
                        continue
                    if seen % stride:
                        seen += 1
                        continue
                    seen += 1
                    if written >= args.sample:
                        continue
                    dst.write(
                        json.dumps(
                            {
                                "source_row": r.get("source_row"),
                                "id": r.get("id"),
                                "name": r.get("name"),
                                "formula": r.get("r57_formula"),
                                "domain": None,
                                "batch": None,
                                "r20_formula": r.get("r20_formula"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    ids.append(r.get("id"))
                    written += 1

    report = {
        "manifest_dir": str(args.manifest_dir),
        "dataset": args.dataset,
        "total_rows": total,
        "eligible_rows": eligible,
        "stride": stride,
        "written": written,
        "first_ids": ids[:20],
        "last_ids": ids[-10:],
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()

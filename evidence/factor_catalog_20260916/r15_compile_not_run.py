"""Compile-only audit for untouched NOT_RUN formulas in a frozen catalog."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--deadline-seconds", type=float, default=150.0)
    args = ap.parse_args()
    if args.start < 0 or args.limit <= 0 or args.deadline_seconds <= 0:
        ap.error("start must be non-negative; limit/deadline must be positive")
    if args.output.exists() or Path(str(args.output) + ".partial").exists():
        raise FileExistsError(args.output)

    helper_dir = Path(__file__).resolve().parents[1] / "factor_catalog_20260915"
    sys.path.insert(0, str(helper_dir))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor

    source_sha = sha256(args.input)
    parser, engine = build_runtime()
    partial = Path(str(args.output) + ".partial")
    summary_path = args.output.with_suffix(".summary.json")
    started = time.monotonic()
    counts: collections.Counter[str] = collections.Counter()
    selected = processed = 0

    with gzip.open(args.input, "rt", encoding="utf-8-sig", newline="") as src, \
            gzip.open(partial, "xt", encoding="utf-8") as dst:
        for record in csv.DictReader(src):
            if record.get("compile_status") != "NOT_RUN" or not (record.get("id") or "").strip():
                continue
            if selected < args.start:
                selected += 1
                continue
            if processed >= args.limit:
                break
            if time.monotonic() - started >= args.deadline_seconds:
                raise TimeoutError("compile-only audit deadline exceeded")
            formula = record.get("current_formula") or ""
            row = {
                "source_row": int(record["source_row"]),
                "id": record["id"],
                "executed_formula": formula,
                "formula_sha256": hashlib.sha256(formula.encode("utf-8")).hexdigest(),
                "source_csv_sha256": source_sha,
                "scope": "compile_only_no_data_read_not_execution_evidence",
            }
            try:
                if not formula:
                    raise ValueError("MISSING_CURRENT_FORMULA")
                expr = parser.parse(formula)
                bindings, failures = bind_fields(expr)
                row["field_binding_count"] = len(bindings)
                row["field_binding_failures"] = failures
                if failures:
                    raise ValueError("FIELD_BINDING_FAILED")
                factor = Factor(
                    name=str(record["id"]), expr=expr, source_expr=formula,
                    surface="compat_research",
                )
                engine.compile(factor)
                row["compile_status"] = "COMPILED"
            except Exception as exc:
                row.update(
                    compile_status="COMPILE_FAILED",
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            counts[row["compile_status"]] += 1
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed += 1

    os.replace(partial, args.output)
    with gzip.open(args.output, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    summary = {
        "input": str(args.input),
        "input_sha256": source_sha,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "start": args.start,
        "requested": args.limit,
        "processed": processed,
        "counts": dict(counts),
        "seconds": time.monotonic() - started,
        "complete": processed == args.limit,
        "scope": "compile_only_no_data_read_not_execution_evidence",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

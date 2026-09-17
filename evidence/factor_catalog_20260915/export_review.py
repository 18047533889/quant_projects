"""Stream a lightweight review CSV; never overwrite the original factor catalog.

Static status is deliberately separate from real-execution evidence. A smoke
result for a different formula cannot certify a subsequently migrated formula.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import itertools
import os
import tempfile
import json
from pathlib import Path

def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)

def _export_review(source, audit, output, smoke_paths=(), compile_paths=()):
    compilations = {}
    for path in compile_paths:
        for item in records(path):
            compilations[(item["source_row"], item.get("id"))] = {name: item.get(name) for name in ("current_formula", "status", "error")}
            compilations[(item["source_row"], item.get("id"))]["evidence_file"] = path.name
    executions = {}
    for path in smoke_paths:
        # Only explicitly supplied completed gzip evidence is accepted. Corrupt
        # or interrupted files raise rather than silently certifying a prefix.
        for item in records(path):
            key = (item["source_row"], item.get("id"))
            executions[key] = {name: item.get(name) for name in (
                "executed_formula", "status", "error", "value_count", "finite_count",
                "result_hash", "elapsed_seconds", "retry_after_batch_abort",
                "batch_abort_error_type", "batch_abort_error", "backend_path",
                "execution_window", "execution_symbol_count")}
            executions[key]["evidence_file"] = path.name
    counts = collections.Counter()
    fields = ["source_row", "id", "name", "original_formula", "current_formula", "original_fields",
              "original_tables", "domain", "pit", "logic", "batch", "migration_changes",
              "bindings", "static_status", "static_reason", "compile_status", "compile_reason", "execution_status",
              "execution_reason", "value_count", "finite_count", "result_hash",
              "retry_after_batch_abort", "batch_abort_error_type", "batch_abort_error",
              "current_fields", "current_tables", "backend_path",
              "execution_evidence_file", "execution_validation_scope",
              "execution_window", "execution_symbol_count",
              "compile_evidence_file", "compile_validation_scope"]
    ids = set()
    # Exclusive creation prevents accidentally overwriting the input or an
    # earlier delivered review. Write to a caller-selected new version path.
    opener = gzip.open if str(output).endswith(".gz") else open
    with opener(output, "xt", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for original, checked in itertools.zip_longest(records(source), records(audit)):
            if original is None or checked is None:
                raise ValueError("source/audit row counts differ")
            key = (original["source_row"], original.get("id"))
            if key != (checked["source_row"], checked.get("id")):
                raise ValueError(f"source/audit identity mismatch at {key}")
            if original.get("formula") != checked.get("formula"):
                raise ValueError(f"source/audit original formula mismatch at {key}")
            if key[1]:
                if key[1] in ids:
                    raise ValueError(f"duplicate factor id: {key[1]}")
                ids.add(key[1])
            row = {name: original.get(name.removeprefix("original_"), "") for name in fields[:11]}
            row["current_formula"] = checked.get("current_formula", original.get("formula", ""))
            for name in ("migration_changes", "bindings"):
                row[name] = json.dumps(checked.get(name, []), ensure_ascii=False)
            current_bindings = checked.get("bindings", [])
            row["current_fields"] = "|".join(sorted({
                f"{b['table']}.{b['column']}" for b in current_bindings
                if b.get("table") and b.get("column")}))
            row["current_tables"] = "|".join(sorted({
                b["table"] for b in current_bindings if b.get("table")}))
            row["static_status"] = checked["status"]
            row["static_reason"] = checked.get("error") or json.dumps(checked.get("unresolved", []), ensure_ascii=False)
            row["compile_status"] = "NOT_RUN"
            compiled = compilations.get(key)
            if compiled and compiled["current_formula"] == row["current_formula"]:
                row["compile_status"] = compiled["status"]
                row["compile_reason"] = compiled["error"] or ""
                row["compile_evidence_file"] = compiled["evidence_file"]
                row["compile_validation_scope"] = (
                    "historical_formula_matched_not_current_code_certification"
                )
            row["execution_status"] = "NOT_RUN"
            result = executions.get(key)
            if result:
                if result["executed_formula"] != row["current_formula"]:
                    row["execution_reason"] = "Earlier evidence uses a different formula; rerun required"
                else:
                    row["execution_status"] = result["status"]
                    row["backend_path"] = json.dumps(result["backend_path"], ensure_ascii=False)
                    row["execution_evidence_file"] = result["evidence_file"]
                    window = result.get("execution_window")
                    row["execution_window"] = json.dumps(window) if window is not None else ""
                    row["execution_symbol_count"] = result.get("execution_symbol_count")
                    # Matching formula text alone cannot certify a changed
                    # implementation. Keep these historical run facts explicit.
                    row["execution_validation_scope"] = (
                        "historical_formula_matched_not_current_code_certification"
                    )
                    row["execution_reason"] = result["error"] or ""
                    for name in ("value_count", "finite_count", "result_hash",
                                 "retry_after_batch_abort", "batch_abort_error_type",
                                 "batch_abort_error"):
                        row[name] = result[name]
            writer.writerow(row)
            counts["rows"] += 1
            counts["static:" + row["static_status"]] += 1
            counts["execution:" + row["execution_status"]] += 1
            counts["compile:" + row["compile_status"]] += 1
    counts["factor_ids"] = len(ids)
    return dict(counts)

def export_review(source, audit, output, smoke_paths=(), compile_paths=()):
    """Publish only a fully reconciled CSV, atomically and without overwriting."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    # Same-filesystem private staging permits atomic no-clobber publication.
    # Only this small output is staged, never the source CSV or repository.
    with tempfile.TemporaryDirectory(prefix=".review-", dir=output.parent) as staging:
        staged = Path(staging) / output.name
        counts = _export_review(source, audit, staged, smoke_paths, compile_paths)
        os.link(staged, output)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--smoke", action="append", default=[], type=Path)
    ap.add_argument("--compile-evidence", action="append", default=[], type=Path)
    args = ap.parse_args()
    print(json.dumps(export_review(args.source, args.audit, args.output, args.smoke, args.compile_evidence), ensure_ascii=False))

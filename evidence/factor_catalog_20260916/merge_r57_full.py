# -*- coding: utf-8 -*-
"""Merge the R57 compile shards into an aggregate manifest and a new revision CSV.

Contract of the delivered CSV:
* every row of the R20 revision is preserved (stable source_row + id),
* the original user formula and the R20 formula are both kept verbatim,
* the R57 formula produced by the current migration chain is added next to them,
* the compile verdict is the CURRENT tree's verdict, not a copy of R20's,
* execution columns are carried over from R20 but explicitly marked as NOT
  re-verified under R57 -- this file does not claim R57 execution,
* the code identity (git head + digest of every engine/data_access source file)
  is recorded so the manifest cannot be confused with another tree.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

CSV_FIELDS = [
    "source_row", "id", "name",
    "original_formula", "r20_formula", "r57_formula", "migration_changes",
    "compile_status", "compile_error_type", "compile_error",
    "binding_failure_count", "ops_call_count",
    "ops_missing", "ops_not_production_admitted", "ops_experimental",
    "domain", "pit", "logic", "batch", "original_fields", "original_tables",
    "r20_compile_status", "r20_execution_status",
    "r57_execution_status", "r57_execution_note",
    "code_head", "code_hashes_digest",
]

R20_KEEP = ("name", "original_formula", "domain", "pit", "logic", "batch",
            "original_fields", "original_tables", "compile_status", "execution_status")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="R20 revision CSV (gzip)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--csv-out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--expected-total", type=int, default=0)
    args = ap.parse_args()

    shard_files = sorted(args.outdir.glob("full-s*.jsonl.gz"))
    if not shard_files:
        raise SystemExit("no shard files")
    print(f"shard files: {len(shard_files)}", flush=True)

    counts: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    missing_ops: collections.Counter[str] = collections.Counter()
    nonprod_ops: collections.Counter[str] = collections.Counter()
    migrated = 0
    heads: set[str] = set()
    hash_digests: set[str] = set()
    per_shard: dict[str, dict] = {}
    ids_seen: set[tuple[str, str]] = set()
    total = 0

    for path in shard_files:
        summary_path = Path(str(path)[: -len(".jsonl.gz")] + ".summary.json")
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            per_shard[path.stem] = {
                "counts": s.get("counts"),
                "source_factor_count": s.get("source_factor_count"),
                "seconds": s.get("seconds"),
                "catalog_size": (s.get("registry_surface") or {}).get("catalog_size"),
            }
            hb, ha = s.get("code_hashes_before") or {}, s.get("code_hashes_after") or {}
            if hb != ha:
                raise SystemExit(f"code changed during shard {path.name}")
            hash_digests.add(hashlib.sha256(
                json.dumps(hb, sort_keys=True).encode()).hexdigest())
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                total += 1
                key = (str(rec.get("source_row")), rec["id"])
                if key in ids_seen:
                    raise SystemExit(f"duplicate factor identity {key}")
                ids_seen.add(key)
                counts[rec.get("compile_status") or "NULL"] += 1
                if rec.get("compile_status") != "COMPILED":
                    errors[f'{rec.get("compile_error_type")}: {str(rec.get("compile_error"))[:160]}'] += 1
                if rec.get("migration_changes"):
                    migrated += 1
                for op in rec.get("ops_used") or []:
                    if not op.get("resolved"):
                        missing_ops[op["call"]] += 1
                    elif not op.get("production_admitted"):
                        nonprod_ops[op["call"]] += 1
        print(f"  read {path.name}: total={total}", flush=True)

    aggregate = {
        "shards": len(shard_files),
        "total_rows": total,
        "counts": dict(counts),
        "rows_with_migration_changes": migrated,
        "distinct_code_hash_digests": sorted(hash_digests),
        "code_hashes_consistent_across_shards": len(hash_digests) == 1,
        "operators_missing_from_surface": dict(missing_ops.most_common()),
        "operators_not_production_admitted": dict(nonprod_ops.most_common()),
        "top_error_groups": dict(errors.most_common(100)),
        "per_shard": per_shard,
        "scope": "compile_only_real_engine_strict_field_bind_no_read_no_execution",
    }
    if args.expected_total and total != args.expected_total:
        aggregate["TOTAL_MISMATCH"] = {"expected": args.expected_total, "actual": total}
    args.summary_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in aggregate.items()
                      if k not in ("per_shard", "top_error_groups")}, ensure_ascii=False)[:2500], flush=True)

    # ---- second pass: stream R20 for classification columns only, then write CSV ----
    r20: dict[tuple[str, str], dict] = {}
    with gzip.open(args.source, "rt", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("id"):
                r20[(row.get("source_row"), row["id"])] = {k: row.get(k, "") for k in R20_KEEP}
    print(f"r20 rows indexed: {len(r20)}", flush=True)

    head = ""
    if (args.outdir / "full-s0.summary.json").exists():
        head = json.loads((args.outdir / "full-s0.summary.json").read_text()).get("git_head") or ""
    code_digest = sorted(hash_digests)[0] if len(hash_digests) == 1 else "INCONSISTENT"

    written = 0
    with gzip.open(args.csv_out, "wt", encoding="utf-8", newline="") as dst:
        w = csv.DictWriter(dst, fieldnames=CSV_FIELDS)
        w.writeheader()
        for path in shard_files:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    key = (str(rec.get("source_row")), rec["id"])
                    base = r20.get(key) or {}
                    ops = rec.get("ops_used") or []
                    w.writerow({
                        "source_row": rec.get("source_row"),
                        "id": rec["id"],
                        "name": base.get("name") or rec.get("name") or "",
                        "original_formula": rec.get("original_formula") or "",
                        "r20_formula": rec.get("r20_formula") or "",
                        "r57_formula": rec.get("r57_formula") or "",
                        "migration_changes": json.dumps(rec.get("migration_changes") or [], ensure_ascii=False),
                        "compile_status": rec.get("compile_status") or "",
                        "compile_error_type": rec.get("compile_error_type") or "",
                        "compile_error": rec.get("compile_error") or "",
                        "binding_failure_count": len(rec.get("binding_failures") or []),
                        "ops_call_count": len(ops),
                        "ops_missing": ",".join(sorted({o["call"] for o in ops if not o.get("resolved")})),
                        "ops_not_production_admitted": ",".join(
                            sorted({o["call"] for o in ops if o.get("resolved") and not o.get("production_admitted")})),
                        "ops_experimental": ",".join(
                            sorted({o["call"] for o in ops if o.get("status") == "experimental"})),
                        "domain": base.get("domain", ""),
                        "pit": base.get("pit", ""),
                        "logic": base.get("logic", ""),
                        "batch": base.get("batch", ""),
                        "original_fields": base.get("original_fields", ""),
                        "original_tables": base.get("original_tables", ""),
                        "r20_compile_status": base.get("compile_status", ""),
                        "r20_execution_status": base.get("execution_status", ""),
                        "r57_execution_status": "NOT_RUN",
                        "r57_execution_note": "R57 is a compile-only manifest; R20 execution status is carried for reference only and was not re-verified under the current tree.",
                        "code_head": head,
                        "code_hashes_digest": code_digest,
                    })
                    written += 1
    print(f"csv rows written: {written} -> {args.csv_out}", flush=True)
    aggregate["csv_rows_written"] = written
    aggregate["csv_path"] = str(args.csv_out)
    aggregate["csv_sha256"] = hashlib.sha256(args.csv_out.read_bytes()).hexdigest()
    args.summary_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

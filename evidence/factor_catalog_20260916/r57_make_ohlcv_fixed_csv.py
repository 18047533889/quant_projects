# -*- coding: utf-8 -*-
"""Apply the repaired R19 OHLCV migration to a catalog CSV, column-preserving.

Reads a catalog CSV (R20 or R57 shape), rewrites only the current_formula /
r57_formula cells that still carry the generic five-argument OHLCV call for the
four previously-missed indicators, and writes a new CSV with every other column
untouched.

Usage:
    make_ohlcv_fixed_csv.py --source <in.csv.gz> --out <out.csv.gz> \
        [--formula-col current_formula] [--appended-col <copy-of-source>]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from pathlib import Path

TARGETS = ("FisherTransform", "CoppockCurve", "QQE", "ElderRay")


def touched(formula: str) -> bool:
    return any("%s(field('open'" % t in formula for t in TARGETS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--formula-col", default="current_formula")
    ap.add_argument("--append-note-col", default="migration_changes")
    args = ap.parse_args()

    from factor_engine.tools.catalog_r19_operator_recipes import migrate_formula

    with gzip.open(args.source, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        total = changed = failed = 0
        by_op: dict[str, int] = {}
        for row in reader:
            total += 1
            f = row.get(args.formula_col) or ""
            if not touched(f):
                writer.writerow(row)
                continue
            try:
                new, notes = migrate_formula(f, row.get("logic") or "")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("MIGRATE-FAIL %s: %s" % (row.get("id"), exc), file=sys.stderr)
                writer.writerow(row)
                continue
            if new == f:
                writer.writerow(row)
                continue
            row = dict(row)
            row[args.formula_col] = new
            note_col = args.append_note_col
            if note_col in row and notes:
                try:
                    existing = json.loads(row[note_col]) if row[note_col] else []
                except (TypeError, ValueError):
                    existing = [row[note_col]]
                row[note_col] = json.dumps(list(existing) + [
                    n for n in notes if "OHLCV" in n or "SEMANTIC_REPAIR" in n
                ], ensure_ascii=False)
            writer.writerow(row)
            changed += 1
            for t in TARGETS:
                if "%s(field('open'" % t in f:
                    by_op[t] = by_op.get(t, 0) + 1
                    break

    payload = buf.getvalue()
    with gzip.open(args.out, "wt", encoding="utf-8", newline="") as dst:
        dst.write(payload)

    print(json.dumps({
        "source": str(args.source),
        "out": str(args.out),
        "total_rows": total,
        "changed_rows": changed,
        "migration_failures": failed,
        "by_operator": by_op,
        "formula_col": args.formula_col,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Build an execution input for the four indicators the R19 OHLCV repair missed.

Takes exactly the catalog rows whose current_formula still carries the generic
``Ind(Open, High, Low, Close, Volume)`` call, applies the repaired R19 migration,
and writes the record shape smoke_catalog.py consumes so the repaired formulas
can be executed end to end rather than merely compiled.

Read-only w.r.t. the repo; writes only --out / --report.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

TARGETS = ("FisherTransform", "CoppockCurve", "QQE", "ElderRay")


def has_generic_call(formula: str) -> bool:
    return any("%s(field('open'" % t in formula for t in TARGETS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    from factor_engine.tools.catalog_r19_operator_recipes import migrate_formula

    written = 0
    skipped_unrepaired = 0
    by_op: dict[str, int] = {}
    ids: list[str] = []
    with gzip.open(args.source, "rt", encoding="utf-8") as fh, \
            gzip.open(args.out, "wt", encoding="utf-8") as dst:
        for row in csv.DictReader(fh):
            f = row.get("current_formula") or ""
            if not has_generic_call(f):
                continue
            new, changes = migrate_formula(f, row.get("logic") or "")
            if new == f:
                skipped_unrepaired += 1
                continue
            dst.write(json.dumps({
                "source_row": row.get("source_row"),
                "id": row.get("id"),
                "name": row.get("name"),
                "formula": new,
                "domain": row.get("domain") or None,
                "batch": row.get("batch") or None,
                "r20_formula": f,
                "r21_repair": changes,
            }, ensure_ascii=False) + "\n")
            written += 1
            ids.append(row.get("id"))
            for t in TARGETS:
                if "%s(field('open'" % t in f:
                    by_op[t] = by_op.get(t, 0) + 1
                    break

    report = {
        "source": str(args.source),
        "written": written,
        "skipped_unrepaired": skipped_unrepaired,
        "by_operator": by_op,
        "first_ids": ids[:10],
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Show the R19 rewrite applied to the real catalog rows it previously missed.

Reads the R20 CSV, finds the rows still carrying the generic five-argument OHLCV
call for FisherTransform / CoppockCurve / QQE / ElderRay, runs the repaired R19
migration on their current_formula, and prints before/after plus a compile check.
"""
from __future__ import annotations

import csv
import gzip
import sys

from factor_engine.api.dsl_parser import DSLParser
from factor_engine.ir.analyzer import Analyzer
from factor_engine.tools.catalog_r19_operator_recipes import migrate_formula

TARGETS = ("FisherTransform", "CoppockCurve", "QQE", "ElderRay")


def main() -> int:
    path = sys.argv[1]
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = row.get("current_formula") or ""
            if any(f"({t}(field('open'" in f or f"{t}(field('open'" in f for t in TARGETS):
                rows.append(row)
    print("rows still carrying the generic OHLCV call:", len(rows))

    seen: set[str] = set()
    repaired_n = 0
    compile_ok = 0
    compile_bad: list[tuple[str, str]] = []
    for row in rows:
        f = row["current_formula"]
        new, changes = migrate_formula(f, row.get("logic") or "")
        if new == f:
            continue
        repaired_n += 1
        for t in TARGETS:
            if f"{t}(field('open'" in f and t not in seen:
                seen.add(t)
                print()
                print("### %s (example id %s)" % (t, row["id"]))
                print("  BEFORE:", f[:230])
                print("  AFTER :", new[:230])
                print("  note  :", changes[0][:120] if changes else "")
        try:
            Analyzer().lower(DSLParser(surface="compat_research").parse(new))
            compile_ok += 1
        except Exception as exc:  # noqa: BLE001
            compile_bad.append((row["id"], "%s: %s" % (type(exc).__name__, exc)))

    print()
    print("repaired rows:", repaired_n, "| compile ok:", compile_ok, "| compile bad:", len(compile_bad))
    for rid, err in compile_bad[:10]:
        print("   BAD", rid, err[:160])
    missing = [t for t in TARGETS if t not in seen]
    print("targets not exercised:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

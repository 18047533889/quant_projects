"""Check whether the holder_top10_* names really are missing operators.

compile_r57_full.py classifies an operator as "missing" by asking
OperatorRegistry.resolve_canonical_optional, but dsl_parser registers these five
names as native parser macros under the compat_research / all surfaces. If so,
they expand at parse time and never need a registry entry, and the earlier
"compiled green but will explode" call was wrong.

This script (a) reports the compile_status of the catalog rows that reference
them, and (b) tries to parse and lower each macro for real.
"""
from __future__ import annotations

import collections
import csv
import gzip
import sys

MACROS = (
    "holder_top10_pledge_ratio",
    "holder_top10_daily_id_churn",
    "holder_top10_two_day_rank_migration",
    "holder_top10_weighted_std",
    "holder_top10_disclosure_count",
)


def main() -> int:
    path = sys.argv[1]
    per_op_rows = collections.Counter()
    status = collections.defaultdict(collections.Counter)
    examples: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = row.get("r57_formula") or ""
            for m in MACROS:
                if m + "(" in f:
                    per_op_rows[m] += 1
                    status[m][row.get("compile_status")] += 1
                    examples.setdefault(m, f)
    print("=== catalog rows referencing each macro ===")
    for m in MACROS:
        print("  %6d  %-38s compile_status=%s" % (
            per_op_rows[m], m, dict(status[m])))

    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    print()
    print("=== parse + lower under compat_research ===")
    for m in MACROS:
        src = examples.get(m) or '%s("StockTopTenShareholder")' % m
        try:
            plan = DSLParser(surface="compat_research").parse(src)
            ir = Analyzer().lower(plan)
            print("  OK   %-38s lowered, node type %s" % (m, type(ir).__name__))
        except Exception as exc:  # noqa: BLE001
            print("  FAIL %-38s %s: %s" % (m, type(exc).__name__, str(exc)[:110]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

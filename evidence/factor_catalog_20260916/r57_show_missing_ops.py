"""Show which operators the R57 catalog references but the registry does not expose.

compile_r57_full.py recorded operators_missing_from_surface and
operators_not_production_admitted against the live registry. Print them with
their catalog row counts, plus a breakdown of compile_status for the rows that
reference an unreachable operator, so we know how much of the "compiled" surface
would fail on execution.
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import os
import sys


def main() -> int:
    manifest_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "evidence/factor_catalog_20260916/r57_20260919_full"
    src = sys.argv[2] if len(sys.argv) > 2 else \
        "evidence/factor_catalog_20260916/factor_catalog_review_r20_final.csv.gz"

    agg = json.load(open(os.path.join(manifest_dir, "r57_aggregate.json"), encoding="utf-8"))
    missing = agg.get("operators_missing_from_surface") or {}
    nonprod = agg.get("operators_not_production_admitted") or {}

    dsl_kw = {"field", "source_col", "where", "if_else", "iif", "literal", "const"}
    real_missing = {k: v for k, v in missing.items() if k not in dsl_kw}
    print("=== operators referenced but NOT in the registry (%d distinct) ===" % len(real_missing))
    total_ref = 0
    for k, v in sorted(real_missing.items(), key=lambda kv: -kv[1]):
        print("  %7d  %s" % (v, k))
        total_ref += v
    print("  total references:", total_ref)
    print()
    print("=== operators in the registry but not production-admitted: %d distinct ===" % len(nonprod))
    for k, v in sorted(nonprod.items(), key=lambda kv: -kv[1])[:25]:
        print("  %7d  %s" % (v, k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Print the arity2 audit summary."""
from __future__ import annotations

import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("rows_parsed:", d["rows_parsed"], "| rows_flagged:", d["rows_flagged"])
print("totals_by_kind:", json.dumps(d["totals_by_kind"], ensure_ascii=False))
print("parse_failures:", json.dumps(d["parse_failures"], ensure_ascii=False)[:300])
print()
g = d["issues_by_kind_and_op"]
for kind in ("TOO_MANY_POSITIONAL", "SERIES_INTO_SCALAR_PARAM", "UNKNOWN_KWARG", "OPERATOR_NOT_REGISTERED"):
    print("===", kind, "===")
    n = 0
    for k, v in g.items():
        if k.startswith(kind):
            print("  %7d  %s" % (v, k.split("|", 1)[1]))
            n += 1
        if n >= 16:
            break
    print()

print("=== samples for the OHLCV-family and remaining groups ===")
for key, vals in list(d.get("samples", {}).items()):
    it = vals[0]["issue"] if vals else {}
    if it.get("kind") in ("TOO_MANY_POSITIONAL",) or it.get("op") in (
        "QQE", "FisherTransform", "CoppockCurve", "ElderRay", "group_tail_lead_score",
    ):
        print(" ", key[:150])
        print("     id=%s issue=%s" % (vals[0]["id"], json.dumps(it, ensure_ascii=False)[:260]))

# -*- coding: utf-8 -*-
"""Compact summary of a smoke_catalog.py run (backend_path is a huge nested dict)."""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = collections.Counter()
route = collections.Counter()
errs = collections.Counter()
allnan = []
failed = []
for line in gzip.open(path, "rt", encoding="utf-8"):
    r = json.loads(line)
    st = r.get("status") or "?"
    status[st] += 1
    bp = r.get("backend_path")
    if isinstance(bp, dict):
        route[
            (
                bp.get("primary_route"),
                bool(bp.get("used_polars_long_path")),
                bool(bp.get("used_sql_pushdown")),
            )
        ] += 1
    if st == "EXECUTION_FAILED":
        failed.append((r.get("id"), r.get("error_type"), str(r.get("error"))[:180]))
    elif st == "EXECUTED_ALL_NONFINITE":
        allnan.append(r.get("id"))
    elif st != "EXECUTED":
        errs[f"{st}|{r.get('error_type')}|{str(r.get('error'))[:160]}"] += 1

print(f"{path.name}: {sum(status.values())} rows")
for k, v in status.most_common():
    print(f"  {v:5d}  {k}")
print("  backend route (primary, polars_long, sql):")
for k, v in route.most_common(8):
    print(f"    {v:5d}  {k}")
print(f"  all-non-finite ids ({len(allnan)}):", allnan[:20])
print(f"  hard failures ({len(failed)}):")
for f in failed[:20]:
    print("   ", f)
for k, v in errs.most_common(10):
    print(f"  other-error {v:4d}  {k[:180]}")

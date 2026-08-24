# -*- coding: utf-8 -*-
"""R30 Phase 1b: risky-name + pit_safe=false scan on the FULL load_all registry.

Reuses the 1436-canonical inventory saved by the caller.  Runs entirely inside
the same process that already ran load_all() (73s), so scan here is cheap.
"""
from __future__ import annotations

import sys
import json
import time

sys.path.insert(0, ".")

t0 = time.time()
from factor_engine.cleaned_operators import load_all
load_all()
from factor_engine.cleaned_operators.registry import OperatorRegistry as R
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy

# --- risky names across the FULL registry (executable = has runtime backend) ---
RISKY = ("rand_", "shuffle", "sample", "lead", "next", "bfill", "interpolate")
risky = {}
for c in R.list_canonical():
    cl = c.lower()
    if any(p in cl for p in RISKY):
        backends = sorted(R._operators.get(c, {}).keys())
        risky[c] = {
            "surface": classify_canonical(c),
            "backends": backends,
            "status": R._catalog.get(c, {}).get("status"),
            "has_executable": bool(backends),
        }
print("RISKY (any):", len(risky))
for k in sorted(risky):
    print("  ", k, risky[k])

# --- pit_safe=False per surface (full registry) ---
pit = {}
for c in R.list_canonical():
    backends_map = R._operators.get(c)
    if not backends_map:
        continue
    op = backends_map.get("pandas_numpy") or next(iter(backends_map.values()))
    try:
        pol = infer_operator_policy(op, canonical=c)
    except Exception:
        continue
    if not pol.pit_safe:
        pit.setdefault(classify_canonical(c), []).append(c)
print("\nPIT_SAFE=FALSE per surface:")
for s in sorted(pit):
    print(" ", s, len(pit[s]))
    for c in sorted(pit[s]):
        print("    ", c)

# --- research / unsafe / internal / legacy on full registry ---
for s in ("research", "unsafe", "internal", "legacy"):
    lst = [c for c in R.list_canonical() if classify_canonical(c) == s]
    print(f"\n[{s}] ({len(lst)}):")
    for c in sorted(lst):
        print("   ", c, "status=", R._catalog.get(c, {}).get("status"))

# save
out = {"risky": risky, "pit_false": {s: sorted(v) for s, v in pit.items()}}
json.dump(out, open("factor_engine/docs/r30_phase1b_scan.json", "w"), indent=2, sort_keys=True)
print("\nsaved factor_engine/docs/r30_phase1b_scan.json  elapsed", round(time.time() - t0, 1), "s")

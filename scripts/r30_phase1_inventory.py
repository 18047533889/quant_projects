# -*- coding: utf-8 -*-
"""R30 Phase 1: fresh current canonical inventory (no hardcoded 1362).

Prints the current registry state so subsequent phases work off reality.
"""
from __future__ import annotations

import sys
import json
import time

sys.path.insert(0, ".")

t0 = time.time()
import factor_engine.cleaned_operators  # noqa: F401
from factor_engine.cleaned_operators.registry import OperatorRegistry

# classify
try:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical, is_dsl_name_allowed
except Exception as e:  # pragma: no cover
    classify_canonical = None
    is_dsl_name_allowed = None
    print("classify import err:", e)

cats = OperatorRegistry._catalog
ops = OperatorRegistry._operators
alis = OperatorRegistry._aliases

# Fresh canonical set = registered runtime keys (not hardcoded)
runtime_canons = sorted(set(ops.keys()) | set(cats.keys()))
with_runtime = sorted(k for k in runtime_canons if ops.get(k))

surfaces = {}
if classify_canonical is not None:
    for c in with_runtime:
        try:
            surfaces[c] = classify_canonical(c)
        except Exception as e:
            surfaces[c] = f"ERR:{e}"

from collections import Counter
sc = Counter(surfaces.values())
print("fresh runtime canonicals:", len(with_runtime))
print("fresh catalog canonicals:", len(runtime_canons))
print("surface counts:", dict(sc))
print("aliases:", len(alis))
print("load time:", round(time.time() - t0, 1), "s")

# inventory of risky names present in registry (active = has executable backend)
RISKY_PREFIX = ("rand_", "shuffle", "sample", "lead", "next", "bfill", "fillna_interpolate", "causal_bfill")
risky = {}
for c in with_runtime:
    cl = c.lower()
    if any(p in cl for p in ("rand_", "shuffle", "sample", "lead", "next", "bfill", "interpolate")):
        risky[c] = {
            "surface": surfaces.get(c),
            "backends": sorted(ops.get(c, {}).keys()),
            "status": cats.get(c, {}).get("status"),
        }
print("\nRISKY ACTIVE NAMES:", len(risky))
for k in sorted(risky):
    print("  ", k, risky[k])

# pit_safe=False on daily surface
print("\nDAILY PIT_SAFE=FALSE:")
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
pit_false = []
for c in with_runtime:
    if surfaces.get(c) == "daily":
        try:
            op = ops[c].get("pandas_numpy") or next(iter(ops[c].values()))
            pol = infer_operator_policy(op, canonical=c)
            if not pol.pit_safe:
                pit_false.append((c, cats.get(c, {}).get("status")))
        except Exception:
            pass
for c, st in sorted(pit_false):
    print("  ", c, "status=", st)
print("count:", len(pit_false))

# research lifecycle present in catalog
research_life = sorted(c for c, v in cats.items() if str(v.get("status", "")).lower() == "research")
print("\nRESEARCH status count:", len(research_life))

# save snapshot
snap = {
    "head": "700f0e7e20acd7bb65b44e372d5c8df6f5739df0",
    "runtime_canonicals": with_runtime,
    "catalog_canonicals": runtime_canons,
    "surface_counts": dict(sc),
    "n_aliases": len(alis),
    "risky_active": risky,
    "daily_pit_safe_false": [c for c, _ in pit_false],
}
with open("factor_engine/docs/r30_phase1_inventory.json", "w") as f:
    json.dump(snap, f, indent=2, sort_keys=True)
print("\nsaved factor_engine/docs/r30_phase1_inventory.json")

# -*- coding: utf-8 -*-
"""R30 Phase 3: per-module operator status scan to drive the
PRODUCTION_LOAD_MODULES / RESEARCH_LOAD_MODULES / INTERNAL_KERNEL_MODULES split.

For each module in ``_LOAD_MODULES`` + ``_REVIEWED_EXTENSIONS``, report how many
canonicals it registered and their lifecycle status (production / research /
experimental / deprecated / stub / doc_only / implemented-unknown).
"""
from __future__ import annotations

import sys
import json
from collections import Counter, defaultdict

sys.path.insert(0, ".")

# Import in the same order as the loader but STOP before the promotion layers
# mutate statuses, so we observe the RAW registered lifecycle.
import factor_engine.cleaned_operators as co

from factor_engine.cleaned_operators.registry import OperatorRegistry

# Snapshot raw registered statuses the same way _load_all_impl does (the loader
# snapshots BEFORE dedupe/hardening rewrite statuses).
raw_status: dict[str, str] = {}
for canonical, catalog in OperatorRegistry._catalog.items():
    raw_status[canonical] = str((catalog or {}).get("status", "implemented") or "implemented").lower()

# Map canonical -> registering module (first backend class module).
canon_to_mod: dict[str, str] = {}
for canonical, backends in OperatorRegistry._operators.items():
    for backend, op in backends.items():
        mod = type(op).__module__
        canon_to_mod[canonical] = mod
        break

# modules of interest
ALL_MODULES = list(co._LOAD_MODULES) + list(co._REVIEWED_EXTENSIONS)

per_mod: dict[str, dict] = {}
for mod in ALL_MODULES:
    per_mod[mod] = {"total": 0, "status": Counter(), "canonicals": []}

for canonical in sorted(canon_to_mod):
    mod = canon_to_mod[canonical]
    if mod in per_mod:
        st = raw_status.get(canonical, "implemented")
        per_mod[mod]["total"] += 1
        per_mod[mod]["status"][st] += 1
        per_mod[mod]["canonicals"].append(canonical)

# Also: which top-level packages register research/experimental/deprecated canonicals
pkg_status = Counter()
for canonical, st in raw_status.items():
    if st in ("research", "experimental", "deprecated"):
        mod = canon_to_mod.get(canonical, "?")
        pkg = ".".join(mod.split(".")[:3]) if mod.startswith("cleaned_operators") else mod
        pkg_status[(pkg, st)] += 1

print("=== modules that register research/experimental/deprecated canonicals ===")
flagged = []
for mod in sorted(per_mod):
    st = per_mod[mod]["status"]
    if st.get("research", 0) or st.get("experimental", 0) or st.get("deprecated", 0) or st.get("doc_only", 0) or st.get("stub", 0):
        flagged.append(mod)
        print(f"{mod}: total={per_mod[mod]['total']} {dict(st)}")
        # print non-implemented canonicals
        nonimpl = [c for c in per_mod[mod]['canonicals'] if raw_status.get(c, 'implemented') not in ('implemented', 'production')]
        if nonimpl:
            print(f"    non-implemented: {nonimpl[:8]}")

print("\n=== top-level package status rollup ===")
for (pkg, st), cnt in sorted(pkg_status.items()):
    print(f"  {pkg:60s} {st:14s} {cnt}")

json.dump({
    "per_module": {m: {"total": v["total"], "status": dict(v["status"])} for m, v in per_mod.items()},
    "flagged_modules": flagged,
}, open("docs/r30_phase3_module_scan.json", "w"), indent=2, sort_keys=True)
print("\nsaved docs/r30_phase3_module_scan.json")

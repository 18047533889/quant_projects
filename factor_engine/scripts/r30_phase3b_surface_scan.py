# -*- coding: utf-8 -*-
"""R30 Phase 3b: per-module SURFACE scan AFTER full load_all (promotion layers
applied).  Drives which modules must move to RESEARCH_LOAD_MODULES /
INTERNAL_KERNEL_MODULES.

A module belongs to the production loader when every canonical it registers
lands on daily/extended and is production-certified; any module that registers
research/unsafe-surface canonicals must be opt-in research.
"""
from __future__ import annotations

import sys
import json
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from cleaned_operators import load_all
load_all()

from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_surface import classify_canonical

canon_to_mod: dict[str, str] = {}
for canonical, backends in OperatorRegistry._operators.items():
    for backend, op in backends.items():
        canon_to_mod[canonical] = type(op).__module__
        break

import cleaned_operators as co
ALL_MODULES = list(co._LOAD_MODULES) + list(co._REVIEWED_EXTENSIONS)

per_mod: dict[str, dict] = {}
for mod in ALL_MODULES:
    per_mod[mod] = {"total": 0, "surface": Counter()}

for canonical in sorted(canon_to_mod):
    mod = canon_to_mod[canonical]
    if mod in per_mod:
        per_mod[mod]["total"] += 1
        per_mod[mod]["surface"][classify_canonical(canonical)] += 1

# modules that register any research/unsafe/internal surface
print("=== modules registering research/unsafe/internal surface ===")
for mod in sorted(per_mod):
    s = per_mod[mod]["surface"]
    nonprod = sum(s.get(k, 0) for k in ("research", "unsafe", "internal", "legacy"))
    if nonprod:
        print(f"{mod}: total={per_mod[mod]['total']} {dict(s)}")

print("\n=== R30-named modules ===")
for mod in ["cleaned_operators.research_polars",
            "cleaned_operators.cross_section.panel_model",
            "cleaned_operators.research_transform",
            "cleaned_operators.dmd",
            "cleaned_operators.research_spectral",
            "cleaned_operators.ts_model.dynamic_regression",
            "cleaned_operators.ts_model.ar_meanrev",
            "cleaned_operators.ts_model.state_space",
            "cleaned_operators.ts_model.volatility",
            "cleaned_operators.ts_model.complexity",
            "cleaned_operators.ts_model.wavelet_spectral",
            "cleaned_operators.ts_model.sequence_anomaly",
            "cleaned_operators.ts_model.path_signature",
            "cleaned_operators.ts_model.polars_regression"]:
    if mod in per_mod:
        print(f"{mod}: total={per_mod[mod]['total']} {dict(per_mod[mod]['surface'])}")
    else:
        print(f"{mod}: <no direct canonicals (helper module)>")

json.dump({
    "per_module_surface": {m: {"total": v["total"], "surface": dict(v["surface"])} for m, v in per_mod.items()},
}, open("docs/r30_phase3b_surface_scan.json", "w"), indent=2, sort_keys=True)
print("\nsaved")

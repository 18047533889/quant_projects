# -*- coding: utf-8 -*-
"""R30 Phase 3c: per-canonical final surface + certification for the modules
R30 §7 names, to decide production vs research loader placement precisely."""
from __future__ import annotations

import sys
import json

sys.path.insert(0, ".")
from factor_engine.cleaned_operators import load_all
load_all()
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_surface import classify_canonical

canon_to_mod = {}
for canonical, backends in OperatorRegistry._operators.items():
    for backend, op in backends.items():
        canon_to_mod[canonical] = type(op).__module__
        break

TARGET_MODS = [
    "factor_engine.cleaned_operators.research_polars",
    "factor_engine.cleaned_operators.cross_section.panel_model",
    "factor_engine.cleaned_operators.research_transform",
    "factor_engine.cleaned_operators.dmd",
    "factor_engine.cleaned_operators.research_spectral",
    "factor_engine.cleaned_operators.ts_model.dynamic_regression",
    "factor_engine.cleaned_operators.ts_model.ar_meanrev",
    "factor_engine.cleaned_operators.ts_model.state_space",
    "factor_engine.cleaned_operators.ts_model.volatility",
    "factor_engine.cleaned_operators.ts_model.complexity",
    "factor_engine.cleaned_operators.ts_model.wavelet_spectral",
    "factor_engine.cleaned_operators.ts_model.sequence_anomaly",
    "factor_engine.cleaned_operators.ts_model.path_signature",
    "factor_engine.cleaned_operators.ts_model.polars_regression",
]

from collections import Counter
by_cert = Counter()
per_mod = {}
for mod in TARGET_MODS:
    canons = [c for c, m in canon_to_mod.items() if m == mod]
    rows = []
    for c in sorted(canons):
        cat = OperatorRegistry._catalog.get(c, {})
        cert = cat.get("production_certified")
        rows.append({"canonical": c, "surface": classify_canonical(c),
                     "production_certified": cert, "status": cat.get("status")})
        by_cert[(classify_canonical(c), bool(cert))] += 1
    per_mod[mod] = rows

for mod, rows in per_mod.items():
    if not rows:
        print(f"{mod}: <no canonicals>")
        continue
    cert_counts = Counter((r["surface"], bool(r["production_certified"])) for r in rows)
    print(f"{mod}: {len(rows)} canonicals {dict((str(k), v) for k, v in cert_counts.items())}")

print("\n=== aggregate by (surface, certified) ===")
for k, v in sorted(by_cert.items()):
    print("  ", k, v)

json.dump({m: r for m, r in per_mod.items()}, open("docs/r30_phase3c_detail.json", "w"), indent=2, sort_keys=True)
print("saved")

# -*- coding: utf-8 -*-
"""R30 Phase 14: per-canonical final review for every current canonical.

One row per canonical with the disposition / audit dimensions; the disposition
is derived from actual registry state (surface / lifecycle / pit_safe / six-gate
certification / tombstone), not hardcoded.  Every retained canonical must carry
a real test (checked against the test suite path registry in a later phase).
"""
from __future__ import annotations

import sys
import json
import csv

sys.path.insert(0, ".")
from factor_engine.cleaned_operators import load_all
load_all()

from factor_engine.cleaned_operators.registry import OperatorRegistry as R
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
from factor_engine.cleaned_operators.operator_spec import _infer_panel_params, _infer_status, _compute_allow_in_production
from factor_engine.cleaned_operators.tombstones import ALL_TOMBSTONED_NAMES
from factor_engine.runtime.execution_contract import execution_contract

rows = []
for canonical in sorted(R.list_canonical()):
    backends_map = R._operators.get(canonical, {})
    op = backends_map.get("pandas_numpy") or (next(iter(backends_map.values())) if backends_map else None)
    catalog = R._catalog.get(canonical, {})
    surface = classify_canonical(canonical)
    status = _infer_status(catalog) if op is not None else None
    policy = infer_operator_policy(op, canonical=canonical) if op is not None else None
    pit_safe = bool(getattr(policy, "pit_safe", False)) if policy else False
    production_certified = bool(catalog.get("production_certified"))
    shape_preserving = bool(getattr(policy, "shape_preserving", True)) if policy else True
    allow = False
    if op is not None and status is not None:
        allow = _compute_allow_in_production(
            canonical, status=status, pit_safe=pit_safe, shape_preserving=shape_preserving
        )

    # Final disposition (R30 §4 taxonomy).
    if canonical in ALL_TOMBSTONED_NAMES or (canonical in R._catalog and not backends_map and not catalog.get("source")):
        disposition = "DELETE_TOMBSTONE"
    elif surface == "research":
        disposition = "MOVE_RESEARCH_ISOLATED"
    elif surface == "internal":
        disposition = "MOVE_INTERNAL"
    elif surface == "unsafe":
        disposition = "DELETE_COMPLETELY"
    elif surface == "legacy":
        disposition = "DELETE_TOMBSTONE"
    elif allow:
        disposition = "KEEP_PRODUCTION"
    elif surface in ("daily", "extended"):
        disposition = "KEEP_PRODUCTION_CONTEXTUAL" if not pit_safe else "KEEP_PRODUCTION_PENDING_EVIDENCE"
    else:
        disposition = "KEEP_INTERMEDIATE"

    ec = execution_contract(canonical) if op is not None else None
    rows.append({
        "canonical": canonical,
        "aliases": sorted(R._aliases.get(k) for k in (R._aliases or {}) if R._aliases[k] == canonical),
        "source": catalog.get("source", ""),
        "backends": sorted(backends_map.keys()) or [""],
        "surface": surface,
        "lifecycle": str(status or ""),
        "role": "factor" if surface in ("daily", "extended") else surface,
        "public_loaded": bool(backends_map),
        "raw_registry_callable": bool(backends_map),
        "pit_safe": pit_safe,
        "production_certified": production_certified,
        "allow_in_production": allow,
        "state_model": str(getattr(ec, "state_model", "")),
        "chunking": str(getattr(ec, "chunking", "")),
        "legacy_seed_fallback": bool(getattr(ec, "legacy_seed_fallback", False)),
        "final_disposition": disposition,
    })

# counts by disposition
from collections import Counter
disc = Counter(r["final_disposition"] for r in rows)
print("per-canonical rows:", len(rows))
for d, c in sorted(disc.items()):
    print(f"  {d}: {c}")

csv_fields = list(rows[0].keys())
with open("evidence/factor_engine/r30/R30_PER_CANONICAL_FINAL_REVIEW.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    w.writerows(rows)
json.dump(rows, open("evidence/factor_engine/r30/R30_PER_CANONICAL_FINAL_REVIEW.json", "w"), indent=2, sort_keys=True)
print("saved R30_PER_CANONICAL_FINAL_REVIEW.csv/json")

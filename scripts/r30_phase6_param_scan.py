# -*- coding: utf-8 -*-
"""R30 Phase 6: quantify param-kind + param-role explicitness on production ops."""
from __future__ import annotations

import sys
import json
from collections import Counter

sys.path.insert(0, ".")
from cleaned_operators import load_all
load_all()

from cleaned_operators.registry import OperatorRegistry as R
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.operator_spec import _infer_panel_params
from cleaned_operators.base import ParamRole, MISSING

# production = daily + extended surface
prod = [c for c in R.list_canonical() if classify_canonical(c) in ("daily", "extended") and R._operators.get(c)]

stats = {
    "production_total": len(prod),
    "declared_panel_params": 0,
    "fallback_panel_params": 0,
    "no_param_specs": 0,
    "param_role_cover": 0,
    "scalar_params_total": 0,
    "scalar_params_with_role": 0,
    "scalar_params_missing_role": 0,
}
missing_role_examples = []
fallback_examples = []
for c in sorted(prod):
    op = R.get(c, mode="any")
    meta = getattr(op, "metadata", None)
    if meta is None:
        continue
    declared = tuple(getattr(meta, "panel_params", None) or ())
    cat = R._catalog.get(c, {})
    if declared:
        stats["declared_panel_params"] += 1
    else:
        inferred = _infer_panel_params(op, meta, cat)
        if inferred:
            stats["fallback_panel_params"] += 1
            if len(fallback_examples) < 10:
                fallback_examples.append(c)
    specs = getattr(meta, "param_specs", None) or {}
    if not specs:
        stats["no_param_specs"] += 1
        continue
    # scalar params = params that are NOT panel params (declared or inferred)
    panels = set(declared) or set(_infer_panel_params(op, meta, cat))
    scalars = [p for p in (getattr(meta, "param_names", None) or ()) if p not in panels]
    for p in scalars:
        spec = specs.get(p)
        if spec is None:
            continue
        stats["scalar_params_total"] += 1
        role = getattr(spec, "param_role", None)
        if role is not None:
            stats["scalar_params_with_role"] += 1
        else:
            stats["scalar_params_missing_role"] += 1
            if len(missing_role_examples) < 20:
                missing_role_examples.append(f"{c}.{p}")

print("=== R30 Phase 6 param scan ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
print("fallback panel examples:", fallback_examples)
print("missing role examples:", missing_role_examples)

json.dump({"stats": stats, "fallback_examples": fallback_examples,
           "missing_role_examples": missing_role_examples},
          open("docs/r30_phase6_param_scan.json", "w"), indent=2, sort_keys=True)
print("saved")

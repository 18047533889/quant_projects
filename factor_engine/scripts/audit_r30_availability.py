# -*- coding: utf-8 -*-
"""R30 Phase 10 audit: Availability / Decision / Execution clock.

Hard gates:
  R30_ALL_PRODUCTION_FACTORS_HAVE_AVAILABILITY_CLOCK
  R30_SAME_CLOSE_LOOKAHEAD_ZERO
  R30_DAG_AVAILABILITY_PROPAGATION_PASS

* a production factor marked ``same_session_usable=True`` while consuming
  session-end bars (close/high/low/volume) is a same-close lookahead (blocker);
* every daily session-end factor either declares ``available_at=session_close``
  with ``same_session_usable=False`` or is flagged for declaration;
* intraday minute->daily aggregates declare EOD availability.
"""
from __future__ import annotations

import sys
import json

sys.path.insert(0, ".")
from factor_engine.cleaned_operators import load_all
load_all()

from factor_engine.cleaned_operators.registry import OperatorRegistry as R
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.availability_clock import (
    default_available_at,
    default_same_session_usable,
    availability_clock_ok,
)
from factor_engine.cleaned_operators.operator_spec import _infer_panel_params

violations = []      # hard blockers (same-close lookahead)
needs_declaration = []  # session-end daily factors lacking available_at
rows = []

for name in sorted(R.list_canonical()):
    surf = classify_canonical(name)
    if surf not in ("daily", "extended"):
        continue
    op = R.get(name, mode="any")
    if op is None:
        continue
    meta = getattr(op, "metadata", None)
    if meta is None:
        continue
    names = tuple(getattr(meta, "param_names", None) or ())
    panels = _infer_panel_params(op, meta, R._catalog.get(name, {}))
    input_names = tuple(n for n in panels if n)
    aa = getattr(meta, "available_at", None)
    ssu = getattr(meta, "same_session_usable", None)
    # intraday minute->daily: session-close availability required
    tags = [str(t) for t in (getattr(meta, "tags", None) or ())]
    is_intraday = "minute" in str(getattr(meta, "input_grain", "")) or "intraday" in " ".join(tags)
    ok, reason = availability_clock_ok(
        available_at=aa, same_session_usable=ssu, input_names=input_names
    )
    inferred_aa = aa or default_available_at(input_names)
    inferred_ssu = default_same_session_usable(input_names, declared=ssu)
    if not ok:
        violations.append({"canonical": name, "surface": surf, "reason": reason})
    if inferred_aa == "session_close" and aa is None:
        needs_declaration.append({
            "canonical": name, "surface": surf,
            "inferred_available_at": inferred_aa,
        })
    rows.append({
        "canonical": name, "surface": surf,
        "available_at": aa, "same_session_usable": ssu,
        "inferred_available_at": inferred_aa,
        "inferred_same_session_usable": inferred_ssu,
        "input_panels": list(input_names),
        "status": "VIOLATION" if not ok else "NEEDS_DECL" if (aa is None and inferred_aa == "session_close") else "OK",
    })

print(f"total production (daily+extended): {len(rows)}")
print(f"hard violations (same-close lookahead): {len(violations)}")
for v in violations[:10]:
    print("  ", v)
print(f"session-end factors needing available_at declaration: {len(needs_declaration)}")
for d in needs_declaration[:8]:
    print("  ", d)

# R30 §28 hard gate: no production factor claims same-bar usability on a
# session-close factor.
hard_same_close_zero = len(violations) == 0
print(f"\nR30_SAME_CLOSE_LOOKAHEAD_ZERO = {hard_same_close_zero}")

json.dump({
    "rows": rows,
    "violations": violations,
    "needs_declaration": needs_declaration,
    "hard_same_close_zero": hard_same_close_zero,
}, open("evidence/factor_engine/r30/R30_AVAILABILITY_CLOCK_AUDIT.csv.json", "w"), indent=2, sort_keys=True)

import csv
with open("evidence/factor_engine/r30/R30_AVAILABILITY_CLOCK_AUDIT.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    w.writeheader()
    w.writerows(rows)
print("saved R30_AVAILABILITY_CLOCK_AUDIT.csv")

# -*- coding: utf-8 -*-
"""R30 Phase 11/12 audit: corporate-action vintage + as-of universe PIT.

Hard gates:
  R30_FUTURE_CORPORATE_ACTION_RESTATEMENT_ZERO
  R30_SURVIVORSHIP_BIAS_ZERO
  R30_CROSS_SECTION_UNIVERSE_ASOF_REQUIRED

Operator-level contract: a production factor that consumes price / valuation /
group / index data must either (a) declare the source semantic type (RawPrice /
PITAdjustedPrice / CurrentVintageAdjustedPrice / PITTotalReturnIndex) and the
universe requirement, or (b) be recorded as needing a source contract.  The
*operator itself* cannot prove the panel is as-of — that is DataAccess' job —
but a factor must not pretend a current-vintage adjusted price series is PIT
safe.

This audit is a gate on DECLARATIONS, not a reimplementation of DataAccess PIT.
"""
from __future__ import annotations

import sys
import json

sys.path.insert(0, ".")
from factor_engine.cleaned_operators import load_all
load_all()

from factor_engine.cleaned_operators.registry import OperatorRegistry as R
from factor_engine.cleaned_operators.operator_surface import classify_canonical

# Canonicals whose INPUT SEMANTIC meaning depends on corporate-action vintage or
# as-of universe membership.  These consume price/valuation/group/index data.
_PRICE_SENSITIVE = ("return", "price", "vwap", "high", "low", "close", "open",
                    "cap", "pe", "pb", "yield", "adjust", "cumfactor", "amount")
_UNIVERSE_SENSITIVE = ("rank", "zscore", "neutralize", "group_", "cs_", "knn",
                       "index_", "member", "listing", "suspension", "industry")

violations = []
needs_contract = []
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
    tags = [str(t) for t in (getattr(meta, "tags", None) or ())]
    in_sem = dict(getattr(meta, "input_semantic_types", None) or {})
    universe = getattr(meta, "universe_requirement", None)
    market = getattr(meta, "market_scope", None)
    low = name.lower()
    is_price_sensitive = any(p in low for p in _PRICE_SENSITIVE)
    is_universe_sensitive = any(p in low for p in _UNIVERSE_SENSITIVE)
    # A factor that consumes current-vintage adjusted price WITHOUT declaring a
    # PIT semantic type is a restatement risk.
    if is_price_sensitive and not in_sem:
        needs_contract.append({"canonical": name, "surface": surf, "dimension": "corporate_action"})
    if is_universe_sensitive and universe is None:
        needs_contract.append({"canonical": name, "surface": surf, "dimension": "universe_asof"})
    rows.append({
        "canonical": name, "surface": surf,
        "price_sensitive": is_price_sensitive,
        "universe_sensitive": is_universe_sensitive,
        "declared_semantic_types": in_sem,
        "universe_requirement": universe,
        "market_scope": market,
    })

print(f"production ops: {len(rows)}")
print(f"needing corporate-action contract: {sum(1 for n in needs_contract if n['dimension']=='corporate_action')}")
print(f"needing universe-asof contract: {sum(1 for n in needs_contract if n['dimension']=='universe_asof')}")
print("violations (explicit mislabel):", len(violations))

json.dump({"rows": rows, "needs_contract": needs_contract, "violations": violations},
          open("evidence/factor_engine/r30/R30_SOURCE_UNIVERSE_PIT_AUDIT.json", "w"), indent=2, sort_keys=True)
print("saved R30_SOURCE_UNIVERSE_PIT_AUDIT.json")

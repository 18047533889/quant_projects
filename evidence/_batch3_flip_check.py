# -*- coding: utf-8 -*-
"""Batch 3: confirm the execution-kind classification flip for the 13 declared ops.

Before the patch `canonical_polars_kind(..., production_mode=True)` returned
UNSUPPORTED for every one of them (recorded in evidence/_resolve_batch3.json).
This script re-reads the live registry after the patch.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import (  # noqa: E402
    canonical_polars_kind,
    get_physical_spec,
)

OPS = [
    "price_impact", "efficiency_ratio", "turnover_zscore",
    "fin_quarter_from_cumulative", "relative_volume", "intra_realized_skewness",
    "amihud_illiquidity", "intra_realized_variance", "ts_pct", "signed_sqrt",
    "sigmoid", "intra_realized_kurtosis", "log_abs",
]

load_all()

rows = []
bad = []
for c in OPS:
    kind = canonical_polars_kind(c, production_mode=True)
    kv = getattr(kind, "value", str(kind))
    op = OperatorRegistry.get(c, "polars", mode="any")
    spec = get_physical_spec(op)
    rec = {
        "operator": c,
        "polars_kind_production": kv,
        "spec_present": spec is not None,
        "execution_kind": getattr(getattr(spec, "execution_kind", None), "value", None),
        "supports_lazy": getattr(spec, "supports_lazy", None),
        "supports_streaming": getattr(spec, "supports_streaming", None),
        "materializes_full_panel": getattr(spec, "materializes_full_panel", None),
        "kernel_identity": getattr(spec, "kernel_identity", None),
        "backends": sorted(OperatorRegistry.backends_for(c)),
    }
    rows.append(rec)
    if kv != "polars_native":
        bad.append(c)
    print(json.dumps(rec, ensure_ascii=False), flush=True)

out = os.environ.get("FLIP_OUT") or "/home/sunhaiwei/quant_projects/evidence/_batch3_classification_flip.json"
json.dump({"n": len(rows), "flipped": len(rows) - len(bad), "not_flipped": bad, "rows": rows},
          open(out, "w"), indent=1, ensure_ascii=False)
print(f"\nFLIPPED {len(rows) - len(bad)}/{len(rows)}  NOT_FLIPPED={bad}")
print("WROTE", out)
sys.exit(0 if not bad else 1)

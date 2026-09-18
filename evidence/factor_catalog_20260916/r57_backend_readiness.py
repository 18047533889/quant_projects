# -*- coding: utf-8 -*-
"""Backend / speed readiness matrix for the operator surface.

Answers: which operators can actually run on a fast (Polars-native / streaming)
path in production mode, how many fall back or are unclassified, and how many
factor rows in the current catalog depend on each class.
"""
from __future__ import annotations

import collections
import json
import sys

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

cat = OperatorRegistry._catalog
kinds = collections.Counter()
accelerators = collections.Counter()
backend_sets = collections.Counter()
no_spec = []
polars_delegate_like = []

for canonical in sorted(cat):
    try:
        backends = tuple(sorted(map(str, OperatorRegistry.backends_for(canonical) or [])))
    except Exception:  # noqa: BLE001
        backends = ()
    backend_sets[backends] += 1
    if "polars" not in backends:
        kinds["NO_POLARS_SLOT"] += 1
        continue
    op = OperatorRegistry.get(canonical, "polars", mode="any")
    if op is None:
        kinds["POLARS_SLOT_UNRESOLVABLE"] += 1
        continue
    spec = getattr(op, "_physical_spec", None)
    if spec is None:
        no_spec.append(canonical)
        kinds["NO_EXPLICIT_PHYSICAL_SPEC"] += 1
        src = str(((cat[canonical].get("backend_meta") or {}).get("polars") or {}).get("source") or "")
        if "bridge" in src or "delegate" in src or "fallback" in src:
            polars_delegate_like.append((canonical, src))
        continue
    ek = getattr(spec.execution_kind, "value", spec.execution_kind)
    kinds[f"SPEC:{ek}"] += 1
    acc = getattr(spec.accelerator, "value", spec.accelerator)
    accelerators[str(acc)] += 1

print("=== polars slot classification (production-relevant) ===")
for k, v in kinds.most_common():
    print(f"  {k:40s} {v}")
print("\n=== explicit accelerator declarations ===")
for k, v in accelerators.most_common():
    print(f"  {k:40s} {v}")

print("\n=== backend sets (top 15) ===")
for k, v in backend_sets.most_common(15):
    print(f"  {v:6d}  {k}")

print(f"\n=== NO explicit PhysicalImplementationSpec: {len(no_spec)} ===")
print("  first 40:", no_spec[:40])
print(f"\n  ... of which bridge/delegate-looking source: {len(polars_delegate_like)}")
for c, s in polars_delegate_like[:20]:
    print(f"     {c:45s} source={s}")

# --- polars_long / streaming path ---
print("\n=== polars_long (streaming) policy ===")
try:
    from factor_engine.backend import polars_long_policy as plp

    for name in dir(plp):
        obj = getattr(plp, name)
        if isinstance(obj, (set, frozenset, list, tuple)) and len(obj) > 3:
            head = list(obj)[:5]
            print(f"  {name:45s} n={len(obj):5d} head={head}")
except Exception as exc:  # noqa: BLE001
    print("  unavailable:", type(exc).__name__, exc)

print("\n=== sql/duckdb slot coverage ===")
sql_like = [c for c in cat if any(b in ("sql", "duckdb_sql", "duckdb") for b in (cat[c].get("backends") or []))]
print(f"  canonicals with a sql-ish backend: {len(sql_like)}")

# --- dependency weighting from the R57 manifest, if available ---
if len(sys.argv) > 1:
    import collections as C
    import gzip
    from pathlib import Path

    spec_set = set(cat) - set(no_spec)
    dep = C.Counter()
    rows = 0
    for path in sorted(Path(sys.argv[1]).glob("full-s*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                rows += 1
                for op in rec.get("ops_used") or []:
                    canon = op.get("canonical")
                    if not canon:
                        dep["UNRESOLVED_CALL"] += 1
                    elif canon in no_spec:
                        dep["OP_WITHOUT_PHYSICAL_SPEC"] += 1
                    else:
                        dep["OP_WITH_PHYSICAL_SPEC"] += 1
    print(f"\n=== R57 manifest dependency weighting (rows={rows}) ===")
    for k, v in dep.most_common():
        print(f"  {k:35s} {v}")

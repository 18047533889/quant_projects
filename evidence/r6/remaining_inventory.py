"""Compact remaining-contract inventory; does not certify numerical execution."""
import json
from collections import defaultdict
from factor_engine.runtime.operator_snapshot import build_runtime_operator_snapshot

snapshot = build_runtime_operator_snapshot(profile="runtime")
groups = defaultdict(list)
for row in snapshot["operators"]:
    if row["research_callable"]:
        continue
    backends = row.get("backends") or []
    reference = next((b for b in backends if b["backend"] == "pandas_numpy"), backends[0] if backends else {})
    kernels = reference.get("bound_kernels") or []
    source = (kernels[0].get("source_file") if kernels else reference.get("source_file")) or "UNBOUND"
    unknown = sorted({
        name for b in backends
        for name, field in b.get("parameter_schema", {}).get("properties", {}).items()
        if field.get("x-factor-engine-verification") == "unknown"
    })
    groups[source].append({"canonical": row["canonical"], "unknown_parameters": unknown})
print(json.dumps({
    "schema_version": "factor_engine.r6.remaining_contracts.v1",
    "catalog_digest": snapshot["catalog_digest"],
    "counts": snapshot["counts"],
    "scope": "Runtime contracts only; no numerical or production certification.",
    "groups": [
        {"source": source, "count": len(rows), "operators": rows}
        for source, rows in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    ],
}, ensure_ascii=False, indent=2))

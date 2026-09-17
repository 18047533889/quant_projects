from __future__ import annotations

import json
import hashlib
import runpy
import sys

from factor_engine.api.source_ref import decode_source_ref, looks_like_source_ref
from factor_engine.planner import batch_data_request as batch_module
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    PhysicalBatchGlobalOptimizer,
)


original_build = batch_module.build_batch_data_request


def diagnostic_build(*args, **kwargs):
    request = original_build(*args, **kwargs)
    missing = []
    for name, scope in request.column_source_scope_keys.items():
        if name not in request.column_scan_costs:
            ref = decode_source_ref(name) if looks_like_source_ref(name) else None
            missing.append(
                {
                    "table": getattr(ref, "table", ""),
                    "field": getattr(ref, "field", ""),
                    "params": ref.params_dict() if ref is not None else {},
                    "scope": scope,
                }
            )
    print(
        "R20_DIAG_REQUEST "
        + json.dumps(
            {
                "groups": len(request.groups),
                "degraded": [item.to_dict() for item in request.degraded_planning],
                "bound_columns": len(request.column_source_scope_keys),
                "costed_columns": len(request.column_scan_costs),
                "missing": missing,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return request


batch_module.build_batch_data_request = diagnostic_build
original_derive = PhysicalBatchGlobalOptimizer._derive_node_estimates


def diagnostic_derive(all_nodes, node_graph, **kwargs):
    costs = kwargs.get("column_scan_costs") or {}
    estimates = original_derive(all_nodes, node_graph, **kwargs)
    zero = []
    for node_id, values in estimates.items():
        if not all(int(value or 0) > 0 for value in values):
            node = all_nodes[node_id]
            attrs = dict(getattr(node, "attrs", None) or {})
            name = str(attrs.get("name") or "")
            item = {
                "id": node_id,
                "op": str(getattr(node, "op", "") or ""),
                "rows": values[0],
                "bytes": values[1],
                "memory": values[2],
            }
            if name and looks_like_source_ref(name):
                ref = decode_source_ref(name)
                identity = (ref.table, ref.field, tuple(sorted(ref.params_dict().items())))
                equivalent = []
                for key in costs:
                    if not looks_like_source_ref(key):
                        continue
                    candidate = decode_source_ref(key)
                    candidate_identity = (
                        candidate.table,
                        candidate.field,
                        tuple(sorted(candidate.params_dict().items())),
                    )
                    if candidate_identity == identity:
                        equivalent.append({
                            "exact": key == name,
                            "len": len(key),
                            "sha256": hashlib.sha256(key.encode()).hexdigest(),
                        })
                item.update(
                    table=ref.table,
                    field=ref.field,
                    params=ref.params_dict(),
                    exact_cost_key_match=name in costs,
                    name_len=len(name),
                    name_sha256=hashlib.sha256(name.encode()).hexdigest(),
                    canonical_equivalent=equivalent,
                    cost_type=type(costs.get(name)).__name__,
                    cost_rows_attr=getattr(costs.get(name), "estimated_rows", None),
                    cost_bytes_attr=getattr(costs.get(name), "projection_bytes", None),
                    cost_repr=repr(costs.get(name))[:500],
                )
            elif name:
                item["name"] = name
            zero.append(item)
    print("R20_DIAG_ZERO " + json.dumps(zero, ensure_ascii=False), flush=True)
    return estimates


PhysicalBatchGlobalOptimizer._derive_node_estimates = staticmethod(diagnostic_derive)
sys.argv = [
    "smoke_catalog.py",
    "--input", "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/r20-holder-macro-smoke-input.jsonl.gz",
    "--output", "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/r20-holder-macro-diag-output.jsonl.gz",
    "--limit", "3",
    "--start", "2026-04-20",
    "--end", "2026-04-24",
    "--backend", "auto",
    "--max-rss-mib", "2048",
    "--timeout-seconds", "180",
    "--batch-only",
    "--batch-size", "3",
]
runpy.run_path(
    "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260915/smoke_catalog.py",
    run_name="__main__",
)

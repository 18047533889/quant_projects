import json
import sys
from pathlib import Path

from factor_engine.runtime.multibackend.batch_global_optimizer import (
    PhysicalBatchGlobalOptimizer,
)


old = PhysicalBatchGlobalOptimizer._build_physical_plan


def probe(self, *args, **kwargs):
    plan = old(self, *args, **kwargs)
    node_graph = args[6]
    node_estimates = args[8]
    all_nodes = args[9]
    target_regions = {
        region.region_id: region
        for region in plan.regions
        if region.region_id in {
            "region_4_pandas_numpy_pandas_long",
            "region_9_pandas_numpy_pandas_long",
        }
    }
    if len(target_regions) == 2:
        print(json.dumps({
            "event": "target_edge_regions",
            "regions": {
                region_id: [
                    {
                        "id": node_id,
                        "op": all_nodes[node_id].op,
                        "attrs": dict(all_nodes[node_id].attrs or {}),
                        "children": node_graph.get(node_id, []),
                    }
                    for node_id in region.node_ids
                ]
                for region_id, region in target_regions.items()
            },
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "producer": edge.producer_region,
                    "consumer": edge.consumer_region,
                }
                for edge in plan.edges
                if edge.producer_region in target_regions
                and edge.consumer_region in target_regions
            ],
        }, default=str), flush=True)
    zero = []
    for region in plan.regions:
        if region.estimated_rows <= 0:
            zero.append({
                "region_id": region.region_id,
                "backend": region.backend.value,
                "representation": region.representation.value,
                "nodes": [
                    {
                        "id": node_id,
                        "op": all_nodes[node_id].op,
                        "attrs": dict(all_nodes[node_id].attrs or {}),
                        "children": node_graph.get(node_id, []),
                        "estimate": node_estimates[node_id],
                    }
                    for node_id in region.node_ids
                ],
            })
    if zero:
        print(json.dumps({"event": "zero_regions", "regions": zero}, default=str), flush=True)
    zero_edges = [
        {
            "edge_id": edge.edge_id,
            "producer_region": edge.producer_region,
            "consumer_region": edge.consumer_region,
            "estimated_rows": edge.estimated_rows,
            "estimated_bytes": edge.estimated_bytes,
        }
        for edge in plan.edges
        if edge.estimated_rows <= 0 or edge.estimated_bytes <= 0
    ]
    if zero_edges:
        producer_nodes = []
        for node_id, node in all_nodes.items():
            if any(node_id in edge["edge_id"] for edge in zero_edges):
                producer_nodes.append({
                    "id": node_id,
                    "op": node.op,
                    "attrs": dict(node.attrs or {}),
                    "children": node_graph.get(node_id, []),
                    "child_ops": [all_nodes[child].op for child in node_graph.get(node_id, [])],
                    "child_estimates": [node_estimates[child] for child in node_graph.get(node_id, [])],
                    "estimate": node_estimates[node_id],
                })
        print(json.dumps({
            "event": "zero_edges", "edges": zero_edges,
            "producer_nodes": producer_nodes,
        }, default=str), flush=True)
    return plan


PhysicalBatchGlobalOptimizer._build_physical_plan = probe
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factor_catalog_20260915"))
from smoke_catalog import main

sys.argv = [
    "smoke_catalog.py",
    "--input", "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/factor_catalog_review_r14a.williams_daily_smoke.input.jsonl.gz",
    "--output", "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/r14-edge-region-diagnostic-r7.output.jsonl.gz",
    "--limit", "40",
    "--batch-size", "40",
    "--batch-only",
    "--backend", "auto",
    "--max-rss-mib", "1800",
    "--timeout-seconds", "20",
    "--deadline-mode", "external-watchdog",
    "--daily-only",
]
main()

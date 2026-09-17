"""Bounded scaling probe with distinct roots over a real panel-shaped subtree."""

from __future__ import annotations

import argparse
import json
import resource
import time
from types import SimpleNamespace

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    PhysicalBatchGlobalOptimizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, required=True)
    args = parser.parse_args()
    count = args.factors
    if count <= 0 or count > 100_000:
        raise SystemExit("--factors must be in [1, 100000]")

    column = PlanNode("column", attrs={"name": "close"})
    window = PlanNode("literal", attrs={"value": 20})
    shared = PlanNode("ts_mean", inputs=(column, window), node_id="shared_mean")
    roots = {}
    for index in range(count):
        ref = PlanNode("plan_ref", attrs={"sid": "shared_mean"})
        constant = PlanNode("literal", attrs={"value": index})
        roots[f"factor_{index}"] = PlanNode("add", inputs=(ref, constant))

    ctx = SimpleNamespace(
        run_mode="research",
        runtime_stats={
            "row_count_estimate": 2560,
            "estimated_bytes": 163840,
            "estimated_memory_bytes": 163840,
        },
    )
    column_scan_costs = {
        "close": SimpleNamespace(estimated_rows=2560, projection_bytes=163840)
    }
    optimizer = PhysicalBatchGlobalOptimizer(
        forced_backend="pandas_numpy",
        max_optimization_ms=180_000,
        max_candidate_plans=1,
    )
    phase_seconds = {}

    def timed(name, original):
        def call(*call_args, **call_kwargs):
            started = time.perf_counter()
            try:
                return original(*call_args, **call_kwargs)
            finally:
                phase_seconds[name] = phase_seconds.get(name, 0.0) + (
                    time.perf_counter() - started
                )
        return call

    optimizer._derive_node_estimates = timed(
        "shape_derivation", optimizer._derive_node_estimates
    )
    optimizer._optimize_graph = timed("backend_assignment", optimizer._optimize_graph)
    optimizer._build_physical_plan = timed(
        "region_build", optimizer._build_physical_plan
    )
    started = time.perf_counter()
    result = optimizer.optimize_batch(
        roots,
        {"shared_mean": shared},
        {},
        ctx,
        column_scan_costs=column_scan_costs,
    )
    total_seconds = time.perf_counter() - started
    measured = sum(phase_seconds.values())
    phase_seconds["other"] = max(0.0, total_seconds - measured)
    plan = result.physical_plan
    print(json.dumps({
        "factor_count": count,
        "logical_input_count": count,
        "expected_unshared_root_nodes": 3 * count,
        "expected_shared_subtree_nodes": 3,
        "deduplicated_node_count": len(result.discovered_node_ids),
        "region_count": len(plan.regions),
        "edge_count": len(plan.edges),
        "total_seconds": total_seconds,
        "phase_seconds": phase_seconds,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "forced_backend": "pandas_numpy",
        "panel_rows": 2560,
        "panel_bytes": 163840,
        "reads_real_data": False,
        "production_ready": result.production_ready,
        "execution_ready": result.execution_ready,
        "limits": {"timeout_seconds": 180, "max_rss_mib": 2048},
    }, sort_keys=True))


if __name__ == "__main__":
    main()

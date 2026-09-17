"""Bounded synthetic scaling probe for the batch-global physical planner."""

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

    shared = PlanNode("literal", attrs={"value": 1}, node_id="shared")
    roots = {
        f"factor_{index}": PlanNode("plan_ref", attrs={"sid": "shared"})
        for index in range(count)
    }
    ctx = SimpleNamespace(
        run_mode="research",
        runtime_stats={
            "row_count_estimate": 1,
            "estimated_bytes": 32,
            "estimated_memory_bytes": 32,
        },
    )
    started = time.perf_counter()
    result = PhysicalBatchGlobalOptimizer(
        forced_backend="pandas_numpy",
        max_optimization_ms=180_000,
        max_candidate_plans=1,
    ).optimize_batch(roots, {"shared": shared}, {}, ctx)
    seconds = time.perf_counter() - started
    plan = result.physical_plan
    print(json.dumps({
        "factor_count": count,
        "deduplicated_node_count": len(result.discovered_node_ids),
        "region_count": len(plan.regions),
        "edge_count": len(plan.edges),
        "build_seconds": seconds,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "production_ready": result.production_ready,
        "execution_ready": result.execution_ready,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

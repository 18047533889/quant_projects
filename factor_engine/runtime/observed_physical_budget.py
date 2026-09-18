"""Refine compute admission from admitted, still-owned aligned source buffers.

Raw read-wave reservations are never changed. Unknown evidence retains the
pre-read contract. This is estimated compute workspace on measured input shape,
not a measurement of the future computation's peak memory.
"""
from dataclasses import replace
from types import SimpleNamespace

from factor_engine.planner.physical_factor_dag import TASK_CSE_SHARED, TASK_ROOT, TASK_SOURCE_SCAN
from factor_engine.runtime.multibackend.batch_global_optimizer import PhysicalBatchGlobalOptimizer, _canonical_cost_op
from factor_engine.planner.backend_region import TransferEdge, TransferTransform


# Budget reductions require a proven fixed-width, index-preserving output.
# Unknown operators still execute under their original conservative contracts.
_OBSERVED_NUMERIC_OPS = frozenset({
    "abs", "neg", "add", "subtract", "multiply", "divide", "safe_div_null",
    "square", "sqrt", "eq", "ne", "lt", "le", "gt", "ge",
    "and", "or", "not", "is_null", "is_finite", "where",
})


def make_observed_budget_refresher(optimization, original_task_contracts):
    nodes = {}
    stack = [*optimization.logical_roots.values(), *optimization.logical_shared_nodes.values()]
    seen = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        node_id = optimization.discovered_node_ids.get(id(node))
        if node_id is not None:
            nodes[node_id] = node
        stack.extend(node.inputs)
    graph = {}
    for node_id, node in nodes.items():
        children = [optimization.discovered_node_ids.get(id(child)) for child in node.inputs]
        if node.op == "plan_ref":
            sid = str(node.attrs.get("sid") or "")
            definition = optimization.logical_shared_nodes.get(sid)
            child_id = optimization.discovered_node_ids.get(id(definition))
            children = [child_id]
        graph[node_id] = children
    physical = optimization.physical_plan
    regions = {r.region_id: r for r in physical.regions}
    node_region = {n: r.region_id for r in physical.regions for n in r.node_ids}
    incoming = {key: [] for key in regions}
    for edge in physical.edges:
        incoming[edge.consumer_region].append(edge)
    completed = set()

    def refresh(scheduler, plan, committed, remaining):
        import pandas as pd
        from pandas.api.types import is_numeric_dtype, is_complex_dtype
        ledger = plan.meta.setdefault("observed_physical_budget", {})
        dag = plan.physical_dag
        for task_id in sorted(remaining):
            if task_id in completed:
                continue
            task = dag.tasks[task_id]
            if task.task_type not in {TASK_ROOT, TASK_CSE_SHARED}:
                continue
            def retain(reason):
                ledger[task_id] = {"status": "retained_pre_read_contract", "reason": reason}
                # Only source readiness can improve within this immutable batch.
                # Avoid repeating deep measurements of permanently unsupported
                # tasks on every scheduler tick in very large DAGs.
                if reason != "source ancestors not committed":
                    completed.add(task_id)
            original = original_task_contracts.get(task_id)
            logical_id = task.factor_name if task.task_type == TASK_ROOT else task_id.split(":", 1)[-1]
            if original is None or logical_id not in node_region:
                retain("missing logical floor or physical root")
                continue
            source_tasks, pending, visited = set(), list(task.inputs), set()
            while pending:
                parent_id = pending.pop()
                if parent_id in visited:
                    continue
                visited.add(parent_id)
                parent = dag.tasks.get(parent_id)
                if parent is None:
                    continue
                if parent.task_type == TASK_SOURCE_SCAN:
                    source_tasks.add(parent_id)
                pending.extend(parent.inputs)
            if not source_tasks or not source_tasks.issubset(committed):
                retain("source ancestors not committed")
                continue
            refs = []
            for source_id in source_tasks:
                ref = scheduler._buffer_results.get(source_id)
                held = any(
                    source_id in tids
                    and scheduler._wave_refs.get(wid) is ref
                    and task_id in scheduler._wave_pending_consumers.get(wid, ())
                    for wid, tids in scheduler._wave_source_tasks.items()
                )
                if ref is None or not held:
                    break
                refs.append(ref)
            else:
                required_regions, pending = set(), [node_region[logical_id]]
                while pending:
                    rid = pending.pop()
                    if rid in required_regions:
                        continue
                    required_regions.add(rid)
                    pending.extend(edge.producer_region for edge in incoming[rid])
                required_nodes = {n for rid in required_regions for n in regions[rid].node_ids}
                # Every dependency of every node in a merged region must be covered,
                # including another root's columns sharing that region.
                if any(n not in nodes or any(c not in nodes for c in graph[n]) for n in required_nodes):
                    retain("incomplete discovered graph")
                    continue
                if any(
                    nodes[n].op not in {"column", "literal", "plan_ref"}
                    and _canonical_cost_op(nodes[n]) not in _OBSERVED_NUMERIC_OPS
                    for n in required_nodes
                ) or any(
                    nodes[n].op == "literal"
                    and (
                        type(nodes[n].attrs.get("value")) not in (bool, int, float, type(None))
                        or (type(nodes[n].attrs.get("value")) is int
                            and nodes[n].attrs["value"].bit_length() > 63)
                    )
                    for n in required_nodes
                ):
                    retain("intermediate fixed-width numeric output not proven")
                    continue
                needed = {str(nodes[n].attrs.get("name") or "") for n in required_nodes if nodes[n].op == "column"}
                loaded = {}
                ambiguous = False
                for ref in refs:
                    for name, value in (getattr(ref, "meta", {}).get("loaded_columns") or {}).items():
                        if name not in needed:
                            continue
                        if name in loaded and loaded[name] is not value:
                            # No value equality checks or source-identity guesses.
                            ambiguous = True
                        loaded[name] = value
                if ambiguous or needed - loaded.keys():
                    retain("missing or ambiguous source column ownership")
                    continue
                observations = {}
                index = None
                for name in needed:
                    value = loaded[name]
                    if not isinstance(value, pd.Series) or not is_numeric_dtype(value.dtype) or is_complex_dtype(value.dtype) or len(value) == 0:
                        break
                    if index is not None and not index.equals(value.index):
                        break
                    index = value.index
                    size = int(value.memory_usage(index=True, deep=True))
                    # Allow numeric results to widen Boolean/integer inputs to Float64.
                    size = max(size, int(value.index.memory_usage(deep=True)) + len(value) * 8)
                    if size <= 0:
                        break
                    observations[name] = SimpleNamespace(estimated_rows=len(value), projection_bytes=size)
                if len(observations) != len(needed) or not needed:
                    retain("non-numeric, empty or incompatible aligned input")
                    continue
                shadow = {}
                for n in required_nodes:
                    node = nodes[n]
                    attrs = dict(node.attrs)
                    if node.op == "column":
                        for key in ("estimated_rows", "row_count_estimate", "estimated_bytes",
                                    "byte_count_estimate", "estimated_memory_bytes", "memory_bytes_estimate"):
                            attrs.pop(key, None)
                    shadow[n] = replace(node, attrs=attrs)
                subgraph = {n: graph[n] for n in required_nodes}
                if any(c not in required_nodes for children in subgraph.values() for c in children):
                    retain("physical region closure lacks logical dependency")
                    continue
                try:
                    estimates = PhysicalBatchGlobalOptimizer._derive_node_estimates(
                        shadow, subgraph, roots=(), global_rows=None,
                        global_bytes=None, global_memory=None, column_scan_costs=observations,
                    )
                    if any(min(estimates[n]) <= 0 for n in required_nodes):
                        retain("unknown intermediate output shape")
                        continue
                    # The executor retains region outputs within this task. Keep
                    # every ancestor workspace; no identity-edge exemption.
                    workspace = sum(
                        estimates[n][2] + PhysicalBatchGlobalOptimizer._node_additional_workspace_bytes(
                            shadow[n], optimization.per_node_choices[n], estimates[n][0]
                        )
                        for rid in required_regions for n in regions[rid].node_ids
                    )
                    transfer = 0
                    for rid in required_regions:
                        for edge in incoming[rid]:
                            if not isinstance(edge, TransferEdge):
                                raise ValueError("untyped transfer boundary")
                            observed = sum(estimates[n][1] for n in regions[edge.producer_region].node_ids)
                            identity = (
                                edge.transform == TransferTransform.SAME_BACKEND_NATIVE
                                and edge.source_backend == edge.target_backend
                                and edge.source_representation == edge.target_representation
                                and not any((edge.requires_sort, edge.requires_repartition,
                                             edge.requires_reshape, edge.requires_dtype_cast))
                            )
                            # Non-identity conversions retain their original bound
                            # and allow source/interchange/target to coexist.
                            transfer += observed if identity else max(edge.estimated_bytes, 3 * observed)
                    peak = max(original.peak_memory_bytes, workspace + transfer + original.output_bytes)
                    contract = replace(original, peak_memory_bytes=peak,
                                       estimate_basis=original.estimate_basis + "+observed-aligned-shape")
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    retain(type(exc).__name__ + ": " + str(exc)[:240])
                    continue
                dag.tasks[task_id] = replace(task, resource_contract=contract)
                ledger[task_id] = {
                    "status": "refined", "source_task_ids": sorted(source_tasks),
                    "region_ids": sorted(required_regions), "observed_columns": len(observations),
                    "workspace_bytes": workspace, "transfer_bytes": transfer,
                    "logical_floor_bytes": original.peak_memory_bytes,
                    "admission_peak_bytes": contract.admissible_peak_bytes,
                    "basis": "measured-input-shape-plus-estimated-compute-workspace",
                }
                completed.add(task_id)
                continue
            retain("read-wave buffer no longer held")
    return refresh

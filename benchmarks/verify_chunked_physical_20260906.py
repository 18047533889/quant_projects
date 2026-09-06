"""Serial public-planner verification; run from factor_engine with PYTHONPATH=.:..."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from factor_engine.planner import PlanNode, Optimizer, compile_many_chunked, iter_compile_many_chunked
from factor_engine.planner.dag import DAGPlan, FactorPlan
from factor_engine.planner.compiler_pass import (
    CompilerPassManager, PassContract, PassContext, IRKind, SemanticEquivalence,
    NumericEquivalence, InvariantResult, PassInvariantError, PassLegalityError,
)
import factor_engine.planner.physical_lowerer as lowerer


def fails(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    literal = lambda value: PlanNode(op="literal", attrs={"value": value})
    ref = lambda sid: PlanNode(op="plan_ref", attrs={"sid": sid})
    definitions = {"inner": literal(3), "outer": ref("inner"), "unused": literal(9)}
    dag = DAGPlan(roots=[FactorPlan("a", ref("outer")), FactorPlan("b", ref("inner"))],
                  shared_nodes=definitions)
    physical = compile_many_chunked(dag, chunk_size=1, rows=2, instruments=1)
    assert isinstance(physical, list) and len(physical) == 2
    assert set(physical[0].tasks) >= {"cse:inner", "cse:outer", "root:a"}
    assert "cse:unused" not in physical[0].tasks
    assert "cse:outer" not in physical[1].tasks
    assert "cse:outer" in physical[0].tasks["root:a"].inputs
    assert "root:a" in physical[0].tasks["cse:outer"].consumers
    order = physical[0].topological_order()
    assert order.index("cse:inner") < order.index("cse:outer") < order.index("root:a")
    assert dag.shared_nodes == definitions
    # Execute both independent physical chunks through the actual scheduler.
    # The tiny literal/ref evaluator is an independent numerical oracle for
    # shared materialization order and closure completeness, not a market scan.
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.resource_broker import ResourceBroker
    observed = {}
    for part in physical:
        shared = {}

        def evaluate(node):
            if node.op == "literal":
                return node.attrs["value"]
            assert node.op == "plan_ref"
            return shared[node.attrs["sid"]]

        def materialize(sid, node):
            shared[sid] = evaluate(node)
            return True

        scheduler = AdaptiveBatchScheduler(
            broker=ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=2), max_concurrency=2,
        )
        scheduler._execution_policy = "thread"
        try:
            scheduler.run(part, backend=None,
                          ctx=SimpleNamespace(run_mode="research", runtime_stats={}, shared_result_cache=shared),
                          execute_root=lambda task: evaluate(task.node_ref),
                          materialize_shared=materialize,
                          result_handler=lambda name, result: observed.__setitem__(name, result))
        finally:
            scheduler.executor.shutdown()
    assert observed == {"a": 3, "b": 3}

    calls = []
    original = lowerer.lower_batch_dag

    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with patch.object(lowerer, "lower_batch_dag", track):
        iterator = iter_compile_many_chunked(dag, chunk_size=1)
        assert calls == []
        first = next(iterator)
        assert len(calls) == 1
        del first
        next(iterator)
        assert len(calls) == 2
    for size in [0, -1, 1.5, True, "2", lowerer._get_adaptive_dag_width_limit() + 1]:
        fails(lambda size=size: compile_many_chunked(dag, chunk_size=size))
    fails(lambda: compile_many_chunked(dag.roots, chunk_size=1))
    fails(lambda: compile_many_chunked([FactorPlan("x", ref(""))], chunk_size=1))
    fails(lambda: compile_many_chunked(DAGPlan([FactorPlan("x", ref("a"))],
                                               {"a": ref("b"), "b": ref("a")}), chunk_size=1))
    fails(lambda: compile_many_chunked(DAGPlan([FactorPlan("x", ref("a"))],
                                               {"a": ref("missing")}), chunk_size=1))
    assert compile_many_chunked([], chunk_size=1) == []
    fails(lambda: compile_many_chunked([FactorPlan("x", literal(1)), FactorPlan("x", literal(2))],
                                       chunk_size=1))
    assert len(compile_many_chunked(dag.roots, shared_nodes=definitions, chunk_size=1)) == 2
    with patch.object(lowerer, "_get_adaptive_chunk_size", return_value=99), \
            patch.object(lowerer, "_get_adaptive_dag_width_limit", return_value=1):
        assert len(compile_many_chunked(dag)) == 2

    # Public optimizer boundary: this checkout has no Optimizer.compile method.
    # Exercise supported trace API and report the unavailable named gate honestly.
    optimizer = Optimizer()
    plan = PlanNode(op="add", inputs=(literal(1), literal(2)))
    optimized, traces = optimizer.optimize_with_pass_trace(plan)
    assert optimized.op == "literal" and optimized.attrs["value"] == 3
    assert len(optimizer.optimize_with_trace(plan)) == 3
    assert all(inv.passed for trace in traces for inv in trace.invariants)
    contract = dict(name="probe", input_ir=IRKind.NORMALIZED_IR,
                    output_ir=IRKind.NORMALIZED_IR,
                    semantic_equivalence=SemanticEquivalence.IDENTITY)
    bad_invariant = SimpleNamespace(
        contract=PassContract(**contract, numeric_equivalence=NumericEquivalence.EXACT_VALUE,
                              invariants=(lambda *args: InvariantResult("probe", False),)),
        run=lambda node, context: node,
    )
    fails(lambda: CompilerPassManager([bad_invariant]).run(plan), PassInvariantError)
    bad_numeric = SimpleNamespace(
        contract=PassContract(**contract, numeric_equivalence=NumericEquivalence.RESEARCH_APPROXIMATE),
        run=lambda node, context: node,
    )
    fails(lambda: CompilerPassManager([bad_numeric]).run(plan, context=PassContext(production=True)),
          PassLegalityError)
    print(json.dumps({
        "chunk_closure_and_dependencies": "PASS", "lazy_lowering": "PASS",
        "chunk_scheduler_numerical_oracle": observed,
        "invalid_chunks_missing_refs_cycles": "PASS",
        "optimized": {"op": optimized.op, "attrs": dict(optimized.attrs)},
        "stages": [{"name": trace.name, "semantic": trace.semantic_equivalence.value,
                    "numeric": trace.numeric_equivalence.name,
                    "invariants": [{"name": i.name, "passed": i.passed} for i in trace.invariants]}
                   for trace in traces],
        "legacy_trace_api": "PASS", "invariant_fail_closed": "PASS",
        "production_numeric_fail_closed": "PASS",
        "Optimizer.compile": "NOT_AVAILABLE" if not hasattr(optimizer, "compile") else "NOT_RUN",
    }, indent=2))


if __name__ == "__main__":
    main()

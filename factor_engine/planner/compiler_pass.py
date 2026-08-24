"""Typed compiler-pass contracts and execution traces for logical-plan rewrites.

This module is intentionally additive: current compiler stages still carry ``PlanNode``
while pass contracts make stage transitions, legality, invariants, equivalence, and cost
observable. Later IR types can replace individual stages without changing the manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.plan_hash import structural_key


class IRKind(str, Enum):
    PARSED_EXPR = "parsed_expr"
    BOUND_EXPR = "bound_expr"
    TYPED_SEMANTIC_IR = "typed_semantic_ir"
    NORMALIZED_IR = "normalized_ir"
    LOGICAL_QUERY_GRAPH = "logical_query_graph"
    OPTIMIZED_LOGICAL_GRAPH = "optimized_logical_graph"
    PHYSICAL_QUERY_GRAPH = "physical_query_graph"
    EXECUTABLE_REGIONS = "executable_regions"


class SemanticEquivalence(str, Enum):
    IDENTITY = "identity"
    VALUE_PRESERVING = "value_preserving"
    RANK_PRESERVING = "rank_preserving"
    RESEARCH_APPROXIMATE = "research_approximate"


class NumericEquivalence(IntEnum):
    EXACT_VALUE = 0
    IEEE_EQUIVALENT = 1
    TOLERANCE_EQUIVALENT = 2
    RANK_EQUIVALENT = 3
    RESEARCH_APPROXIMATE = 4


@dataclass(frozen=True)
class NumericPolicy:
    """Maximum numeric relaxation allowed for compiler rewrites."""

    maximum_equivalence: NumericEquivalence = NumericEquivalence.IEEE_EQUIVALENT

    @classmethod
    def production_default(cls) -> "NumericPolicy":
        return cls(NumericEquivalence.IEEE_EQUIVALENT)

    @classmethod
    def research_default(cls) -> "NumericPolicy":
        return cls(NumericEquivalence.RESEARCH_APPROXIMATE)

    def permits(self, equivalence: NumericEquivalence) -> bool:
        return equivalence <= self.maximum_equivalence


@dataclass(frozen=True)
class CompilerCost:
    scan_bytes: int = 0
    decoded_bytes: int = 0
    conversion_bytes: int = 0
    kernel_work: float = 0.0
    memory_lifetime_bytes: int = 0
    scheduler_queue_cost: float = 0.0
    write_bytes: int = 0

    @property
    def ranking_key(self) -> tuple[float, ...]:
        return (
            float(self.scan_bytes),
            float(self.decoded_bytes),
            float(self.conversion_bytes),
            float(self.kernel_work),
            float(self.memory_lifetime_bytes),
            float(self.scheduler_queue_cost),
            float(self.write_bytes),
        )


@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    detail: str = ""


InvariantCheck = Callable[[PlanNode, PlanNode, "PassContext"], InvariantResult]
LegalityCheck = Callable[[PlanNode, "PassContext"], bool]
CostEstimator = Callable[[PlanNode, PlanNode, "PassContext"], CompilerCost]


@dataclass(frozen=True)
class PassContract:
    name: str
    input_ir: IRKind
    output_ir: IRKind
    semantic_equivalence: SemanticEquivalence
    numeric_equivalence: NumericEquivalence
    invariants: tuple[InvariantCheck, ...] = ()
    legality: LegalityCheck | None = None
    cost_estimator: CostEstimator | None = None
    description: str = ""


@dataclass(frozen=True)
class PassContext:
    production: bool = False
    numeric_policy: NumericPolicy = field(default_factory=NumericPolicy.production_default)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@runtime_checkable
class CompilerPass(Protocol):
    contract: PassContract

    def run(self, plan: PlanNode, context: PassContext) -> PlanNode:
        ...


@dataclass(frozen=True)
class PassTrace:
    name: str
    input_ir: IRKind
    output_ir: IRKind
    semantic_equivalence: SemanticEquivalence
    numeric_equivalence: NumericEquivalence
    input_hash: str
    output_hash: str
    changed: bool
    duration_ns: int
    cost: CompilerCost
    invariants: tuple[InvariantResult, ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class CompilationResult:
    plan: PlanNode
    ir_kind: IRKind
    traces: tuple[PassTrace, ...]


class CompilerPassError(RuntimeError):
    pass


class PassContractError(CompilerPassError):
    pass


class PassInvariantError(CompilerPassError):
    pass


class PassLegalityError(CompilerPassError):
    pass


class CompilerPassManager:
    """Run typed passes with fail-closed contract and invariant enforcement.

    REM-197: 维护一个 class-level 生产运行计数器，证明 pass manager 在真实生产路径中执行。
    """

    _production_run_count: int = 0

    def __init__(self, passes: Sequence[CompilerPass] = ()) -> None:
        self._passes = tuple(passes)
        self._validate_pipeline()

    @classmethod
    def get_production_run_count(cls) -> int:
        """REM-197: 返回生产模式执行次数，用于可达性测试。"""
        return cls._production_run_count

    @classmethod
    def reset_production_run_count(cls) -> None:
        """REM-197: 重置生产模式计数器，测试专用。"""
        cls._production_run_count = 0

    @property
    def passes(self) -> tuple[CompilerPass, ...]:
        return self._passes

    def _validate_pipeline(self) -> None:
        for current, following in zip(self._passes, self._passes[1:]):
            if current.contract.output_ir != following.contract.input_ir:
                raise PassContractError(
                    f"compiler pass kind mismatch: {current.contract.name} outputs "
                    f"{current.contract.output_ir.value}, but {following.contract.name} "
                    f"expects {following.contract.input_ir.value}"
                )

    def run(
        self,
        plan: PlanNode,
        *,
        context: PassContext | None = None,
        input_ir: IRKind | None = None,
    ) -> CompilationResult:
        ctx = context or PassContext()
        kind = input_ir or (
            self._passes[0].contract.input_ir
            if self._passes
            else IRKind.LOGICAL_QUERY_GRAPH
        )
        current = plan
        traces: list[PassTrace] = []

        # REM-197: increment production counter when genuinely running production
        if ctx.production:
            type(self)._production_run_count += 1

        for compiler_pass in self._passes:
            contract = compiler_pass.contract
            if kind != contract.input_ir:
                raise PassContractError(
                    f"pass {contract.name} expects {contract.input_ir.value}, got {kind.value}"
                )
            if not ctx.numeric_policy.permits(contract.numeric_equivalence):
                raise PassLegalityError(
                    f"pass {contract.name} requires {contract.numeric_equivalence.name}, "
                    f"policy allows through {ctx.numeric_policy.maximum_equivalence.name}"
                )
            if contract.legality is not None and not contract.legality(current, ctx):
                raise PassLegalityError(f"pass {contract.name} is not legal in this context")

            before_hash = structural_key(current)
            started = perf_counter_ns()
            output = compiler_pass.run(current, ctx)
            duration_ns = perf_counter_ns() - started
            if not isinstance(output, PlanNode):
                raise PassContractError(
                    f"pass {contract.name} returned {type(output).__name__}, expected PlanNode"
                )

            invariant_results = tuple(check(current, output, ctx) for check in contract.invariants)
            failed = tuple(result for result in invariant_results if not result.passed)
            if failed:
                summary = "; ".join(
                    f"{result.name}: {result.detail or 'failed'}" for result in failed
                )
                raise PassInvariantError(f"pass {contract.name} invariant failure: {summary}")

            after_hash = structural_key(output)
            cost = (
                contract.cost_estimator(current, output, ctx)
                if contract.cost_estimator is not None
                else CompilerCost()
            )
            traces.append(
                PassTrace(
                    name=contract.name,
                    input_ir=contract.input_ir,
                    output_ir=contract.output_ir,
                    semantic_equivalence=contract.semantic_equivalence,
                    numeric_equivalence=contract.numeric_equivalence,
                    input_hash=before_hash,
                    output_hash=after_hash,
                    changed=before_hash != after_hash,
                    duration_ns=duration_ns,
                    cost=cost,
                    invariants=invariant_results,
                )
            )
            current = output
            kind = contract.output_ir

        return CompilationResult(plan=current, ir_kind=kind, traces=tuple(traces))

    @staticmethod
    def choose_lowest_cost(results: Sequence[CompilationResult]) -> CompilationResult:
        """Choose among already-legal candidates using the declared cost vector."""
        if not results:
            raise ValueError("at least one legal compilation candidate is required")

        def total(result: CompilationResult) -> tuple[float, ...]:
            values = [trace.cost.ranking_key for trace in result.traces]
            if not values:
                return CompilerCost().ranking_key
            return tuple(sum(parts) for parts in zip(*values))

        return min(results, key=total)


def invariant_root_semantics_preserved(
    before: PlanNode, after: PlanNode, context: PassContext
) -> InvariantResult:
    del context
    passed = dict(before.semantic_attrs) == dict(after.semantic_attrs)
    return InvariantResult(
        name="ROOT_SEMANTICS_PRESERVED",
        passed=passed,
        detail="root semantic_attrs changed" if not passed else "",
    )


def invariant_source_dependencies_preserved(
    before: PlanNode, after: PlanNode, context: PassContext
) -> InvariantResult:
    del context
    from factor_engine.planner.source_dependencies import build_source_dependency_manifest

    before_dependencies = set(build_source_dependency_manifest(before))
    after_dependencies = set(build_source_dependency_manifest(after))
    passed = before_dependencies == after_dependencies
    return InvariantResult(
        name="SOURCE_DEPENDENCIES_PRESERVED",
        passed=passed,
        detail=(
            f"before={sorted(before_dependencies)!r}, after={sorted(after_dependencies)!r}"
            if not passed
            else ""
        ),
    )


def invariant_leaf_columns_preserved(
    before: PlanNode, after: PlanNode, context: PassContext
) -> InvariantResult:
    del context

    def columns(root: PlanNode) -> set[tuple[tuple[str, Any], ...]]:
        found: set[tuple[tuple[str, Any], ...]] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node.op in {"column", "col"}:
                found.add(tuple(sorted(dict(node.attrs).items())))
            stack.extend(node.inputs)
        return found

    before_columns = columns(before)
    after_columns = columns(after)
    passed = before_columns == after_columns
    return InvariantResult(
        name="LEAF_COLUMNS_PRESERVED",
        passed=passed,
        detail=(
            f"before={sorted(before_columns)!r}, after={sorted(after_columns)!r}"
            if not passed
            else ""
        ),
    )


DEFAULT_REWRITE_INVARIANTS: tuple[InvariantCheck, ...] = (
    invariant_root_semantics_preserved,
    invariant_source_dependencies_preserved,
    invariant_leaf_columns_preserved,
)

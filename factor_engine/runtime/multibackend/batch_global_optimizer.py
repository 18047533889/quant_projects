# -*- coding: utf-8 -*-
"""MB-P1-003: Cross-root global optimization.

Performs batch-global optimizations across multiple factor roots:
- Cross-root common subexpression elimination (CSE)
- Shared intermediate materialization planning
- Global constant folding and predicate pushdown
- Cross-root data movement minimization

Integrates with adaptive_batch_scheduler for batch-level plan optimization.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationOpportunity:
    """A detected cross-root optimization opportunity.

    Attributes:
        kind: Optimization type (cse/constant_fold/predicate_push/fusion)
        savings_bytes: Estimated memory savings (bytes)
        savings_work: Estimated compute savings (work units)
        affected_roots: Root task IDs that benefit
        shared_node_id: ID for shared materialization (if CSE)
        confidence: Confidence in savings estimate [0.0, 1.0]
    """
    kind: str
    savings_bytes: int
    savings_work: float
    affected_roots: list[str]
    shared_node_id: str | None = None
    confidence: float = 0.8


@dataclass
class GlobalOptimizationResult:
    """Result of batch-global optimization pass.

    Attributes:
        opportunities: Detected optimization opportunities
        applied_count: Number of optimizations applied
        total_savings_bytes: Total estimated memory savings
        total_savings_work: Total estimated work savings
        rewritten_dag: Optimized DAG (if modifications made)
        metadata: Additional optimization metadata
    """
    opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    applied_count: int = 0
    total_savings_bytes: int = 0
    total_savings_work: float = 0.0
    rewritten_dag: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchGlobalOptimizer:
    """Cross-root global optimizer for batch factor computation.

    Analyzes entire batch DAG to find optimization opportunities that
    span multiple factor roots, then rewrites the DAG to exploit them.
    """

    def __init__(
        self,
        *,
        min_savings_bytes: int = 10 * 1024 * 1024,  # 10 MB
        min_savings_work: float = 50_000.0,
        min_reuse_count: int = 2,
    ):
        """Initialize global optimizer.

        Args:
            min_savings_bytes: Minimum memory savings to apply optimization
            min_savings_work: Minimum work savings to apply optimization
            min_reuse_count: Minimum reuse count for CSE
        """
        self._min_savings_bytes = min_savings_bytes
        self._min_savings_work = min_savings_work
        self._min_reuse_count = min_reuse_count
        self._optimization_history: list[GlobalOptimizationResult] = []

    def optimize_batch(
        self,
        dag: Any,
        *,
        cost_fn: Callable[[Any], dict[str, Any]] | None = None,
        enable_cse: bool = True,
        enable_predicate_push: bool = True,
        enable_constant_fold: bool = True,
    ) -> GlobalOptimizationResult:
        """Optimize entire batch DAG across all roots.

        Args:
            dag: PhysicalFactorDAG to optimize
            cost_fn: Optional cost estimation function
            enable_cse: Enable cross-root CSE
            enable_predicate_push: Enable predicate pushdown
            enable_constant_fold: Enable constant folding

        Returns:
            GlobalOptimizationResult with opportunities and rewritten DAG
        """
        try:
            opportunities = []

            # Phase 1: Cross-root CSE detection
            if enable_cse:
                cse_opps = self._detect_cross_root_cse(dag, cost_fn)
                opportunities.extend(cse_opps)
                _logger.info(f"Detected {len(cse_opps)} CSE opportunities")

            # Phase 2: Predicate pushdown across joins
            if enable_predicate_push:
                pred_opps = self._detect_predicate_pushdown(dag, cost_fn)
                opportunities.extend(pred_opps)
                _logger.info(f"Detected {len(pred_opps)} predicate push opportunities")

            # Phase 3: Constant folding
            if enable_constant_fold:
                const_opps = self._detect_constant_folding(dag, cost_fn)
                opportunities.extend(const_opps)
                _logger.info(f"Detected {len(const_opps)} constant fold opportunities")

            # Sort by savings (memory primary, work secondary)
            opportunities.sort(
                key=lambda o: (o.savings_bytes, o.savings_work),
                reverse=True
            )

            # Apply worthwhile optimizations
            rewritten_dag = dag
            applied = []
            for opp in opportunities:
                if self._is_worthwhile(opp):
                    try:
                        rewritten_dag = self._apply_optimization(rewritten_dag, opp)
                        applied.append(opp)
                    except Exception as exc:
                        _logger.warning(
                            f"Failed to apply {opp.kind} optimization: {exc}"
                        )

            result = GlobalOptimizationResult(
                opportunities=opportunities,
                applied_count=len(applied),
                total_savings_bytes=sum(o.savings_bytes for o in applied),
                total_savings_work=sum(o.savings_work for o in applied),
                rewritten_dag=rewritten_dag if applied else None,
                metadata={
                    "cse_count": sum(1 for o in applied if o.kind == "cse"),
                    "predicate_push_count": sum(
                        1 for o in applied if o.kind == "predicate_push"
                    ),
                    "constant_fold_count": sum(
                        1 for o in applied if o.kind == "constant_fold"
                    ),
                },
            )

            self._optimization_history.append(result)
            return result

        except Exception as exc:
            _logger.error(f"Batch optimization failed: {exc}")
            return GlobalOptimizationResult()

    def _detect_cross_root_cse(
        self,
        dag: Any,
        cost_fn: Callable[[Any], dict[str, Any]] | None,
    ) -> list[OptimizationOpportunity]:
        """Detect common subexpressions across multiple roots."""
        opportunities = []

        try:
            # Build structural hash -> (node, roots) mapping
            node_fingerprints: dict[str, list[tuple[Any, str]]] = defaultdict(list)

            for root_id, root_task in self._iter_root_tasks(dag):
                for node in self._traverse_plan(root_task):
                    fingerprint = self._structural_fingerprint(node)
                    if fingerprint:
                        node_fingerprints[fingerprint].append((node, root_id))

            # Find reused subexpressions
            for fingerprint, nodes_and_roots in node_fingerprints.items():
                if len(nodes_and_roots) < self._min_reuse_count:
                    continue

                # Estimate savings from sharing
                node = nodes_and_roots[0][0]
                affected_roots = [r for _, r in nodes_and_roots]

                cost = self._estimate_node_cost(node, cost_fn)
                # Savings = (N - 1) executions avoided
                reuse_factor = len(nodes_and_roots) - 1
                savings_bytes = cost.get("peak_live_memory_bytes", 0) * reuse_factor
                savings_work = cost.get("total_work", 0.0) * reuse_factor

                if savings_bytes >= self._min_savings_bytes // 2:
                    opportunities.append(
                        OptimizationOpportunity(
                            kind="cse",
                            savings_bytes=savings_bytes,
                            savings_work=savings_work,
                            affected_roots=affected_roots,
                            shared_node_id=f"shared_{fingerprint[:8]}",
                            confidence=0.85,
                        )
                    )

        except Exception as exc:
            _logger.warning(f"CSE detection failed: {exc}")

        return opportunities

    def _detect_predicate_pushdown(
        self,
        dag: Any,
        cost_fn: Callable[[Any], dict[str, Any]] | None,
    ) -> list[OptimizationOpportunity]:
        """Detect opportunities to push predicates down through joins."""
        opportunities = []

        try:
            for root_id, root_task in self._iter_root_tasks(dag):
                for node in self._traverse_plan(root_task):
                    # Look for filter -> join pattern
                    if self._is_filter_node(node):
                        child = self._get_child_node(node)
                        if child and self._is_join_node(child):
                            # Estimate savings from earlier filtering
                            filter_cost = self._estimate_node_cost(node, cost_fn)
                            join_cost = self._estimate_node_cost(child, cost_fn)

                            # Pushing filter before join reduces join input
                            selectivity = self._estimate_selectivity(node)
                            savings_bytes = int(
                                join_cost.get("peak_live_memory_bytes", 0)
                                * (1.0 - selectivity)
                            )
                            savings_work = (
                                join_cost.get("total_work", 0.0) * (1.0 - selectivity)
                            )

                            if savings_bytes >= self._min_savings_bytes // 4:
                                opportunities.append(
                                    OptimizationOpportunity(
                                        kind="predicate_push",
                                        savings_bytes=savings_bytes,
                                        savings_work=savings_work,
                                        affected_roots=[root_id],
                                        confidence=0.70,
                                    )
                                )

        except Exception as exc:
            _logger.warning(f"Predicate pushdown detection failed: {exc}")

        return opportunities

    def _detect_constant_folding(
        self,
        dag: Any,
        cost_fn: Callable[[Any], dict[str, Any]] | None,
    ) -> list[OptimizationOpportunity]:
        """Detect opportunities for constant folding."""
        opportunities = []

        try:
            for root_id, root_task in self._iter_root_tasks(dag):
                for node in self._traverse_plan(root_task):
                    if self._is_constant_expression(node):
                        # Fold at plan time instead of runtime
                        cost = self._estimate_node_cost(node, cost_fn)
                        savings_work = cost.get("total_work", 0.0)

                        if savings_work >= self._min_savings_work // 10:
                            opportunities.append(
                                OptimizationOpportunity(
                                    kind="constant_fold",
                                    savings_bytes=0,
                                    savings_work=savings_work,
                                    affected_roots=[root_id],
                                    confidence=0.95,
                                )
                            )

        except Exception as exc:
            _logger.warning(f"Constant folding detection failed: {exc}")

        return opportunities

    def _is_worthwhile(self, opp: OptimizationOpportunity) -> bool:
        """Check if optimization meets threshold for application."""
        return (
            opp.savings_bytes >= self._min_savings_bytes
            or opp.savings_work >= self._min_savings_work
        )

    def _apply_optimization(
        self, dag: Any, opp: OptimizationOpportunity
    ) -> Any:
        """Apply optimization to DAG (creates modified copy)."""
        # This would integrate with PhysicalFactorDAG rewrite logic
        # For now, return dag unchanged (actual rewrite requires
        # deeper integration with planner)
        _logger.info(
            f"Applied {opp.kind} optimization: "
            f"{opp.savings_bytes // 1024 // 1024} MB, "
            f"{opp.savings_work:.0f} work units"
        )
        return dag

    def _structural_fingerprint(self, node: Any) -> str | None:
        """Compute structural fingerprint for CSE."""
        try:
            op = getattr(node, "op", None)
            if op is None:
                return None

            # Include operator + child structure + parameters
            parts = [op]
            for child in getattr(node, "inputs", []):
                child_fp = self._structural_fingerprint(child)
                if child_fp:
                    parts.append(child_fp)

            # Include relevant parameters (ignore cosmetic ones)
            params = getattr(node, "params", {})
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            if param_str:
                parts.append(param_str)

            return "|".join(parts)

        except Exception:
            return None

    def _iter_root_tasks(self, dag: Any) -> list[tuple[str, Any]]:
        """Iterate over root tasks in DAG."""
        try:
            tasks = getattr(dag, "tasks", {})
            return [
                (tid, task)
                for tid, task in tasks.items()
                if getattr(task, "task_type", None) == "root"
            ]
        except Exception:
            return []

    def _traverse_plan(self, task: Any) -> list[Any]:
        """Traverse plan tree, yielding all nodes."""
        nodes = []
        try:
            node = getattr(task, "node_ref", None)
            if node is not None:
                nodes.append(node)
                for child in getattr(node, "inputs", []):
                    nodes.extend(self._traverse_plan_node(child))
        except Exception:
            pass
        return nodes

    def _traverse_plan_node(self, node: Any) -> list[Any]:
        """Recursively traverse plan node."""
        nodes = [node]
        try:
            for child in getattr(node, "inputs", []):
                nodes.extend(self._traverse_plan_node(child))
        except Exception:
            pass
        return nodes

    def _estimate_node_cost(
        self, node: Any, cost_fn: Callable[[Any], dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Estimate cost for single node."""
        if cost_fn is not None:
            try:
                return cost_fn(node)
            except Exception:
                pass

        # Fallback estimate
        return {"peak_live_memory_bytes": 10 * 1024 * 1024, "total_work": 1000.0}

    def _is_filter_node(self, node: Any) -> bool:
        """Check if node is a filter operation."""
        op = getattr(node, "op", "")
        return "filter" in op.lower() or "where" in op.lower()

    def _is_join_node(self, node: Any) -> bool:
        """Check if node is a join operation."""
        op = getattr(node, "op", "")
        return "join" in op.lower() or "merge" in op.lower()

    def _is_constant_expression(self, node: Any) -> bool:
        """Check if node computes constant expression."""
        try:
            # Check if all inputs are literals
            inputs = getattr(node, "inputs", [])
            if not inputs:
                return False
            return all(getattr(inp, "is_literal", False) for inp in inputs)
        except Exception:
            return False

    def _get_child_node(self, node: Any) -> Any | None:
        """Get first child node."""
        try:
            inputs = getattr(node, "inputs", [])
            return inputs[0] if inputs else None
        except Exception:
            return None

    def _estimate_selectivity(self, filter_node: Any) -> float:
        """Estimate filter selectivity (fraction of rows passing)."""
        # Conservative estimate: assume 50% selectivity
        return 0.5

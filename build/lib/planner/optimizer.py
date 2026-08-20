"""逻辑计划轻量优化：常量折叠、composite lowering 与 fastpath 改写。"""

from __future__ import annotations

import math

from .logical_plan import PlanNode


class Optimizer:
    """轻量计划优化；最终输出保证 canonical op/parameter spelling。

    REM-011/REM-012: Production 执行通过 CompilerPassManager 统一路由，确保
    PassContract 与 NumericPolicy 在实际执行中生效。
    """

    def __init__(self, *, allow_semantic_rewrites: bool = False) -> None:
        self.allow_semantic_rewrites = bool(allow_semantic_rewrites)

    def optimize(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        """优化计划并返回最终结果。

        REM-011: 实际执行通过 CompilerPassManager，确保 PassContract、
        invariant 检查与 NumericPolicy 在生产环境真实生效。
        """
        from planner.compiler_pass import NumericPolicy, PassContext
        from planner.optimizer_passes import build_optimizer_pass_manager

        numeric_policy = (
            NumericPolicy.production_default()
            if production
            else NumericPolicy.research_default()
        )
        ctx = PassContext(
            production=production,
            numeric_policy=numeric_policy,
        )
        return build_optimizer_pass_manager(self).run(plan, context=ctx).plan

    def lower_only(self, plan: PlanNode) -> PlanNode:
        from planner.composite_lowering import lower_composite_operators
        from planner.canonicalize_params import (
            canonicalize_plan_parameters,
            validate_plan_params,
        )

        folded = self._fold_literals(plan)
        validate_plan_params(folded)
        lowered = lower_composite_operators(folded)
        return canonicalize_plan_parameters(self._fold_literals(lowered))

    def optimize_with_trace(
        self, plan: PlanNode, *, production: bool = False
    ) -> tuple[PlanNode, PlanNode, tuple[tuple[str, tuple[str, ...]], ...]]:
        """返回 (final, folded, lowering_trace) 三元组，保持遗留 API 兼容性。

        REM-011: 内部通过 pass manager 执行，但对外保持原有返回格式。
        """
        from planner.compiler_pass import NumericPolicy, PassContext
        from planner.composite_lowering import build_lowering_trace
        from planner.optimizer_passes import build_optimizer_pass_manager

        folded = self._fold_literals(plan)
        # 通过 pass manager 执行完整流程
        numeric_policy = (
            NumericPolicy.production_default()
            if production
            else NumericPolicy.research_default()
        )
        ctx = PassContext(production=production, numeric_policy=numeric_policy)
        result = build_optimizer_pass_manager(self).run(folded, context=ctx)
        final = result.plan

        # 从 traces 中提取 lowering 前后的计划，构造遗留格式的 trace
        # traces[0] = literal_fold, traces[1] = parameter_validation, traces[2] = composite_lowering
        # 我们需要 lowering 前（fold+validate后）与 lowering 后（post_lowering_fold前）
        # 由于 pass manager 只记录 hash，我们需要重新执行 lowering 的 trace 构建
        from planner.canonicalize_params import validate_plan_params
        from planner.composite_lowering import lower_composite_operators

        validate_plan_params(folded, production=production)
        lowered = lower_composite_operators(folded)
        refolded = self._fold_literals(lowered)
        trace = build_lowering_trace(folded, refolded)

        return final, folded, trace

    def optimize_with_pass_trace(
        self, plan: PlanNode, *, production: bool = False
    ) -> tuple[PlanNode, tuple]:
        """返回 (final_plan, pass_traces) 二元组，暴露完整的 pass manager trace。

        REM-011: 新式 API，返回 CompilerPassManager 产生的结构化 trace。
        """
        from planner.compiler_pass import NumericPolicy, PassContext
        from planner.optimizer_passes import build_optimizer_pass_manager

        numeric_policy = (
            NumericPolicy.production_default()
            if production
            else NumericPolicy.research_default()
        )
        ctx = PassContext(production=production, numeric_policy=numeric_policy)
        result = build_optimizer_pass_manager(self).run(plan, context=ctx)
        return result.plan, result.traces

    def _optimize_legacy_reference(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        """REM-011: 遗留实现的精确副本，仅用于测试 parity oracle。

        此方法是语义参考实现，用于证明新 pass-manager 路由与旧手链输出完全一致。
        生产代码不得调用此方法。
        """
        from planner.canonicalize_params import (
            canonicalize_plan_parameters,
            validate_plan_params,
        )
        from planner.composite_lowering import lower_composite_operators
        from planner.rewrite_fastpath import rewrite_plan_for_fastpath

        folded = self._fold_literals(plan)
        # R6 P0-04: parameters must be validated BEFORE composite lowering so a
        # lowering can never truncate a 5.9 or accept a NaN/Inf that runtime
        # validation rejects.  Lowering only reads already-canonicalized values.
        validate_plan_params(folded, production=production)
        lowered = lower_composite_operators(folded)
        refolded = self._fold_literals(lowered)
        rewritten = rewrite_plan_for_fastpath(
            refolded,
            allow_semantic_rewrites=(self.allow_semantic_rewrites and not production),
        )
        return canonicalize_plan_parameters(rewritten)

    def _fold_literals(self, node: PlanNode) -> PlanNode:
        inputs = [self._fold_literals(c) for c in node.inputs]
        n = PlanNode(
            op=node.op,
            inputs=inputs,
            attrs=dict(node.attrs),
            semantic_attrs=dict(node.semantic_attrs),
            node_id=node.node_id,
        )
        if n.op in ("add", "subtract", "sub", "multiply", "mul", "divide", "div") and len(inputs) == 2:
            a, b = inputs
            if a.op == "literal" and b.op == "literal":
                va, vb = float(a.attrs["value"]), float(b.attrs["value"])
                if n.op == "add":
                    out = va + vb
                elif n.op in ("subtract", "sub"):
                    out = va - vb
                elif n.op in ("multiply", "mul"):
                    out = va * vb
                else:
                    if vb == 0.0:
                        return n
                    out = va / vb
                # Audit #389 finite gate: 1e308 * 1e308 overflows to +inf during
                # constant folding.  Folding would manufacture an ``inf`` literal
                # that the runtime's numeric semantics may treat differently from
                # an evaluated multiply (e.g. protected_div / NaN handling).  When
                # the folded result is a non-finite float, DON'T fold — leave the
                # operator node in place and let the runtime apply its own
                # arithmetic semantics to the operands.
                if isinstance(out, float) and not math.isfinite(out):
                    return n
                return PlanNode(
                    op="literal",
                    attrs={"value": out},
                    inputs=[],
                    semantic_attrs=dict(n.semantic_attrs),
                    node_id=n.node_id,
                )
        if n.op == "nary_add" and inputs and all(c.op == "literal" for c in inputs):
            out = sum(float(c.attrs["value"]) for c in inputs)
            # Audit #389: same finite gate for nary sums (e.g. 1e308 + 1e308).
            if isinstance(out, float) and not math.isfinite(out):
                return n
            return PlanNode(
                op="literal",
                attrs={"value": out},
                inputs=[],
                semantic_attrs=dict(n.semantic_attrs),
                node_id=n.node_id,
            )
        if n.op == "nary_mul" and inputs and all(c.op == "literal" for c in inputs):
            prod = 1.0
            for c in inputs:
                prod *= float(c.attrs["value"])
            # Audit #389: same finite gate for nary products.
            if isinstance(prod, float) and not math.isfinite(prod):
                return n
            return PlanNode(
                op="literal",
                attrs={"value": prod},
                inputs=[],
                semantic_attrs=dict(n.semantic_attrs),
                node_id=n.node_id,
            )
        return n

# -*- coding: utf-8 -*-
"""Normalize legacy parameter spellings after all plan rewrites.

Analyzer canonicalizes user-authored keyword aliases before IR construction, but
optimizer/composite rules can synthesize new ``PlanNode.attrs`` later.  This
final pass guarantees execution layers only receive canonical parameters.

Also hosts the R6 P0-04 pre-lowering parameter validator and the R6 P1-19
``ParameterCanonicalizer``: composite lowering must never truncate / re-legalize
a parameter that runtime validation rejects, and algebraically-identical
parameter values (proportional weights, pure scale, float noise) must hash to
the same factor.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from planner.logical_plan import PlanNode


def validate_plan_params(plan: PlanNode, *, production: bool = True) -> PlanNode:
    """R6 P0-04: validate every operator node's scalar ``attrs`` against its
    declared ``ParamSpec`` BEFORE composite lowering runs.

    The pipeline contract is ``Parse → canonical parameter validation → typed IR
    → lowering → backend``.  Lowering must receive already-canonicalized values;
    this walk is the planning-time enforcement point.  Only operators that
    declare a ``ParamSpec`` / ``param_types`` are enforced — legacy operators
    without contracts are left untouched so nothing regresses.
    """
    from backend.parameter_aliases import normalize_parameter_aliases
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.base import _normalise_integer
    from backend.operator_errors import OperatorParameterError

    def _validate_op(node: PlanNode) -> None:
        op = str(node.op or "")
        if op in {"column", "literal", "plan_ref", "materialized_series"}:
            return
        canonical = OperatorRegistry._aliases.get(op, op)
        meta = getattr(OperatorRegistry.get(canonical), "metadata", None)
        if meta is None:
            return
        param_names = list(getattr(meta, "param_names", None) or [])
        if not param_names:
            return
        specs = getattr(meta, "param_specs", None) or {}
        types = getattr(meta, "param_types", None) or {}
        attrs = normalize_parameter_aliases(canonical, dict(node.attrs))
        # ``_normalise_integer`` is the SAME resolver the runtime uses in
        # ``validate_operator_call``: ParamSpec.dtype -> param_types ->
        # integer-name whitelist.  Reusing it gives planning-time == runtime
        # parity, so a ``window=5.9`` that runtime rejects is also rejected
        # here before any lowering can truncate it.
        for key, value in attrs.items():
            if key not in param_names:
                continue
            # bool must NOT be skipped — it is the bool-as-int case that
            # ``_normalise_integer`` rejects.  Only structured/non-scalar values
            # (lists, dicts, panels) are not plan scalar params.
            if not isinstance(value, (int, float, str, bool)):
                continue
            try:
                _normalise_integer(value, key, types.get(key), specs.get(key))
            except OperatorParameterError as exc:
                raise OperatorParameterError(
                    f"{canonical}.{key}: planning-time parameter validation "
                    f"failed (R6 P0-04): {exc}"
                ) from exc

    for node in _walk(plan):
        _validate_op(node)
    return plan


def _walk(node: PlanNode):
    yield node
    for child in node.inputs:
        yield from _walk(child)


def canonicalize_parameter_values(attrs: dict[str, Any]) -> dict[str, Any]:
    """R6 P1-19: canonicalize parameter VALUES so algebraically-identical
    factors share one AST hash.

    - floats are rounded to 12 significant digits (float-noise equality);
    - proportional weight arrays (``*_weight``/``*_weights``/``weights``) are
      normalized to unit sum when they are finite and non-empty;
    - booleans stay booleans; ints stay ints.
    """
    import math

    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool)):
            num = float(value)
            if math.isfinite(num) and num != 0.0:
                out[key] = _round_sig(num, 12)
                continue
            out[key] = value
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            seq = list(value)
            if key.endswith("weight") or key.endswith("weights"):
                nums = [float(v) for v in seq if isinstance(v, (int, float))]
                total = sum(nums)
                if nums and math.isfinite(total) and abs(total) > 1e-12:
                    out[key] = tuple(_round_sig(v / total, 12) for v in nums)
                    continue
            out[key] = tuple(_round_sig(float(v), 12) if isinstance(v, (int, float)) else v for v in seq)
            continue
        out[key] = value
    return out


def _round_sig(value: float, digits: int) -> float:
    import math

    if value == 0.0 or not math.isfinite(value):
        return value
    try:
        shift = digits - int(math.floor(math.log10(abs(value)))) - 1
    except (ValueError, OverflowError):
        return value
    factor = 10.0 ** shift
    return math.floor(value * factor + 0.5) / factor


class ParameterCanonicalizer:
    """R6 P1-19: turn an operator node's parameter dict into a canonical form
    used for AST-hash dedup.  Two plans that differ only by proportional weight
    scale, pure float noise, or inactive parameters hash identically.

    The dedup chain is explicitly: ``canonicalize`` -> canonical parameter dict
    -> ``hash_key`` produces the equivalence-class key -> the caller (CSE /
    factor_dedup) keeps one representative per key.  No mutable ``_seen`` cache
    is stored here: it would only ever grow across calls (memory leak) and the
    caller already owns the key->node map.  Audit #388: removed the dead
    ``self._seen`` field that was initialized but never read or written.
    """

    def __init__(self) -> None:
        pass

    def canonicalize(self, canonical: str, attrs: dict[str, Any]) -> dict[str, Any]:
        return canonicalize_parameter_values(attrs)

    def hash_key(self, canonical: str, attrs: dict[str, Any]) -> tuple:
        canon = self.canonicalize(canonical, attrs)
        return (canonical,) + tuple(sorted((k, _freeze(v)) for k, v in canon.items()))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def canonicalize_plan_parameters(plan: PlanNode) -> PlanNode:
    """Return a structurally equivalent plan with canonical attrs everywhere."""
    from backend.parameter_aliases import normalize_parameter_aliases
    from cleaned_operators.registry import OperatorRegistry

    inputs = [canonicalize_plan_parameters(child) for child in plan.inputs]
    op = str(plan.op)
    attrs = dict(plan.attrs)
    if op not in {"column", "literal", "plan_ref", "materialized_series"}:
        canonical = OperatorRegistry._aliases.get(op, op)
        attrs = normalize_parameter_aliases(canonical, attrs)
        attrs = canonicalize_parameter_values(attrs)
        op = canonical
    return PlanNode(op=op, inputs=inputs, attrs=attrs, node_id=plan.node_id)

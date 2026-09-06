# -*- coding: utf-8 -*-
"""Normalize legacy parameter spellings after all plan rewrites.

Analyzer canonicalizes user-authored keyword aliases before IR construction, but
optimizer/composite rules can synthesize new ``PlanNode.attrs`` later.  This
final pass guarantees execution layers only receive canonical parameters.

Also hosts the R6 P0-04 pre-lowering parameter validator and the R6 P1-19
``ParameterCanonicalizer``: composite lowering must never truncate / re-legalize
a parameter that runtime validation rejects, and algebraically-identical
parameter values (declared-scale weights, float noise) must hash to the same
factor.

R13 NEW-P0-17/18/19/20 (execution vs hash split):

* EXECUTION parameters are NEVER rewritten for dedup.  ``canonicalize_plan_parameters``
  only re-spells parameter *names* (alias → canonical) and canonicalizes the op
  name; every scalar/vector VALUE the user passed is carried into the execution
  plan verbatim — a ``0.123456789012345`` stays ``0.123456789012345``.
* HASH canonicalization lives on the *hash* side only (``canonicalize_parameter_values``
  / ``structural_key`` / ``ParameterCanonicalizer.hash_key``): float-noise is
  rounded to 12 significant digits, and a parameter is unit-sum-normalized ONLY
  when its ``ParamSpec`` explicitly declares ``equivalence="positive_scale"``.
  Weight normalization is never inferred from the parameter NAME.
* A vector parameter is never silently length-changed: any non-numeric element
  in a declared scale-equivalent vector is a hard error, and non-weight sequences
  keep every element (numerics rounded, others verbatim).
* ``validate_plan_params`` runs the SAME authority the runtime call gate uses —
  per-param dtype/min/max/choices/finite, ``active_when``, the common
  window/min_periods + k relations, and every declared ``RelationalParamSpec`` —
  so lowering can never see a logically-illegal combination.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from factor_engine.planner.logical_plan import PlanNode


def _operator_contract(canonical: str):
    """Lazily resolve ``(operator, metadata, param_specs, param_types)`` for a canonical."""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
    except Exception:
        operator = None
    if operator is None:
        return None, None, {}, {}
    meta = getattr(operator, "metadata", None)
    if meta is None:
        return operator, None, {}, {}
    return (
        operator,
        meta,
        getattr(meta, "param_specs", None) or {},
        getattr(meta, "param_types", None) or {},
    )


def validate_plan_params(plan: PlanNode, *, production: bool = True) -> PlanNode:
    """R6 P0-04 + R13 NEW-P0-20: validate every operator node's scalar ``attrs``
    against its declared contract BEFORE composite lowering runs.

    The pipeline contract is ``Parse → canonical parameter validation → typed IR
    → lowering → backend``.  Lowering must receive already-canonicalized values;
    this walk is the planning-time enforcement point.  Only operators that
    declare a ``ParamSpec`` / ``param_types`` are enforced — legacy operators
    without contracts are left untouched so nothing regresses.

    R13 NEW-P0-20: validation is the FULL runtime authority, not just the integer
    normalizer — per-param dtype/min/max/choices/finite (via ``_normalise_integer``),
    ``active_when`` (dead-knob rejection), the common ``min_periods <= window`` /
    ``k <= window`` relations, and every declared ``RelationalParamSpec`` are all
    enforced here with the SAME helpers ``validate_operator_call`` uses, so a
    logically-illegal parameter combination can never survive past planning.
    """
    from factor_engine.backend.parameter_aliases import normalize_parameter_aliases
    from factor_engine.cleaned_operators.base import (
        _enforce_active_when,
        _kernel_param_defaults,
        _validate_common_integer_relations,
        _validate_relational_specs,
    )
    from factor_engine.cleaned_operators.common.strict_params import normalize_and_validate_scalar_param
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.operator_errors import OperatorParameterError

    def _validate_op(node: PlanNode) -> None:
        op = str(node.op or "")
        if op in {"column", "literal", "plan_ref", "materialized_series"}:
            return
        canonical = OperatorRegistry._aliases.get(op, op)
        operator, meta, specs, types = _operator_contract(canonical)
        if operator is None or meta is None:
            return
        param_names = list(getattr(meta, "param_names", None) or [])
        if not param_names:
            return
        attrs = normalize_parameter_aliases(canonical, dict(node.attrs))
        # Positionally-passed scalar parameters appear as ``literal`` child
        # inputs, NOT ``node.attrs`` (the runtime binds every evaluated input in
        # child order — panels and literals mixed — to the kernel signature, so
        # the index-into-param_names mapping is the same one the analyzer's
        # ``_operator_param_values`` uses).  Validating on ``attrs`` alone would
        # skip these params entirely (and run relational specs on an empty
        # bound, raising a misleading "fast < slow" error for a valid call).
        for _idx, _child in enumerate(node.inputs):
            if _idx >= len(param_names):
                break
            # R19-003: ``"value" in child.attrs`` (NOT ``attrs.get("value") is not
            # None``) — an explicit ``None`` positional literal is a real bound
            # value that must enter ParamSpec validation, active_when,
            # relational-constraint and hash identity, never be conflated with a
            # missing literal.
            if str(getattr(_child, "op", "")) == "literal" and "value" in _child.attrs:
                attrs.setdefault(param_names[_idx], _child.attrs["value"])
        defaults = _kernel_param_defaults(operator) or {}

        # per-param scalar validation (dtype / min / max / choices / finite).
        # R19-002: ``normalize_and_validate_scalar_param`` is the SAME unified
        # authority the runtime call gate uses — planning-time and runtime share
        # one declared ParamSpec type domain, so ``window=5.9`` that runtime
        # rejects is also rejected here, and np.int64/np.float64/Decimal/enum
        # scalars are validated (not silently skipped because they are not
        # ``isinstance(value, (int, float, str, bool))``).  Structured/vector
        # values (lists, dicts) are passed through unchanged by the unified entry
        # exactly as the runtime does.
        validated: dict[str, Any] = {}
        for key, value in attrs.items():
            if key not in param_names:
                continue
            try:
                validated[key] = normalize_and_validate_scalar_param(
                    canonical, key, value, phase="planning",
                    declared_type=types.get(key), spec=specs.get(key),
                )
            except OperatorParameterError as exc:
                raise OperatorParameterError(
                    f"{canonical}.{key}: planning-time parameter validation "
                    f"failed (R6 P0-04 / R13 NEW-P0-20): {exc}"
                ) from exc

        # merge canonical defaults so relation/active_when judgements see the
        # same value set the kernel receives at runtime.
        bound = dict(defaults)
        bound.update(validated)

        try:
            # dead-knob / active_when rejection (same helper as runtime).
            _enforce_active_when(meta, (), bound, defaults)
            # common cross-parameter relations (min_periods<=window, k<=window).
            _validate_common_integer_relations(meta, (), bound)
            # declared RelationalParamSpec expressions.
            _validate_relational_specs(meta, operator, (), bound)
        except Exception as exc:  # noqa: BLE001 — wrap-and-raise, never swallow
            raise OperatorParameterError(
                f"{canonical}: planning-time relational/active_when validation "
                f"failed (R13 NEW-P0-20): {exc}"
            ) from exc

    for node in _walk(plan):
        _validate_op(node)
    return plan


def _walk(node: PlanNode):
    yield node
    for child in node.inputs:
        yield from _walk(child)


def canonicalize_parameter_values(
    attrs: Mapping[str, Any], *, canonical: str | None = None
) -> dict[str, Any]:
    """HASH-SIDE parameter canonicalization (R13 NEW-P0-17).

    This function is for CACHE / DEDUP KEYS ONLY — it is NEVER used to build the
    execution plan.  It merges algebraically-identical factor arguments onto one
    key:

    - floats are rounded to 12 significant digits (float-noise equality);
    - a SEQUENCE parameter is unit-sum normalized ONLY when its ``ParamSpec``
      declares ``equivalence="positive_scale"`` (never guessed from the name);
    - booleans stay booleans; ints stay ints.

    R13 NEW-P0-19: a vector parameter is never silently length-changed.  A
    declared scale-equivalent vector containing any non-numeric element raises
    (the whole parameter is rejected, not filtered).  Non-weight sequences keep
    every element: numerics are rounded, everything else is preserved verbatim.

    ``canonical`` (optional) resolves the operator's declared ``ParamSpec`` so
    weight normalization is gated on ``equivalence="positive_scale"``; without a
    canonical, NO name-based normalization is applied (R13 NEW-P0-18).
    """
    import math

    _, _, param_specs, _ = (
        _operator_contract(canonical) if canonical is not None else (None, None, {}, {})
    )

    def _is_declared_scale_equivalent(key: str) -> bool:
        spec = (param_specs or {}).get(key)
        return spec is not None and getattr(spec, "equivalence", None) == "positive_scale"

    def _round_num(value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        num = float(value)
        if math.isfinite(num) and num != 0.0:
            return _round_sig(num, 12)
        return value

    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool)):
            out[key] = _round_num(value)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            seq = list(value)
            if _is_declared_scale_equivalent(key):
                # strict vector: every element must be numeric — never drop.
                if not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in seq
                ):
                    from factor_engine.backend.operator_errors import OperatorParameterError

                    raise OperatorParameterError(
                        f"{key}: a scale-equivalent weight vector must contain "
                        f"only numeric elements (got a non-numeric element); the "
                        "whole parameter is rejected rather than silently "
                        "filtered (R13 NEW-P0-19)"
                    )
                nums = [float(v) for v in seq]
                total = sum(nums)
                if nums and math.isfinite(total) and abs(total) > 1e-12:
                    out[key] = tuple(_round_sig(v / total, 12) for v in nums)
                    continue
                out[key] = tuple(seq)
                continue
            # non-weight / undeclared sequence: keep EVERY element, length stable.
            out[key] = tuple(_round_num(v) for v in seq)
            continue
        out[key] = value
    return out


def exact_parameter_values(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return values for an exact execution/CSE identity without approximation.

    This deliberately does not apply significant-digit rounding or declared
    search equivalences.  Containers are copied into deterministic immutable
    shapes, while scalar types and values (including large integers and signed
    zero) are preserved for the typed plan encoder.
    """

    def _exact(value: Any) -> Any:
        if isinstance(value, Mapping):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("exact parameter mappings require string keys")
            return {k: _exact(value[k]) for k in sorted(value)}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(_exact(v) for v in value)
        return value

    if not all(isinstance(k, str) for k in attrs):
        raise TypeError("exact parameter attrs require string keys")
    return {k: _exact(attrs[k]) for k in sorted(attrs)}


def _round_sig(value: float, digits: int) -> float:
    import math

    if value == 0.0 or not math.isfinite(value):
        return value
    try:
        shift = digits - int(math.floor(math.log10(abs(value)))) - 1
    except (ValueError, OverflowError):
        return value
    try:
        factor = 10.0 ** shift
        return math.floor(value * factor + 0.5) / factor
    except OverflowError:
        # Only the scale construction for subnormal finite values takes this
        # fallback. Normal values retain the legacy half-up behavior exactly.
        return float(format(value, f".{digits}g"))


class ParameterCanonicalizer:
    """R6 P1-19: turn an operator node's parameter dict into a canonical form
    used for AST-hash dedup.  Two plans that differ only by declared-scale
    proportional weight, pure float noise, or inactive parameters hash identically.

    The dedup chain is explicitly: ``canonicalize`` -> canonical parameter dict
    -> ``hash_key`` produces the equivalence-class key -> the caller (CSE /
    factor_dedup) keeps one representative per key.  No mutable ``_seen`` cache
    is stored here: it would only ever grow across calls (memory leak) and the
    caller already owns the key->node map.  Audit #388: removed the dead
    ``self._seen`` field that was initialized but never read or written.

    R13 NEW-P0-17/18: this is the HASH side — the canonical form produced here is
    a dedup key, never the execution plan's attrs.  Weight normalization is gated
    on ``ParamSpec.equivalence == "positive_scale"`` (resolved via ``canonical``),
    not on the parameter name.
    """

    def __init__(self, canonical: str | None = None) -> None:
        self.canonical = canonical

    def canonicalize(self, attrs: dict[str, Any]) -> dict[str, Any]:
        return canonicalize_parameter_values(attrs, canonical=self.canonical)

    def hash_key(self, attrs: dict[str, Any]) -> tuple:
        canon = self.canonicalize(attrs)
        # R19-001: a stable tuple schema — ``(canonical, sorted-items)``.  The old
        # ``str + tuple`` was illegal (TypeError: can only concatenate str not
        # tuple to str); the canonical form must be a deterministic, cross-process
        # hashable key: same params -> same key, different semantic params ->
        # different key, and the element order inside each pair is stable so the
        # tuple is order-invariant regardless of dict iteration order.
        return (
            "search-equivalence-v1",
            self.canonical or "",
            tuple(sorted((k, _freeze(v)) for k, v in canon.items())),
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def canonicalize_plan_parameters(plan: PlanNode) -> PlanNode:
    """Return a structurally equivalent EXECUTION plan with canonical SPELLING
    everywhere — op name canonicalized and parameter names re-spelled from
    aliases — while every VALUE stays exactly as the user passed it
    (R13 NEW-P0-17: dedup canonicalization never mutates execution parameters).

    ``semantic_attrs`` is carried through untouched (R13 NEW-P0-14): the typed
    layer's unit / grain / availability / price-basis / source identity must
    survive every optimizer rewrite.
    """
    from factor_engine.backend.parameter_aliases import normalize_parameter_aliases
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    inputs = [canonicalize_plan_parameters(child) for child in plan.inputs]
    op = str(plan.op)
    attrs = dict(plan.attrs)
    if op not in {"column", "literal", "plan_ref", "materialized_series"}:
        canonical = OperatorRegistry._aliases.get(op, op)
        attrs = normalize_parameter_aliases(canonical, attrs)
        op = canonical
    return PlanNode(
        op=op,
        inputs=inputs,
        attrs=attrs,
        semantic_attrs=dict(plan.semantic_attrs),
        node_id=plan.node_id,
    )

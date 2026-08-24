# -*- coding: utf-8 -*-
"""Semantic Closure shared layer (Master Spec Part 0 / P0).

This package is the framework-level front door for the Operator Semantic
Closure: a central MissingPolicy / WindowSemantics vocabulary, the SameAxis
axis-contract guards, and a registry-level closure audit that mechanically
reports the release invariants (Part BY) across the whole public catalog.

The shared layer deliberately does NOT edit ``cleaned_operators.base`` — it
sits beside it as a declarative side-registry so operators and audits can adopt
the vocabulary without touching the frozen core dataclasses.
"""
from factor_engine.cleaned_operators.closure.axis_contract import (
    SameAxisError,
    assert_same_axes,
    assert_same_columns,
    assert_same_index,
    assert_unique_columns,
    assert_unique_index,
    axis_contract_for,
    declare_axis_contract,
    has_axis_contract,
    same_columns,
    same_index,
)
from factor_engine.cleaned_operators.closure.missing_policy import (
    MissingPolicy,
    current_required_family_for,
    declare_current_required_family,
    declare_missing_policy,
    missing_policy_for,
    policy_value,
)
from factor_engine.cleaned_operators.closure.window_semantics import (
    WindowSemantics,
    declare_window_semantics,
    semantics_value,
    window_semantics_for,
)
from factor_engine.cleaned_operators.closure.declared_policies import declare_all as declare_round14_policies  # noqa: E402

__all__ = [
    "SameAxisError",
    "assert_same_axes",
    "assert_same_columns",
    "assert_same_index",
    "assert_unique_columns",
    "assert_unique_index",
    "axis_contract_for",
    "declare_axis_contract",
    "has_axis_contract",
    "same_columns",
    "same_index",
    "MissingPolicy",
    "current_required_family_for",
    "declare_current_required_family",
    "declare_missing_policy",
    "missing_policy_for",
    "policy_value",
    "WindowSemantics",
    "declare_window_semantics",
    "semantics_value",
    "window_semantics_for",
]

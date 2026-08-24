# -*- coding: utf-8 -*-
"""R10 #18/#19/#20/#28/#29/#30 — parameter-semantics review fixes (2026-08).

* R10 #18 — ``active_when`` is resolvable WITHOUT the controller being
  explicitly provided (the controller's ``ParamSpec.default`` is used), and an
  INACTIVE parameter explicitly bound to a non-default value is rejected.
* R10 #19 — parameter alias single binding: canonical + alias (or two aliases
  of the same canonical) bound together is a duplicate logical binding error.
* R10 #20 — ParameterSensitivity distinguishes COMPOSITIONAL_SENSITIVITY from
  TERMINAL_RANK_EQUIVALENCE; the compositional signature binds raw-value
  behavior so ``f -> 2f`` / ``f -> f+1`` are NOT judged insensitive.
* R10 #28 — explicit ``panel_params`` / ``input_fields`` metadata takes
  precedence over the legacy numeric-control-name heuristic.
* R10 #29 — a failed contract build returns :data:`CONTRACT_ERROR` and excludes
  the operator from production search (no name-guessing fallback).
* R10 #30 — a relational predicate that RAISES is a :data:`CONTRACT_ERROR`
  marker, never ``_PROBE_NOT_APPLICABLE``.

The ``_SpecStub`` / ``_RelStub`` classes mirror ``cleaned_operators.base.ParamSpec``
/ ``RelationalParamSpec`` (duck-typed by ``parameter_canonicalizer``) so the
tests stay fast and self-contained.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factor_engine.parameter_canonicalizer import (
    CONTRACT_ERROR,
    ParameterCanonicalizer,
    ParameterContractError,
    _PROBE_NOT_APPLICABLE,
    _check_relational_specs,
    _output_signature,
    _probe_value,
    build_operator_contract,
    classify_panel_scalar_params,
    operator_contract_searchable,
    sensitivity_verified,
)


class _SpecStub:
    """Duck-typed stand-in for ``cleaned_operators.base.ParamSpec``."""

    def __init__(
        self,
        dtype=None,
        min=None,
        max=None,
        choices=None,
        active_when=None,
        default=None,
    ):
        self.dtype = dtype
        self.min = min
        self.max = max
        self.choices = choices
        self.active_when = active_when
        self.default = default


class _RelStub:
    """Duck-typed stand-in for ``cleaned_operators.base.RelationalParamSpec``."""

    def __init__(self, check):
        self._check = check

    def check(self, bound):
        return self._check(bound)


class _BoomRel:
    """A relational predicate whose ``check`` RAISES (a genuine contract bug)."""

    def check(self, bound):  # noqa: ARG002
        raise RuntimeError("relational interpreter bug")


# ---------------------------------------------------------------------------
# R10 #18 — active_when resolvable without the controller being provided
# ---------------------------------------------------------------------------


def test_active_when_controller_omitted_uses_default_dependent_active():
    # Controller ``mode`` is OMITTED from the param dict; its declared default
    # ``"A"`` makes the dependent ``w`` ACTIVE, so legal probe alternatives are
    # produced (not just the current value).
    specs = {
        "mode": _SpecStub(dtype=str, choices=("A", "B"), default="A"),
        "w": _SpecStub(
            dtype=int, min=2, max=20, default=5, active_when=("mode", ("A",))
        ),
    }
    v0 = _probe_value("w", 0, 5, {}, specs["w"], None, specs)
    v1 = _probe_value("w", 1, 5, {}, specs["w"], None, specs)
    assert isinstance(v0, int) and 2 <= v0 <= 20
    # Active knob -> probe 1 is a distinct legal alternative, not NOT_APPLICABLE.
    assert v1 is not _PROBE_NOT_APPLICABLE
    assert v1 != v0


def test_active_when_controller_omitted_uses_default_dependent_inactive():
    # Controller default ``"B"`` makes ``w`` INACTIVE -> probing is a no-op.
    specs = {
        "mode": _SpecStub(dtype=str, choices=("A", "B"), default="B"),
        "w": _SpecStub(
            dtype=int, min=2, max=20, default=5, active_when=("mode", ("A",))
        ),
    }
    assert _probe_value("w", 0, 5, {}, specs["w"], None, specs) == 5
    assert _probe_value("w", 1, 5, {}, specs["w"], None, specs) is _PROBE_NOT_APPLICABLE


def test_active_when_inactive_parameter_explicit_non_default_rejected():
    specs = {
        "mode": _SpecStub(dtype=str, choices=("A", "B"), default="B"),
        "w": _SpecStub(
            dtype=int, min=2, max=20, default=5, active_when=("mode", ("A",))
        ),
    }
    pc = ParameterCanonicalizer("op", [], param_specs=specs)
    # Controller omitted -> default "B" -> inactive; w=10 is a non-default value.
    with pytest.raises(ValueError, match="inactive"):
        pc.canonical_key({"w": 10})
    # The canonical default (dead knob set to its no-op value) is tolerated.
    assert pc.canonical_key({"w": 5}) == (("w", "5"),)
    # Explicit controller value "B" also marks w inactive -> reject.
    with pytest.raises(ValueError, match="inactive"):
        pc.canonical_key({"w": 10, "mode": "B"})
    # Active controller value "A" allows a genuine knob.
    assert pc.canonical_key({"w": 7, "mode": "A"}) == (("mode", "'A'"), ("w", "7"))


def test_active_when_dependent_omitted_never_rejected():
    specs = {
        "mode": _SpecStub(dtype=str, choices=("A", "B"), default="B"),
        "w": _SpecStub(
            dtype=int, min=2, max=20, default=5, active_when=("mode", ("A",))
        ),
    }
    pc = ParameterCanonicalizer("op", [], param_specs=specs)
    # The inactive dependent is simply absent -> fine.
    assert pc.canonical_key({}) == ()


# ---------------------------------------------------------------------------
# R10 #19 — parameter alias single binding
# ---------------------------------------------------------------------------


def test_alias_duplicate_binding_canonical_plus_alias_rejected():
    pc = ParameterCanonicalizer("op", [], param_aliases={"d": "window"})
    with pytest.raises(ValueError, match="duplicate logical binding"):
        pc.canonical_key({"window": 20, "d": 30})


def test_alias_duplicate_binding_two_aliases_rejected():
    pc = ParameterCanonicalizer("op", [], param_aliases={"d": "window", "w": "window"})
    with pytest.raises(ValueError, match="duplicate logical binding"):
        pc.canonical_key({"d": 30, "w": 30})


def test_alias_single_binding_accepted_and_maps_to_canonical_key():
    pc = ParameterCanonicalizer("op", [], param_aliases={"d": "window"})
    # One alias bound alone is fine and canonicalizes to the same key as the
    # canonical spelling.
    assert pc.canonical_key({"d": 30}) == pc.canonical_key({"window": 30})
    # A single canonical param without any alias is unchanged.
    assert pc.canonical_key({"window": 30}) == (("window", "30"),)


def test_no_alias_declared_no_duplicate_detection_needed():
    pc = ParameterCanonicalizer("op", [])
    assert pc.canonical_key({"window": 20, "other": 30}) == (
        ("other", "30"),
        ("window", "20"),
    )


# ---------------------------------------------------------------------------
# R10 #20 — compositional vs terminal-rank-equivalence sensitivity
# ---------------------------------------------------------------------------


def test_compositional_mode_scaling_not_judged_insensitive():
    spec = _SpecStub(dtype=float, choices=(1.0, 2.0))

    def evaluate(p):
        k = p["k"]
        return np.array([1.0, 2.0, 3.0, 4.0]) * k

    # f vs 2f: same finite-mask, same cross-sectional rank, DIFFERENT raw values.
    assert _output_signature(np.array([1.0, 2.0, 3.0, 4.0])) != _output_signature(
        np.array([2.0, 4.0, 6.0, 8.0])
    )
    assert sensitivity_verified(
        "op", ["k"], evaluate, {"k": 1.0},
        probes=2, param_specs={"k": spec}, sensitivity_mode="compositional",
    )


def test_compositional_mode_additive_shift_not_judged_insensitive():
    spec = _SpecStub(dtype=float, choices=(1.0, 2.0))

    def evaluate(p):
        k = p["k"]
        return np.array([1.0, 2.0, 3.0, 4.0]) + k

    assert _output_signature(np.array([1.0, 2.0, 3.0, 4.0])) != _output_signature(
        np.array([2.0, 3.0, 4.0, 5.0])
    )
    assert sensitivity_verified(
        "op", ["k"], evaluate, {"k": 1.0},
        probes=2, param_specs={"k": spec}, sensitivity_mode="compositional",
    )


def test_terminal_rank_equivalence_mode_scaling_judged_insensitive():
    spec = _SpecStub(dtype=float, choices=(1.0, 2.0))

    def evaluate(p):
        k = p["k"]
        return np.array([1.0, 2.0, 3.0, 4.0]) * k

    # Same rank AND same min-max-normalized value pattern -> one signature.
    assert _output_signature(
        np.array([1.0, 2.0, 3.0, 4.0]), sensitivity_mode="terminal_rank_equivalence"
    ) == _output_signature(
        np.array([2.0, 4.0, 6.0, 8.0]), sensitivity_mode="terminal_rank_equivalence"
    )
    assert not sensitivity_verified(
        "op", ["k"], evaluate, {"k": 1.0},
        probes=2, param_specs={"k": spec},
        sensitivity_mode="terminal_rank_equivalence",
    )


def test_signature_binds_panel_structural_identity():
    idx1 = pd.date_range("2024-01-01", periods=2)
    df1 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx1, columns=["A", "B"])
    df2 = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.date_range("2024-01-02", periods=2),
        columns=["A", "B"],
    )
    df3 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx1, columns=["A", "C"])
    # Identical values but a different index or different columns must NOT collide.
    assert _output_signature(df1) != _output_signature(df2)
    assert _output_signature(df1) != _output_signature(df3)


def test_sensitivity_mode_rejects_unknown_policy():
    with pytest.raises(ValueError, match="sensitivity_mode"):
        sensitivity_verified(
            "op", ["k"], lambda p: np.array([1.0, 2.0]), {"k": 1.0},
            sensitivity_mode="bogus",
        )


# ---------------------------------------------------------------------------
# R10 #30 — relational predicate execution errors are CONTRACT_ERROR
# ---------------------------------------------------------------------------


def test_relational_predicate_raise_is_contract_error_not_not_applicable():
    result = _check_relational_specs(10, "window", {"window": 10}, [_BoomRel()])
    assert result is CONTRACT_ERROR
    assert result is not _PROBE_NOT_APPLICABLE
    # The marker propagates through _probe_value.
    spec = _SpecStub(dtype=int, min=1, max=10)
    assert (
        _probe_value("window", 0, 10, {"window": 10}, spec, [_BoomRel()])
        is CONTRACT_ERROR
    )


def test_relational_predicate_false_still_not_applicable():
    # A predicate that returns False (infeasible combination) is STILL
    # NOT_APPLICABLE — only a raising predicate is a CONTRACT_ERROR.
    rel = _RelStub(lambda b: b["window"] >= 4 * b["k"] + 1)
    assert (
        _probe_value("window", 1, 5, {"window": 5, "k": 1}, _SpecStub(dtype=int), [rel])
        is _PROBE_NOT_APPLICABLE
    )


def test_relational_predicate_raise_fails_closed_in_sensitivity_gate():
    spec = _SpecStub(dtype=float, choices=(1.0, 2.0))

    def evaluate(p):
        return np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ParameterContractError, match="CONTRACT_ERROR"):
        sensitivity_verified(
            "op", ["k"], evaluate, {"k": 1.0},
            probes=2, param_specs={"k": spec}, relational_specs=[_BoomRel()],
        )


# ---------------------------------------------------------------------------
# R10 #28 — explicit panel metadata beats the name heuristic
# ---------------------------------------------------------------------------


def test_explicit_panel_metadata_beats_name_heuristic():
    scalar, panel = classify_panel_scalar_params(
        ["level", "x", "window", "side", "ratio"],
        panel_params=("level", "side"),
        input_fields=("x",),
    )
    # Declared panel inputs stay panel even though their names look like
    # numeric controls.
    assert "level" in panel
    assert "side" in panel
    assert "x" in panel
    # Undeclared numeric-control-looking names are classified scalar.
    assert "window" in scalar
    assert "ratio" in scalar


def test_name_heuristic_still_fallback_when_no_metadata():
    scalar, panel = classify_panel_scalar_params(["level", "close", "volume"])
    assert "level" in scalar    # numeric-control name -> scalar
    assert "close" in panel     # non-control name -> panel
    assert "volume" in panel


# ---------------------------------------------------------------------------
# R10 #29 — fail-closed contract build
# ---------------------------------------------------------------------------


def _meta(
    name="op",
    param_names=(),
    param_specs=None,
    param_aliases=None,
    panel_params=(),
    input_fields=(),
    scalar_params=(),
):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            param_names=list(param_names),
            param_specs=dict(param_specs or {}),
            param_aliases=dict(param_aliases or {}),
            panel_params=tuple(panel_params),
            input_fields=tuple(input_fields),
            scalar_params=tuple(scalar_params),
        )
    )


def test_contract_build_failure_excludes_operator_from_production_search():
    # No metadata at all -> contract cannot be built -> CONTRACT_ERROR.
    broken = object()
    contract = build_operator_contract(broken)
    assert contract is CONTRACT_ERROR
    assert not operator_contract_searchable(contract)


def test_contract_build_failure_dead_spec_excluded():
    # A ParamSpec key that is not a param_name/alias is a dead spec -> fail closed.
    bad = _meta(param_names=["x"], param_specs={"ghost": _SpecStub()})
    assert build_operator_contract(bad) is CONTRACT_ERROR
    assert not operator_contract_searchable(build_operator_contract(bad))


def test_contract_build_failure_bad_alias_target_excluded():
    # An alias pointing at an undeclared param is a broken contract.
    bad = _meta(param_names=["x"], param_aliases={"d": "window"})
    assert build_operator_contract(bad) is CONTRACT_ERROR


def test_contract_build_success_is_searchable():
    good = _meta(
        name="op",
        param_names=["x", "window"],
        param_specs={"window": _SpecStub(dtype=int)},
        param_aliases={"d": "window"},
        panel_params=("x",),
    )
    contract = build_operator_contract(good)
    assert contract is not CONTRACT_ERROR
    assert operator_contract_searchable(contract)
    assert contract["canonical"] == "op"
    assert "window" in contract["param_names"]
    assert contract["panel_params"] == ("x",)

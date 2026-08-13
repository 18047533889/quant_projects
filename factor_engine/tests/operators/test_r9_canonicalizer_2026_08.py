# -*- coding: utf-8 -*-
"""R9-P1-037/038/039/040 — ParameterCanonicalizer review fixes (2026-08).

* R9-P1-037 — ``pure_scale`` is only dropped from the canonical key in an
  explicit ``terminal_rank_equivalence`` mode; the default compositional mode
  keeps the scale ``k``, and rank-equivalence keys are tagged.
* R9-P1-038 — a sequence parameter containing ``bool`` is rejected unless its
  ``ParamSpec`` declares ``dtype=bool``.
* R9-P1-039 — the sensitivity signature is a multi-level hash (finite-mask /
  rank-vector / quantized normalized value), so two vectors with identical
  mean/std/count but different patterns no longer collide.
* R9-P1-040 — ``_probe_value`` draws only legal values from the canonical
  grammar (dtype/choices/min/max/active_when/relational specs); a probe index
  with no legal alternative yields ``_PROBE_NOT_APPLICABLE``.

The ``_SpecStub`` / ``_RelStub`` classes mirror ``cleaned_operators.base.ParamSpec``
/ ``RelationalParamSpec`` (duck-typed by ``parameter_canonicalizer``).  They are
used instead of importing the operator package so this file stays fast and
self-contained.
"""
from __future__ import annotations

import numpy as np
import pytest

from parameter_canonicalizer import (
    ParamNormalizer,
    ParameterCanonicalizer,
    _PROBE_NOT_APPLICABLE,
    _output_signature,
    _probe_value,
    is_rank_equivalence_key,
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


# ---------------------------------------------------------------------------
# R9-P1-037 — pure_scale de-duplication is compositional-mode-gated
# ---------------------------------------------------------------------------


def test_pure_scale_stays_in_key_in_compositional_mode():
    pc = ParameterCanonicalizer("op", [ParamNormalizer("k", pure_scale=True)])
    key = pc.canonical_key({"k": 2.0, "x": 1.0})
    # Default (compositional) mode: the scale MUST remain in the canonical key.
    assert ("k", "2.0") in key
    assert not is_rank_equivalence_key(key)
    # k*f(x) and k'*f(x) are DIFFERENT expressions once composed.
    assert pc.canonical_key({"k": 3.0, "x": 1.0}) != pc.canonical_key({"k": 2.0, "x": 1.0})


def test_pure_scale_may_be_dropped_in_terminal_rank_equivalence_mode_and_tagged():
    pc = ParameterCanonicalizer(
        "op", [ParamNormalizer("k", pure_scale=True)], terminal_rank_equivalence=True
    )
    key = pc.canonical_key({"k": 2.0, "x": 1.0})
    # Rank-equivalence mode: the scale may be omitted, but the key is tagged.
    assert is_rank_equivalence_key(key)
    assert ("k", "2.0") not in key
    # In rank-equivalence mode 2*f(x) and 3*f(x) collapse to the same tagged key.
    assert pc.canonical_key({"k": 3.0, "x": 1.0}) == key


def test_rank_equivalence_tag_never_leaks_into_compositional_key():
    pc = ParameterCanonicalizer("op", [ParamNormalizer("k", pure_scale=True)])
    key = pc.canonical_key({"k": 2.0, "x": 1.0})
    assert key[0] != "__rank_equivalence__"
    assert not is_rank_equivalence_key(key)


# ---------------------------------------------------------------------------
# R9-P1-038 — bool is not a number in a sequence parameter
# ---------------------------------------------------------------------------


def test_bool_in_sequence_rejected_without_bool_dtype_spec():
    pc = ParameterCanonicalizer(
        "op", [], param_specs={"weights": _SpecStub(dtype=float)}
    )
    with pytest.raises(ValueError, match="dtype=bool"):
        pc.canonical_key({"weights": [1.0, 2.0, True]})
    with pytest.raises(ValueError, match="dtype=bool"):
        pc.canonical_key({"weights": [1, False, 3]})


def test_bool_in_sequence_rejected_when_no_spec_at_all():
    pc = ParameterCanonicalizer("op", [])
    with pytest.raises(ValueError, match="dtype=bool"):
        pc.canonical_key({"weights": [1.0, 2.0, True]})


def test_bool_in_sequence_allowed_when_spec_declares_bool_dtype():
    pc = ParameterCanonicalizer(
        "op", [], param_specs={"mask": _SpecStub(dtype=bool)}
    )
    key = pc.canonical_key({"mask": [True, False, True], "window": 5})
    assert ("mask", "[True, False, True]") in key


def test_pure_numeric_sequence_still_accepted():
    pc = ParameterCanonicalizer("op", [])
    key = pc.canonical_key({"weights": [1.0, 2.0, 3.0], "window": 5})
    assert ("weights", "[1.0, 2.0, 3.0]") in key
    # Mixed int/float (no bool) is still a legal numeric sequence.
    pc.canonical_key({"weights": [1, 2.0, 3]})


# ---------------------------------------------------------------------------
# R9-P1-039 — multi-level sensitivity signature
# ---------------------------------------------------------------------------


def test_signature_distinguishes_same_mean_std_count_different_patterns():
    # Same multiset, different order -> identical mean/std/count but a different
    # value pattern.  The old mean|std|count signature collided; the multi-level
    # hash must not.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    assert abs(a.mean() - b.mean()) < 1e-12
    assert abs(a.std() - b.std()) < 1e-12
    assert a.size == b.size
    assert _output_signature(a) != _output_signature(b)


def test_signature_distinguishes_nan_placement():
    # Same finite values, different NaN placement -> finite-mask hash differs.
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([np.nan, 1.0, 2.0, 4.0, 5.0])
    assert _output_signature(a) != _output_signature(b)


def test_signature_equivalent_for_identical_vectors():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, np.nan, 4.0])
    assert _output_signature(a) == _output_signature(b)


def test_sensitivity_gate_catches_different_patterns_same_statistics():
    # A param whose legal values drive DIFFERENT value patterns that all share
    # the same mean/std/count.  The old mean|std|count signature collapsed all
    # three outputs into one set entry (gate -> False); the multi-level
    # signature distinguishes them (gate -> True).
    spec = _SpecStub(dtype=float, choices=(1.0, 2.0, 3.0))

    def evaluate(p):
        k = p["k"]
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        if k == 1.0:
            return base
        if k == 2.0:
            return base[::-1]      # same multiset, reversed pattern
        return np.roll(base, 1)    # same multiset, rotated pattern

    assert sensitivity_verified(
        "op", ["k"], evaluate, {"k": 1.0},
        probes=3, param_specs={"k": spec},
    )


# ---------------------------------------------------------------------------
# R9-P1-040 — _probe_value respects ParamSpec (dtype/choices/min/max)
# ---------------------------------------------------------------------------


def test_probe_int_is_never_fractional():
    spec = _SpecStub(dtype=int, min=2, max=10)
    for probe in range(5):
        value = _probe_value("window", probe, 8, {"window": 8}, spec)
        if value is _PROBE_NOT_APPLICABLE:
            continue
        assert isinstance(value, int)
        assert 2 <= value <= 10


def test_probe_int_never_outside_choices():
    spec = _SpecStub(dtype=int, choices=(2, 4, 8))
    seen = []
    for probe in range(5):
        value = _probe_value("k", probe, 4, {"k": 4}, spec)
        if value is _PROBE_NOT_APPLICABLE:
            continue
        assert value in (2, 4, 8)
        seen.append(value)
    # three distinct legal choices are probed, then not-applicable
    assert sorted(seen) == [2, 4, 8]
    assert _probe_value("k", 3, 4, {"k": 4}, spec) is _PROBE_NOT_APPLICABLE


def test_probe_singleton_int_marks_later_probes_not_applicable():
    spec = _SpecStub(dtype=int, min=5, max=5)
    assert _probe_value("w", 0, 5, {"w": 5}, spec) == 5
    assert _probe_value("w", 1, 5, {"w": 5}, spec) is _PROBE_NOT_APPLICABLE


def test_probe_respects_active_when():
    spec = _SpecStub(dtype=int, min=2, max=20, active_when=("mode", ("A",)))
    # Inactive branch: only the current value is proposed.
    assert _probe_value("w", 0, 5, {"w": 5, "mode": "B"}, spec) == 5
    assert _probe_value("w", 1, 5, {"w": 5, "mode": "B"}, spec) is _PROBE_NOT_APPLICABLE
    # Active branch: normal grid.
    v = _probe_value("w", 0, 5, {"w": 5, "mode": "A"}, spec)
    assert isinstance(v, int) and 2 <= v <= 20


def test_probe_respects_relational_spec():
    spec = _SpecStub(dtype=int, min=1, max=10)
    rel = _RelStub(lambda b: b["window"] >= 4 * b["k"] + 1)
    params = {"window": 5, "k": 1}
    # probe 0 -> 5 (5 >= 5, legal); probe 1 -> 1 (1 >= 5, illegal -> N/A).
    assert _probe_value("window", 0, 5, params, spec, [rel]) == 5
    assert _probe_value("window", 1, 5, params, spec, [rel]) is _PROBE_NOT_APPLICABLE


def test_probe_float_respects_bounds():
    spec = _SpecStub(dtype=float, min=0.0, max=1.0)
    for probe in range(5):
        value = _probe_value("alpha", probe, 0.5, {"alpha": 0.5}, spec)
        if value is _PROBE_NOT_APPLICABLE:
            continue
        assert 0.0 <= value <= 1.0


def test_sensitivity_gate_skips_not_applicable_probes_and_is_false_on_singleton():
    # A singleton int parameter has exactly ONE legal value -> never sensitive.
    spec = _SpecStub(dtype=int, min=7, max=7)
    assert not sensitivity_verified(
        "op", ["w"], lambda p: np.array([float(p["w"]), 1.0]),
        {"w": 7}, param_specs={"w": spec}, probes=3,
    )
    # A two-choice int parameter is sensitive when the output depends on it.
    spec2 = _SpecStub(dtype=int, choices=(2, 4))
    assert sensitivity_verified(
        "op", ["w"], lambda p: np.array([float(p["w"]), 1.0, 2.0]),
        {"w": 2}, param_specs={"w": spec2}, probes=3,
    )


def test_probe_bool_toggles():
    spec = _SpecStub(dtype=bool)
    assert _probe_value("flag", 0, True, {"flag": True}, spec) is True
    assert _probe_value("flag", 1, True, {"flag": True}, spec) is False

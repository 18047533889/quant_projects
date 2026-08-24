# -*- coding: utf-8 -*-
"""R7-219/220/222/223/224: single strict ParamSpec binder.

A declared ``ParamSpec`` is the sole authority for a parameter's dtype, bounds,
choices, active_when and default.  Alias spellings resolve to their canonical
target before validation.  No name-whitelist heuristics may leak into a
spec'd parameter.
"""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import (
    MISSING,
    OperatorMetadata,
    ParamSpec,
    _normalise_call,
    _normalise_integer,
    _validate_param_spec,
)


def _panel():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.RangeIndex(3))


# ---------------------------------------------------------------------------
# #219 ParamSpec(int, min=None) accepts -1, 0, 1 — no legacy >=1 fallback
# ---------------------------------------------------------------------------

def test_int_min_none_accepts_negative_and_zero():
    for v in (-1, 0, 1):
        assert _normalise_integer(v, "myparam", spec=ParamSpec(dtype=int)) == v


def test_int_min_zero_rejects_negative():
    with pytest.raises(Exception):
        _normalise_integer(-1, "p", spec=ParamSpec(dtype=int, min=0))


def test_int_min_one_rejects_zero():
    with pytest.raises(Exception):
        _normalise_integer(0, "p", spec=ParamSpec(dtype=int, min=1))


def test_int_window_name_keeps_min_when_spec_bound():
    # Even for a legacy window-ish name, a spec min=0 is authoritative (not the
    # name-whitelist >=1 fallback).
    assert _normalise_integer(0, "window", spec=ParamSpec(dtype=int, min=0)) == 0
    with pytest.raises(Exception):
        _normalise_integer(-1, "window", spec=ParamSpec(dtype=int, min=0))


def test_no_spec_keeps_legacy_name_whitelist():
    # Only operators WITHOUT a spec fall back to the name heuristic.
    with pytest.raises(Exception):
        _normalise_integer(0, "window")  # legacy >=1
    assert _normalise_integer(1, "window") == 1


# ---------------------------------------------------------------------------
# #220 strict dtype validation via _validate_param_spec
# ---------------------------------------------------------------------------

def test_bool_strict_type():
    assert _validate_param_spec(True, "flag", ParamSpec(dtype=bool), None, None) is True
    with pytest.raises(Exception):
        _validate_param_spec(1, "flag", ParamSpec(dtype=bool), None, None)
    with pytest.raises(Exception):
        _validate_param_spec(1.0, "flag", ParamSpec(dtype=bool), None, None)


def test_str_strict_type():
    assert _validate_param_spec("abc", "s", ParamSpec(dtype=str), None, None) == "abc"
    with pytest.raises(Exception):
        _validate_param_spec(5, "s", ParamSpec(dtype=str), None, None)


def test_float_rejects_bool():
    with pytest.raises(Exception):
        _validate_param_spec(True, "x", ParamSpec(dtype=float), None, None)
    with pytest.raises(Exception):
        _validate_param_spec(float("nan"), "x", ParamSpec(dtype=float), None, None)
    assert _validate_param_spec(0.5, "x", ParamSpec(dtype=float), None, None) == 0.5


def test_int_rejects_bool_and_fraction():
    with pytest.raises(Exception):
        _validate_param_spec(True, "n", ParamSpec(dtype=int), None, None)
    with pytest.raises(Exception):
        _validate_param_spec(5.9, "n", ParamSpec(dtype=int), None, None)


def test_choices_exact_match_for_float():
    spec = ParamSpec(dtype=float, choices=(0.05, 0.1, 0.2))
    assert _validate_param_spec(0.1, "q", spec, None, None) == 0.1
    with pytest.raises(Exception):
        _validate_param_spec(0.15, "q", spec, None, None)


# ---------------------------------------------------------------------------
# #222 MISSING sentinel: None is a real default, MISSING is undeclared
# ---------------------------------------------------------------------------

def test_missing_sentinel_distinct_from_none():
    assert ParamSpec(dtype=float).default is MISSING
    assert ParamSpec(dtype=float, default=None).default is None
    assert MISSING is not None


def test_active_when_uses_explicit_none_default():
    # mode="A" dead knob pinned to an explicit None default is tolerated.
    meta = OperatorMetadata(
        name="op",
        category="ts",
        param_names=["x", "mode", "center"],
        panel_params=("x",),
        param_specs={
            "mode": ParamSpec(dtype=str, choices=("A", "B"), default="A"),
            "center": ParamSpec(dtype=float, active_when=("mode", ("B",)), default=None),
        },
    )
    # mode defaults to A -> center inactive; center=None (its real default) OK.
    args, kwargs = _normalise_call(meta, (_panel(),), {"center": None})
    assert kwargs.get("center") is None
    # mode="A" + center=1.0 (not the default) -> rejected.
    with pytest.raises(Exception):
        _normalise_call(meta, (_panel(),), {"center": 1.0, "mode": "A"})
    # mode="B" + center=1.0 -> active, accepted.
    args, kwargs = _normalise_call(meta, (_panel(),), {"center": 1.0, "mode": "B"})
    assert kwargs["mode"] == "B"


# ---------------------------------------------------------------------------
# #223 alias spelling validates against the canonical target's ParamSpec
# ---------------------------------------------------------------------------

def test_alias_validates_against_canonical_spec():
    meta = OperatorMetadata(
        name="ts_foo",
        category="ts",
        param_names=["window"],
        param_aliases={"d": "window"},
        param_specs={"window": ParamSpec(dtype=int, min=2, max=10)},
    )
    _, kwargs = _normalise_call(meta, (), {"d": 5})
    assert kwargs["d"] == 5
    with pytest.raises(Exception):
        _normalise_call(meta, (), {"d": 1})  # window min=2
    with pytest.raises(Exception):
        _normalise_call(meta, (), {"d": 11})  # window max=10


def test_explicit_alias_must_point_at_declared_param():
    meta = OperatorMetadata(
        name="op",
        category="ts",
        param_names=["window"],
        param_aliases={"d": "nonexistent"},
    )
    with pytest.raises(Exception):
        _normalise_call(meta, (), {"d": 5})


# ---------------------------------------------------------------------------
# #224 panel arity vs total positional arity
# ---------------------------------------------------------------------------

def test_total_positional_arity_enforced():
    meta = OperatorMetadata(
        name="ts_corr",
        category="ts",
        param_names=["x", "y", "window"],
        panel_params=("x", "y"),
        scalar_params=("window",),
        panel_arity=2,
        total_positional_arity=3,
    )
    a = _panel()
    _normalise_call(meta, (a, a, 5), {})  # OK
    with pytest.raises(Exception):
        _normalise_call(meta, (a, a, 5, 6), {})  # over-long


def test_derived_arity_from_panel_plus_scalar():
    meta = OperatorMetadata(
        name="op",
        category="ts",
        param_names=["x", "y", "window"],
        panel_params=("x", "y"),
        scalar_params=("window",),
    )
    a = _panel()
    _normalise_call(meta, (a, a, 5), {})
    with pytest.raises(Exception):
        _normalise_call(meta, (a, a, 5, 6), {})

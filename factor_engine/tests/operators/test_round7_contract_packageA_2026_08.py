# -*- coding: utf-8 -*-
"""Round-7 Package A: canonical-contract acceptance tests (2026-08-09).

Pins the audit's Parameter / DSL / Registry layer P0s:

* ParamSpec ``dtype=int`` / ``dtype=float`` reject non-numeric values — a
  declared-int param receiving the string ``"20"`` is a contract violation, not
  a silent ``int("20")`` (the DSL parser converts numeric strings at parse time).
* ``ParamSpec.choices`` is enforced for EVERY dtype (the old float branch
  returned before the choices check, so ``tail_fraction`` / ``quantile`` /
  ``alpha`` accepted any float).
* The global legacy alias set is restricted: an alias kwarg is accepted only
  when the operator declares the canonical parameter it is a synonym for — a
  kernel must not silently swallow ``alpha=`` it never declares.
* ``ParamSpec.active_when`` is enforced at runtime: an inactive parameter must
  be unprovided or equal to its canonical default, or the call is rejected.
* The manifest distinguishes ``input_fields`` (panel data) from
  ``scalar_parameters`` — ``window`` / ``lag`` are never listed as data fields.
* The registry catalog persists the full canonical contract (param_specs /
  compatible_units / param_aliases / input_grain / output_grain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    validate_operator_call,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

_rng = np.random.default_rng(0)
_IDX = pd.date_range("2024-01-01", periods=40, freq="B")
_COLS = ["A", "B", "C"]


def _mk(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((40, 3)), index=_IDX, columns=_COLS)


class _FakeOp:
    def __init__(self, name, params, specs, kernel):
        self.metadata = OperatorMetadata(
            name=name, category="test", param_names=params, param_specs=specs
        )
        self._kernel = kernel

    def _calculate_series(self, x, **_):
        return x

    def calculate(self, *args, **kwargs):
        pa, pk = validate_operator_call(self, tuple(args), dict(kwargs))
        return self._kernel(*pa, **pk)

    def validate_params(self, *a, **k):
        return True


# ---------------------------------------------------------------------------
# §1  ParamSpec dtype strictness (items 1-2)
# ---------------------------------------------------------------------------

def test_declared_int_rejects_string():
    """window='20' must raise — the runtime never int()s a string (item 1)."""
    op = _FakeOp(
        "fake_int", ["x", "window"],
        {"window": ParamSpec(dtype=int, min=2)},
        lambda x, window=60, **_: x,
    )
    with pytest.raises(Exception, match="integer"):
        op.calculate(_mk(), window="20")
    # integral float is still fine (20.0 == 20)
    out = op.calculate(_mk(), window=20.0)
    assert out.shape == (40, 3)


def test_declared_float_rejects_string_and_nonfinite():
    op = _FakeOp(
        "fake_float", ["x", "alpha"],
        {"alpha": ParamSpec(dtype=float, min=0.0, max=1.0)},
        lambda x, alpha=0.1, **_: x,
    )
    with pytest.raises(Exception, match="real number"):
        op.calculate(_mk(), alpha="0.05")
    with pytest.raises(Exception, match="finite"):
        op.calculate(_mk(), alpha=float("nan"))
    out = op.calculate(_mk(), alpha=0.05)
    assert out.shape == (40, 3)


def test_float_choices_are_enforced():
    """ParamSpec(dtype=float, choices=(...)) must reject off-choice floats — the
    old float branch returned before the choices check (item 2)."""
    op = _FakeOp(
        "fake_choice", ["x", "alpha"],
        {"alpha": ParamSpec(dtype=float, min=0.0, max=1.0, choices=(0.05, 0.1, 0.2))},
        lambda x, alpha=0.1, **_: x,
    )
    with pytest.raises(Exception, match="allowed choice"):
        op.calculate(_mk(), alpha=0.13729)
    out = op.calculate(_mk(), alpha=0.1)
    assert out.shape == (40, 3)


def test_int_choices_are_enforced():
    op = _FakeOp(
        "fake_int_choice", ["x", "k"],
        {"k": ParamSpec(dtype=int, min=1, choices=(2, 4, 8))},
        lambda x, k=2, **_: x,
    )
    with pytest.raises(Exception, match="allowed choice"):
        op.calculate(_mk(), k=3)


# ---------------------------------------------------------------------------
# §2  Legacy alias restriction (item 3)
# ---------------------------------------------------------------------------

def test_legacy_alias_accepted_only_when_canonical_declared():
    """d=5 stays legal for ts_mean (declares window); alpha= on ts_std (no
    alpha/halflife/lambda_param) must be rejected instead of silently swallowed."""
    x = _mk(seed=3)
    tsm = OperatorRegistry.get("ts_mean", "pandas_numpy") or OperatorRegistry.get("ts_mean")
    out = tsm.calculate(x, d=5)
    assert out.shape == (40, 3)
    tstd = OperatorRegistry.get("ts_std", "pandas_numpy") or OperatorRegistry.get("ts_std")
    with pytest.raises(Exception, match="undeclared keyword"):
        tstd.calculate(x, alpha=0.1)


def test_alias_swallowed_kwarg_removed_from_surface_ops():
    """Regression: an operator whose kernel declares recent_window/prior_window
    (NOT window) must reject the dead window= kwarg (the old behavior silently
    swallowed it and manufactured identical ASTs)."""
    x = _mk(seed=4)
    op = OperatorRegistry.get("ts_beta_break_score", "pandas_numpy") or OperatorRegistry.get(
        "ts_beta_break_score"
    )
    with pytest.raises(Exception, match="undeclared keyword"):
        op.calculate(x, x, window=120, recent_window=30, prior_window=90)


# ---------------------------------------------------------------------------
# §3  active_when runtime enforcement (item 4)
# ---------------------------------------------------------------------------

def test_active_when_inactive_non_default_rejected():
    """candlestick_pattern: homing_pigeon reads no window/penetration knob, so
    body_window=5 (non-default) must raise; the canonical default is a no-op."""
    op = OperatorRegistry.get("candlestick_pattern", "pandas_numpy") or OperatorRegistry.get(
        "candlestick_pattern"
    )
    frames = tuple(_mk(seed=10 + i) for i in range(4))
    with pytest.raises(Exception, match="inactive"):
        op.calculate(*frames, pattern="homing_pigeon", body_window=5)
    with pytest.raises(Exception, match="inactive"):
        op.calculate(*frames, pattern="homing_pigeon", penetration=0.5)
    out = op.calculate(*frames, pattern="homing_pigeon", body_window=10, penetration=0.3)
    assert out.shape == (40, 3)


def test_active_when_active_param_allowed():
    """long_legged_doji reads body_window AND shadow_window -> both are active."""
    op = OperatorRegistry.get("candlestick_pattern", "pandas_numpy") or OperatorRegistry.get(
        "candlestick_pattern"
    )
    frames = tuple(_mk(seed=20 + i) for i in range(4))
    out = op.calculate(*frames, pattern="long_legged_doji", body_window=7, shadow_window=9)
    assert out.shape == (40, 3)


# ---------------------------------------------------------------------------
# §4  Manifest panel-vs-scalar split (item 10)
# ---------------------------------------------------------------------------

def test_manifest_input_fields_are_panels_only():
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec, spec_to_manifest_entry

    e = spec_to_manifest_entry(build_operator_spec("ts_mean"))
    assert "x" in e["input_fields"]
    assert "window" not in e["input_fields"]
    assert "window" in e["scalar_parameters"]
    assert "lag" not in e["input_fields"]


# ---------------------------------------------------------------------------
# §5  Registry catalog full contract (item 9)
# ---------------------------------------------------------------------------

def test_registry_catalog_persists_full_contract():
    catalog = OperatorRegistry.catalog()
    entry = catalog.get("ts_mean") or {}
    assert "param_specs" in entry and isinstance(entry["param_specs"], dict)
    assert "compatible_units" in entry
    assert "param_aliases" in entry
    assert "input_grain" in entry
    assert "output_grain" in entry
    # backend_meta must still carry per-backend source
    assert "pandas_numpy" in (entry.get("backend_meta") or {})


# ---------------------------------------------------------------------------
# §6  DSL numeric-string coercion (item 1 companion)
# ---------------------------------------------------------------------------

def test_dsl_numeric_string_coerced_at_parser():
    from factor_engine.api.dsl_parser import _coerce_numeric_string

    assert _coerce_numeric_string("20") == 20 and isinstance(_coerce_numeric_string("20"), int)
    assert _coerce_numeric_string("0.05") == 0.05 and isinstance(_coerce_numeric_string("0.05"), float)
    assert _coerce_numeric_string("-3") == -3
    assert _coerce_numeric_string("doji") is None
    assert _coerce_numeric_string("upper") is None

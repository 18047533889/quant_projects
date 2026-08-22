# -*- coding: utf-8 -*-
"""R30 §9/§24: production operators carry explicit panel/scalar kinds and
explicit ParamRole on every searchable scalar.

* ``ts_threshold_cycle_period(x, lower, upper, window)`` infers panel=(x,),
  not (x, lower, upper) — required scalars are never misclassified as panels;
* no production scalar resolves to the silent ECONOMIC fallback
  (``missing_role_defaults_to_searchable`` is 0 across daily/extended).
"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_spec import _infer_panel_params
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.base import missing_role_defaults_to_searchable


def test_required_scalar_thresholds_not_misclassified_as_panels():
    op = OperatorRegistry.get("ts_threshold_cycle_period")
    meta = getattr(op, "metadata", None)
    panels = _infer_panel_params(op, meta, OperatorRegistry._catalog.get("ts_threshold_cycle_period", {}))
    assert tuple(panels) == ("x",)
    assert "lower" not in panels and "upper" not in panels


def test_envelope_scalars_not_panels():
    for c in ("ts_envelope_boundary_dwell", "ts_envelope_pressure"):
        op = OperatorRegistry.get(c)
        meta = getattr(op, "metadata", None)
        panels = _infer_panel_params(op, meta, OperatorRegistry._catalog.get(c, {}))
        assert "lower" not in panels and "upper" not in panels


def test_no_production_scalar_resolves_to_silent_economic():
    viol = []
    for c in OperatorRegistry.list_canonical():
        if classify_canonical(c) not in ("daily", "extended"):
            continue
        op = OperatorRegistry.get(c, mode="any")
        if op is None:
            continue
        meta = getattr(op, "metadata", None)
        if meta is None:
            continue
        declared = tuple(getattr(meta, "panel_params", None) or ())
        panels = set(declared) or set(_infer_panel_params(op, meta, OperatorRegistry._catalog.get(c, {})))
        specs = getattr(meta, "param_specs", None) or {}
        names = tuple(getattr(meta, "param_names", None) or ())
        for p in names:
            if p in panels:
                continue
            spec = specs.get(p)
            if spec is None:
                continue
            if missing_role_defaults_to_searchable(spec):
                viol.append(f"{c}.{p}")
    assert viol == [], f"{len(viol)} scalars resolve to silent ECONOMIC: {viol[:10]}"


def test_technical_window_helpers_carry_horizon_role():
    from cleaned_operators.technical.indicators_v2 import _WIN_GE2, _WIN_GE1, _POS_FLOAT
    from cleaned_operators.base import ParamRole

    assert _WIN_GE2.param_role is ParamRole.HORIZON
    assert _WIN_GE1.param_role is ParamRole.HORIZON
    assert _POS_FLOAT.param_role is ParamRole.STATE_THRESHOLD

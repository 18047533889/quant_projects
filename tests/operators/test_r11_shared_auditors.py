# -*- coding: utf-8 -*-
"""Round-11 §37 shared machine auditors — pytest wrapper.

Runs the cross-cutting audits (prefix-invariance A, stateful-contract discovery
B, default-history C, unit-algebra D) over a bounded operator sample so CI
enforces them without paying the full-registry cost.
"""
from __future__ import annotations

import importlib.util
import os
import random

# Load the audit modules by ABSOLUTE FILE PATH: the monorepo root ships a
# ``scripts`` package that shadows factor_engine's under pytest (the conftest's
# insert loop leaves the quant root at sys.path[0]).
_FE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_FE_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_audit = _load_module("scripts/audit_r11_longtail.py", "audit_r11_longtail_under_test")
_harness = _load_module("scripts/audit_all_factor_production.py", "audit_all_factor_production_under_test")

_canonical_sample = _audit._canonical_sample
audit_default_parameter_history = _audit.audit_default_parameter_history
audit_prefix_invariance = _audit.audit_prefix_invariance
audit_stateful_contract_discovery = _audit.audit_stateful_contract_discovery
audit_unit_algebra = _audit.audit_unit_algebra
_panels = _harness._panels


_AUDIT_TARGETS = (
    # P0 pivot/extrema rewrite targets
    "ts_confirmed_pivot_high", "ts_confirmed_pivot_low", "structural_level_density",
    "structural_level_nearest_distance", "structural_level_strength",
    "extrema_divergence", "extrema_confirmation_rate",
    # the audit's named stateful families + cross_event
    "ts_state_age_percentile", "ts_state_exit_hazard", "ts_state_residual_life",
    "ts_cusum_pressure", "event_decay_asof", "ts_time_since_change",
    "ts_hysteresis_state", "ts_hysteresis_age", "ts_state_integral",
    "ts_state_entry_strength", "cross_event",
    # representative rolling / event kernels
    "ts_mean", "ts_std", "ts_rank", "ts_autocorr", "ts_beta", "ts_sharpe",
    "ts_event_spacing_mean", "ts_event_spacing_cv", "ts_transition_count",
    "update_clock_activity", "report_filing_delay", "directional_change_state",
)


# Well-declared operators the audits must not false-flag (rolling / causal /
# event kernels with proper window params and history contracts).
_WELL_DECLARED = (
    "ts_mean", "ts_std", "ts_var", "ts_rank", "ts_median", "ts_skew",
    "ts_autocorr", "ts_beta", "ts_corr", "ts_cov", "ts_sharpe", "ts_delay",
    "ts_delta", "ts_sum", "ts_max", "ts_min", "ts_zscore", "ts_ema", "RSI_WILDER",
    "ts_positive_ratio", "ts_negative_ratio", "ts_transition_count",
    "ts_event_spacing_mean", "ts_event_spacing_cv", "ts_time_since_change",
)


def _sample() -> list[str]:
    """CI-bounded sample: the round's named audit targets plus well-declared
    operators that must survive the audits without false flags.

    The comprehensive sweep (including the pre-existing pattern-family history
    backlog) stays available via the CLI
    (``python3 scripts/audit_r11_longtail.py --sample N``); pytest keeps the
    gate focused on the operators THIS round changed + known-good rolling
    kernels so it stays green under heavy concurrent load.
    """
    import factor_engine.cleaned_operators as co

    co.load_all()
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    all_c = list(OperatorRegistry.list_canonical())
    wanted = list(_AUDIT_TARGETS) + list(_WELL_DECLARED)
    return [c for c in wanted if c in all_c]


def test_audit_a_prefix_invariance() -> None:
    panels = _panels(rows=420, columns=6)
    errors = audit_prefix_invariance(_sample(), panels)
    assert not errors, "\n".join(errors)


def test_audit_b_stateful_contract_discovery() -> None:
    panels = _panels(rows=420, columns=6)
    errors = audit_stateful_contract_discovery(_sample(), panels)
    assert not errors, "\n".join(errors)


def test_audit_c_default_parameter_history() -> None:
    panels = _panels(rows=420, columns=6)
    errors = audit_default_parameter_history(_sample(), panels)
    assert not errors, "\n".join(errors)


def test_audit_d_unit_algebra() -> None:
    errors = audit_unit_algebra()
    assert not errors, "\n".join(errors)

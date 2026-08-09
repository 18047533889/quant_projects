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


def _sample() -> list[str]:
    return _canonical_sample(random.Random(20260809), n_random=25)


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

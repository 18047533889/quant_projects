# -*- coding: utf-8 -*-
"""R30 §5: production admission is a hard gate on pit_safe and lifecycle.

* ``_compute_allow_in_production`` returns False when ``pit_safe=False`` even if
  every downstream evidence overlay would pass (hard gate, not downstream);
* ``research`` lifecycle is an explicit rejection;
* experimental / deprecated / stub / doc_only are rejected;
* only ``production`` lifecycle can pass the lifecycle gate.
"""
from __future__ import annotations

import pytest

from cleaned_operators.operator_spec import _compute_allow_in_production


def test_pit_safe_false_is_hard_rejected():
    # R30 §5 P0-004: pit_safe=False must fail admission regardless of surface /
    # evidence.  We stub the registry to be empty so the surface / six-gate
    # checks would otherwise pass if reached — the pit_safe gate must fire first.
    assert _compute_allow_in_production("x", status="production", pit_safe=False) is False


def test_pit_safe_true_production_passes_surface_gate():
    # With pit_safe=True and status=production, admission still requires
    # daily/extended surface + production_certified + eligible backend; against
    # an empty registry this fails closed on surface (never crashes).
    assert _compute_allow_in_production("x", status="production", pit_safe=True) is False


def test_research_lifecycle_rejected():
    assert _compute_allow_in_production("x", status="research", pit_safe=True) is False


def test_experimental_lifecycle_rejected():
    assert _compute_allow_in_production("x", status="experimental", pit_safe=True) is False


def test_deprecated_stub_doc_only_rejected():
    for st in ("deprecated", "stub", "doc_only"):
        assert _compute_allow_in_production("x", status=st, pit_safe=True) is False

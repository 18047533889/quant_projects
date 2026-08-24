# -*- coding: utf-8 -*-
"""R30 §8: raw ``OperatorRegistry.get`` carries a production-mode gate.

* default ``get(name)`` returns only daily/extended-surface operators;
* research / unsafe / internal / legacy operators return ``None`` unless
  ``mode="any"`` / ``mode="research"`` is passed;
* tombstoned names raise ``RemovedOperatorError`` regardless of mode;
* production execution path (``get_preferred``) is gated by the same surface
  classification through ``BackendRouter``.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.tombstones import RemovedOperatorError


def test_production_mode_excludes_research_surface():
    for name in ("ts_dmd_dominant_frequency", "ts_dmd_mode_concentration"):
        assert OperatorRegistry.get(name) is None


def test_production_mode_excludes_internal_surface():
    for name in ("constant", "identity", "protected_div"):
        assert OperatorRegistry.get(name) is None


def test_production_mode_excludes_unsafe_surface():
    for name in ("tan", "cot", "sec"):
        assert OperatorRegistry.get(name) is None


def test_production_mode_excludes_legacy_surface():
    assert OperatorRegistry.get("cube") is None


def test_any_mode_allows_research_surface():
    assert OperatorRegistry.get("ts_dmd_dominant_frequency", mode="any") is not None
    assert OperatorRegistry.get("tan", mode="any") is not None
    assert OperatorRegistry.get("cube", mode="any") is not None


def test_production_mode_allows_daily_surface():
    assert OperatorRegistry.get("ts_mean") is not None
    assert OperatorRegistry.get("ts_mean", mode="any") is not None


def test_tombstone_raises_in_any_mode():
    with pytest.raises(RemovedOperatorError):
        OperatorRegistry.get("rand_uniform", mode="any")
    with pytest.raises(RemovedOperatorError):
        OperatorRegistry.get("bfill", mode="research")

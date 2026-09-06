# -*- coding: utf-8 -*-
"""R10 #4 regression tests: an execution-contract lookup failure must NEVER
silently fall a stateful operator to stateless/independent.  Production raises
``ExecutionContractResolutionError``; research falls back with the explicit
``UNKNOWN_EXECUTION_CONTRACT`` marker and full-history (fail-closed) semantics."""
from __future__ import annotations

import pytest

from factor_engine.runtime.execution_contract import (
    ExecutionContractResolutionError,
    execution_contract,
    history_requirement,
)


def test_normal_resolution_unaffected():
    # ts_ema is a checkpoint-capable recursive operator; must stay recursive.
    c = execution_contract("ts_ema")
    assert c.state_model == "recursive"
    assert c.chunking in {"checkpoint", "required_full_history"}
    assert c.resolution_error is None


def test_stateless_operator_is_still_stateless():
    c = execution_contract("ts_mean")
    assert c.state_model == "stateless"
    assert c.chunking == "independent"


def test_production_lookup_failure_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("checkpoint registry corrupt")

    import factor_engine.stateful_contract as stateful_contract

    monkeypatch.setattr(stateful_contract.StatefulCheckpointRegistry, "get", _boom)
    with pytest.raises(ExecutionContractResolutionError):
        execution_contract("ts_ema", production=True)


def test_research_lookup_failure_marks_unknown_not_stateless(monkeypatch):
    # An operator NOT covered by the stateful seed is the dangerous fail-open
    # path: before the fix, a checkpoint-registry error silently resolved it to
    # stateless/independent.  Now research falls back with the UNKNOWN marker.
    def _boom(*a, **k):
        raise RuntimeError("checkpoint registry corrupt")

    import factor_engine.stateful_contract as stateful_contract

    monkeypatch.setattr(stateful_contract.StatefulCheckpointRegistry, "get", _boom)
    c = execution_contract("nonsense_operator_xyz")  # research default
    assert c.resolution_error == "UNKNOWN_EXECUTION_CONTRACT"
    # fail-closed: an unresolved contract is treated as full-history, NOT
    # stateless and NOT a finite independent window.
    assert c.state_model == "unknown"
    assert c.requires_full_history is True


def test_history_requirement_production_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("checkpoint registry corrupt")

    import factor_engine.stateful_contract as stateful_contract

    monkeypatch.setattr(stateful_contract.StatefulCheckpointRegistry, "get", _boom)
    with pytest.raises(ExecutionContractResolutionError):
        history_requirement("nonsense_operator_xyz", {"window": 20}, production=True)


def test_history_requirement_research_unknown_is_full_history(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("checkpoint registry corrupt")

    import factor_engine.stateful_contract as stateful_contract

    monkeypatch.setattr(stateful_contract.StatefulCheckpointRegistry, "get", _boom)
    req = history_requirement("nonsense_operator_xyz", {"window": 20})
    assert req.kind == "full_history"  # unknown contract => conservative

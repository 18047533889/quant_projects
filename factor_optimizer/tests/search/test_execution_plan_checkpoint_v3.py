import json
import os

import pytest


def _session():
    from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
    from factor_optimizer.search.runner import SearchConfig, SearchSession

    budget = SearchBudget(max_trials=3, max_evaluations=3, max_cost_units=3.0)
    config = SearchConfig(budget=budget, enable_multifidelity=False)
    return SearchSession("v3-plan", config, BudgetTracker(budget))


def test_checkpoint_binds_execution_plan_and_rejects_late_mutation(tmp_path):
    session = _session()
    session.config.plateau_threshold = 0.5

    with pytest.raises(ValueError, match="configuration changed"):
        session.checkpoint(tmp_path / "checkpoint.json")


def test_checkpoint_is_complete_replace_and_roundtrips_plan_hash(tmp_path):
    from factor_optimizer.search.runner import SearchSession

    session = _session()
    path = tmp_path / "checkpoint.json"
    path.write_text('{"old": true}', encoding="utf-8")
    session.checkpoint(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution_plan_hash"] == session.execution_plan_hash
    assert not [name for name in os.listdir(tmp_path) if name.startswith(".fo-checkpoint-")]
    restored = SearchSession.resume(path)
    assert restored.execution_plan_hash == session.execution_plan_hash


def test_tampered_config_cannot_reuse_checkpoint_plan_hash(tmp_path):
    from factor_optimizer.search.runner import SearchSession

    session = _session()
    path = tmp_path / "checkpoint.json"
    session.checkpoint(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["plateau_threshold"] = 0.25
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="execution plan hash"):
        SearchSession.resume(path)

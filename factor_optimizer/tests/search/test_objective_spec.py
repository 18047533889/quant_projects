"""ObjectiveSpec contract tests (FO-P0-06).

Covers:
- fail-closed validation at construction (unknown direction, empty metric,
  non-string metric)
- to_dict / from_dict round-trip
- pickle round-trip
- equality by content (and hashability)
- spec/field disagreement in SearchConfig raises
- legacy bare ``objective_direction`` field still works and is reconciled
- checkpoint/resume preserves the authoritative spec
"""

import json
import pickle

import pytest

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.search_budget import SearchBudget, BudgetTracker
from factor_optimizer.search.runner import SearchConfig, SearchSession


# ---------------------------------------------------------------------------
# ObjectiveSpec construction: fail-closed validation
# ---------------------------------------------------------------------------

def test_objective_spec_valid_construction():
    spec = ObjectiveSpec(metric_name="rank_ic", direction="maximize")
    assert spec.metric_name == "rank_ic"
    assert spec.direction == "maximize"

    spec_min = ObjectiveSpec(metric_name="loss", direction="minimize")
    assert spec_min.direction == "minimize"


@pytest.mark.parametrize("direction", ["sideways", "", "MAXIMIZE", "max", None, 1, True])
def test_objective_spec_rejects_unknown_direction(direction):
    with pytest.raises(ValueError, match="direction"):
        ObjectiveSpec(metric_name="rank_ic", direction=direction)


@pytest.mark.parametrize("metric", ["", "   ", None, 42, True, ["score"]])
def test_objective_spec_rejects_empty_or_invalid_metric(metric):
    with pytest.raises(ValueError, match="metric_name"):
        ObjectiveSpec(metric_name=metric, direction="maximize")


def test_objective_spec_is_frozen():
    spec = ObjectiveSpec(metric_name="rank_ic", direction="maximize")
    with pytest.raises(Exception):
        spec.direction = "minimize"
    with pytest.raises(Exception):
        spec.metric_name = "other"
    # Content is still readable after the failed mutation attempts.
    assert spec.direction == "maximize"
    assert spec.metric_name == "rank_ic"


# ---------------------------------------------------------------------------
# Serialization: to_dict / from_dict / pickle / equality
# ---------------------------------------------------------------------------

def test_objective_spec_to_from_dict_round_trip():
    spec = ObjectiveSpec(metric_name="rank_ic", direction="maximize")
    data = spec.to_dict()
    assert data == {"metric_name": "rank_ic", "direction": "maximize"}
    assert ObjectiveSpec.from_dict(data) == spec


def test_objective_spec_json_round_trip():
    spec = ObjectiveSpec(metric_name="loss", direction="minimize")
    payload = json.dumps(spec.to_dict())
    assert ObjectiveSpec.from_dict(json.loads(payload)) == spec


def test_objective_spec_pickle_round_trip():
    spec = ObjectiveSpec(metric_name="rank_ic", direction="maximize")
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
    assert restored is not spec


def test_objective_spec_equality_by_content():
    a = ObjectiveSpec(metric_name="rank_ic", direction="maximize")
    b = ObjectiveSpec(metric_name="rank_ic", direction="maximize")
    c = ObjectiveSpec(metric_name="loss", direction="maximize")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    # Content-defined dict keys work.
    assert {a: "x"}[b] == "x"


@pytest.mark.parametrize(
    "data, exc",
    [
        ({"metric_name": "score", "direction": "diagonal"}, ValueError),
        ({"metric_name": "", "direction": "maximize"}, ValueError),
        ({"metric_name": "score"}, ValueError),   # direction missing -> fail closed
        (None, TypeError),
        ("maximize", TypeError),
    ],
)
def test_objective_spec_from_dict_fail_closed(data, exc):
    with pytest.raises(exc):
        ObjectiveSpec.from_dict(data)


def test_objective_spec_from_dict_defaults_missing_metric_to_score():
    """Backward-compatible tolerance: a bare direction dict maps to 'score'."""
    spec = ObjectiveSpec.from_dict({"direction": "maximize"})
    assert spec == ObjectiveSpec(metric_name="score", direction="maximize")


def test_objective_spec_from_direction():
    assert ObjectiveSpec.from_direction("maximize") == ObjectiveSpec(
        metric_name="score", direction="maximize"
    )
    with pytest.raises(ValueError):
        ObjectiveSpec.from_direction("diagonal")


# ---------------------------------------------------------------------------
# SearchConfig reconciliation: spec is the authority
# ---------------------------------------------------------------------------

def _config(**overrides):
    budget = SearchBudget(max_trials=50, max_evaluations=50, max_cost_units=500.0)
    return SearchConfig(budget=budget, **overrides)


def test_config_default_spec():
    config = _config()
    assert config.objective_spec == ObjectiveSpec(
        metric_name="score", direction="maximize"
    )
    assert config.objective_direction == "maximize"


def test_config_legacy_field_is_reconciled_into_spec():
    config = _config(objective_direction="minimize")
    assert config.objective_spec == ObjectiveSpec(
        metric_name="score", direction="minimize"
    )
    assert config.objective_direction == "minimize"


def test_config_spec_is_authority_for_direction():
    config = _config(objective_spec=ObjectiveSpec("loss", "minimize"))
    assert config.objective_direction == "minimize"
    assert config.objective_spec.direction == "minimize"
    assert config.objective_spec.metric_name == "loss"


def test_config_spec_and_field_agree():
    config = _config(
        objective_direction="minimize",
        objective_spec=ObjectiveSpec("loss", "minimize"),
    )
    assert config.objective_direction == "minimize"
    assert config.objective_spec.metric_name == "loss"


def test_config_spec_and_field_disagreement_raises():
    with pytest.raises(ValueError, match="disagree"):
        _config(
            objective_direction="maximize",
            objective_spec=ObjectiveSpec("loss", "minimize"),
        )


def test_config_invalid_bare_field_still_rejected():
    with pytest.raises(ValueError, match="objective_direction"):
        _config(objective_direction="sideways")


def test_config_rejects_non_objective_spec():
    with pytest.raises(TypeError, match="ObjectiveSpec"):
        _config(objective_spec={"metric_name": "score", "direction": "maximize"})


def test_config_to_dict_carries_spec_not_bare_field():
    config = _config(objective_spec=ObjectiveSpec("loss", "minimize"))
    data = config.to_dict()
    assert data["objective_spec"] == {"metric_name": "loss", "direction": "minimize"}
    assert "objective_spec" in data
    # The bare field is no longer a serialized twin that can drift.
    assert "objective_direction" not in data


def test_config_dict_round_trip_preserves_spec():
    config = _config(objective_spec=ObjectiveSpec("loss", "minimize"))
    restored = SearchConfig.from_dict(config.to_dict())
    assert restored.objective_spec == ObjectiveSpec("loss", "minimize")
    assert restored.objective_direction == "minimize"
    assert restored.to_dict() == config.to_dict()


def test_config_legacy_checkpoint_dict_still_restores():
    """A pre-spec checkpoint payload (bare objective_direction) still works."""
    data = {
        "budget": SearchBudget().to_dict(),
        "plateau_window": 20,
        "plateau_threshold": 0.001,
        "enable_multifidelity": True,
        "max_concurrency": 1,
        "evaluation_cost_units": None,
        "execution_mode": "research_only",
        "require_evaluation_protocol": True,
        "objective_direction": "minimize",
    }
    restored = SearchConfig.from_dict(data)
    assert restored.objective_direction == "minimize"
    assert restored.objective_spec.direction == "minimize"
    assert restored.objective_spec.metric_name == "score"


# ---------------------------------------------------------------------------
# Checkpoint / resume preserves the spec
# ---------------------------------------------------------------------------

def test_checkpoint_resume_preserves_objective_spec(tmp_path):
    config = _config(
        objective_spec=ObjectiveSpec("loss", "minimize"),
        plateau_window=10,
        plateau_threshold=0.0,
        enable_multifidelity=False,
    )
    session = SearchSession(
        session_id="spec-ckpt",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
    )
    path = str(tmp_path / "session.json")
    session.checkpoint(path)

    restored = SearchSession.resume(path)
    assert restored.config.objective_spec == ObjectiveSpec("loss", "minimize")
    assert restored.config.objective_direction == "minimize"
    # The restored spec still drives incumbent selection.
    assert restored.update_best("t1", 0.9)
    assert restored.update_best("t2", 0.2)
    assert restored.best_score == 0.2
    assert restored.best_trial_id == "t2"


def test_checkpoint_resume_preserves_spec_through_config_only_roundtrip():
    """A checkpoint payload surviving a bare json dump keeps the spec."""
    config = _config(objective_spec=ObjectiveSpec("sharpe", "maximize"))
    data = json.loads(json.dumps(config.to_dict()))
    restored_config = SearchConfig.from_dict(data)
    assert restored_config.objective_spec == ObjectiveSpec("sharpe", "maximize")
    assert restored_config.objective_direction == "maximize"

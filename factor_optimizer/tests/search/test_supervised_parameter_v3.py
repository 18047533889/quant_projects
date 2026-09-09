import pytest

from factor_optimizer.contracts.campaign_store import CampaignStateError, SQLiteCampaignStore
from factor_optimizer.search.supervised_parameter import (
    U_CENTER_GRID, TAIL_CUTOFF_GRID, fit_supervised_parameter,
)


def test_validation_and_test_cannot_refit_u_center():
    scores = {value: -(value - 0.45) ** 2 for value in U_CENTER_GRID}
    for role in ("VALIDATION", "TEST"):
        with pytest.raises(ValueError, match="only be fitted on TRAIN"):
            fit_supervised_parameter(
                parent_factor_id="parent", repair_family="U_SHAPE_REPAIR",
                parameter_name="center", candidate_grid=U_CENTER_GRID,
                train_scores=scores, split_role=role, train_split_ref="train-v1",
                training_evidence_ref="qe:train:1", objective_id="net_utility",
            )


def test_frozen_train_choice_ignores_future_scores_and_roundtrips_sqlite(tmp_path):
    train = {value: -(value - 0.45) ** 2 for value in U_CENTER_GRID}
    state = fit_supervised_parameter(
        parent_factor_id="parent", repair_family="U_SHAPE_REPAIR",
        parameter_name="center", candidate_grid=U_CENTER_GRID,
        train_scores=train, split_role="TRAIN", train_split_ref="train-v1",
        training_evidence_ref="qe:train:1", objective_id="net_utility",
    )
    assert state.value == 0.45
    store = SQLiteCampaignStore(tmp_path / "state.sqlite3")
    store.freeze_supervised_parameter("campaign", state)
    restored = SQLiteCampaignStore(store.path).supervised_parameter("campaign", "parent", "U_SHAPE_REPAIR")
    assert restored == state
    # Arbitrarily favorable test data has no API into recipe application.
    assert restored.recipe_parameters(split_role="TEST") == {"center": 0.45}


def test_tail_cutoff_grid_is_finite_and_second_fit_cannot_replace_state(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "tail.sqlite3")
    first = fit_supervised_parameter(
        parent_factor_id="tail", repair_family="TAIL_SATURATION", parameter_name="cutoff",
        candidate_grid=TAIL_CUTOFF_GRID,
        train_scores={0.85: 0.1, 0.90: 0.2, 0.95: 0.15}, split_role="TRAIN",
        train_split_ref="train", training_evidence_ref="qe:tail:train", objective_id="stability",
    )
    store.freeze_supervised_parameter("campaign", first)
    changed = fit_supervised_parameter(
        parent_factor_id="tail", repair_family="TAIL_SATURATION", parameter_name="cutoff",
        candidate_grid=TAIL_CUTOFF_GRID,
        train_scores={0.85: 0.3, 0.90: 0.2, 0.95: 0.1}, split_role="TRAIN",
        train_split_ref="other", training_evidence_ref="qe:test-leak", objective_id="stability",
    )
    with pytest.raises(CampaignStateError, match="already frozen"):
        store.freeze_supervised_parameter("campaign", changed)

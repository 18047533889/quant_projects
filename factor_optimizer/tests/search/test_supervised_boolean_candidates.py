import pytest

from factor_optimizer.search.supervised_parameter import FrozenSupervisedParameter, fit_supervised_parameter


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("location", ["value", "frozen_grid", "fit_grid", "score_key"])
def test_boolean_candidate_is_not_numeric_evidence(flag, location):
    value = float(flag)
    with pytest.raises((TypeError, ValueError)):
        if location in ("value", "frozen_grid"):
            FrozenSupervisedParameter("p", "U_SHAPE_REPAIR", "center",
                flag if location == "value" else value,
                (flag,) if location == "frozen_grid" else (value,),
                "train", "evidence", "objective")
        else:
            fit_supervised_parameter(parent_factor_id="p", repair_family="U_SHAPE_REPAIR",
                parameter_name="center", candidate_grid=(flag,) if location == "fit_grid" else (value,),
                train_scores={flag if location == "score_key" else value: 1.0}, split_role="TRAIN",
                train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")


def test_numeric_candidates_keep_selection_and_hash_roundtrip():
    result = fit_supervised_parameter(parent_factor_id="p", repair_family="U_SHAPE_REPAIR",
        parameter_name="center", candidate_grid=(0, 0.5, 1), train_scores={0: 1.0, 0.5: 2.0, 1: 1.0},
        split_role="TRAIN", train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")
    assert result.value == 0.5
    assert FrozenSupervisedParameter(**result.__dict__) == result

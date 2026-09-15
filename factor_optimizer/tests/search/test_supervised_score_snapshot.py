from collections.abc import Mapping
import pytest

from factor_optimizer.search.supervised_parameter import fit_supervised_parameter


def fit(scores):
    return fit_supervised_parameter(parent_factor_id="p", repair_family="U_SHAPE_REPAIR",
        parameter_name="center", candidate_grid=(0.4, 0.5), train_scores=scores,
        split_role="TRAIN", train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")


class ChangingScores(Mapping):
    def __init__(self):
        self.reads = {0.4: 0, 0.5: 0}

    def __iter__(self):
        return iter(self.reads)

    def __len__(self):
        return 2

    def __getitem__(self, key):
        self.reads[key] += 1
        return ({0.4: 3.0, 0.5: 1.0} if self.reads[key] == 1 else {0.4: -100.0, 0.5: 100.0})[key]


def test_selection_uses_the_values_that_were_validated():
    scores = ChangingScores()
    assert fit(scores).value == 0.4
    assert scores.reads == {0.4: 1, 0.5: 1}


class InconsistentScores(ChangingScores):
    def values(self):
        return [1.0, 1.0]

    def __getitem__(self, key):
        return float("nan")


def test_actual_selected_score_values_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        fit(InconsistentScores())

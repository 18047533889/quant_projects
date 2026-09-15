import pytest

from factor_optimizer.search.supervised_parameter import fit_supervised_parameter


class ObservedScores(dict):
    reads = 0

    def __iter__(self):
        self.reads += 1
        return super().__iter__()


@pytest.mark.parametrize("grid", [(), (0.5, 0.5), (float("inf"),), (float("nan"),)])
@pytest.mark.parametrize("family", ["U_SHAPE_REPAIR", "TAIL_HINGE"])
def test_invalid_grid_rejected_before_score_access(grid, family):
    scores = ObservedScores({v: 1.0 for v in grid})
    with pytest.raises(ValueError):
        fit_supervised_parameter(parent_factor_id="p", repair_family=family,
            parameter_name="center", candidate_grid=grid, train_scores=scores,
            split_role="TRAIN", train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")
    assert scores.reads == 0


def test_valid_grid_reads_scores_and_keeps_tie_break():
    scores = ObservedScores({0.4: 1.0, 0.5: 1.0})
    result = fit_supervised_parameter(parent_factor_id="p", repair_family="U_SHAPE_REPAIR",
        parameter_name="center", candidate_grid=(0.4, 0.5), train_scores=scores,
        split_role="TRAIN", train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")
    assert scores.reads > 0
    assert result.value == 0.5

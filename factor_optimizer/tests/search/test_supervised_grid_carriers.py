import pytest

from factor_optimizer.search.supervised_parameter import FrozenSupervisedParameter, fit_supervised_parameter


@pytest.mark.parametrize("grid", ["01", b"01", bytearray(b"01"), {0.0: "ignored", 1.0: "ignored"}])
@pytest.mark.parametrize("operation", ["freeze", "fit"])
def test_ambiguous_grid_carrier_rejected(grid, operation):
    # Bytes would otherwise become ASCII codes, not the intended numeric grid.
    numeric = (48.0, 49.0) if isinstance(grid, (bytes, bytearray)) else (0.0, 1.0)
    with pytest.raises((TypeError, ValueError)):
        if operation == "freeze":
            FrozenSupervisedParameter("p", "U_SHAPE_REPAIR", "center", numeric[0],
                grid, "train", "evidence", "objective")
        else:
            fit_supervised_parameter(parent_factor_id="p", repair_family="U_SHAPE_REPAIR",
                parameter_name="center", candidate_grid=grid, train_scores={v: 1.0 for v in numeric},
                split_role="TRAIN", train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")


@pytest.mark.parametrize("carrier", [list, tuple, iter])
def test_numeric_iterable_still_supported(carrier):
    result = fit_supervised_parameter(parent_factor_id="p", repair_family="U_SHAPE_REPAIR",
        parameter_name="center", candidate_grid=carrier([0.4, 0.5]), train_scores={0.4: 1.0, 0.5: 2.0},
        split_role="TRAIN", train_split_ref="train", training_evidence_ref="evidence", objective_id="objective")
    assert result.value == 0.5
    assert result.candidate_grid == (0.4, 0.5)

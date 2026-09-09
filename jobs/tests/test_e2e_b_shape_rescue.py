import pytest

from jobs.e2e_b_shape_rescue import run_e2e_b_shape_rescue


@pytest.mark.parametrize("kind,branch", [
    ("u", "U_SHAPE_RESCUE"),
    ("inverted_u", "INVERTED_U_RESCUE"),
])
def test_low_linear_ic_shape_survives_and_parent_children_are_oos_reevaluated(kind, branch):
    result = run_e2e_b_shape_rescue(kind)
    parent_id = f"parent|{result.parent_dsl}"
    assert abs(result.parent_train_evaluation.get_metric("rank_ic", parent_id).value) < .15
    assert result.policy_branch == branch
    assert result.frozen_center.value in result.frozen_center.candidate_grid
    assert result.frozen_center.train_split_ref.startswith("split:")
    assert set(result.oos_evaluations) == {"parent", "left", "right", "recombined"}
    recombined = result.oos_evaluations["recombined"]
    rid = f"recombined|{result.child_dsls['recombined']}"
    assert recombined.get_metric("rank_ic", rid).value > .9
    assert recombined.artifacts["quantile_returns_daily"].values.shape == (20, 10, 1)
    assert result.trace["oos_role"] == "EVALUATION_ONLY"


def test_oos_labels_cannot_change_train_frozen_center():
    normal = run_e2e_b_shape_rescue("u")
    poisoned = run_e2e_b_shape_rescue("u", poison_oos=True)
    assert normal.frozen_center.state_hash == poisoned.frozen_center.state_hash
    assert normal.train_scores == poisoned.train_scores
    normal_ic = normal.oos_evaluations["recombined"].get_metric(
        "rank_ic", f"recombined|{normal.child_dsls['recombined']}"
    ).value
    poisoned_ic = poisoned.oos_evaluations["recombined"].get_metric(
        "rank_ic", f"recombined|{poisoned.child_dsls['recombined']}"
    ).value
    assert normal_ic == pytest.approx(-poisoned_ic, abs=.03)


def test_monotonic_parent_uses_distinct_policy_and_no_center_search():
    result = run_e2e_b_shape_rescue("monotonic")
    assert result.policy_branch == "MONOTONIC_NO_SHAPE_REPAIR"
    assert result.frozen_center is None
    assert result.train_scores == {}
    assert result.child_dsls == {}
    assert set(result.oos_evaluations) == {"parent"}


def test_branch_is_derived_from_public_train_shape_evidence():
    result = run_e2e_b_shape_rescue("u")
    parent_id = f"parent|{result.parent_dsl}"
    public = result.parent_train_evaluation
    u_score = public.get_metric("u_shape_score", parent_id).value
    inv_score = public.get_metric("inverted_u_score", parent_id).value
    mono = public.get_metric("quantile_monotonicity", parent_id).value
    curvature = public.get_metric("quantile_curvature", parent_id).value
    assert result.policy_branch == (
        "MONOTONIC_NO_SHAPE_REPAIR" if mono >= max(u_score, inv_score)
        else "U_SHAPE_RESCUE" if curvature > 0 else "INVERTED_U_RESCUE"
    )

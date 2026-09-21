"""Regression cases for advertised repair/search functionality."""
import pytest

from factor_optimizer.search.conditional_search import build_conditional_tree
from factor_optimizer.search.supervised_parameter import fit_supervised_parameter
from factor_optimizer.search.tiered_evaluation import EvaluationTier


def test_sma_is_a_valid_causal_smoothing_proposal():
    family = build_conditional_tree(["CAUSAL_SMOOTHING"])[0]
    family.validate_params({"method": "SMA", "natural_time_scale_relative": 0.5})


def test_inverted_u_center_can_be_frozen_from_train_evidence():
    frozen = fit_supervised_parameter(parent_factor_id="p", repair_family="INVERTED_U_REPAIR",
        parameter_name="center", candidate_grid=[0.4, 0.5, 0.6], train_scores={0.4: 1., 0.5: 1., 0.6: 1.},
        split_role="TRAIN", train_split_ref="train:1", training_evidence_ref="scores:1", objective_id="rank_ic")
    assert frozen.recipe_parameters(split_role="TEST") == {"center": 0.5}


def test_nonfinite_funnel_cost_cannot_be_scheduled():
    with pytest.raises(ValueError):
        EvaluationTier("full", float("inf"), 4)

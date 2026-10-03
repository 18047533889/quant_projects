from __future__ import annotations

import json

import pytest

from factor_optimizer.candidate_catalog import optimizer_candidate_specs
from factor_optimizer.research_batch import BatchOptimizationConfig


def _key(item):
    family, parameters = item
    return family, json.dumps(parameters, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("family", ["U_SHAPE_REPAIR", "INVERTED_U_REPAIR"])
def test_shape_grid_keeps_existing_order_and_appends_asymmetric_cases(family):
    specs = optimizer_candidate_specs(BatchOptimizationConfig(
        families=(family,), maximum_candidates=128))
    parameters = [params for name, params in specs if name == family]
    expected_symmetric = [
        {"center": center, "power": power, "asymmetry": False}
        for center in (.35, .5, .65) for power in (1., 2.)
    ]
    expected_asymmetric = [
        {"center": center, "power": power, "asymmetry": True}
        for center in (.35, .5, .65) for power in (1., 2.)
    ]
    assert parameters == expected_symmetric + expected_asymmetric


def test_robust_scale_catalog_covers_six_branches_with_prior_first():
    specs = optimizer_candidate_specs(BatchOptimizationConfig(
        families=("ROBUST_SCALE",), maximum_candidates=128))
    parameters = [params for _, params in specs]
    assert parameters == [
        {"scale": "mad", "center": "median"},
        {"scale": "mad", "center": "mean"},
        {"scale": "iqr", "center": "median"},
        {"scale": "iqr", "center": "mean"},
        {"scale": "std", "center": "median"},
        {"scale": "std", "center": "mean"},
    ]


def test_representation_rank_appends_cs_min_without_reordering_old_grid():
    specs = optimizer_candidate_specs(BatchOptimizationConfig(
        families=("REPRESENTATION_RANK",), maximum_candidates=128))
    parameters = [params for _, params in specs]
    old_grid = [{"rank_axis": "cross_sectional", "tie_method": "average", "window": 20}]
    old_grid += [
        {"rank_axis": "ts", "tie_method": tie, "window": window}
        for tie in ("average", "min") for window in (5, 10, 20)
    ]
    assert parameters[:-1] == old_grid
    assert parameters[-1] == {
        "rank_axis": "cross_sectional", "tie_method": "min", "window": 20}


def test_expanded_catalog_is_unique_filtered_and_within_default_static_budget():
    config = BatchOptimizationConfig(maximum_candidates=128)
    specs = optimizer_candidate_specs(config)
    assert len(specs) == 64
    assert len({_key(item) for item in specs}) == len(specs)
    for family in {family for family, _ in specs}:
        filtered = optimizer_candidate_specs(BatchOptimizationConfig(
            families=(family,), maximum_candidates=128))
        assert all(candidate_family == family for candidate_family, _ in filtered)
        assert filtered == [item for item in specs if item[0] == family]


def test_expanded_declared_static_budget_fails_closed_before_selection():
    # The 64 static specs plus four smoothing/decay sign companions require 68
    # catalog slots. A smaller declared limit fails before research execution.
    with pytest.raises(ValueError, match="candidate budget"):
        optimizer_candidate_specs(BatchOptimizationConfig(maximum_candidates=67))
    assert len(optimizer_candidate_specs(BatchOptimizationConfig(
        maximum_candidates=68))) == 64

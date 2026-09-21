"""Admissible smoothing grids and immutable evidence identity."""

from dataclasses import FrozenInstanceError

import pytest

from factor_optimizer.adapters.preprocessing import (
    compile_admissible_smoothing_grid,
    compile_smoothing_repair,
)


def test_default_grid_covers_all_five_methods_with_only_fp_admissible_plans():
    plans = compile_admissible_smoothing_grid(
        natural_time_scale=20.0,
        training_context_ref="train:grid-v1",
    )
    transforms = {plan.transform for plan in plans}
    assert transforms == {
        "trailing_sma", "ewma", "one_sided_iir_lowpass", "kama", "kalman_local_level",
    }
    assert len({plan.identity for plan in plans}) == len(plans)
    for plan in plans:
        assert plan.family == "CAUSAL_SMOOTHING"
        assert plan.training_context_ref == "train:grid-v1"


def test_grid_filters_by_both_fo_relative_domain_and_fp_method_domain():
    plans = compile_admissible_smoothing_grid(
        natural_time_scale=100.0,
        training_context_ref="train:large-scale",
        target_time_scales=(3.0, 10.0, 13.0, 20.0, 60.0),
    )
    by_transform = {}
    for plan in plans:
        by_transform.setdefault(plan.transform, []).append(dict(plan.parameters))
    assert [p["alpha"] for p in by_transform["one_sided_iir_lowpass"]] == pytest.approx(
        [1.0 - 2.0 ** (-1.0 / 10.0), 1.0 - 2.0 ** (-1.0 / 13.0)]
    )
    assert [p["halflife"] for p in by_transform["ewma"]] == [10.0, 13.0, 20.0, 60.0]
    assert [p["window"] for p in by_transform["trailing_sma"]] == [10, 13, 20, 60]


def test_plan_identity_is_stable_evidence_bound_and_plan_is_frozen():
    kwargs = dict(
        family="CAUSAL_SMOOTHING",
        parameters={"method": "EWMA", "natural_time_scale_relative": 0.5},
        natural_time_scale=20.0,
    )
    left = compile_smoothing_repair(**kwargs, training_context_ref="train:A")
    same = compile_smoothing_repair(**kwargs, training_context_ref="train:A")
    other_evidence = compile_smoothing_repair(**kwargs, training_context_ref="train:B")
    assert left.identity == same.identity
    assert left.identity != other_evidence.identity
    with pytest.raises(FrozenInstanceError):
        left.training_context_ref = "train:forged"

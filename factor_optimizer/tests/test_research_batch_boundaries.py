"""Adversarial data-contract tests for the research-only batch optimizer."""

from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

from factor_optimizer.research_batch import (
    BatchOptimizationConfig,
    automatic_time_split,
    optimize_factor_batch,
)


def _fixture(*, times=240, assets=24):
    rng = np.random.default_rng(912)
    signal = np.stack([
        rng.permutation(np.linspace(-1.0, 1.0, assets)) for _ in range(times)
    ])
    labels_array = signal ** 2
    factors = np.stack((-labels_array, signal), axis=-1)
    time_axis = AxisRef("time", "int", times, np.arange(times))
    asset_axis = AxisRef(
        "asset", "str", assets, np.array([f"a{i}" for i in range(assets)])
    )
    batch = FactorBatch(("reverse", "shape"), time_axis, asset_axis, factors)
    labels = LabelBundle(
        "forward",
        labels_array,
        1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)),
        asset_axis=asset_axis,
    )
    return batch, labels


def _config(**overrides):
    values = dict(
        families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR"),
        bootstrap_draws=99,
    )
    values.update(overrides)
    return BatchOptimizationConfig(selection_objective='rank_ic', **values)


def test_long_forward_labels_are_purged_by_actual_end_time_not_nominal_horizon():
    _, labels = _fixture()
    ends = list(labels.label_end_time)
    # Nominal horizon remains one, but the final five TRAIN labels mature in
    # VALIDATION and the final five VALIDATION labels mature in TEST.
    for i in range(139, 144):
        ends[i] = 150 + (i - 139)
    for i in range(187, 192):
        ends[i] = 200 + (i - 187)
    attacked = replace(labels, label_end_time=tuple(ends))

    split = automatic_time_split(attacked, _config())

    assert set(range(139, 144)).isdisjoint(split.train_indices)
    assert set(range(187, 192)).isdisjoint(split.validation_indices)
    assert max(attacked.label_end_time[i] for i in split.train_indices) < 143
    assert max(attacked.label_end_time[i] for i in split.validation_indices) < 191


def test_validation_features_cannot_change_any_train_candidate_score():
    batch, labels = _fixture()
    config = _config()
    baseline = optimize_factor_batch(
        batch, labels, config=config, allow_research=True
    )
    values = np.array(batch.values, copy=True)
    values[144:192] = 1e12
    poisoned = replace(batch, values=values)

    attacked = optimize_factor_batch(
        poisoned, labels, config=config, allow_research=True
    )

    for factor_id in batch.factor_ids:
        left = baseline.factors[factor_id]
        right = attacked.factors[factor_id]
        left_scores = [
            (x["family"], x["parameters"], x.get("train_gain"))
            for x in left.candidates
        ]
        right_scores = [
            (x["family"], x["parameters"], x.get("train_gain"))
            for x in right.candidates
        ]
        assert left_scores == right_scores


def test_test_label_values_validity_and_windows_do_not_pollute_research_result():
    batch, labels = _fixture()
    validity = np.ones_like(labels.values, dtype=bool)
    baseline_labels = replace(labels, validity=validity)
    baseline = optimize_factor_batch(
        batch, baseline_labels, config=_config(), allow_research=True
    )

    values = np.array(labels.values, copy=True)
    values[192:] = np.nan
    changed_validity = validity.copy()
    changed_validity[192:] = False
    ends = list(labels.label_end_time)
    ends[192:] = range(1000, 1048)
    attacked_labels = replace(
        labels,
        values=values,
        validity=changed_validity,
        label_end_time=tuple(ends),
    )
    attacked = optimize_factor_batch(
        batch, attacked_labels, config=_config(), allow_research=True
    )

    assert baseline.split.identity == attacked.split.identity
    for factor_id in batch.factor_ids:
        left = baseline.factors[factor_id]
        right = attacked.factors[factor_id]
        assert left.plan_identity == right.plan_identity
        assert left.train_gain == right.train_gain
        assert left.validation_lower_bound == right.validation_lower_bound
        assert left.status == right.status
    assert baseline.test_evaluated is False
    assert attacked.test_evaluated is False
    assert baseline.execution_mode == attacked.execution_mode == "research_only"


def test_factor_validity_false_is_equivalent_to_missing_raw_value():
    batch, labels = _fixture()
    validity = np.ones_like(batch.values, dtype=bool)
    validity[20:35, :4, 0] = False
    masked = replace(batch, validity=validity)
    missing_values = np.array(batch.values, copy=True)
    missing_values[20:35, :4, 0] = np.nan
    missing = replace(batch, values=missing_values)

    left = optimize_factor_batch(masked, labels, config=_config(), allow_research=True)
    right = optimize_factor_batch(missing, labels, config=_config(), allow_research=True)

    assert left.factors["reverse"].status == right.factors["reverse"].status
    assert left.factors["reverse"].train_gain == right.factors["reverse"].train_gain
    np.testing.assert_allclose(
        left.optimized.values[:, :, 0],
        right.optimized.values[:, :, 0],
        equal_nan=True,
    )


@pytest.mark.parametrize("families", ["SIGN_ORIENTATION", (True,), ("",)])
def test_family_configuration_rejects_non_tuple_or_non_name_members(families):
    with pytest.raises(ValueError, match="families"):
        BatchOptimizationConfig(selection_objective='rank_ic', families=families)


@pytest.mark.parametrize("field", [
    "train_fraction",
    "validation_fraction",
    "minimum_coverage",
    "confidence_level",
    "minimum_improvement",
    "natural_time_scale",
])
def test_boolean_and_nan_numeric_configuration_is_rejected(field):
    with pytest.raises((TypeError, ValueError)):
        BatchOptimizationConfig(selection_objective='rank_ic', **{field: True})
    with pytest.raises((TypeError, ValueError)):
        BatchOptimizationConfig(selection_objective='rank_ic', **{field: float("nan")})


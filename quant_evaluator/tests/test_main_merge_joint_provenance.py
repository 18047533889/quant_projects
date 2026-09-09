"""Actual QE producer pairing identities, independent of FA test envelopes."""
from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.resampling import ResamplingPlan
from quant_evaluator.metrics.robustness import compute_joint_block_bootstrap


PAIR_FIELDS = ("resampling_plan_content_hash", "sample_identity_hash",
               "time_identity_hash", "common_mask_hash")


def test_actual_draws_and_common_grid_are_named_and_permutation_invariant():
    plan = ResamplingPlan(tuple(range(12)), "synthetic_daily", 3, 8, 17)
    values = np.column_stack((np.arange(12.), np.arange(12.) ** 2))
    batch = compute_joint_block_bootstrap(values, plan, ("a", "b"))
    single = compute_joint_block_bootstrap(values[:, 1:], plan, ("b",))
    permuted = compute_joint_block_bootstrap(values[:, ::-1], plan, ("b", "a"))
    for key in PAIR_FIELDS:
        assert len(batch.provenance[key]) == 64
        assert batch.provenance[key] == single.provenance[key] == permuted.provenance[key]
    np.testing.assert_array_equal(batch.samples[:, 1], single.samples[:, 0])
    np.testing.assert_array_equal(batch.samples[:, ::-1], permuted.samples)
    expected = values[plan.indices()].mean(axis=1)
    np.testing.assert_array_equal(batch.samples, expected)


@pytest.mark.parametrize("change", [
    {"seed": 18}, {"block_length": 2}, {"clock_ref": "other_clock"},
    {"time_ids": tuple(range(1, 13))},
    {"segment_ids": (0,) * 6 + (1,) * 6},
])
def test_changed_real_plan_or_time_cannot_keep_pairing_identity(change):
    plan = ResamplingPlan(tuple(range(12)), "synthetic_daily", 3, 8, 17)
    values = np.arange(12.)[:, None]
    original = compute_joint_block_bootstrap(values, plan, ("a",))
    changed = compute_joint_block_bootstrap(values, replace(plan, **change), ("a",))
    assert original.provenance["resampling_plan_content_hash"] != changed.provenance["resampling_plan_content_hash"]
    assert original.provenance["sample_identity_hash"] != changed.provenance["sample_identity_hash"]


def test_missing_column_never_obtains_usable_complete_grid_samples():
    plan = ResamplingPlan(tuple(range(12)), "synthetic_daily", 3, 8, 17)
    valid = np.arange(12.)[:, None]
    incomplete = valid.copy()
    incomplete[3] = np.nan
    good = compute_joint_block_bootstrap(valid, plan, ("good",))
    bad = compute_joint_block_bootstrap(incomplete, plan, ("bad",))
    mixed = compute_joint_block_bootstrap(np.column_stack((valid, incomplete)), plan, ("good", "bad"))
    assert np.isnan(bad.samples).all() and np.isnan(mixed.samples[:, 1]).all()
    assert bad.provenance["common_mask_hash"] != good.provenance["common_mask_hash"]
    assert mixed.provenance["common_mask_hash"] == good.provenance["common_mask_hash"]
    np.testing.assert_array_equal(mixed.samples[:, 0], good.samples[:, 0])

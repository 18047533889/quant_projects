"""Independent small-panel checks for interaction metrics and axis contracts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.interactions.complementarity import (
    compute_complementarity_score,
    compute_interaction_strength,
)
from quant_evaluator.metrics.interactions.conditional import (
    compute_conditional_ic,
    compute_incremental_ic,
    compute_partial_ic,
)
from quant_evaluator.metrics.interactions.pairwise import (
    compute_pairwise_artifacts,
    compute_pairwise_correlation,
    compute_rolling_pairwise_correlation,
)
from quant_evaluator.metrics.interactions.substitution import (
    compute_marginal_contribution,
    compute_substitution_effect,
)
from quant_evaluator.runtime.evaluator import evaluate


def _bundles(values, labels, *, validity=None, label_validity=None, assets=None):
    t, n, f = values.shape
    if assets is None:
        assets = tuple(f"s{i}" for i in range(n))
    asset_axis = AxisRef("asset", "str", n, assets)
    time_axis = AxisRef("time", "int", t, np.arange(t))
    fb = FactorBatch(
        tuple(f"f{i}" for i in range(f)), time_axis, asset_axis,
        np.ascontiguousarray(values), validity=validity,
    )
    lb = LabelBundle(
        "y", np.ascontiguousarray(labels), 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
        asset_axis=asset_axis, validity=label_validity,
    )
    return fb, lb


def _sample():
    rng = np.random.default_rng(20260926)
    t, n = 9, 48
    a = rng.normal(size=(t, n))
    b = 0.35 * a + rng.normal(size=(t, n))
    c = rng.normal(size=(t, n))
    values = np.stack((a, b, c), axis=2)
    labels = 0.4 * a + 0.3 * b + rng.normal(scale=0.7, size=(t, n))
    values[rng.random(values.shape) < 0.06] = np.nan
    labels[rng.random(labels.shape) < 0.05] = np.nan
    labels[1] = np.nan
    validity = rng.random(values.shape) > 0.04
    label_validity = rng.random(labels.shape) > 0.04
    return _bundles(values, labels, validity=validity, label_validity=label_validity)


def _corr(x, y, method="pearson"):
    if len(x) < 2 or np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return np.nan
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    return float(np.corrcoef(x, y)[0, 1])


def _joint(fb, lb, t, i, j):
    values = fb.values[t]
    labels = lb.values[t]
    mask = np.isfinite(values[:, i]) & np.isfinite(values[:, j]) & np.isfinite(labels)
    if fb.validity is not None:
        mask &= fb.validity[t, :, i] & fb.validity[t, :, j]
    if lb.validity is not None:
        mask &= lb.validity[t]
    return values[mask, i], values[mask, j], labels[mask]


@pytest.mark.parametrize("method", ("pearson", "spearman"))
def test_substitution_and_complementarity_against_joint_sample_oracle(method):
    fb, lb = _sample()
    got = compute_substitution_effect(fb, lb, 0, 1, method=method, min_assets=8)
    expected = np.full((3, fb.num_times), np.nan)
    for t in range(fb.num_times):
        a, b, y = _joint(fb, lb, t, 0, 1)
        if len(a) < 8:
            continue
        expected[0, t] = _corr(a, y, method)
        expected[1, t] = _corr(b, y, method)
        za = (a - a.mean()) / a.std()
        zb = (b - b.mean()) / b.std()
        expected[2, t] = _corr((za + zb) / 2, y, method)
    for actual, oracle in zip(got, expected):
        np.testing.assert_allclose(actual, oracle, rtol=0, atol=1e-12, equal_nan=True)

    valid = np.all(np.isfinite(expected), axis=0)
    lift = np.abs(expected[2, valid]) - np.maximum(
        np.abs(expected[0, valid]), np.abs(expected[1, valid])
    )
    score, mean_lift, std_lift = compute_complementarity_score(
        fb, lb, 0, 1, method=method, min_assets=8, min_periods=2,
    )
    assert mean_lift == pytest.approx(float(lift.mean()), abs=1e-12)
    assert std_lift == pytest.approx(float(lift.std(ddof=1)), abs=1e-12)
    assert score == pytest.approx(float(lift.mean() / lift.std(ddof=1)), abs=1e-12)
    assert np.isnan(got[0][1]) and np.isnan(got[1][1]) and np.isnan(got[2][1])


def test_conditional_and_interaction_against_direct_numpy_oracle():
    fb, lb = _sample()
    conditional, counts = compute_conditional_ic(
        fb, lb, 0, quantiles=3, min_assets=5,
    )
    expected = np.full_like(conditional, np.nan)
    expected_counts = np.zeros_like(counts)
    interaction, additive = compute_interaction_strength(
        fb, lb, 0, 1, min_assets=8,
    )
    expected_interaction = np.full(fb.num_times, np.nan)
    expected_additive = np.full(fb.num_times, np.nan)
    for t in range(fb.num_times):
        conditioning = fb.values[t, :, 0].copy()
        conditioning[~fb.validity[t, :, 0]] = np.nan
        finite = np.isfinite(conditioning)
        if finite.sum() >= 5:
            edges = np.percentile(conditioning[finite], np.linspace(0, 100, 4))
            groups = np.digitize(conditioning, edges[1:-1])
            groups[~finite] = -1
            for q in range(3):
                member = groups == q
                if member.sum() < 5:
                    continue
                for f in range(fb.num_factors):
                    mask = (member & np.isfinite(fb.values[t, :, f])
                            & np.isfinite(lb.values[t])
                            & fb.validity[t, :, f] & lb.validity[t])
                    expected_counts[t, f, q] = mask.sum()
                    if mask.sum() >= 5:
                        expected[t, f, q] = _corr(
                            fb.values[t, mask, f], lb.values[t, mask],
                        )
        a, b, y = _joint(fb, lb, t, 0, 1)
        if len(a) >= 8:
            za = (a - a.mean()) / (a.std() + 1e-8)
            zb = (b - b.mean()) / (b.std() + 1e-8)
            expected_interaction[t] = _corr(za * zb, y)
            expected_additive[t] = _corr((za + zb) / 2, y)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(conditional, expected, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(interaction, expected_interaction, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(additive, expected_additive, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("method", ("pearson", "spearman"))
def test_pairwise_dense_sparse_and_rolling_against_joint_finite_oracle(method):
    fb, _ = _sample()
    dense = compute_pairwise_correlation(fb, method=method, min_obs=8)
    rolling = compute_rolling_pairwise_correlation(fb, window=3, method=method, min_obs=8)
    sparse = compute_pairwise_artifacts(
        fb, method=method, min_obs=8,
        candidate_pairs=((0, 1),), max_pairs=1,
        window_ref="all", universe_ref="u", sample_ref="s",
        windows={"early": (0, 3), "late": (3, fb.num_times)},
    )[0]
    for i in range(fb.num_factors):
        for j in range(fb.num_factors):
            mask = (np.isfinite(fb.values[:, :, i]) & np.isfinite(fb.values[:, :, j])
                    & fb.validity[:, :, i] & fb.validity[:, :, j])
            oracle = _corr(fb.values[:, :, i][mask], fb.values[:, :, j][mask], method)
            assert dense[i, j] == pytest.approx(oracle, abs=1e-12)
            assert dense[i, j] == pytest.approx(dense[j, i], abs=1e-12)
            for t in range(2, fb.num_times):
                wm = mask[t - 2:t + 1]
                x = fb.values[t - 2:t + 1, :, i][wm]
                y = fb.values[t - 2:t + 1, :, j][wm]
                assert rolling[t, i, j] == pytest.approx(_corr(x, y, method), abs=1e-12)
    assert sparse.pair_count == sum(sparse.daily_pair_counts)
    assert sparse.correlation == pytest.approx(dense[0, 1], abs=1e-12)
    for window in sparse.windows:
        start, end = window.start_index, window.end_index
        mask = (np.isfinite(fb.values[start:end, :, 0])
                & np.isfinite(fb.values[start:end, :, 1])
                & fb.validity[start:end, :, 0] & fb.validity[start:end, :, 1])
        oracle = _corr(
            fb.values[start:end, :, 0][mask],
            fb.values[start:end, :, 1][mask], method,
        )
        assert window.pair_count == int(mask.sum())
        assert window.correlation == pytest.approx(oracle, abs=1e-12)


def test_factor_permutation_duplicate_and_near_constant_semantics():
    fb, lb = _sample()
    order = (2, 0, 1)
    permuted = FactorBatch(
        tuple(fb.factor_ids[i] for i in order), fb.time_axis, fb.asset_axis,
        fb.values[:, :, order], validity=fb.validity[:, :, order],
    )
    original = compute_pairwise_correlation(fb, min_obs=8)
    changed = compute_pairwise_correlation(permuted, min_obs=8)
    np.testing.assert_allclose(changed, original[np.ix_(order, order)], atol=1e-12, equal_nan=True)
    ab = compute_substitution_effect(fb, lb, 0, 1, min_assets=8)
    ba = compute_substitution_effect(fb, lb, 1, 0, min_assets=8)
    np.testing.assert_allclose(ab[0], ba[1], atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ab[1], ba[0], atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ab[2], ba[2], atol=1e-12, equal_nan=True)
    permuted_ab = compute_substitution_effect(permuted, lb, 1, 2, min_assets=8)
    for x, y in zip(ab, permuted_ab):
        np.testing.assert_allclose(x, y, atol=1e-12, equal_nan=True)
    original_conditional = compute_conditional_ic(
        fb, lb, 0, quantiles=3, min_assets=5,
    )
    permuted_conditional = compute_conditional_ic(
        permuted, lb, 1, quantiles=3, min_assets=5,
    )
    np.testing.assert_allclose(
        permuted_conditional[0], original_conditional[0][:, order, :],
        atol=1e-12, equal_nan=True,
    )
    np.testing.assert_array_equal(
        permuted_conditional[1], original_conditional[1][:, order, :],
    )
    original_interaction = compute_interaction_strength(fb, lb, 0, 1, min_assets=8)
    permuted_interaction = compute_interaction_strength(
        permuted, lb, 1, 2, min_assets=8,
    )
    for x, y in zip(original_interaction, permuted_interaction):
        np.testing.assert_allclose(x, y, atol=1e-12, equal_nan=True)
    original_complementarity = compute_complementarity_score(
        fb, lb, 0, 1, min_assets=8, min_periods=2,
    )
    permuted_complementarity = compute_complementarity_score(
        permuted, lb, 1, 2, min_assets=8, min_periods=2,
    )
    np.testing.assert_allclose(
        original_complementarity, permuted_complementarity, atol=1e-12,
    )

    rng = np.random.default_rng(17)
    signal = rng.normal(size=(5, 40))
    duplicate, labels = _bundles(
        np.stack((signal, signal), axis=2), signal.copy(),
    )
    a, b, combined = compute_substitution_effect(duplicate, labels, 0, 1, min_assets=8)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(a, combined, atol=1e-12)
    assert compute_pairwise_correlation(duplicate, min_obs=8)[0, 1] == pytest.approx(1.0)
    assert compute_complementarity_score(
        duplicate, labels, 0, 1, min_assets=8, min_periods=2,
    )[1] == pytest.approx(0.0, abs=1e-12)
    constant = FactorBatch(
        duplicate.factor_ids, duplicate.time_axis, duplicate.asset_axis,
        np.stack((signal, np.full_like(signal, 1e-18)), axis=2),
    )
    _, weak, combined_weak = compute_substitution_effect(
        constant, labels, 0, 1, min_assets=8,
    )
    assert np.isnan(weak).all() and np.isnan(combined_weak).all()
    assert np.isnan(compute_pairwise_correlation(constant, min_obs=8)[1, 1])
    assert np.isnan(
        compute_rolling_pairwise_correlation(
            constant, window=2, min_obs=8,
        )[-1, 1, 1]
    )
    sparse_constant = compute_pairwise_artifacts(
        constant, min_obs=8, candidate_pairs=((0, 1),), max_pairs=1,
        window_ref="w", universe_ref="u", sample_ref="s",
    )[0]
    assert sparse_constant.correlation is None
    assert sparse_constant.status.value == "constant_input"


def test_explicit_asset_order_mismatch_fails_at_low_level_and_public_entry():
    t, n = 25, 40
    signal = np.tile(np.arange(n, dtype=float), (t, 1))
    fb, lb = _bundles(np.stack((signal, 2 * signal + 1), axis=2), signal)
    reversed_labels = LabelBundle(
        "y", signal[:, ::-1].copy(), 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
        asset_axis=AxisRef("asset", "str", n, tuple(reversed(lb.asset_axis.values))),
    )
    calls = (
        lambda: compute_substitution_effect(fb, reversed_labels, 0, 1, min_assets=8),
        lambda: compute_marginal_contribution(fb, reversed_labels, 1, (0,), min_assets=8),
        lambda: compute_conditional_ic(fb, reversed_labels, 0, quantiles=2, min_assets=5),
        lambda: compute_incremental_ic(fb, reversed_labels, (0,), 1, min_assets=8),
        lambda: compute_partial_ic(fb, reversed_labels, 1, (0,), min_assets=8),
        lambda: compute_complementarity_score(fb, reversed_labels, 0, 1, min_assets=8, min_periods=2),
        lambda: compute_interaction_strength(fb, reversed_labels, 0, 1, min_assets=8),
        lambda: evaluate(fb, reversed_labels, metrics=("rank_ic",)),
    )
    for call in calls:
        with pytest.raises(InvalidContractError, match="asset coordinates"):
            call()

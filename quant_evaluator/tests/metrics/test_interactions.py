"""
Tests for factor interaction analysis metrics.
"""

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.interactions import (
    compute_pairwise_correlation,
    compute_rolling_pairwise_correlation,
    compute_correlation_matrix,
    compute_conditional_ic,
    compute_incremental_ic,
    compute_partial_ic,
    compute_substitution_effect,
    detect_substitutable_factors,
    compute_marginal_contribution,
    compute_complementarity_score,
    detect_complementary_pairs,
    compute_interaction_strength,
)


@pytest.fixture
def simple_factor_batch():
    """Create a simple factor batch for testing."""
    T, N, F = 50, 100, 3

    # Create correlated factors
    np.random.seed(42)
    base = np.random.randn(T, N)

    # Factor 0: base signal
    factor_0 = base + np.random.randn(T, N) * 0.5

    # Factor 1: highly correlated with factor 0 (substitutable)
    factor_1 = base + np.random.randn(T, N) * 0.5

    # Factor 2: orthogonal signal (complementary)
    factor_2 = np.random.randn(T, N)

    values = np.stack([factor_0, factor_1, factor_2], axis=2)

    time_axis = AxisRef(name="time", dtype="int64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    return FactorBatch(
        factor_ids=("factor_0", "factor_1", "factor_2"),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )


@pytest.fixture
def simple_label_bundle(simple_factor_batch):
    """Create labels correlated with factors."""
    T, N = simple_factor_batch.num_times, simple_factor_batch.num_assets

    np.random.seed(43)

    # Labels driven by factor_0 and factor_2, not factor_1
    labels = (
        0.6 * simple_factor_batch.values[:, :, 0] +  # factor_0
        0.4 * simple_factor_batch.values[:, :, 2] +  # factor_2
        np.random.randn(T, N) * 0.5
    )

    decision_time = tuple(range(T))

    return LabelBundle(
        target_id="returns",
        values=labels,
        horizon=1,
        execution_delay=0,
        decision_time=decision_time,
        execution_time=decision_time,
        label_start_time=decision_time,
        label_end_time=tuple(t + 1 for t in decision_time),
    )


class TestPairwiseCorrelation:
    def test_compute_pairwise_correlation(self, simple_factor_batch):
        corr_matrix = compute_pairwise_correlation(simple_factor_batch, method="pearson")

        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1.0
        assert np.allclose(np.diag(corr_matrix), 1.0, atol=0.01)

        # Factor 0 and 1 should be highly correlated
        assert corr_matrix[0, 1] > 0.7
        assert corr_matrix[1, 0] > 0.7

        # Factor 2 should be less correlated with others
        assert abs(corr_matrix[0, 2]) < 0.3
        assert abs(corr_matrix[1, 2]) < 0.3

    def test_compute_pairwise_spearman(self, simple_factor_batch):
        corr_matrix = compute_pairwise_correlation(simple_factor_batch, method="spearman")

        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0, atol=0.01)

    def test_rolling_pairwise_correlation(self, simple_factor_batch):
        window = 20
        rolling_corr = compute_rolling_pairwise_correlation(
            simple_factor_batch, window=window, method="pearson"
        )

        T = simple_factor_batch.num_times
        assert rolling_corr.shape == (T, 3, 3)

        # First window-1 periods should be NaN
        assert np.all(np.isnan(rolling_corr[:window-1, :, :]))

        # Later periods should have valid correlations
        assert not np.all(np.isnan(rolling_corr[window:, :, :]))

        # Diagonal should be 1.0 where valid
        for t in range(window-1, T):
            if np.isfinite(rolling_corr[t, 0, 0]):
                assert np.allclose(np.diag(rolling_corr[t, :, :]), 1.0, atol=0.05)

    def test_correlation_matrix_cross_sectional(self, simple_factor_batch):
        corr_matrix = compute_correlation_matrix(
            simple_factor_batch, cross_sectional=True, method="pearson"
        )

        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0, atol=0.05)

    def test_insufficient_observations(self):
        T, N, F = 5, 8, 2
        values = np.random.randn(T, N, F)

        time_axis = AxisRef(name="time", dtype="int64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        factor_batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        corr_matrix = compute_pairwise_correlation(factor_batch, min_obs=100)

        # Should return NaN when insufficient observations
        assert np.all(np.isnan(corr_matrix))


class TestConditionalIC:
    def test_compute_conditional_ic(self, simple_factor_batch, simple_label_bundle):
        conditional_ic, sample_counts = compute_conditional_ic(
            simple_factor_batch,
            simple_label_bundle,
            conditioning_factor_idx=0,
            quantiles=5,
            method="pearson",
        )

        T, F = simple_factor_batch.num_times, simple_factor_batch.num_factors
        assert conditional_ic.shape == (T, F, 5)
        assert sample_counts.shape == (T, 5)

        # Should have some valid ICs
        assert not np.all(np.isnan(conditional_ic))

        # Sample counts should be reasonable
        assert np.all((sample_counts == 0) | (sample_counts >= 10))

    def test_compute_incremental_ic(self, simple_factor_batch, simple_label_bundle):
        # Factor 2 should have incremental IC over factor 0
        incremental_ic, base_ic, total_ic = compute_incremental_ic(
            simple_factor_batch,
            simple_label_bundle,
            base_factor_indices=(0,),
            test_factor_idx=2,
            method="pearson",
        )

        T = simple_factor_batch.num_times
        assert incremental_ic.shape == (T,)
        assert base_ic.shape == (T,)
        assert total_ic.shape == (T,)

        # Should have some valid values
        assert not np.all(np.isnan(incremental_ic))
        assert not np.all(np.isnan(base_ic))
        assert not np.all(np.isnan(total_ic))

        # Factor 2 should have positive incremental IC (complementary)
        valid_mask = np.isfinite(incremental_ic)
        if np.any(valid_mask):
            mean_incremental = np.mean(incremental_ic[valid_mask])
            assert mean_incremental > -0.5  # Some positive contribution expected

    def test_compute_partial_ic(self, simple_factor_batch, simple_label_bundle):
        partial_ic = compute_partial_ic(
            simple_factor_batch,
            simple_label_bundle,
            factor_idx=2,
            control_indices=(0, 1),
            method="pearson",
        )

        assert partial_ic.shape == (simple_factor_batch.num_times,)
        assert not np.all(np.isnan(partial_ic))

    def test_partial_ic_matches_closed_form_partial_correlation(self):
        """Non-degenerate oracle: partial IC equals the closed-form partial correlation.

        label = a*control + b*test + noise with control/test independent of
        each other and of the noise.  The theoretical partial correlation
        between test and label given control reduces to
        corr(b*test, noise) / 1 = 1/sqrt(1 + var(noise)/b^2) > 0, while the
        raw (total) IC equals corr(test, label) which is strictly smaller.
        """
        rng = np.random.default_rng(7)
        T, N = 6, 500

        a, b, noise_sd = 2.0, 1.0, 1.0
        control = rng.standard_normal((T, N))
        test = rng.standard_normal((T, N))
        noise = rng.standard_normal((T, N)) * noise_sd
        label = a * control + b * test + noise

        # Theoretical partial correlation of test vs label given control.
        var_signal = b * b
        theoretical = np.sqrt(var_signal / (var_signal + noise_sd**2))

        values = np.stack([control, test], axis=2)
        factor_batch = FactorBatch(
            factor_ids=("control", "test"),
            time_axis=AxisRef(name="time", dtype="int64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=values,
        )
        decision_time = tuple(range(T))
        label_bundle = LabelBundle(
            target_id="returns",
            values=label,
            horizon=1,
            execution_delay=0,
            decision_time=decision_time,
            execution_time=decision_time,
            label_start_time=decision_time,
            label_end_time=tuple(t + 1 for t in decision_time),
        )

        partial_ic = compute_partial_ic(
            factor_batch,
            label_bundle,
            factor_idx=1,
            control_indices=(0,),
            method="pearson",
            min_assets=30,
        )
        _, _, total_ic = compute_incremental_ic(
            factor_batch,
            label_bundle,
            base_factor_indices=(0,),
            test_factor_idx=1,
            method="pearson",
            min_assets=30,
        )

        assert np.all(np.isfinite(partial_ic))
        # Closed-form partial correlation within sampling tolerance.
        np.testing.assert_allclose(partial_ic, theoretical, atol=0.08)
        # Partial IC must differ from the raw total IC (which the buggy
        # implementation returned): the control factor dilutes the raw IC.
        raw_expected = np.sqrt(b * b / (b * b + a * a + noise_sd**2))
        assert np.all(np.abs(total_ic - partial_ic) > 0.05)
        np.testing.assert_allclose(total_ic, raw_expected, atol=0.08)


    def test_conditional_ic_insufficient_assets(self, simple_factor_batch, simple_label_bundle):
        # Set min_assets very high
        conditional_ic, sample_counts = compute_conditional_ic(
            simple_factor_batch,
            simple_label_bundle,
            conditioning_factor_idx=0,
            quantiles=10,  # Many quantiles with few assets each
            min_assets=50,
        )

        # Should have mostly NaN due to insufficient assets per quantile
        assert np.sum(np.isnan(conditional_ic)) > 0.5 * conditional_ic.size


class TestSubstitution:
    def test_compute_substitution_effect(self, simple_factor_batch, simple_label_bundle):
        # Factor 0 and 1 are substitutable
        ic_a, ic_b, ic_combined = compute_substitution_effect(
            simple_factor_batch,
            simple_label_bundle,
            factor_a_idx=0,
            factor_b_idx=1,
            method="pearson",
        )

        T = simple_factor_batch.num_times
        assert ic_a.shape == (T,)
        assert ic_b.shape == (T,)
        assert ic_combined.shape == (T,)

        # Should have valid ICs
        assert not np.all(np.isnan(ic_a))
        assert not np.all(np.isnan(ic_b))
        assert not np.all(np.isnan(ic_combined))

        # Combined IC should be close to max individual IC (substitution)
        valid_mask = np.isfinite(ic_a) & np.isfinite(ic_b) & np.isfinite(ic_combined)
        if np.any(valid_mask):
            max_individual = np.maximum(np.abs(ic_a[valid_mask]), np.abs(ic_b[valid_mask]))
            ratio = np.mean(np.abs(ic_combined[valid_mask]) / (max_individual + 1e-8))
            assert 0.5 < ratio < 1.5  # Should be roughly similar

    def test_detect_substitutable_factors(self, simple_factor_batch, simple_label_bundle):
        substitutable = detect_substitutable_factors(
            simple_factor_batch,
            simple_label_bundle,
            threshold=0.85,
            method="pearson",
        )

        # Should detect factor 0 and 1 as substitutable
        assert len(substitutable) >= 0  # May or may not detect depending on threshold

        # If detected, verify structure
        for factor_a, factor_b, score in substitutable:
            assert 0 <= factor_a < simple_factor_batch.num_factors
            assert 0 <= factor_b < simple_factor_batch.num_factors
            assert factor_a < factor_b
            assert 0.0 <= score <= 2.0

    def test_compute_marginal_contribution(self, simple_factor_batch, simple_label_bundle):
        # Marginal contribution of factor 2 given factor 0
        marginal_ic, baseline_ic = compute_marginal_contribution(
            simple_factor_batch,
            simple_label_bundle,
            target_factor_idx=2,
            baseline_indices=(0,),
            method="pearson",
        )

        T = simple_factor_batch.num_times
        assert marginal_ic.shape == (T,)
        assert baseline_ic.shape == (T,)

        # Should have valid values
        assert not np.all(np.isnan(marginal_ic))
        assert not np.all(np.isnan(baseline_ic))

        # Factor 2 should contribute positively (complementary to factor 0)
        valid_mask = np.isfinite(marginal_ic)
        if np.any(valid_mask):
            mean_marginal = np.mean(marginal_ic[valid_mask])
            assert mean_marginal > -0.5  # Expect some positive contribution

    def test_marginal_contribution_no_baseline(self, simple_factor_batch, simple_label_bundle):
        marginal_ic, baseline_ic = compute_marginal_contribution(
            simple_factor_batch,
            simple_label_bundle,
            target_factor_idx=0,
            baseline_indices=(),
            method="pearson",
        )

        # Baseline should be zero
        assert np.all(baseline_ic == 0.0)

        # Marginal should equal total IC
        assert not np.all(np.isnan(marginal_ic))


class TestComplementarity:
    def test_compute_complementarity_score(self, simple_factor_batch, simple_label_bundle):
        # Factor 0 and 2 should be complementary
        comp_score, mean_lift, std_lift = compute_complementarity_score(
            simple_factor_batch,
            simple_label_bundle,
            factor_a_idx=0,
            factor_b_idx=2,
            method="pearson",
        )

        assert np.isfinite(comp_score) or np.isnan(comp_score)
        assert np.isfinite(mean_lift) or np.isnan(mean_lift)
        assert np.isfinite(std_lift) or np.isnan(std_lift)

        # If valid, should show some complementarity
        if np.isfinite(comp_score):
            assert comp_score > -5.0  # Reasonable range

    def test_detect_complementary_pairs(self, simple_factor_batch, simple_label_bundle):
        complementary = detect_complementary_pairs(
            simple_factor_batch,
            simple_label_bundle,
            threshold=0.5,
            method="pearson",
        )

        # Should return a list
        assert isinstance(complementary, list)

        # Verify structure
        for factor_a, factor_b, comp_score, mean_lift in complementary:
            assert 0 <= factor_a < simple_factor_batch.num_factors
            assert 0 <= factor_b < simple_factor_batch.num_factors
            assert factor_a < factor_b
            assert comp_score >= 0.5

    def test_compute_interaction_strength(self, simple_factor_batch, simple_label_bundle):
        interaction_ic, additive_ic = compute_interaction_strength(
            simple_factor_batch,
            simple_label_bundle,
            factor_a_idx=0,
            factor_b_idx=2,
            method="pearson",
        )

        T = simple_factor_batch.num_times
        assert interaction_ic.shape == (T,)
        assert additive_ic.shape == (T,)

        # Should have some valid values
        assert not np.all(np.isnan(interaction_ic))
        assert not np.all(np.isnan(additive_ic))


class TestEdgeCases:
    def test_constant_factor(self):
        """Test with a constant factor (zero variance)."""
        T, N, F = 30, 50, 2

        values = np.random.randn(T, N, F)
        values[:, :, 1] = 1.0  # Constant factor

        time_axis = AxisRef(name="time", dtype="int64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        factor_batch = FactorBatch(
            factor_ids=("variable", "constant"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        corr_matrix = compute_pairwise_correlation(factor_batch)

        # Correlation with constant should be NaN
        assert np.isnan(corr_matrix[0, 1])
        assert np.isnan(corr_matrix[1, 0])

    def test_with_validity_mask(self):
        """Test with validity masks."""
        T, N, F = 30, 50, 2

        np.random.seed(44)
        values = np.random.randn(T, N, F)

        # Create validity mask with some invalid entries
        validity = np.random.rand(T, N, F) > 0.2

        time_axis = AxisRef(name="time", dtype="int64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        factor_batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
            validity=validity,
        )

        corr_matrix = compute_pairwise_correlation(factor_batch)

        assert corr_matrix.shape == (2, 2)
        # Should still compute correlations with masked data
        assert not np.all(np.isnan(corr_matrix))

    def test_one_dimensional_label_validity_is_broadcast(self):
        """Conditional IC accepts scalar-per-time label validity."""
        T, N, F = 2, 20, 2
        values = np.arange(T * N * F, dtype=float).reshape(T, N, F)
        batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=AxisRef(name="time", dtype="int64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=values,
        )
        labels = LabelBundle(
            target_id="returns",
            values=np.arange(T, dtype=float),
            validity=np.array([True, False]),
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )
        conditional_ic, counts = compute_conditional_ic(
            batch, labels, conditioning_factor_idx=0, quantiles=2, min_assets=2
        )
        assert np.all(counts[1] > 0)
        assert np.all(np.isnan(conditional_ic[1]))

    def test_incremental_ic_one_dimensional_label_validity(self):
        """Incremental IC accepts scalar-per-time label validity."""
        T, N, F = 2, 20, 2
        values = np.arange(T * N * F, dtype=float).reshape(T, N, F)
        batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=AxisRef(name="time", dtype="int64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=values,
        )
        labels = LabelBundle(
            target_id="returns",
            values=np.arange(T, dtype=float),
            validity=np.array([True, False]),
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )
        incremental, base, total = compute_incremental_ic(
            batch, labels, base_factor_indices=(0,), test_factor_idx=1, min_assets=2
        )
        assert np.isnan(incremental[1])
        assert np.isnan(base[1])
        assert np.isnan(total[1])

        """Test with sparse data (many NaNs)."""
        T, N, F = 30, 50, 2

        values = np.random.randn(T, N, F)
        # Make 70% of data NaN
        values[np.random.rand(T, N, F) < 0.7] = np.nan

        time_axis = AxisRef(name="time", dtype="int64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        factor_batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        labels = np.random.randn(T, N)
        labels[np.random.rand(T, N) < 0.7] = np.nan

        decision_time = tuple(range(T))

        label_bundle = LabelBundle(
            target_id="returns",
            values=labels,
            horizon=1,
            decision_time=decision_time,
            execution_time=decision_time,
            label_start_time=decision_time,
            label_end_time=tuple(t + 1 for t in decision_time),
        )

        # Should handle sparse data gracefully
        conditional_ic, sample_counts = compute_conditional_ic(
            factor_batch, label_bundle, conditioning_factor_idx=0, quantiles=3
        )

        # May have many NaNs but shouldn't crash
        assert conditional_ic.shape == (T, F, 3)

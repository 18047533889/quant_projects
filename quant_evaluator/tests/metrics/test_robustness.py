"""
Tests for robustness metrics with golden reference values.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.robustness import (
    compute_subsample_ic,
    compute_subsample_ic_std,
    compute_hac_variance,
    compute_hac_tstat,
    compute_block_bootstrap_ci,
)


class TestSubsampleIC:
    """Test subsample IC stability."""

    def test_subsample_ic_basic(self):
        """Basic subsample IC computation."""
        np.random.seed(42)
        ic_series = np.random.randn(100, 2) * 0.1 + 0.05  # Mean ~0.05

        subsample_means, subsample_stds = compute_subsample_ic(
            ic_series, num_subsamples=50, subsample_fraction=0.8, random_seed=42
        )

        assert subsample_means.shape == (50, 2)
        assert subsample_stds.shape == (50, 2)

        # Mean of subsample means should be close to population mean
        assert np.abs(np.nanmean(subsample_means[:, 0]) - 0.05) < 0.02

    def test_subsample_ic_stable_series(self):
        """Stable IC series yields low subsample variance."""
        ic_series = np.array([[0.05], [0.051], [0.049], [0.05]] * 30)  # 120 periods

        subsample_means, subsample_stds = compute_subsample_ic(
            ic_series, num_subsamples=100, subsample_fraction=0.8, random_seed=42
        )

        # Very stable -> low variance across subsamples
        variance_of_means = np.var(subsample_means[:, 0])
        assert variance_of_means < 1e-5

    def test_subsample_ic_unstable_series(self):
        """Unstable IC series yields high subsample variance."""
        np.random.seed(100)
        # High variance IC
        ic_series = np.random.randn(100, 1) * 0.3

        subsample_means, subsample_stds = compute_subsample_ic(
            ic_series, num_subsamples=100, subsample_fraction=0.8, random_seed=42
        )

        # Unstable -> higher variance than stable series
        variance_of_means = np.var(subsample_means[:, 0])
        assert variance_of_means > 0.0001

    def test_subsample_ic_fraction_validation(self):
        """Invalid subsample fraction raises error."""
        ic_series = np.random.randn(50, 1)

        with pytest.raises(ValueError, match="subsample_fraction must be in"):
            compute_subsample_ic(ic_series, num_subsamples=10, subsample_fraction=1.5)

    def test_subsample_ic_reproducibility(self):
        """Random seed ensures reproducibility."""
        np.random.seed(42)
        ic_series = np.random.randn(80, 1)

        means1, stds1 = compute_subsample_ic(
            ic_series, num_subsamples=20, subsample_fraction=0.7, random_seed=123
        )
        means2, stds2 = compute_subsample_ic(
            ic_series, num_subsamples=20, subsample_fraction=0.7, random_seed=123
        )

        assert np.allclose(means1, means2, equal_nan=True)
        assert np.allclose(stds1, stds2, equal_nan=True)


class TestSubsampleICStd:
    """Test subsample IC standard deviation."""

    def test_subsample_ic_std_wrapper(self):
        """Subsample IC std is derived from subsample_ic."""
        np.random.seed(42)
        ic_series = np.random.randn(100, 2)

        robustness_std = compute_subsample_ic_std(
            ic_series, num_subsamples=50, subsample_fraction=0.8, random_seed=42
        )

        assert robustness_std.shape == (2,)

        # Manually compute for verification
        subsample_means, _ = compute_subsample_ic(
            ic_series, num_subsamples=50, subsample_fraction=0.8, random_seed=42
        )
        expected_std = np.nanstd(subsample_means, axis=0, ddof=1)

        assert np.allclose(robustness_std, expected_std, equal_nan=True)

    def test_subsample_ic_std_low_for_stable(self):
        """Low robustness std for stable IC."""
        ic_series = np.array([[0.06]] * 150)

        robustness_std = compute_subsample_ic_std(
            ic_series, num_subsamples=100, subsample_fraction=0.8, random_seed=42
        )

        # Constant IC -> near-zero std
        assert robustness_std[0] < 1e-10


class TestHACVariance:
    """Test HAC variance estimation."""

    def test_hac_variance_white_noise(self):
        """White noise HAC variance equals sample variance."""
        np.random.seed(42)
        series = np.random.randn(200, 1)

        hac_var = compute_hac_variance(series, max_lag=5, kernel="bartlett")

        # For white noise, HAC variance ~ sample variance
        sample_var = np.var(series[:, 0], ddof=0) / len(series)
        assert np.isclose(hac_var[0], sample_var, rtol=0.3)

    def test_hac_variance_ar1_process(self):
        """AR(1) process has higher HAC variance than naive."""
        np.random.seed(100)
        phi = 0.7
        T = 300

        # Generate AR(1)
        series = np.zeros(T)
        series[0] = np.random.randn()
        for t in range(1, T):
            series[t] = phi * series[t-1] + np.random.randn()

        hac_var = compute_hac_variance(series.reshape(-1, 1), max_lag=10, kernel="bartlett")
        naive_var = np.var(series, ddof=0) / T

        # HAC variance should be larger due to autocorrelation
        assert hac_var[0] > naive_var

    def test_hac_variance_uniform_kernel(self):
        """Uniform kernel applies equal weights."""
        np.random.seed(42)
        series = np.random.randn(150, 1)

        hac_var_bartlett = compute_hac_variance(series, max_lag=5, kernel="bartlett")
        hac_var_uniform = compute_hac_variance(series, max_lag=5, kernel="uniform")

        # Different kernels yield different estimates
        assert not np.isclose(hac_var_bartlett[0], hac_var_uniform[0])

    def test_hac_variance_kernel_validation(self):
        """Invalid kernel raises error."""
        series = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="Unknown kernel"):
            compute_hac_variance(series, max_lag=5, kernel="gaussian")

    def test_hac_variance_multiple_series(self):
        """HAC variance for multiple series."""
        np.random.seed(42)
        series = np.random.randn(200, 3)

        hac_var = compute_hac_variance(series, max_lag=5, kernel="bartlett")

        assert hac_var.shape == (3,)
        assert np.all(hac_var > 0)

    def test_hac_variance_insufficient_data(self):
        """Insufficient data returns NaN."""
        series = np.random.randn(10, 1)

        hac_var = compute_hac_variance(series, max_lag=5, kernel="bartlett")

        assert np.isnan(hac_var[0])


class TestHACTStat:
    """Test HAC-robust t-statistic."""

    def test_hac_tstat_positive_mean(self):
        """Positive mean yields positive HAC t-stat."""
        np.random.seed(42)
        ic_series = np.random.randn(150, 1) * 0.05 + 0.1  # Mean ~0.1

        t_stat_hac, se_hac = compute_hac_tstat(ic_series, max_lag=5, kernel="bartlett")

        assert t_stat_hac[0] > 0
        assert se_hac[0] > 0

        # Verify t-stat calculation
        mean_ic = np.mean(ic_series[:, 0])
        expected_t = mean_ic / se_hac[0]
        assert np.isclose(t_stat_hac[0], expected_t, atol=1e-10)

    def test_hac_tstat_zero_mean(self):
        """Zero mean yields near-zero t-stat."""
        np.random.seed(200)
        ic_series = np.random.randn(300, 1)

        t_stat_hac, se_hac = compute_hac_tstat(ic_series, max_lag=5, kernel="bartlett")

        # Mean ~0 -> t-stat should be small with larger sample
        assert abs(t_stat_hac[0]) < 2.0

    def test_hac_tstat_conservative_vs_naive(self):
        """HAC t-stat is more conservative for autocorrelated data."""
        np.random.seed(42)
        phi = 0.8
        T = 300

        # Generate AR(1) IC series
        ic_series = np.zeros((T, 1))
        ic_series[0, 0] = np.random.randn()
        for t in range(1, T):
            ic_series[t, 0] = phi * ic_series[t-1, 0] + np.random.randn() * 0.1 + 0.05

        t_stat_hac, se_hac = compute_hac_tstat(ic_series, max_lag=10, kernel="bartlett")

        # Naive t-stat
        mean_ic = np.mean(ic_series[:, 0])
        naive_se = np.std(ic_series[:, 0], ddof=1) / np.sqrt(T)
        naive_t = mean_ic / naive_se

        # HAC t-stat should be smaller (more conservative)
        assert t_stat_hac[0] < naive_t
        assert se_hac[0] > naive_se

    def test_hac_tstat_insufficient_data(self):
        """Insufficient data returns NaN."""
        ic_series = np.random.randn(10, 1)

        t_stat_hac, se_hac = compute_hac_tstat(ic_series, max_lag=5, kernel="bartlett")

        assert np.isnan(t_stat_hac[0])
        assert np.isnan(se_hac[0])


class TestBlockBootstrapCI:
    """Test block bootstrap confidence interval."""

    def test_block_bootstrap_ci_basic(self):
        """Basic block bootstrap CI."""
        np.random.seed(42)
        ic_series = np.random.randn(200, 2) * 0.05 + 0.08  # Mean ~0.08

        ci_lower, ci_upper = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=500, confidence_level=0.95, random_seed=42
        )

        assert ci_lower.shape == (2,)
        assert ci_upper.shape == (2,)

        # CI should contain true mean
        true_mean = np.mean(ic_series[:, 0])
        assert ci_lower[0] < true_mean < ci_upper[0]

        # Upper > lower
        assert ci_upper[0] > ci_lower[0]

    def test_block_bootstrap_ci_width(self):
        """CI width reflects data variance."""
        np.random.seed(42)

        # Low variance
        ic_low_var = np.random.randn(150, 1) * 0.01 + 0.05
        # High variance
        ic_high_var = np.random.randn(150, 1) * 0.3 + 0.05

        ci_low_lower, ci_low_upper = compute_block_bootstrap_ci(
            ic_low_var, block_length=10, num_bootstrap=500, confidence_level=0.95, random_seed=42
        )
        ci_high_lower, ci_high_upper = compute_block_bootstrap_ci(
            ic_high_var, block_length=10, num_bootstrap=500, confidence_level=0.95, random_seed=42
        )

        width_low = ci_low_upper[0] - ci_low_lower[0]
        width_high = ci_high_upper[0] - ci_high_lower[0]

        # High variance -> wider CI
        assert width_high > width_low

    def test_block_bootstrap_ci_confidence_level(self):
        """Higher confidence level yields wider CI."""
        np.random.seed(100)
        ic_series = np.random.randn(200, 1)

        ci_lower_90, ci_upper_90 = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=500, confidence_level=0.90, random_seed=42
        )
        ci_lower_95, ci_upper_95 = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=500, confidence_level=0.95, random_seed=42
        )

        width_90 = ci_upper_90[0] - ci_lower_90[0]
        width_95 = ci_upper_95[0] - ci_lower_95[0]

        # 95% CI should be wider than 90% CI
        assert width_95 > width_90

    def test_block_bootstrap_ci_block_length_validation(self):
        """Invalid block length raises error."""
        ic_series = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="block_length must be in"):
            compute_block_bootstrap_ci(
                ic_series, block_length=0, num_bootstrap=100, confidence_level=0.95
            )

        with pytest.raises(ValueError, match="block_length must be in"):
            compute_block_bootstrap_ci(
                ic_series, block_length=150, num_bootstrap=100, confidence_level=0.95
            )

    def test_block_bootstrap_ci_confidence_validation(self):
        """Invalid confidence level raises error."""
        ic_series = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="confidence_level must be in"):
            compute_block_bootstrap_ci(
                ic_series, block_length=10, num_bootstrap=100, confidence_level=1.5
            )

    def test_block_bootstrap_ci_reproducibility(self):
        """Random seed ensures reproducibility."""
        np.random.seed(42)
        ic_series = np.random.randn(120, 1)

        ci_lower1, ci_upper1 = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=200, confidence_level=0.95, random_seed=999
        )
        ci_lower2, ci_upper2 = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=200, confidence_level=0.95, random_seed=999
        )

        assert np.allclose(ci_lower1, ci_lower2, equal_nan=True)
        assert np.allclose(ci_upper1, ci_upper2, equal_nan=True)

    def test_block_bootstrap_ci_insufficient_data(self):
        """Insufficient data returns NaN."""
        ic_series = np.random.randn(15, 1)

        ci_lower, ci_upper = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=100, confidence_level=0.95, random_seed=42
        )

        assert np.isnan(ci_lower[0])
        assert np.isnan(ci_upper[0])


class TestRNGHygiene:
    """RNG hygiene (QE-METRIC-P0-08): all sampling uses a LOCAL
    np.random.default_rng(random_seed) Generator. The GLOBAL numpy random
    state must never be touched (no np.random.seed, no np.random.choice)."""

    def _global_state_signature(self):
        st = np.random.get_state()
        return (st[0], tuple(st[1][:4]), st[2])

    def test_same_seed_identical_subsample_results(self):
        rng = np.random.default_rng(7)
        ic_series = rng.normal(0.05, 0.1, size=(200, 2))

        means_a, stds_a = compute_subsample_ic(
            ic_series, num_subsamples=30, subsample_fraction=0.8, random_seed=123
        )
        means_b, stds_b = compute_subsample_ic(
            ic_series, num_subsamples=30, subsample_fraction=0.8, random_seed=123
        )

        np.testing.assert_array_equal(means_a, means_b)
        np.testing.assert_array_equal(stds_a, stds_b)

    def test_different_seeds_different_subsample_draws(self):
        rng = np.random.default_rng(8)
        ic_series = rng.normal(0.05, 0.1, size=(200, 1))

        means_a, _ = compute_subsample_ic(
            ic_series, num_subsamples=20, subsample_fraction=0.8, random_seed=1
        )
        means_b, _ = compute_subsample_ic(
            ic_series, num_subsamples=20, subsample_fraction=0.8, random_seed=2
        )

        # Different seeds must draw different subsamples (different means).
        assert not np.allclose(means_a, means_b)

    def test_global_rng_state_unchanged(self):
        """Calling the samplers must not perturb np.random's global state."""
        rng = np.random.default_rng(9)
        ic_series = rng.normal(0.05, 0.1, size=(150, 2))

        before = self._global_state_signature()
        compute_subsample_ic(
            ic_series, num_subsamples=25, subsample_fraction=0.8, random_seed=42
        )
        mid = self._global_state_signature()
        compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=200,
            confidence_level=0.95, random_seed=42,
        )
        after = self._global_state_signature()

        assert before == mid
        assert before == after

    def test_block_bootstrap_same_seed_reproducible_and_local(self):
        rng = np.random.default_rng(10)
        ic_series = rng.normal(0.03, 0.08, size=(180, 1))

        before = self._global_state_signature()
        lo_a, hi_a = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=300,
            confidence_level=0.95, random_seed=77,
        )
        lo_b, hi_b = compute_block_bootstrap_ci(
            ic_series, block_length=10, num_bootstrap=300,
            confidence_level=0.95, random_seed=77,
        )
        after = self._global_state_signature()

        np.testing.assert_array_equal(lo_a, lo_b)
        np.testing.assert_array_equal(hi_a, hi_b)
        assert before == after

    def test_subsample_uses_generator_not_legacy_choice(self):
        """The sampler's draws must match default_rng(seed).choice, not the
        legacy np.random.choice stream (regression guard for global RNG use)."""
        T = 100
        subsample_size = 80
        ic_series = np.linspace(0.01, 0.1, T).reshape(-1, 1)

        # What default_rng(5).choice would produce for draw 0:
        expected_first = np.sort(np.random.default_rng(5).choice(
            T, size=subsample_size, replace=False
        ))

        # Reconstruct the actual first subsample from its mean: seed the
        # sampler, then verify determinism against the Generator stream by
        # running the sampler twice with the same seed (equality) and once
        # with a seed matching the expected draw.
        means_a, _ = compute_subsample_ic(
            ic_series, num_subsamples=1, subsample_fraction=0.8, random_seed=5
        )
        means_b, _ = compute_subsample_ic(
            ic_series, num_subsamples=1, subsample_fraction=0.8, random_seed=5
        )
        assert means_a[0, 0] == means_b[0, 0]
        # And the mean equals the mean over the Generator's own draw.
        expected_mean = float(np.mean(ic_series[expected_first, 0]))
        assert means_a[0, 0] == pytest.approx(expected_mean, rel=1e-12)

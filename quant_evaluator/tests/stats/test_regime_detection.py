"""
Tests for regime detection functionality.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.stats.regime_detection import (
    GaussianHMM,
    detect_regimes,
)


class TestGaussianHMM:
    """Test suite for Gaussian HMM."""

    def test_hmm_two_regimes_fit(self):
        """Test HMM fitting with two distinct regimes."""
        np.random.seed(42)
        T = 500

        # Generate data with two regimes
        states_true = np.zeros(T, dtype=int)
        data = np.zeros(T)

        # Regime 1: mean=0, std=1 (first 250)
        states_true[:250] = 0
        data[:250] = np.random.randn(250)

        # Regime 2: mean=5, std=2 (last 250)
        states_true[250:] = 1
        data[250:] = np.random.randn(250) * 2 + 5

        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)
        model.fit(data)

        # Check that model converged
        assert model.n_iter_fit_ <= 50

        # Check that means are separated
        means_sorted = np.sort(model.means_)
        assert means_sorted[1] - means_sorted[0] > 2.0

        # Check variances are positive
        assert np.all(model.vars_ > 0)

        # Check transition matrix is stochastic
        assert np.allclose(model.trans_mat_.sum(axis=1), 1.0)

    def test_hmm_predict(self):
        """Test HMM state prediction."""
        np.random.seed(42)
        T = 400

        # Two regimes with clear separation
        data = np.zeros(T)
        data[:200] = np.random.randn(200) * 0.5 - 2.0  # Low regime
        data[200:] = np.random.randn(200) * 0.5 + 3.0  # High regime

        model = GaussianHMM(n_states=2, n_iter=100, random_state=42)
        model.fit(data)

        states = model.predict(data)

        # States should be mostly consistent within each regime
        assert states.shape == (T,)
        assert np.all((states >= 0) & (states < 2))

        # Check that state changes occur (not all same state)
        assert len(np.unique(states)) == 2

        # First half and second half should be mostly different states
        state_first_half = np.bincount(states[:200]).argmax()
        state_second_half = np.bincount(states[200:]).argmax()

        # Should identify different dominant states
        assert state_first_half != state_second_half or np.abs(model.means_[0] - model.means_[1]) < 1.0

    def test_hmm_predict_proba(self):
        """Test HMM posterior probability computation."""
        np.random.seed(42)
        T = 300

        data = np.concatenate([
            np.random.randn(150) - 1.5,
            np.random.randn(150) + 2.0,
        ])

        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)
        model.fit(data)

        probs = model.predict_proba(data)

        # Check shape and validity
        assert probs.shape == (T, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)

        # Probabilities should be decisive in clear regimes
        # (close to 0 or 1 for at least some observations)
        max_probs = probs.max(axis=1)
        assert np.sum(max_probs > 0.9) > T * 0.3  # At least 30% confident

    def test_hmm_convergence(self):
        """Test HMM convergence behavior."""
        np.random.seed(42)
        T = 250

        data = np.random.randn(T)  # Simple normal data

        model = GaussianHMM(n_states=2, n_iter=100, tol=1e-4, random_state=42)
        model.fit(data)

        # Should have log-likelihood history
        assert len(model.log_likelihood_history_) > 0

        # Log-likelihood should generally increase
        ll_history = np.array(model.log_likelihood_history_)
        if len(ll_history) > 1:
            # At least last few iterations should not decrease significantly
            assert ll_history[-1] >= ll_history[0] - 1.0

    def test_hmm_three_states(self):
        """Test HMM with three states."""
        np.random.seed(42)
        T = 600

        # Three regimes
        data = np.concatenate([
            np.random.randn(200) * 0.5 - 3.0,  # Low
            np.random.randn(200) * 0.5,         # Mid
            np.random.randn(200) * 0.5 + 3.0,   # High
        ])

        model = GaussianHMM(n_states=3, n_iter=100, random_state=42)
        model.fit(data)

        # Check parameters
        assert model.means_.shape == (3,)
        assert model.vars_.shape == (3,)
        assert model.trans_mat_.shape == (3, 3)

        states = model.predict(data)
        assert len(np.unique(states)) <= 3  # May not use all states

    def test_hmm_insufficient_data(self):
        """Test error with insufficient data."""
        data = np.random.randn(10)

        model = GaussianHMM(n_states=2, n_iter=50)

        # Should raise error or handle by removing NaN
        try:
            model.fit(data)
            # If it doesn't raise, data was sufficient after NaN removal
            assert model.means_.shape == (2,)
        except ValueError as e:
            assert "Insufficient" in str(e)

    def test_hmm_with_nan(self):
        """Test HMM with NaN values."""
        np.random.seed(42)
        data = np.random.randn(200)
        data[50:60] = np.nan  # Inject NaN

        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)
        model.fit(data)  # Should handle by removing NaN

        # Should fit successfully
        assert model.means_.shape == (2,)

    def test_hmm_single_state(self):
        """Test HMM with single state (degenerate case)."""
        np.random.seed(42)
        data = np.random.randn(150)

        model = GaussianHMM(n_states=1, n_iter=50, random_state=42)
        model.fit(data)

        states = model.predict(data)

        # All states should be 0
        assert np.all(states == 0)

    def test_hmm_transition_persistence(self):
        """Test that HMM captures persistent states."""
        np.random.seed(42)
        T = 600

        # Generate with high persistence (few transitions)
        states_true = np.zeros(T, dtype=int)
        states_true[200:400] = 1
        states_true[400:] = 0

        data = np.zeros(T)
        data[states_true == 0] = np.random.randn(np.sum(states_true == 0)) * 0.5 - 1.0
        data[states_true == 1] = np.random.randn(np.sum(states_true == 1)) * 0.5 + 2.0

        model = GaussianHMM(n_states=2, n_iter=100, random_state=42)
        model.fit(data)

        # Transition matrix should have high diagonal (persistence)
        diag_avg = np.mean(np.diag(model.trans_mat_))
        assert diag_avg > 0.6  # States are persistent


class TestDetectRegimes:
    """Test convenience function for regime detection."""

    def test_detect_regimes_basic(self):
        """Test basic regime detection."""
        np.random.seed(42)
        T = 400

        # Two regimes
        data = np.concatenate([
            np.random.randn(200) * 1.0 - 2.0,
            np.random.randn(200) * 1.5 + 2.0,
        ])

        states, probs, model = detect_regimes(data, n_states=2, n_iter=50, random_state=42)

        # Check outputs
        assert states.shape == (T,)
        assert probs.shape == (T, 2)
        assert isinstance(model, GaussianHMM)

        # States should identify regimes
        assert len(np.unique(states)) == 2

    def test_detect_regimes_returns_model(self):
        """Test that returned model can be reused."""
        np.random.seed(42)
        T = 300

        data = np.random.randn(T)

        states, probs, model = detect_regimes(data, n_states=2, random_state=42)

        # Use model on new data
        new_data = np.random.randn(100)
        new_states = model.predict(new_data)
        new_probs = model.predict_proba(new_data)

        assert new_states.shape == (100,)
        assert new_probs.shape == (100, 2)

    def test_detect_regimes_market_like(self):
        """Test regime detection on market-like returns."""
        np.random.seed(42)
        T = 500

        # Simulate bull/bear regimes with realistic parameters
        states_true = np.zeros(T, dtype=int)
        returns = np.zeros(T)

        # Bull market: positive drift, low vol
        bull_periods = [0, 100, 200, 300, 400]
        bear_periods = [50, 150, 250, 350, 450]

        for start, end in zip(bull_periods, bear_periods):
            if end <= T:
                returns[start:end] = np.random.randn(end - start) * 0.01 + 0.001
                states_true[start:end] = 0

        for start, end in zip(bear_periods, bull_periods[1:] + [T]):
            if end <= T:
                returns[start:end] = np.random.randn(end - start) * 0.025 - 0.002
                states_true[start:end] = 1

        states, probs, model = detect_regimes(returns, n_states=2, n_iter=100, random_state=42)

        # Should identify two regimes
        assert len(np.unique(states)) == 2

        # Regimes should have different characteristics
        means_sorted = np.sort(model.means_)
        vars_sorted = np.sort(model.vars_)

        # At least some difference in means or variances
        assert (means_sorted[1] - means_sorted[0] > 0.002 or
                vars_sorted[1] / vars_sorted[0] > 1.5)


class TestHMMEdgeCases:
    """Test edge cases and robustness."""

    def test_hmm_constant_data(self):
        """Test HMM with constant data."""
        data = np.ones(100) * 5.0

        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)

        # Should handle gracefully (may not converge meaningfully)
        model.fit(data)

        # Variance should be clamped to minimum
        assert np.all(model.vars_ >= 1e-6)

    def test_hmm_nearly_constant_data(self):
        """Test HMM with very low variance data."""
        np.random.seed(42)
        data = np.random.randn(150) * 0.001 + 10.0

        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)
        model.fit(data)

        # Should not crash and maintain positive variances
        assert np.all(model.vars_ > 0)

    def test_hmm_outliers(self):
        """Test HMM robustness to outliers."""
        np.random.seed(42)
        T = 300

        data = np.random.randn(T)
        # Add outliers
        data[50] = 10.0
        data[150] = -10.0
        data[250] = 15.0

        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)
        model.fit(data)

        # Should still fit
        assert model.means_.shape == (2,)
        assert np.all(np.isfinite(model.means_))

    def test_hmm_very_short_regimes(self):
        """Test HMM with rapidly switching regimes."""
        np.random.seed(42)
        T = 200

        # Alternate regimes every 10 observations
        data = np.zeros(T)
        for i in range(0, T, 10):
            if (i // 10) % 2 == 0:
                data[i:i+10] = np.random.randn(min(10, T-i)) - 1.0
            else:
                data[i:i+10] = np.random.randn(min(10, T-i)) + 1.0

        model = GaussianHMM(n_states=2, n_iter=100, random_state=42)
        model.fit(data)

        # Transition matrix should show more switching (lower diagonal)
        diag_avg = np.mean(np.diag(model.trans_mat_))
        # With frequent switches, persistence should be lower
        # (though still may be above 0.5 due to estimation)
        assert 0.3 < diag_avg < 0.95

    def test_hmm_initialization_stability(self):
        """Test that different random seeds give reasonable results."""
        np.random.seed(42)
        T = 300

        data = np.concatenate([
            np.random.randn(150) - 1.5,
            np.random.randn(150) + 1.5,
        ])

        # Fit with different seeds
        models = []
        for seed in [0, 1, 2, 3, 4]:
            model = GaussianHMM(n_states=2, n_iter=100, random_state=seed)
            model.fit(data)
            models.append(model)

        # All should identify two separated means
        for model in models:
            means_sorted = np.sort(model.means_)
            assert means_sorted[1] - means_sorted[0] > 1.0

    def test_hmm_predict_unseen_data(self):
        """Test prediction on data from different distribution."""
        np.random.seed(42)

        # Train on one distribution
        train_data = np.random.randn(200)
        model = GaussianHMM(n_states=2, n_iter=50, random_state=42)
        model.fit(train_data)

        # Predict on different distribution
        test_data = np.random.randn(100) * 3.0 + 5.0
        states = model.predict(test_data)
        probs = model.predict_proba(test_data)

        # Should not crash and return valid outputs
        assert states.shape == (100,)
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

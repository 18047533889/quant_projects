"""
Fail-closed regression tests for risk stress testing / VaR-CVaR.

Each test pins one verified fail-open defect:
1. apply_hypothetical_scenario with <2 finite values -> NaN result, never an
   unshocked passthrough copy of the input.
3. compute_scenario_impact with unknown metric names -> ValueError listing
   the unknown metric(s).
4. compute_worst_case_scenarios must never rank NaN windows as worst-case
   winners.
5. confidence_level outside (0, 1) -> ValueError (1.0 previously caused
   division by zero -> inf VaR/CVaR).
2/6. Determinism / RNG hygiene: repeated calls with the same seed give
   identical results, and the global numpy RNG state is never touched
   (verified via np.random.get_state() sentinel). Note: the current
   stress_testing.py / var_cvar.py contain no stochastic sampling at all,
   so the determinism assertions hold trivially and will keep guarding if
   sampling is (re)introduced with local default_rng generators.
"""

import pytest
import numpy as np

from quant_evaluator.metrics.risk.stress_testing import (
    apply_hypothetical_scenario,
    compute_scenario_impact,
    compute_worst_case_scenarios,
)
from quant_evaluator.metrics.risk.var_cvar import (
    compute_var,
    compute_var_historical,
    compute_var_parametric,
    compute_var_cornish_fisher,
    compute_cvar,
    compute_var_cvar,
)


def _rng_data(seed: int = 7, T: int = 100) -> np.ndarray:
    """Local-generator data factory (never touches global RNG)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T) * 0.02


class TestDefect1HypotheticalScenarioFailClosed:
    """Defect 1: <2 finite values must yield NaN, not unshocked copy."""

    def test_all_nan_column_marks_nan_not_passthrough(self):
        returns = np.column_stack([
            _rng_data(seed=1, T=50),
            np.full(50, np.nan),
        ])
        shocked = apply_hypothetical_scenario(
            returns, shock_size=-0.10, shock_type="absolute"
        )
        # Healthy column is shocked; degenerate column must be NaN,
        # NOT an unshocked copy of the input (all-NaN in == all-NaN out,
        # so the distinguishing assertion lives in the single-finite-value
        # test below where the input is NOT all NaN).
        assert np.all(np.isnan(shocked[:, 1]))
        assert np.allclose(shocked[:, 0], returns[:, 0] - 0.10)

    def test_single_finite_value_marks_nan(self):
        ret = np.full(30, np.nan)
        ret[0] = 0.01
        shocked = apply_hypothetical_scenario(ret, shock_size=-0.10)
        # Fail-open would return the unshocked value 0.01 here;
        # fail-closed must invalidate the whole result to NaN.
        assert np.all(np.isnan(shocked))

    def test_1d_all_nan_marks_nan(self):
        shocked = apply_hypothetical_scenario(
            np.full(10, np.nan), shock_size=-0.05
        )
        assert np.all(np.isnan(shocked))

    def test_healthy_column_still_shocked(self):
        ret = _rng_data(seed=2, T=50)
        shocked = apply_hypothetical_scenario(ret, shock_size=-0.10)
        assert np.allclose(shocked, ret - 0.10)

    def test_mixed_multi_factor(self):
        rng = np.random.default_rng(3)
        good = rng.standard_normal(40) * 0.02
        one_finite = np.full(40, np.nan)
        one_finite[5] = 0.02
        returns = np.column_stack([good, one_finite])
        shocked = apply_hypothetical_scenario(returns, shock_size=-0.10)
        assert np.all(np.isnan(shocked[:, 1]))
        assert np.allclose(shocked[:, 0], good - 0.10)


class TestDefect3UnknownMetricFailClosed:
    """Defect 3: unknown metric names in scenario config -> ValueError."""

    def test_unknown_single_metric_raises(self):
        baseline = _rng_data(seed=4, T=100)
        scenario = _rng_data(seed=5, T=50) - 0.01
        with pytest.raises(ValueError, match="sharpe_ratio"):
            compute_scenario_impact(baseline, scenario, metrics=["sharpe_ratio"])

    def test_unknown_metric_named_in_message(self):
        baseline = _rng_data(seed=4, T=100)
        scenario = _rng_data(seed=5, T=50)
        with pytest.raises(ValueError, match="nonexistent_metric"):
            compute_scenario_impact(
                baseline, scenario, metrics=["mean", "nonexistent_metric"]
            )

    def test_multiple_unknown_metrics_all_listed(self):
        baseline = _rng_data(seed=4, T=100)
        scenario = _rng_data(seed=5, T=50)
        with pytest.raises(ValueError) as excinfo:
            compute_scenario_impact(
                baseline, scenario, metrics=["bad_one", "bad_two"]
            )
        msg = str(excinfo.value)
        assert "bad_one" in msg and "bad_two" in msg

    def test_valid_metrics_still_accepted(self):
        baseline = _rng_data(seed=4, T=100)
        scenario = _rng_data(seed=5, T=50)
        impact = compute_scenario_impact(
            baseline, scenario,
            metrics=["mean", "sharpe", "max_drawdown", "var", "cvar"],
        )
        for key in ("mean_baseline", "sharpe_baseline", "max_drawdown_baseline",
                    "var_baseline", "cvar_baseline"):
            assert key in impact


class TestDefect4WorstCaseNeverNaN:
    """Defect 4: NaN metric values must never win the worst-case ranking."""

    @staticmethod
    def _returns_with_nan_window(T: int = 120, window: int = 20,
                                 seed: int = 11) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ret = rng.standard_normal(T) * 0.02
        # A fully-NaN window: cumulative product becomes NaN.
        ret[40:40 + window] = np.nan
        return ret

    def test_nan_window_not_selected_as_worst(self):
        ret = self._returns_with_nan_window()
        worst = compute_worst_case_scenarios(
            ret, n_scenarios=10, window_size=20, metric="cumulative"
        )
        assert len(worst) > 0
        for s in worst:
            assert np.isfinite(s["metric_value"]), (
                f"NaN metric_value leaked into worst-case ranking: {s}"
            )
        # No returned scenario may overlap the NaN-only window [40, 59].
        for s in worst:
            assert not (s["start_idx"] < 60 and s["end_idx"] > 40)

    def test_nan_window_not_selected_volatility(self):
        # nanstd of an all-NaN window -> NaN (with >=2 NaNs, ddof=1 -> NaN).
        ret = self._returns_with_nan_window(T=120, window=20, seed=12)
        worst = compute_worst_case_scenarios(
            ret, n_scenarios=10, window_size=20, metric="volatility"
        )
        for s in worst:
            assert np.isfinite(s["metric_value"])

    def test_partial_nan_window_excluded_when_metric_nan(self):
        rng = np.random.default_rng(13)
        ret = rng.standard_normal(100) * 0.02
        ret[30:50] = np.nan  # window fully NaN
        worst = compute_worst_case_scenarios(
            ret, n_scenarios=5, window_size=20, metric="cumulative"
        )
        assert all(np.isfinite(s["metric_value"]) for s in worst)

    def test_all_finite_windows_unchanged_behavior(self):
        rng = np.random.default_rng(14)
        ret = rng.standard_normal(200) * 0.02
        ret[50:70] = -0.05
        worst = compute_worst_case_scenarios(
            ret, n_scenarios=5, window_size=20, metric="cumulative"
        )
        assert len(worst) == 5
        for i in range(len(worst) - 1):
            assert worst[i]["metric_value"] <= worst[i + 1]["metric_value"]


class TestDefect5ConfidenceLevelValidation:
    """Defect 5: confidence_level outside (0, 1) -> ValueError."""

    @pytest.mark.parametrize("bad_cl", [1.0, 0.0, -0.5, 1.5, 2.0])
    @pytest.mark.parametrize(
        "func,kwargs",
        [
            (compute_var, {}),
            (compute_var_historical, {}),
            (compute_var_parametric, {}),
            (compute_var_cornish_fisher, {}),
            (compute_cvar, {}),
            (compute_var_cvar, {}),
        ],
    )
    def test_invalid_confidence_level_raises(self, bad_cl, func, kwargs):
        returns = _rng_data(seed=6, T=100)
        with pytest.raises(ValueError, match="confidence_level"):
            if func is compute_var_cvar:
                func(returns, confidence_level=bad_cl, **kwargs)
            else:
                func(returns, confidence_level=bad_cl, **kwargs)

    def test_valid_boundary_inside_range_accepted(self):
        returns = _rng_data(seed=6, T=200)
        # Values strictly inside (0, 1) must still work.
        assert np.isfinite(compute_var(returns, confidence_level=0.001))
        assert np.isfinite(compute_cvar(returns, confidence_level=0.999,
                                        method="parametric"))

    def test_cvar_parametric_cl_one_no_inf(self):
        returns = _rng_data(seed=6, T=200)
        # Previously: division by zero -> -inf/inf. Now fails closed.
        with pytest.raises(ValueError):
            compute_cvar(returns, confidence_level=1.0, method="parametric")


class TestDefects2And6RngHygiene:
    """Defects 2/6: determinism and global-RNG hygiene."""

    @staticmethod
    def _global_state_signature() -> tuple:
        state = np.random.get_state()
        # (MT19937 internal key first words, pos) uniquely identifies state.
        return (tuple(state[1][:4]), state[2])

    def test_global_rng_state_untouched_by_owned_modules(self):
        # Warm up global RNG to a known, non-default point.
        np.random.seed(12345)
        np.random.random(17)
        before = self._global_state_signature()

        ret = _rng_data(seed=8, T=100)
        apply_hypothetical_scenario(ret, shock_size=-0.05)
        compute_worst_case_scenarios(ret, n_scenarios=3, window_size=10)
        compute_scenario_impact(ret, ret * 0.5, metrics=["mean", "var", "cvar"])
        compute_var(ret, confidence_level=0.95)
        compute_cvar(ret, confidence_level=0.95, method="parametric")
        compute_var_cvar(ret, confidence_level=0.99)

        after = self._global_state_signature()
        assert before == after, (
            "Global numpy RNG state was mutated by risk metrics; "
            "use np.random.default_rng(seed) local generators instead."
        )

    def test_repeated_calls_identical_results(self):
        ret = _rng_data(seed=9, T=200)
        r1 = apply_hypothetical_scenario(ret, shock_size=-0.07)
        r2 = apply_hypothetical_scenario(ret, shock_size=-0.07)
        assert np.array_equal(r1, r2)

        w1 = compute_worst_case_scenarios(ret, n_scenarios=5, window_size=15)
        w2 = compute_worst_case_scenarios(ret, n_scenarios=5, window_size=15)
        assert len(w1) == len(w2)
        for s1, s2 in zip(w1, w2):
            assert s1["start_idx"] == s2["start_idx"]
            assert s1["metric_value"] == s2["metric_value"]

    def test_no_global_seed_usage_in_owned_modules(self):
        # Guard: neither owned module may call the legacy global RNG API.
        import inspect
        import quant_evaluator.metrics.risk.stress_testing as st
        import quant_evaluator.metrics.risk.var_cvar as vc

        for module in (st, vc):
            source = inspect.getsource(module)
            assert "np.random.seed(" not in source, (
                f"{module.__name__} uses global np.random.seed"
            )
            for forbidden in ("np.random.randn(", "np.random.random(",
                              "np.random.choice(", "np.random.rand("):
                assert forbidden not in source, (
                    f"{module.__name__} uses legacy global RNG call {forbidden}"
                )

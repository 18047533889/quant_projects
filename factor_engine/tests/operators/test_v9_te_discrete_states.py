from collections import Counter
import numpy as np
import pandas as pd
import pytest


def _oracle(target, source, lag=1, states=2, source_states=None):
    triples = [(int(target[i + lag]), int(target[i]), int(source[i]))
               for i in range(len(target) - lag)
               if np.isfinite([target[i + lag], target[i], source[i]]).all()]
    counts = Counter(triples)
    mass = {key: counts[key] + 0.5 for key in np.ndindex(states, states, source_states or states)}
    total = sum(mass.values())
    probability = {key: value / total for key, value in mass.items()}
    result = 0.0
    for (future, current, driver), p in probability.items():
        px = sum(q for (a, b, c), q in probability.items() if b == current)
        pfx = sum(q for (a, b, c), q in probability.items() if a == future and b == current)
        pxy = sum(q for (a, b, c), q in probability.items() if b == current and c == driver)
        result += p * np.log(p * px / (pfx * pxy))
    return result


@pytest.mark.parametrize("states", [2, 3])
def test_ordered_states_match_independent_smoothed_conditional_information(states):
    from factor_engine.cleaned_operators import advanced_information as m
    source = np.random.default_rng(33).integers(states, size=256).astype(float)
    target = np.r_[0.0, source[:-1]]
    actual = m._transfer_entropy_window(target, source, 3, 1, 30)
    assert actual == pytest.approx(_oracle(target, source, states=states), abs=1e-12)
    assert actual > 0.5
    status = m.last_te_binning_status()
    assert status["requested_bins"] == 3
    assert status["effective_x_bins"] == status["effective_y_bins"] == states


def test_ties_and_adjacent_extreme_categories_never_merge():
    from factor_engine.cleaned_operators import advanced_information as m
    values = np.array([1e300, np.nextafter(1e300, np.inf)])
    edges = m._quantile_edges(np.repeat(values, [99, 1]), 3)
    assert edges.size == 3
    np.testing.assert_array_equal(np.digitize(values, edges) - 1, [0, 1])
    constant = np.ones(100)
    assert np.isnan(m._transfer_entropy_window(constant, constant, 3, 1, 30))
    assert m._quantile_edges(constant, 3).size == 2


def test_larger_alphabet_preserves_quantile_policy_without_jitter():
    from factor_engine.cleaned_operators import advanced_information as m
    values = np.r_[np.zeros(90), np.arange(1.0, 11.0)]
    expected = np.unique(np.quantile(values, np.linspace(0, 1, 4)))
    np.testing.assert_array_equal(m._quantile_edges(values, 3), expected)


def test_target_state_first_seen_at_endpoint_is_retained():
    from factor_engine.cleaned_operators import advanced_information as m
    source = np.random.default_rng(123).integers(2, size=256).astype(float)
    target = np.r_[0.0, source[:-1]]
    target[-1] = 2.0
    expected = _oracle(target, source, states=3, source_states=2)
    actual = m._transfer_entropy_window(target, source, 3, 1, 30)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert m.last_te_binning_status()["effective_x_bins"] == 3


def test_continuous_quantile_interpolation_is_finite_at_extreme_units():
    from factor_engine.cleaned_operators import advanced_information as m
    values = np.array([-1e308, -9e307, 9e307, 1e308])
    np.testing.assert_array_equal(m._quantile_edges(values, 2), [-1e308, 0.0, 1e308])


def test_gaps_keep_physical_lag_and_surrogates_share_category_policy():
    from factor_engine.cleaned_operators import advanced_information as m
    source = np.random.default_rng(93).integers(2, size=256).astype(float)
    target = np.r_[0.0, source[:-1]]
    source[43:51] = np.nan
    target[90:94] = np.nan
    expected = _oracle(target, source)
    assert m._transfer_entropy_window(target, source, 3, 1, 30) == pytest.approx(expected, abs=1e-12)
    nulls = []
    for offset in m._SURROGATE_OFFSETS:
        shuffled = source.copy()
        finite = np.isfinite(shuffled)
        shuffled[finite] = np.roll(shuffled[finite], -offset)
        nulls.append(_oracle(target, shuffled))
    actual = m._effective_transfer_entropy_window(target, source, 3, 1, 30)
    assert actual == pytest.approx(expected - np.mean(nulls), abs=1e-12)
    strength, lag = m._te_peak_window(target, source, 3, 30)
    assert strength > 0.5 and lag == pytest.approx(0.1)


def test_independent_binary_control_and_public_prefix():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators import advanced_information as m
    rng = np.random.default_rng(23)
    source = rng.integers(2, size=280).astype(float)
    independent = rng.integers(2, size=280).astype(float)
    assert m._transfer_entropy_window(independent, source, 3, 1, 30) < 0.05
    ensure_cleaned_loaded()
    op = OperatorRegistry.get("ts_transfer_entropy", "pandas_numpy", mode="research")
    target = pd.DataFrame({"A": np.r_[0.0, source[:-1]]})
    driver = pd.DataFrame({"A": source})
    actual = op.calculate(target, driver, window=256, bins=3)
    pd.testing.assert_frame_equal(actual.iloc[:260], op.calculate(target.iloc[:260], driver.iloc[:260], window=256, bins=3))
    assert actual.iloc[-1, 0] > 0.5

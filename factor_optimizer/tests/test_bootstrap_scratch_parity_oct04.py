"""Reference parity tests for a bounded moving-block bootstrap helper."""
import importlib
import math

import numpy as np
import pytest


def _api():
    module_name = "factor_optimizer.research_bootstrap"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        pytest.fail("moving-block bootstrap API is not implemented")


def _old_lower_bound(differences, *, block_length, horizon,
                     minimum_validation_days, bootstrap_draws, seed,
                     confidence_level):
    """Independent oracle copied from the pre-extraction implementation."""
    length = max(block_length, int(horizon))
    n = len(differences)
    if n < 3 * length:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_draws):
        starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
        values = np.concatenate([differences[s:s + length] for s in starts])[:n]
        values = values[np.isfinite(values)]
        if len(values) < minimum_validation_days:
            return None
        draws.append(float(values.mean()))
    return float(np.quantile(draws, (1 - confidence_level) / 2))


def _compare(differences, *, block_length, horizon, minimum_validation_days,
             bootstrap_draws=31, seed=20261004, confidence_level=.95):
    kwargs = dict(
        block_length=block_length,
        horizon=horizon,
        minimum_validation_days=minimum_validation_days,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
        confidence_level=confidence_level,
    )
    expected = _old_lower_bound(differences, **kwargs)
    actual = _api().moving_block_lower_bound(differences, **kwargs)
    assert actual == expected


def test_exact_seeded_parity_with_nondivisible_tail_and_nan_inf_positions():
    # Defect caught: dropping invalid dates before block construction or using
    # a different final-block width changes the sampled observations.
    values = np.array([
        .25, np.nan, -.5, .75, np.inf, -.125, .375, -.625, 1.0,
        np.nan, .5, -.25, .125, -.875, np.inf, .625, -.375,
    ], dtype=np.float64)
    _compare(values, block_length=4, horizon=2, minimum_validation_days=5)


def test_exact_seeded_parity_when_horizon_exceeds_configured_block_length():
    # Defect caught: failing to use max(block_length, horizon) for sampling.
    values = np.array([
        -.4, .2, .1, np.nan, .3, -.5, .8, .2, -.1, .7, .5, -.3,
        .4, -.2, .9, .1, -.8, .6, .2, -.1, .3,
    ])
    _compare(values, block_length=2, horizon=5, minimum_validation_days=4,
             seed=17, bootstrap_draws=43)


def test_insufficient_sampled_finite_observations_returns_none():
    # Defect caught: counting unique rows or omitting the per-draw minimum gate.
    values = np.array([1.0, np.nan, np.inf, -np.inf, np.nan, .5,
                       np.nan, np.nan, np.nan, 2.0, np.nan, np.nan])
    _compare(values, block_length=3, horizon=1, minimum_validation_days=5,
             bootstrap_draws=15, seed=3)


def test_short_series_returns_none_before_bootstrap():
    # Defect caught: resampling when fewer than three effective blocks exist.
    values = np.array([.2, np.nan, -.1, .4, .3, .1])
    _compare(values, block_length=2, horizon=3, minimum_validation_days=1)


def test_cancellation_sensitive_samples_match_old_mean_and_quantile_exactly():
    # Defect caught: replacing per-draw NumPy mean with differently grouped
    # prefix-sum arithmetic that moves a decision near its threshold.
    values = np.array([
        1e16, 1., -1e16, .5, -1e16, 1e16, 3., -2.,
        np.nan, 1e16, -1e16, .25, -.75, 5., -4., 2.,
    ])
    _compare(values, block_length=3, horizon=2, minimum_validation_days=4,
             bootstrap_draws=41, seed=99, confidence_level=.9)


def test_repeated_call_is_seed_deterministic_and_matches_reference():
    values = np.array([.5, -.25, np.nan, .75, .125, -.5, .25, .375,
                       -.125, .625, -.75, .875, .1, -.2, .3])
    kwargs = dict(block_length=4, horizon=1, minimum_validation_days=3,
                  bootstrap_draws=51, seed=812, confidence_level=.9)
    expected = _old_lower_bound(values, **kwargs)
    api = _api()
    assert api.moving_block_lower_bound(values, **kwargs) == expected
    assert api.moving_block_lower_bound(values, **kwargs) == expected


@pytest.mark.parametrize("name,bad_value", [
    ("block_length", True), ("block_length", 1.5), ("block_length", 0),
    ("horizon", False), ("horizon", 1.5), ("horizon", 0),
    ("minimum_validation_days", True), ("minimum_validation_days", 1.5),
    ("minimum_validation_days", 0), ("bootstrap_draws", True),
    ("bootstrap_draws", 1.5), ("bootstrap_draws", 0), ("seed", True),
    ("seed", 1.5),
])
def test_integer_arguments_reject_bool_fractional_and_nonpositive_values(name, bad_value):
    kwargs = dict(block_length=2, horizon=1, minimum_validation_days=1,
                  bootstrap_draws=5, seed=7, confidence_level=.95)
    kwargs[name] = bad_value
    with pytest.raises((TypeError, ValueError)):
        _api().moving_block_lower_bound(np.arange(12, dtype=float), **kwargs)


@pytest.mark.parametrize("confidence_level", [True, np.nan, np.inf, -np.inf, 0., 1., -0.1, 1.1])
def test_confidence_level_must_be_finite_and_strictly_between_zero_and_one(confidence_level):
    with pytest.raises((TypeError, ValueError)):
        _api().moving_block_lower_bound(
            np.arange(12, dtype=float), block_length=2, horizon=1,
            minimum_validation_days=1, bootstrap_draws=5, seed=7,
            confidence_level=confidence_level)


@pytest.mark.parametrize("differences", [np.ones((3, 4)), np.array([1., 2j]),
                                           np.array([True, False]), np.array(["1", "2"])])
def test_differences_must_be_one_dimensional_real_numeric(differences):
    with pytest.raises((TypeError, ValueError)):
        _api().moving_block_lower_bound(
            differences, block_length=2, horizon=1, minimum_validation_days=1,
            bootstrap_draws=5, seed=7, confidence_level=.95)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_exact_seeded_parity_preserves_real_numeric_input_dtype(dtype):
    if dtype is np.int64:
        raw = [1000000, 1, -1000000, 0, -3, 4, 0, 0, 0, 5, -2, 1]
    else:
        raw = [1e6, 1., -1e6, .5, -3., 4., np.nan, .25,
               -.75, 5., -2., 1.5]
    values = np.array(raw, dtype=dtype)
    _compare(values, block_length=3, horizon=2, minimum_validation_days=4,
             bootstrap_draws=23, seed=31, confidence_level=.9)

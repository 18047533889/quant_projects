import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.markov_dynamics import (
    TsMarkovPersistence,
    TsActiveInformationStorage,
    _bin,
    _equilibrium_distance_series,
    _quantile_edges,
    _quasipotential_depth_series,
    _run_kernel,
)


def test_quantile_states_survive_small_units_and_positive_affine_change():
    values = np.sin(np.linspace(0.0, 8.0, 180)) + 0.2 * np.cos(np.linspace(0.0, 17.0, 180))
    base = _bin(values, _quantile_edges(values, 5))
    tiny = values * 1e-13
    shifted = 1e6 + values
    assert np.array_equal(base, _bin(tiny, _quantile_edges(tiny, 5)))
    assert np.array_equal(base, _bin(shifted, _quantile_edges(shifted, 5)))
    constant = np.full(20, 1e-20)
    assert np.unique(_bin(constant, _quantile_edges(constant, 5))).size == 1


def test_equilibrium_exact_isolated_zero_and_plateau_are_stable_roots():
    series = np.array([-1.0])
    common = {
        "state": np.array([[0.0]]),
        "counts": np.array([[[3.0, 3.0, 3.0]]]),
        "centers": np.array([[[-1.0, 1.0, 2.0]]]),
        "D1": np.array([[[1.0 / 3.0, 0.0, -1.0]]]),
    }
    out = _equilibrium_distance_series(series, common, 0, 60, 3)
    assert out[0] == pytest.approx(-2.0)

    plateau = dict(common)
    plateau["centers"] = np.array([[[-2.0, 0.0, 2.0, 4.0]]])
    plateau["counts"] = np.array([[[3.0, 3.0, 3.0, 3.0]]])
    plateau["D1"] = np.array([[[1.0, 0.0, 0.0, -1.0]]])
    out2 = _equilibrium_distance_series(series, plateau, 0, 60, 3)
    assert out2[0] == pytest.approx(-2.0)


def test_quasipotential_uses_adjacent_basin_ridges():
    potential = np.array([10.0, 0.0, 2.0, 1.0, 3.0, 0.0, 12.0, 11.0])
    d2 = np.ones(8)
    d1 = np.zeros(8)
    d1[:-1] = -np.diff(potential) * (d2[:-1] + 1e-12)
    res = {
        "state": np.array([[1.0]]),
        "counts": np.array([[np.full(8, 3.0)]]),
        "D1": np.array([[d1]]),
        "D2": np.array([[d2]]),
        "centers": np.array([[np.arange(8, dtype=float)]]),
    }
    assert _quasipotential_depth_series(res, 0, 3)[0] == pytest.approx(2.0)


@pytest.mark.parametrize("bins, history, exact_window", [(2, 1, 21), (3, 2, 137)])
def test_ais_owns_grid_and_rejects_impossible_window_before_kernel(bins, history, exact_window):
    op = TsActiveInformationStorage()
    x = pd.DataFrame({"A": np.sin(np.arange(220, dtype=float))})
    with pytest.raises(ValueError, match="five observations per joint cell"):
        op.calculate(x, window=exact_window - 1, bins=bins, history_length=history)
    result = op.calculate(x, window=exact_window, bins=bins, history_length=history)
    assert result.shape == x.shape


def test_ais_rejects_markov_only_bin_counts():
    op = TsActiveInformationStorage()
    x = pd.DataFrame({"A": np.arange(200, dtype=float)})
    with pytest.raises(ValueError):
        op.calculate(x, window=180, bins=5, history_length=1)


def test_selected_kernel_outputs_match_full_without_quadratic_retention():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(90, 6)))
    selected_keys = {"state", "counts", "D1", "centers"}
    full = _run_kernel(x, 30, 5, 1, 3)
    selected = _run_kernel(x, 30, 5, 1, 3, outputs=selected_keys)
    assert set(selected) == selected_keys
    for key in selected_keys:
        np.testing.assert_allclose(selected[key], full[key], equal_nan=True)
    assert "P" not in selected and "N_obs" not in selected
    selected_bytes = sum(value.nbytes for value in selected.values())
    full_bytes = sum(value.nbytes for value in full.values())
    assert selected_bytes < full_bytes / 3


def test_selected_public_kernel_preserves_prefix_and_overlap_chunk_replay():
    rng = np.random.default_rng(11)
    x = pd.DataFrame({"A": rng.normal(size=180), "B": rng.normal(size=180)})
    op = TsMarkovPersistence()
    kwargs = dict(window=40, bins=3, lag=1, min_count=3, min_state_support=3, min_history=10)
    full = op.calculate(x, **kwargs)
    prefix = op.calculate(x.iloc[:125], **kwargs)
    pd.testing.assert_frame_equal(prefix, full.iloc[:125])

    split = 110
    overlap = kwargs["window"]
    chunk = op.calculate(x.iloc[split - overlap :], **kwargs)
    replay = chunk.iloc[overlap:].copy()
    pd.testing.assert_frame_equal(replay, full.iloc[split:])

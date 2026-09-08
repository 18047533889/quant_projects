from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.model_family_kernels import fit_pca_block
from factor_engine.cleaned_operators.cross_section.panel_model import (
    _moe_forecast,
    _pca_resid,
    last_fit_telemetry,
)
from factor_engine.cleaned_operators.cross_section.pca_state import PCAState


def _projection(state):
    return state.loadings.T @ state.loadings


def test_pca_nan_and_signed_infinity_share_one_training_missing_policy():
    base = np.array([
        [1.0, 4.0, 2.0],
        [2.0, np.nan, 4.0],
        [3.0, 8.0, 7.0],
        [5.0, 9.0, 11.0],
        [8.0, 12.0, 16.0],
    ])
    states = []
    for missing in (np.nan, np.inf, -np.inf):
        sample = base.copy()
        sample[1, 1] = missing
        state = PCAState.from_window(
            sample, 1, min_coverage_ratio=0.7, absolute_min_obs=2
        )
        assert state is not None
        states.append(state)

    reference = states[0]
    for state in states[1:]:
        np.testing.assert_allclose(state.mu, reference.mu, rtol=0, atol=0)
        np.testing.assert_allclose(state.sd, reference.sd, rtol=0, atol=0)
        np.testing.assert_allclose(
            _projection(state), _projection(reference), rtol=1e-12, atol=1e-12
        )


def test_pca_residual_restores_current_nonfinite_mask():
    training = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 3.0, 4.0],
        [3.0, 5.0, 8.0],
        [5.0, 8.0, 13.0],
    ])
    state = PCAState.from_window(training, 1)
    assert state is not None
    for missing in (np.nan, np.inf, -np.inf):
        row = np.array([8.0, missing, 21.0])
        residual = state.resid(row)
        assert np.isnan(residual[1])
        assert np.isfinite(residual[[0, 2]]).all()
        canonical_residual = _pca_resid(training, row, 1)
        assert np.isnan(canonical_residual[1])
        assert np.isfinite(canonical_residual[[0, 2]]).all()


def test_pca_constant_and_all_missing_columns_are_not_identifiable():
    training = np.array([
        [1.0, 7.0, np.nan],
        [2.0, 7.0, np.inf],
        [4.0, 7.0, -np.inf],
        [8.0, 7.0, np.nan],
    ])
    assert PCAState.from_window(training, 1) is None


def test_shared_pca_block_restores_current_nonfinite_mask():
    training = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 3.0, 4.0],
        [3.0, 5.0, 8.0],
        [5.0, 8.0, 13.0],
    ])
    block = fit_pca_block(training, 1)
    assert block is not None
    residual = block.resid(np.array([8.0, np.inf, 21.0]), 1)
    assert np.isnan(residual[1])
    assert np.isfinite(residual[[0, 2]]).all()


def test_shared_pca_block_nan_and_infinity_have_equal_finite_variance_state():
    base = np.array([
        [1.0, 4.0, 2.0],
        [2.0, np.nan, 4.0],
        [3.0, 8.0, 7.0],
        [5.0, 9.0, 11.0],
        [8.0, 12.0, 16.0],
    ])
    blocks = []
    for missing in (np.nan, np.inf, -np.inf):
        sample = base.copy()
        sample[1, 1] = missing
        block = fit_pca_block(sample, 1)
        assert block is not None
        assert np.isfinite(block.var_ret).all()
        blocks.append(block)
    for block in blocks[1:]:
        np.testing.assert_allclose(block.var_ret, blocks[0].var_ret, rtol=0, atol=0)
        np.testing.assert_allclose(
            block.commonality(np.ones(3), 1),
            blocks[0].commonality(np.ones(3), 1),
            rtol=1e-12,
            atol=1e-12,
        )


def test_moe_extreme_gate_is_stable_and_records_normalized_active_experts():
    n = 31
    index = pd.RangeIndex(n)
    x_values = np.linspace(-2.0, 2.0, n)
    market_values = np.linspace(0.0, 1.0, n)
    market_values[-1] = 1e300
    y = pd.DataFrame({"A": 1.5 + 2.0 * x_values}, index=index)
    x = pd.DataFrame({"A": x_values}, index=index)
    market = pd.DataFrame({"A": market_values}, index=index)

    result = _moe_forecast(y, (x,), market, window=30, n_experts=2)
    assert np.isfinite(result.iloc[-1, 0])
    telemetry = last_fit_telemetry()
    assert telemetry["fit_status"] == "ok"
    assert telemetry["all_invalid"] is False
    assert telemetry["active_experts"] == 2
    assert telemetry["weight_sum"] == 1.0


def test_moe_one_valid_expert_gets_unit_weight():
    n = 31
    index = pd.RangeIndex(n)
    market_values = np.linspace(0.0, 1.0, n)
    x_values = np.linspace(-2.0, 2.0, n)
    x_values[:15] = np.nan
    y_values = 1.5 + 2.0 * np.linspace(-2.0, 2.0, n)
    y = pd.DataFrame({"A": y_values}, index=index)
    x = pd.DataFrame({"A": x_values}, index=index)
    market = pd.DataFrame({"A": market_values}, index=index)

    result = _moe_forecast(y, (x,), market, window=30, n_experts=2)
    np.testing.assert_allclose(result.iloc[-1, 0], y_values[-1], rtol=1e-12, atol=1e-12)
    telemetry = last_fit_telemetry()
    assert telemetry["fit_status"] == "ok"
    assert telemetry["active_experts"] == 1
    assert telemetry["weight_sum"] == 1.0


def test_moe_all_invalid_is_missing_with_explicit_fit_status():
    n = 20
    index = pd.RangeIndex(n)
    market_values = np.linspace(0.0, 1.0, n)
    x = pd.DataFrame({"A": np.ones(n)}, index=index)
    y = pd.DataFrame({"A": np.linspace(1.0, 2.0, n)}, index=index)
    market = pd.DataFrame({"A": market_values}, index=index)

    result = _moe_forecast(y, (x,), market, window=19, n_experts=2)
    assert np.isnan(result.iloc[-1, 0])
    telemetry = last_fit_telemetry()
    assert telemetry["fit_status"] == "all_experts_invalid"
    assert telemetry["all_invalid"] is True
    assert telemetry["active_experts"] == 0
    assert telemetry["weight_sum"] == 0.0


def _moe_inputs(n=31):
    index = pd.RangeIndex(n)
    x_values = np.linspace(-2.0, 2.0, n)
    market_values = np.linspace(0.0, 1.0, n)
    y = pd.DataFrame({"A": 1.5 + 2.0 * x_values}, index=index)
    x = pd.DataFrame({"A": x_values}, index=index)
    market = pd.DataFrame({"A": market_values}, index=index)
    return y, x, market


def test_moe_debug_telemetry_tracks_final_nan_cell_and_new_call_identity():
    y, x, market = _moe_inputs()
    valid = _moe_forecast(y, (x,), market, window=30, n_experts=2)
    assert np.isfinite(valid.iloc[-1, 0])
    first = last_fit_telemetry()

    terminal_nan_market = market.copy()
    terminal_nan_market.iloc[-1, 0] = np.nan
    result = _moe_forecast(
        y, (x,), terminal_nan_market, window=30, n_experts=2
    )
    assert np.isnan(result.iloc[-1, 0])
    telemetry = last_fit_telemetry()
    assert telemetry["debug_only"] is True
    assert telemetry["fit_status"] == "current_market_state_nonfinite"
    assert telemetry["row"] == len(y) - 1
    assert telemetry["col"] == 0
    assert telemetry["call_id"] != first["call_id"]


def test_moe_debug_telemetry_resets_before_structural_error():
    y, x, market = _moe_inputs()
    _moe_forecast(y, (x,), market, window=30, n_experts=2)
    first = last_fit_telemetry()

    with pytest.raises(ValueError, match="market_state is a required"):
        _moe_forecast(y, (x,), None, window=30, n_experts=2)

    telemetry = last_fit_telemetry()
    assert telemetry["debug_only"] is True
    assert telemetry["fit_status"] == "missing_market_state"
    assert telemetry["call_id"] != first["call_id"]
    assert telemetry["row"] is None
    assert telemetry["col"] is None


def test_moe_debug_telemetry_is_isolated_across_threads():
    y, x, market = _moe_inputs()

    def run(terminal_nan):
        local_market = market.copy()
        if terminal_nan:
            local_market.iloc[-1, 0] = np.nan
        _moe_forecast(y, (x,), local_market, window=30, n_experts=2)
        return last_fit_telemetry()

    with ThreadPoolExecutor(max_workers=2) as pool:
        valid, terminal_nan = pool.map(run, (False, True))

    assert valid["fit_status"] == "ok"
    assert terminal_nan["fit_status"] == "current_market_state_nonfinite"
    assert valid["call_id"] != terminal_nan["call_id"]
    assert valid["row"] == terminal_nan["row"] == len(y) - 1
    assert valid["col"] == terminal_nan["col"] == 0

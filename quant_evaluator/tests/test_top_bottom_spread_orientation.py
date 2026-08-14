"""
Regression tests for QE2-P0-001: top-bottom spread orientation bug.

Bug: compute_top_bottom_spread() formerly defaulted to top_q=0, bottom_q=-1,
computing Low - High instead of High - Low.  This reversed the spread sign,
Sharpe ratio sign, and all directional factor judgments.

Fix: defaults are now top_q=None (→ n_quantiles-1, the highest group) and
bottom_q=0 (the lowest group).

Quantile encoding convention enforced here:
    quantile 0   = LOWEST factor value group
    quantile N-1 = HIGHEST factor value group
"""

import numpy as np
import pytest

from quant_evaluator.metrics.quantile import (
    compute_top_bottom_spread,
    compute_quantile_returns,
)
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quintile_returns(top_val: float, bottom_val: float) -> np.ndarray:
    """
    Build a minimal (T=3, n_quantiles=5, F=1) quantile_returns array where
    quantile 4 (highest group) returns top_val and quantile 0 (lowest group)
    returns bottom_val.  Middle quantiles are set to 0.0.
    """
    T, Q, F = 3, 5, 1
    q_ret = np.zeros((T, Q, F), dtype=np.float64)
    q_ret[:, 4, :] = top_val    # highest group
    q_ret[:, 0, :] = bottom_val  # lowest group
    return q_ret


def _make_decile_returns(top_val: float, bottom_val: float) -> np.ndarray:
    """Same as above but for deciles (Q=10)."""
    T, Q, F = 3, 10, 1
    q_ret = np.zeros((T, Q, F), dtype=np.float64)
    q_ret[:, 9, :] = top_val    # highest group (quantile 9)
    q_ret[:, 0, :] = bottom_val  # lowest group (quantile 0)
    return q_ret


# ---------------------------------------------------------------------------
# Test 1: Default parameters produce High - Low (positive for positive factor)
# ---------------------------------------------------------------------------

def test_default_params_spread_orientation():
    """
    QE2-P0-001: Default call must return top_group - bottom_group > 0
    when the top group outperforms the bottom group.

    Before the fix, default top_q=0 pointed to the LOWEST group, making the
    spread negative for a positive factor.
    """
    q_ret = _make_quintile_returns(top_val=0.05, bottom_val=-0.03)
    spread = compute_top_bottom_spread(q_ret)  # uses defaults

    assert spread.shape == (3, 1), f"Expected (3,1), got {spread.shape}"
    expected = 0.05 - (-0.03)  # 0.08
    np.testing.assert_allclose(
        spread,
        expected,
        atol=1e-12,
        err_msg=(
            "QE2-P0-001 FAIL: default spread is wrong. "
            f"Got {spread[0, 0]:.6f}, expected {expected:.6f}. "
            "Old bug would have returned -0.08 (Low - High)."
        ),
    )


# ---------------------------------------------------------------------------
# Test 2: Spread > 0 for a positively predictive factor (end-to-end)
# ---------------------------------------------------------------------------

def test_positive_factor_spread_is_positive():
    """
    QE2-P0-001: For a factor where higher value predicts higher return,
    the default spread (top - bottom) must be strictly positive.

    Construct factor values = [0,1,2,...,N-1] so quantile 0 gets the lowest
    values and quantile 4 gets the highest.  Labels are set equal to factor
    values, guaranteeing top quantile mean > bottom quantile mean.
    """
    np.random.seed(0)
    T, N = 10, 50  # 50 assets → 10 per quintile

    # Factor: strictly increasing 0..N-1 repeated over T periods
    factor = np.tile(np.arange(N, dtype=float), (T, 1))  # (T, N)
    factor_3d = factor[:, :, np.newaxis]  # (T, N, 1)

    # Labels = factor value → positive monotone relationship
    labels = factor.copy()  # (T, N)

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    batch = FactorBatch(
        factor_ids=("factor_0",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_3d,
    )
    bundle = LabelBundle(
        target_id="ret_1d",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )

    q_returns, q_counts = compute_quantile_returns(batch, bundle, n_quantiles=5, min_assets=5)
    spread = compute_top_bottom_spread(q_returns)  # (T, 1)

    mean_spread = np.nanmean(spread)
    assert mean_spread > 0, (
        f"QE2-P0-001 FAIL: positive factor produced negative spread {mean_spread:.6f}. "
        "Sign inversion indicates the old bug is still present."
    )

    # Also verify top quantile mean > bottom quantile mean in the raw returns
    top_mean = np.nanmean(q_returns[:, -1, 0])
    bot_mean = np.nanmean(q_returns[:, 0, 0])
    assert top_mean > bot_mean, (
        f"Quantile 4 mean ({top_mean:.4f}) should exceed quantile 0 mean ({bot_mean:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 3: Explicit top_q / bottom_q parameters work correctly
# ---------------------------------------------------------------------------

def test_explicit_top_q_bottom_q():
    """
    QE2-P0-001: Explicit top_q=4, bottom_q=0 must give same result as defaults.
    And explicit top_q=0, bottom_q=4 must give the negated result (old bug value).
    """
    q_ret = _make_quintile_returns(top_val=0.06, bottom_val=-0.02)

    spread_default = compute_top_bottom_spread(q_ret)
    spread_explicit = compute_top_bottom_spread(q_ret, top_q=4, bottom_q=0)
    spread_inverted = compute_top_bottom_spread(q_ret, top_q=0, bottom_q=4)

    np.testing.assert_allclose(
        spread_default, spread_explicit, atol=1e-12,
        err_msg="Explicit top_q=4, bottom_q=0 should equal default.",
    )
    np.testing.assert_allclose(
        spread_default, -spread_inverted, atol=1e-12,
        err_msg="Inverted top_q=0, bottom_q=4 should negate the default spread.",
    )

    # Confirm the default (correct) sign
    assert np.all(spread_default > 0), "Spread should be positive when top > bottom"
    assert np.all(spread_inverted < 0), "Inverted spread should be negative"


# ---------------------------------------------------------------------------
# Test 4: Quintile vs decile — n_quantiles auto-detection
# ---------------------------------------------------------------------------

def test_quintile_vs_decile_auto_top():
    """
    QE2-P0-001: With top_q=None the function must auto-detect n_quantiles-1
    for both quintiles (n=5) and deciles (n=10).
    """
    top_val, bottom_val = 0.10, -0.05
    expected_spread = top_val - bottom_val  # 0.15

    spread_quintile = compute_top_bottom_spread(_make_quintile_returns(top_val, bottom_val))
    spread_decile = compute_top_bottom_spread(_make_decile_returns(top_val, bottom_val))

    np.testing.assert_allclose(spread_quintile, expected_spread, atol=1e-12,
                               err_msg="Quintile spread mismatch")
    np.testing.assert_allclose(spread_decile, expected_spread, atol=1e-12,
                               err_msg="Decile spread mismatch")

    # Prove the top index differs: quintile uses col 4, decile uses col 9
    assert _make_quintile_returns(top_val, bottom_val).shape[1] == 5
    assert _make_decile_returns(top_val, bottom_val).shape[1] == 10


# ---------------------------------------------------------------------------
# Test 5: Negative-index backward compatibility (-1 still works as top)
# ---------------------------------------------------------------------------

def test_negative_index_top_q_still_valid():
    """
    QE2-P0-001: Passing top_q=-1 (Python negative index for last element)
    must produce the same result as top_q=None (auto) or top_q=n_quantiles-1.

    Test_multi_factor_evaluation.py already used top_q=-1, bottom_q=0 as a
    workaround for the old bug.  That workaround must remain correct.
    """
    q_ret = _make_quintile_returns(top_val=0.04, bottom_val=-0.01)

    spread_auto = compute_top_bottom_spread(q_ret)             # top_q=None
    spread_neg1 = compute_top_bottom_spread(q_ret, top_q=-1)   # negative index
    spread_pos4 = compute_top_bottom_spread(q_ret, top_q=4)    # explicit

    np.testing.assert_allclose(spread_auto, spread_neg1, atol=1e-12,
                               err_msg="top_q=-1 must equal top_q=None (auto)")
    np.testing.assert_allclose(spread_auto, spread_pos4, atol=1e-12,
                               err_msg="top_q=-1 must equal top_q=4 for quintiles")

    assert np.all(spread_auto > 0), "All three variants should be positive"


# ---------------------------------------------------------------------------
# Test 6: Zero spread when top and bottom groups have equal returns
# ---------------------------------------------------------------------------

def test_zero_spread_when_groups_equal():
    """
    QE2-P0-001: When top group and bottom group have identical returns,
    spread must be exactly zero regardless of the direction fix.
    """
    T, Q, F = 4, 5, 2
    q_ret = np.full((T, Q, F), 0.03, dtype=np.float64)  # all quantiles same

    spread = compute_top_bottom_spread(q_ret)
    np.testing.assert_allclose(spread, 0.0, atol=1e-12,
                               err_msg="Equal group returns must yield zero spread")


# ---------------------------------------------------------------------------
# Test 7: Multi-factor shape check
# ---------------------------------------------------------------------------

def test_multi_factor_shape_preserved():
    """
    QE2-P0-001: Shape (T, F) is preserved for multi-factor input.
    """
    T, Q, F = 6, 5, 4
    q_ret = np.random.RandomState(99).randn(T, Q, F)

    spread = compute_top_bottom_spread(q_ret)

    assert spread.shape == (T, F), (
        f"Expected spread shape (T={T}, F={F}), got {spread.shape}"
    )
    # Verify: spread[t, f] == q_ret[t, Q-1, f] - q_ret[t, 0, f]
    expected = q_ret[:, Q - 1, :] - q_ret[:, 0, :]
    np.testing.assert_allclose(spread, expected, atol=1e-12)


if __name__ == "__main__":
    print("Running QE2-P0-001 top-bottom spread orientation regression tests...\n")

    test_default_params_spread_orientation()
    print("  PASS  test_default_params_spread_orientation")

    test_positive_factor_spread_is_positive()
    print("  PASS  test_positive_factor_spread_is_positive")

    test_explicit_top_q_bottom_q()
    print("  PASS  test_explicit_top_q_bottom_q")

    test_quintile_vs_decile_auto_top()
    print("  PASS  test_quintile_vs_decile_auto_top")

    test_negative_index_top_q_still_valid()
    print("  PASS  test_negative_index_top_q_still_valid")

    test_zero_spread_when_groups_equal()
    print("  PASS  test_zero_spread_when_groups_equal")

    test_multi_factor_shape_preserved()
    print("  PASS  test_multi_factor_shape_preserved")

    print("\nAll QE2-P0-001 regression tests passed.")

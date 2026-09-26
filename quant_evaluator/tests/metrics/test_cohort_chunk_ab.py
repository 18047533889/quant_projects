"""Cohort PnL full-output goldens from the pre-chunk scalar daily loop.

The goldens were captured on 2026-09-26 from the unmodified, committed
compute_cohort_pnl implementation. Each digest includes every returned array,
its key, shape, dtype, and exact bytes. The scenarios exercise H=1, a
truncated tail, a holding window longer than the 32-day chunk, missing
returns/prices, explicit eligibility, costs, and both terminal policies.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.portfolio_inputs import TradeEligibilityPanel
from quant_evaluator.metrics.probe_portfolio._core import compute_cohort_pnl


_CASES = (
    ("short", 9, 8, 1, False, "ongoing", True,
     "80d6dd78f0de8474fdb358ea707a9c81fa337f823f55ac19bc00db7420a603f1"),
    ("tail", 9, 8, 20, False, "liquidate_at_end", True,
     "b899e4084dd491d35cd7a4b24e7aaa4f1e7a46c879ddd964da3c147dc21627b7"),
    ("chunk", 75, 50, 40, False, "ongoing", True,
     "395340087b7f6310a490d2f269f1876c63f07e9043649088f1800822ba454122"),
    ("eligible", 75, 50, 20, True, "liquidate_at_end", True,
     "7757a691cec925493c6c7e83371caa62de9d692e8edd6661263ca368837df36d"),
    ("no_tradable", 41, 25, 7, False, "liquidate_at_end", False,
     "14c2b4ee27dbd59443d7c0f166c9cd4ebe4f3f7db6b669e983dda8330e31ef42"),
)


def _digest(result):
    digest = hashlib.sha256()
    for key in sorted(result):
        values = np.ascontiguousarray(result[key])
        digest.update(key.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    "name,t,n,holding,eligible,terminal_policy,require_tradable,expected",
    _CASES,
)
def test_chunked_cohort_matches_scalar_loop_golden(
    name, t, n, holding, eligible, terminal_policy, require_tradable, expected,
):
    rng = np.random.default_rng(20260926 + t + n + holding)
    factor = rng.normal(size=(t, n))
    returns = rng.normal(scale=0.01, size=(t, n))
    vwap = np.ones((t, n))
    opening = np.ones((t, n))
    weights = rng.random((t, n))
    factor[rng.random((t, n)) < 0.03] = np.nan
    returns[rng.random((t, n)) < 0.03] = np.nan
    vwap[rng.random((t, n)) < 0.02] = np.nan
    opening[rng.random((t, n)) < 0.02] = np.nan
    weights[rng.random((t, n)) < 0.02] = np.nan

    permissions = None
    if eligible:
        time_axis = AxisRef("t", "int", t, np.arange(t))
        asset_axis = AxisRef(
            "a", "str", n, tuple(f"s{i}" for i in range(n)),
        )
        buy = rng.random((t, n)) > 0.12
        sell = np.ones((t, n), dtype=bool)
        borrow = rng.random((t, n)) > 0.12
        cover = np.ones((t, n), dtype=bool)
        permissions = TradeEligibilityPanel(
            buy, sell, borrow, cover, np.zeros((t, n), dtype=int),
            time_axis, asset_axis, "test", "test",
        )

    result = compute_cohort_pnl(
        factor, returns, vwap,
        next_open=opening,
        weight_matrix=weights,
        n_quantiles=2,
        holding=holding,
        long_weight=0.6,
        short_weight=-0.4,
        per_side_cost=0.003,
        min_bucket_size=2,
        require_tradable=require_tradable,
        trade_eligibility=permissions,
        terminal_position_policy=terminal_policy,
    )
    assert _digest(result) == expected, name


def test_missing_held_return_marks_day_unknown_after_other_cohorts_add():
    # At t=3, the day-0 cohort owns asset 3, whose return is missing.
    # The day-1 cohort owns assets 0/1 and contributes a finite return later
    # in the loop. Both pnl and active return must remain NaN.
    factor = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (8, 1))
    factor[1] = np.array([3.0, 2.0, 1.0, 0.0])
    returns = np.zeros((8, 4))
    returns[3, 3] = np.nan
    result = compute_cohort_pnl(
        factor, returns, np.ones((8, 4)),
        n_quantiles=2, holding=3,
        long_weight=1.0, short_weight=0.0,
    )
    assert np.isnan(result["pnl_net"][3])
    assert np.isnan(result["active_ret"][3])
    assert np.isfinite(result["pnl_net"][2])
    assert result["gross_exposure"][3] > 0

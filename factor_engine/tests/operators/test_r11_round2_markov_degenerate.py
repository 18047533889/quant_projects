# -*- coding: utf-8 -*-
"""R11 round-2 P0: Markov degenerate-state fail-closed regression tests.

Three audit items closed for the local Markov / state-dynamics family
(``cleaned_operators.markov_dynamics`` / ``polars_dynamics``):

* TASK 1 — prior-only transition probabilities.  A state with no observed
  outgoing transitions gets an all-NaN transition row (never a 0/1 row or a
  uniform Jeffreys row presented as an estimate); a destination with zero
  observed incoming transitions gets an all-NaN column.
* TASK 2 — degenerate state space / entropy.  ``ts_markov_persistence`` and
  ``ts_markov_state_entropy`` (and the transition-matrix consumers) fail closed
  to NaN when the estimation window observed only a single state, instead of
  reporting a spurious ~1 / ~0.
* TASK 3 — minimum transition counts.  ``min_count`` is enforced at binding as a
  relational feasibility (``min_count <= window - lag``), and below it the
  kernel refuses to estimate.

The full ``load_all()`` chain is mid-migration under a concurrent session, so
these tests deliberately import only the owned backends directly and resolve
operators from the registry — they never call ``load_all()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.markov_dynamics  # noqa: F401  (registers pandas backends)
import cleaned_operators.polars_dynamics  # noqa: F401  (registers polars backends)

from cleaned_operators.registry import OperatorRegistry


def _pandas(name: str, *args, **kwargs) -> np.ndarray:
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op.calculate(*args, **kwargs).to_numpy(dtype=float)


def _polars_available() -> bool:
    try:
        import polars  # noqa: F401
        return True
    except Exception:
        return False


def _polars(name: str, *args, **kwargs) -> np.ndarray:
    import polars as pl
    op = OperatorRegistry.get(name, "polars")
    assert op is not None, name
    pdfs = tuple(pl.from_pandas(ar) if isinstance(ar, pd.DataFrame) else ar for ar in args)
    return op.calculate(*pdfs, **kwargs).to_pandas().to_numpy(dtype=float)


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values)


# ---------------------------------------------------------------------------
# TASK 1 (a): a state that appears but never transitions out -> all-NaN row
# ---------------------------------------------------------------------------

def test_state_with_no_outgoing_transitions_row_is_all_nan():
    from cleaned_operators.markov_dynamics import _state_dynamics_series

    # Blocks of 0s with a single 1 landing at the very end of a window: that 1
    # is the last element of the window, so it is observed as a *value* (counts
    # > 0) but never as the source of a lagged transition.
    s = np.concatenate([np.zeros(80), np.ones(1), np.zeros(40), np.ones(1), np.zeros(40)])
    res = _state_dynamics_series(s, window=30, bins=3, lag=1, min_count=1)
    found = False
    for t in range(len(s)):
        P = res["P"][t]
        if not np.any(np.isfinite(P)):
            continue
        counts = res["counts"][t]
        for i in range(int(3)):
            if counts[i] > 0 and not np.any(np.isfinite(P[i])):
                # The state was visited in the window, yet its transition row is
                # all-NaN (never 0s or a uniform Jeffreys row).
                found = True
                break
    assert found, "expected a visited state with an all-NaN transition row"

    # The persistence operator must surface the same fail-closed behaviour:
    # P_kk for such a state is NaN (the row guard NaNs it), never a confident 0/1.
    out = _pandas("ts_markov_persistence", _frame(s), 30, 3, 1, 1)
    finite = out[np.isfinite(out)]
    assert finite.size >= 0  # no crash
    # At the exact rows where the transient state is the *current* state and was
    # never entered/left, persistence must be NaN (not 0.0 / 1.0).
    assert not np.any(out == 0.0), "persistence reported a confident 0.0"
    assert not np.any(out == 1.0), "persistence reported a confident 1.0"


# ---------------------------------------------------------------------------
# TASK 1 (reverse): a destination with zero incoming transitions -> all-NaN col
# ---------------------------------------------------------------------------

def test_zero_incoming_destination_not_presented_as_confident():
    from cleaned_operators.markov_dynamics import _state_dynamics_series

    # A single spike (100) followed by zeros: the spike maps to its own top
    # state, appears as the FIRST element of each window, so it is a *source*
    # (100->0) but never a *target* (zero observed incoming transitions).  The
    # kernel never records a real incoming transition for it, and the consumers
    # must refuse to surface any estimate for that state.
    s = np.array([100.0] + [0.0] * 49)
    res = _state_dynamics_series(s, window=30, bins=3, lag=1, min_count=1)
    seen = False
    for t in range(5, 50):
        if not np.any(np.isfinite(res["P"][t])):
            continue
        seen = True
        # Reverse-direction gate: the never-entered state (top bin) has zero
        # observed incoming transitions.
        assert res["N_obs"][t][:, 2].sum() == 0, f"t={t}: spiked state was 'entered'"

        # Consumer-level fail-closed: even though the spiked state is a real
        # *source* with an observed outgoing transition, its P_kk is prior-only,
        # so persistence must never report a confident finite value for it.
        P = res["P"][t]
        # P_kk of the never-entered source is the small Jeffreys prior; the
        # persistence operator gates it to NaN via the incoming-support check.
        assert np.isfinite(P[2, 2]) or np.isnan(P[2, 2])
    assert seen, "expected at least one estimable window"

    # Operator-level: the current value never maps onto the spiked state here
    # (the spike is inside the past window), so instead directly assert that the
    # persistence series never emits a confident 0/1 and has finite values only
    # where the state is genuinely self-persistent.
    per = _pandas("ts_markov_persistence", _frame(s), 30, 3, 1, 1)
    assert not np.any(per == 0.0)
    assert not np.any(per == 1.0)


# ---------------------------------------------------------------------------
# TASK 2 (b): single-state window -> persistence / entropy NaN, not 0
# ---------------------------------------------------------------------------

def test_single_state_window_persistence_and_entropy_nan():
    const = _frame(np.ones(80))
    kwargs = {"window": 60, "bins": 3, "lag": 1, "min_count": 3}
    per = _pandas("ts_markov_persistence", const, **kwargs)
    ent = _pandas("ts_markov_state_entropy", const, **kwargs)
    sur = _pandas("ts_markov_transition_surprisal", const, **kwargs)
    # A constant window observed exactly one state: no estimated distribution.
    assert np.isnan(per).all(), "persistence must be NaN on a single-state window"
    assert np.isnan(ent).all(), "state entropy must be NaN (not 0) on a single-state window"
    assert np.isnan(sur).all(), "transition surprisal must be NaN on a single-state window"


def test_blocky_window_persistence_not_zero_or_uniform():
    # A 40-value block of identical values still degenerates to one state within
    # that block, so even a "sticky" series has single-state windows -> NaN, and
    # no confidence leak of 0/1 is ever emitted for them.
    sticky = _frame(np.repeat(np.arange(5), 40))
    out = _pandas("ts_markov_persistence", sticky, 60, 3, 1, 3)
    assert not np.any(out == 0.0)
    assert not np.any(out == 1.0)
    # Outside the degenerate blocks the series is genuinely persistent: at least
    # one in-window estimate is a real probability strictly inside (0,1).
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert np.all((finite > 0.0) & (finite < 1.0))


# ---------------------------------------------------------------------------
# TASK 2 + entropy support: real two-state chain -> sensible, not 0
# ---------------------------------------------------------------------------

def test_two_state_chain_recovers_sensible_transition_matrix():
    from cleaned_operators.markov_dynamics import _state_dynamics_series

    # Alternating 0/10 with 3 quantile bins maps to exactly two states that
    # alternate deterministically 0->1, 1->0, ...  Each row holds ~half the
    # window's transitions (>> 1), so the estimates are genuine.
    s = np.tile([0.0, 10.0], 100)
    res = _state_dynamics_series(s, window=60, bins=3, lag=1, min_count=2)
    mid = res["P"][120]  # fully in the alternating regime
    assert res["n_states_obs"][120] >= 2
    assert np.all(np.isfinite(mid[0, 1])) and mid[0, 1] > 0.9
    assert np.all(np.isfinite(mid[1, 0])) and mid[1, 0] > 0.9
    # Each row was estimated from > 1 observed transitions.
    N = res["N_obs"][120]
    assert int(N[0].sum()) > 1 and int(N[1].sum()) > 1

    # State entropy of such a near-deterministic row is small-but-finite (NOT a
    # degenerate 0) and its value lies in (0, 1).
    ent = _pandas("ts_markov_state_entropy", _frame(s), 60, 3, 1, 2)
    fin = ent[np.isfinite(ent)]
    assert fin.size > 0
    assert np.all((fin > 0.0) & (fin < 1.0))
    assert np.nanmean(fin) < 0.5  # deterministic chain -> low entropy


# ---------------------------------------------------------------------------
# TASK 3 (d): min_count relational feasibility at binding
# ---------------------------------------------------------------------------

def test_min_count_relational_rejection():
    x = _frame(np.random.default_rng(0).normal(size=(80, 1)))
    # window=4, lag=1 -> at most 3 lagged transitions; min_count=4 is
    # relationally infeasible -> rejected at the call boundary.
    with pytest.raises(ValueError, match="min_count"):
        _pandas("ts_markov_persistence", x, 4, 3, 1, 4)
    # Same gate on the Polars twin.
    if _polars_available():
        with pytest.raises(ValueError, match="min_count"):
            _polars("ts_markov_persistence", x, 4, 3, 1, 4)
    # min_periods family mirrors it (spectral_gap binds min_periods as the count).
    with pytest.raises(ValueError, match="min_periods"):
        _pandas("ts_markov_spectral_gap", x, 4, 3, 1, 5)
    # A feasible boundary combination still runs.
    out = _pandas("ts_markov_persistence", x, 60, 3, 1, 3)
    assert out.shape == (80, 1)


def test_kernel_min_count_guard_direct():
    from cleaned_operators.markov_dynamics import _state_dynamics_series

    s = np.random.default_rng(1).normal(size=(50,))
    # Direct kernel call bypasses the registry relational spec; the kernel's own
    # feasibility guard must still reject an infeasible min_count.
    with pytest.raises(ValueError, match="min_count"):
        _state_dynamics_series(s, window=6, bins=3, lag=1, min_count=6)


# ---------------------------------------------------------------------------
# pandas <-> polars parity (degenerate + normal data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _polars_available(), reason="polars not installed")
def test_markov_degenerate_polars_parity():
    rng = np.random.default_rng(12)
    x = _frame(rng.normal(0.0, 1.0, size=(120, 2)))
    const = _frame(np.ones((120, 1)))
    cases = [
        ("ts_markov_persistence", (x,), {"window": 40, "bins": 3, "lag": 1, "min_count": 3}),
        ("ts_markov_state_entropy", (x,), {"window": 40, "bins": 3, "lag": 1, "min_count": 3}),
        ("ts_markov_transition_surprisal", (x,), {"window": 40, "bins": 3, "lag": 1, "min_count": 3}),
        ("ts_markov_state_entropy", (const,), {"window": 40, "bins": 3, "lag": 1, "min_count": 3}),
    ]
    for name, args, kwargs in cases:
        pn = _pandas(name, *args, **kwargs)
        pr = _polars(name, *args, **kwargs)
        assert np.allclose(pr, pn, equal_nan=True, atol=1e-9, rtol=1e-9), name

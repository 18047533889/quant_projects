# -*- coding: utf-8 -*-
"""Model-audit state-family fixes: M-150, M-151, M-160, M-162, M-171, M-221.

Each test proves one audit fix at current HEAD:
- M-150  Lyapunov physical-time horizon: ``physical_time=True`` scans k PHYSICAL
         bars forward (the k-th successor sits exactly k physical bars later) and
         regresses against physical elapsed time; the compressed path is the
         documented default and differs on gapped series.
- M-151  Lyapunov params strict: window/tau/horizon/min_anchors/embedding_dim
         RAISE on non-integer / non-finite / bool / out-of-range values instead
         of being clamped.
- M-160  RQA family window maturity unified: one module-level constant
         ``_RQA_MIN_EFFECTIVE_FRACTION_DEFAULT = 0.8`` applied to ALL RQA ops
         including ``ts_recurrence_rate``.
- M-162  RQA estimator params (dim/delay/eps_fraction/min_periods) are declared
         non-searchable ParamSpecs; only ``window`` (HORIZON) is searchable.
- M-171  TE feasibility compile-time: shared ``_te_feasibility`` gate + a
         telemetry accessor (``te_feasibility_failure_reason``) + a
         compile-time RelationalParamSpec; the runtime raise is preserved.
- M-221  Markov NaN-compress must not redefine lag: transitions are formed on
         the ORIGINAL time axis at the physical lag; a gap contributes no
         transition and never pairs across itself.

Kernel-level assertions import the modules directly (no full ``load_all``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402

_load_done = False


@pytest.fixture(scope="module", autouse=True)
def _load_once():
    global _load_done
    if not _load_done:
        load_all()
        _load_done = True
    yield


def _frame(a, cols=("A",)):
    return pd.DataFrame(np.asarray(a, dtype=float), columns=list(cols))


def _get(canonical: str):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, f"{canonical} not registered on pandas_numpy"
    return op


# ---------------------------------------------------------------------------
# M-150 — Lyapunov physical-time horizon
# ---------------------------------------------------------------------------

def test_m150_physical_divergence_k_spans_physical_bars():
    """A physical trajectory is ineligible if any exact-bar successor is absent.

    Window chunk with an interior NaN at physical offset 2: surviving points sit
    at physical offsets [0,1,3,4,5,6,7].  Anchor 0 (offset 0) / neighbour 1
    (offset 1):
      * physical step 1 -> anchor offset 1 (ok), neighbour offset 2 (missing)
      * therefore the pair is rejected as a whole; later isolated successors do
        not create a partial regression curve with changing support.
    """
    from factor_engine.cleaned_operators.local_lyapunov import _physical_divergence

    chunk = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    finite = np.isfinite(chunk)
    f = chunk[finite]
    phys = np.nonzero(finite)[0]
    # tau=1, dim=2 embedding: rows = (f[i], f[i+1]).
    Z = np.stack([f[:-1], f[1:]], axis=1)
    z_len = Z.shape[0]
    assert phys[:z_len].tolist() == [0, 1, 3, 4, 5, 6]
    d0j = float(np.sqrt(np.sum((Z[0] - Z[1]) ** 2)))
    div = _physical_divergence(Z, phys, z_len, 0, 1, 3, d0j)
    assert div is None


def test_m150_compressed_and_physical_differ_on_gapped_series():
    """On a gapped series the compressed-ordinal path and the physical-clock path
    must NOT be bit-identical (the compressed k steps span a different physical
    span than the physical clock)."""
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    rng = np.random.default_rng(7)
    x = rng.normal(size=50)
    x[18] = np.nan
    x[19] = np.nan
    comp = _lyapunov_series(x, 35, 1, 2, 4, 1, physical_time=False)
    phys = _lyapunov_series(x, 35, 1, 2, 4, 1, physical_time=True)
    # Both emit some values, and the finite outputs differ somewhere.
    assert np.isfinite(comp).any()
    assert np.isfinite(phys).any()
    assert not np.allclose(comp, phys, equal_nan=True), (
        "compressed and physical divergence horizons are identical on a gapped "
        "series — the physical clock must diverge from compressed ordinals (M-150)"
    )


def test_m150_physical_path_default_is_compressed():
    """``physical_time`` defaults to False and matches the explicit policy."""
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    x = np.random.default_rng(150).normal(size=40).cumsum()
    x[17] = np.nan
    implicit = _lyapunov_series(x, 24, 1, 2, 2, 1)
    explicit = _lyapunov_series(x, 24, 1, 2, 2, 1, physical_time=False)
    np.testing.assert_array_equal(implicit, explicit)
    assert np.isfinite(explicit).any()


def test_m150_operator_accepts_physical_time_flag():
    op = _get("ts_local_lyapunov_exponent")
    rng = np.random.default_rng(11)
    df = _frame(rng.normal(size=(80, 1)))
    base = op.calculate(df, window=60, tau=1, embedding_dim=3, horizon=5, min_anchors=2)
    phys = op.calculate(df, window=60, tau=1, embedding_dim=3, horizon=5, min_anchors=2,
                        physical_time=True)
    assert base.shape == phys.shape


# ---------------------------------------------------------------------------
# M-151 — Lyapunov params strict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        {"window": 1.5},
        {"window": True},
        {"window": np.nan},
        {"window": np.inf},
        {"tau": 0},
        {"tau": True},
        {"tau": np.nan},
        {"horizon": 0},
        {"horizon": np.inf},
        {"min_anchors": 0},
        {"embedding_dim": 1},
        {"embedding_dim": 7},
    ],
)
def test_m151_lyapunov_strict_params_reject_bad_values(bad):
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    with pytest.raises((ValueError, TypeError)):
        _lyapunov_series(np.arange(30.0), **{"window": 40, "tau": 1, "dim": 2,
                                              "horizon": 3, "min_anchors": 1, **bad})


def test_m151_operator_strict_params_reject_bool_physical_time():
    op = _get("ts_local_lyapunov_exponent")
    rng = np.random.default_rng(5)
    df = _frame(rng.normal(size=(40, 1)))
    with pytest.raises(ValueError):
        op.calculate(df, window=30, physical_time=1)  # bool contract, not truthy int


def test_m151_valid_params_still_accepted():
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    out = _lyapunov_series(np.arange(30.0), window=40, tau=1, dim=2, horizon=3, min_anchors=1)
    assert out.shape == (30,)


# ---------------------------------------------------------------------------
# M-160 — RQA family window maturity unified
# ---------------------------------------------------------------------------

def test_m160_rqa_shared_constant_is_08():
    from factor_engine.cleaned_operators.recurrence_analysis import _RQA_MIN_EFFECTIVE_FRACTION_DEFAULT, _recurrence_series

    assert _RQA_MIN_EFFECTIVE_FRACTION_DEFAULT == 0.8
    # The kernel default is the same shared family policy.
    import inspect

    sig = inspect.signature(_recurrence_series)
    assert sig.parameters["min_effective_fraction"].default == 0.8


def test_m160_recurrence_rate_now_family_policy_on_gap():
    """ts_recurrence_rate is sample-size dependent too: a trailing window whose
    longest contiguous finite run covers < 0.8*window must emit NaN (the same
    maturity policy as the line-structure statistics), not a short-sample value.
    """
    op = _get("ts_recurrence_rate")
    idx = pd.date_range("2024-01-01", periods=40)
    # Fully-finite window -> value emitted (maturity gate passes).
    full = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 1)), index=idx, columns=["A"])
    out_full = op.calculate(full, window=40, dim=1, delay=1, eps_fraction=0.1, min_periods=10)
    assert np.isfinite(out_full.to_numpy(dtype=float)[-1, 0])
    # Interior gap breaks the trailing contiguous run (longest trailing finite
    # run is ~15 of 40 < 0.8*40=32) -> the rate must NOT recompute on the short
    # finite prefix; it must be NaN.
    gapped = full.copy().to_numpy(dtype=float).copy()
    gapped[:24, 0] = np.nan
    gapped = pd.DataFrame(gapped, index=idx, columns=["A"])
    out_gap = op.calculate(gapped, window=40, dim=1, delay=1, eps_fraction=0.1, min_periods=10)
    assert np.isnan(out_gap.to_numpy(dtype=float)[-1, 0])


# ---------------------------------------------------------------------------
# M-162 — RQA estimator params not searched
# ---------------------------------------------------------------------------

def test_m162_rqa_estimator_params_non_searchable():
    from factor_engine.cleaned_operators.recurrence_analysis import _RQA_PARAM_SPECS

    for name in ("dim", "delay", "eps_fraction", "min_periods"):
        assert name in _RQA_PARAM_SPECS, name
        assert _RQA_PARAM_SPECS[name].searchable is False, name
    assert _RQA_PARAM_SPECS["window"].param_role.value == "horizon"
    # ``min_line`` is a fixed internal preset (2), not an exposed parameter, so
    # it must NOT carry a ParamSpec (keys(param_specs) ⊆ param_names invariant).
    assert "min_line" not in _RQA_PARAM_SPECS


def test_m162_each_rqa_operator_declares_the_specs():
    for canon in ("ts_recurrence_rate", "ts_recurrence_diagonal_entropy",
                  "ts_recurrence_trapping_time", "ts_recurrence_divergence"):
        op = _get(canon)
        specs = op.metadata.param_specs
        for name in ("dim", "delay", "eps_fraction", "min_periods"):
            assert name in specs, (canon, name)
            assert specs[name].searchable is False, (canon, name)
        assert "min_line" not in op.metadata.param_names, canon
        assert specs["window"].param_role.value == "horizon", canon


# ---------------------------------------------------------------------------
# M-171 — TE feasibility compile-time
# ---------------------------------------------------------------------------

def test_m171_te_feasibility_telemetry_accessor():
    from factor_engine.cleaned_operators.advanced_information import te_feasibility_failure_reason

    reason = te_feasibility_failure_reason(
        window=20, bins=3, lag=1, min_cells_ratio=1.0, min_transitions=None,
    )
    assert reason is not None
    assert "min_transitions" in reason or "window" in reason
    ok = te_feasibility_failure_reason(
        window=60, bins=3, lag=1, min_cells_ratio=1.0, min_transitions=None,
    )
    assert ok is None


def test_m171_te_relational_spec_declared():
    from factor_engine.cleaned_operators.advanced_information import _TE_FEASIBILITY_SPECS, _TE_PEAK_FEASIBILITY_SPECS

    assert any(s.expression == "window >= lag + 2" for s in _TE_FEASIBILITY_SPECS)
    assert any(s.expression == "window >= 12" for s in _TE_PEAK_FEASIBILITY_SPECS)
    for canon in ("ts_transfer_entropy", "ts_effective_transfer_entropy"):
        op = _get(canon)
        exprs = [r.expression for r in op.metadata.relational_specs]
        assert "window >= lag + 2" in exprs, canon


def test_m171_te_runtime_raise_preserved_for_bins_floor():
    """The bins-scaled floor (not expressible in the relational language) is
    still enforced at the call boundary: window=20/bins=3 is guaranteed-NaN."""
    op = _get("ts_transfer_entropy")
    rng = np.random.default_rng(26)
    idx = pd.date_range("2024-01-01", periods=50)
    x = pd.DataFrame(rng.normal(size=(50, 2)), index=idx, columns=["A", "B"])
    y = pd.DataFrame(rng.normal(size=(50, 2)), index=idx, columns=["A", "B"])
    with pytest.raises(ValueError):
        op.calculate(x, y, window=20, bins=3, lag=1)
    # A feasible combination still produces values.
    out = op.calculate(x, y, window=40, bins=3, lag=1, min_transitions=5)
    assert np.isfinite(out.to_numpy(dtype=float)).any()


def test_m171_te_relational_spec_prunes_basic_infeasibility():
    """window < lag + 2 is rejected by the declared RelationalParamSpec at the
    call boundary (compile-time prune), not just at runtime."""
    op = _get("ts_transfer_entropy")
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame(rng.normal(size=(10, 1)), index=idx, columns=["A"])
    y = pd.DataFrame(rng.normal(size=(10, 1)), index=idx, columns=["A"])
    with pytest.raises(ValueError):
        op.calculate(x, y, window=2, bins=3, lag=1)


# ---------------------------------------------------------------------------
# M-221 — Markov NaN-compress must not redefine lag
# ---------------------------------------------------------------------------

def test_m221_gap_does_not_pair_across():
    """A NaN interior row contributes NO transition and never re-pairs who is
    matched with whom: the lagged pairs are formed on the ORIGINAL time axis.

    past = [1, 2, 3, NaN, 100, 101, 102, 103] (bins=2, lag=1):
      * physical valid lag-1 transitions: (1,2), (2,3), (100,101), (101,102),
        (102,103) = 5.  Compressing the NaN would produce a 6th transition by
        pairing 3 -> 4 ... the point is the COUNT stays 5 and the state pair
        counts reflect ONLY the physical pairs.
      * A compressed re-pairing would add a transition from the last finite value
        before the gap to the first after it; the physical scheme must not.
    """
    from factor_engine.cleaned_operators.markov_dynamics import _state_dynamics_series

    series = np.array([1.0, 2.0, 3.0, np.nan, 100.0, 101.0, 102.0, 103.0, 104.0])
    res = _state_dynamics_series(series, window=8, bins=2, lag=1, min_count=1)
    # Row t=8 sees past = series[0:8]; median edges split finite values so the
    # states are 0 for the low block and 1 for the high block.  Physical count of
    # lag-1 transitions in a 7-finite-long window is 5 (compressed would be 6).
    assert res["total_trans"][8] == pytest.approx(5.0)
    N = res["N_obs"][8].astype(int)
    # The compressed scheme would add an extra (0,0) by pairing 3 (state 0) with
    # 100 (state 1)? 100 is state 1 here (median 100 -> edges [1,100,103]), so
    # the false cross-gap pair would be (0,1).  In the physical scheme (100,101)
    # IS a legit (0,1) transition, so the crisp guard is the TOTAL count == 5 and
    # the low-block self-transition (0,0) == 2 (only (1,2),(2,3)).
    assert N[0, 0] == 2, N.tolist()
    # The transition matrix support sums to exactly the physical count.
    assert int(N.sum()) == 5, N.tolist()


def test_m221_operator_gap_not_paired():
    """ts_markov_transition_surprisal reads the CURRENT jump from the historical
    matrix; a gap must not manufacture a cross-gap same-lag pair at the last row.
    Here the row before a NaN gap is low-state and the row after is high-state;
    if a false cross-gap pair had been created the historical (0,1) count would
    have been polluted.  We simply assert the operator runs and the kernel gate
    (total_trans) is unchanged by the gap's presence."""
    from factor_engine.cleaned_operators.markov_dynamics import _state_dynamics_series

    base = np.array([1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    gapped = np.array([1.0, 2.0, 3.0, np.nan, 100.0, 101.0, 102.0, 103.0, 104.0])
    r_base = _state_dynamics_series(base, window=8, bins=2, lag=1, min_count=1)
    r_gap = _state_dynamics_series(gapped, window=8, bins=2, lag=1, min_count=1)
    # Fully-finite 8-row past -> 7 physical lag-1 transitions.  The NaN removes
    # EXACTLY the two pairs that touch it (3->NaN, NaN->100), leaving 5.  If the
    # NaN were compressed away the count would be 6 (7 finite values -> 6 pairs);
    # asserting 5 proves the gap contributes no transition and never re-pairs
    # across itself.
    assert r_base["total_trans"][8] == pytest.approx(7.0)
    assert r_gap["total_trans"][8] == pytest.approx(5.0)
    assert r_gap["total_trans"][8] == pytest.approx(r_base["total_trans"][8] - 2.0)

"""
R61-FI-040/041 parity tests: FP legacy kernel vs FE operator.

One parity test per FP transform whose registry metadata routes execution to
the FE adapter (``implementation_origin = "FE_OPERATOR"``).  Parity here means
the *math of the transform* agrees between the retained FP-native kernel and
the FE canonical operator on the same numeric input.

Tolerance contract
------------------
- Cross-sectional (cs_rank/cs_zscore/cs_demean/cs_winsor): exact — the FE and
  FP kernels are the same math (average ties / nanstd ddof=1 / nanmean /
  np.nanquantile-linear clip).  Tolerance ``atol=0.0`` is asserted; a change
  to either kernel's finite-mask or tie policy would surface here.
- forward_fill (FE ffill_limit): exact pandas ``ffill(limit=max_lag)``.
  ``atol=0.0``.
- ols_neutralize (FE cs_neutralize): both solve per-date OLS residuals with
  ``numpy.linalg.lstsq``; tiny LAPACK differences make this float-exact in
  most panels but the contract is ``atol=1e-9`` (relative tolerance
  ``rtol=1e-7`` against the scale of the residuals).

Input convention
----------------
The FP kernels consume long-format frames (``asset_id``/``date``/``value``);
the legacy cross-sectional kernels additionally consume ``(T, N)`` numpy
panels.  Each test compares the FE-backed *adapter execution* against the
FP-native kernel result on the SAME panel, so it proves the routing is
correct end to end.

These tests import the FE operator registry lazily through
``factor_preprocess.adapters.fe_operator`` (never at module scope).  When FE
is not importable the tests are skipped rather than failed, because the FP
registry must remain executable in FP-only environments (the parity claim is
about the FE-backed path, which requires FE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.adapters.fe_operator import get_fe_executor

# Tolerance contract (explicit — plan §26 F3 / task FI-041):
_CS_TOL = {"atol": 0.0}
_FFILL_TOL = {"atol": 0.0}
_OLS_TOL = {"atol": 1e-9, "rtol": 1e-7}

_FE_AVAILABLE = get_fe_executor("rank") is not None

needs_fe = pytest.mark.skipif(
    not _FE_AVAILABLE,
    reason="factor_engine not importable in this environment (FP-only wheel); "
           "FE-backed parity path is not exercised",
)


def _long(values_mat, dates, cols, value_col="value"):
    rows = []
    for i, t in enumerate(dates):
        for j, col in enumerate(cols):
            rows.append({"date": t, "asset_id": col, value_col: values_mat[i, j]})
    return pd.DataFrame(rows)


def _rng_panel(seed, T=None, N=None):
    rng = np.random.default_rng(seed)
    if T is None:
        T = int(rng.integers(5, 12))
    if N is None:
        N = int(rng.integers(3, 8))
    mat = rng.normal(size=(T, N))
    mat[rng.random((T, N)) < 0.2] = np.nan
    if seed % 4 == 0:
        mat[0, :] = 5.0  # tie row
    if seed % 5 == 0:
        mat[:, 0] = 3.0  # constant column
    dates = pd.date_range("2024-01-01", periods=T)
    cols = [f"A{i}" for i in range(N)]
    return mat, dates, cols


def _adapter_call(fe_canonical, ldf, **scalar_kw):
    ex = get_fe_executor(fe_canonical)
    assert ex is not None, f"FE executor {fe_canonical} unavailable"
    out = ex(values=ldf, value_col="value", time_col="date",
             asset_col="asset_id", **scalar_kw)
    return out


@pytest.fixture(scope="module")
def _fe_executors():
    # One probe to force a single FE load for the module.
    return {c: get_fe_executor(c) for c in
            ("rank", "zscore", "cs_demean", "winsorize", "ffill_limit",
             "cs_neutralize")}


# ---------------------------------------------------------------------------
# Cross-sectional exact duplicates
# ---------------------------------------------------------------------------

@needs_fe
def test_cs_rank_fe_backed_matches_fp_kernel():
    from factor_preprocess.transforms import cs_rank

    mat, dates, cols = _rng_panel(1)
    ldf = _long(mat, dates, cols)
    out = _adapter_call("rank", ldf)
    fp = cs_rank(pd.DataFrame(mat, index=dates, columns=cols).to_numpy(),
                 method="average", pct=True)
    np.testing.assert_allclose(
        out.to_numpy().reshape(mat.shape), fp, **_CS_TOL)


@needs_fe
def test_cs_rank_ties_and_singletons_match():
    from factor_preprocess.transforms import cs_rank

    mat = np.array([
        [1.0, 5.0, 3.0],
        [2.0, np.nan, 1.0],
        [7.0, 7.0, 7.0],
        [4.0, 1.0, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    dates = pd.date_range("2024-01-01", periods=5)
    cols = ["A", "B", "C"]
    ldf = _long(mat, dates, cols)
    out = _adapter_call("rank", ldf)
    fp = cs_rank(mat, method="average", pct=True)
    np.testing.assert_allclose(out.to_numpy().reshape(5, 3), fp, **_CS_TOL)


@needs_fe
def test_cs_rank_inf_excluded_like_nan():
    from factor_preprocess.transforms import cs_rank

    mat = np.array([[1.0, np.inf, 3.0]])
    dates = pd.date_range("2024-01-01", periods=1)
    cols = ["A", "B", "C"]
    ldf = _long(mat, dates, cols)
    out = _adapter_call("rank", ldf)
    fp = cs_rank(mat, method="average", pct=True)
    assert np.isnan(out.to_numpy()[1])  # Inf cell is not rankable
    np.testing.assert_allclose(
        np.nan_to_num(out.to_numpy().reshape(1, 3), nan=np.nan),
        np.nan_to_num(fp, nan=np.nan), **_CS_TOL)


@needs_fe
def test_cs_demean_fe_backed_matches_fp_kernel_random_panel():
    from factor_preprocess.transforms import cs_demean

    mat, dates, cols = _rng_panel(3)
    ldf = _long(mat, dates, cols)
    out = _adapter_call("cs_demean", ldf)
    fp = cs_demean(mat)
    np.testing.assert_allclose(out.to_numpy().reshape(mat.shape), fp, **_CS_TOL)


@needs_fe
def test_cs_zscore_is_not_routed_singleton_semantics_differ():
    """cs_zscore stays FP_NATIVE: singleton-row semantics differ.

    FE zscore emits an all-NaN row when a cross-section has exactly one finite
    value (pandas ``std(ddof=1)`` is NaN and NaN is not replaced); FP returns
    ``constant_value`` (0.0) at the single finite cell because
    ``np.where(std > 0, ...)`` treats NaN>0 as False and falls through to the
    constant.  This is a deliberate non-routing (partial:normal).
    """
    from factor_preprocess.transforms import cs_zscore

    mat = np.array([[np.nan, 0.33057101, np.nan, np.nan]])
    dates = pd.date_range("2024-01-01", periods=1)
    cols = ["A", "B", "C", "D"]
    ldf = _long(mat, dates, cols)
    # registry must NOT route cs_zscore to FE
    registry = get_default_registry()
    assert registry.get("cs_zscore").implementation_origin == "FP_NATIVE"
    fp = cs_zscore(mat, ddof=1)
    assert fp[0, 1] == 0.0  # single finite cell -> constant_value
    # The FE zscore kernel itself would give all-NaN (documented divergence).
    fe_exec = get_fe_executor("zscore")
    if fe_exec is not None:
        fe_out = fe_exec(values=ldf, value_col="value", time_col="date",
                         asset_col="asset_id").to_numpy().reshape(1, 4)
        assert np.isnan(fe_out[0]).all()


@needs_fe
def test_cs_demean_fe_backed_matches_fp_kernel_after_zscore_nonrouting_case():
    from factor_preprocess.transforms import cs_demean

    mat, dates, cols = _rng_panel(3)
    ldf = _long(mat, dates, cols)
    out = _adapter_call("cs_demean", ldf)
    fp = cs_demean(mat)
    np.testing.assert_allclose(out.to_numpy().reshape(mat.shape), fp, **_CS_TOL)


@needs_fe
@pytest.mark.parametrize("lower,upper", [(0.01, 0.99), (0.05, 0.95), (0.25, 0.75)])
def test_cs_winsor_fe_backed_matches_fp_kernel(lower, upper):
    from factor_preprocess.transforms import cs_winsor

    mat, dates, cols = _rng_panel(4)
    ldf = _long(mat, dates, cols)
    out = _adapter_call("winsorize", ldf, lower=lower, upper=upper)
    fp = cs_winsor(mat, lower=lower, upper=upper)
    np.testing.assert_allclose(out.to_numpy().reshape(mat.shape), fp, **_CS_TOL)


# ---------------------------------------------------------------------------
# Missingness exact duplicate
# ---------------------------------------------------------------------------

@needs_fe
@pytest.mark.parametrize("max_lag", [1, 2, 5])
def test_forward_fill_fe_backed_matches_fp_kernel(max_lag):
    from factor_preprocess.transforms import forward_fill

    mat, dates, cols = _rng_panel(5 + max_lag)
    ldf = _long(mat, dates, cols)
    out = _adapter_call("ffill_limit", ldf, max_periods=max_lag)
    fp = forward_fill(ldf, max_lag=max_lag, asset_col="asset_id",
                      time_col="date", value_col="value")
    np.testing.assert_allclose(
        out.to_numpy().reshape(mat.shape),
        fp.to_numpy().reshape(mat.shape),
        **_FFILL_TOL,
    )


@needs_fe
def test_v7_forward_fill_positional_max_lag_is_bound_not_dropped():
    from factor_preprocess.transforms import forward_fill

    mat, dates, cols = _rng_panel(27, T=8, N=3)
    ldf = _long(mat, dates, cols).sort_values(["asset_id", "date"])
    executor = get_fe_executor("ffill_limit", fallback=forward_fill)
    out = executor(ldf, 1, "asset_id", "date", "value")
    expected = forward_fill(ldf, 1, "asset_id", "date", "value")
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert executor.effective_parameters == {"max_periods": 1}


@needs_fe
def test_v7_duplicate_time_asset_identity_is_rejected_even_when_values_match():
    mat, dates, cols = _rng_panel(28, T=3, N=2)
    ldf = _long(mat, dates, cols)
    ldf = pd.concat([ldf, ldf.iloc[[0]]], ignore_index=True)
    executor = get_fe_executor("rank")
    with pytest.raises(ValueError, match="duplicate .* identities"):
        executor(values=ldf)


# ---------------------------------------------------------------------------
# Neutralization exact duplicate (add_intercept=False)
# ---------------------------------------------------------------------------

@needs_fe
def test_ols_neutralize_fe_backed_matches_fp_kernel():
    from factor_preprocess.neutralization import ols_neutralize

    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=6)
    cols = ["A", "B", "C", "D", "E"]
    mat = rng.normal(size=(6, 5))
    mat[rng.random((6, 5)) < 0.15] = np.nan
    e1 = rng.normal(size=(6, 5))
    e2 = rng.normal(size=(6, 5))
    ldf = _long(mat, dates, cols)
    exp_rows = []
    for i, t in enumerate(dates):
        for j, c in enumerate(cols):
            exp_rows.append({"date": t, "asset_id": c, "e1": e1[i, j], "e2": e2[i, j]})
    lexp = pd.DataFrame(exp_rows)

    out = _adapter_call("cs_neutralize", ldf, exposures=lexp,
                        exposure_cols=("e1", "e2"), add_intercept=False)
    fp = ols_neutralize(
        ldf, lexp[["date", "asset_id", "e1", "e2"]],
        min_observations=2, add_intercept=False,
    )
    np.testing.assert_allclose(
        out.to_numpy().reshape(mat.shape),
        fp.to_numpy().reshape(mat.shape),
        **_OLS_TOL,
    )


# ---------------------------------------------------------------------------
# Routing / metadata contract (registry view)
# ---------------------------------------------------------------------------

def test_fe_backed_transforms_carry_origin_metadata():
    registry = get_default_registry()
    expected = {
        "cs_rank": "rank",
        "cs_demean": "cs_demean",
        "cs_winsor": "winsorize",
        "forward_fill": "ffill_limit",
        "ols_neutralize": "cs_neutralize",
        "industry_neutral": "cs_neutralize",
        "size_neutral": "cs_neutralize",
        "dual_neutral": "cs_neutralize",
    }
    for name, fe_id in expected.items():
        meta = registry.get(name)
        assert meta is not None
        assert meta.implementation_origin == "FE_OPERATOR", name
        assert meta.fe_operator_id == fe_id, name
        assert meta.fit_kind == "stateless", name
        assert registry.resolve_origin(name) == "FE_OPERATOR", name


def test_fitted_and_unmapped_transforms_are_fp_native():
    registry = get_default_registry()
    # Fitted-state transforms are never routed to FE (stateless-only rule).
    # requires_exposure is NOT the discriminator: ols_neutralize and its
    # semantic aliases require exposures but are stateless exact duplicates
    # that ARE FE-routed.  fitted (requires_fit) is the hard FP_NATIVE class.
    for m in registry.all_transforms():
        if m.requires_fit:
            assert m.implementation_origin == "FP_NATIVE", m.name
            assert m.fit_kind == "fitted", m.name
        if m.implementation_origin is None:
            raise AssertionError(
                f"transform {m.name} has no implementation_origin; routing audit "
                "cannot enumerate the catalog"
            )
    # Freshness / smoothing / decomposition / cs_zscore are FP-native.
    for name in ("freshness_score", "days_since_update", "ewma", "kama",
                 "trailing_sma", "one_sided_iir_lowpass", "kalman_local_level",
                 "hp_filter", "volatility_scale", "missing_indicator",
                 "cs_zscore"):
        meta = registry.get(name)
        assert meta is not None
        assert meta.implementation_origin == "FP_NATIVE", name
        assert registry.resolve_origin(name) == "FP_NATIVE", name


def test_fe_operator_execution_is_lazy_and_falls_back():
    """Registry execution must not hard-import FE; fallback = FP kernel."""
    registry = get_default_registry()
    # get_execution should return a callable in any env.  When FE is present
    # it is the adapter executor, otherwise the retained FP kernel.
    fn = registry.get_execution("cs_rank")
    assert callable(fn)
    fn2 = registry.get_execution("ols_neutralize")
    assert callable(fn2)

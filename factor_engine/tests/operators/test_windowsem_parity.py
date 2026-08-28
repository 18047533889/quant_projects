# -*- coding: utf-8 -*-
"""WINDOW-SEMANTICS PARITY — pandas reference vs polars/sql-emitter divergence
regressions (2026-08-28 WINDOW_SEMANTICS_PARITY).

Every test compares the ACTIVE polars implementation against the pandas
reference operator on the SAME panel (rows with NaN and ±Inf in the middle), and
asserts FULL matrix parity: identical NaN mask + identical finite values.

Producers pinned by this file:
  - ts_std/ts_var/ts_rank/ts_skew/ts_autocorr/ts_sharpe/ts_moment/ts_sum_decay/
    ts_poly2_coeff/ts_median/ts_quantile/ts_mean/ts_sum/ts_max/ts_min: ±Inf
    treated as missing inside the polars rolling kernels (pandas reference drops
    Inf from windows).
  - ts_log_return: ±Inf input is censored to null (NEW-200 invalid price).
  - ts_kurt: polars must match the ACTIVE pandas StableTsKurt (full finite
    windows, min_periods=window, unbiased Fisher excess) — NOT partial-window
    rolling.kurt.
  - ts_moment: cold-start rows < window are NaN; partial windows with >=1 finite
    sample compute the central moment.
  - ts_beta: polars delegates to the single reference kernel (exact parity).
  - ts_delta/ts_pct: ±Inf in the shift column is propagated identically (both
    pandas and polars emit ±Inf on an Inf - finite / Inf/finite ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("polars")
pytest.importorskip("pandas")

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _panel(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = np.abs(rng.standard_normal((n, 3)))
    data[5, 1] = np.nan          # mid NaN
    data[12, 2] = np.inf         # mid +Inf
    data[20, 2] = -np.inf        # mid -Inf
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(data, index=idx, columns=["AAA", "BBB", "CCC"])


def _assert_parity(canonical: str, pdf: pd.DataFrame, pkwargs: dict, plkwargs: dict | None = None) -> None:
    op_pd = OperatorRegistry.get(canonical, backend="pandas_numpy", mode="any")
    op_pl = OperatorRegistry.get(canonical, backend="polars", mode="any")
    assert op_pd is not None, f"{canonical}: missing pandas_numpy backend"
    assert op_pl is not None, f"{canonical}: missing polars backend"

    plf = pl.from_pandas(pdf.reset_index(drop=True))
    pk = plkwargs if plkwargs is not None else pkwargs
    r_pd = op_pd.calculate(pdf.copy(), **pkwargs)
    r_pl = op_pl.calculate(plf.clone(), **pk)
    r_pl_pd = r_pl.to_pandas()

    a_pd = r_pd.to_numpy(dtype=float)
    a_pl = r_pl_pd.to_numpy(dtype=float)

    np.testing.assert_array_equal(
        np.isnan(a_pd), np.isnan(a_pl),
        err_msg=f"{canonical}: NaN mask differs from pandas reference",
    )
    # finite cells must match exactly (both NaN and ±Inf handled above);
    # values that are ±Inf in BOTH sides are equal.
    pd_fin = ~np.isnan(a_pd)
    np.testing.assert_allclose(
        a_pl[pd_fin], a_pd[pd_fin], rtol=1e-9, atol=1e-9, equal_nan=True,
        err_msg=f"{canonical}: finite values differ from pandas reference",
    )


def test_ts_std_inf_is_missing() -> None:
    _assert_parity("ts_std", _panel(), {"window": 10})


def test_ts_var_inf_is_missing() -> None:
    _assert_parity("ts_var", _panel(), {"window": 10})


def test_ts_rank_inf_is_missing() -> None:
    _assert_parity("ts_rank", _panel(), {"window": 10})


def test_ts_log_return_inf_censored() -> None:
    _assert_parity("ts_log_return", _panel(), {"d": 2})


def test_ts_moment_partial_and_cold_start() -> None:
    _assert_parity("ts_moment", _panel(), {"d": 10, "k": 3})


def test_ts_sharpe_inf_is_missing() -> None:
    _assert_parity("ts_sharpe", _panel(), {"window": 10})


def test_ts_poly2_coeff_inf_is_missing() -> None:
    _assert_parity("ts_poly2_coeff", _panel(), {"d": 10})


def test_ts_median_inf_is_missing() -> None:
    _assert_parity("ts_median", _panel(), {"window": 10})


def test_ts_quantile_inf_is_missing() -> None:
    _assert_parity("ts_quantile", _panel(), {"d": 10, "q": 0.5})


def test_ts_sum_decay_inf_is_missing() -> None:
    _assert_parity("ts_sum_decay", _panel(), {"window": 10})


def test_ts_kurt_full_window_reference() -> None:
    _assert_parity("ts_kurt", _panel(), {"window": 10})


def test_ts_skew_inf_is_missing() -> None:
    _assert_parity("ts_skew", _panel(), {"window": 10})


def test_ts_autocorr_inf_is_missing() -> None:
    _assert_parity("ts_autocorr", _panel(), {"window": 10, "lag": 1})


def test_ts_mean_inf_is_missing() -> None:
    _assert_parity("ts_mean", _panel(), {"window": 10})


def test_ts_sum_inf_is_missing() -> None:
    _assert_parity("ts_sum", _panel(), {"window": 10})


def test_ts_max_inf_is_missing() -> None:
    _assert_parity("ts_max", _panel(), {"window": 10})


def test_ts_min_inf_is_missing() -> None:
    _assert_parity("ts_min", _panel(), {"window": 10})


def test_ts_delta_inf_propagated_identically() -> None:
    _assert_parity("ts_delta", _panel(), {"n": 2})


def test_ts_pct_inf_propagated_identically() -> None:
    _assert_parity("ts_pct", _panel(), {"d": 2})


def test_ts_kurt_poison_partial_window_does_not_emit() -> None:
    """POISON (non-vacuous): the OLD polars ts_kurt emitted a value on partial
    windows (row 3 = first 4 samples).  The pandas reference (StableTsKurt)
    keeps rows < window NaN.  Assert that behaviour is now pinned.
    """
    pdf = _panel()
    plf = pl.from_pandas(pdf.reset_index(drop=True))
    op_pd = OperatorRegistry.get("ts_kurt", backend="pandas_numpy", mode="any")
    op_pl = OperatorRegistry.get("ts_kurt", backend="polars", mode="any")
    r_pd = op_pd.calculate(pdf.copy(), window=10)
    r_pl = op_pl.calculate(plf.clone(), window=10).to_pandas()
    assert np.isnan(r_pd.iloc[3]["AAA"]), "pandas reference must be NaN at warmup row 3"
    assert np.isnan(r_pl.iloc[3]["AAA"]), "polars must NOT emit partial-window kurtosis"
    assert np.isfinite(r_pd.iloc[12]["AAA"]) and np.isfinite(r_pl.iloc[12]["AAA"])

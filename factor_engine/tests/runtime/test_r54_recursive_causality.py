from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.cleaned_operators.technical.adaptive_filters  # noqa: F401
from factor_engine.cleaned_operators.registry import OperatorRegistry


CASES = {
    "ts_mcginley_dynamic": {"window": 3, "power": 4.0},
    "ts_one_euro_filter": {"min_cutoff": 1.0, "beta": 0.01},
    "ts_nlms_filter": {"order": 3, "mu": 0.1, "eps": 1e-6},
    "ts_rls_filter": {"order": 3, "lambda_": 0.99, "delta": 1.0},
}


def _panel() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=24, freq="B")
    base = 100.0 + np.arange(24, dtype=float) + np.sin(np.arange(24))
    return pd.DataFrame({"B": base * 2.0, "A": base}, index=index)


def _polars(panel: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(panel.reset_index(names="date"))


@pytest.mark.parametrize("canonical", CASES)
def test_recursive_filters_are_prefix_causal_and_backend_identical(canonical: str) -> None:
    panel = _panel()
    kwargs = CASES[canonical]
    pandas_op = OperatorRegistry.get(canonical, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(canonical, backend="polars")

    full_pd = pandas_op.calculate(panel, **kwargs)
    prefix_pd = pandas_op.calculate(panel.iloc[:15], **kwargs)
    pd.testing.assert_frame_equal(full_pd.iloc[:15], prefix_pd, check_exact=False, rtol=1e-12, atol=1e-12)

    full_pl = polars_op.calculate(_polars(panel), **kwargs)
    prefix_pl = polars_op.calculate(_polars(panel.iloc[:15]), **kwargs)
    assert full_pl.columns == ["date", "B", "A"]
    assert full_pl["date"].to_list() == panel.index.to_list()
    np.testing.assert_allclose(
        full_pl.select("B", "A").head(15).to_numpy(),
        prefix_pl.select("B", "A").to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    np.testing.assert_allclose(
        full_pd[["B", "A"]].to_numpy(),
        full_pl.select("B", "A").to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_mcginley_gap_reseeds_only_after_complete_finite_window(missing: float) -> None:
    panel = _panel()
    panel.iloc[8, :] = missing
    kwargs = CASES["ts_mcginley_dynamic"]

    got_pd = OperatorRegistry.get("ts_mcginley_dynamic", backend="pandas_numpy").calculate(panel, **kwargs)
    got_pl = OperatorRegistry.get("ts_mcginley_dynamic", backend="polars").calculate(_polars(panel), **kwargs)

    assert got_pd.iloc[8:11].isna().all().all()
    expected_seed = panel.iloc[9:12].mean()
    pd.testing.assert_series_equal(got_pd.iloc[11], expected_seed, check_names=False)
    assert np.isfinite(got_pd.iloc[11:].to_numpy()).all()
    np.testing.assert_allclose(
        got_pd.to_numpy(), got_pl.select("B", "A").to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("missing", [np.inf, -np.inf])
def test_mcginley_leading_inf_waits_for_finite_seed_and_large_seed_stays_finite(missing: float) -> None:
    panel = _panel()
    panel.iloc[0, :] = missing
    panel.iloc[1:4, :] = np.array([[1e308, 5e307], [1e308, 5e307], [1e308, 5e307]])
    kwargs = CASES["ts_mcginley_dynamic"]

    got_pd = OperatorRegistry.get("ts_mcginley_dynamic", backend="pandas_numpy").calculate(panel, **kwargs)
    got_pl = OperatorRegistry.get("ts_mcginley_dynamic", backend="polars").calculate(_polars(panel), **kwargs)

    assert got_pd.iloc[:3].isna().all().all()
    np.testing.assert_allclose(got_pd.iloc[3].to_numpy(), [1e308, 5e307], rtol=0.0, atol=0.0)
    assert np.isfinite(got_pd.iloc[3].to_numpy()).all()
    np.testing.assert_allclose(
        got_pd.to_numpy(), got_pl.select("B", "A").to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("canonical", ["ts_one_euro_filter", "ts_nlms_filter", "ts_rls_filter"])
def test_other_recursive_filters_recover_after_interior_missing(canonical: str) -> None:
    panel = _panel()
    panel.iloc[8, :] = np.nan
    kwargs = CASES[canonical]
    got_pd = OperatorRegistry.get(canonical, backend="pandas_numpy").calculate(panel, **kwargs)
    got_pl = OperatorRegistry.get(canonical, backend="polars").calculate(_polars(panel), **kwargs)

    assert np.isfinite(got_pd.iloc[-4:].to_numpy()).all()
    np.testing.assert_allclose(
        got_pd.to_numpy(), got_pl.select("B", "A").to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )

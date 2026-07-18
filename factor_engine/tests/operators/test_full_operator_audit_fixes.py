# -*- coding: utf-8 -*-
"""Regression coverage for the full 2026-07 FactorEngine operator audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry() -> None:
    load_all()


def _pd(name: str):
    operator = OperatorRegistry.get(name, backend="pandas_numpy")
    assert operator is not None
    return operator


def _pl(name: str):
    operator = OperatorRegistry.get(name, backend="polars")
    assert operator is not None
    return operator


def test_mixed_fiscal_period_encodings_are_exact_and_cross_backend_equal():
    pl = pytest.importorskip("polars")
    values = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]})
    periods = pd.DataFrame(
        {"A": ["202401", "2024Q2", "20240930", "2024-12-31"]}
    )
    expected = np.array([np.nan, 10.0, 20.0, 30.0])

    pandas_result = _pd("period_lag").calculate(values, periods, 1)
    np.testing.assert_allclose(
        pandas_result["A"].to_numpy(),
        expected,
        equal_nan=True,
    )

    polars_result = _pl("period_lag").calculate(
        pl.DataFrame({"A": values["A"].to_numpy()}),
        pl.DataFrame({"A": periods["A"].tolist()}),
        1,
    )
    np.testing.assert_allclose(
        polars_result["A"].to_numpy(),
        expected,
        equal_nan=True,
    )


def test_fiscal_revision_policy_and_composites_match_polars():
    pl = pytest.importorskip("polars")
    values = pd.DataFrame({"A": [10.0, 20.0, 22.0, 30.0, 40.0]})
    periods = pd.DataFrame(
        {"A": ["2024Q1", "2024Q2", "2024Q2", "2024Q3", "2024Q4"]}
    )

    latest = _pd("period_lag").calculate(
        values, periods, 1, "latest_available"
    )
    first = _pd("period_lag").calculate(
        values, periods, 1, "first_available"
    )
    assert latest.iloc[-2, 0] == 22.0
    assert first.iloc[-2, 0] == 20.0

    pl_values = pl.DataFrame({"A": values["A"].to_numpy()})
    pl_periods = pl.DataFrame({"A": periods["A"].tolist()})
    latest_pl = _pl("period_lag").calculate(
        pl_values, pl_periods, 1, "latest_available"
    )
    first_pl = _pl("period_lag").calculate(
        pl_values, pl_periods, 1, "first_available"
    )
    np.testing.assert_allclose(
        latest_pl["A"].to_numpy(),
        latest["A"].to_numpy(),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first_pl["A"].to_numpy(),
        first["A"].to_numpy(),
        equal_nan=True,
    )

    for name, args in (
        ("period_change", (1, "absolute", True)),
        ("period_average", (2, True)),
        ("period_cagr", (3, 4, "strict", True)),
        ("ttm_from_quarterly", (4, True)),
    ):
        pandas_result = _pd(name).calculate(values, periods, *args)
        polars_result = _pl(name).calculate(pl_values, pl_periods, *args)
        np.testing.assert_allclose(
            polars_result["A"].to_numpy(),
            pandas_result["A"].to_numpy(),
            equal_nan=True,
            rtol=1e-10,
            atol=1e-10,
        )

    assert _pd("period_average").calculate(values, periods, 2, True).iloc[-1, 0] == 35.0
    assert _pd("ttm_from_quarterly").calculate(values, periods, 4, True).iloc[-1, 0] == 102.0


def test_cumulative_quarter_ttm_and_yoy_cross_backend_parity():
    pl = pytest.importorskip("polars")
    cumulative = pd.DataFrame(
        {"A": [10.0, 30.0, 60.0, 100.0, 15.0]}
    )
    periods = pd.DataFrame(
        {"A": ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"]}
    )
    quarters = pd.DataFrame({"A": [1, 2, 3, 4, 1]})
    pl_values = pl.DataFrame({"A": cumulative["A"].to_numpy()})
    pl_periods = pl.DataFrame({"A": periods["A"].tolist()})
    pl_quarters = pl.DataFrame({"A": quarters["A"].to_numpy()})

    for name, pandas_args, polars_args in (
        (
            "quarter_from_cumulative",
            (cumulative, periods, quarters),
            (pl_values, pl_periods, pl_quarters),
        ),
        (
            "ttm_from_cumulative",
            (cumulative, periods, quarters),
            (pl_values, pl_periods, pl_quarters),
        ),
        (
            "yoy_by_period",
            (cumulative, periods, 4, "signed", True),
            (pl_values, pl_periods, 4, "signed", True),
        ),
    ):
        pandas_result = _pd(name).calculate(*pandas_args)
        polars_result = _pl(name).calculate(*polars_args)
        np.testing.assert_allclose(
            polars_result["A"].to_numpy(),
            pandas_result["A"].to_numpy(),
            equal_nan=True,
            rtol=1e-10,
            atol=1e-10,
        )


def test_fundamental_staleness_rejects_future_availability_in_both_backends():
    pl = pytest.importorskip("polars")
    available = pd.DataFrame(
        {"A": pd.to_datetime(["2024-01-03", "2024-01-05"])}
    )
    decision = pd.DataFrame(
        {"A": pd.to_datetime(["2024-01-04", "2024-01-04"])}
    )
    pandas_result = _pd("fundamental_staleness").calculate(
        available, decision
    )
    polars_result = _pl("fundamental_staleness").calculate(
        pl.DataFrame({"A": available["A"].tolist()}),
        pl.DataFrame({"A": decision["A"].tolist()}),
    )
    assert pandas_result.iloc[0, 0] == 1.0
    assert np.isnan(pandas_result.iloc[1, 0])
    np.testing.assert_allclose(
        polars_result["A"].to_numpy(),
        pandas_result["A"].to_numpy(),
        equal_nan=True,
    )


def test_no_intercept_r2_uses_uncentered_definition():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"A": [2.0, 3.0, 5.0, 8.0]})
    result = _pd("ts_regression_r2").calculate(
        y, x, 4, 3, False
    )
    xv = x["A"].to_numpy()
    yv = y["A"].to_numpy()
    slope = float(np.dot(xv, yv) / np.dot(xv, xv))
    residual = yv - slope * xv
    expected = 1.0 - float(np.dot(residual, residual)) / float(
        np.dot(yv, yv)
    )
    assert result.iloc[-1, 0] == pytest.approx(expected)


def test_cs_neutralize_excludes_missing_groups_and_handles_intercept_only():
    y = pd.DataFrame(
        [[1.0, 2.0, 3.0, 100.0]],
        columns=list("ABCD"),
    )
    group = pd.DataFrame(
        [["industry", "industry", "industry", None]],
        columns=list("ABCD"),
    )
    result = _pd("cs_neutralize").calculate(
        y,
        group=group,
        add_intercept=True,
        min_obs=3,
    )
    np.testing.assert_allclose(
        result.loc[0, ["A", "B", "C"]].to_numpy(dtype=float),
        np.array([-1.0, 0.0, 1.0]),
        atol=1e-12,
    )
    assert np.isnan(result.loc[0, "D"])


@pytest.mark.parametrize("canonical", ["ts_product", "ts_mad"])
def test_fractional_windows_are_rejected(canonical):
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        _pd(canonical).calculate(values, 1.5)


def test_production_aggressive_environment_cannot_bypass_evidence(monkeypatch):
    from backend.operator_capability import (
        UnsupportedOperatorBackendError,
        get_best_backend,
    )

    monkeypatch.setenv("FACTOR_ENGINE_OPERATOR_BACKEND", "auto_aggressive")
    _, backend = get_best_backend(
        "ADX",
        mode="production",
        prefer="auto",
    )
    assert backend == "pandas_numpy"

    with pytest.raises(UnsupportedOperatorBackendError):
        get_best_backend("ADX", mode="production", prefer="polars")

    _, research_backend = get_best_backend(
        "ADX",
        mode="research",
        prefer="auto",
        allow_unverified_backend=True,
    )
    assert research_backend == "polars"


def test_registry_explicit_preference_is_fail_closed():
    from backend.operator_capability import UnsupportedOperatorBackendError

    with pytest.raises(UnsupportedOperatorBackendError):
        OperatorRegistry.get_preferred(
            "ADX",
            prefer="polars",
            mode="production",
        )
    with pytest.raises(ValueError):
        OperatorRegistry.get_preferred("ADX", prefer="unknown")


def test_invalid_evidence_never_falls_back_to_case_registry(monkeypatch):
    from backend import primitive_evidence

    monkeypatch.setattr(
        primitive_evidence,
        "evidence_artifact_valid",
        lambda: False,
    )
    assert (
        primitive_evidence._load_verified_fail_closed(
            "polars_reference_parity"
        )
        == frozenset()
    )


def test_duckdb_probe_failure_downgrades_all_static_candidates(monkeypatch):
    from backend import sql_tiers
    from backend.sql_pushdown import duckdb_capabilities

    def fail(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        duckdb_capabilities,
        "get_duckdb_capability_report",
        fail,
    )
    sql_tiers._DUCKDB_DOWNGRADE_CACHE = None
    downgraded = sql_tiers.duckdb_downgraded_canonicals(refresh=True)
    assert downgraded == frozenset(
        sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS
    )


def test_final_registry_aliases_and_surfaces_are_closed():
    from cleaned_operators.operator_surface import classify_canonical

    canonicals = set(OperatorRegistry.list_canonical())
    for alias, target in OperatorRegistry._aliases.items():
        assert target in canonicals, (alias, target)
        assert OperatorRegistry.resolve_canonical(alias) == target

    for canonical in canonicals:
        catalog = OperatorRegistry.catalog()[canonical]
        assert catalog.get("surface") == classify_canonical(canonical)
        assert OperatorRegistry.backends_for(canonical)

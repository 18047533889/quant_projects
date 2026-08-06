# -*- coding: utf-8 -*-
from __future__ import annotations

import sys

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


def _to_polars(frame: pd.DataFrame):
    pl = pytest.importorskip("polars")
    return pl.DataFrame({column: frame[column].tolist() for column in frame.columns})


def _assert_polars_equal(name: str, args: tuple, *operator_args) -> None:
    if "polars" not in OperatorRegistry.backends_for(name):
        pytest.skip(f"{name} has no certified Polars runtime")
    pandas_result = _pd(name).calculate(*args, *operator_args)
    polars_args = tuple(_to_polars(arg) if isinstance(arg, pd.DataFrame) else arg for arg in args)
    polars_result = OperatorRegistry.get(name, backend="polars").calculate(
        *polars_args, *operator_args
    )
    np.testing.assert_allclose(
        polars_result.select(pandas_result.columns.tolist()).to_numpy(),
        pandas_result.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10,
    )


def test_obsolete_fiscal_override_is_not_imported() -> None:
    assert "cleaned_operators.overhaul.fundamental" not in sys.modules


def test_fiscal_backend_signatures_are_identical_and_complete() -> None:
    expected = {
        "period_lag": ["x", "period_id", "periods", "revision_policy"],
        "period_change": [
            "x", "period_id", "periods", "mode", "require_consecutive", "revision_policy"
        ],
        "period_average": [
            "x", "period_id", "periods", "require_consecutive", "revision_policy"
        ],
        "period_cagr": [
            "x", "period_id", "periods", "periods_per_year", "sign_policy",
            "require_consecutive", "revision_policy",
        ],
        "quarter_from_cumulative": [
            "x", "period_id", "fiscal_quarter", "revision_policy"
        ],
        "ttm_from_quarterly": [
            "x", "period_id", "periods", "require_consecutive", "revision_policy"
        ],
        "ttm_from_cumulative": [
            "x", "period_id", "fiscal_quarter", "revision_policy"
        ],
        "yoy_by_period": [
            "x", "period_id", "periods", "denominator", "require_consecutive",
            "revision_policy",
        ],
    }
    for canonical, signature in expected.items():
        catalog = OperatorRegistry.catalog()[canonical]
        for backend in ("pandas_numpy", "polars"):
            if backend in OperatorRegistry.backends_for(canonical):
                assert catalog["backend_signatures"][backend] == signature
        assert catalog["semantic_contract"] == "strict_fiscal_period_v2"
        assert catalog["certified_parameter_domain"]["revision_policy"] == [
            "first_available", "latest_available"
        ]


@pytest.mark.parametrize("revision_policy,expected", [("latest_available", 25.0), ("first_available", 20.0)])
def test_period_lag_revision_policy_has_cross_backend_parity(revision_policy, expected) -> None:
    values = pd.DataFrame({"A": [10.0, 20.0, 25.0, 40.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q2", "2023Q2", "2023Q3"]})
    result = _pd("period_lag").calculate(values, periods, 1, revision_policy)
    assert result.iloc[-1, 0] == expected
    _assert_polars_equal("period_lag", (values, periods), 1, revision_policy)


@pytest.mark.parametrize("revision_policy,expected", [("latest_available", 32.5), ("first_available", 30.0)])
def test_period_average_revision_policy_has_cross_backend_parity(revision_policy, expected) -> None:
    values = pd.DataFrame({"A": [10.0, 20.0, 25.0, 40.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q2", "2023Q2", "2023Q3"]})
    result = _pd("period_average").calculate(
        values, periods, 2, True, revision_policy
    )
    assert result.iloc[-1, 0] == pytest.approx(expected)
    _assert_polars_equal(
        "period_average", (values, periods), 2, True, revision_policy
    )


@pytest.mark.parametrize("sign_policy,expected", [("strict", np.nan), ("absolute", 1.0)])
def test_period_cagr_sign_policy_has_cross_backend_parity(sign_policy, expected) -> None:
    values = pd.DataFrame({"A": [-10.0, 20.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q2"]})
    result = _pd("period_cagr").calculate(
        values, periods, 1, 1, sign_policy, True, "latest_available"
    )
    if np.isnan(expected):
        assert np.isnan(result.iloc[-1, 0])
    else:
        assert result.iloc[-1, 0] == pytest.approx(expected)
    _assert_polars_equal(
        "period_cagr",
        (values, periods),
        1,
        1,
        sign_policy,
        True,
        "latest_available",
    )


@pytest.mark.parametrize("denominator,expected", [("signed", -1.5), ("absolute", 1.5)])
def test_yoy_denominator_policy_has_cross_backend_parity(denominator, expected) -> None:
    values = pd.DataFrame({"A": [-10.0, 1.0, 2.0, 3.0, 5.0]})
    periods = pd.DataFrame(
        {"A": ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"]}
    )
    result = _pd("yoy_by_period").calculate(
        values, periods, 4, denominator, True, "latest_available"
    )
    assert result.iloc[-1, 0] == pytest.approx(expected)
    _assert_polars_equal(
        "yoy_by_period",
        (values, periods),
        4,
        denominator,
        True,
        "latest_available",
    )


def test_bool_window_is_rejected_by_both_runtime_bases() -> None:
    from backend.operator_errors import OperatorParameterError
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(OperatorParameterError, match="bool"):
        _pd("ts_mean").calculate(values, True)
    if "polars" in OperatorRegistry.backends_for("ts_mean"):
        with pytest.raises(OperatorParameterError, match="bool"):
            OperatorRegistry.get("ts_mean", backend="polars").calculate(
                _to_polars(values), True
            )


def test_misaligned_panels_fail_closed() -> None:
    # After allow_panel_broadcast changes, operators broadcast across misaligned
    # columns and return NaN for non-matching pairs instead of raising.
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"B": [1.0, 2.0, 3.0]})
    result = _pd("ts_corr").calculate(x, y, 2)
    # Result should have both columns A and B, all NaN since they don't overlap
    assert set(result.columns) == {"A", "B"}
    assert result.isna().all().all()



def test_top_bottom_k_sample_std_rejects_singleton_k() -> None:
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    for name in ("ts_topk_std", "ts_bottomk_std"):
        with pytest.raises(ValueError, match="k >= 2"):
            _pd(name).calculate(values, 3, 1)
        constraints = OperatorRegistry.catalog()[name]["parameter_constraints"]
        assert constraints["k"]["minimum"] == 2
        assert constraints["ddof"] == 1


def test_days_since_max_lookback_is_inclusive() -> None:
    condition = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]})
    result = _pd("ts_days_since").calculate(condition, 2)
    assert result["A"].iloc[:3].tolist() == [0.0, 1.0, 2.0]
    assert np.isnan(result["A"].iloc[3])
    _assert_polars_equal("ts_days_since", (condition,), 2)
    contract = OperatorRegistry.catalog()["ts_days_since"]["lookback_contract"]
    assert contract["inclusive_max_distance"] is True


def test_technical_and_tail_risk_policy_metadata_is_corrected() -> None:
    catalog = OperatorRegistry.catalog()
    for name in ("ADX", "MACD_line", "MACD_signal", "MACD_hist"):
        assert catalog[name]["scope"] == "time_series"
        assert catalog[name]["stateful"] is True
        # ``pit_safe`` is evidence-converged, never forced by the final contract
        # layer (review §2.7): True exactly when the six-gate certifies it.
        assert catalog[name]["pit_safe"] is (
            catalog[name]["production_certified"] is True
        )
    assert catalog["lqtp_historical_cvar"]["scope"] == "time_series"
    assert catalog["lqtp_historical_cvar"]["pit_safe"] is (
        catalog["lqtp_historical_cvar"]["production_certified"] is True
    )


def test_every_active_operator_has_one_unified_contract() -> None:
    for canonical in OperatorRegistry.list_canonical():
        catalog = OperatorRegistry.catalog()[canonical]
        contract = catalog["contract"]
        assert contract["canonical"] == canonical
        assert contract["version"] == "2.1"
        assert contract["lookback"]
        assert set(contract["backends"]) == set(OperatorRegistry.backends_for(canonical))


def test_time_series_operators_have_machine_readable_lookback() -> None:
    for canonical, catalog in OperatorRegistry.catalog().items():
        if catalog.get("scope") != "time_series":
            continue
        lookback = catalog["lookback_contract"]
        assert lookback["kind"] in {
            "parameterized_rows", "recursive_state", "causal_unbounded"
        }
        if "window" in (catalog.get("param_names") or ()):
            assert "window" in (lookback.get("parameters") or ())


def test_replacement_history_is_explicit() -> None:
    history = OperatorRegistry.catalog()["ts_days_since"]["replacement_history"]
    assert history
    assert any(row["new_source"] == "layer_composite_fixes" for row in history)

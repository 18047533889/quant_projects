from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _load(name: str):
    return importlib.import_module(f"factor_engine.cleaned_operators.{name}")


def _load_canonical(name: str):
    return _load(name)


def _event_case(curve: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = 40
    response = np.zeros(n, dtype=float)
    event = np.zeros(n, dtype=float)
    for anchor in (1, 10, 20):
        event[anchor] = 1.0
        response[anchor + 1 : anchor + 1 + len(curve)] = curve
    return pd.DataFrame({"A": response}), pd.DataFrame({"A": event})


@pytest.mark.parametrize("curve", [[1.0] * 4, [-1.0] * 4])
def test_m42_public_reversal_valid_no_flip_is_zero(curve):
    er = _load("event_response")
    response, event = _event_case(curve)
    out = er.EventResponseReversalStrength().calculate(
        response, event, history_window=40, horizon=4, min_events=3
    )
    assert out.iloc[-1, 0] == 0.0


@pytest.mark.parametrize(
    ("curve", "expected"),
    [([1.0, 1.0, -1.0, -1.0], -2.0), ([-1.0, -1.0, 1.0, 1.0], 2.0)],
)
def test_m42_public_reversal_true_flip_formula_unchanged(curve, expected):
    er = _load("event_response")
    response, event = _event_case(curve)
    out = er.EventResponseReversalStrength().calculate(
        response, event, history_window=40, horizon=4, min_events=3
    )
    assert out.iloc[-1, 0] == pytest.approx(expected)


def test_m42_undefined_and_missing_cohorts_remain_nan():
    er = _load("event_response")
    response, event = _event_case([1.0] * 4)
    no_events = pd.DataFrame(0.0, index=response.index, columns=response.columns)
    assert np.isnan(
        er.EventResponseReversalStrength().calculate(
            response, no_events, history_window=40, horizon=4, min_events=1
        ).iloc[-1, 0]
    )
    response.iloc[12, 0] = np.nan
    assert np.isnan(
        er.EventResponseReversalStrength().calculate(
            response, event, history_window=40, horizon=4, min_events=3
        ).iloc[-1, 0]
    )
    one_lag, one_event = _event_case([1.0])
    assert np.isnan(
        er.EventResponseReversalStrength().calculate(
            one_lag, one_event, history_window=40, horizon=1, min_events=3
        ).iloc[-1, 0]
    )


@pytest.mark.parametrize(
    ("turn_value", "expected"),
    [(0.0, 1.0), (1e-16, np.exp(-20e-16)), (0.5, np.exp(-10.0))],
)
def test_m44_public_old_mass_is_independent_turnover_diagnostic(turn_value, expected):
    ts = _load("turnover_survival")
    price = pd.DataFrame({"A": np.full(21, 100.0)})
    turnover = pd.DataFrame({"A": np.full(21, turn_value)})
    out = ts.TsTurnoverOldMass().calculate(price, turnover, window=20)
    assert out.iloc[-1, 0] == pytest.approx(expected, rel=1e-14, abs=1e-16)


@pytest.mark.parametrize("bad", [np.nan, -0.1])
def test_m44_unknown_or_negative_turnover_keeps_old_mass_nan(bad):
    ts = _load("turnover_survival")
    price = pd.DataFrame({"A": np.full(21, 100.0)})
    turnover = pd.DataFrame({"A": np.full(21, 0.5)})
    turnover.iloc[10, 0] = bad
    out = ts.TsTurnoverOldMass().calculate(price, turnover, window=20)
    assert np.isnan(out.iloc[-1, 0])


PRICE_CLASSES = (
    "TsTurnoverReferencePrice",
    "TsTurnoverCostDispersion",
    "TsTurnoverProfitShare",
    "TsTurnoverNearCostMass",
    "TsTurnoverCostQuantileDistance",
    "TsTurnoverCostEntropy",
    "TsTurnoverCostModeDistance",
    "TsTurnoverCostSkew",
    "TsTurnoverCostEntropyVolScaled",
)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.nan, np.inf])
def test_m45_positive_turnover_invalid_historical_price_blocks_all_cost_outputs(bad_price):
    ts = _load("turnover_survival")
    price = pd.DataFrame({"A": np.full(21, 100.0)})
    turnover = pd.DataFrame({"A": np.full(21, 0.5)})
    price.iloc[10, 0] = bad_price
    for class_name in PRICE_CLASSES:
        out = getattr(ts, class_name)().calculate(price, turnover, window=20)
        assert np.isnan(out.iloc[-1, 0]), class_name
    old = ts.TsTurnoverOldMass().calculate(price, turnover, window=20)
    assert old.iloc[-1, 0] == pytest.approx(np.exp(-10.0))


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.nan, np.inf])
def test_m45_zero_turnover_invalid_price_adds_no_unknown_cost_mass(bad_price):
    ts = _load("turnover_survival")
    price = pd.DataFrame({"A": np.linspace(90.0, 110.0, 21)})
    turnover = pd.DataFrame({"A": np.full(21, 0.5)})
    price.iloc[10, 0] = bad_price
    turnover.iloc[10, 0] = 0.0
    out = ts.TsTurnoverReferencePrice().calculate(price, turnover, window=20)
    assert np.isfinite(out.iloc[-1, 0])


def test_m44_m45_prefix_causality_and_current_row_exclusion():
    ts = _load("turnover_survival")
    price = pd.DataFrame({"A": np.linspace(90.0, 110.0, 30)})
    turnover = pd.DataFrame({"A": np.full(30, 0.5)})
    base = ts.TsTurnoverReferencePrice().calculate(price, turnover, window=20)
    poisoned_price = pd.concat([price, pd.DataFrame({"A": [-1.0]})], ignore_index=True)
    poisoned_turn = pd.concat([turnover, pd.DataFrame({"A": [0.5]})], ignore_index=True)
    extended = ts.TsTurnoverReferencePrice().calculate(
        poisoned_price, poisoned_turn, window=20
    )
    np.testing.assert_allclose(base.to_numpy(), extended.iloc[:-1].to_numpy(), equal_nan=True)


def test_active_polars_consumers_use_corrected_shared_kernels():
    pl = pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all
    load_all()
    response, event = _event_case([1.0] * 4)
    response_pl = pl.from_pandas(response)
    event_pl = pl.from_pandas(event)
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    er_cls = type(OperatorRegistry.get("event_response_reversal_strength", "polars"))
    assert er_cls.__module__ == "factor_engine.cleaned_operators.polars_dynamics"
    er_out = er_cls().calculate(
        response_pl, event_pl, history_window=40, horizon=4, min_events=3
    )
    assert er_out["A"][-1] == 0.0

    price = pl.DataFrame({"A": np.full(21, 100.0)})
    turnover = pl.DataFrame({"A": np.zeros(21)})
    chip_cls = type(OperatorRegistry.get("ts_turnover_old_mass", "polars"))
    assert chip_cls.__module__ == "factor_engine.cleaned_operators.polars_chip_tail"
    chip_out = chip_cls().calculate(price, turnover, window=20)
    assert chip_out["A"][-1] == pytest.approx(1.0)

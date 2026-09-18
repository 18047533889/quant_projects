"""Default-binding contract for the live Polars event-response registration."""
import inspect

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _panels():
    n = 180
    event = np.zeros(n)
    event[np.arange(8, 150, 14)] = 1.0
    response = 0.02 * np.sin(np.arange(n) / 5.0)
    for anchor in np.flatnonzero(event):
        width = min(11, n - anchor)
        response[anchor:anchor + width] += 1.7 * np.exp(-0.32 * np.arange(width))
    pandas_response = pd.DataFrame({"A": response, "B": response[::-1]})
    pandas_event = pd.DataFrame({"A": event, "B": event[::-1]})
    return pandas_response, pandas_event, pl.from_pandas(pandas_response), pl.from_pandas(pandas_event)


def test_live_polars_registration_exposes_authored_defaults():
    load_all()
    operator = OperatorRegistry.get("event_response_decay_rate", "polars", mode="any")
    assert type(operator).__module__ == "factor_engine.cleaned_operators.polars_dynamics"
    signature = inspect.signature(operator._contract_callable)
    assert signature.parameters["history_window"].default == 120
    assert signature.parameters["horizon"].default == 10
    assert signature.parameters["min_events"].default == 3


def test_omitted_defaults_match_explicit_defaults_and_pandas_backend():
    load_all()
    py, pe, ly, le = _panels()
    pandas_op = OperatorRegistry.get("event_response_decay_rate", "pandas_numpy", mode="any")
    polars_op = OperatorRegistry.get("event_response_decay_rate", "polars", mode="any")
    pandas_implicit = pandas_op.calculate(py, pe).to_numpy()
    pandas_explicit = pandas_op.calculate(py, pe, history_window=120, horizon=10, min_events=3).to_numpy()
    polars_implicit = polars_op.calculate(ly, le).to_numpy()
    polars_explicit = polars_op.calculate(ly, le, history_window=120, horizon=10, min_events=3).to_numpy()
    assert np.isfinite(pandas_implicit).sum() > 0
    np.testing.assert_allclose(pandas_implicit, pandas_explicit, equal_nan=True)
    np.testing.assert_allclose(polars_implicit, polars_explicit, equal_nan=True)
    np.testing.assert_allclose(polars_implicit, pandas_implicit, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_omitted_defaults_remain_future_prefix_invariant():
    load_all()
    _, _, response, event = _panels()
    cut = 145
    changed_response = response.with_columns(
        pl.when(pl.int_range(pl.len()) >= cut).then(pl.col("A") * -11 + 7).otherwise(pl.col("A")).alias("A")
    )
    changed_event = event.with_columns(
        pl.when(pl.int_range(pl.len()) >= cut).then(1 - pl.col("A")).otherwise(pl.col("A")).alias("A")
    )
    op = OperatorRegistry.get("event_response_decay_rate", "polars", mode="any")
    before = op.calculate(response, event).to_numpy()
    after = op.calculate(changed_response, changed_event).to_numpy()
    np.testing.assert_allclose(before[:cut], after[:cut], equal_nan=True)

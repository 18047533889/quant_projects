"""A capped streak of d one-step returns needs d prior price rows."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import own_history_requirement
from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.ir.analyzer import Analyzer


@pytest.mark.parametrize("d", [1, 5, 20])
def test_digital_count_history_is_number_of_transitions(d):
    load_all()
    req = own_history_requirement("digital_count", {"d": d, "threshold": 0.01, "run": 1})
    assert req.rows >= d
    analysis = Analyzer().lower(F("digital_count")(col("close"), d=d, threshold=0.01, run=1))
    assert analysis.lookback >= d


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_declared_warmup_replays_capped_streak_exactly(backend):
    load_all()
    d = 20
    req = own_history_requirement("digital_count", {"d": d, "threshold": 0.01, "run": 3})
    frame = pd.DataFrame({"A": 1.0 + np.arange(45) * 0.001})
    cut = 35
    op = OperatorRegistry.get("digital_count", backend, mode="research")
    def execute(data):
        arg = pl.from_pandas(data) if backend == "polars" else data
        return op.calculate(arg, d=d, threshold=0.01, run=3)["A"].to_numpy()
    full = execute(frame)
    chunk = execute(frame.iloc[cut - req.rows:])
    np.testing.assert_allclose(chunk[req.rows:], full[cut:])
    assert (full[cut:] == d).all()

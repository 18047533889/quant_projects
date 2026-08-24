"""Direct registry calls enforce the same rolling-parameter contract as Planner."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    ensure_cleaned_loaded()


@pytest.mark.parametrize(
    ("operator", "kwargs"),
    [
        ("ts_rank", {"window": 1.5}),
        ("ts_rank", {"window": True}),
        ("ts_sharpe", {"window": 1}),
        ("ts_sharpe", {"ann_factor": float("nan")}),
        ("ts_autocorr", {"window": 5, "lag": 5}),
        ("ts_autocorr", {"window": 5, "lag": 1.5}),
    ],
)
def test_pandas_rolling_operators_reject_invalid_parameters(operator, kwargs):
    value = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(OperatorParameterError):
        OperatorRegistry.get(operator, backend="pandas_numpy").calculate(value, **kwargs)


def test_native_polars_ts_rank_rejects_legacy_runtime_alias():
    pl = pytest.importorskip("polars")
    value = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(OperatorParameterError, match="window"):
        OperatorRegistry.get("ts_rank", backend="polars").calculate(value, d=2)

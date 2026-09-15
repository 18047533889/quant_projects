from __future__ import annotations

import numpy as np
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _linear_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def test_ts_quantile_integer_panel_preserves_linear_interpolation() -> None:
    load_all()
    raw = [1, 2, None, 4, 8, 16, 32]
    window = 4
    q = 0.35
    expected = []
    for end in range(1, len(raw) + 1):
        finite = [float(v) for v in raw[max(0, end - window):end] if v is not None]
        expected.append(_linear_quantile(finite, q))

    operator = OperatorRegistry.get("ts_quantile", backend="polars")
    actual = operator.calculate(pl.DataFrame({"A": raw}), d=window, q=q)

    assert actual.schema["A"] == pl.Float64
    np.testing.assert_allclose(actual["A"].to_numpy(), expected, rtol=0, atol=1e-14)

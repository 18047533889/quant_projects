from __future__ import annotations

import numpy as np
import pandas as pd


def test_ultimate_oscillator_six_argument_dsl_uses_documented_default_weights() -> None:
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    load_all()
    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    t = np.arange(40, dtype=float)
    close = pd.DataFrame(
        {
            "A": 100 + 0.2 * t + np.sin(t / 2),
            "B": 80 + 0.1 * t + 1.7 * np.cos(t / 3),
        },
        index=dates,
    )
    high = close.add(1.0 + 0.1 * np.cos(t), axis=0)
    low = close.sub(1.0 + 0.1 * np.sin(t), axis=0)
    data = {}
    for name, frame in {"high": high, "low": low, "close": close}.items():
        series = frame.stack()
        series.index.names = ["timestamp", "instrument"]
        data[name] = series

    expression = "zscore(UltimateOscillator(high, low, close, 5, 10, 20))"
    parser = DSLParser(surface="compat_research", dialect="native")
    parsed = parser.parse(expression)
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=data),
        run_mode="research",
    )
    actual = engine.run(Factor(name="default_weights", expr=parsed))["result"]
    explicit = parser.parse(
        "zscore(UltimateOscillator(high, low, close, 5, 10, 20, 4.0, 2.0, 1.0))"
    )
    expected_series = engine.run(Factor(name="explicit_weights", expr=explicit))["result"]
    assert np.isfinite(actual.to_numpy()).any()
    pd.testing.assert_series_equal(
        actual.sort_index(), expected_series.sort_index(), check_names=False
    )


def test_ultimate_oscillator_param_specs_expose_kernel_weight_defaults() -> None:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    operator = OperatorRegistry.get("UltimateOscillator", backend="pandas_numpy")
    assert operator is not None
    specs = operator.metadata.param_specs
    assert specs["short_weight"].default == 4.0
    assert specs["medium_weight"].default == 2.0
    assert specs["long_weight"].default == 1.0

from __future__ import annotations

import numpy as np
import pandas as pd


def test_obv_dsl_execution_matches_legacy_numeric_kernel() -> None:
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.technical.signal import OBV as LegacyOBV
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    load_all()
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    price = pd.DataFrame(
        {"A": [10, 11, 11, 9, np.nan, 10, 12, 12, 11, 13, 12, 14],
         "B": [20, 19, 21, 21, 22, 20, 20, 23, 22, np.nan, 24, 25]},
        index=dates,
        dtype=float,
    )
    volume = pd.DataFrame(
        {"A": np.arange(100.0, 112.0), "B": np.arange(200.0, 212.0)},
        index=dates,
    )
    data = {}
    for name, frame in {"close": price, "volume": volume}.items():
        series = frame.stack(dropna=False)
        series.index.names = ["timestamp", "instrument"]
        data[name] = series

    expr = DSLParser(surface="compat_research", dialect="native").parse(
        "OBV(close, volume)"
    )
    actual = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=data),
        run_mode="research",
    ).run(Factor(name="obv", expr=expr))["result"].sort_index()
    expected = LegacyOBV().calculate(price, volume).stack(dropna=False)
    expected.index.names = ["timestamp", "instrument"]
    pd.testing.assert_series_equal(
        actual, expected.sort_index(), check_names=False, rtol=0.0, atol=0.0
    )
    for instrument in ("A", "B"):
        assert actual.loc[(dates[0], instrument)] == 0.0


def test_obv_is_registered_with_two_panel_series_contract() -> None:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.operator_surface import classify_canonical
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    assert classify_canonical("OBV") == "extended"
    for backend in ("pandas_numpy", "polars"):
        operator = OperatorRegistry.get("OBV", backend=backend)
        assert operator is not None
        assert operator.metadata.param_names == ["price", "volume"]
        assert operator.metadata.return_type == "series"

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.dsl_parser import DSLParser
from factor_engine.backend.operator_types import OPERATOR_SIGNATURES, TypeKind
from factor_engine.ir.analyzer import Analyzer, TypedInputContractError


LIMIT_EVENT_FORMULAS = (
    ("ashare_limit_up_touch", "ashare_limit_up_touch(field('high', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)"),
    ("ashare_limit_down_touch", "ashare_limit_down_touch(field('low', table='StockDailyBar'), field('low_limit', table='StockDailyBar'), 0.005)"),
    (
        "ashare_limit_one_price",
        "ashare_limit_one_price(field('open', table='StockDailyBar'), field('high', table='StockDailyBar'), field('low', table='StockDailyBar'), field('close', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), field('low_limit', table='StockDailyBar'), 'up', 0.005)",
    ),
    ("ashare_limit_failed", "ashare_limit_failed(field('high', table='StockDailyBar'), field('close', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)"),
    (
        "ashare_open_at_upper_limit",
        "ashare_open_at_upper_limit(field('open', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)",
    ),
    (
        "ashare_limit_open_failed",
        "ashare_limit_open_failed(field('open', table='StockDailyBar'), field('low', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)",
    ),
)


def _lower(source: str):
    return Analyzer().lower(DSLParser(surface="compat_research").parse(source)).ir


@pytest.mark.parametrize("canonical,formula", LIMIT_EVENT_FORMULAS)
def test_limit_event_signatures_and_ir_output_are_formal_event_bool(canonical, formula):
    assert OPERATOR_SIGNATURES[canonical].output is TypeKind.SERIES_BOOL
    event = _lower(formula)
    assert event.attrs["dtype"] == "bool"
    assert event.semantic_attrs["semantic_kind"] == "EventBool"


def test_no_read_factor_engine_compiles_nested_limit_event_response():
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.sources.datasource import DataSource

    class NoReadDataSource(DataSource):
        def _reject(self, method, *args, **kwargs):
            raise AssertionError(f"compile must not read through {method}")

        def load_column(self, *args, **kwargs):
            return self._reject("load_column", *args, **kwargs)

        def load_columns(self, *args, **kwargs):
            return self._reject("load_columns", *args, **kwargs)

        def prefetch_columns(self, *args, **kwargs):
            return self._reject("prefetch_columns", *args, **kwargs)

        def scan_polars_long(self, *args, **kwargs):
            return self._reject("scan_polars_long", *args, **kwargs)

        def scan_index_long(self, *args, **kwargs):
            return self._reject("scan_index_long", *args, **kwargs)

    formula = (
        'event_historical_response_mean(field("ret", table="StockDailyBarAdj"), '
        'ashare_limit_up_touch(field("high", table="StockDailyBar"), '
        'field("high_limit", table="StockDailyBar"), 0.005))'
    )
    load_all()
    expr = DSLParser(surface="compat_research").parse(formula)
    _plan, analysis = FactorEngine(PandasBackend(), NoReadDataSource(), run_mode="research").compile(
        Factor(name="limit-event-response", expr=expr, source_expr=formula,
               surface="compat_research")
    )
    root = analysis.ir
    assert root.inputs[0].semantic_attrs["semantic_kind"] == "ReturnDecimal"
    assert root.inputs[1].semantic_attrs["semantic_kind"] == "EventBool"


def test_continuous_price_is_still_rejected_as_event_input():
    with pytest.raises(TypedInputContractError, match="parameter 'event'.*PriceContinuous"):
        _lower(
            'event_historical_response_mean(ret, '
            'field("close", table="StockDailyBarAdj"))'
        )


def test_limit_event_runtime_keeps_zero_one_nan_missing_semantics():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    index = pd.date_range("2025-01-01", periods=4)
    high = pd.DataFrame({"A": [9.9, 10.0, np.nan, 10.1]}, index=index)
    limit = pd.DataFrame({"A": [10.0, 10.0, 10.0, np.nan]}, index=index)
    result = OperatorRegistry.get(
        "ashare_limit_up_touch", backend="pandas_numpy"
    ).calculate(high, limit, tick_tolerance=0.005)
    np.testing.assert_allclose(
        result["A"].to_numpy(), np.array([0.0, 1.0, np.nan, np.nan]), equal_nan=True
    )

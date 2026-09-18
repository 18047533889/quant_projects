"""R27: final rolling defaults, backend contracts, DSL binding and history."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.ir.analyzer import Analyzer
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource
from tests.runtime.test_r23_extreme_numeric_batch import _coherent_broker


DEFAULTS = {
    "ts_mean": {"window": 20, "min_periods": 1},
    "ts_sum": {"window": 20, "min_periods": 1},
    "ts_std": {"window": 20, "ddof": 1},
    "ts_var": {"window": 20, "ddof": 1, "min_periods": 1},
    "ts_min": {"window": 20},
    "ts_max": {"window": 20},
    "ts_median": {"window": 20},
    "ts_rank": {"window": 20, "min_periods": 1},
    "ts_quantile": {"d": 20, "q": 0.5},
    "ts_skew": {"window": 20},
    "ts_kurt": {"window": 20},
    "ts_mad": {"window": 20, "min_periods": None, "scale": 1.0},
    "ts_product": {"window": 20, "min_periods": None, "skipna": True},
    "ts_argmax": {"window": 20, "min_periods": 1},
    "ts_argmin": {"window": 20, "min_periods": 1},
    "ts_decay_linear": {"window": 20},
    "ts_time_slope": {"window": 20, "min_periods": None},
    "ts_zscore": {
        "window": 20,
        "min_periods": 1,
        "null_policy": "ignore",
        "nan_policy": "propagate",
        "includes_current_bar": True,
        "ddof": 1,
        "zero_std_policy": "zero",
    },
    "ts_delta": {"n": 1},
    "ts_delay": {"n": 1},
    "ts_topk_mean": {"window": 20, "k": 5, "min_periods": 1},
    "ts_bottomk_mean": {"window": 20, "k": 5, "min_periods": 1},
}


def _spec(spec):
    return (
        spec.dtype,
        spec.min,
        spec.max,
        spec.choices,
        spec.searchable,
        spec.active_when,
        spec.history_semantics,
        spec.history_formula,
        spec.default,
        spec.equivalence,
        spec.param_role,
    )


def test_final_backend_contracts_defaults_and_analyzer_history_agree():
    load_all()
    for name, defaults in DEFAULTS.items():
        catalog = OperatorRegistry._catalog[name]
        assert {key: spec.default for key, spec in catalog["param_specs"].items()} == defaults
        pandas_meta = OperatorRegistry.get(name, "pandas_numpy", mode="research").metadata
        polars_meta = OperatorRegistry.get(name, "polars", mode="research").metadata
        assert list(pandas_meta.param_names) == list(polars_meta.param_names) == list(catalog["param_names"])
        assert {key: _spec(value) for key, value in pandas_meta.param_specs.items()} == {
            key: _spec(value) for key, value in polars_meta.param_specs.items()
        }
        omitted = Analyzer().lower(F(name)(col("x")))
        explicit = Analyzer().lower(F(name)(col("x"), **defaults))
        assert omitted.lookback == explicit.lookback, name


def test_run_many_omitted_defaults_equal_explicit_and_core_oracles():
    load_all()
    dates = pd.date_range("2026-01-01", periods=32)
    assets = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])
    values = np.column_stack([
        1.01 + np.arange(32) / 1000,
        1.03 + np.sin(np.arange(32) / 4) / 100,
    ])
    source = InMemorySeriesSource(data={"x": pd.Series(values.ravel(), index=index)})
    source.instrument_filter = tuple(assets)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {"x": "float64"}
    engine = FactorEngine(build_backend("pandas"), source, run_mode="research")
    engine.resource_broker = _coherent_broker()
    factors = []
    for name, defaults in DEFAULTS.items():
        factors.extend([
            Factor(name=f"{name}__omitted", expr=F(name)(col("x"))),
            Factor(name=f"{name}__explicit", expr=F(name)(col("x"), **defaults)),
        ])
    results = engine.run_many(factors)["results"]
    for name in DEFAULTS:
        pd.testing.assert_series_equal(
            results[f"{name}__omitted"],
            results[f"{name}__explicit"],
            check_names=False,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )

    wide = pd.DataFrame(values, index=dates, columns=assets)
    oracles = {
        "ts_mean": wide.rolling(20, min_periods=1).mean(),
        "ts_sum": wide.rolling(20, min_periods=1).sum(),
        "ts_std": wide.rolling(20, min_periods=1).std(ddof=1),
        "ts_var": wide.rolling(20, min_periods=1).var(ddof=1),
        "ts_min": wide.rolling(20, min_periods=1).min(),
        "ts_max": wide.rolling(20, min_periods=1).max(),
        "ts_median": wide.rolling(20, min_periods=1).median(),
        "ts_argmax": wide.rolling(20, min_periods=1).apply(
            lambda a: len(a) - 1 - np.flatnonzero(a == np.nanmax(a))[-1], raw=True
        ),
        "ts_argmin": wide.rolling(20, min_periods=1).apply(
            lambda a: len(a) - 1 - np.flatnonzero(a == np.nanmin(a))[-1], raw=True
        ),
    }
    for name, expected_wide in oracles.items():
        expected = expected_wide.stack(dropna=False).rename_axis(index.names)
        expected.index = expected.index.set_levels(
            [expected.index.levels[0], expected.index.levels[1]]
        )
        pd.testing.assert_series_equal(
            results[f"{name}__omitted"].sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )

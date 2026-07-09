# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER9：扩展 ts / group / cum / EWM / 比较逻辑 parity CI。"""

from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from cleaned_operators.operator_policy import (
    POLARS_PARITY_VERIFIED_TIER9,
    POLARS_PRODUCTION_SAFE,
)
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0], index=idx)
    y = pd.Series([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 40.0], index=idx)
    grp = pd.Series([1, 1, 1, 1, 2, 2, 2, 2], index=idx, dtype=float)
    z = pd.Series(
        [float("nan"), 2.0, 3.0, 4.0, 5.0, float("nan"), 7.0, 8.0],
        index=idx,
    )
    return InMemorySeriesSource(data={"x": x, "y": y, "grp": grp, "z": z})


def _run_pair(source, expr):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
        a = eng_pd.run(Factor(name="t", expr=expr))["result"]
        b = eng_pl.run(Factor(name="t", expr=expr))["result"]
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
    return a, b


def _parity_exprs():
    x, y, z = col("x"), col("y"), col("z")
    gt = make_cleaned_call_factory("gt")
    return {
        "ts_mad": make_cleaned_call_factory("ts_mad")(x, 2),
        "ts_skew": make_cleaned_call_factory("ts_skew")(x, 3),
        "ts_kurt": make_cleaned_call_factory("ts_kurt")(x, 4),
        "ts_quantile": make_cleaned_call_factory("ts_quantile")(x, 3, 0.5),
        "ts_product": make_cleaned_call_factory("ts_product")(x, 2),
        "ts_argmax": make_cleaned_call_factory("ts_argmax")(x, 2),
        "ts_argmin": make_cleaned_call_factory("ts_argmin")(x, 2),
        "ts_regression": make_cleaned_call_factory("ts_regression")(x, y, 3),
        "ts_ratio": make_cleaned_call_factory("ts_ratio")(x),
        "ts_max_buildup": make_cleaned_call_factory("ts_max_buildup")(x, 2),
        "ts_moment": make_cleaned_call_factory("ts_moment")(x, 3, 2),
        "ts_ratio": make_cleaned_call_factory("ts_ratio")(x),
        "Slope": make_cleaned_call_factory("Slope")(x, 3),
        "c_percentile": make_cleaned_call_factory("c_percentile")(x, 0.5),
        "cs_mad": make_cleaned_call_factory("cs_mad")(x),
        "cs_mad_zscore": make_cleaned_call_factory("cs_mad_zscore")(x),
        "quantile": make_cleaned_call_factory("quantile")(x, 0.5),
        "group_percentile": make_cleaned_call_factory("group_percentile")(x, col("grp"), 0.5),
        "group_decay_linear": make_cleaned_call_factory("group_decay_linear")(x, col("grp"), 2),
        "group_winsorize": make_cleaned_call_factory("group_winsorize")(x, col("grp")),
        "ewm_std": make_cleaned_call_factory("ewm_std")(x, 3),
        "ewm_var": make_cleaned_call_factory("ewm_var")(x, 3),
        "ewm_corr": make_cleaned_call_factory("ewm_corr")(x, y, 3),
        "ewm_cov": make_cleaned_call_factory("ewm_cov")(x, y, 3),
        "cum_sum": make_cleaned_call_factory("cum_sum")(x),
        "cum_max": make_cleaned_call_factory("cum_max")(x),
        "cum_min": make_cleaned_call_factory("cum_min")(x),
        "cum_prod": make_cleaned_call_factory("cum_prod")(x),
        "cum_delta": make_cleaned_call_factory("cum_delta")(x),
        "expanding_std": make_cleaned_call_factory("expanding_std")(x),
        "expanding_rank": make_cleaned_call_factory("expanding_rank")(x),
        "expanding_mean": make_cleaned_call_factory("expanding_mean")(x),
        "expanding_sum": make_cleaned_call_factory("expanding_sum")(x),
        "count": make_cleaned_call_factory("count")(x),
        "power": make_cleaned_call_factory("power")(x, 2),
        "floor": make_cleaned_call_factory("floor")(x),
        "ceil": make_cleaned_call_factory("ceil")(x),
        "inverse": make_cleaned_call_factory("inverse")(x),
        "minimum": make_cleaned_call_factory("minimum")(x, y),
        "maximum": make_cleaned_call_factory("maximum")(x, y),
        "signed_sqrt": make_cleaned_call_factory("signed_sqrt")(x),
        "signed_log": make_cleaned_call_factory("signed_log")(x),
        "log_abs": make_cleaned_call_factory("log_abs")(x),
        "is_nan": make_cleaned_call_factory("is_nan")(z),
        "gt": gt(x, y),
        "lt": make_cleaned_call_factory("lt")(x, y),
        "ge": make_cleaned_call_factory("ge")(x, y),
        "le": make_cleaned_call_factory("le")(x, y),
        "eq": make_cleaned_call_factory("eq")(x, y),
        "ne": make_cleaned_call_factory("ne")(x, y),
        "and_": make_cleaned_call_factory("and_")(gt(x, y), gt(y, x)),
        "or_": make_cleaned_call_factory("or_")(gt(x, y), gt(y, x)),
        "not_": make_cleaned_call_factory("not_")(gt(x, y)),
        "fillna_interpolate": make_cleaned_call_factory("fillna_interpolate")(z),
    }


def test_parity_tier9_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER9 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER9))
def test_polars_parity_verified_tier9(source, canonical):
    exprs = _parity_exprs()
    assert canonical in exprs, f"缺少 parity fixture: {canonical}"
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-5, atol=1e-5)

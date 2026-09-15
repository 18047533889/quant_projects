import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.cleaned_operators.robust_tail  # noqa: F401
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "ts_lower_partial_moment", "ts_upper_partial_moment", "ts_expected_shortfall",
    "ts_quantile_skew", "ts_quantile_kurtosis", "ts_tail_ratio",
    "ts_extreme_cluster_ratio",
)


def _op(name, backend):
    return OperatorRegistry.get(name, backend=backend)


def _last(result):
    if isinstance(result, pl.DataFrame):
        result = result.to_pandas()
    return float(result.iloc[-1, 0])


def _call(name, backend, values, positional, *args, **kwargs):
    panel = pd.DataFrame({"A": values}, dtype=float)
    if backend == "polars":
        panel = pl.from_pandas(panel)
    op = _op(name, backend)
    return op.calculate(panel, *args) if positional else op.calculate(x=panel, **kwargs)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("positional", [False, True])
def test_registry_backends_match_independent_tail_references(backend, positional):
    x = np.arange(1.0, 11.0)
    cases = [
        ("ts_lower_partial_moment", x, (5, 9.0, 2.0, 5), {"window": 5, "threshold": 9.0, "order": 2.0, "min_periods": 5}, 2.8),
        ("ts_upper_partial_moment", x, (5, 7.0, 2.0, 5), {"window": 5, "threshold": 7.0, "order": 2.0, "min_periods": 5}, 2.8),
        ("ts_expected_shortfall", x, (5, 0.4, "lower", 2), {"window": 5, "q": 0.4, "side": "lower", "min_tail_count": 2}, 6.5),
        ("ts_quantile_skew", np.array([0., 1., 2., 3., 20.]), (5, .1, .5, .9, 5), {"window": 5, "q_low": .1, "q_mid": .5, "q_high": .9, "min_periods": 5}, None),
        ("ts_quantile_kurtosis", np.arange(8.0), (8, (.1, .9), (.25, .75), 8), {"window": 8, "outer": (.1, .9), "inner": (.25, .75), "min_periods": 8}, None),
        ("ts_tail_ratio", np.array([-8., -4., -2., 1., 3.]), (5, .2, .8, 5), {"window": 5, "q_low": .2, "q_high": .8, "min_periods": 5}, None),
        ("ts_extreme_cluster_ratio", np.array([1., 8., 9., np.nan, 2., 10., 11.]), (7, 7.0, .9, "upper", 2), {"window": 7, "threshold": 7.0, "q": .9, "side": "upper", "min_periods": 2}, .5),
    ]
    for name, values, args, kwargs, expected in cases:
        if name == "ts_quantile_skew":
            ql, qm, qh = np.quantile(values, [.1, .5, .9])
            expected = (qh + ql - 2 * qm) / (qh - ql)
        elif name == "ts_quantile_kurtosis":
            expected = (np.quantile(values, .9) - np.quantile(values, .1)) / (np.quantile(values, .75) - np.quantile(values, .25))
        elif name == "ts_tail_ratio":
            expected = abs(np.quantile(values, .8)) / abs(np.quantile(values, .2))
        result = _call(name, backend, values, positional, *args, **kwargs)
        assert _last(result) == pytest.approx(expected)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_prefix_causality_and_invalid_domains(backend):
    base = np.array([-3., 1., 4., 8., 2., 10., 6., 12.])
    for name in NAMES:
        op = _op(name, backend)
        x = pd.DataFrame({"A": base})
        longer = pd.DataFrame({"A": np.r_[base, 999.]})
        if backend == "polars":
            x, longer = pl.from_pandas(x), pl.from_pandas(longer)
        first = op.calculate(x)
        prefix = op.calculate(longer)
        if backend == "polars":
            first, prefix = first.to_pandas(), prefix.to_pandas()
        pd.testing.assert_frame_equal(first, prefix.iloc[:-1].reset_index(drop=True))
    panel = pl.DataFrame({"A": base}) if backend == "polars" else pd.DataFrame({"A": base})
    with pytest.raises((TypeError, ValueError)):
        _op("ts_tail_ratio", backend).calculate(panel, window=5, min_periods=6)
    with pytest.raises((TypeError, ValueError)):
        _op("ts_quantile_kurtosis", backend).calculate(panel, window=8, outer=(.3, .7), inner=(.2, .8), min_periods=8)
    with pytest.raises((TypeError, ValueError)):
        _op("ts_extreme_cluster_ratio", backend).calculate(panel, threshold=-1.0, side="absolute")


def test_all_scalar_contracts_have_defaults_types_and_roles():
    for name in NAMES:
        metadata = _op(name, "pandas_numpy").metadata
        assert set(metadata.param_specs) == set(metadata.param_names) - {"x"}
        for spec in metadata.param_specs.values():
            assert spec.default is not None
            assert spec.dtype is not None or name == "ts_extreme_cluster_ratio"
            assert spec.param_role is not None


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_structured_tuple_and_union_contracts_preserve_legacy_inputs(backend):
    panel = pd.DataFrame({"A": np.arange(8.0)})
    if backend == "polars":
        panel = pl.from_pandas(panel)
    kurt = _op("ts_quantile_kurtosis", backend)
    from_list = kurt.calculate(panel, window=8, outer=[.1, .9], inner=[.25, .75], min_periods=8)
    assert _last(from_list) == pytest.approx((np.quantile(np.arange(8.0), .9) - np.quantile(np.arange(8.0), .1)) / 3.5)
    for bad in ((.1,), (.1, .9, .95), (.1, np.inf), (.1, True)):
        with pytest.raises((TypeError, ValueError)):
            kurt.calculate(panel, window=8, outer=bad, inner=(.25, .75), min_periods=8)

    cluster = _op("ts_extreme_cluster_ratio", backend)
    tied_panel = pd.DataFrame({"A": [0., 1., 2., 3., 4., 5., 7., 7.]})
    if backend == "polars":
        tied_panel = pl.from_pandas(tied_panel)
    assert np.isfinite(_last(cluster.calculate(tied_panel, window=8, threshold="QuAnTiLe", min_periods=2)))
    numeric_string = cluster.calculate(panel, window=8, threshold="4.0", q=.9, side="upper", min_periods=2)
    numeric_float = cluster.calculate(panel, window=8, threshold=4.0, q=.9, side="upper", min_periods=2)
    assert _last(numeric_string) == pytest.approx(_last(numeric_float))
    for bad in (np.inf, "inf", "not-a-threshold", True):
        with pytest.raises((TypeError, ValueError)):
            cluster.calculate(panel, window=8, threshold=bad, min_periods=2)


def test_final_polars_registry_uses_declared_extreme_delegate():
    import factor_engine.cleaned_operators.polars_native.ts_advanced_batch1  # noqa: F401
    op = _op("ts_extreme_cluster_ratio", "polars")
    assert type(op).__name__ == "TSExtremeClusterRatioPolarsNative"
    assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"

import numpy as np
import pandas as pd
import polars as pl
import pytest
from decimal import Decimal, localcontext
import subprocess
import sys
import warnings
from factor_engine.cleaned_operators.polars_native.cs_zscore_finite_anchor_v2 import (
    _finite_anchor_expr,
    cs_zscore_finite_anchor_v2_numpy,
)


def oracle(row, ddof=1, constant=0.0):
    with localcontext() as ctx:
        ctx.prec = 1200
        vals = [Decimal.from_float(float(x)) for x in row if np.isfinite(x)]
        if any(np.isinf(x) for x in row):
            return [constant if not np.isnan(x) else np.nan for x in row]
        if len(vals) <= ddof or len(vals) < 2 or min(vals) == max(vals):
            return [constant if not np.isnan(x) else np.nan for x in row]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / (len(vals) - ddof)
        sd = var.sqrt()
        z = [
            (
                float((Decimal.from_float(float(x)) - mean) / sd)
                if np.isfinite(x)
                else np.nan
            )
            for x in row
        ]
        return z


@pytest.mark.parametrize(
    "row",
    [
        [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6],
        [np.finfo(float).max, -np.finfo(float).max, 0.0, 1.0],
        [
            np.nextafter(0.0, 1.0),
            2 * np.nextafter(0.0, 1.0),
            4 * np.nextafter(0.0, 1.0),
            8 * np.nextafter(0.0, 1.0),
        ],
        [2.0, 2.0, 2.0, np.nan],
        [1.0, np.inf, 3.0, np.nan],
        [1.0, 2.0, np.nan, 4.0],
    ],
)
@pytest.mark.parametrize("ddof", [0, 1, 2, 4])
def test_numeric_matches_decimal(row, ddof):
    panel = pd.DataFrame([row], columns=list("abcd"))
    expected = oracle(row, ddof)
    ref = cs_zscore_finite_anchor_v2_numpy(panel, ddof=ddof).to_numpy()[0]
    got = _finite_anchor_expr(pl.from_pandas(panel), ddof, 0.0).to_numpy()[0]
    np.testing.assert_allclose(ref, expected, rtol=5e-12, atol=5e-12, equal_nan=True)
    np.testing.assert_allclose(got, expected, rtol=5e-12, atol=5e-12, equal_nan=True)
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fp = cs_zscore(np.asarray(row, dtype=np.float64), ddof=ddof)
    np.testing.assert_allclose(fp, expected, rtol=5e-12, atol=5e-12, equal_nan=True)


def test_time_axis_and_null_preserved():
    p = pl.DataFrame(
        {"date": ["d1", "d2"], "a": [1.0, None], "b": [2.0, 3.0], "c": [4.0, 5.0]}
    )
    out = _finite_anchor_expr(p, 1, 0.0)
    assert out["date"].equals(p["date"]) and out["a"][1] is None


@pytest.mark.parametrize("ddof", [1.5, -1, True])
def test_rejects_invalid_ddof(ddof):
    with pytest.raises(ValueError):
        cs_zscore_finite_anchor_v2_numpy(pd.DataFrame({"a": [1.0], "b": [2.0]}), ddof)


@pytest.mark.parametrize("constant", [np.longdouble("1e400"), 1 + 2j, np.inf, np.nan])
def test_rejects_nonfinite_or_nonreal_constant(constant):
    with pytest.raises(ValueError):
        cs_zscore_finite_anchor_v2_numpy(
            pd.DataFrame({"a": [1.0], "b": [2.0]}), constant_value=constant
        )


def test_all_null_polars_column_and_float64_reference():
    p = pl.DataFrame({"date": ["d1"], "a": [None], "b": [2], "c": [4]})
    result = _finite_anchor_expr(p, 1, 0.0)
    assert result.schema["a"] == pl.Float64
    assert result["a"][0] is None
    integer = cs_zscore_finite_anchor_v2_numpy(
        pd.DataFrame({"a": [2**60], "b": [2**60 + 2]})
    )
    assert all(dtype == np.dtype("float64") for dtype in integer.dtypes)


def test_empty_axis_only_and_reserved_prefix_inputs():
    empty = pl.DataFrame(schema={"date": pl.String, "a": pl.Float64, "b": pl.Float64})
    assert _finite_anchor_expr(empty, 1, 0.0).height == 0
    with pytest.raises(ValueError):
        _finite_anchor_expr(pl.DataFrame({"date": ["d"]}), 1, 0.0)
    with pytest.raises(ValueError):
        _finite_anchor_expr(pl.DataFrame(), 1, 0.0)
    with pytest.raises(ValueError):
        cs_zscore_finite_anchor_v2_numpy(pd.DataFrame({"date": ["d"]}))
    with pytest.raises(ValueError):
        cs_zscore_finite_anchor_v2_numpy(pd.DataFrame())
    with pytest.raises(TypeError):
        _finite_anchor_expr(pl.Series("a", [1.0, 2.0]), 1, 0.0)
    with pytest.raises(TypeError):
        cs_zscore_finite_anchor_v2_numpy(pd.Series([1.0, 2.0]))
    panel = pl.DataFrame({"__csa_count": [2.0], "a": [1.0], "b": [3.0]})
    result = _finite_anchor_expr(panel, 1, 0.0)
    assert result.columns == panel.columns
    np.testing.assert_allclose(result.to_numpy(), [oracle([2.0, 1.0, 3.0])])
    two_columns = _finite_anchor_expr(
        pl.DataFrame({"left": [1.0], "right": [3.0]}), 1, 0.0
    )
    np.testing.assert_allclose(two_columns.to_numpy(), [oracle([1.0, 3.0], ddof=1)])


def test_wide_ddof_short_circuit_and_complex_rejection():
    result = _finite_anchor_expr(pl.DataFrame({"a": [1.0], "b": [2.0]}), 10**100, 4.0)
    assert result["a"][0] == 4.0
    complex_panel = pd.DataFrame({"a": np.array([1 + 2j]), "b": np.array([2 + 3j])})
    with pytest.raises(TypeError):
        cs_zscore_finite_anchor_v2_numpy(complex_panel)
    bool_panel = pd.DataFrame({"a": pd.Series([True], dtype="boolean"), "b": [False]})
    with pytest.raises(TypeError):
        cs_zscore_finite_anchor_v2_numpy(bool_panel)
    with pytest.raises(TypeError):
        _finite_anchor_expr(pl.DataFrame({"a": [True], "b": [False]}), 0, 0.0)
    string_panel = pl.DataFrame({"a": ["not-a-number"], "b": ["also-not-a-number"]})
    with pytest.raises(TypeError):
        _finite_anchor_expr(string_panel, 0, 4.0)
    huge_ddof = _finite_anchor_expr(
        pl.DataFrame({"a": [1.0], "b": [2.0]}), 10**1000, 4.0
    )
    assert huge_ddof["a"][0] == huge_ddof["b"][0] == 4.0
    with pytest.raises(TypeError):
        _finite_anchor_expr(pl.DataFrame({"a": ["x"], "b": ["y"]}), 10**1000, 4.0)


def test_explicit_registration_in_isolated_process():
    script = r"""
from factor_engine.cleaned_operators.registry import OperatorRegistry
print("lifecycle-before", OperatorRegistry.lifecycle())
assert OperatorRegistry.lifecycle() == "building"
from factor_engine.cleaned_operators.polars_native.cs_zscore_finite_anchor_v2 import register_cs_zscore_finite_anchor_v2
assert register_cs_zscore_finite_anchor_v2() == ("cs_zscore_finite_anchor_v2",)
reference = OperatorRegistry.get("cs_zscore_finite_anchor_v2", "pandas_numpy", mode="any")
native = OperatorRegistry.get("cs_zscore_finite_anchor_v2", "polars", mode="any")
assert reference is not None and native is not None
import pandas as pd, polars as pl
pandas = pd.DataFrame([[1e16, 1e16 + 2, 1e16 + 4]], columns=["a", "b", "c"])
polars = pl.from_pandas(pandas)
left = reference.calculate(pandas, 0, 2.0)
right = native.calculate(polars, 0, 2.0)
kw_left = reference.calculate(pandas, ddof=0, constant_value=2.0)
kw_right = native.calculate(polars, ddof=0, constant_value=2.0)
import numpy as np
np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(kw_left.to_numpy(), kw_right.to_numpy(), rtol=1e-12, atol=1e-12)
for op, panel in ((reference, pandas), (native, polars)):
    try: op.calculate(panel, ddof=1.5)
    except (TypeError, ValueError): pass
    else: raise AssertionError("fractional ddof accepted")
    try: op.calculate(panel, unsupported=True)
    except (TypeError, ValueError): pass
    else: raise AssertionError("unknown kwarg accepted")
bool_pandas = pd.DataFrame({"a": pd.Series([True], dtype="boolean"), "b": [False]})
bool_polars = pl.DataFrame({"a": [True], "b": [False]})
for op, panel in ((reference, bool_pandas), (native, bool_polars)):
    try: op.calculate(panel)
    except (TypeError, ValueError): pass
    else: raise AssertionError("boolean numeric panel accepted")
from factor_engine.backend.operator_capability import backend_status
assert backend_status("cs_zscore_finite_anchor_v2", "polars", production_mode=True) != "production_safe"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

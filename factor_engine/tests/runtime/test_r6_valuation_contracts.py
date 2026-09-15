"""Real final-registry valuation calls, not metadata-only certification."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.valuation import ops_v2 as module
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

def panels(op):
    index = pd.bdate_range("2024-01-01", periods=16)
    grid = np.arange(96, dtype=float).reshape(16, 6)
    result = {}
    for i, name in enumerate(op.metadata.panel_params):
        values = 2 + (i+1) * .13 * grid + np.sin(grid / (i+2)) * .2
        result[name] = pd.DataFrame(values, index=index, columns=list("ABCDEF"))
    if "change_date" in result:
        result["change_date"] = pd.DataFrame(index[0], index=index, columns=list("ABCDEF"))
    return result

@pytest.mark.parametrize("name", module._CANONICALS)
def test_each_final_valuation_positional_keyword_prefix_and_schema(name):
    op = OperatorRegistry.get(name, "pandas_numpy", mode="research")
    inputs = panels(op)
    actual = op.calculate(**inputs)
    pd.testing.assert_frame_equal(actual, op.calculate(*inputs.values()))
    first, *rest = inputs
    pd.testing.assert_frame_equal(actual, op.calculate(inputs[first], **{k: inputs[k] for k in rest}))
    assert actual.shape == next(iter(inputs.values())).shape
    assert np.isfinite(actual.to_numpy()).any(), name
    assert not np.isinf(actual.to_numpy()).any(), name
    short = op.calculate(**{k: v.iloc[:10] for k, v in inputs.items()})
    pd.testing.assert_frame_equal(short, actual.iloc[:10])
    _, _, verified = _parameter_contract(op, op.metadata.panel_params)
    assert verified, name
    if len(inputs) > 1:
        shifted = dict(inputs)
        key = list(inputs)[-1]
        shifted[key] = inputs[key].rename(columns={"A": "wrong"})
        with pytest.raises(ValueError, match="axes|misaligned"):
            op.calculate(**shifted)

@pytest.mark.parametrize("name,a,b,expected", [
    ("free_float_ratio", 4., 10., .4),
    ("free_to_circulating_ratio", 4., 8., .5),
    ("a_share_cap_ratio", 9., 10., .9),
    ("free_float_turnover", 2., 8., .25),
    ("valuation_pe_ttm_lyr_gap", -4., 2., np.log(2)),
    ("valuation_pcf_definition_gap", -4., 2., np.log(2)),
    ("valuation_pe_gap_signed_log", -4., 2., -np.log(5)-np.log(3)),
    ("valuation_pcf_gap_signed_log", -4., 2., -np.log(5)-np.log(3)),
    ("valuation_pe_gap_positive", 4., 2., np.log(2)),
    ("valuation_pcf_gap_positive", 4., 2., np.log(2)),
    ("market_cap_free_cap_gap", 4., 2., np.log(2)),
    ("valuation_growth_mismatch", .2, .1, .1),
])
def test_independent_elementwise_values(name, a, b, expected):
    op = OperatorRegistry.get(name, "pandas_numpy", mode="research")
    inputs = [pd.DataFrame([[x]]) for x in (a, b)]
    assert op.calculate(*inputs).iat[0, 0] == pytest.approx(expected)

def test_disagreement_never_winsorizes_missing_or_uses_unscaled_constant():
    op = OperatorRegistry.get("valuation_cashflow_disagreement", "pandas_numpy", mode="research")
    x = pd.DataFrame([[1., 2., 3., 4., 5., np.inf]])
    result = op.calculate(x, x*2, x*3, x/10, x/20)
    assert np.isfinite(result.iloc[0, :5]).all()
    assert np.isnan(result.iat[0, 5])
    constants = [pd.DataFrame([[float(i)] * 6]) for i in range(1, 6)]
    assert op.calculate(*constants).isna().all().all()

def test_quality_wls_independent_linear_case_and_weights():
    op = OperatorRegistry.get("valuation_quality_mismatch", "pandas_numpy", mode="research")
    q = pd.DataFrame([[1., 2., 3., 4., 5., 6.]])
    w = pd.DataFrame([[1., 2., 3., 4., 5., 6.]])
    np.testing.assert_allclose(op.calculate(3+2*q, q, w), 0., atol=1e-12)
    y = q*q
    result = op.calculate(y, q, w)
    design = np.column_stack([np.ones(6), q.iloc[0]])
    weight = np.diag(w.iloc[0])
    beta = np.linalg.solve(design.T @ weight @ design, design.T @ weight @ y.iloc[0])
    np.testing.assert_allclose(result.iloc[0], y.iloc[0] - design @ beta, atol=1e-12)

def test_change_operators_independent_reference():
    index = pd.bdate_range("2024-01-05", periods=4)
    total = pd.DataFrame({"A": [10., 20., 25., 50.]}, index=index)
    circulating = pd.DataFrame({"A": [5., 12., 20., 30.]}, index=index)
    op = OperatorRegistry.get("capital_change_magnitude", "pandas_numpy", mode="research")
    np.testing.assert_allclose(op.calculate(total)["A"], [np.nan, 1., .25, 1.], equal_nan=True)
    for name in ("circulating_cap_ratio_change", "circulating_cap_unlock_proxy"):
        op = OperatorRegistry.get(name, "pandas_numpy", mode="research")
        np.testing.assert_allclose(op.calculate(circulating,total)["A"], [np.nan,.1,.2,-.2], equal_nan=True)
    dates = pd.DataFrame({"A": [pd.NaT, pd.Timestamp("2024-01-06"), pd.NaT, pd.NaT]}, index=index)
    op = OperatorRegistry.get("capital_change_age", "pandas_numpy", mode="research")
    np.testing.assert_allclose(op.calculate(change_date=dates)["A"], [np.nan,0.,1.,2.], equal_nan=True)

@pytest.mark.parametrize("scale", [True, 0., -1., np.inf, np.nan])
def test_scale_invalid_domain(scale):
    op = OperatorRegistry.get("valuation_growth_mismatch", "pandas_numpy", mode="research")
    with pytest.raises(ValueError):
        op.calculate(pd.DataFrame([[.2]]), pd.DataFrame([[.1]]), scale=scale)

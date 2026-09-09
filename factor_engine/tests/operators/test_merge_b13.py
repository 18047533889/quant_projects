import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.feature_geometry import TsBetaBreakScore, _true_beta


@pytest.mark.parametrize("xs,ys", [(1.,1.), (1e-7,1.), (1.,1e-13),
    (1e-200,1e200), (1e200,1e-200)])
def test_beta_break_two_to_three_dimensionless(xs, ys):
    x = np.sin(.51*np.arange(40.))
    y = x*np.r_[np.full(20,2.),np.full(20,3.)]
    out = TsBetaBreakScore().calculate(pd.DataFrame(y*ys), pd.DataFrame(x*xs),
                                      recent_window=20, prior_window=20)
    assert out.iloc[-1,0] == pytest.approx(.2, rel=1e-12)


@pytest.mark.parametrize("prior,recent,expected", [(2.,2.,0.), (-2.,3.,1.), (2.,-3.,-1.)])
def test_true_betas_not_correlations(prior,recent,expected):
    x = np.sin(.51*np.arange(40.))
    y = x*np.r_[np.full(20,prior),np.full(20,recent)]+7
    out = TsBetaBreakScore().calculate(pd.DataFrame(y),pd.DataFrame(x+100),
        recent_window=20,prior_window=20)
    assert out.iloc[-1,0] == pytest.approx(expected,abs=1e-12)


def test_direct_beta_restores_original_units():
    x = np.arange(20.)*1e-100
    y = np.arange(20.)*3e100
    assert _true_beta(y,x,5) == pytest.approx(3e200)


def test_constant_predictor_fails_closed_and_unit_is_dimensionless():
    op = TsBetaBreakScore()
    x = pd.DataFrame(np.ones(40))
    assert op.calculate(x,x,recent_window=20,prior_window=20).isna().all().all()
    assert "unit:dimensionless" in op.metadata.tags
    assert op.metadata.output_unit == "dimensionless"


def test_full_type_authority_has_dimensionless_unit_and_no_removed_window():
    from factor_engine.backend.operator_types import OPERATOR_SIGNATURES
    from factor_engine.api.columns import col
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.ir.analyzer import Analyzer
    signature = OPERATOR_SIGNATURES["ts_beta_break_score"]
    assert signature.output_unit == "dimensionless"
    assert [arg.name for arg in signature.inputs] == ["y", "x", "recent_window", "prior_window"]
    call = CleanedCall("ts_beta_break_score", (col("test_y"),col("test_x")),
                       (("recent_window",20),("prior_window",20)))
    assert Analyzer().lower(call).ir.semantic_attrs["unit"] == "dimensionless"

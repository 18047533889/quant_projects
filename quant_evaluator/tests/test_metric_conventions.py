import numpy as np
import pytest
from quant_evaluator.metrics.portfolio_stats import (
    compute_compound_annualized_return, compute_calmar_ratio, compute_maximum_drawdown,
    compute_long_short_returns, compute_sharpe_ratio,
)
from quant_evaluator.metrics.probe_portfolio.sharpe import compute_annualized_return


def test_cagr_and_calmar_share_one_numerator():
    r = np.tile([.1, -.08], 20)
    expected = np.prod(1+r)**(252/len(r))-1
    assert compute_compound_annualized_return(r) == pytest.approx(expected)
    assert compute_annualized_return(r) == pytest.approx(expected)
    assert compute_calmar_ratio(r) == pytest.approx(expected / compute_maximum_drawdown(r)[0])


@pytest.mark.parametrize('r', [[.1,-1.2,-2.,10.], [-1.,5.,3.], [-1.01,-1.01,.3]])
def test_annual_return_cannot_recover_after_bankruptcy(r):
    assert compute_annualized_return(np.array(r)) == -1.


def test_batch_mask_broadcast_matches_individual_factors():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(30, 40, 3)); r = rng.normal(0,.01,size=(30,40))
    mask = np.ones((30,40), bool); mask[:,0] = False
    batched = compute_long_short_returns(x,r,validity_mask=mask)
    for f in range(3):
        single = compute_long_short_returns(x[:,:,f],r,validity_mask=mask)
        for a,b in zip(batched,single): np.testing.assert_allclose(a[:,f],b,equal_nan=True)
    with pytest.raises(ValueError): compute_long_short_returns(x,r,validity_mask=np.ones((30,2),bool))


def test_empty_sharpe_has_same_output_contract():
    assert np.isscalar(compute_sharpe_ratio(np.array([])))
    assert np.isnan(compute_sharpe_ratio(np.array([])))
    assert compute_sharpe_ratio(np.empty((0,3))).shape == (3,)


def test_missing_policy_and_zero_drawdown_are_explicit():
    r = np.array([.1,np.nan,-.05])
    assert compute_compound_annualized_return(r,2) == pytest.approx(1.1*.95-1)
    with pytest.raises(ValueError): compute_compound_annualized_return(r,missing_return_policy='fail')
    assert np.isnan(compute_calmar_ratio(np.full(30,.01)))
    with pytest.raises(ValueError): compute_compound_annualized_return(r,periods_per_year=0)

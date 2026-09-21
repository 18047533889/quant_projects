import numpy as np
import pandas as pd
import pytest

from factor_preprocess.neutralization.ols import ols_neutralize, compute_exposures


def sample():
    # Residual [1,-1,-1,1] is orthogonal to intercept and x.
    values = pd.DataFrame({'date': [1]*4, 'asset_id': list('abcd'),
                           'value': [2., 2., 4., 8.]}, index=[7, 7, 2, 9])
    exposures = pd.DataFrame({'date': [1]*4, 'asset_id': list('abcd'),
                              'value': [0., 1., 2., 3.]})
    return values, exposures


def test_same_named_exposure_does_not_regress_factor_on_itself():
    values, exposures = sample()
    result = ols_neutralize(values, exposures, min_observations=3)
    np.testing.assert_allclose(result, [1., -1., -1., 1.], atol=1e-12)
    assert result.index.equals(values.index)


def test_coefficient_report_handles_same_named_exposure():
    values, exposures = sample()
    result = compute_exposures(values, exposures)
    np.testing.assert_allclose(result[['intercept', 'value_coef']], [[1., 2.]], atol=1e-12)


@pytest.mark.parametrize('fn', [ols_neutralize, compute_exposures])
def test_duplicate_exposure_keys_are_rejected_before_join(fn):
    values, exposures = sample()
    exposures = exposures.rename(columns={'value': 'size'})
    with pytest.raises(ValueError, match='unique'):
        fn(values, pd.concat([exposures, exposures.iloc[:1]]))


def test_insufficient_integer_values_produce_nan_not_integer_sentinel():
    values, exposures = sample()
    values['value'] = values['value'].astype(int)
    result = ols_neutralize(values, exposures.rename(columns={'value': 'size'}))
    assert result.isna().all()


def test_exposure_order_and_unrelated_factor_metadata_do_not_change_fit():
    values, exposures = sample()
    values['size'] = [100., 10., -5., 99.]
    exposures = exposures.rename(columns={'value': 'size'}).iloc[::-1]
    result = ols_neutralize(values, exposures, min_observations=3)
    np.testing.assert_allclose(result, [1., -1., -1., 1.], atol=1e-12)


def test_future_cross_sections_do_not_change_past_residuals():
    values, exposures = sample()
    future = values.assign(date=2, value=[1e9, -1e9, 5e8, -5e8])
    future_exposures = exposures.assign(date=2)
    result = ols_neutralize(pd.concat([values, future]),
                           pd.concat([exposures, future_exposures]), min_observations=3)
    np.testing.assert_allclose(result.iloc[:4], [1., -1., -1., 1.], atol=1e-12)


def test_redundant_exposure_columns_do_not_make_valid_projection_unavailable():
    values, exposures = sample()
    exposures = exposures.rename(columns={'value': 'size'})
    for k in range(8):
        exposures[f'redundant_{k}'] = exposures['size']*(k+1)
    result = ols_neutralize(values, exposures, min_observations=3)
    np.testing.assert_allclose(result, [1., -1., -1., 1.], atol=1e-12)


def test_intercept_is_counted_when_rejecting_saturated_neutralization():
    values = pd.DataFrame({'date': [1]*3, 'asset_id': list('abc'),
                           'value': [2., 7., -1.]})
    exposures = values[['date', 'asset_id']].assign(size=[-1., 0., 1.],
                                                  dummy=[0., 1., 0.])
    result = ols_neutralize(values, exposures, min_observations=3)
    assert result.isna().all(), "a saturated projection has no residual degrees of freedom"


def test_fully_explained_signal_cannot_be_ranked_from_roundoff_noise():
    values, exposures = sample()
    exposures = exposures.rename(columns={'value': 'size'})
    values['value'] = .1 + .3*exposures['size'].to_numpy()
    result = ols_neutralize(values, exposures, min_observations=3)
    np.testing.assert_array_equal(result, np.zeros(4))


def test_absent_industry_columns_cannot_change_numeric_projection():
    rng = np.random.default_rng(453)
    values = pd.DataFrame({'date': [1]*30, 'asset_id': np.arange(30),
                           'value': rng.normal(size=30)*1e12})
    exposures = values[['date', 'asset_id']].assign(size=rng.normal(size=30))
    ordinary = ols_neutralize(values, exposures)
    expanded = pd.concat([exposures, pd.DataFrame(
        np.zeros((30, 100)), columns=[f'absent_{j}' for j in range(100)])], axis=1)
    padded = ols_neutralize(values, expanded)
    np.testing.assert_array_equal(padded, ordinary)

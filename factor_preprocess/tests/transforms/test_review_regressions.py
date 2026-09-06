import numpy as np
import pandas as pd
from factor_preprocess.transforms.freshness import days_since_update, freshness_score, stale_data_indicator


def test_freshness_is_asset_local_with_duplicate_interleaved_index():
    values = pd.DataFrame({
        'asset_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'date': pd.to_datetime(['2026-01-01', '2026-01-01', '2026-01-02', '2026-01-02', '2026-01-04', '2026-01-04']),
        'value': [10.0, np.nan, np.nan, 20.0, np.nan, np.nan],
    }, index=[0, 0, 1, 1, 2, 2])
    age = days_since_update(values)
    np.testing.assert_allclose(age, [0, np.nan, 1, 0, 3, 2], equal_nan=True)
    pd.testing.assert_index_equal(age.index, values.index)
    np.testing.assert_allclose(freshness_score(values, 1), [1, 0, .5, 1, .125, .25])
    np.testing.assert_allclose(stale_data_indicator(values, 1), [0, np.nan, 0, 0, 1, 1], equal_nan=True)
    poisoned = values.copy()
    poisoned.iloc[-2:, poisoned.columns.get_loc('value')] = 999.0
    np.testing.assert_allclose(days_since_update(poisoned).iloc[:4], age.iloc[:4], equal_nan=True)


def test_freshness_preserves_fractional_calendar_days_with_timezone():
    values = pd.DataFrame({'asset_id': ['A', 'A'], 'date': pd.date_range('2026-01-01', periods=2, freq='12h', tz='Asia/Hong_Kong'), 'value': [1.0, np.nan]})
    np.testing.assert_allclose(days_since_update(values), [0, .5])

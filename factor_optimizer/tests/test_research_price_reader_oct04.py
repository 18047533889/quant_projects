"""Session-preserving reader behavior without COS or a copied dataset."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from factor_optimizer.research_price_reader import read_session_vwap


class Store:
    def __init__(self, sessions, duplicate=False):
        self.sessions = sessions
        self.calls = []
        self.duplicate = duplicate

    def read(self, name, **kwargs):
        self.calls.append(kwargs)
        start, end = kwargs['time_range']
        dates = self.sessions[(self.sessions >= start) & (self.sessions <= end)]
        dates = dates[dates != self.sessions[1]]
        rows = pd.DataFrame({'TradeDate': dates, 'Symbol': 'a', 'AdjVwap': 2.0})
        if self.duplicate:
            rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
        return SimpleNamespace(to_pandas=lambda: rows)


def test_chunked_read_preserves_missing_session_and_asset_axes():
    dates = pd.bdate_range('2020-01-01', periods=130)
    store = Store(dates)
    result = read_session_vwap(store, dates, ['a', 'b'])
    assert result.index.equals(dates)
    assert list(result.columns) == ['a', 'b']
    assert np.isnan(result.loc[dates[1], 'a'])
    assert np.isnan(result['b']).all()
    assert result.loc[dates[2], 'a'] == 2.0
    assert len(store.calls) == 3
    assert [c['query_budget'].max_scan_files for c in store.calls] == [64, 64, 2]


def test_duplicate_price_observation_is_rejected():
    dates = pd.bdate_range('2020-01-01', periods=3)
    with pytest.raises(ValueError, match='duplicate'):
        read_session_vwap(Store(dates, duplicate=True), dates, ['a'])


@pytest.mark.parametrize('bad', [[pd.NaT], ['2020-01-02', '2020-01-01'],
                                ['2020-01-01', '2020-01-01']])
def test_invalid_session_axis_is_rejected_before_read(bad):
    store = Store(pd.bdate_range('2020-01-01', periods=3))
    with pytest.raises(ValueError, match='sessions'):
        read_session_vwap(store, bad, ['a'])
    assert store.calls == []

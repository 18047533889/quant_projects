"""Read identities must survive conversion; handles close on success/error."""
from types import SimpleNamespace
import pandas as pd
import pytest
from factor_optimizer.research_price_reader import read_session_vwap


class Handle:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.closed = False
        self.read_identity = SimpleNamespace(
            digest='identity', source_snapshot='snapshot', provenance_status='available')

    def to_pandas(self):
        if self.fail:
            raise RuntimeError('conversion failed')
        return pd.DataFrame({'TradeDate': [pd.Timestamp('2020-01-02')],
                             'Symbol': ['a'], 'AdjVwap': [2.0]})

    def close(self):
        self.closed = True


def test_reader_keeps_source_identity_and_closes_handle():
    handle = Handle()
    store = SimpleNamespace(read=lambda *args, **kwargs: handle)
    result = read_session_vwap(store, ['2020-01-02'], ['a'])
    receipt = result.attrs['data_access_price_reads'][0]
    assert receipt['read_identity_digest'] == 'identity'
    assert receipt['source_snapshot'] == 'snapshot'
    assert receipt['provenance_status'] == 'available'
    assert receipt['rows'] == 1
    assert handle.closed


def test_reader_closes_handle_after_conversion_failure():
    handle = Handle(fail=True)
    store = SimpleNamespace(read=lambda *args, **kwargs: handle)
    with pytest.raises(RuntimeError, match='conversion failed'):
        read_session_vwap(store, ['2020-01-02'], ['a'])
    assert handle.closed


def test_absent_identity_is_explicitly_unavailable():
    handle = Handle()
    handle.read_identity = None
    store = SimpleNamespace(read=lambda *args, **kwargs: handle)
    result = read_session_vwap(store, ['2020-01-02'], ['a'])
    receipt = result.attrs['data_access_price_reads'][0]
    assert receipt['read_identity_digest'] is None
    assert receipt['source_snapshot'] is None
    assert receipt['provenance_status'] == 'unavailable'

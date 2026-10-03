"""Warmup windows preserve real metadata estimates without reading data."""
from types import SimpleNamespace
import pandas as pd
import pytest
from factor_engine.storage.time_window import WindowedDataSource

class Source:
    instrument_filter = ('A', 'B')
    def __init__(self):
        self.calls = []
        self.receipt = SimpleNamespace(estimated_rows=8, selected_bytes=128)
    def estimate_scan_cost(self, *, fields, time_range=None, instruments=None):
        self.calls.append((fields, time_range, instruments))
        return self.receipt
    def load_column(self, name):
        raise AssertionError('cost estimation must not read data')

def test_window_estimate_forwards_real_receipt_and_intersected_scope():
    source = Source()
    window = WindowedDataSource(source, start_date='2026-01-02', end_date='2026-01-08')
    receipt = window.estimate_scan_cost(fields=('close',),
        time_range=('2026-01-04', '2026-01-12'), instruments=('B',))
    assert receipt is source.receipt
    fields, span, instruments = source.calls[0]
    assert fields == ('close',) and instruments == ('B',)
    assert tuple(map(pd.Timestamp, span)) == (pd.Timestamp('2026-01-04'), pd.Timestamp('2026-01-08'))

def test_window_estimate_without_caller_bounds_uses_window_and_inner_instruments():
    source = Source()
    window = WindowedDataSource(source, start_date='2026-01-02', end_date='2026-01-08')
    assert window.estimate_scan_cost(fields=('close',)) is source.receipt
    assert source.calls[0][2] == ('A', 'B')
    assert tuple(map(pd.Timestamp, source.calls[0][1])) == (pd.Timestamp('2026-01-02'), pd.Timestamp('2026-01-08'))

def test_fields_only_estimator_is_a_conservative_unfiltered_receipt():
    class FieldsOnly(Source):
        def estimate_scan_cost(self, *, fields):
            self.calls.append(fields)
            return self.receipt
    source = FieldsOnly()
    window = WindowedDataSource(source, start_date='2026-01-02')
    assert window.estimate_scan_cost(fields=('close',)) is source.receipt
    assert source.calls == [('close',)]

def test_estimator_errors_propagate_without_retry_or_fake_rows():
    class Broken(Source):
        def estimate_scan_cost(self, *, fields, time_range=None, instruments=None):
            self.calls.append(fields)
            raise RuntimeError('snapshot mismatch')
    source = Broken()
    with pytest.raises(RuntimeError, match='snapshot mismatch'):
        WindowedDataSource(source).estimate_scan_cost(fields=('close',))
    assert len(source.calls) == 1

def test_missing_estimator_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match='scan-cost'):
        WindowedDataSource(object()).estimate_scan_cost(fields=('close',))

def test_disjoint_cost_scopes_rejected_without_calling_estimator():
    source = Source()
    window = WindowedDataSource(source, start_date='2026-01-05', end_date='2026-01-08')
    with pytest.raises(ValueError, match='disjoint'):
        window.estimate_scan_cost(fields=('close',), time_range=('2026-01-01', '2026-01-03'))
    assert source.calls == []

def test_cost_bounds_keep_utc_instants_of_aware_window():
    source = Source()
    window = WindowedDataSource(source,
        start_date=pd.Timestamp('2026-01-02 09:30', tz='Asia/Shanghai'),
        end_date=pd.Timestamp('2026-01-02 15:00', tz='Asia/Shanghai'), intraday=True)
    window.estimate_scan_cost(fields=('close',), time_range=(
        pd.Timestamp('2026-01-02 01:00', tz='UTC'), pd.Timestamp('2026-01-02 08:00', tz='UTC')))
    assert source.calls[0][1] == (pd.Timestamp('2026-01-02 01:30', tz='UTC'),
                                  pd.Timestamp('2026-01-02 07:00', tz='UTC'))

@pytest.mark.parametrize('span', [(pd.NaT, None), ('2026-01-02',), '2026-01-02'])
def test_invalid_cost_bounds_rejected_before_estimator(span):
    source = Source()
    with pytest.raises(ValueError):
        WindowedDataSource(source).estimate_scan_cost(fields=('close',), time_range=span)
    assert source.calls == []

def test_dataset_keyword_reaches_estimator_that_accepts_it():
    class DatasetSource(Source):
        def estimate_scan_cost(self, *, dataset, fields, time_range=None, instruments=None):
            self.calls.append(dataset)
            return self.receipt
    source = DatasetSource()
    assert WindowedDataSource(source).estimate_scan_cost(
        fields=('close',), dataset='ashare_stock_daily_adj') is source.receipt
    assert source.calls == ['ashare_stock_daily_adj']

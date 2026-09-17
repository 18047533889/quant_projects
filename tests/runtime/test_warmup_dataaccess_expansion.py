from types import SimpleNamespace

from factor_engine.runtime.warmup_service import _switch_engine_to_window
from factor_engine.storage.sources.data_access_source import DataAccessSource


def test_warmup_expands_read_and_preserves_contracts_without_mutating_source():
    source = DataAccessSource(dataset='ashare_stock_daily_adj', fields={'close': 'AdjClose'},
        start_date='2026-04-27', end_date='2026-04-30', instrument_filter=[],
        run_mode='production', production=True, strict_unknown_fields=True,
        pit_enforce=True, enforce_mining_gate=True, read_auto=False,
        params={'lazy_scan': True})
    engine = SimpleNamespace(data_source=source, cache=None,
        with_data_source=lambda data, **kwargs: SimpleNamespace(data_source=data))
    window = SimpleNamespace(actual_load_start='2025-04-03',
        actual_load_end='2026-04-30', requested_start='2026-04-27')
    result = _switch_engine_to_window(engine, window, source_bar_freq='1d', bars_per_day=1)
    actual = result.data_source
    assert actual.start_date == '2025-04-03'
    assert actual.end_date == '2026-04-30'
    assert source.start_date == '2026-04-27'
    assert actual.instrument_filter == []
    assert actual.production and actual.strict_unknown_fields and actual.pit_enforce
    assert actual.enforce_mining_gate
    assert actual.run_mode == 'production'
    assert actual.read_auto is False
    assert actual.fields == source.fields
    assert actual is not source

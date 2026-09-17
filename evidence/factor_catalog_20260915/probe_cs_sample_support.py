"""Real-data check of cross-section support; no production panel publication."""
import json
import os
import time

os.environ['ASHARE_PARQUET_ROOT'] = '/home/sunhaiwei/cos_data'
os.environ.setdefault('DATA_ACCESS_SKIP_COS_MIRROR', '1')
os.environ['DATA_ACCESS_RUN_MODE'] = 'interactive_research'

from data_access import get_store
from factor_engine.api.dsl_parser import DSLParser
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.sources.data_access_source import DataAccessSource
from smoke_catalog import bind_fields, summarize_result

load_all()
store = get_store()
ids = store.read_arrow('ashare_stock_daily_adj', columns=['Symbol'],
                       time_range=('2026-04-30', '2026-04-30'), limit=64)
symbols = list(dict.fromkeys(ids.column('Symbol').to_pylist()))
assert 20 <= len(symbols) <= 64
formula = 'cs_multi_robust_resid(pe_ratio, market_cap, pb_ratio, roe, add_intercept=True)'
expr = DSLParser(surface='compat_research').parse(formula)
bindings, failures = bind_fields(expr)
assert not failures, failures
fields = {b['input']: b['column'] for b in bindings if b['dataset'] == 'ashare_stock_daily_adj'}
source = DataAccessSource(dataset='ashare_stock_daily_adj', fields=fields,
    start_date='2025-01-01', end_date='2026-04-30', instrument_filter=symbols,
    run_mode='interactive_research', production=False, read_auto=False)
engine = FactorEngine(PandasBackend(), source, run_mode='research')
started = time.monotonic()
result = engine.run(Factor(name='source_row_37_support_probe', expr=expr,
                          source_expr=formula, surface='compat_research'))
summary = summarize_result(result)
print(json.dumps({'event': 'cross_section_support_proof', 'source_row': 37,
    'executed_formula': formula, 'symbols': symbols, 'symbol_count': len(symbols),
    'seconds': time.monotonic()-started, **summary}), flush=True)
assert summary['finite_count'] > 0, 'adequate universe still produced no finite values'
source.clear_cache()

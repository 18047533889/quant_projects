#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the 22 truncated daily factor matrices to full 2016-01-04..2026-08-24 coverage.

The 22 "new-mined" daily factors (from /tmp/new50_selected.json, excluding the 4
alphasage which already have full coverage) exist in factor_matrices_all only for
2016-01-04..2018-06-30 (the intake eval window). This script recomputes them over
the full daily panel using FactorEngine with each factor's fe_formula expression
tree (the exact same pipeline land_runner.py used), verified 0-diff against the
existing 2016-2018 slice.

Output: weekly_backtest_output/factor_matrices_all_2026daily/<page>.parquet
        (2588 x 5461 float32, index 2016-01-04..2026-08-24)
"""
import sys, os, json, time, argparse, warnings, resource
sys.path.insert(0, '/home/sunhaiwei/quant_projects')
warnings.filterwarnings("ignore")
os.environ['ASHARE_PARQUET_ROOT'] = os.path.expanduser('~/cos_data')
os.environ['DATA_ACCESS_SKIP_COS_MIRROR'] = '1'
os.environ['DATA_ACCESS_RUN_MODE'] = 'interactive_research'
os.environ['FACTOR_ENGINE_RUN_MODE'] = 'research'
os.environ['OMP_NUM_THREADS'] = '31'

import numpy as np
import pandas as pd
import factor_engine.cleaned_operators as co
co.load_all(include_research=False)
from factor_engine.storage.factory import build_data_source
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.api.factor import Factor
import factor_engine.api.cleaned_ops as co2
from factor_engine.api.columns import col

PROJECT = '/home/sunhaiwei/quant_projects'
FV_DIR = os.path.join(PROJECT, 'weekly_backtest_output', 'factor_matrices_all')
OUT_DIR = os.path.join(PROJECT, 'weekly_backtest_output', 'factor_matrices_all_2026daily')
START, END = '2016-01-04', '2026-08-24'
LOG = '/tmp/rob26_rebuild_22daily.log'
PROGRESS = '/tmp/rob26_rebuild_22daily.json'

OPS = {}
for nm in ['power','ts_pct','ts_topk_sum','ts_skew','ts_std','sqrt','zscore','cs_resid','cs_demean',
           'ts_cov','ts_kurt','ts_max','ts_argmax','ts_delta','winsorize','round','signed_sqrt',
           'sigmoid','scale','log','abs','where','lt','gt','is_nan','rank','ts_mean','ewm_mean',
           'ts_zscore','clip','multiply','divide','add','subtract','neg','sign','tanh','log1p',
           'price_spread_deviation','ts_min','ts_sum','ewm_std','ewm_var','ts_rank','pct_change',
           'ts_corr','log10','ewma','ema','ts_argmax_age','signed_power','if_else','eq','delay',
           'max','min','winsor','ts_delay','ts_delay1','ts_delay2','cs_zscore','cs_sum','cs_mean',
           'log10','subtract','lt','gt','ge','le','ne','and','or','not','swap','ref','pow']:
    try:
        OPS[nm] = co2.make_cleaned_call_factory(nm)
    except Exception:
        pass

COLMAP = {'Return': col('Return'), 'Volume': col('Volume'), 'AdjOpen': col('AdjOpen'),
          'AdjClose': col('AdjClose'), 'AdjHigh': col('AdjHigh'), 'AdjLow': col('AdjLow'),
          'AdjPreClose': col('AdjPreClose'), 'AdjAmount': col('AdjAmount'), 'AdjVwap': col('AdjVwap'),
          'PreClose': col('AdjPreClose'), 'Open': col('AdjOpen'), 'Close': col('AdjClose'),
          'High': col('AdjHigh'), 'Low': col('AdjLow'), 'Amount': col('AdjAmount')}

def build(node):
    kind = node.get('kind')
    if kind == 'column':
        return COLMAP.get(node['name'], col(node['name']))
    if kind == 'literal':
        return node['value']
    op = node.get('op')
    args = [build(a) for a in node.get('args', [])]
    if op not in OPS:
        raise ValueError(f'no op {op}')
    return OPS[op](*args)

def plog(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    REF = pd.read_parquet(os.path.join(FV_DIR, 'abnormality_asymmetry.parquet'))
    IDX = REF.index
    COLS = list(REF.columns)
    plog(f'[main] ref {REF.shape} {IDX[0]}..{IDX[-1]}')

    d26 = json.load(open('/tmp/new50_selected.json'))
    TRUNC = ['tail_asymmetry_volume_confirmed','overnight_gap_fragility','intraday_conviction_volume_gate',
             'downside_upside_tail_imbalance_volume_','ovni_stable_curvature','ovni_vol_align',
             'adaptive_vol_asym_v2','intraday_overnight_gap_smoothed','vol_range_conviction_smooth',
             'relative_volume_return_coherence_regime','intraday_persistence_share_ewma',
             'vol_momentum_divergence_stable','exhaustion_local_norm','pv_dir_alignment',
             'vol_trend_stability','intraday_phase_share_ewma_volume_quality','overnight_gap_fragility_log',
             'signed_vol_ret_alignment','crash_gap_overnight_fragility','range_amplitude_volatility',
             'intraday_downside_variance_share_logit','crash_gap_fragility_signed_sqrt']
    entries = {e['page_name']: e for e in d26}

    ds = build_data_source({'type':'data_access','dataset':'ashare_stock_daily_adj',
                            'start_date':START,'end_date':END})
    eng = FactorEngine(build_backend('pandas'), ds, run_mode='research')

    result = {}
    for page in TRUNC:
        p = os.path.join(OUT_DIR, page + '.parquet')
        if os.path.exists(p) and os.path.getsize(p) > 1e6:
            result[page] = {'status':'skip','shape':list(pd.read_parquet(p).shape)}
            plog(f'[skip] {page}')
            continue
        entry = entries.get(page)
        if entry is None:
            result[page] = {'status':'fail','error':'no entry'}
            continue
        t0 = time.time()
        try:
            expr = build(json.loads(entry['fe_formula']))
            f = Factor(name='rebuild_'+page, expr=expr)
            out = eng.run(f, market='ashare')
            mat = out['result'].unstack(level=out['result'].index.names[-1])
            mat = mat.reindex(index=IDX, columns=COLS).astype('float32')
            nn = float(mat.notna().mean().mean())
            n26 = int(mat.loc['2026-01-01':'2026-08-24'].notna().sum().sum())
            mat.to_parquet(p)
            result[page] = {'status':'ok','shape':list(mat.shape),'nonnull':round(nn,4),
                            'n26':n26,'t':round(time.time()-t0,1)}
            plog(f"[ok] {page}: nonnull={nn:.3f} n26={n26} t={time.time()-t0:.0f}s")
        except Exception as e:
            result[page] = {'status':'fail','error':f'{type(e).__name__}: {str(e)[:200]}'}
            plog(f"[fail] {page}: {type(e).__name__} {str(e)[:160]}")
        with open(PROGRESS, 'w') as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
    json.dump(result, open(PROGRESS, 'w'), ensure_ascii=False, indent=1)
    plog('[done]')

if __name__ == '__main__':
    main()

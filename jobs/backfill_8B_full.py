#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量重算 8 个 B 类空因子（HASCODE 日线类）覆盖写回 factor_matrices_all/。
分批（每 800 股）加载行情，逐因子逐批算，结果拼接后写盘。控制内存。
用法：/home/sunhaiwei/quant_projects/.venv/bin/python jobs/backfill_8B_full.py > /tmp/backfill_8B.log 2>&1 &
"""
import sys, json, time, re, os
sys.path.insert(0, '.'); sys.path.insert(0, 'jobs'); sys.path.insert(0, 'scripts/archive/jobs')
import numpy as np, pandas as pd
import backfill_28_factors as B
import extend_all_456 as E
from patch_alpha_tools import patch_tools
patch_tools()

T0 = time.time()
targets = ['breakout_hold_persistence_20d','defended_close_high_retest_participation_20_5',
           'downside_shock_accumulation_asym','failed_retest_reversion_20_5',
           'market_phase_atr_expansion','skew_weighted_volume_event_intensity',
           'skew_weighted_volume_event_intensity_close','trend_path_efficiency_5_60']

symbols = list(pd.read_parquet('weekly_backtest_output/factor_matrices_all/abnormality_asymmetry.parquet').columns)
print(f"[全量] 股票池 {len(symbols)} 只, 起始 {time.time()-T0:.0f}s", flush=True)
start, end = "2019-01-02", "2026-08-24"

# 预取每个因子的 code（每因子只改一次）
codes = {}
for page in targets:
    code0 = B.LQTP_PAGES.get(page, {}).get('code') or ''
    code = code0.replace('df_copy','df')
    code = re.sub(r"^import talib.*$", "", code, flags=re.M)
    code = code.replace("talib.TRANGE(", "E.talib.TRANGE(").replace("talib.ATR(", "E.talib.ATR(")
    code = re.sub(r"alpha_tools\.decompose_return_volatility_direction\(([^,]+), min_periods=\d+, window=(\d+)",
                  r"alpha_tools.decompose_return_volatility_direction(\1, window=\2, min_periods=\2", code)
    codes[page] = code

# 逐因子：逐批算，拼接写盘
for page in targets:
    print(f"\n== {page} ==", flush=True)
    parts = {}
    for bi in range(0, len(symbols), 800):
        batch = symbols[bi:bi+800]
        mkt = {}
        for c in ["open","high","low","close","volume","amount"]:
            mkt[c] = B.load_wide(batch, start, end, {c.title() if c!='volume' else 'Volume': c})[c]
        mkt["ret"] = mkt["close"].pct_change(fill_method=None)
        mkt["abs_ret"] = mkt["ret"].abs()
        mkt["trange"] = mkt["high"] - mkt["low"]
        mkt["prior_high"] = mkt["high"].shift(1)
        mkt["prior_low"] = mkt["low"].shift(1)
        mkt["prior_close"] = mkt["close"].shift(1)
        mkt["close_shift_5"] = mkt["close"].shift(5)
        dates = mkt["close"].index
        batch2 = list(mkt["close"].columns)
        interm = E._build_intermediates(mkt, batch2, dates)
        for k in ["ret","abs_ret","trange","prior_high","prior_low","prior_close","close_shift_5"]:
            if k in mkt:
                interm[k] = mkt[k]
        m = E._execute_factor_code(codes[page], mkt, batch2, dates, intermediates=interm)
        parts[bi] = m
        print(f"  批{bi//800+1}/{(len(symbols)+799)//800}: valid={m.notna().mean().mean()*100:.1f}% ({time.time()-T0:.0f}s)", flush=True)
        del mkt, interm, m
        import gc; gc.collect()
    full = pd.concat([parts[bi] for bi in sorted(parts)], axis=1)
    sub = full.astype('float32').dropna(axis=1, how='all')
    B.FV_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(B.FV_DIR / f"{page}.parquet")
    print(f"  ✓ 写 {page}: {sub.shape} valid={full.notna().mean().mean()*100:.1f}% ({time.time()-T0:.0f}s)", flush=True)

print(f"\n[完成] 8 个 B 类, 总耗时 {time.time()-T0:.0f}s", flush=True)
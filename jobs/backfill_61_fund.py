#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补齐 61 个空因子（财务数据缺失型）：用新 /tmp/fund_* 列重算并写回 factor_matrices_all/。

复用 backfill_adj_parallel.run_hascode_parallel 通道（HASCODE code 执行），
但传入指定空因子列表；4 个分钟类（amount_weighted_impact / skew_weighted_*）
走 run_minute_pages；market_phase_atr_expansion 用日线等价 code。
产出写回 weekly_backtest_output/factor_matrices_all/<page>.parquet（不覆盖已非空的）。

用法：/home/sunhaiwei/quant_projects/.venv/bin/python jobs/backfill_61_fund.py [--smoke]
"""
import sys, json, time, os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path("/home/sunhaiwei/quant_projects")
for p in [str(PROJECT), str(PROJECT / "vectorbt_qs"), str(PROJECT / "factor_preprocess"),
          str(PROJECT / "jobs"), str(PROJECT / "scripts" / "archive" / "jobs")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import backfill_28_factors as B
import extend_all_456 as E

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"

# 新增基本面列缓存（喂给 mkt dict）
FUND_COLS = ["pb_lf", "pe_ttm", "mkt_cap_float", "free_turn",
             "debttoassets", "roe_ttm2", "roa2_ttm2", "ps_ttm", "pcf_ocf_ttm",
             "free_float_shares", "qfa_yoygr", "forecast_incap_chgr_mid",
             "style_gate_size_large", "style_gate_size_small",
             "style_gate_liquidity_high", "style_gate_momentum_high"]

# 分钟类（需要 StockMinuteBar，样本宇宙）
MINUTE_EMPTY = {
    "amount_weighted_impact",
    "skew_weighted_volume_event_intensity",
    "skew_weighted_volume_event_intensity_close",
}
# market_phase_atr_expansion：talib.TRANGE 在逐 symbol 注入时失真，用日线等价 code
MARKET_PHASE_CODE = '''def factor_market_phase_atr_expansion(df):
    tr = df['high'] - df['low']
    pc = df['close'].shift(1)
    tr_v = pd.concat([tr, (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    atr = tr_v.rolling(20, min_periods=5).mean()
    atr_long_mean = tr_v.rolling(120, min_periods=20).mean().replace(0.0, np.nan)
    signal = atr / atr_long_mean
    signal = signal.fillna(1.0)
    signal.name = 'factor_market_phase_atr_expansion'
    return signal'''


def is_empty(page):
    fp = FV_DIR / f"{page}.parquet"
    if not fp.exists():
        return True
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(fp)
        names = t.schema.names
        if len(names) <= 1:
            return True
        return False
    except Exception:
        return True


def run_hascode_parallel(pages, symbols, dates):
    """复用 extend_all_456._compute_factor_chunk（ProcessPool 并发批量）。"""
    import pickle as _pk
    import tempfile
    import re
    from concurrent.futures import ProcessPoolExecutor, as_completed
    mkt = {}
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        p = Path(f"/tmp/mkt_{c}.parquet")
        mkt[c] = pd.read_parquet(p) if p.exists() else None
    dates_all = mkt["close"].index
    for fc in FUND_COLS:
        p = Path(f"/tmp/fund_{fc}.parquet")
        mkt[fc] = pd.read_parquet(p) if p.exists() else None
    if mkt.get("debttoassets") is None:
        mkt["debttoassets"] = pd.DataFrame(np.nan, index=dates_all, columns=symbols)
    print(f"[b61] 构建 intermediates ...", flush=True)
    intermediates = E._build_intermediates(mkt, symbols, dates_all)
    tmp = Path(tempfile.gettempdir())
    mkt_path = tmp / "backfill61_mkt.pkl"
    int_path = tmp / "backfill61_int.pkl"
    with open(mkt_path, "wb") as fh:
        _pk.dump({"market_data": mkt, "symbols": symbols, "dates": dates_all}, fh)
    all_used = set()
    for pg in pages:
        code = B.LQTP_PAGES.get(pg, {}).get("code", "") or ""
        if pg == "market_phase_atr_expansion":
            code = MARKET_PHASE_CODE
        all_used |= set(re.findall(r"""df(?:_copy)?\[["']([a-zA-Z_][a-zA-Z0-9_]*)["']\]""", code))
    all_used |= {"close", "open", "high", "low", "volume", "amount", "close_price", "TradingDay"}
    all_used = {c for c in all_used if not c.startswith("factor_")}
    slim = {k: v for k, v in intermediates.items() if k in all_used}
    with open(int_path, "wb") as fh:
        _pk.dump(slim, fh)
    print(f"[b61] intermediates 精简 {len(intermediates)} -> {len(slim)}", flush=True)
    del intermediates
    import gc; gc.collect()
    n_workers = min(8, (os.cpu_count() or 4) - 2)
    res = {}
    defs = []
    for pg in pages:
        d = dict(B.LQTP_PAGES.get(pg, {}))
        if pg == "market_phase_atr_expansion":
            d["code"] = MARKET_PHASE_CODE
        if d.get("factor_name"):
            defs.append(d)
    batches = [defs[i::n_workers] for i in range(n_workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(E._compute_factor_chunk, b, str(mkt_path), str(int_path), symbols, dates_all.tolist())
                   for b in batches if b]
        for fut in as_completed(futures):
            cr, cok = fut.result(timeout=7200)
            res.update(cr)
    print(f"[b61] HASCODE 并发完成 {len(defs)} 个 {time.time()-t0:.0f}s", flush=True)
    return res, dates_all


def main():
    smoke = "--smoke" in sys.argv
    empty = [p for p in json.load(open("/tmp/empty_factors.json")) if is_empty(p)]
    print(f"[b61] 空因子 {len(empty)}", flush=True)
    if not empty:
        print("[b61] 无空因子", flush=True)
        return
    minute = [p for p in empty if p in MINUTE_EMPTY]
    daily_exec = [p for p in empty if p not in MINUTE_EMPTY]
    print(f"[b61] 分钟 {len(minute)} {minute}, HASCODE {len(daily_exec)}", flush=True)

    all_res = {}
    t0 = time.time()
    # A. 分钟类（sample 500 串行）
    if minute and not smoke:
        symbols_full, sample500 = B.get_symbols(full=True)
        all_res.update(B.run_minute_pages(minute, sample500, B.FULL_START, B.FULL_END, False))
        print(f"[b61] 分钟完成 {time.time()-t0:.0f}s", flush=True)
    # B. HASCODE 日线（并发，全宇宙）
    if daily_exec:
        symbols_full, _ = B.get_symbols(full=True)
        res, dates_all = run_hascode_parallel(daily_exec, symbols_full, None)
        for k, v in res.items():
            pn = k.replace("factor_", "").replace("_flipped", "")
            all_res[pn] = v
        print(f"[b61] HASCODE 完成 {len(daily_exec)} 个 {time.time()-t0:.0f}s", flush=True)

    if smoke:
        print("[b61] smoke 模式不写盘", flush=True)
        return
    # 写盘：只覆盖仍为空的文件
    written = 0
    for page, mat in all_res.items():
        if not is_empty(page):
            print(f"  [skip] {page} 已非空", flush=True)
            continue
        if mat is None or not isinstance(mat, pd.DataFrame) or mat.empty:
            print(f"  [empty-res] {page}", flush=True)
            continue
        mat = mat.reindex(index=dates_all, columns=symbols_full) if page not in MINUTE_EMPTY else mat
        sub = mat.astype(np.float32).dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print(f"  [all-null] {page} 仍全空，不覆盖", flush=True)
            continue
        sub.to_parquet(FV_DIR / f"{page}.parquet")
        pct = sub.notna().mean().mean() * 100
        print(f"  [OK] {page}: {sub.shape} nonnull={pct:.1f}%", flush=True)
        written += 1
    still = [p for p in json.load(open("/tmp/empty_factors.json")) if is_empty(p)]
    print(f"[b61] 写盘 {written} 个, 仍空 {len(still)} 个: {still}", flush=True)


if __name__ == "__main__":
    main()
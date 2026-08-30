#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后复权并行 backfill：把当前 85 空因子补齐。
- HASCODE 日线：复用 extend_all_456._compute_factor_chunk（ProcessPool 并发批量）。
- 分钟类：复用 backfill_28.run_minute_pages（串行，先跑这部分少因子）。
- DSL：复用 backfill_28.run_dsl_pages。
全部用后复权 Adj 数据（rebuild_tmp_cache_adj 已重建 /tmp/mkt_*，backfill_28.DAILY 已切 StockDailyBarAdj）。
"""
import sys, json, time, importlib
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

def current_empty():
    out = []
    for f in sorted(FV_DIR.glob("*.parquet")):
        try:
            if not pd.read_parquet(f).shape[1]:
                out.append(f.stem)
        except Exception:
            out.append(f.stem)
    return out

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


def run_hascode_parallel(daily_exec, symbols, start, end):
    """用 extend_all_456 的并发架构（_compute_factor_chunk ProcessPool）跑 HASCODE 日线。"""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pickle as _pk
    mkt = {}
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        p = Path(f"/tmp/mkt_{c}.parquet")
        mkt[c] = pd.read_parquet(p) if p.exists() else None
    dates = mkt["close"].index
    # 通配加载全部财务列缓存（/tmp/fund_*.parquet），列名 = 文件名去 fund_ 前缀
    fund_cols = []
    for fp in sorted(Path("/tmp").glob("fund_*.parquet")):
        fc = fp.stem[len("fund_"):]
        mkt[fc] = pd.read_parquet(fp)
        fund_cols.append(fc)
    if "debttoassets" not in mkt or mkt.get("debttoassets") is None:
        mkt["debttoassets"] = pd.DataFrame(np.nan, index=dates, columns=symbols)
    intermediates = E._build_intermediates(mkt, symbols, dates)
    # 补传 _build_intermediates 硬编码 FUND_COLS 之外的 glob 新列（close_raw/free_cap 等）
    for fc in fund_cols:
        if fc in mkt and mkt[fc] is not None and fc not in intermediates:
            intermediates[fc] = mkt[fc].reindex(index=dates, columns=symbols)
    import tempfile
    tmp = Path(tempfile.gettempdir())
    mkt_path = tmp / "backfill_adj_mkt.pkl"
    int_path = tmp / "backfill_adj_int.pkl"
    with open(mkt_path, "wb") as fh:
        _pk.dump({"market_data": mkt, "symbols": symbols, "dates": dates}, fh)
    all_used = set()
    for pg in daily_exec:
        code = B.LQTP_PAGES.get(pg, {}).get("code", "")
        if pg == "market_phase_atr_expansion":
            code = MARKET_PHASE_CODE
        all_used |= set(__import__("re").findall(r"df_copy\['([a-zA-Z_][a-zA-Z0-9_]*)+'\]", code))
        all_used |= set(__import__("re").findall(r"df_copy\[\"([a-zA-Z_][a-zA-Z0-9_]*)+", code))
        all_used |= set(__import__("re").findall(r"df\[['\\\"]([a-zA-Z_][a-zA-Z0-9_]*)+['\\\"]\]", code))
    all_used |= {"close", "open", "high", "low", "volume", "amount", "close_price", "TradingDay"}
    all_used = {c for c in all_used if not c.startswith("factor_")}
    slim = {k: v for k, v in intermediates.items() if k in all_used}
    with open(int_path, "wb") as fh:
        _pk.dump(slim, fh)
    del intermediates
    n_workers = min(8, (os.cpu_count() or 4) - 2)
    res = {}
    # _compute_factor_chunk 期待 chunk 为 dict 列表（factor 定义），把 page 名转成定义
    defs = []
    for pg in daily_exec:
        d = dict(B.LQTP_PAGES.get(pg, {}))
        if pg == "market_phase_atr_expansion":
            d["code"] = MARKET_PHASE_CODE
        if d.get("factor_name"):
            defs.append(d)
    batches = [defs[i::n_workers] for i in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(E._compute_factor_chunk, b, str(mkt_path), str(int_path), symbols, dates.tolist())
                   for b in batches if b]
        for fut in as_completed(futures):
            cr, cok = fut.result(timeout=3600)
            res.update(cr)
    return res, dates

def main():
    smoke = "--smoke" in sys.argv
    empty = current_empty()
    print(f"[backfill-par] 空因子 {len(empty)}", flush=True)
    if not empty:
        print("[backfill-par] 无空因子", flush=True)
        return
    start, end = ("2024-01-02", "2024-01-03") if smoke else (B.FULL_START, B.FULL_END)
    symbols, minute_symbols = B.get_symbols(full=not smoke)

    minute = [p for p in empty if p in B.MINUTE_PAGES]
    dsl_only = sorted(set(empty) & B.DSL_ONLY_PAGES)
    dsl_fallback = [p for p in empty if p not in B.MINUTE_PAGES and p not in B.DSL_ONLY_PAGES
                    and not ((B.LQTP_PAGES.get(p, {}).get("code") or "") and "NotImplementedError" not in (B.LQTP_PAGES.get(p, {}).get("code") or ""))]
    daily_exec = [p for p in empty if p not in B.MINUTE_PAGES and p not in B.DSL_ONLY_PAGES
                  and (B.LQTP_PAGES.get(p, {}).get("code") or "") and "NotImplementedError" not in (B.LQTP_PAGES.get(p, {}).get("code") or "")]
    print(f"[backfill-par] 分钟 {len(minute)}, HASCODE {len(daily_exec)}, DSL {len(dsl_only)}, 其它fallback {len(dsl_fallback)}", flush=True)

    all_res = {}
    t0 = time.time()
    # A. 分钟类（sample500，串行；只 6 因子）
    # 分钟类用 StockMinuteBar（无 Adj 源，复权单日恒定不影响日内形态）
    minute = [p for p in empty if p in B.MINUTE_PAGES and not (FV_DIR/f"{p}.parquet").exists()]
    if minute and not smoke:
        all_res.update(B.run_minute_pages(minute, minute_symbols or symbols, start, end, smoke))
        print(f"[backfill-par] 分钟完成 {time.time()-t0:.0f}s", flush=True)
    # B. HASCODE 日线（并发，全宇宙）—— 含 3 个分钟型因子的 panel 等价实现
    if daily_exec:
        res, _dates = run_hascode_parallel(daily_exec, symbols, start, end)
        for k, v in res.items():
            pn = k.replace("factor_", "").replace("_flipped", "")
            all_res[pn] = v
        print(f"[backfill-par] HASCODE 完成 {len(daily_exec)} 个 {time.time()-t0:.0f}s", flush=True)
    # C. DSL（语义实现）
    if dsl_only:
        all_res.update(B.run_dsl_pages(dsl_only, symbols, start, end, smoke))
        print(f"[backfill-par] DSL 完成 {time.time()-t0:.0f}s", flush=True)
    # D. 其它 fallback（无 code 或 NotImplementedError）—— 尝试 DSL 兜底（B.run_dsl_pages 覆盖的）
    if dsl_fallback:
        all_res.update(B.run_dsl_pages(dsl_fallback, symbols, start, end, smoke))
        print(f"[backfill-par] fallback DSL 完成 {time.time()-t0:.0f}s", flush=True)

    if not smoke:
        B.save_mats(all_res, smoke=False)
    still = current_empty()
    print(f"[backfill-par] 补 {len(all_res)} 个, 仍空 {len(still)} 个, 总耗时 {time.time()-t0:.0f}s", flush=True)
    if still:
        print("  仍空:", still[:25], flush=True)

if __name__ == "__main__":
    main()

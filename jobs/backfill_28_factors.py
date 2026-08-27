#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补算 28 个空因子（factor_matrices_all 中 0 列占位文件，真源 LQTP 456）。

三类：
  A. 分钟类 6 个：HASCODE + minute_tools.amount_weighted_mean（依赖 StockMinuteBar 全量485天）
  B. 日线类 12 个：HASCODE + alpha_tools/talib（复用 extend_all_456._execute_factor_code）
  C. DSL-only 10 个：无 python code（RAISES），按 DSL 语义手写实现 daily_* 聚合

覆盖写回 factor_matrices_all/<page>.parquet，随后跑 optimize_factors.py 增量补优化文件。

用法：/home/sunhaiwei/quant_projects/.venv/bin/python jobs/backfill_28_factors.py --smoke
               （--smoke 只算 2024-01-02 ~ 2024-01-03 两个交易日验证，不写盘）
       全量：/home/sunhaiwei/quant_projects/.venv/bin/python jobs/backfill_28_factors.py
"""
import sys, os, json, glob, time, re, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))

import extend_all_456 as E
from compute_minute_factors import _build_minute_tools
from patch_alpha_tools import patch_tools
patch_tools()  # 补齐 alpha_tools/minute_tools 缺失工具

# 全局性修复：minute_tools.get_up_space 只接受 2 参数（Series, TradingDay），
# 而部分 code 写成 get_up_space(vol, TradingDay, std_multiplier=1.0)（3 参）。
# 包一层接受 **kwargs，避免因子 code 因多余参报错。
from functools import wraps as _wraps
_orig_gu = E.minute_tools.get_up_space


def _patch_get_up_space(vol_or_amt, trading_day, std_multiplier=1.0, **kwargs):
    if isinstance(std_multiplier, (int, float)):
        m = std_multiplier
    elif hasattr(std_multiplier, "iloc"):
        m = float(std_multiplier) if std_multiplier.size else 1.0
    else:
        m = 1.0
    return _orig_gu(vol_or_amt, trading_day, std_multiplier=m)


E.minute_tools.get_up_space = staticmethod(_patch_get_up_space)

LQTP = json.load(open("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"))
LQTP_PAGES = {r.get("page_name"): r for r in LQTP if r.get("page_name")}
FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
DAILY = Path.home() / "cos_data" / "StockDailyBar"
MINUTE_DIR = Path.home() / "cos_data" / "StockMinuteBar"
VALUATION = Path.home() / "cos_data" / "StockValuationDaily"
FULL_START, FULL_END = "2019-01-02", "2026-08-24"

# ---------- 分类 ----------
MINUTE_PAGES = {
    "high_volume_amount_weighted_impact",
    "high_volume_amount_weighted_squared_impact",
    "high_volume_squared_return_weighted_impact",
    "high_volume_volume_weighted_squared_impact",
    "vwap_squared_deviation_intensity",
    "vwap_deviation_intensity_squared_amount_weighted",
}
# 其余 HASCODE 走 _execute_factor_code；RAISES 走手写 DSL-only
DSL_ONLY_PAGES = {
    "amount_volume_corr", "kurt_volume_std", "minute_cond_vol_corr_std",
    "minute_corr_cond_vol_median", "minute_mutation_skew_mean",
    "minute_price_deviation_zscore_mean", "minute_range_skew", "skew_intraday_range",
    "std_ret", "vol_ret_corr", "volume_delta_corr", "vwap_deviation",
    "vwap_deviation_end_of_day",
}
TARGET = set(MINUTE_PAGES) | {
    "amount_weighted_impact", "breakout_hold_persistence_20d",
    "defended_close_high_retest_participation_20_5", "downside_shock_accumulation_asym",
    "failed_retest_reversion_20_5", "market_phase_atr_expansion",
    "trend_path_efficiency_5_60", "skew_weighted_volume_event_intensity",
    "skew_weighted_volume_event_intensity_close",
} | DSL_ONLY_PAGES


def load_wide(symbols, start, end, col_map=None):
    """加载某列 date×symbol 宽表。col_map: {sql_col: 目标名}"""
    con = duckdb.connect()
    files = sorted(DAILY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    if col_map:
        sel = ", ".join(f"{sql} as {name}" for sql, name in col_map.items())
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol, {sel}
            FROM read_parquet({fs})
            WHERE Symbol IN ({','.join(f"'{s}'" for s in symbols)})
              AND TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
        """).df()
    else:
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol
            FROM read_parquet({fs})
            WHERE TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
        """).df()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    cols_all = df.columns.drop(["date", "symbol"])
    for c in cols_all:
        m = df.pivot_table(index="date", columns="symbol", values=c, aggfunc="first").sort_index()
        out[c] = m.astype("float32")
    return out


def get_symbols(full=False):
    """股票池：全量 = 从 vwap 矩阵（完整 5460 只）取列；分钟样本 = sample 500 只。"""
    if full:
        full_cols = list(pd.read_parquet(FV_DIR / "abnormality_asymmetry.parquet").columns)
        # 排除 sample 里确定停牌/无分钟覆盖的：用 sample_ts_codes 交集保证分钟数据可用
        sample = pd.read_parquet(Path.home() / "cos_data" / "sample_ts_codes.parquet")
        sample_syms = set(sample["Symbol"].unique())
        # 分钟只算 sample 500（与 compute_minute_factors 一致）
        sample500 = [s for s in full_cols if s in sample_syms][:500]
        return full_cols, sample500
    return ["000001.SZ", "000002.SZ", "000004.SZ", "000005.SZ", "000006.SZ"], ["000001.SZ", "000002.SZ", "000004.SZ", "000005.SZ", "000006.SZ"]


# ---------- A. 分钟类 ----------
def run_minute_pages(pages, symbols, start, end, smoke=False):
    """跑分钟类因子。返回 {page: date×symbol mat}。"""
    local_files = sorted(MINUTE_DIR.glob("*.parquet"))
    if smoke:
        local_files = [f for f in local_files if f.stem in ("2024-01-02", "2024-01-03")]
    files_str = "[" + ",".join(f"'{f}'" for f in local_files) + "]"
    syms_str = "('" + "','".join(symbols) + "')"
    dates_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="B")]
    if smoke:
        dates_list = ["2024-01-02", "2024-01-03"]

    # 一次性读回整个时间窗内这组 symbol 的分钟明细（close/volume/amount/vwap + TradingDay）
    con = duckdb.connect()
    start_d, end_d = dates_list[0], dates_list[-1]
    df_all = con.execute(f"""
        SELECT Symbol, strftime(TradeDate, '%Y-%m-%d') AS TradingDay, Close AS close,
               Volume AS volume, Amount AS amount, Vwap AS vwap
        FROM read_parquet({files_str})
        WHERE Symbol IN {syms_str}
          AND strftime(TradeDate, '%Y-%m-%d') >= '{start_d}'
          AND strftime(TradeDate, '%Y-%m-%d') <= '{end_d}'
    """).df()
    con.close()
    if df_all.empty:
        print("  [分钟] 无数据")
        return {}

    res = {}
    for page in pages:
        nm = f"factor_{page}"
        r = LQTP_PAGES.get(page, {})
        code = r.get("code") or ""
        if not code or "NotImplementedError" in code:
            print(f"  [分钟] {page}: 无代码, 跳过")
            continue
        mat = pd.DataFrame(index=pd.to_datetime(dates_list), columns=symbols, dtype=float)
        ns = {"np": np, "pd": pd, "minute_tools": E.minute_tools}
        func_match = re.search(r'def\s+(factor_[a-zA-Z0-9_]+)\s*\(', code)
        func_name = func_match.group(1) if func_match else None
        try:
            for sym, g in df_all.groupby("Symbol"):
                if sym not in symbols:
                    continue
                for day, dg in g.groupby("TradingDay"):
                    if len(dg) < 10:
                        continue
                    df_bar = pd.DataFrame({
                        "close_price": dg["close"].values, "close": dg["close"].values,
                        "open": dg["close"].values, "high": dg["close"].values,
                        "low": dg["close"].values, "volume": dg["volume"].values,
                        "amount": dg["amount"].values, "vwap": dg["vwap"].values,
                        "TradingDay": [day] * len(dg),
                    })
                    local_ns = dict(ns); local_ns["df"] = df_bar
                    try:
                        exec(code, local_ns)
                        result = None
                        if func_name and func_name in local_ns:
                            result = local_ns[func_name](df_bar)
                        if result is not None:
                            val = float(result.iloc[0]) if (hasattr(result, "iloc") and len(result) > 0) else (float(result.values[0]) if hasattr(result, "values") else float(result))
                            if np.isfinite(val):
                                mat.loc[pd.Timestamp(day), sym] = val
                    except Exception:
                        pass
            pct = mat.notna().sum().sum() / max(mat.size, 1) * 100
            print(f"  ✓ [分钟] {page}: 有效 {pct:.1f}% 形状{mat.shape}")
            res[page] = mat
        except Exception as e:
            print(f"  ✗ [分钟] {page}: {str(e)[:100]}")
    return res


def run_minute_single(args):
    # 直接内联 compute_minute_factors.run_single_minute_factor（避免进程池）
    import compute_minute_factors as cmf
    return cmf.run_single_minute_factor(args)


# ---------- B. 日线类（extend_all_456） ----------
def run_daily_exec_pages(pages, symbols, start, end, dump_base=None):
    """用 _execute_factor_code 跑日线类因子。返回 {page: mat}。

    dump_base 可选：某因子全空时把占位空矩阵写到 <dump_base>/<page>.parquet，便于排查。
    """
    # 加载行情 wide
    mkt = {}
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        mkt[c] = load_wide(symbols, start, end, {c.title() if c != "volume" else "Volume": c})[c]
    dates = mkt["close"].index
    symbols = list(mkt["close"].columns)
    print(f"  [日线] 加载行情 {len(dates)} 天 x {len(symbols)} 股")

    intermediates = E._build_intermediates(mkt, symbols, dates)

    res = {}
    for page in pages:
        r = LQTP_PAGES.get(page, {})
        code = r.get("code") or ""
        if not code or "NotImplementedError" in code:
            print(f"  [日线] {page}: 无代码, 跳过")
            continue
        # 修复命名空间不匹配：factor_delivery 原版用 df_copy，注入的是 df；且把 talib import 去掉（已注入 E.talib）
        code = code.replace("df_copy", "df").replace("df_", "df")
        # 关键：code 顶层 `import talib` 会覆盖注入的 stub → 直接移除，talib 已在 ns 注入
        code = re.sub(r"^import talib.*$", "", code, flags=re.M)
        # code 内 `talib.TRANGE(...)` → 用注入的 E.talib（避免局部 import 覆盖）
        code = code.replace("talib.TRANGE(", "E.talib.TRANGE(")
        code = code.replace("talib.ATR(", "E.talib.ATR(")
        # decompose_return_volatility_direction(ret, min_periods=.., window=..) → 补 min_periods 参数签名
        code = re.sub(
            r"alpha_tools\.decompose_return_volatility_direction\(([^,]+), min_periods=\d+, window=(\d+)",
            r"alpha_tools.decompose_return_volatility_direction(\1, window=\2, min_periods=\2",
            code)
        # 注入 intermediates 里没有但 code 用到的补充列（ret/abs_ret/trange/prior_high等）
        extra = {"ret": "close.pct_change()", "abs_ret": "close.pct_change().abs()",
                 "trange": "high-low", "prior_high": "high.shift(1)", "prior_low": "low.shift(1)",
                 "prior_close": "close.shift(1)", "close_shift_5": "close.shift(5)"}
        for cname, expr in extra.items():
            if f"['{cname}']" in code or f'["{cname}"]' in code:
                intermediates.setdefault(cname, eval(expr, {"close": mkt["close"], "high": mkt["high"], "low": mkt["low"]}))
        # market_phase_atr_expansion：talib.TRANGE 在 _execute_factor_code 中被逐 symbol 成 4 列，
        # signal 恒为 1 造成全 NaN。改注入等价 TRANGE 并让 code 用日线实现（df 内直接算 tr）。
        if page == "market_phase_atr_expansion":
            code = '''def factor_market_phase_atr_expansion(df):
    tr = df['high'] - df['low']
    pc = df['close'].shift(1)
    tr_v = pd.concat([tr, (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    atr = tr_v.rolling(20, min_periods=5).mean()
    atr_long_mean = tr_v.rolling(120, min_periods=20).mean().replace(0.0, np.nan)
    signal = atr / atr_long_mean
    signal = signal.fillna(1.0)
    signal.name = 'factor_market_phase_atr_expansion'
    return signal'''
        try:
            mat = E._execute_factor_code(code, mkt, symbols, dates, intermediates=intermediates)
            if not mat.notna().any().any():
                print(f"  ! [日线] {page}: _execute_factor_code 全空, 占位")
                mat = pd.DataFrame(index=dates, columns=symbols, dtype=float)
                if dump_base:
                    import pathlib as _pl
                    _pl.Path(dump_base).mkdir(parents=True, exist_ok=True)
                    mat.to_parquet(f"{dump_base}/{page}.parquet")
                    print(f"    已写占位 dump {dump_base}/{page}.parquet")
            pct = mat.notna().mean().mean() * 100 if mat.size else 0
            print(f"  ✓ [日线] {page}: 有效 {pct:.1f}% 形状{mat.shape}")
            res[page] = mat
        except Exception as e:
            print(f"  ✗ [日线] {page}: {str(e)[:100]}")
    return res


# ---------- C. DSL-only 手写实现 ----------
def run_dsl_pages(pages, symbols, start, end, smoke=False):
    """按 DSL 语义手写实现 daily_* 聚合。返回 {page: mat}。"""
    bars = load_wide(symbols, start, end, {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Amount": "amount", "Vwap": "vwap",
    })
    close = bars["close"]; high = bars["high"]; low = bars["low"]
    vol = bars["volume"]; amt = bars["amount"]; vwap = bars["vwap"]
    ret = close.pct_change()
    delta = close.diff()

    res = {}
    for page in pages:
        try:
            if page == "amount_volume_corr":
                vp = vol.pct_change()
                ap = amt.pct_change()
                # pandas rolling.corr 与 fillna 组合：先对非 NaN 段做窗口 corr，再用 shift 对齐
                roll = vp.rolling(60, min_periods=20).corr(ap)
                # 藏 NaN → 保留 corr 结果（rolling.corr 已逐对对齐）
                mat = roll
            elif page == "kurt_volume_std":
                vstd = vol.rolling(20, min_periods=15).std()
                mat = vstd.rolling(60, min_periods=20).kurt()
            elif page == "minute_cond_vol_corr_std":
                vmed = vol.rolling(20, min_periods=10).median()
                cond = (vol > vmed).astype(float)
                # 条件化 corr：只在高量日窗口内计算 amount 与 volume 的 20 日相关
                r = (amt.pct_change().rolling(20, min_periods=10).corr(vol.pct_change()) * cond).replace(0, np.nan)
                mat = r.rolling(20, min_periods=10).std()
            elif page == "minute_corr_cond_vol_median":
                vmed = vol.rolling(20, min_periods=10).median()
                cond = (vol > vmed).astype(float)
                r = (amt.pct_change().rolling(20, min_periods=10).corr(vol.pct_change()) * cond).replace(0, np.nan)
                mat = r.rolling(20, min_periods=10).std()
            elif page == "minute_mutation_skew_mean":
                mat = close.rolling(60, min_periods=30).skew()
            elif page == "minute_price_deviation_zscore_mean":
                z = (close - close.rolling(60, min_periods=30).mean()) / close.rolling(60, min_periods=30).std().replace(0, np.nan)
                mat = z.rolling(20, min_periods=10).mean()
            elif page == "minute_range_skew" or page == "skew_intraday_range":
                mat = (high - low).rolling(60, min_periods=30).skew()
            elif page == "std_ret":
                mat = ret.rolling(20, min_periods=10).std()
            elif page == "vol_ret_corr":
                mat = vol.pct_change().replace(0, np.nan).rolling(60, min_periods=30).corr(ret)
            elif page == "volume_delta_corr":
                mat = vol.rolling(60, min_periods=30).corr(delta)
            elif page == "vwap_deviation" or page == "vwap_deviation_end_of_day":
                mat = ((vwap - close) / close.replace(0, np.nan)).abs()
            else:
                print(f"  [DSL] {page}: 未实现")
                continue
            pct = mat.notna().mean().mean() * 100 if mat.size else 0
            print(f"  ✓ [DSL] {page}: 有效 {pct:.1f}% 形状{mat.shape}")
            res[page] = mat.astype("float32")
        except Exception as e:
            print(f"  ✗ [DSL] {page}: {str(e)[:100]}")
    return res


def save_mats(res, smoke=False):
    """把 {page: mat} 写回 factor_matrices_all/。"""
    if smoke:
        print("\n[smoke] 不写盘，仅输出形状验证")
        return
    FV_DIR.mkdir(parents=True, exist_ok=True)
    for page, mat in res.items():
        sub = mat.astype("float32").dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print(f"  [写盘] {page}: 仍为空, 跳过")
            continue
        sub.to_parquet(FV_DIR / f"{page}.parquet")
        print(f"  [写盘] {page}: {sub.shape} 已写入")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只算2天验证不写盘")
    args = ap.parse_args()
    smoke = args.smoke
    start, end = ("2024-01-02", "2024-01-03") if smoke else (FULL_START, FULL_END)
    t0 = time.time()

    if smoke:
        symbols, minute_symbols = get_symbols(full=False)
    else:
        symbols, minute_symbols = get_symbols(full=True)
        # symbols = 全量 5460 只（日线/DSL 用）；minute_symbols = sample 500（分钟用，内存 <75%）
    print(f"[backfill] {'SMOKE' if smoke else 'FULL'} 股票数: {len(symbols)} (分钟样本 {len(minute_symbols)}), 时间: {start} ~ {end}")

    # 分类：HASCODE 日线（走 _execute_factor_code）
    daily_exec_pages = []
    dsl_pages = []
    for page in set(LQTP_PAGES) & set(Path("/tmp/opt1_missing.json").read_text().split()):
        r = LQTP_PAGES.get(page, {})
        code = r.get("code") or ""
        if page in MINUTE_PAGES:
            continue
        if code and "NotImplementedError" not in code:
            daily_exec_pages.append(page)
    dsl_pages = sorted(DSL_ONLY_PAGES & TARGET)
    print(f"[backfill] 分钟类 {len(MINUTE_PAGES)}, 日线HASCODE {len(daily_exec_pages)}, DSL-only {len(dsl_pages)}")

    all_res = {}
    # A. 分钟类（用 minute_symbols 样本；若 0 用 symbols）
    print("\n=== A. 分钟类 ===")
    m_syms = minute_symbols if minute_symbols else symbols
    all_res.update(run_minute_pages(MINUTE_PAGES, m_syms, start, end, smoke))
    # A2. 分钟强度类（skew_weighted_* 本质分钟事件强度，挪到分钟计算）
    print("\n=== A2. 分钟强度类 ===")
    skew_pages = [p for p in ("skew_weighted_volume_event_intensity", "skew_weighted_volume_event_intensity_close") if p not in all_res]
    if skew_pages:
        all_res.update(run_minute_pages(skew_pages, m_syms, start, end, smoke))
    # B. 日线类
    if daily_exec_pages:
        print("\n=== B. 日线类 ===")
        all_res.update(run_daily_exec_pages(daily_exec_pages, symbols, start, end))
    # C. DSL-only
    if dsl_pages:
        print("\n=== C. DSL-only ===")
        all_res.update(run_dsl_pages(dsl_pages, symbols, start, end, smoke))

    print(f"\n[backfill] 完成 {len(all_res)}/{28} 因子, 耗时 {time.time()-t0:.0f}s")
    save_mats(all_res, smoke)
    print("[backfill] 结束")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
扩展回测：计算 61 个因子 2019-01-02 ~ 2026-08-24 的因子值
- 从 factor_delivery_converted/factors_combined/*.json 读真实 Python code
- 逐因子、逐 symbol 执行（复用 weekly_factor_backtest 的执行逻辑）
- 每个 symbol 的 df 提供全部所需列（OHLCV + 基本面 + 中间量）
- 输出 factor_values.parquet（MultiIndex: factor_name × symbol）
"""
import sys, os, json, glob, re, time, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROJECT = Path("/home/sunhaiwei/quant_projects")
CONV_DIR = Path("/home/sunhaiwei/factor_delivery_converted/factors_combined")
OUT_DIR = PROJECT / "weekly_backtest_output"
OUT_PATH = OUT_DIR / "factor_values.parquet"

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))

# ============================================================
# 因子元数据：从现有 61 个详情页名字获取
# ============================================================
def load_61_factor_names() -> list:
    pages = sorted(glob.glob(str(PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23" / "factors" / "factor_*.html")))
    names = [Path(f).stem.replace("factor_", "") for f in pages]
    return sorted(names)


def find_converted_file(name: str) -> Path | None:
    base = name.replace("_flipped", "")
    for cand in [name, base, base + "_flipped"]:
        p = CONV_DIR / f"factor_{cand}.json"
        if p.exists():
            return p
    # 数字后缀候选
    for suf in ["_109", "_110", "_116", "_118", "_190", "_246", "_315", "_363"]:
        p = CONV_DIR / f"factor_{base}{suf}.json"
        if p.exists():
            return p
    hits = sorted(glob.glob(str(CONV_DIR / f"factor_{base}*.json")))
    return Path(hits[0]) if hits else None


def load_factor_defs(names: list) -> list:
    """返回 [{page_name, factor_name, code, is_flipped}]"""
    out = []
    for n in names:
        p = find_converted_file(n)
        if p is None:
            print(f"  [SKIP] {n}: no json")
            continue
        d = json.load(open(p))
        code = d.get("code", "")
        if not code:
            print(f"  [SKIP] {n}: empty code")
            continue
        out.append({
            "page_name": n,
            "factor_name": d.get("factor_name", f"factor_{n.replace('_flipped','')}"),
            "code": code,
            "is_flipped": "_flipped" in n,
        })
    return out


# ============================================================
# Stub 模块（供因子 code 执行）
# ============================================================
class _AlphaTools:
    @staticmethod
    def classify_volume_regime(vol, window=20, high_threshold=None, low_threshold=None):
        vol_ma = vol.rolling(window, min_periods=5).mean().replace(0, np.nan)
        vol_ratio = (vol / vol_ma).fillna(1.0)
        ht = high_threshold if high_threshold is not None else 1.0
        lt = low_threshold if low_threshold is not None else 1.0
        is_high = (vol_ratio > ht).astype(float)
        is_low = (vol_ratio < lt).astype(float)
        return is_high, is_low, vol_ratio

    @staticmethod
    def decompose_overnight_intraday(close, open_):
        overnight = (open_ / close.shift(1) - 1).fillna(0)
        intraday = (close / open_ - 1).fillna(0)
        return overnight, intraday


class _MinuteTools:
    @staticmethod
    def get_up_space(vol_or_amt, trading_day, std_multiplier=1.0):
        vol_ma = vol_or_amt.rolling(20, min_periods=5).mean().replace(0, np.nan)
        vol_std = vol_or_amt.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((vol_or_amt - vol_ma) / vol_std).fillna(0)
        is_high = (z > std_multiplier).astype(float)
        return is_high, z


class _TalibStub:
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        if not hasattr(high, "shift"):
            high = pd.Series(high); low = pd.Series(low); close = pd.Series(close)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(timeperiod, min_periods=1).mean()

    @staticmethod
    def EMA(data, timeperiod=30):
        if not hasattr(data, "ewm"):
            data = pd.Series(data)
        return data.ewm(span=timeperiod, adjust=False, min_periods=1).mean()

    @staticmethod
    def SMA(data, timeperiod=30):
        if not hasattr(data, "rolling"):
            data = pd.Series(data)
        return data.rolling(timeperiod, min_periods=1).mean()

    @staticmethod
    def STDDEV(data, timeperiod=5, nbdev=1):
        if not hasattr(data, "rolling"):
            data = pd.Series(data)
        return data.rolling(timeperiod, min_periods=1).std()


alpha_tools = _AlphaTools()
minute_tools = _MinuteTools()
talib = _TalibStub()


# ============================================================
# 中间量（复用 weekly_factor_backtest._build_intermediates）
# ============================================================
def _build_intermediates(mkt, symbols, dates):
    """从 OHLCV + 基本面 预计算中间列（DataFrame, date×symbol）"""
    close = mkt.get("close", pd.DataFrame(index=dates, columns=symbols))
    op = mkt.get("open", pd.DataFrame(index=dates, columns=symbols))
    hi = mkt.get("high", pd.DataFrame(index=dates, columns=symbols))
    lo = mkt.get("low", pd.DataFrame(index=dates, columns=symbols))
    vol = mkt.get("volume", pd.DataFrame(index=dates, columns=symbols))
    amt = mkt.get("amount", pd.DataFrame(index=dates, columns=symbols))

    def reindex(df):
        if df is None or df.empty:
            return pd.DataFrame(np.nan, index=dates, columns=symbols)
        return df.reindex(index=dates, columns=symbols, copy=False)

    close_ = reindex(close)
    op_ = reindex(op)
    hi_ = reindex(hi)
    lo_ = reindex(lo)
    vol_ = reindex(vol)
    amt_ = reindex(amt)

    r_t = close_.pct_change().fillna(0)
    intraday_ret = (close_ - op_).fillna(0)
    range_ = hi_ - lo_
    vwap_approx = amt_.replace(0, np.nan) / vol_.replace(0, np.nan)
    vwap_approx = vwap_approx.fillna(close_)

    cols = {}
    for name, df in [
        ("close", close_), ("open", op_), ("high", hi_), ("low", lo_),
        ("volume", vol_), ("amount", amt_), ("range", range_),
        ("r_t", r_t), ("intraday_ret", intraday_ret),
        ("C", close_), ("V", vol_),
        ("vol_ratio", (vol_ / vol_.rolling(20, min_periods=1).mean()).fillna(1)),
        ("overnight_return", (op_ / close_.shift(1) - 1).fillna(0)),
        ("weighted", vwap_approx),
        ("close_price", close_),
        ("prior_high", hi_.shift(1)),
        ("prior_low", lo_.shift(1)),
        ("prior_close", close_.shift(1)),
        ("prior_bar_close", close_.shift(1)),
        ("prior_15", close_.shift(15)),
        ("bar_low", lo_),
        ("shift_close_5", close_.shift(5)),
    ]:
        cols[name] = df

    for span in [5, 10, 15, 20, 24, 30, 45, 63]:
        cols[f"EMA_{span}"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"EWMA_{span}"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"ema_{span}d"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"ema_span_{span}"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"EMA_vol_{span}"] = vol_.ewm(span=span, adjust=False).mean()
        cols[f"EWMA_ret_{span}"] = r_t.ewm(span=span, adjust=False).mean()
        cols[f"EMA_ret_{span}"] = r_t.ewm(span=span, adjust=False).mean()

    for span in [5, 10, 15, 20, 30, 40, 60]:
        cols[f"mean_{span}d"] = close_.rolling(span, min_periods=1).mean()
        cols[f"sum_{span}"] = close_.rolling(span, min_periods=1).sum()
        cols[f"vol_{span}d"] = vol_.rolling(span, min_periods=1).mean()
        cols[f"vol_rolling_{span}"] = vol_.rolling(span, min_periods=1).mean()
        cols[f"std_{span}d"] = close_.rolling(span, min_periods=2).std()
        cols[f"vol_std_{span}d"] = vol_.rolling(span, min_periods=2).std()

    for span in [20, 30, 60]:
        cols[f"vol_realized_{span}d"] = r_t.rolling(span, min_periods=2).std() * np.sqrt(252)
        cols[f"up_vol_{span}"] = (r_t * (r_t > 0)).rolling(span, min_periods=2).std() * np.sqrt(252)
        cols[f"down_vol_{span}"] = (-r_t * (r_t < 0)).rolling(span, min_periods=2).std() * np.sqrt(252)

    cols["volume_rank_20"] = vol_.rolling(20, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    cols["volume_rank_30"] = vol_.rolling(30, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    vol_ma20 = vol_.rolling(20, min_periods=5).mean()
    cols["transition_into_high_volume"] = (vol_ / vol_ma20).fillna(1)

    prev_close = close_.shift(1)
    tr1 = hi_ - lo_
    tr2 = (hi_ - prev_close).abs()
    tr3 = (lo_ - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for span in [14, 21]:
        cols[f"atr_{span}"] = tr.rolling(span, min_periods=1).mean()
        cols[f"atr_pct_{span}"] = cols[f"atr_{span}"] / close_

    for span in [20, 30]:
        cols[f"vol_sum_{span}d"] = vol_.rolling(span, min_periods=1).sum()

    size_z_val = np.log(vol_.rolling(20, min_periods=5).mean())
    mkt_cap = np.log(close_ * vol_)
    cols["size_z"] = (size_z_val - size_z_val.rolling(20, min_periods=5).mean()) / size_z_val.rolling(20, min_periods=5).std()
    cols["size_z"] = cols["size_z"].fillna(0).clip(-10, 10)
    cols["intercept"] = pd.DataFrame(1.0, index=dates, columns=symbols)

    vol_std20 = vol_.rolling(20, min_periods=5).std()
    vol_ma20b = vol_.rolling(20, min_periods=5).mean()
    cols["style_gate_resvol_high"] = (vol_std20 / vol_ma20b.replace(0, np.nan)).fillna(1)

    for span in [5, 10, 20, 40]:
        cols[f"sma_{span}"] = close_.rolling(span, min_periods=1).mean()
        cols[f"ts_mean_{span}d"] = close_.rolling(span, min_periods=1).mean()
        cols[f"ts_mean_close_{span}"] = close_.rolling(span, min_periods=1).mean()

    cols["free_turn"] = vol_.rolling(20, min_periods=5).mean()
    cols["turn_20d"] = vol_.rolling(20, min_periods=5).mean()
    cols["pb_lf"] = close_ / (close_ + 1e-10)
    cols["vwap_dev"] = (close_ - vwap_approx) / close_.replace(0, np.nan)

    vol_5 = r_t.rolling(5, min_periods=2).std()
    vol_20 = r_t.rolling(20, min_periods=5).std()
    cols["vol_ratio_fwd"] = (vol_5 / vol_20.replace(0, np.nan)).fillna(1)

    cols["daily_mean_close"] = close_
    vol_median = vol_.rolling(20, min_periods=5).median()
    cols["classify_vol_tool"] = (vol_ / vol_median.replace(0, np.nan)).fillna(1)
    cols["vol_tool_adaptive"] = cols["classify_vol_tool"]

    cols["sum_20_ret"] = r_t.rolling(20, min_periods=1).sum()
    cols["sum_20_abs_ret"] = r_t.abs().rolling(20, min_periods=1).sum()
    cols["ret_shifted"] = r_t.shift(1)
    cols["close_ret_5d"] = close_.pct_change(5).fillna(0)
    cols["close_shifted"] = close_.shift(1)

    # 真实基本面列（来自 StockValuationDaily）
    for fc in ["pb_lf", "pe_ttm", "mkt_cap_float", "free_turn"]:
        if fc in mkt and mkt[fc] is not None:
            cols[fc] = reindex(mkt[fc])

    # abs_ret 列
    cols["abs_ret"] = r_t.abs()

    return cols


# ============================================================
# 因子执行器（复用 weekly_factor_backtest._execute_factor_code）
# ============================================================
def _execute_factor_code(code, mkt, symbols, dates, intermediates=None):
    close = mkt.get("close", pd.DataFrame(0.0, index=dates, columns=symbols))
    open_ = mkt.get("open", pd.DataFrame(0.0, index=dates, columns=symbols))
    high = mkt.get("high", pd.DataFrame(0.0, index=dates, columns=symbols))
    low = mkt.get("low", pd.DataFrame(0.0, index=dates, columns=symbols))
    vol = mkt.get("volume", pd.DataFrame(0.0, index=dates, columns=symbols))
    amt = mkt.get("amount", pd.DataFrame(0.0, index=dates, columns=symbols))

    close = close.fillna(0.0)
    open_ = open_.fillna(close)
    high = high.fillna(close)
    low = low.fillna(close)
    vol = vol.fillna(0.0)
    amt = amt.fillna(0.0)

    ns = {
        "np": np, "pd": pd,
        "alpha_tools": alpha_tools,
        "minute_tools": minute_tools,
        "talib": talib,
        "__intermediates__": intermediates or {},
    }

    func_match = re.search(r'def\s+(factor_[a-zA-Z0-9_]+)\s*\(', code)
    func_name = func_match.group(1) if func_match else None

    # 只注入该因子 code 实际引用到的中间列（内存优化，16 workers 不再各持全量 14GB）
    used_mid = set(re.findall(r"df_copy\['([a-zA-Z_][a-zA-Z0-9_]*)+'\]", code))
    used_mid |= set(re.findall(r"df_copy\[\"([a-zA-Z_][a-zA-Z0-9_]*)+", code))
    used_mid |= set(re.findall(r"df\[['\\\"]([a-zA-Z_][a-zA-Z0-9_]*)+['\\\"]\]", code))
    # 把含 "factor_" 的列名过滤掉（那些是输出列，不是输入）
    used_mid = {c for c in used_mid if not c.startswith("factor_")}
    used_cols_set = set()
    for cname, cdf in (intermediates or {}).items():
        if cname in used_mid:
            used_cols_set.add(cname)
    used_cols_set |= {"close", "open", "high", "low", "volume", "amount", "close_price", "TradingDay"}
    if "TradingDay" in code:
        used_cols_set.add("TradingDay")

    sym_series = {}
    for sym in symbols:
        df = pd.DataFrame({
            "close": close[sym].values,
            "open": open_[sym].values,
            "high": high[sym].values,
            "low": low[sym].values,
            "volume": vol[sym].values,
            "amount": amt[sym].values,
            "close_price": close[sym].values,
            "TradingDay": dates,
        }, index=dates)
        df.index.name = "date"

        # 中间列 + 基本面列（只注入用到的）
        for cname in used_cols_set:
            if cname in df.columns:
                continue
            cdf = (intermediates or {}).get(cname)
            if isinstance(cdf, pd.DataFrame) and sym in cdf.columns:
                df[cname] = cdf[sym].reindex(dates).values

        local_ns = dict(ns)
        local_ns["df"] = df
        try:
            exec(code, local_ns)
            result = None
            if func_name and func_name in local_ns:
                result = local_ns[func_name](df)
            if result is None:
                for col in df.columns:
                    if col.startswith("factor_"):
                        result = df[col]
                        break
            if result is not None:
                if isinstance(result, pd.Series):
                    # 标量型因子（如 amount_weighted_squared_impact）会返回单元素 Series
                    if result.name and str(result.name).startswith("factor_"):
                        if len(result) == 1:
                            # 展开到全时间轴
                            val = result.iloc[0] if np.isfinite(result.iloc[0]) else np.nan
                            sym_series[sym] = pd.Series(val, index=dates, name=result.name)
                        else:
                            sym_series[sym] = result.reindex(dates)
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        if col.startswith("factor_"):
                            sym_series[sym] = result[col].reindex(dates)
                            break
        except Exception:
            pass

    if not sym_series:
        return pd.DataFrame(index=dates, columns=symbols)
    mat = pd.DataFrame(index=dates, columns=symbols)
    for sym, s in sym_series.items():
        if sym in mat.columns:
            mat[sym] = s.reindex(dates).values
    return mat.astype(float)


def _compute_factor_chunk(chunk, mkt_path, int_path, symbols, dates_list):
    import pickle as _pickle
    import numpy as np
    import pandas as pd
    with open(mkt_path, "rb") as fh:
        cache = _pickle.load(fh)
    mkt = cache["market_data"]
    syms = cache["symbols"]
    dts = pd.DatetimeIndex(dates_list)
    with open(int_path, "rb") as fh:
        intermediates = _pickle.load(fh)

    results = {}
    ok = 0
    for f in chunk:
        name = f["factor_name"]
        code = f.get("code", "") or ""
        if not code or "NotImplementedError" in code:
            results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
            continue
        try:
            mat = _execute_factor_code(code, mkt, syms, dts, intermediates=intermediates)
            if mat.empty:
                results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
                continue
            mat = mat.reindex(index=dts, columns=syms)
            valid_pct = mat.notna().sum().sum() / mat.size
            if valid_pct > 0.05:
                results[name] = mat
                ok += 1
            else:
                results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
        except Exception:
            results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
    return results, ok


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("扩展回测: 61 因子 2019-01-02 ~ 2026-08-24")
    print("=" * 60)

    # 1. 因子定义
    names = load_61_factor_names()
    print(f"[1] 因子页数量: {len(names)}")
    defs = load_factor_defs(names)
    print(f"[1] 可用因子 code: {len(defs)}")

    # 2. 股票池 + 行情矩阵
    print("[2] 加载行情矩阵 ...")
    mkt = {}
    for col in ["open", "high", "low", "close", "volume", "amount", "pre_close", "adj_factor"]:
        p = Path(f"/tmp/mkt_{col}.parquet")
        mkt[col] = pd.read_parquet(p) if p.exists() else None
    # 基本面列
    for fc in ["pb_lf", "pe_ttm", "mkt_cap_float", "free_turn"]:
        p = Path(f"/tmp/fund_{fc}.parquet")
        mkt[fc] = pd.read_parquet(p) if p.exists() else None

    dates = mkt["close"].index
    symbols = list(mkt["close"].columns)
    print(f"[2] dates {dates[0]} ~ {dates[-1]} ({len(dates)} 天), {len(symbols)} 股")

    # 3. 中间量
    print("[3] 构建中间量 ...")
    intermediates = _build_intermediates(mkt, symbols, dates)
    print(f"[3] {len(intermediates)} 个中间列")

    # 4. 并行计算因子
    # 只 pickle 精简后的 intermediates（按 code 用到的列）以控制 worker 内存
    import pickle as _pickle, tempfile
    tmp = Path(tempfile.gettempdir())
    mkt_path = tmp / "mkt_data_cache_20260825.pkl"
    int_path = tmp / "intermediates_cache_20260825.pkl"
    with open(mkt_path, "wb") as fh:
        _pickle.dump({"market_data": mkt, "symbols": symbols, "dates": dates}, fh)

    # 聚合所有因子 code 用到的中间列（减少 worker 内 intermediates 体积）
    all_used = set()
    for d in defs:
        code = d["code"]
        all_used |= set(re.findall(r"df_copy\['([a-zA-Z_][a-zA-Z0-9_]*)+'\]", code))
        all_used |= set(re.findall(r"df_copy\[\"([a-zA-Z_][a-zA-Z0-9_]*)+", code))
        all_used |= set(re.findall(r"df\[['\\\"]([a-zA-Z_][a-zA-Z0-9_]*)+['\\\"]\]", code))
        all_used |= {"close", "open", "high", "low", "volume", "amount", "close_price", "TradingDay"}
    all_used = {c for c in all_used if not c.startswith("factor_")}
    slim_int = {k: v for k, v in intermediates.items() if k in all_used}
    print(f"[3b] 精简中间列: {len(intermediates)} -> {len(slim_int)}")
    with open(int_path, "wb") as fh:
        _pickle.dump(slim_int, fh)
    del intermediates, slim_int

    n_workers = min(16, os.cpu_count() or 4)
    chunk_size = max(1, len(defs) // n_workers)
    chunks = [defs[i:i + chunk_size] for i in range(0, len(defs), chunk_size)]

    results = {}
    success = 0
    print(f"[4] 并行计算 {len(defs)} 个因子, {n_workers} workers ...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_compute_factor_chunk, ch, str(mkt_path), str(int_path), symbols, dates.tolist()) for ch in chunks]
        for fut in as_completed(futures):
            cr, cok = fut.result(timeout=900)
            results.update(cr)
            success += cok
            print(f"  [4] 完成 {len(results)}/{len(defs)} 因子")

    # 5. 组装 MultiIndex parquet
    print("[5] 组装 factor_values.parquet ...")
    cols = []
    mats = {}
    for d in defs:
        pn = d["page_name"]
        fn = d["factor_name"]
        mat = results.get(fn)
        if mat is None:
            # 部分 code 里函数名与 factor_name 不完全一致，按 page_name 兜底找
            for k, v in results.items():
                if pn in k or k.replace("factor_", "") == pn.replace("_flipped", ""):
                    mat = v
                    fn = k
                    break
        if mat is None:
            print(f"  [WARN] {pn}: 无结果")
            continue
        # 存到 page_name 列（与详情页一致）
        for sym in mat.columns:
            mats[(pn, sym)] = mat[sym].values
        print(f"  [5] {pn}: valid={mat.notna().mean().mean():.1%}")

    idx = dates
    fv = pd.DataFrame(mats, index=idx)
    fv.columns = pd.MultiIndex.from_tuples(fv.columns, names=["factor", "symbol"])
    fv = fv.sort_index(axis=1)
    fv = fv.astype(np.float32)
    print(f"[5] final: {fv.shape}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fv.to_parquet(OUT_PATH)
    print(f"[6] saved: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB), 用时 {time.time()-t0:.0f}s")
    print(f"[6] 成功因子: {success}/{len(defs)}")


if __name__ == "__main__":
    main()

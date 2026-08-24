#!/usr/bin/env python3
"""
为分钟因子补充 minute_tools.amount_weighted_mean，
下载分钟数据，并行跑 7 个缺失的因子。
"""

from __future__ import annotations
import sys, os, json, re, tempfile as _tempfile, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))

import weekly_factor_backtest as _wfb

COS_BUCKET = "quantsociety-cold-data-1425188104"
COS_SYNC = "sudo -n /usr/local/libexec/quantsociety-cos/research-cos sync"
LOCAL_DATA_ROOT = Path.home() / "cos_data"


def _cos_download_file(key: str, destination: Path) -> bool:
    import subprocess
    try:
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = f'{COS_SYNC} "cos://{COS_BUCKET}/{key}" "{dst}"'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000
    except Exception:
        return False


def _download_minute_bars(start: str, end: str) -> list:
    from datetime import datetime as dt
    minute_dir = LOCAL_DATA_ROOT / "StockMinuteBar"
    minute_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in minute_dir.glob("*.parquet")}

    sdt = dt.strptime(start, "%Y-%m-%d")
    edt = dt.strptime(end, "%Y-%m-%d")
    needed = []
    cur = sdt
    while cur <= edt:
        fname = cur.strftime("%Y-%m-%d")
        if fname not in existing:
            needed.append(fname)
        cur += timedelta(days=1)

    if not needed:
        print(f"[分钟] 已有 {len(existing)} 天本地文件")
        return list(minute_dir.glob("*.parquet"))

    print(f"[分钟] 需下载 {len(needed)} 天...")
    done = 0
    for day in needed:
        dst = minute_dir / f"{day}.parquet"
        if dst.exists():
            continue
        key = f"clean_data/ashare/lqtp_data/StockMinuteBar/{day}.parquet"
        if _cos_download_file(key, dst):
            done += 1
            if done % 20 == 0:
                print(f"[分钟] {done}/{len(needed)}")

    print(f"[分钟] 下载完成: {done} 个新文件")
    return list(minute_dir.glob("*.parquet"))


# ============================================================
# 增强 minute_tools
# ============================================================
def _build_minute_tools():
    class EnhancedMinuteTools:
        @staticmethod
        def get_up_space(vol_or_amt, trading_day, std_multiplier=1.0):
            vol_ma = vol_or_amt.rolling(20, min_periods=5).mean().replace(0, np.nan)
            vol_std = vol_or_amt.rolling(20, min_periods=5).std().replace(0, np.nan)
            z = ((vol_or_amt - vol_ma) / vol_std).fillna(0)
            is_high = (z > std_multiplier).astype(float)
            return is_high, z

        @staticmethod
        def amount_weighted_mean(values, amounts, trading_days):
            """按交易日分组，计算 amount 加权的 values 均值"""
            df_local = pd.DataFrame({"value": values, "amount": amounts, "date": trading_days})
            df_local = df_local.dropna(subset=["amount", "value"])
            if df_local.empty:
                return pd.Series(dtype=float)

            def w_mean(group):
                amt = group["amount"].values
                val = group["value"].values
                total = np.nansum(amt)
                if total == 0 or np.isnan(total):
                    return np.nan
                return np.nansum(amt * val) / total

            result = df_local.groupby("date", sort=True).apply(w_mean, include_groups=False)
            result.index = pd.to_datetime(result.index)
            return result

    return EnhancedMinuteTools()


# ============================================================
# 核心：跑单个因子（处理所有股票/日期）
# ============================================================
def run_single_minute_factor(args) -> tuple:
    """Worker: 对一个因子计算每日因子值，返回 DataFrame"""
    import warnings as _warn
    _warn.filterwarnings("ignore", message="Calling float on a single element Series")
    _warn.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import duckdb

    factor_dict, files_str, syms_str, symbols, dates_list = args
    nm = factor_dict["factor_name"]
    code = factor_dict["code"]
    enhanced_tools = _build_minute_tools()

    func_match = re.search(r'def\s+(factor_[a-zA-Z0-9_]+)\s*\(', code)
    func_name = func_match.group(1) if func_match else None

    dates_idx = pd.to_datetime(dates_list)
    mat = pd.DataFrame(np.nan, index=dates_idx, columns=symbols, dtype=float)

    ns_base = {
        "np": np, "pd": pd,
        "minute_tools": enhanced_tools,
        "__intermediates__": {},
    }

    conn = duckdb.connect(database=":memory:")

    for i, dt in enumerate(dates_list):
        try:
            df = conn.execute(f"""
                SELECT Symbol, Close, Volume, Amount, Vwap
                FROM read_parquet({files_str})
                WHERE Symbol IN {syms_str} AND TradeDate = '{dt}'
                ORDER BY Symbol
            """).fetchdf()
        except Exception:
            continue

        if df.empty:
            continue

        day_dt = pd.Timestamp(dt)
        sym_groups = df.groupby("Symbol")

        for sym, sym_df in sym_groups:
            if sym not in symbols or len(sym_df) < 10:
                continue

            df_bar = pd.DataFrame({
                "close_price": sym_df["Close"].values,
                "close": sym_df["Close"].values,
                "open": sym_df["Close"].values,
                "high": sym_df["Close"].values,
                "low": sym_df["Close"].values,
                "volume": sym_df["Volume"].values.astype(float),
                "amount": sym_df["Amount"].values,
                "vwap": sym_df["Vwap"].values,
                "TradingDay": [dt] * len(sym_df),
            })

            local_ns = dict(ns_base)
            local_ns["df"] = df_bar

            try:
                exec(code, local_ns)
                result = None
                if func_name and func_name in local_ns:
                    result = local_ns[func_name](df_bar)
                if result is not None:
                    val = float(result.iloc[0]) if (hasattr(result, "iloc") and len(result) > 0) else (float(result.values[0]) if hasattr(result, "values") else float(result))
                    if np.isfinite(val):
                        mat.loc[day_dt, sym] = val
            except Exception:
                pass

        if (i + 1) % 100 == 0:
            print(f"  [{nm}] {i+1}/{len(dates_list)} 天")

    conn.close()
    return nm, mat


# ============================================================
# 主流程
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2024-01-02")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    out_dir = _PROJECT_ROOT / "weekly_backtest_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "backtest_results.json"

    with open(results_path) as f:
        existing = json.load(f)

    # 7 个需要分钟数据的因子
    minute_names = [
        "factor_amount_weighted_impact",
        "factor_high_volume_amount_weighted_squared_impact",
        "factor_high_volume_volume_weighted_squared_impact",
        "factor_high_volume_amount_weighted_impact",
        "factor_high_volume_squared_return_weighted_impact",
        "factor_vwap_squared_deviation_intensity",
        "factor_vwap_deviation_intensity_squared_amount_weighted",
    ]

    # 加载 code
    factors = []
    for nm in minute_names:
        fname = _PROJECT_ROOT / "factor_delivery_converted" / "factors_combined" / f"{nm}.json"
        if fname.exists():
            with open(fname) as f:
                d = json.load(f)
            code = d.get("code", "")
            if code and "NotImplementedError" not in code:
                factors.append({
                    "factor_name": nm,
                    "code": code,
                    "status": "EXEC",
                })

    print(f"[因子] {len(factors)} 个分钟因子待计算")

    # 股票池
    symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
    syms_str = "('" + "','".join(symbols) + "')"
    dates_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start, args.end, freq="B")]

    # 下载分钟数据
    print("\n[分钟数据] 下载中...")
    local_files = _download_minute_bars(args.start, args.end)
    files_str = "[" + ",".join(f"'{f}'" for f in sorted(local_files)) + "]"

    # 行情（用于回测）
    market_data = _wfb.load_market_data(symbols, args.start, args.end)
    fwd_rets_path = Path(_tempfile.gettempdir()) / "fwd_rets_minute.parquet"
    market_data["close"].to_parquet(str(fwd_rets_path))

    # ============================================================
    # Step 1: 并行计算所有因子的分钟因子值
    # ============================================================
    print(f"\n[计算] {len(factors)} 个因子值（{len(dates_list)} 天 x {len(symbols)} 股）...")

    tasks = [
        (f, files_str, syms_str, symbols, dates_list)
        for f in factors
    ]

    all_mats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_single_minute_factor, t): t[0]["factor_name"] for t in tasks}
        for future in as_completed(futures):
            nm = futures[future]
            try:
                name, mat = future.result(timeout=1200)
                all_mats[name] = mat
                pct = mat.notna().sum().sum() / mat.size * 100
                print(f"  ✓ {name}: 有效数据 {pct:.1f}%, 形状 {mat.shape}")
            except Exception as e:
                print(f"  ✗ {nm}: {e}")

    # 合并因子值
    if not all_mats:
        print("[错误] 没有成功计算的因子")
        return

    fv_combined = pd.concat(all_mats, axis=1)
    fv_combined.columns = pd.MultiIndex.from_tuples(
        [(c, s) for c, s in fv_combined.columns]
    )
    print(f"\n[因子值] 合并后: {fv_combined.shape}")

    # 加载行情（用于获取真实交易日）
    market_data = _wfb.load_market_data(symbols, args.start, args.end)
    # 获取真实交易日列表（只有这些日期才有效）
    actual_trading_dates = market_data["close"].index
    print(f"[数据] 真实交易日: {len(actual_trading_dates)} 天")

    # 保存因子值
    existing_fv = pd.read_parquet(str(out_dir / "factor_values.parquet"))

    # 过滤：分钟因子日期只保留真实交易日
    fv_combined = fv_combined.loc[fv_combined.index.intersection(actual_trading_dates)]
    print(f"[因子值] 过滤到交易日: 保留 {len(fv_combined)} 天")

    combined_fv = pd.concat([existing_fv, fv_combined], axis=1)
    combined_fv.to_parquet(str(out_dir / "factor_values.parquet"))
    print(f"[因子值] 已更新: {combined_fv.shape}")

    # ============================================================
    # Step 2: 回测
    # ============================================================
    print(f"\n[回测] {len(all_mats)} 个因子...")
    new_results = {}
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for nm, mat in all_mats.items():
            f_dict = next((f for f in factors if f["factor_name"] == nm), {"factor_name": nm, "code": "", "status": "EXEC"})
            # 过滤到实际交易日
            mat_filtered = mat.loc[mat.index.intersection(actual_trading_dates)]
            future = pool.submit(
                _wfb._backtest_single_factor,
                (mat_filtered.to_dict(), f_dict),
                str(fwd_rets_path),
                10_000_000.0, 0.2, 0.2,
            )
            futures[future] = nm

        for future in as_completed(futures):
            nm = futures[future]
            try:
                _, f_dict, result = future.result(timeout=300)
                new_results[nm] = result
                if result.get("success"):
                    s = result["stats"]
                    print(f"  ✓ {nm}: Sharpe={s.get('Sharpe Ratio', 0):+.2f}, IC={s.get('IC', 0):+.4f}")
                else:
                    print(f"  ✗ {nm}: {result.get('error', '?')[:80]}")
            except Exception as e:
                print(f"  ✗ {nm}: {e}")

    # ============================================================
    # Step 3: 合并结果 + 翻转
    # ============================================================
    all_results = {**existing, **new_results}

    flip_map = {}
    for name, res in all_results.items():
        if name.endswith("_flipped") or not res.get("success"):
            continue
        ic = res["stats"].get("IC", 0)
        if ic < -0.0001:
            flip_name = name + "_flipped"
            if flip_name not in all_results:
                flip_map[name] = flip_name

    for orig, flipped in flip_map.items():
        s = all_results[orig]["stats"]
        flip_stats = {
            **s,
            "Sharpe Ratio": -s["Sharpe Ratio"],
            "Annualized Return (%)": -s["Annualized Return (%)"],
            "Max Drawdown (%)": abs(s["Max Drawdown (%)"]) if s["Max Drawdown (%)"] < 0 else s["Max Drawdown (%)"],
            "IC": -s["IC"],
            "Rank IC": -s["Rank IC"],
        }
        all_results[flipped] = {"success": True, "stats": flip_stats}

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # 汇总
    pos = sum(1 for v in all_results.values() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0)
    total_ok = sum(1 for v in all_results.values() if v.get("success"))
    new_ok = sum(1 for v in new_results.values() if v.get("success"))
    new_pos = sum(1 for v in new_results.values() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0)

    print("\n" + "=" * 60)
    print("  分钟因子完成")
    print(f"  新增成功: {new_ok} 个 | 正 Sharpe: {new_pos} 个")
    print(f"  累计成功: {total_ok} 个 | 正 Sharpe: {pos} 个")
    print(f"  新增翻转: {len(flip_map)} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()

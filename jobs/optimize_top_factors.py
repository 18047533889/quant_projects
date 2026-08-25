#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段2：Top 因子参数寻优（factor_optimizer SearchRunner）。

从阶段1结果选 Top N（按优化后 RankIC IR 排序），对每个 Top 因子做参数寻优：
用 factor_optimizer 的 SearchRunner + parameter_tune/window_adjust 变异，
objective = 最大化 RankIC IR。proposal_fn 生成"预处理参数变体"作为 Trial，
evaluation_fn 用 quant_evaluator 算 RankIC IR。

输出：
  weekly_backtest_output/optimized_top/<page>.parquet   （寻优后因子）
  weekly_backtest_output/optimized_top_meta.json        （变异记录 + 公式化表述）
"""
import sys, os, json, glob, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))
sys.path.insert(0, str(PROJECT / "factor_optimizer"))
sys.path.insert(0, str(PROJECT / "factor_preprocess"))

from quant_evaluator.metrics.ic import _spearman_rank_correlation
from factor_preprocess.registry.transforms import create_default_registry

OPT_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
OUT_DIR = PROJECT / "weekly_backtest_output" / "optimized_top"
META_PATH = PROJECT / "weekly_backtest_output" / "optimized_top_meta.json"
OPT1_META = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
DAILY = Path.home() / "cos_data" / "StockDailyBar"
START, END = "2019-01-02", "2026-08-24"
TOP_N = 30

_reg = create_default_registry()
cs_winsor = _reg.get_function("cs_winsor")
cs_zscore = _reg.get_function("cs_zscore")

_HAS_VWAP = None


def load_vwap():
    """加载 VWAP 矩阵（vwap-to-vwap 收益口径）。"""
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    files = sorted(DAILY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, Vwap as vwap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    m = df.pivot_table(index='date', columns='symbol', values='vwap', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    _HAS_VWAP = m.sort_index()
    return _HAS_VWAP


def daily_rankic_ir(factor_mat, vwap):
    """返回 RankIC IR（vwap-to-vwap 收益口径）。"""
    common = factor_mat.index.intersection(vwap.index)
    fv = factor_mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols]; vv = vv[cols]
    fwd = vv.pct_change().shift(-1)  # vwap-to-vwap
    T = fv.shape[0]
    ic = np.full(T, np.nan)
    fv_a = fv.values; fwd_a = fwd.values
    for t in range(T):
        m = fv_a[t]; r = fwd_a[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < 20:
            continue
        ic[t] = _spearman_rank_correlation(m[mask], r[mask])
    s = pd.Series(ic, index=common).dropna()
    if len(s) == 0:
        return 0.0, 0.0
    mean = float(s.mean()); std = float(s.std())
    return mean, (mean / std if std > 1e-9 else 0.0)


def apply_params(mat, winsor_lower, winsor_upper, zscore, smooth_win):
    """按参数应用预处理。返回变换后矩阵。"""
    out = mat.copy()
    if winsor_lower is not None:
        out = pd.DataFrame(cs_winsor(out.values, lower=winsor_lower, upper=winsor_upper, axis=1),
                           index=mat.index, columns=mat.columns)
    if zscore:
        out = pd.DataFrame(cs_zscore(out.values, axis=1, ddof=1),
                            index=mat.index, columns=mat.columns)
    if smooth_win and smooth_win > 1:
        out = out.rolling(smooth_win, min_periods=1).mean()
    return out


def param_search(page, mat, vwap):
    """对单个因子做参数网格寻优。返回 (best_params, best_ir, best_mat, best_mean)。"""
    # 参数网格
    winsor_opts = [(0.01, 0.99), (0.02, 0.98), (0.05, 0.95), None]
    zscore_opts = [True, False]
    smooth_opts = [1, 3, 5, 10]
    best_ir = -1e9; best_mean = 0.0; best_params = None; best_mat = None
    for wl, wu in winsor_opts:
        for zs in zscore_opts:
            for sw in smooth_opts:
                try:
                    tmat = apply_params(mat, wl, wu, zs, sw)
                    mean, ir = daily_rankic_ir(tmat, vwap)
                    if ir > best_ir:
                        best_ir = ir; best_mean = mean; best_params = (wl, wu, zs, sw)
                        best_mat = tmat
                except Exception:
                    continue
    return best_params, best_ir, best_mat, best_mean


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vwap = load_vwap()
    common_idx = vwap.index
    common_cols = vwap.columns

    # 读阶段1 meta，选 Top N
    if not OPT1_META.exists():
        print("[opt2] 无阶段1 meta，先跑 optimize_factors.py")
        return
    opt1 = json.loads(OPT1_META.read_text())
    ranked = sorted(opt1.items(), key=lambda kv: kv[1].get("best_rankic_ir", 0), reverse=True)
    top = [page for page, m in ranked[:TOP_N] if m.get("best_rankic_ir", 0) > 0]
    print(f"[opt2] Top {len(top)} 因子 (vwap-to-vwap 口径)", flush=True)

    meta = {}
    t0 = time.time()
    for i, page in enumerate(top):
        fpath = OPT_DIR / f"{page}.parquet"
        if not fpath.exists():
            continue
        mat = pd.read_parquet(fpath).reindex(index=common_idx, columns=common_cols)
        if mat.shape[1] == 0:
            continue
        best_params, best_ir, best_mat, best_mean = param_search(page, mat, vwap)
        if best_mat is not None:
            sub = best_mat.astype("float32").dropna(axis=1, how="all")
            sub.to_parquet(OUT_DIR / f"{page}.parquet")
        # 公式化表述
        wl, wu, zs, sw = best_params
        steps = []
        if wl is not None:
            steps.append(f"cs_winsor({wl}, {wu})")
        if zs:
            steps.append("cs_zscore")
        if sw and sw > 1:
            steps.append(f"rolling_mean({sw})")
        meta[page] = {
            "best_params": {"winsor": [wl, wu], "zscore": zs, "smooth": sw},
            "best_rankic_ir": best_ir,
            "best_mean_rankic": best_mean,
            "steps": steps,
            "formula": " → ".join(steps) if steps else "原始因子",
        }
        if i % 5 == 0:
            print(f"  {i}/{len(top)} {page} ir={best_ir:.3f} 耗时{time.time()-t0:.0f}s", flush=True)

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"[opt2] 完成 {len(meta)} 因子, 存 {META_PATH}, 耗时{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

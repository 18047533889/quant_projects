#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1：456 因子预处理自动化 + 择优。

对每个因子：
1. 按需判定：检测 DSL 是否已含 cs_zscore/cs_rank/cs_demean/cs_winsorize/neutralize/standardize 等算子，
   避免重复预处理；检测因子值分布（极端值比例、是否已标准化、是否已中性化）。
2. 择优：生成 2-4 个预处理变体（原始 / 去极值 / 去极值+标准化 / 去极值+标准化+中性化 / 全管线），
   用 quant_evaluator 各算一次 RankIC/IR，选最优变体作为"优化后因子"。
3. 公式化表述：记录每个因子的预处理步骤链。

输出：
  weekly_backtest_output/optimized_factors/<page>.parquet   （优化后因子矩阵）
  weekly_backtest_output/optimized_meta.json                （步骤链 + 最优变体 + 指标对比）
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
sys.path.insert(0, str(PROJECT / "factor_preprocess"))

from quant_evaluator.metrics.ic import _spearman_rank_correlation
from factor_preprocess.registry.transforms import create_default_registry

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OUT_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
META_PATH = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
LQTP_ALL = json.load(open("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"))
DAILY = Path.home() / "cos_data" / "StockDailyBar"
INDUSTRY = Path.home() / "cos_data" / "StockIndustry"
VALUATION = Path.home() / "cos_data" / "StockValuationDaily"
START, END = "2019-01-02", "2026-08-24"

# factor_engine 已含的预处理算子（检测到则跳过对应步骤）
FE_PREPROC_OPS = [
    "cs_zscore", "cs_rank", "cs_demean", "cs_winsorize", "cs_winsor",
    "neutralize", "industry_neutral", "market_cap_neutralize", "size_neutralize",
    "standardize", "zscore", "rank", "cs_scale", "winsorize", "normalize",
    "c_zscore", "c_rank", "c_demean", "c_winsorize", "c_scale",
]

_reg = create_default_registry()
cs_winsor = _reg.get_function("cs_winsor")
cs_zscore = _reg.get_function("cs_zscore")
cs_rank = _reg.get_function("cs_rank")
cs_scale = _reg.get_function("cs_scale")

_HAS_CLOSE = None
_HAS_VWAP = None
_HAS_IND = None
_HAS_MCAP = None


def load_close():
    global _HAS_CLOSE
    if _HAS_CLOSE is not None:
        return _HAS_CLOSE
    files = sorted(DAILY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, Close as close
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    m = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    _HAS_CLOSE = m.sort_index()
    return _HAS_CLOSE


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


def load_industry():
    global _HAS_IND
    if _HAS_IND is not None:
        return _HAS_IND
    files = sorted(INDUSTRY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, IndustryName
        FROM read_parquet({fs})
        WHERE IndustrySource = 'sw_l1' AND TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    df = df.drop_duplicates(subset=['date', 'symbol'], keep='first')
    m = df.pivot_table(index='date', columns='symbol', values='IndustryName', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    _HAS_IND = m.sort_index()
    return _HAS_IND


def load_mktcap():
    global _HAS_MCAP
    if _HAS_MCAP is not None:
        return _HAS_MCAP
    files = sorted(VALUATION.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, CirculatingMarketCap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    df["log_mktcap"] = np.log(df["CirculatingMarketCap"].replace(0, np.nan))
    m = df.pivot_table(index='date', columns='symbol', values='log_mktcap', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    _HAS_MCAP = m.sort_index()
    return _HAS_MCAP


def daily_rankic(factor_mat, vwap):
    """逐日 spearman rankic，收益口径 = vwap-to-vwap（vwap.pct_change().shift(-1)）。
    返回 (ic_series, mean_ric, ric_ir)。"""
    common = factor_mat.index.intersection(vwap.index)
    fv = factor_mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols]; vv = vv[cols]
    fwd = vv.pct_change().shift(-1)  # vwap-to-vwap 收益
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
        return s, 0.0, 0.0
    mean = float(s.mean()); std = float(s.std())
    return s, mean, (mean / std if std > 1e-9 else 0.0)


def neutralize_wide(fv, vwap, industry, mktcap):
    """逐日横截面回归取残差（行业哑变量 + log市值）。fv: date×symbol。"""
    common = fv.index.intersection(vwap.index)
    fv = fv.reindex(index=common)
    ind = industry.reindex(index=common, columns=fv.columns)
    mcap = mktcap.reindex(index=common, columns=fv.columns)
    ind_mat = ind.fillna("").values
    uniq = sorted({x for row in ind_mat for x in row if x})
    T, N = fv.shape
    K = len(uniq) + 2
    X = np.zeros((T, N, K), dtype=np.float64)
    X[:, :, 0] = 1.0
    for i, name in enumerate(uniq):
        X[:, :, 1 + i] = (ind_mat == name).astype(np.float64)
    X[:, :, 1 + len(uniq)] = mcap.values
    for i in range(len(uniq)):
        colsum = X[:, :, 1 + i].sum(axis=1)
        X[colsum == 0, :, 1 + i] = np.nan
    fv_a = fv.values.astype(np.float64)
    resid = np.full((T, N), np.nan)
    for t in range(T):
        y = fv_a[t]; Xt = X[t]
        ok = np.isfinite(y) & np.isfinite(Xt).all(axis=1)
        if ok.sum() < 30:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(Xt[ok], y[ok], rcond=None)
            resid[t, ok] = y[ok] - Xt[ok] @ coef
        except Exception:
            pass
    return pd.DataFrame(resid, index=common, columns=fv.columns)


def detect_dsl_preproc(page):
    """检测 DSL 是否已含预处理算子。返回已含的算子列表。"""
    rec = next((r for r in LQTP_ALL if r.get("page_name") == page), None)
    if not rec:
        return []
    dsl = (rec.get("dsl") or "") + " " + (rec.get("lqtp_formula") or "") + " " + (rec.get("fe_formula") or "")
    dsl_l = dsl.lower()
    found = [op for op in FE_PREPROC_OPS if op.lower() in dsl_l]
    return found


def build_variants(mat, vwap, industry, mktcap, dsl_ops):
    """生成预处理变体。返回 [(name, matrix, steps)]。"""
    variants = []
    # 变体0: 原始
    variants.append(("raw", mat.copy(), ["原始因子"]))

    # 判定是否已标准化 / 已中性化
    vals = mat.values
    finite = vals[np.isfinite(vals)]
    std_global = float(np.nanstd(finite)) if len(finite) > 1 else 0.0
    already_std = 0.5 < std_global < 2.0
    already_zscore = "cs_zscore" in dsl_ops or "zscore" in dsl_ops or "standardize" in dsl_ops
    already_neutral = any(op in dsl_ops for op in ["neutralize", "industry_neutral", "market_cap_neutralize", "size_neutralize"])
    already_winsor = "cs_winsor" in dsl_ops or "winsorize" in dsl_ops or "cs_winsorize" in dsl_ops

    # 变体1: 去极值 (若未去极值)
    if not already_winsor:
        w = cs_winsor(mat.values, lower=0.01, upper=0.99, axis=1)
        w = pd.DataFrame(w, index=mat.index, columns=mat.columns)
        variants.append(("winsor", w, ["cs_winsor(0.01, 0.99)"]))
    else:
        variants.append(("winsor", mat.copy(), ["（DSL 已含去极值，跳过）"]))

    # 变体2: 去极值 + 标准化 (若未标准化)
    base = variants[-1][1]
    if not already_zscore:
        z = cs_zscore(base.values, axis=1, ddof=1)
        z = pd.DataFrame(z, index=mat.index, columns=mat.columns)
        steps = variants[-1][2] + ["cs_zscore"]
        variants.append(("winsor_zscore", z, steps))
    else:
        variants.append(("winsor_zscore", base.copy(), variants[-1][2] + ["（DSL 已含标准化，跳过）"]))

    # 变体3: 去极值 + 标准化 + 中性化 (若未中性化)
    base2 = variants[-1][1]
    if not already_neutral:
        neu = neutralize_wide(base2, vwap, industry, mktcap)
        steps = variants[-1][2] + ["ols_neutralize(行业+log市值)"]
        variants.append(("winsor_zscore_neutral", neu, steps))
    else:
        variants.append(("winsor_zscore_neutral", base2.copy(), variants[-1][2] + ["（DSL 已含中性化，跳过）"]))

    return variants


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vwap = load_vwap()
    industry = load_industry()
    mktcap = load_mktcap()
    common_idx = vwap.index
    common_cols = vwap.columns
    industry = industry.reindex(index=common_idx, columns=common_cols)
    mktcap = mktcap.reindex(index=common_idx, columns=common_cols)

    names = sorted([f.stem for f in FV_DIR.glob("*.parquet")]) if FV_DIR.exists() else []
    print(f"[opt1] 因子数: {len(names)} (vwap-to-vwap 口径)", flush=True)

    meta = {}
    t0 = time.time()
    for i, page in enumerate(names):
        fpath = FV_DIR / f"{page}.parquet"
        try:
            mat = pd.read_parquet(fpath)
        except Exception:
            continue
        if mat.shape[1] == 0 or mat.isna().all().all():
            meta[page] = {"error": "empty", "best": "raw", "steps": ["（无有效值）"], "variants": {}}
            continue
        mat = mat.reindex(index=common_idx, columns=common_cols)

        dsl_ops = detect_dsl_preproc(page)
        variants = build_variants(mat, vwap, industry, mktcap, dsl_ops)

        # 择优：各变体算 RankIC/IR（vwap-to-vwap）
        best_name = "raw"; best_ir = -1e9; best_mean = 0.0
        var_metrics = {}
        for vname, vmat, vsteps in variants:
            s, mean, ir = daily_rankic(vmat, vwap)
            var_metrics[vname] = {"mean_rankic": mean, "rankic_ir": ir, "n": len(s)}
            if ir > best_ir:
                best_ir = ir; best_name = vname; best_mean = mean

        # 取最优变体矩阵
        best_mat = None
        for vname, vmat, vsteps in variants:
            if vname == best_name:
                best_mat = vmat
                best_steps = vsteps
                break

        # 写优化后因子
        if best_mat is not None:
            sub = best_mat.astype("float32").dropna(axis=1, how="all")
            sub.to_parquet(OUT_DIR / f"{page}.parquet")

        meta[page] = {
            "best": best_name,
            "best_mean_rankic": best_mean,
            "best_rankic_ir": best_ir,
            "steps": best_steps,
            "dsl_preproc_ops": dsl_ops,
            "variants": var_metrics,
        }
        if i % 50 == 0:
            print(f"  {i}/{len(names)} {page} best={best_name} ir={best_ir:.3f} 耗时{time.time()-t0:.0f}s", flush=True)

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"[opt1] 完成 {len(meta)} 因子, meta 存 {META_PATH}, 耗时{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

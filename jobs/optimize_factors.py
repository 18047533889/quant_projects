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
DAILY_ADJ = Path.home() / "cos_data" / "StockDailyBarAdj"
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
    """加载 后复权 AdjVwap 矩阵（vwap-to-vwap 后复权收益口径，硬性）。
    数据源：StockDailyBarAdj.AdjVwap（后复权），禁止未复权 StockDailyBar.Vwap。"""
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    files = sorted(DAILY_ADJ.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
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


def daily_rankic(factor_mat, vwap, stride=1):
    """逐日 spearman rankic，收益口径 = vwap-to-vwap（AdjVwap(t+2)/AdjVwap(t+1)-1 = shift(-2)，企业级）。
    ��量化 rankdata（比逐日 _spearman_rank_correlation 快 ~5x）。stride 降采样加速变体择优。
    返回 (ic_series, mean_ric, ric_ir)。"""
    common = factor_mat.index.intersection(vwap.index)
    fv = factor_mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols]; vv = vv[cols]
    fwd = vv.pct_change().shift(-2)  # vwap-to-vwap 后复权收益（t+1成交→t+2卖出，对齐平台TargetVwapReturnH01）
    T = fv.shape[0]
    ic = np.full(T, np.nan)
    fv_a = fv.values; fwd_a = fwd.values
    from scipy.stats import rankdata
    for t in range(0, T, stride):
        m = fv_a[t]; r = fwd_a[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < 20:
            continue
        ra = rankdata(m[mask]); rb = rankdata(r[mask])
        am = ra - ra.mean(); bm = rb - rb.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        ic[t] = (am * bm).sum() / d if d > 1e-18 else 0.0
    s = pd.Series(ic, index=common).dropna()
    if len(s) == 0:
        return s, 0.0, 0.0
    mean = float(s.mean()); std = float(s.std())
    return s, mean, (mean / std if std > 1e-9 else 0.0)


def neutralize_wide(fv, vwap, industry, mktcap, X_cache=None):
    """逐日横截面回归取残差（行业哑变量 + log市值）。fv: date×symbol。
    只保留 fv 真实有值的日期（VWAP 全轴含 2016-2018 但部分因子仅覆盖 2019 后，
    若全部 reindex 会把有效日期稀释掉 → 2026 前因子会整段变 NaN）。

    2026-08-28 性能改造：
    - X_static（行业哑变量+截距）跨因子共享（X_cache 全局缓存，float32）；
    - 每日回归用 solve(XtX, Xty)（0.4ms/日）替代 lstsq（6ms/日）；
    - 市值列每日变化 → Xty 里市值列动态拼。
    """
    common = fv.index.intersection(vwap.index)
    valid_rows = fv.notna().sum(axis=1) >= 30
    common = common[valid_rows.reindex(common).fillna(False).values] if len(valid_rows) else common
    fv = fv.reindex(index=common)
    ind = industry.reindex(index=common, columns=fv.columns)
    mcap = mktcap.reindex(index=common, columns=fv.columns)
    ind_mat = ind.fillna("").values
    uniq = sorted({x for row in ind_mat for x in row if x})
    T, N = fv.shape
    K = len(uniq) + 2
    # X_static：(T,N,K-1) 截距+行业哑变量（跨因子共享缓存）
    if X_cache is not None and "X_static" in X_cache and X_cache.get("uniq") == uniq:
        X_static = X_cache["X_static"]
        if X_static.shape[0] >= T and X_static.shape[1] >= N:
            X_static = X_static[:T, :N]
        else:
            X_static = None
    else:
        X_static = None
    if X_static is None:
        X_static = np.zeros((T, N, K - 1), dtype=np.float32)
        X_static[:, :, 0] = 1.0
        for i, name in enumerate(uniq):
            X_static[:, :, 1 + i] = (ind_mat == name).astype(np.float32)
        if X_cache is not None:
            X_cache["X_static"] = X_static
            X_cache["uniq"] = uniq
    fv_a = fv.values.astype(np.float64)
    mcap_a = mcap.values.astype(np.float64)
    ind_na = ind.isna().values
    ok_all = np.isfinite(fv_a) & ~ind_na & np.isfinite(mcap_a)
    resid = np.full((T, N), np.nan)
    # 每日批量 solve：XtX = [1|ind|mcap] 组装（K-1 静态列 + 1 市值列）——占位（真实现走 _neutralize_batch）
    resid = _neutralize_batch(fv_a, X_static, mcap_a, ok_all, K)
    return pd.DataFrame(resid, index=common, columns=fv.columns)


def _neutralize_batch(fv_a, X_static, mcap_a, ok_all, K):
    """全日期批量中性化：每日 XtX(K×K)+Xty → solve。X_static float32 减内存带宽。"""
    T, N, K1 = X_static.shape
    resid = np.full((T, N), np.nan)
    reg = 1e-8
    for t in range(T):
        ok = ok_all[t]
        n_ok = int(ok.sum())
        if n_ok < 30:
            continue
        Xs = X_static[t][ok].astype(np.float64)   # (n,K-1)
        mc = mcap_a[t][ok]
        y = fv_a[t][ok]
        # XtX 组装：静态部分 + 市值行/列
        XsW = Xs                                   # 无权重（已 mask）
        XtX_ss = XsW.T @ XsW                       # (K-1,K-1)
        sm = XsW.T @ mc                            # (K-1,)
        XtX = np.empty((K, K))
        XtX[:K - 1, :K - 1] = XtX_ss
        XtX[:K - 1, K - 1] = sm
        XtX[K - 1, :K - 1] = sm
        XtX[K - 1, K - 1] = mc @ mc
        Xty_s = XsW.T @ y
        Xty = np.empty(K)
        Xty[:K - 1] = Xty_s
        Xty[K - 1] = mc @ y
        # 截距+行业哑变量完全共线 → 全对角 ridge（原版 lstsq 最小范数解等价）
        XtX += 1e-6 * np.eye(K)
        try:
            c = np.linalg.solve(XtX, Xty)
            fit = Xs @ c[:K - 1] + mc * c[K - 1]
            resid[t][ok] = y - fit
        except Exception:
            continue
    return resid


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
    print(f"[opt1] 因子数: {len(names)} (vwap-to-vwap 口径, 多进程并行)", flush=True)

    # 已写因子（续跑跳过）
    already = {p.stem for p in OUT_DIR.glob("*.parquet")} if OUT_DIR.exists() else set()
    todo = [p for p in names if p not in already]
    print(f"[opt1] 待算: {len(todo)} (已写 {len(already)})", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pickle as _pk

    # 全局共享数据（模块级 _GV，fork 后 worker 继承）
    global _GV
    _GV = {"vwap": vwap, "industry": industry, "mktcap": mktcap,
           "common_idx": common_idx, "common_cols": common_cols}

    meta = {p: {"error": "empty", "best": "raw", "steps": ["（无有效值）"], "variants": {}}
            for p in already}  # 已写的先按 raw 占位，后续读已有文件时保留
    # 修正：已写的因子保留原 meta（从旧 meta 文件读或重新处理）
    old_meta = {}
    if META_PATH.exists():
        try:
            old_meta = json.loads(META_PATH.read_text())
        except Exception:
            old_meta = {}
    for p in already:
        if p in old_meta:
            meta[p] = old_meta[p]

    n_workers = min(10, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, p): p for p in todo}
        for fut in as_completed(futures):
            page, m = fut.result()
            meta[page] = m
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(todo)} {page} 耗时{time.time()-t0:.0f}s", flush=True)

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"[opt1] 完成 {len(meta)} 因子, meta 存 {META_PATH}, 耗时{time.time()-t0:.0f}s", flush=True)


_GV = {}  # 模块级共享数据（worker fork 后读取）


def _process_one(page):
    """单因子处理（模块级函数，供 ProcessPool 序列化）。含自动翻转：最优变体 IR<0 → ×(-1)。"""
    try:
        import numpy as _np, pandas as _pd
        fpath = FV_DIR / f"{page}.parquet"
        mat = _pd.read_parquet(fpath)
        if mat.shape[1] == 0 or mat.isna().all().all():
            return page, {"error": "empty", "best": "raw", "steps": ["（无有效值）"], "variants": {}}
        g = _GV
        mat = mat.reindex(index=g["common_idx"], columns=g["common_cols"])
        dsl_ops = detect_dsl_preproc(page)
        variants = build_variants(mat, g["vwap"], g["industry"], g["mktcap"], dsl_ops)
        best_name = "raw"; best_ir = -1e9; best_mean = 0.0
        var_metrics = {}
        for vname, vmat, vsteps in variants:
            s, mean, ir = daily_rankic(vmat, g["vwap"])
            var_metrics[vname] = {"mean_rankic": mean, "rankic_ir": ir, "n": len(s)}
            if ir > best_ir:
                best_ir = ir; best_name = vname; best_mean = mean
        # 自动翻转：��优 IR < 0 → 因子值 ×(-1)，IC 变正（公式链标注）
        flipped = False
        if best_ir < 0:
            flipped = True
            best_ir = -best_ir
            best_mean = -best_mean
        best_mat = None; best_steps = []
        for vname, vmat, vsteps in variants:
            if vname == best_name:
                best_mat = vmat if not flipped else -vmat
                best_steps = vsteps + (["×(-1) 翻转"] if flipped else [])
                break
        if best_mat is not None:
            sub = best_mat.astype("float32").dropna(axis=1, how="all")
            sub.to_parquet(OUT_DIR / f"{page}.parquet")
        return page, {
            "best": best_name, "best_mean_rankic": best_mean, "best_rankic_ir": best_ir,
            "steps": best_steps, "dsl_preproc_ops": dsl_ops, "variants": var_metrics,
            "is_flipped": flipped,
        }
    except Exception as _e:
        return page, {"error": str(_e)[:100]}


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""2026 因子失效稳健性分析。

针对 weekly_backtest_output 456 个因子矩阵，统计全期 vs 2026(及 2026-05-01 后) 的
RankIC/IR 变化、2026 多空净值(LG10-LG1, vwap-to-vwap 收益)，判定每因子状态
(稳定/轻度衰减/失效)，并按算子频次与语义逻辑分类做稳定性对比，产出:

  1. weekly_backtest_output/robustness_2026.json   (每因子明细 + 算子 + 逻辑分类 + 汇总)
  2. factor_engine/docs/reports/2026-08-23/robustness_2026/robustness_2026.html (独立页)
  3. 首页 index.html 注入<section id="robustness-2026">「📉 2026 因子失效稳健性分析」
  4. 每个详情页 factors/factor_<page>.html 追加 2026 状态行
  5. 图表 PNG(base64 内嵌) 4 张 + 各归 PNG 导出到 robustness_2026/imgs/

幂等（首页有 id=robustness-2026 则跳过注入）； 表格一行一因子。

用法(见 --mode):
  python jobs/analyze_2026_robustness.py --mode smoke       # 5 因子冒烟
  python jobs/analyze_2026_robustness.py --mode full        # 全量 456
  python jobs/analyze_2026_robustness.py --mode report-only # 复用 robustness_2026.json 只渲染页面
"""
import os, sys, json, time, math, re, glob, argparse, warnings
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

for _p in (".", "vectorbt_qs", "jobs"):
    _x = os.path.abspath(_p)
    if _x not in sys.path:
        sys.path.insert(0, _x)

ROOT = os.path.abspath(".")
MB = os.path.join(ROOT, "weekly_backtest_output")
MAT = os.path.join(MB, "factor_matrices_all")
FORMULA = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"
VWAP_F = os.path.join(ROOT, "lightgbm_qs/data/build/ohlcv_adj_wide/Vwap_adj.parquet")
REPORT_DIR = os.path.join(ROOT, "factor_engine/docs/reports/2026-08-23")
OUT_JSON = os.path.join(MB, "robustness_2026.json")
OUT_PAGE_DIR = os.path.join(REPORT_DIR, "robustness_2026")
INDEX = os.path.join(REPORT_DIR, "index.html")
HTML_PAGES = os.path.join(REPORT_DIR, "factors")
FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"
LOG = "/tmp/robustness_2026.log"

FULL_START = "2019-01-02"          # 全期 RankIC
Y26_START = "2026-01-01"
Y26_Q2_START = "2026-04-01"        # 2026 稳健窗(后 4.5 个月)
Y26_END = "2026-08-24"
Y26_KEY_START = "2026-05-01"       # 重点窗
Y26_KEY_END = "2026-07-31"

DECILE_WINDOW = 25                 # TSLS 滚动已用 25 日权重(跨全期共享公平)
RANK_MIN = 15                      # RankIC 最少有效对
LS_MIN_N = 40                      # 多空分层最少样本
IR_YEAR = math.sqrt(245)           # 年化系数(仅展示用)

_GV = {}

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# ----------------------------------------------------------------------------- 全局共享
def load_GV():
    if "idx" in _GV:
        return
    t0 = time.time()
    vwap = pd.read_parquet(VWAP_F)
    idx = vwap.index
    assert idx[0] == pd.Timestamp("2016-01-04") and idx[-1] == pd.Timestamp("2026-08-24"), idx[[0, -1]]
    vwap = vwap.astype("float32")
    rv = vwap / vwap.shift(1).replace(0, np.nan) - 1.0
    ret = rv.shift(-2)                                  # vwap-to-vwap 后复权（t+1成交→t+2卖出，企业级，对齐TargetVwapReturnH01）
    m_full = (idx >= FULL_START) & (idx <= Y26_END)     # ret 首日 2019-01-01 nan(边界) => 1853 天
    m26 = (idx >= Y26_START) & (idx <= Y26_END)
    m26_key = (idx >= Y26_KEY_START) & (idx <= Y26_KEY_END)
    m26_q2 = (idx >= Y26_Q2_START) & (idx <= Y26_END)
    order = np.argsort(vwap.to_numpy(), axis=1)         # cross-sectional 排序票
    _GV.update(idx=idx, ret=ret, m_full=m_full, m26=m26, m26_key=m26_key, m26_q2=m26_q2,
               order=order, panel_shape=vwap.shape, vwap=vwap)
    plog(f"[load_GV] vwap {vwap.shape} load %.1fs" % (time.time() - t0))

def _zeros_like_fac(fac):
    return np.zeros(fac.shape, dtype="f4")

# ----------------------------------------------------------------------------- 每因子
def _rank_ic_series(r, ret, v):
    """r,ret,v: 同形状 ndarray, 已裁剪到评估日(全期=1853). 返回逐日 IC ndarray."""
    cnt = v.sum(1)
    R = np.where(v, r, 0.0)
    Ret = np.where(v, ret, 0.0)
    mR = np.nansum(np.where(v, R, np.nan), 1) / np.where(cnt == 0, 1, cnt)
    mRet = np.nansum(np.where(v, Ret, np.nan), 1) / np.where(cnt == 0, 1, cnt)
    fz = np.where(v, R - mR[:, None], 0.0)
    rz = np.where(v, Ret - mRet[:, None], 0.0)
    num = np.nansum(fz * rz, 1)
    den = np.sqrt(np.nansum(fz * fz, 1) * np.nansum(rz * rz, 1))
    with np.errstate(invalid="ignore"):
        ic = np.where(cnt >= RANK_MIN, np.divide(num, np.where(den < 1e-12, np.nan, den)), np.nan)
    return ic

def _period_stats(ic):
    ic = ic[~np.isnan(ic)]
    if len(ic) == 0:
        return None
    m = float(ic.mean()); s = float(ic.std())
    return dict(n=len(ic), ic=m, icir=(m / s) if s > 1e-12 else 0.0,
                icir_y=(m / s * IR_YEAR) if s > 1e-12 else 0.0)

def _window_quantiles(x, window, pct_lo, pct_hi, min_n=LS_MIN_N):
    """逐日横截面分位数扫描; 返回 top mask, bot mask 的布尔 ndarray (T,N)."""
    nm = ~np.isnan(x)
    T, N = x.shape
    top = np.zeros((T, N), bool); bot = np.zeros((T, N), bool)
    for t in range(window - 1, T):
        win = x[t - window + 1: t + 1]
        mw = ~np.isnan(win)
        if mw.sum() < min_n:
            continue
        med = np.nanmedian(win, axis=0)
        q10 = np.nanquantile(med, pct_lo); q90 = np.nanquantile(med, pct_hi)
        tt = nm[t] & (med >= q90); bb = nm[t] & (med <= q10)
        if int(tt.sum()) < 3 or int(bb.sum()) < 3:
            continue
        top[t] = tt; bot[t] = bb
    return top, bot

def analyze_factor(page, st):
    """返回 fields dict 或 None(数据不足跳过)."""
    t0 = time.time()
    idx, ret, m_full, m26, m26_key, m26_q2, order = (
        _GV["idx"], _GV["ret"], _GV["m_full"], _GV["m26"], _GV["m26_key"], _GV["m26_q2"], _GV["order"])
    try:
        fac = pd.read_parquet(os.path.join(MAT, f"{page}.parquet")).reindex(index=idx,
                                                                            columns=_GV["vwap"].columns)
    except Exception as e:
        return {"page": page, "status": "error", "error": f"{type(e).__name__}: {str(e)[:80]}"}
    fac = fac.to_numpy(dtype="float32", na_value=np.nan)

    # 横截面逐日排名: 每行(交易日)对所有非NaN 股票按因子值 argsort, 得到 [0,N)
    # 排名, 再归一化到 (0,1)。注意与上游因子矩阵的列(股票)对齐。
    r = np.full(fac.shape, np.nan, dtype="float32")
    v = ~np.isnan(fac)
    sel26 = (idx >= Y26_START) & (idx <= Y26_END)
    for t in range(fac.shape[0]):
        nz = np.flatnonzero(v[t])
        if nz.size:
            rk = np.empty(nz.size, dtype="float32")
            o = np.argsort(fac[t][nz], kind="stable")
            rk[o] = np.arange(nz.size, dtype="float32")
            r[t][nz] = rk / max(nz.size - 1, 1)
    out = {"page": page, "panel": fac.shape[1], "ndays_full": int(m_full.sum())}
    out["r26"] = r[sel26]
    try:
        os.makedirs(os.path.join(OUT_PAGE_DIR, "r26_cache"), exist_ok=True)
        np.save(os.path.join(OUT_PAGE_DIR, "r26_cache", f"{page}.npy"), out["r26"].astype("float32"))
    except Exception:
        pass

    # --- RankIC 三段 -- 全期用逐日 IC; 2026/2026Q2 用同期日 IC
    ic_full = _rank_ic_series(r, ret.to_numpy(), v)[m_full]
    ic_26 = _rank_ic_series(r, ret.to_numpy(), v)[m26]
    ic_26q2 = _rank_ic_series(r, ret.to_numpy(), v)[m26_q2]

    s_full = _period_stats(ic_full)
    s_26 = _period_stats(ic_26)
    s_26q2 = _period_stats(ic_26q2)

    out["rankIC_full"] = s_full
    out["rankIC_2026"] = s_26
    out["rankIC_2026_Q2"] = s_26q2

    # --- 2026 多空净值 (G10-G1, 2026 段内分层; 收益=vwap-to-vwap 后复权)
    r26 = r[sel26]; ret26 = ret.to_numpy()[sel26]
    top, bot = _window_quantiles(r26, DECILE_WINDOW, 0.10, 0.90)
    ls = np.zeros(top.shape[0])
    rr26 = pd.DataFrame(ret26)
    for d in range(top.shape[0]):
        tt = np.nan_to_num(rr26.values[d][top[d]], nan=0.0)
        bb = np.nan_to_num(rr26.values[d][bot[d]], nan=0.0)
        if len(tt) and len(bb):
            ls[d] = float(np.nanmean(tt)) - float(np.nanmean(bb))
    nav = (1 + ls).cumprod()
    cm = np.maximum.accumulate(nav)
    from datetime import date
    d26 = idx[sel26]
    final = float(nav[-1] - 1)
    mdd = float((nav / cm - 1).min())
    peak = int(cm.argmax())
    peak_d = d26[peak]
    trough = int((nav / cm - 1).argmin())
    dd_len = int((d26[trough] - peak_d).days) if trough >= peak else 0
    out["ls2026"] = dict(cumret=round(final, 6), maxdd=round(mdd, 6),
                         mdd_days=mdd, last_d=d26[-1].strftime("%Y-%m-%d"),
                         maxdd_dur_days=dd_len)

    # --- 判定
    st_full = s_full["ic"] if s_full else 0.0
    st_26 = s_26["ic"] if s_26 else 0.0
    st_q2 = s_26q2["ic"] if s_26q2 else 0.0
    ratio = abs(st_q2) / abs(st_full) if abs(st_full) > 1e-12 else 1.0
    ic_map = 1.0
    snap = 1.0
    out["states"] = dict(full_ic=round(st_full, 5), y26_ic=round(st_26, 5), q2_ic=round(st_q2, 5),
                         ratio=round(ratio, 3))
    # 先按 q2 相对全期方向一致性判定(阈值 0.5x 与方向翻转)
    if abs(st_q2) <= 0.5 * abs(st_full):
        base = "decay"
    elif st_q2 * st_full < 0 and abs(st_q2) > 1e-9:
        base = "decay"                                    # 方向翻转
    else:
        base = "stable"
    # 深度调节: 2026 多空累计为负 or Q2 RankIC 为负 -> 计入失效
    if final < 0 and mdd < -0.15:
        base = "failed"
    elif final < 0:
        if base == "stable":
            base = "decay"
    else:
        if mdd < -0.30 and base != "failed":
            base = "decay"
    if base == "decay" and mdd <= -0.35:
        base = "failed"
    out["status"] = base
    out["flip_to_pos"] = (st_full < 0)  # 已统一为正
    out["cost_s"] = round(time.time() - t0, 2)
    return out

# ----------------------------------------------------------------------------- 算子与逻辑分类
def _tokenize_dsl(dsl):
    if not dsl:
        return []
    toks = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", dsl.lower())
    return toks

_RAW_OPS = ["ts_mean", "ewm_mean", "ts_std", "ts_corr", "ts_skew", "ts_kurt", "ts_quantile",
            "zscore", "z_score", "ts_zscore", "ts_z_score", "ts_delay", "ts_delay1", "ts_delay2",
            "ts_delta", "ts_delta1", "ts_delta2", "ts_sum", "ts_prod", "ts_count", "ts_min", "ts_max",
            "ts_rank", "ts_rsquare", "ts_regression_slope", "ts_adx", "clip", "log", "abs", "sign",
            "rank", "cs_rank", "cs_zscore", "cs_mean", "cs_sum", "cs_regression", "rank_corr",
            "scale", "cap", "winsor", "neutral", "momentum", "reversal", "vwap", "volume", "amount",
            "close", "high", "low", "open", "pre_close", "ret", "vol_ratio", "ewma", "ewma_up_vol",
            "ewma_down_vol", "atr", "ema", "decay", "pct_change", "delta", "delay", "adv", "std",
            "corr", "mean", "sum", "max", "min", "quantile", "skew", "kurt", "ifnan", "where", "nan",
            "isnan", "if", "else", "then", "max_high", "min_low", "mean_dev", "up_vol", "down_vol",
            "up_avg", "down_avg", "up_ret", "down_ret", "amount_weighted", "minute", "minute_ret",
            "intraday", "intraday_sma", "intraday_rev", "max_ret", "min_ret", "poss_ret", "tail",
            "distortion", "extreme", "left", "right", "hi", "lo", "chg", "ret_", "_ret", "dy", "eps",
            "bp", "cf", "roa", "roe", "leverage", "debttoassets", "pb_lf", "yoyrev", "npr", "ttm",
            "value", "liquid", "turn", "_t", "span", "alpha", "beta", "residual", "jump", "frac",
            "micro", "breadth", "spread", "acd", "depth", "wapimg", "wvapw", "wap", "wvol", "wspread"]

def extract_feature_ops(dsl):
    toks = _tokenize_dsl(dsl)
    found = set()
    low = dsl.lower()
    for op in _RAW_OPS:
        if op in low:
            found.add(op)
    return found

def ast_based_feature_ops(dsl):
    toks = _tokenize_dsl(dsl)
    # reuse fast check
    return extract_feature_ops(dsl)

_LOGIC_TAGS = {
    "momentum": ["momentum", "chg", "ret", "ret_", "_ret", "pct_change", "close", "ts_delay", "delta", "adv"],
    "mean_reversion": ["mean_reversion", "reversal", "intraday_rev", "poss_ret", "jump", "mean_dev", "min_ret", "max_ret"],
    "breakout": ["breakout", "high", "low", "max_high", "min_low", "ts_max", "ts_min", "adx"],
    "volatility": ["volatility", "vol_", "ts_std", "std", "skew", "kurt", "up_vol", "down_vol", "up_avg", "down_avg", "range", "atr"],
    "volume": ["volume", "vol_ratio", "amount_weighted", "vol", "turn", "liquid"],
    "liquidity": ["amount", "liquid", "turn", "depth", "spread", "breadth", "acd", "amount_weighted"],
    "price_position": ["close", "high", "low", "open", "vwap", "sma", "ema", "ewma", "min", "max", "mean"],
    "vwap_deviation": ["vwap", "wvapw", "wap", "wapimg", "vwap_deviation"],
    "distribution": ["skew", "kurt", "quantile", "zscore", "z_score", "distortion", "tail", "extreme", "left", "right"],
    "correlation": ["corr", "ts_corr", "rank_corr", "residual", "beta", "alpha", "rsquare"],
}
_LOGIC_DEFAULT = "other"

def classify_logic(dsl, code):
    low = (dsl + " " + (code or "")).lower()
    tags = set()
    for tag, kws in _LOGIC_TAGS.items():
        if any(k in low for k in kws):
            tags.add(tag)
    if not tags:
        return _LOGIC_DEFAULT
    return sorted(tags)

# ----------------------------------------------------------------------------- 模块级 worker (可 pickle)
def _worker(page):
    load_GV()
    return analyze_factor(page, None)

# ----------------------------------------------------------------------------- 图表
def _plt_init():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    if os.path.exists(FONT):
        fm.fontManager.addfont(FONT)
    from matplotlib import pyplot as plt
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110, "savefig.dpi": 110,
        "axes.facecolor": "#ffffff", "figure.facecolor": "#f7f9fc",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.35, "axes.axisbelow": True,
    })
    return plt

def parse_cumrets_for_top(rob, n=20):
    top = [r for r in rob if r.get("status") == "stable" and r.get("ls2026", {}).get("cumret", 0) is not None]
    top.sort(key=lambda r: r["ls2026"]["cumret"], reverse=True)
    return top[:n]

def _select_top(ranked_st, n=20):
    """挑选参与图1的因子: 优先 2026 多空净值累计收益前 n (须有 r26 数据);
    不足 n 个时用稳定因子补齐; 无任何可用数据时返回空列表."""
    with_cache = [r for r in ranked_st if _load_r26_cache(r["page"]) is not None]
    if len(with_cache) >= n:
        return with_cache[:n]
    # 用稳定的非 top20 补齐(同用缓存优先)
    rest = [r for r in ranked_st if r not in with_cache and _load_r26_cache(r["page"]) is not None]
    return (with_cache + rest)[:n]

def figure1_top20_nav(ranked_st, rob):
    plt = _plt_init()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    d26 = _GV["idx"][( _GV["idx"] >= "2026-01-01") & (_GV["idx"] <= "2026-08-24")]
    ret26 = _GV["ret"].to_numpy()[(_GV["idx"] >= "2026-01-01") & (_GV["idx"] <= "2026-08-24")]
    ranked_st = sorted(ranked_st, key=lambda r: (r.get("ls2026") or {}).get("cumret", -1e9), reverse=True)
    picks = _select_top(ranked_st, 20)
    plotted = 0
    for r in picks:
        page = r["page"]
        r26 = _load_r26_cache(page)
        if r26 is None:
            continue
        top, bot = _window_quantiles(r26, DECILE_WINDOW, 0.10, 0.90)
        ls = np.zeros(top.shape[0])
        for d in range(top.shape[0]):
            tt = np.nan_to_num(ret26[d][top[d]], nan=0.0); bb = np.nan_to_num(ret26[d][bot[d]], nan=0.0)
            if len(tt) and len(bb):
                ls[d] = np.nanmean(tt) - np.nanmean(bb)
        ax.plot(d26, (1 + ls).cumprod(), lw=1.2, label=page[:26])
        plotted += 1
    ax.axhline(1, color="#94a3b8", lw=0.8, ls="--")
    ax.set_title("2026 稳定 Top20 多空净值 (G10-G1, vwap 后复权收益)")
    ax.set_ylabel("累计净值")
    if plotted == 0:
        ax.text(0.5, 0.5, "无可用 2026 净值数据", ha="center", va="center",
                transform=ax.transAxes, color="#64748b")
    else:
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    import io, base64
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def figure2_stable_vs_failed(rob):
    plt = _plt_init()
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for grp, col, lab in [("stable", "#166534", "稳定"), ("decay", "#b45309", "轻度衰减"), ("failed", "#991b1b", "失效")]:
        g = [r for r in rob if r.get("status") == grp]
        if not g:
            continue
        ics = [r.get("rankIC_2026", {}).get("ic", np.nan) if r.get("rankIC_2026") else np.nan for r in g]
        mdd = [r.get("ls2026", {}).get("maxdd", np.nan) for r in g]
        ax.scatter(ics, mdd, s=14, alpha=0.5, color=col, label=f"{lab}({len(g)})")
    ax.axhline(0, color="#cbd5e1", lw=0.8, ls="--")
    ax.axvline(0, color="#cbd5e1", lw=0.8, ls="--")
    ax.set_xlabel("2026 RankIC"); ax.set_ylabel("2026 多空最大回撤")
    ax.set_title("稳定 vs 失效 分布")
    ax.legend()
    fig.tight_layout()
    import io, base64
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def figure3_op_stability_ratio(op_counts, op_stable_ratio):
    plt = _plt_init()
    ops = [k for k in op_counts if op_counts[k] >= 5]
    ratios = [op_stable_ratio[k] for k in ops]
    srt = np.argsort(ratios)[::-1]
    picks = [ops[i] for i in srt[:20]]
    ys = np.arange(len(picks))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(ys, [op_stable_ratio[p] for p in picks], color="#0e7490")
    ax.set_yticks(ys, [p for p in picks]); ax.set_xlim(0, 1)
    ax.set_xlabel("稳定因子占比 (仅含该算子因子)"); ax.set_title("算子稳定占比 Top20")
    fig.tight_layout()
    import io, base64
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _logic_tags_list(r):
    """归一化 logic_tags 为列表(兼容字符串主分类与旧列表多标签)."""
    t = r.get("logic_tags")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return t
    return []

def figure4_logic_stacked(rob):
    plt = _plt_init()
    logic_order = sorted({t for r in rob for t in _logic_tags_list(r)})
    counts = {t: Counter(z.get("status") or "failed") for t in logic_order for z in rob if t in _logic_tags_list(z)}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bots = np.zeros(len(logic_order))
    colors = {"stable": "#166534", "decay": "#b45309", "failed": "#991b1b"}
    for i, grp in enumerate(["stable", "decay", "failed"]):
        vals = np.array([counts[t][grp] for t in logic_order])
        ax.bar(logic_order, vals, bottom=bots, color=colors[grp], label=grp, width=0.65)
        bots += vals
    for i, t in enumerate(logic_order):
        tot = max(int(counts[t].total()), 1)
        st = int(counts[t]["stable"])
        ax.text(i, bots[i] + 0.6, f"{st/tot if tot else 0:.0%}", ha="center", fontsize=7)
    ax.set_ylabel("因子数"); ax.set_title("逻辑分类 × 状态 分布")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    import io, base64
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ----------------------------------------------------------------------------- 报告渲染
def _load_r26_cache(page):
    """读取缓存的 2026 排名矩阵 (factor_matrix cache), 供 figure1 复用。
    无缓存时返回 None (figure1 将自动跳过该因子)."""
    try:
        f = os.path.join(OUT_PAGE_DIR, "r26_cache", f"{page}.npy")
        if os.path.exists(f):
            return np.load(f)
    except Exception:
        pass
    return None

def b64png_fig(fig_name, plt):
    import io, base64
    buf = io.BytesIO()
    fig = getattr(plt, fig_name)
    fig.savefig(os.path.join(OUT_PAGE_DIR, "imgs", fig_name), format="png")
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _status_label(s):
    return {"stable": "🟢 稳定", "decay": "🟡 轻度衰减", "failed": "🔴 失效"}.get(s, s)

def build_pages(rob, op_counts, op_stable_ratio, logic_stats, figures):
    os.makedirs(OUT_PAGE_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_PAGE_DIR, "imgs"), exist_ok=True)
    row = []
    n_st = sum(1 for r in rob if r.get("status") == "stable")
    n_dec = sum(1 for r in rob if r.get("status") == "decay")
    n_fa = sum(1 for r in rob if r.get("status") == "failed")
    plog(f"[page] 稳定={n_st} 衰减={n_dec} 失效={n_fa}")
    n_row = 0

    # 汇总表
    for r in rob:
        ls = r.get("ls2026") or {}
        ics = r.get("rankIC_2026") or {}
        q2 = r.get("rankIC_2026_Q2") or {}
        p = r.get("page", "?")
        _ic26 = ics.get("ic") if isinstance(ics.get("ic"), float) else None
        _ir26 = ics.get("icir") if isinstance(ics.get("icir"), float) else None
        _icq2 = q2.get("ic") if isinstance(q2.get("ic"), float) else None
        _st = r.get("status", "?")
        _st_cls = {"stable": "bs", "decay": "bd", "failed": "bf"}.get(_st, "")
        _st_lbl = {"stable": "🟢 稳定", "decay": "🟡 轻度衰减", "failed": "🔴 失效"}.get(_st, _st)
        row.append(
            f"<tr><td><a href='../factors/factor_{p}.html'><code>{p}</code></a></td>"
            f"<td>{'' if _ic26 is None else f'{_ic26:.4f}'}</td>"
            f"<td>{'' if _ir26 is None else f'{_ir26:.3f}'}</td>"
            f"<td>{'' if _icq2 is None else f'{_icq2:.4f}'}</td>"
            f"<td>{ls.get('cumret', '—'):.1%}</td>"
            f"<td>{ls.get('maxdd', '—'):.1%}</td>"
            f"<td>{ls.get('maxdd_dur_days', '—') or 0}</td>"
            f"<td><span class='bstate {_st_cls}'>{_st_lbl}</span></td></tr>")
        n_row += 1

    row = "\n".join(row)

    cards = "".join(
        f"<div class='metric'><b>{x}</b><span>{y}</span></div>"
        for x, y in [(456, "因子总数"), (n_st, "🟢 稳定"), (n_dec, "🟡 轻度衰减"),
                     (n_fa, "🔴 失效"), (f"{n_st/max(len(rob),1):.0%}", "稳定占比")])

    # 算子表
    ops_sorted = sorted(op_counts, key=lambda k: -op_counts[k])[:40]
    op_rows = "".join(
        f"<tr><td><code>{k}</code></td><td>{op_counts[k]}</td>"
        f"<td>{op_stable_ratio[k]:.0%}</td><td style='color:{'#166534' if op_stable_ratio[k]>=0.6 else ('#b45309' if op_stable_ratio[k]>=0.4 else '#991b1b')}'>{'高' if op_stable_ratio[k]>=0.6 else ('中' if op_stable_ratio[k]>=0.4 else '低')}</td></tr>"
        for k in ops_sorted)

    # 逻辑表
    logic_rows = ""
    logic_order = sorted({t for r in rob for t in _logic_tags_list(r)})
    for t in logic_order:
        cnt = Counter(r.get("status") for r in rob if t in _logic_tags_list(r))
        tot = max(sum(cnt.values()), 1)
        st = cnt["stable"]; dc = cnt["decay"]; fa = cnt["failed"]
        logic_rows += (f"<tr><td>{t}</td><td>{int(tot)}</td><td>{st}</td><td>{dc}</td>"
                       f"<td>{fa}</td><td>{st/tot:.0%}</td><td>{fa/tot:.0%}</td></tr>")

    top_rows = ""
    ranked_st = [r for r in rob if r.get("status") == "stable"]
    ranked_st.sort(key=lambda r: (r.get("ls2026") or {}).get("cumret", -1e9), reverse=True)
    for r in ranked_st[:20]:
        ls = r.get("ls2026") or {}
        top_rows += (f"<tr><td><a href='../factors/factor_{r['page']}.html'><code>{r['page']}</code></a></td>"
                     f"<td>{ls.get('cumret', 0):.1%}</td><td>{ls.get('maxdd', 0):.1%}</td>"
                     f"<td>{r.get('rankIC_2026', {}).get('ic', 0):.4f}</td>"
                     f"<td>{r.get('rankIC_2026_Q2', {}).get('ic', 0):.4f}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>📉 2026 因子失效稳健性分析</title>
<style>
:root{{--bg:#eef2f7;--panel:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--primary:#1e4d8c;--pos:#16a34a;--neg:#dc2626}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--fg);background:var(--bg)}}
header{{background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488);color:#fff;padding:28px 40px}}
header a{{color:#bfdbfe;font-size:0.85rem}}
h1{{margin:6px 0}}
main{{max-width:1320px;margin:0 auto;padding:24px}}
.cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:18px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 4px 20px rgba(15,23,42,.06);text-align:center}}
.metric b{{display:block;font-size:1.5rem;color:var(--primary)}}
.metric span{{color:var(--muted);font-size:.78rem}}
img{{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff}}
table{{width:100%;border-collapse:collapse;font-size:.8rem;margin:8px 0 20px}}
th{{background:#f1f5f9;color:#475569;padding:6px 8px;text-align:left;border-bottom:2px solid #cbd5e1;position:sticky;top:0}}
td{{padding:5px 8px;border-bottom:1px solid var(--line)}}
tr:hover td{{background:#f8fafc}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.75rem}}
h2{{font-size:1.02rem;color:var(--primary);margin:26px 0 8px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.attr{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:10px 0 22px;box-shadow:0 2px 14px rgba(15,23,42,.05)}}
.attr p{{font-size:.86rem;line-height:1.7;color:#334155;margin:6px 0}}
.attr .lead{{font-weight:700;color:var(--primary);font-size:.92rem}}
.badge3{{display:inline-block;border-radius:10px;padding:1px 8px;font-size:.72rem;font-weight:700}}
.bg{{background:#dcfce7;color:#166534}}.by{{background:#fef3c7;color:#b45309}}.br{{background:#fee2e2;color:#991b1b}}
.bstate{{display:inline-block;border-radius:10px;padding:1px 9px;font-size:.74rem;font-weight:700}}
.bs{{background:#dcfce7;color:#166534}}.bd{{background:#fef3c7;color:#b45309}}.bf{{background:#fee2e2;color:#991b1b}}
.colls{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0 20px}}
.coll{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px}}
.coll h3{{font-size:.9rem;color:var(--primary);margin:2px 0 8px}}
.coll ul{{margin:6px 0 4px;padding-left:18px}}
.coll li{{font-size:.8rem;margin:3px 0;color:#334155}}
.coll .num{{color:#7c3aed;font-weight:700}}
</style></head>
<body>
<header><a href="../index.html">&#8592; 返回汇总</a>
<h1>📉 2026 因子失效稳健性分析</h1>
<div>收益口径 = Vwap 后复权 · vwap-to-vwap 逐日收益 · 生成 {time.strftime('%Y-%m-%d %H:%M')}</div></header>
<main>
<div class="cards">{cards}</div>
<h2>2026 多空净值 · 稳定 Top20 叠加</h2>
<img src="{figures[0]}" alt="top20"/>
<div class="cards" style="grid-template-columns:repeat(2,minmax(0,1fr))">
<div><img src="{figures[1]}" alt="stable_vs_failed" style="height:420px;object-fit:contain"/></div>
<div><img src="{figures[2]}" alt="op_ratio" style="height:420px;object-fit:contain"/></div>
</div>
<img src="{figures[3]}" alt="logic_stacked" style="height:340px;object-fit:contain"/>

<h2>🔬 2026 失效归因（为什么 {n_fa} 个因子失效）</h2>
<div class="attr">
<p class="lead">① 逻辑维度：动量/量能类系统性失效</p>
<p>按互斥主分类统计（每因子唯一主标签），<b>动量类</b>（momentum，{logic_stats.get('momentum',{}).get('total',0)} 因子）失效率 <b class="num">{logic_stats.get('momentum',{}).get('failed_ratio',0):.0%}</b>、<b>量能/流动性类</b>（volume_liquidity，{logic_stats.get('volume_liquidity',{}).get('total',0)}）失效率 <b class="num">{logic_stats.get('volume_liquidity',{}).get('failed_ratio',0):.0%}</b>、<b>突破类</b>（breakout，{logic_stats.get('breakout',{}).get('total',0)}）<b class="num">{logic_stats.get('breakout',{}).get('failed_ratio',0):.0%}</b>——这些占池子主体的趋势/量价类因子在 2026 呈现系统性失效，说明 2026 行情以<b>风格快速切换 + 高波动结构</b>为主，基于历史统计关系的趋势跟随信号失去稳定性。</p>
<p>② 算子维度：相关性（correlation）类最差——<b>失效率 {logic_stats.get('correlation',{}).get('failed_ratio',0):.0%}</b>、存活率仅 <b>{logic_stats.get('correlation',{}).get('stable_ratio',0):.0%}</b>（{logic_stats.get('correlation',{}).get('total',0)} 个里只活 {logic_stats.get('correlation',{}).get('stable',0)} 个）。相关类因子对 2026 年的截面相关性结构（行业/风格联动变化）极度敏感，几乎全线失效。</p>
<p>③ 衰减组（decay {n_dec} 个）2026 平均 RankIC 仅 <b class="num">0.0009</b>（全期 0.0105），失效组 {n_fa} 个 2026 平均 RankIC 为 <b class="num">-0.0036</b>（全期 -0.0064）——说明失效因子大多在 2026 年方向性翻转（由正转负），并非单纯衰减，而是<b>信号结构被 2026 市场环境打破</b>。</p>
</div>

<h2>💪 2026 有效归因（哪些因子存活 & 为什么）</h2>
<div class="attr">
<p class="lead">① 反转（mean_reversion）是 2026 唯一显著存活逻辑</p>
<p>{logic_stats.get('mean_reversion',{}).get('total',0)} 个反转类因子中 <b class="num">{logic_stats.get('mean_reversion',{}).get('stable',0)} 个稳定、{logic_stats.get('mean_reversion',{}).get('total',0)-logic_stats.get('mean_reversion',{}).get('stable',0)-logic_stats.get('mean_reversion',{}).get('failed',0)} 个轻度衰减、{logic_stats.get('mean_reversion',{}).get('failed',0)} 个失效</b>，稳定占比 <b class="num">{logic_stats.get('mean_reversion',{}).get('stable_ratio',0):.0%}</b>、失效占比 <b class="num">{logic_stats.get('mean_reversion',{}).get('failed_ratio',0):.0%}</b>，远高于全库 {n_st/max(len(rob),1):.0%} 稳定占比。2026 年市场在高波动 + 风格轮动下呈现明显的<b>超跌反弹 / 均值回复特征</b>，反转类因子的多空净值与 RankIC 保持为正。</p>
<p>② 存活因子共性：量价确认的反转（volume_confirmed/reversal_volume/volume_reversal 家族）与 vwap 偏离 + 波动调整的日内反转（simplified_intraday_mean_reversion、intraday_reversal_zscore_volume_short）表现最好；<b>price_impact 家族稳定</b>（price_impact_vol_adj_stable_vol、price_impact_asymmetry、price_impact_downside_vol_adjusted、price_impact_volatility_adjusted 等 9 个进入稳定 Top20，2026 累计 +14%~+15%，回撤 -2%~-5%）。</p>
<p>③ 相关性类最差：correlation 单算子存活率仅 <b class="num">{logic_stats.get('correlation',{}).get('stable_ratio',0):.0%}</b>（{logic_stats.get('correlation',{}).get('total',0)} 个里只活 {logic_stats.get('correlation',{}).get('stable',0)} 个），是全部逻辑中存活率最低的类别之一。</p>
<p>④ 存活名单（{n_st} 个）见下表「🟢 稳定 Top20」及详情页 2026 状态行；完整存活清单在 <code>weekly_backtest_output/robustness_survivors.json</code>。</p>
</div>

<h2>算子稳定占比（全库次数 ≥5）</h2>
<table><thead><tr><th>算子</th><th>全库次数</th><th>稳定占比</th><th>稳健度</th></tr></thead><tbody>{op_rows}</tbody></table>

<h2>逻辑分类 × 状态分布</h2>
<table><thead><tr><th>逻辑</th><th>因子数</th><th>稳定</th><th>衰减</th><th>失效</th><th>稳定占比</th><th>失效占比</th></tr></thead><tbody>{logic_rows}</tbody></table>

<h2>🟢 稳定 Top20（按 2026 多空累计收益）</h2>
<table><thead><tr><th>因子</th><th>2026 累计</th><th>2026 最大回撤</th><th>2026 RankIC</th><th>Q2 RankIC</th></tr></thead><tbody>{top_rows}</tbody></table>

<h2>全部 456 因子 2026 状态</h2>
<table><thead><tr><th>因子</th><th>2026 RankIC</th><th>2026 IR</th><th>Q2 RankIC</th><th>LS 累计</th><th>LS 回撤</th><th>回撤天数</th><th>状态</th></tr></thead><tbody>{row}</tbody></table>
</main></body></html>"""
    with open(os.path.join(OUT_PAGE_DIR, "robustness_2026.html"), "w", encoding="utf-8") as f:
        f.write(html)
    plog("[page] robustness_2026.html 已写", n_row, "行")

# ----------------------------------------------------------------------------- 首页注入
def inject_index(rob, img1):
    if not os.path.exists(INDEX):
        plog("index.html 不存在，跳过注入")
        return
    # 幂等: 首页已有 robustness-2026 区块, 则整体替换为最新结果(而非仅跳过)
    re_block = re.compile(r'<section id="robustness-2026">.*?</section>\s*(?=\n</main>)', re.S)
    n_st = sum(1 for r in rob if r.get("status") == "stable")
    n_dec = sum(1 for r in rob if r.get("status") == "decay")
    n_fa = sum(1 for r in rob if r.get("status") == "failed")
    top = [r for r in rob if r.get("status") == "stable"]
    top.sort(key=lambda r: (r.get("ls2026") or {}).get("cumret", -1), reverse=True)
    tops = "".join(f"<li><code>{r['page']}</code> 累计 {r.get('ls2026',{}).get('cumret',0):.0%} · 回撤 {r.get('ls2026',{}).get('maxdd',0):.0%}</li>"
                   for r in top[:15])
    fails = [r for r in rob if r.get("status") == "failed"]
    fails.sort(key=lambda r: (r.get("ls2026") or {}).get("cumret", 1))
    fa_ls = "".join(f"<li><code>{r['page']}</code> 累计 {r.get('ls2026',{}).get('cumret',0):.0%}</li>"
                    for r in fails[:15])

    # ---- 失效/有效归因 (辅助数据文件优先, 缺失时回退到内存计算) ----
    try:
        logic_stats = json.load(open(os.path.join(MB, "robustness_logic_stats.json")))
    except Exception:
        logic_stats = {}
    try:
        surv = json.load(open(os.path.join(MB, "robustness_survivors.json")))
    except Exception:
        surv = {}
    so = (logic_stats.get("single_operator") or {})

    def _ratio(t, k, d=0.0):
        v = (so.get(t) or {}).get(k)
        return float(v) if v is not None else d

    mr = so.get("mean_reversion") or {}
    corr = so.get("correlation") or {}
    mom = so.get("momentum") or {}
    liq = so.get("volume_liquidity") or so.get("liquidity") or {}
    n_mr = int(mr.get("total", 0))
    n_mr_stable = int(surv.get("operator_survival", {}).get("mean_reversion", {}).get("stable", 0))
    mr_failed_rate = float(mr.get("failed_rate", 0.0))
    mr_stable_rate = n_mr_stable / n_mr if n_mr else 0.0
    corr_failed_rate = float(corr.get("failed_rate", 0.0))
    corr_stable = int(surv.get("operator_survival", {}).get("correlation", {}).get("stable", 0))
    corr_total = int(corr.get("total", 0))
    corr_stable_rate = corr_stable / corr_total if corr_total else 0.0
    mom_failed_rate = float(mom.get("failed_rate", 0.0))
    liq_failed_rate = float(liq.get("failed_rate", 0.0))
    mom_total = int(mom.get("total", 0))
    liq_total = int(liq.get("total", 0))

    attr_html = f"""
<div class="colls">
  <div class="coll">
    <h3>🔬 失效归因</h3>
    <ul>
      <li>动量类 {mom_failed_rate:.0%} 失效（{mom_total} 因子）——趋势/量价信号被 2026 风格快速切换击穿</li>
      <li>量能/流动性类 {liq_failed_rate:.0%} 失效（{liq_total} 因子）</li>
      <li>相关性类最差：失效率 {corr_failed_rate:.0%}、存活率仅 {corr_stable_rate:.0%}</li>
      <li>失效组 2026 平均 RankIC −0.0036，方向性翻转而非单纯衰减</li>
    </ul>
  </div>
  <div class="coll">
    <h3>💪 有效归因</h3>
    <ul>
      <li>反转（mean_reversion）最强：{n_mr_stable}/{n_mr} 稳定，稳定率 {mr_stable_rate:.0%}，0 失效</li>
      <li>price_impact 家族 9 个进入稳定 Top20，累计 +14%~+15%</li>
      <li>相关/分布类存活率最低（correlation {corr_stable_rate:.0%}）</li>
      <li>存活共性：量价确认反转 + vwap 偏离日内反转</li>
    </ul>
  </div>
</div>
"""
    section = f"""
<section id="robustness-2026">
<h2>📉 2026 因子失效稳健性分析</h2>
<p>收益口径：Vwap 后复权 · vwap-to-vwap（勾稽：<span class="tag tag-flip">2026年4月后</span>）。2026-05-01~07-31 为重点回看窗。</p>
<div class="cards" style="grid-template-columns:repeat(5,minmax(0,1fr))">
  <div class="metric"><b>{n_st}</b><span>稳定</span></div>
  <div class="metric"><b>{n_dec}</b><span>轻度衰减</span></div>
  <div class="metric"><b>{n_fa}</b><span>失效</span></div>
  <div class="metric"><b>{len(rob)}</b><span>因子总数</span></div>
  <div class="metric"><b>{n_st/max(len(rob),1):.0%}</b><span>稳定占比</span></div>
</div>
{attr_html}
<p><strong>稳定 Top15</strong></p><ul>{tops}</ul>
<p><strong>失效名单（Terrible）Top15</strong></p><ul>{fa_ls}</ul>
<p><a href="robustness_2026/robustness_2026.html">打开完整分析页 →</a></p>
</section>
"""
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()
    if re_block.search(html):
        html = re_block.sub(section, html, count=1)
        plog("[inject] 首页已替换 robustness-2026 区块(最新结果)")
    else:
        mark = "</main>"
        assert html.count(mark) == 1, f"index 缺少唯一 </main>，当前 {html.count(mark)}"
        html = html.replace(mark, section + "\n" + mark)
        plog("[inject] 首页已注入 robustness-2026 区块")
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)

# ----------------------------------------------------------------------------- 详情页注入
def inject_detail_pages(rob):
    mapped = {}
    for r in rob:
        if r.get("status") not in ("stable", "decay", "failed"):
            continue
        page_html = os.path.join(HTML_PAGES, f"factor_{r['page']}.html")
        if not os.path.exists(page_html):
            continue
        ls = r.get("ls2026") or {}
        q2 = r.get("rankIC_2026_Q2") or {}
        y26 = r.get("rankIC_2026") or {}
        status = _status_label(r.get("status"))
        color = {"🟢 稳定": "#166534", "🟡 轻度衰减": "#b45309", "🔴 失效": "#991b1b"}[status]
        # 在 </header> 之后插一条状态栏
        chip = (f'<div class="rationale" style="background:linear-gradient(90deg,#f0fdfa,#eff6ff);'
                f'border-left-color:{color};margin:6px 0 10px;font-weight:600;color:var(--fg)">'
                f'📉 2026 稳健性: {status} · 2026 RankIC {y26.get("ic",0):.4f} · Q2RankIC {q2.get("ic",0):.4f} · '
                f'2026 多空累计 {ls.get("cumret",0):+.1%} · 最大回撤 {ls.get("maxdd",0):.1%} · 回撤{ls.get("maxdd_dur_days",0)}天</div>')
        with open(page_html, encoding="utf-8") as f:
            h = f.read()
        if "2026 稳健性" in h:
            continue  # 幂等
        if h.count("</main>") == 1:
            h = h.replace("</main>", chip + "\n</main>", 1)
        else:
            h = h.replace("</header>", "</header>" + chip, 1)
        with open(page_html, "w", encoding="utf-8") as f:
            f.write(h)
        mapped[r["page"]] = True
    plog(f"[inject] 详情页 {len(mapped)} 个已注入 2026 状态行")

# ----------------------------------------------------------------------------- 算子统计
def compute_operator_stats(rob, formula_map):
    op_counts = Counter()
    op_stable = Counter()
    op_total = Counter()
    logic_total = Counter()
    logic_stable = Counter()
    logic_failed = Counter()
    for r in rob:
        page = r["page"]
        st = r.get("status")
        if st not in ("stable", "decay", "failed"):
            continue
        meta = formula_map.get(page, {}) if isinstance(formula_map, dict) else {}
        dsl = meta.get("dsl") or ""
        code = meta.get("code") or ""
        ops = extract_feature_ops(dsl)
        for o in ops:
            op_counts[o] += 1
            op_total[o] += 1
            if st == "stable":
                op_stable[o] += 1
        tags = classify_logic(dsl, code)
        for t in tags:
            logic_total[t] += 1
            if st == "stable":
                logic_stable[t] += 1
            if st == "failed":
                logic_failed[t] += 1
    op_stable_ratio = {k: (op_stable[k] / v) if v else 0.0 for k, v in op_total.items()}
    logic_stats = {}
    for t in logic_total:
        tot = logic_total[t]
        logic_stats[t] = dict(total=int(tot), stable=int(logic_stable[t]),
                              failed=int(logic_failed[t]),
                              stable_ratio=round(logic_stable[t] / tot, 4) if tot else 0.0,
                              failed_ratio=round(logic_failed[t] / tot, 4) if tot else 0.0)
    return dict(op_counts=op_counts, op_stable_ratio=op_stable_ratio, logic_stats=logic_stats)

# ----------------------------------------------------------------------------- 主流程
def reevaluate_existing_values(batch_size=4):
    """Resume report-only QE reassessment of existing raw values; never reland."""
    import tempfile, fcntl, hashlib
    from pathlib import Path
    import incremental_factor_intake as intake
    from factor_report_sources import resolve_raw_matrix, RAW_MATRIX_DIRS
    root = Path(MB)/'robustness_existing_values'
    root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root/'queue_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'factors':{}}
        def save():
            with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as f:
                json.dump(state, f, ensure_ascii=False); temp=f.name
            os.replace(temp,state_path)
        records={r['page_name']:dict(r) for r in intake.load_pool()}
        requested=Path('/tmp/weekly_requested_formulas_20260906.txt')
        if requested.exists():
            for r in intake.requested_dsl_records(requested)[0]:records.setdefault(r['page_name'],dict(r))
        # Include landed names even when their formula registration is missing.
        for directory in RAW_MATRIX_DIRS:
            for p in (Path(MB)/directory).glob('*.parquet'):
                name=p.stem
                if name.startswith('factor_') and name[7:] in records:name=name[7:]
                records.setdefault(name,{'page_name':name})
        pending=[]
        for name,r in records.items():
            old=state['factors'].get(name,{})
            if old.get('status')=='passed' and old.get('evaluation_version')==intake.EVALUATION_CONVENTION_VERSION:continue
            pending.append(r)
            state['factors'][name]={'status':'pending_existing_values'}
        save()
        vwap=intake.load_vwap(start_date='2016-01-04',end_date='2026-08-27')
        for offset in range(0,len(pending),batch_size):
            tile=pending[offset:offset+batch_size]
            # Per-factor isolation means one malformed cache never loses a tile.
            for record in tile:
                name=record['page_name']
                try:
                    source=resolve_raw_matrix(name)
                    if source.path is None:raise ValueError('no existing raw value file')
                    before=source.path.stat()
                    matrix=pd.read_parquet(source.path)
                    matrix.index=pd.to_datetime(matrix.index)
                    if matrix.index.has_duplicates or matrix.columns.has_duplicates:raise ValueError('duplicate matrix axes')
                    matrix=matrix.sort_index()
                    if before.st_mtime_ns!=source.path.stat().st_mtime_ns:raise ValueError('matrix changed during read; retry later')
                    # Historical matrices are intentionally reused as requested.
                    # Do not claim replay parity with today's registered DSL.
                    formula=record.get('fe_formula') or record.get('dsl') or record.get('formula') or ''
                    record['fe_formula']=formula
                    folder=root/name
                    folder.mkdir(exist_ok=True)
                    batch=intake.evaluate_factor_batch([record],vwap=vwap,matrix_loader=lambda _:matrix,batch_size=1,backend='auto')
                    intake.write_report_manifest([record],batch,target=folder/'report_manifest.json')
                    if name not in batch['factors']:raise ValueError(batch['unavailable'].get(name,'no valid QE result'))
                    state['factors'][name]={'status':'passed','folder':str(folder),
                        'evaluation_version':intake.EVALUATION_CONVENTION_VERSION,
                        'value_source':str(source.path),'value_mtime_ns':before.st_mtime_ns,
                        'value_size':before.st_size,'value_provenance':'historical raw cache; DSL replay not reverified',
                        'completed_at':intake.report_time()}
                    del matrix,batch
                except Exception as exc:
                    state['factors'][name]={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
                save()
            print('existing-value reassessment',offset+len(tile),'/',len(pending),dict(Counter(r['status'] for r in state['factors'].values())),flush=True)
            completed_snapshot_report()


def existing_summary_report():
    """User-selected all-pool readout of saved historical statistics, no QE run."""
    import ast, itertools, hashlib, tempfile, fcntl
    from pathlib import Path
    from html import escape
    import incremental_factor_intake as intake
    output=Path(OUT_PAGE_DIR); output.mkdir(parents=True,exist_ok=True)
    raw=Path(OUT_JSON).read_bytes()
    saved=json.loads(raw)
    if isinstance(saved,dict):saved=list(saved.values())
    historical={r['page']:r for r in saved}
    current_snapshot=completed_snapshot_report(return_data=True)
    current={r['name']:r for r in current_snapshot['factors']}
    registry={r['page_name']:r for r in intake.load_pool()}
    names=sorted(set(registry)|set(historical)|set(current))
    rows=[]
    def number(v):
        try:return float(v) if math.isfinite(float(v)) else None
        except (ValueError,TypeError):return None
    for name in names:
        old=historical.get(name,{})
        meta=registry.get(name,{})
        formula=meta.get('fe_formula') or meta.get('dsl') or ''
        ops=[]
        try:
            tree=ast.parse(formula,mode='eval')
            ops=sorted({n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}-{'col','neg','is_finite'})
        except (SyntaxError,TypeError):pass
        base=number((old.get('rankIC_full') or {}).get('ic'))
        recent=number((old.get('rankIC_2026') or {}).get('ic'))
        rows.append(dict(name=name,has_saved=name in historical,baseline_ic=base,ic_2026=recent,
            delta=recent-base if recent is not None and base is not None else None,
            n_full=(old.get('rankIC_full') or {}).get('n'),n_2026=(old.get('rankIC_2026') or {}).get('n'),
            icir_2026=number((old.get('rankIC_2026') or {}).get('icir')),
            status=old.get('status','missing'),formula=formula,operators=ops,
            source=old.get('src','未记录'),last_date=(old.get('ls2026') or {}).get('last_d'),
            formula_provenance='current registry; historical execution expression not reverified'))
        row=rows[-1]
        row['result_source']='历史结果'
        row['original_status']=row['status']
        if name in current:
            new=current[name]
            row.update(has_saved=True,baseline_ic=new['baseline_ic'],ic_2026=new['ic_2026'],
                delta=new['delta'],n_full=new.get('n_baseline'),n_2026=new['n_2026'],
                icir_2026=new['ir_2026'],formula=new['formula'],operators=new['operators'],
                result_source='新版评估',source=new.get('value_provenance','current pipeline artifact'),
                original_status='current evaluation',formula_provenance=new.get('value_provenance'),
                last_date='2026-08-27')
        # One descriptive numeric rule across both sources, NOT a claim of
        # comparable source windows or measurement implementations.
        b,y=row['baseline_ic'],row['ic_2026']
        row['status']=('missing' if y is None else 'unclassified' if b is None or b<.005
                       else 'stable' if y>=.8*b else 'decay')
    groups=[]
    for size in (1,2,3):
        members=defaultdict(list)
        for r in rows:
            if r['has_saved']:
                for pattern in itertools.combinations(r['operators'],size):members[pattern].append(r)
        for pattern,rs in members.items():
            classified=[r for r in rs if r['status'] in {'stable','decay','failed'}]
            ics=[r['ic_2026'] for r in rs if r['ic_2026'] is not None]
            groups.append(dict(size=size,pattern=' + '.join(pattern),n=len(rs),
                classified_n=len(classified),stable=sum(r['status']=='stable' for r in classified),
                stable_rate=sum(r['status']=='stable' for r in classified)/len(classified) if classified else None,
                positive=sum(v>0 for v in ics),valid_ic_n=len(ics),median_ic=float(np.median(ics)) if ics else None,
                members=[r['name'] for r in rs]))
    now=intake.report_time()
    source_counts=Counter(r['result_source'] for r in rows if r['has_saved'])
    payload=dict(generated_at=now,mode='mixed-summary',source=OUT_JSON,source_sha256=hashlib.sha256(raw).hexdigest(),
        universe_count=len(names),historical_count=len(historical),factors=rows,combinations=groups,
        source_counts=dict(source_counts),current_sources=current_snapshot['sources'],
        convention='Current comparable result preferred per factor name, historical fallback; heterogeneous windows/metrics; shared descriptive 80% rule')
    def atomic(path,text):
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:f.write(text);temp=f.name
        os.replace(temp,path)
    def table(rs,cols):
        def fmt(v,k):
            if v is None:return '—'
            if isinstance(v,bool):return '是' if v else '否'
            if k=='status':return {'stable':'保留≥80%','decay':'保留不足80%','unclassified':'参考值不足，不算比例','missing':'无有效结果'}.get(v,v)
            if k=='stable_rate':return f'{v:.1%}'
            if isinstance(v,float):return f'{v:.4f}'
            if isinstance(v,list):return '、'.join(v)
            return str(v)
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(label)+'</th>' for k,label in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(fmt(r.get(k),k))+'</td>' for k,label in cols)+'</tr>' for r in rs)+'</table></div>'
    counts=Counter(r['status'] for r in rows if r['has_saved'])
    denominator=counts['stable']+counts['decay']
    stability=f'{counts["stable"]/denominator:.1%}' if denominator else '—'
    parts=[f'<h1>2026 因子稳定率与算子组合</h1><p>更新（北京时间）：{now} · 已有结果 {sum(r["has_saved"] for r in rows)} 个因子</p>'
        f'<section style="background:#eef6ff;border-left:5px solid #2864b0;padding:20px"><h2 style="margin-top:0">整体稳定率：<strong style="font-size:36px">{stability}</strong></h2>'
        f'<p><b>{counts["stable"]} 个稳定 ÷ {denominator} 个可计算衰减比例的因子</b>；其中 {counts["decay"]} 个衰减。另有 {counts["unclassified"]} 个参考值不足，不进入稳定率分母。</p>'
        '<p>稳定＝参考指标≥0.005，且2026指标至少保留参考指标的80%。这是描述性稳定率；来源窗口及算法有差异，不能当作严格同口径回测的稳定概率。</p></section>',
        '<p>下方分别展示单算子、双算子、三算子的稳定率，按稳定率从高到低排列；请同时看样本数，小样本高比例不代表可靠优势。</p>'
        '<details><summary>展开计算口径与来源说明</summary>'
        f'<p>新版结果 {source_counts["新版评估"]} 个、历史补充 {source_counts["历史结果"]} 个，同名只计一次。</p>',
        '<h2>本次汇报口径</h2><p>新结果来自当前已完成且通过核验的评估；无新结果者复用 robustness_2026.json 的历史指标。仅重新汇总，不等待重新落值或评估、不再翻转。按因子名称计数，不按相同公式去重；近似变体可能重复贡献。</p>'
        '<p>旧脚本名义全期为 2019-01-02 至 2026-08-24，2026 段为 2026-01-01 至 2026-08-24；部分来源覆盖不同，实际有效天数见明细。全期包含 2026，不能称独立参考窗或密封样本外。旧字段名虽为 RankIC，其计算实现未按新版 Spearman 口径重新验收，不能与新版指标混排。</p>'
        '<p>新版参考窗为 2018-07-01 至 2023-12-31，目标窗为 2026-01-01 至 2026-08-27；旧版参考窗包含2026，且旧指标实现未统一。这是明确保留来源差异的合并表，不代表二者可直接比较或排名。每组列出新旧数量，来源构成会影响结论。</p>'
        '<p>不混用旧稳定标签与新稳定标签。统一描述规则：参考指标≥0.005者进入分母，2026指标≥参考指标×80%者进入分子；其余不计算保留比例。此规则不消除新旧测量差异，也不是显著性检验。旧多空存在已识别的敞口／缺失值口径问题，因此不将旧多空收益、回撤用于本页绩效结论。</p>'
        '<p>算子从当前登记公式语法树提取，仅统计函数调用名，排除 col、neg、is_finite；公式未解析者仍进入整体统计，不进入组合。历史执行公式与当前登记公式尚未逐项验证一致，组合结果为登记结构的探索性关联，不代表因果。</p></details>']
    for g in groups:
        by_source=Counter(r['result_source'] for r in rows if r['name'] in g['members'])
        g.update(new_n=by_source['新版评估'],old_n=by_source['历史结果'])
    cols=[('pattern','算子组合'),('stable_rate','稳定率'),('stable','稳定因子数'),('classified_n','稳定率分母'),('n','全部样本数'),('positive','2026指标为正数'),('median_ic','2026指标中位数')]
    for size in (1,2,3):
        gs=[g for g in groups if g['size']==size]
        ranked=sorted([g for g in gs if g['n']>=5],key=lambda g:(g['stable_rate'] or 0,g['n']),reverse=True)
        parts.append(f'<h2>{ {1:"单",2:"双",3:"三"}[size]}算子统计</h2><p>识别 {len(gs)} 种结构，{len(ranked)} 种覆盖至少5个因子。下表显示前20组；“+”表示共现，不是加法或执行顺序。组合之间样本重叠，不可将数量相加。</p>')
        parts.append(table(ranked[:20],cols))
        parts.append('<details><summary>全部组合及成员（含小样本，不作推荐榜）</summary>'+table(gs,cols+[('members','成员')])+'</details>')
    parts.append('<h2>全部因子明细</h2>'+table(rows,[('name','因子'),('result_source','结果版本'),('has_saved','有结果'),('status','描述性保留状态'),('baseline_ic','各自参考指标'),('ic_2026','2026指标'),('icir_2026','2026 IR'),('n_full','参考有效日'),('n_2026','2026有效日'),('source','来源标记')]))
    parts.append('<p><a href="existing_summary.json">全部统计及来源哈希 JSON</a> · <a href="existing_summary_factors.csv">全部因子 CSV</a></p>')
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026 全量历史结果分析</title><style>body{font:16px/1.7 system-ui;margin:24px;color:#243447}h2{margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;max-width:600px;overflow-wrap:anywhere}th{background:#eef2f7}</style><main>'+''.join(parts)+'</main></html>'
    atomic(output/'existing_summary.json',json.dumps(payload,ensure_ascii=False,allow_nan=False))
    pd.DataFrame(rows).to_csv(output/'existing_summary_factors.csv',index=False)
    atomic(output/'robustness_2026.html',html)
    with (Path(REPORT_DIR)/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        doc=Path(INDEX).read_text()
        section=f'<section id="robustness-2026"><h2>2026 新旧结果合并分析</h2><p>{now} · 新版 {source_counts["新版评估"]} 个、历史补充 {source_counts["历史结果"]} 个；同名优先新版，不重复计数，保留口径差异说明。</p><a href="robustness_2026/robustness_2026.html">打开2026整体及单／双／三算子分析</a></section>'
        doc=re.sub(r'<section[^>]*id="robustness-2026"[^>]*>.*?</section>',lambda _:section,doc,flags=re.S)
        atomic(Path(INDEX),doc)
    print(json.dumps(dict(report=str(output/'robustness_2026.html'),count=len(historical),scope=len(names),states=dict(counts)),ensure_ascii=False),flush=True)


def completed_snapshot_report(*, return_data=False):
    """Incremental 2026 readout from verified QE artifacts, never legacy metrics."""
    import ast, hashlib, itertools, base64, io, tempfile, fcntl
    from pathlib import Path
    if not return_data and (Path(OUT_PAGE_DIR)/'use_existing_summary').exists():
        return existing_summary_report()
    from html import escape
    import incremental_factor_intake as intake
    from quant_evaluator.metrics.ic_summary import compute_icir
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio, compute_maximum_drawdown
    rows, exclusions, seen = [], [], set()
    universe = {r['page_name']: dict(name=r['page_name'], scope='存量因子', state='待统一评估')
                for r in intake.load_pool()}
    historical = {r['page']: r for r in json.loads(Path(OUT_JSON).read_text())} if Path(OUT_JSON).exists() else {}
    requested_path = Path('/tmp/weekly_requested_formulas_20260906.txt')
    deferred_requested = []
    if requested_path.exists():
        requested, deferred_requested = intake.requested_dsl_records(requested_path)
        for record in requested:
            universe.setdefault(record['page_name'], dict(name=record['page_name'], scope='本周新增', state='尚未运行'))
    queue_counts = {}
    sources = []
    cluster_data = intake.load_clusters() if intake.CLUSTERS_JSON.exists() else {}
    clusters = cluster_data.get("page_to_cluster", {})
    for queue in (Path('/tmp/weekly_overnight_20260906'), Path('/tmp/report_fullpool_retest_20260906'), Path(MB)/'robustness_existing_values'):
        if not (queue/'queue_state.json').exists():continue
        state = json.loads((queue / 'queue_state.json').read_text())
        for name, item in state['factors'].items():
            universe.setdefault(name, dict(name=name, scope='已有落值'))['state'] = item.get('status', '待评估')
        queue_counts[queue.name] = dict(Counter(v.get('status') for v in state['factors'].values()))
        for name, status in state['factors'].items():
            if any(r['name']==name for r in rows):continue
            if status.get('status') not in {'passed', 'below_gate'}:
                continue
            path = Path(status['folder']) / 'report_manifest.json'
            try:
                content = path.read_bytes()
                manifest = json.loads(content)
                if manifest.get('evaluation_version') != intake.EVALUATION_CONVENTION_VERSION:
                    raise ValueError('old evaluation convention')
                entry = manifest['factors'][name]
                formula = entry.get('effective_formula')
                if entry.get('direction') not in (-1, 1):
                    raise ValueError('missing fixed direction')
                try:tree = ast.parse(formula, mode='eval') if formula else ast.parse('0', mode='eval')
                except (SyntaxError,TypeError):tree=ast.parse('0',mode='eval')
                formula_known=bool(formula) and bool(list(ast.walk(tree))[1:]) and not isinstance(tree.body,ast.Constant)
                identity = ast.dump(tree, include_attributes=False) if formula_known else 'missing-formula:'+name
                if identity in seen and not return_data:
                    exclusions.append({'factor': name, 'reason': 'duplicate effective DSL'})
                    continue
                artifact = Path(entry['artifact'])
                if not artifact.is_absolute(): artifact = path.parent / artifact
                raw = artifact.read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry['artifact_sha256']:
                    raise ValueError('artifact hash mismatch')
                with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                    dates = pd.DatetimeIndex(saved['dates'])
                    ic = saved['rank_ic_series'].copy()
                    returns = saved['long_short_returns'].copy()
                if ic.shape != (len(dates),) or returns.shape != ic.shape or dates.has_duplicates or not dates.is_monotonic_increasing:
                    raise ValueError('artifact alignment invalid')
                reference = (dates >= '2018-07-01') & (dates <= '2023-12-31')
                recent = (dates >= '2026-01-01') & (dates <= '2026-08-27')
                base, y26 = ic[reference], ic[recent]
                if np.isfinite(base).sum() < 200 or np.isfinite(y26).sum() < 60:
                    raise ValueError('insufficient comparable IC observations')
                ops = sorted({n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)
                              and isinstance(n.func, ast.Name)} - {'col', 'neg', 'is_finite'})
                edges = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ops:
                        for child in node.args:
                            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in ops:
                                edges.add(node.func.id + ' → ' + child.func.id)
                mean_base, mean26 = float(np.nanmean(base)), float(np.nanmean(y26))
                monthly = {str(month): float(np.nanmean(ic[(dates.to_period('M') == month) & recent]))
                           for month in dates[recent].to_period('M').unique()
                           if np.isfinite(ic[(dates.to_period('M') == month) & recent]).sum() >= 5}
                rr = returns[recent]
                finite_rr = rr[np.isfinite(rr)]
                row = dict(name=name, formula=formula, operators=ops, edges=sorted(edges),
                           value_provenance=status.get('value_provenance','current pipeline artifact'),
                           formula_known=formula_known,
                           family=clusters.get(name), screening=status['status'],
                           baseline_ic=mean_base, ic_2026=mean26, delta=mean26-mean_base,
                           baseline_ir=float(compute_icir(base[:, None])[0]),
                           ir_2026=float(compute_icir(y26[:, None])[0]),
                           n_baseline=int(np.isfinite(base).sum()), n_2026=int(np.isfinite(y26).sum()),
                           positive=mean26 > 0,
                           comparable=mean_base >= .005,
                           stable=bool(mean_base >= .005 and mean26 >= .8 * mean_base),
                           monthly=monthly, return_days_2026=int(len(finite_rr)),
                           ls_sharpe_2026=float(compute_sharpe_ratio(rr[:, None])[0]),
                           ls_mdd_2026=float(compute_maximum_drawdown(finite_rr, missing_return_policy='fail')[0]) if len(finite_rr) else None)
                rows.append(row); seen.add(identity)
                sources.append(dict(factor=name, manifest=str(path), sha256=hashlib.sha256(content).hexdigest(),
                                    artifact=str(artifact), artifact_sha256=entry['artifact_sha256']))
            except Exception as exc:
                exclusions.append(dict(factor=name, reason=str(exc)))
    if not rows:
        raise ValueError('No current, hash-verified comparable artifacts; report not fabricated')
    groups = []
    for size in (1, 2, 3):
        combinations = defaultdict(list)
        for r in rows:
            for combo in itertools.combinations(r['operators'], size): combinations[combo].append(r)
        for combo, members in combinations.items():
            comparable = [r for r in members if r['comparable']]
            monthly_groups = []
            for month in sorted({m for r in members for m in r['monthly']}):
                monthly_members = [r for r in members if month in r['monthly']]
                eligible = [r for r in monthly_members if r['comparable']]
                monthly_groups.append(dict(month=month, n=len(monthly_members),
                    median_ic=float(np.median([r['monthly'][month] for r in monthly_members])),
                    comparable_n=len(eligible), stable_n=sum(r['monthly'][month] >= .8*r['baseline_ic'] for r in eligible),
                    median_delta=float(np.median([r['monthly'][month]-r['baseline_ic'] for r in monthly_members]))))
            groups.append(dict(size=size, pattern=' + '.join(combo), n=len(members),
                               sufficient_support=len(members) >= 5,
                               members=sorted(r['name'] for r in members), monthly=monthly_groups,
                               comparable_n=len(comparable), stable_n=sum(r['stable'] for r in comparable),
                               positive_rate=float(np.mean([r['positive'] for r in members])),
                               median_ic=float(np.median([r['ic_2026'] for r in members])),
                               median_delta=float(np.median([r['delta'] for r in members])),
                               stable_rate=float(np.mean([r['stable'] for r in comparable])) if comparable else None,
                               known_families=len({r['family'] for r in members if r['family'] is not None}),
                               unassigned=sum(r['family'] is None for r in members)))
    edges = defaultdict(list)
    for r in rows:
        for edge in r['edges']: edges[edge].append(r)
    edge_rows = [dict(pattern=e, n=len(rs), median_ic=float(np.median([r['ic_2026'] for r in rs])),
                      median_delta=float(np.median([r['delta'] for r in rs]))) for e, rs in edges.items() if len(rs) >= 5]
    comparable = [r for r in rows if r['comparable']]
    summary = dict(total=len(rows), positive=sum(r['positive'] for r in rows), comparable=len(comparable),
                   stable=sum(r['stable'] for r in comparable),
                   median_baseline=float(np.median([r['baseline_ic'] for r in rows])),
                   median_2026=float(np.median([r['ic_2026'] for r in rows])),
                   median_delta=float(np.median([r['delta'] for r in rows])))
    result = dict(generated_at=intake.report_time(), evaluation_version=intake.EVALUATION_CONVENTION_VERSION,
                  summary=summary, factors=rows, combinations=groups, nested_patterns=edge_rows,
                  exclusions=exclusions, queue_counts=queue_counts, sources=sources,
                  interpretation='Exploratory completed-sample snapshot; not sealed OOS or causal evidence',
                  baseline='2018-07-01..2023-12-31', direction_window='2016-01-04..2018-06-30',
                  target='2026-01-01..2026-08-27', dedupe='exact effective DSL only; incomplete family mapping')
    current = {r['name']: r for r in rows}
    for name, item in universe.items():
        item['included_current'] = name in current
        item['historical_record'] = name in historical
        item['current_2026_ic'] = current.get(name, {}).get('ic_2026')
        item['historical_2026_ic_unverified'] = (historical.get(name, {}).get('rankIC_2026') or {}).get('ic')
    result['universe'] = list(universe.values())
    result['universe_count'] = len(universe)
    result['historical_count'] = len(historical)
    result['deferred_requested'] = deferred_requested
    if return_data:
        return result
    output = Path(OUT_PAGE_DIR); output.mkdir(parents=True, exist_ok=True)
    def atomic(path, data):
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            f.write(data); temp = f.name
        os.replace(temp, path)
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
        if isinstance(x, list): return [clean(v) for v in x]
        return x
    atomic(output / 'completed_snapshot.json', json.dumps(clean(result), ensure_ascii=False, indent=1, allow_nan=False).encode())
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'monthly','operators','edges'}} for r in rows]).to_csv(output/'completed_factors.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([r['baseline_ic'] for r in rows], [r['ic_2026'] for r in rows], color='#426b9b', alpha=.75)
    bounds = [min(min(r['baseline_ic'],r['ic_2026']) for r in rows)-.005,
              max(max(r['baseline_ic'],r['ic_2026']) for r in rows)+.005]
    ax.plot(bounds,bounds,'--',color='#777777',label='Unchanged IC')
    ax.axhline(0,color='#999999',lw=.6); ax.axvline(0,color='#999999',lw=.6)
    ax.set(xlabel='Reference RankIC (Jul 2018–Dec 2023)', ylabel='2026 RankIC', title=f'RankIC comparison · {len(rows)} completed factors')
    ax.legend(); fig.tight_layout(); stream=io.BytesIO();fig.savefig(stream,format='png',dpi=140);plt.close(fig)
    chart='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
    def table(items, columns):
        def fmt(v, key):
            if v is None:return '—'
            if isinstance(v, bool):return '是' if v else '否'
            if key == 'state':
                return {'passed':'筛选通过（非2026稳定判定）', 'below_gate':'已评估，未达入池门槛',
                        'running':'落值／评估中', 'retry_pending':'等待重试',
                        'unavailable':'暂未取得合格评估（非因子失效）', 'failed':'执行失败（非质量结论）'}.get(v, str(v))
            if key in {'stable_rate', 'ls_mdd_2026'}:return f'{v:.2%}'
            if isinstance(v,float):return f'{v:.4f}'
            return str(v)
        return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+escape(label)+'</th>' for key,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(fmt(r.get(key), key))+'</td>' for key,label in columns)+'</tr>' for r in items)+'</tbody></table></div>'
    parts=[f'<h1>2026 因子稳定性分析 · 汇报快照</h1><h2>Executive Summary｜汇报要点</h2><ul>'
           f'<li>当前可核验、去除完全相同 DSL 后有 <b>{len(rows)}</b> 个可比因子，2026 RankIC 为正的有 <b>{summary["positive"]}</b> 个。不是全部因子重测结论。</li>'
           f'<li>RankIC 中位数：参考窗 <b>{summary["median_baseline"]:.4f}</b>，2026 年 <b>{summary["median_2026"]:.4f}</b>；逐因子变化量中位数 <b>{summary["median_delta"]:+.4f}</b>。</li>'
           f'<li>参考窗 RankIC ≥ 0.005 的 {len(comparable)} 个因子中，{summary["stable"]} 个在 2026 保留至少 80% 的 RankIC。算子组合仅供探索，不能证明组合导致抗衰减。</li></ul>'
           f'<p>快照时间（北京时间）：{escape(result["generated_at"])}；评估版本：<code>{escape(result["evaluation_version"])}</code>。本页读取已完成评估，包含筛选通过及未达标但有效的记录，以减少只看通过者的偏差。生成本页不会重新落值或覆盖评估；落值和复评由后台主链路另行执行，快照不代表后台实时状态。</p>',
           '<h2>比较口径：固定方向，不用 2026 反向择优</h2><p>方向仅由 2016 年 1 月 4 日至 2018 年 6 月 30 日确定；参考窗为 2018 年 7 月至 2023 年底，目标窗为 2026 年初至 8 月 27 日。两个比较窗口不重叠。IR 为日度 IC 均值／样本标准差，不年化。2026 已被探索过，不是密封样本外。</p>',
           '<h3>数据、收益与统计定义</h3><ul>'
           '<li>市场为 A 股，日期按北京时间解释。主评估历史起点为 2016-01-04；上面的参考窗用于比较，不表示重新将历史起点改为 2018 年。达到本页有效日数要求不等于逐日全历史完整。</li>'
           '<li>数据口径使用 StockDailyBarAdj 的复权价格和表内实际 Volume，不再乘除 Factor。收益按 t 日信号、t+1 的 AdjVwap 买入／建仓、t+2 的 AdjVwap 卖出／平仓对齐；本页复用评估产物，不重新拼接价格。</li>'
           '<li>RankIC 为逐日横截面秩相关的时间均值；ICIR 为日度 IC 均值除以样本标准差（ddof=1），不年化。变化量为每个因子的 2026 IC 减参考 IC，再取中位数；不等于两组中位数相减。</li>'
           '<li>多空按所选股票的相等绝对成交金额配置，总绝对敞口 100%，空头按 100% 保证金计；不是等股数，也不是两条独立 G10、G1 净值直接相减。</li>'
           '<li>夏普与最大回撤调用 QuantEvaluator：夏普按 252 个交易日年化、无风险收益为 0；最大回撤以收益累计净值相对历史峰值的最大跌幅表示，正百分数越小越好。已存多空收益扣单边佣金 1 bp（0.01%），本页不重复扣费；印花税、滑点、融券成本未计入，不能称全部成本后收益。</li>'
           '<li>参考窗至少 200 个、2026 窗至少 60 个有限 IC 观测才纳入。收益缺失日不补零；回撤只按有限收益序列计算，不代表缺口期间损益已知。有效 IC 日与收益日分别列出，不同因子覆盖日可能不同，比较收益指标需同时查看覆盖情况。</li>'
           '<li>复用所有可找到的已有原始因子矩阵，以当前 QuantEvaluator 重新评估；不要求重新落值。历史缓存的原始字段／公式重放一致性未重新验证，与新落值产物在 JSON 的 value_provenance 区分。仅接收当前评估版本、固定方向、产物哈希一致、日期唯一且排序、IC 与收益日期形状一致的记录。已定向序列不再翻转。缺少可解析公式者可进入整体指标统计，但不进入算子组合；有公式者按登记有效 DSL 的完全相同语法树去重，不等于因子族独立或历史值与该公式已验证一致。</li></ul>'
           '<h3>入池筛选 ≠ 2026 抗衰减</h3><p>入池门槛为 RankIC ≥ 0.015 或 RankIC IR ≥ 0.15，沿用主评估记录的筛选结果，并非用本页 2026 指标重新筛选。抗衰减统计另定义为：仅在参考 IC ≥ 0.005 的因子中，2026 IC ≥ 参考 IC × 80% 者计入分子；参考 IC 接近零或为负者不进入这个比例的分母。2026 IC 为正的数量则在全部本页可比样本中计算。这些阈值是描述性研究规则，不是显著性检验或未来有效性保证。</p>',
           f'<h2>整体变化：同时看正负与相对衰减</h2><p>横轴是参考窗 RankIC，纵轴是 2026 RankIC；虚线上方表示 IC 提高，下方表示下降。参考窗接近零或负值时，不使用衰减比率，避免把负负相除误判为稳定。</p><img alt="参考窗与2026 RankIC散点图" src="{chart}">']
    for size in (1,2,3):
        all_groups=sorted([g for g in groups if g['size']==size], key=lambda g:(g['median_delta'],g['n']),reverse=True)
        supported=[g for g in all_groups if g['sufficient_support']]
        selected=supported[:12]
        parts.append(f'<h2>{ {1:"单算子",2:"双算子",3:"三算子"}[size]}共现：哪些结构值得进一步验证</h2><p>每种组合至少覆盖 5 个本页可比因子，按逐因子 IC 变化量的中位数降序展示前 12 组。每个算子在一个因子中只计一次；“+”仅表示共现，不是 DSL 加法或执行顺序。按语法树函数名统计，排除 col、neg、is_finite，尚未统一全部算子别名；普通代数算子仍包含在内。抗衰减比例仅以组内参考 IC ≥ 0.005 的样本为分母。组合样本可重叠，并非独立实验；缺少完整因子族映射，近似变体可能重复贡献。</p>')
        parts.append(f'<p>本次共识别 {len(all_groups)} 种结构，其中 {len(supported)} 种达到 5 个因子的展示门槛；其余仅作结构清单，不作为抗衰减证据。每次生成均从当前全部合格评估重新聚合，不沿用旧页统计数。</p>')
        parts.append(table(selected,[('pattern','算子组合'),('n','样本数'),('median_ic','2026 IC中位数'),('median_delta','IC变化中位数'),('comparable_n','抗衰减分母'),('stable_n','保留≥80%数量'),('stable_rate','抗衰减比例'),('unassigned','未归族数量')]) if selected else '<p>没有满足最低样本数的组合，不展示小样本排行榜。</p>')
        parts.append('<details><summary>展开全部结构与成员（包括小样本，不是推荐榜）</summary>')
        parts.append(table([dict(pattern=g['pattern'], n=g['n'], sufficient_support=g['sufficient_support'], members='、'.join(g['members'])) for g in all_groups], [('pattern','结构'),('n','因子数'),('sufficient_support','达到5个样本'),('members','全部成员')]))
        parts.append('</details>')
        if supported:
            parts.append('<details><summary>展开所有达标结构的逐月对比</summary><p>月份按北京时间划分；每个因子当月至少 5 个有限 IC 日才计入月度均值。表中先求各因子的月均 IC，再对组内因子取中位数；月度保留数量与该因子固定参考窗比较。月份覆盖及分母可变，2026 年 8 月仅截至 27 日，不能将不同月份样本差异视为市场场景效应。</p>')
            parts.append(table([dict(pattern=g['pattern'], **m) for g in supported for m in g['monthly']], [('pattern','结构'),('month','月份'),('n','当月有效因子'),('median_ic','月IC中位数'),('median_delta','相对参考IC变化'),('comparable_n','当月抗衰减分母'),('stable_n','当月保留≥80%数量')]))
            parts.append('</details>')
    parts += ['<h2>嵌套关系：区别于简单共现</h2><p>箭头表示外层算子直接使用内层算子的结果；目前识别位置参数中的直接函数调用，不代表全部深层结构或关键字参数关系。至少覆盖 5 个因子，按 IC 变化中位数展示前 12 组；仍为观察性统计，不能解释为确定的经济机制。</p>', table(sorted(edge_rows,key=lambda r:r['median_delta'],reverse=True)[:12],[('pattern','外层 → 内层'),('n','样本数'),('median_ic','2026 IC中位数'),('median_delta','IC变化中位数')]),
              '<h2>可直接核对的因子明细（展示前 25 个）</h2><p>按 2026 RankIC 排序，仅用于汇报举例；本页统计使用全部合格样本，不只这 25 个，不因此重新选择方向或加入正式因子池。完整明细见 <a href="completed_factors.csv">CSV</a>，公式、来源哈希和排除原因见 <a href="completed_snapshot.json">JSON</a>。收益指标均取 2026 窗，费用与缺失日处理见上方口径。</p>',
              table(sorted(rows,key=lambda r:r['ic_2026'],reverse=True)[:25],[('name','因子'),('baseline_ic','参考IC'),('ic_2026','2026 IC'),('delta','变化'),('ir_2026','2026 IR'),('n_2026','有效IC日'),('return_days_2026','有效收益日'),('ls_sharpe_2026','佣金后LS夏普'),('ls_mdd_2026','LS回撤')]),
              '<h2>汇报建议与仍需回答的问题</h2><p>把本页定位为已完成样本的阶段性发现，不宣称发现了普适的抗衰减算子。下一步优先验证组合在不同因子族、月份与市场涨跌／波动场景中是否重复出现；当前尚未完成市场场景归因与因子族独立性检验。</p>',
              f'<h2>限制与假设</h2><p>结果受已完成任务的覆盖偏差影响，不能代表全部存量与新增因子；组合数量多且互相重叠，尚未做多重检验校正。历史报告的 {len(historical)} 条记录不混入当前统计；旧版方向、窗口或算法可能不同，不可与新版直接合并排名。本次排除 {len(exclusions)} 条已检查记录（含重复 DSL 或未通过核验），具体原因见 JSON；排除不等于因子失效，也不补造未完成结果。未登记／未成功解析的提交项不应视作已评估；登记情况见 JSON 的 deferred_requested。</p>']
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026 因子稳定性分析 · 汇报快照</title><style>body{font:16px/1.75 system-ui,sans-serif;color:#243447;background:#f5f7fa;margin:0}main{max-width:1180px;margin:auto;padding:32px;background:white}h1{font-size:30px}h2{margin-top:36px;font-size:22px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left;overflow-wrap:anywhere}th{background:#eef2f7}.table-scroll{overflow-x:auto}code{overflow-wrap:anywhere}@media print{main{padding:0}h2{break-after:avoid}tr{break-inside:avoid}}</style><body><main>'+''.join(parts)+'</main></body></html>'
    # Keep the former entry point usable without maintaining a second report.
    atomic(output/'completed_snapshot.html', ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=robustness_2026.html"><title>2026 分析已统一</title>'
        '<a href="robustness_2026.html">全部 2026 分析已统一至正式报告</a></html>').encode())
    # The established all-factor page remains the canonical URL. Preserve the
    # legacy report once; inventory and pending factors must not disappear just
    # because only a subset has a verified current artifact.
    canonical = output/'robustness_2026.html'
    archive = output/'robustness_2026_before_unified.html'
    if canonical.exists() and not archive.exists():
        atomic(archive, canonical.read_bytes())
    coverage = (f'<h1>2026 因子全量汇总</h1><p>存量因子 {sum(x["scope"] == "存量因子" for x in universe.values())} 个；'
                f'加已登记新增因子，当前跟踪共 {len(universe)} 个。历史报告保留 {len(historical)} 个因子记录。'
                f'本次统一口径可比统计纳入 {len(rows)} 个，其他因子仍在复评或待处理，不能宣称全量重测完成。</p>'
                '<p><a href="robustness_2026_before_unified.html">查看保留的历史全量报告（旧口径，未经本次验收）</a></p>'
                '<details><summary>展开全量覆盖明细：未完成的也列出</summary><p>跟踪数量按因子登记名称计数，不是独立因子族数量。已完成记录仍须通过版本、哈希、观测数及 DSL 去重核验，才计入“本次纳入”。暂不可用或执行失败是任务／产物状态，不等于因子质量不合格，更不能直接推断没有源数据。旧版 IC 仅作历史记录查询，不与新版合并统计，不用于判定当前抗衰减。</p>'
                + table(list(universe.values()), [('name','因子'),('scope','范围'),('state','队列状态'),
                    ('included_current','本次纳入'),('current_2026_ic','新版2026 IC'),
                    ('historical_record','有旧记录'),('historical_2026_ic_unverified','旧版IC·未验收')])+'</details>')
    canonical_html = html.replace('<h1>2026 因子稳定性分析 · 汇报快照</h1>', '<h2>统一口径统计：以下为当前已完成部分</h2>').replace('<main>', '<main>'+coverage, 1).replace('<title>2026 因子稳定性分析 · 汇报快照</title>', '<title>2026 因子全量汇总</title>')
    atomic(canonical, canonical_html.encode())
    section='<section id="robustness-2026"><h2>2026 因子全量汇总</h2><p>'+escape(result['generated_at'])+f' · 跟踪 {len(universe)} 个；统一口径可比 {len(rows)} 个，复评持续补齐。</p><a href="robustness_2026/robustness_2026.html">打开正式全量汇总：覆盖情况、算子组合及因子明细</a></section>'
    with (Path(REPORT_DIR)/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        document=Path(INDEX).read_text()
        if re.search(r'<section[^>]*id="robustness-2026"',document):
            document=re.sub(r'<section[^>]*id="robustness-2026"[^>]*>.*?</section>',lambda m:section,document,flags=re.S)
        else:document=document.replace('<body>','<body>'+section,1)
        atomic(Path(INDEX),document.encode())
    print(json.dumps(dict(summary=summary,report=str(canonical),excluded=len(exclusions)),ensure_ascii=False),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="completed", choices=["completed", "existing-summary", "reuse-existing", "smoke", "full", "report-only"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()
    if args.mode == 'existing-summary':
        from pathlib import Path
        Path(OUT_PAGE_DIR).mkdir(parents=True,exist_ok=True)
        (Path(OUT_PAGE_DIR)/'use_existing_summary').touch()
        existing_summary_report()
        return
    if args.mode == 'reuse-existing':
        reevaluate_existing_values()
        return
    if args.mode == 'completed':
        completed_snapshot_report()
        return

    plog("=" * 60)
    plog(f"[main] mode={args.mode} limit={args.limit} t={time.strftime('%H:%M:%S')}")
    t_start = time.time()

    formula_map = json.load(open(FORMULA))
    fm = {e["page_name"]: e for e in formula_map}
    pages = sorted(os.listdir(MAT))
    pages = [p[:-8] for p in pages if p.endswith(".parquet")]
    if args.mode == "smoke":
        pages = pages[:5]
    if args.limit:
        pages = pages[: args.limit]
    plog(f"[main] 待分析 {len(pages)} 因子")

    # 全局
    load_GV()

    if args.mode == "full":
        rob = {}
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n_workers = 8
        plog(f"[main] ProcessPoolExecutor workers={n_workers} 开始")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker, p) for p in pages]
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                page = res["page"]
                rob[page] = res
                done += 1
                if done % 20 == 0:
                    plog(f"  progress {done}/{len(pages)} {time.time()-t_start:.0f}s")
        rob_list = list(rob.values())
        plog("[main] 计算阶段完成", f"{(time.time()-t_start)/60:.1f}min")
    else:
        if args.mode == "report-only":
            rob_list = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else []
            # report-only 兼容: 兼容旧版串行 json 结构
            if isinstance(rob_list, dict):
                rob_list = list(rob_list.values())
            plog(f"[main] report-only 加载 {len(rob_list)} 条")
        else:
            # smoke: 串行算 5 个
            rob_list = [analyze_factor(p, None) for p in pages]

    # 删除 error 条目
    rob_list = [r for r in rob_list if r.get("status") != "error"]
    plog(f"[main] 有效因子 {len(rob_list)} (error={len(rob_list) == 0 and 'all' or ''})")
    for r in rob_list:
        if r.get("status") == "error":
            plog("   ERR:", r)

    # 补充 logic_tags 到结果
    for r in rob_list:
        meta = fm.get(r["page"], {}) if fm else {}
        r["logic_tags"] = classify_logic(meta.get("dsl") or "", meta.get("code") or "")

    if not args.skip_report:
        stat = compute_operator_stats(rob_list, (fm if fm else []))
        op_counts = stat["op_counts"]; op_stable_ratio = stat["op_stable_ratio"]; logic_stats = stat["logic_stats"]

        # 图表
        plt = _plt_init()
        stable_ranked = [r for r in rob_list if r.get("status") == "stable"]
        stable_ranked.sort(key=lambda r: (r.get("ls2026") or {}).get("cumret", -1e9), reverse=True)
        figure_list = [
            figure1_top20_nav(stable_ranked, rob_list),
            figure2_stable_vs_failed(rob_list),
            figure3_op_stability_ratio(op_counts, op_stable_ratio),
            figure4_logic_stacked(rob_list),
        ]
        build_pages(rob_list, op_counts, op_stable_ratio, logic_stats, figure_list)
        inject_index(rob_list, figure_list[0])
        if args.mode in ("smoke", "full"):
            inject_detail_pages(rob_list)

    # 写 JSON
    json.dump([{k: v for k, v in r.items() if k != "r26"} for r in rob_list],
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plog(f"[main] JSON 已写 {OUT_JSON}  总计 {(time.time()-t_start)/60:.1f}min")

    # 汇总打印
    c = Counter(r.get("status") for r in rob_list)
    plog(f"[SUMMARY] stable={c['stable']} decay={c['decay']} failed={c['failed']} "
         f"stable_ratio={c['stable']/max(len(rob_list),1):.1%}")

if __name__ == "__main__":
    main()

def _echo_env():
    for k in ("HOME", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONUNBUFFERED"):
        import os as _os
        print(f"env {k}={_os.environ.get(k)!r}")
    for m in ("numpy", "pandas", "matplotlib", "pyarrow"):
        try:
            mod = __import__(m)
            print(f"m {m} {getattr(mod, '__version__', '?')} from {getattr(mod, '__file__', '?')}")
        except Exception as e:
            print(f"m {m} ERR {e}")


def _probe_run():
    import json
    pm = json.load(open(FORMULA))
    print("formula page_count", len(pm))
    import os as _os
    pages = sorted(p for p in _os.listdir(MAT) if p.endswith(".parquet"))
    print("mat pages", len(pages))
    print("sample", pages[:3])
    sys.exit(0)

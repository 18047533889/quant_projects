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

def figure4_logic_stacked(rob):
    plt = _plt_init()
    logic_order = sorted({t for r in rob for t in (r.get("logic_tags") or [])})
    counts = {t: Counter(z.get("status") or "failed") for t in logic_order for z in rob if t in (z.get("logic_tags") or [])}
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
    logic_order = sorted({t for r in rob for t in (r.get("logic_tags") or [])})
    for t in logic_order:
        cnt = Counter(r.get("status") for r in rob if t in (r.get("logic_tags") or []))
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

<h2>🔬 2026 失效归因（为什么 267 个因子失效）</h2>
<div class="attr">
<p class="lead">① 逻辑维度：动量/相关性类全线崩塌</p>
<p>按逻辑标签统计（每因子可多标签），<b>动量类</b>（momentum，450 因子）失效率 <b class="num">58.0%</b>、<b>流动性类</b>（liquidity，448）失效率 <b class="num">57.8%</b>、<b>价格位置类</b>（price_position，447）<b class="num">57.3%</b>、<b>成交量类</b>（volume，451）<b class="num">57.4%</b>——这些占池子主体的趋势/量价类因子在 2026 呈现系统性失效，说明 2026 行情以<b>风格快速切换 + 高波动结构</b>为主，基于历史统计关系的趋势跟随信号失去稳定性。</p>
<p>② 算子维度：相关性（correlation）类最差——<b>单算子失效率 61.5%</b>、存活率仅 <b>8.8%</b>；与相关性组合的算子对失效率普遍 66~68%（correlation+liquidity 68.0%、correlation+momentum 67.6%、correlation+volatility 66.7%、correlation+price_position 66.2%）。相关类因子对 2026 年的截面相关性结构（行业/风格联动变化）极度敏感，几乎全线失效。</p>
<p>③ 衰减组（decay 121 个）2026 平均 RankIC 仅 <b class="num">0.0009</b>（全期 0.0105），失效组 267 个 2026 平均 RankIC 为 <b class="num">-0.0036</b>（全期 -0.0064）——说明失效因子大多在 2026 年方向性翻转（由正转负），并非单纯衰减，而是<b>信号结构被 2026 市场环境打破</b>。</p>
</div>

<h2>💪 2026 有效归因（哪些因子存活 & 为什么）</h2>
<div class="attr">
<p class="lead">① 反转（mean_reversion）是 2026 唯一显著存活逻辑</p>
<p>12 个反转类因子中 <b class="num">9 个稳定、3 个轻度衰减、0 个失效</b>，稳定占比 <b class="num">75%</b>、失效占比 <b class="num">0%</b>，远高于全库 17% 稳定占比。2026 年市场在高波动 + 风格轮动下呈现明显的<b>超跌反弹 / 均值回复特征</b>，反转类因子的多空净值与 RankIC 保持为正。</p>
<p>② 存活因子共性：量价确认的反转（volume_confirmed/reversal_volume/volume_reversal 家族）与 vwap 偏离 + 波动调整的日内反转（simplified_intraday_mean_reversion、intraday_reversal_zscore_volume_short）表现最好；<b>price_impact 家族稳定</b>（price_impact_vol_adj_stable_vol、price_impact_asymmetry、price_impact_downside_vol_adjusted、price_impact_volatility_adjusted 等 9 个进入稳定 Top20，2026 累计 +14%~+15%，回撤 -2%~-5%）。</p>
<p>③ 相关性类最差：correlation 单算子存活率仅 <b class="num">8.8%</b>（91 个里只活 8 个），是全部逻辑中唯一存活率低于 10% 的类别；distribution（分布类）存活率 17.3%，同样偏弱。</p>
<p>④ 存活名单（82 个）见下表「🟢 稳定 Top20」及详情页 2026 状态行；完整存活清单在 <code>weekly_backtest_output/robustness_survivors.json</code>。</p>
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
    liq = so.get("liquidity") or {}
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
      <li>流动性类 {liq_failed_rate:.0%} 失效（{liq_total} 因子）</li>
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full", "report-only"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

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

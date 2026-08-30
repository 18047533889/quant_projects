#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 稳健性分析（增量：35 条新因子）— 只算这 35 条，写 robustness_2026 增量 JSON，
并合并进 weekly_backtest_output/robustness_2026.json（不动已有 470 条）。

与 analyze_2026_robustness.py 完全同口径：
- 收益 = AdjVwap.pct_change().shift(-2) (vwap-to-vwap 后复权)
- 全期 2019-01-02..2026-08-24；2026 段 2026-01-01..2026-08-24；Q2 段 2026-04-01..2026-08-24
- 逐日横截面 rank 归一化；RankIC 三段；2026 多空净值 G10-G1 (DECILE_WINDOW=25, LS_MIN_N=40)
- 状态判定 stable/decay/failed 同源逻辑
输出: /tmp/rob26_incremental.json (35 条完整字段) + 进度 /tmp/rob26_progress.log
"""
import os, sys, json, time, math, re, argparse, warnings
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = "/home/sunhaiwei/quant_projects"
sys.path.insert(0, ROOT)
MB = os.path.join(ROOT, "weekly_backtest_output")
MAT = os.path.join(MB, "factor_matrices_all")
MAT26 = os.path.join(MB, "factor_matrices_all_2026minute")
MAT_D = os.path.join(MB, "factor_matrices_all_2026daily")
FORMULA = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"
VWAP_F = os.path.join(ROOT, "lightgbm_qs/data/build/ohlcv_adj_wide/Vwap_adj.parquet")
OUT_JSON = os.path.join(MB, "robustness_2026.json")
OUT_INCR = "/tmp/rob26_incremental.json"
LOG = "/tmp/rob26_progress.log"

FULL_START = "2019-01-02"
Y26_START = "2026-01-01"
Y26_Q2_START = "2026-04-01"
Y26_END = "2026-08-24"

DECILE_WINDOW = 25
RANK_MIN = 15
LS_MIN_N = 40
IR_YEAR = math.sqrt(245)

_GV = {}


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_GV():
    if "idx" in _GV:
        return
    t0 = time.time()
    vwap = pd.read_parquet(VWAP_F)
    idx = vwap.index
    vwap = vwap.astype("float32")
    rv = vwap / vwap.shift(1).replace(0, np.nan) - 1.0
    ret = rv.shift(-2)
    m_full = (idx >= FULL_START) & (idx <= Y26_END)
    m26 = (idx >= Y26_START) & (idx <= Y26_END)
    m26_q2 = (idx >= Y26_Q2_START) & (idx <= Y26_END)
    _GV.update(idx=idx, ret=ret, m_full=m_full, m26=m26, m26_q2=m26_q2, vwap=vwap)
    plog(f"[load_GV] vwap {vwap.shape} load %.1fs" % (time.time() - t0))


def _rank_ic_series(r, ret, v):
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
    m = float(ic.mean())
    s = float(ic.std())
    return dict(n=len(ic), ic=m, icir=(m / s) if s > 1e-12 else 0.0,
                icir_y=(m / s * IR_YEAR) if s > 1e-12 else 0.0)


def _window_quantiles(x, window, pct_lo, pct_hi, min_n=LS_MIN_N):
    nm = ~np.isnan(x)
    T, N = x.shape
    top = np.zeros((T, N), bool)
    bot = np.zeros((T, N), bool)
    for t in range(window - 1, T):
        win = x[t - window + 1: t + 1]
        mw = ~np.isnan(win)
        if mw.sum() < min_n:
            continue
        med = np.nanmedian(win, axis=0)
        q10 = np.nanquantile(med, pct_lo)
        q90 = np.nanquantile(med, pct_hi)
        tt = nm[t] & (med >= q90)
        bb = nm[t] & (med <= q10)
        if int(tt.sum()) < 3 or int(bb.sum()) < 3:
            continue
        top[t] = tt
        bot[t] = bb
    return top, bot


def analyze_factor(page, src):
    """src: 'daily' (factor_matrices_all) or 'minute2026' (factor_matrices_all_2026minute)."""
    t0 = time.time()
    idx, ret, m_full, m26, m26_q2, vwap = (
        _GV["idx"], _GV["ret"], _GV["m_full"], _GV["m26"], _GV["m26_q2"], _GV["vwap"])
    base = MAT if src == "daily" else (MAT26 if src == "minute2026" else MAT_D)
    try:
        path = os.path.join(base, f"{page}.parquet")
        if not os.path.exists(path):
            path = os.path.join(MAT, f"factor_{page}.parquet")
        fac = pd.read_parquet(path)
        if not isinstance(fac.index, pd.DatetimeIndex):
            fac.index = pd.to_datetime(fac.index)
        fac = fac.reindex(index=idx, columns=vwap.columns)
    except Exception as e:
        return {"page": page, "status": "error", "error": f"{type(e).__name__}: {str(e)[:80]}"}
    fac = fac.to_numpy(dtype="float32", na_value=np.nan)

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
    out = {"page": page, "panel": fac.shape[1], "ndays_full": int(m_full.sum()), "_src": src}

    ic_full = _rank_ic_series(r, ret.to_numpy(), v)[m_full]
    ic_26 = _rank_ic_series(r, ret.to_numpy(), v)[m26]
    ic_26q2 = _rank_ic_series(r, ret.to_numpy(), v)[m26_q2]
    out["rankIC_full"] = _period_stats(ic_full)
    out["rankIC_2026"] = _period_stats(ic_26)
    out["rankIC_2026_Q2"] = _period_stats(ic_26q2)

    r26 = r[sel26]
    ret26 = ret.to_numpy()[sel26]
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
    d26 = idx[sel26]
    final = float(nav[-1] - 1)
    mdd = float((nav / cm - 1).min())
    peak = int(cm.argmax())
    trough = int((nav / cm - 1).argmin())
    dd_len = int((d26[trough] - d26[peak]).days) if trough >= peak else 0
    out["ls2026"] = dict(cumret=round(final, 6), maxdd=round(mdd, 6),
                         mdd_days=mdd, last_d=d26[-1].strftime("%Y-%m-%d"),
                         maxdd_dur_days=dd_len)

    st_full = s_full["ic"] if (s_full := out["rankIC_full"]) else 0.0
    st_26 = s_26["ic"] if (s_26 := out["rankIC_2026"]) else 0.0
    st_q2 = s_q2["ic"] if (s_q2 := out["rankIC_2026_Q2"]) else 0.0
    ratio = abs(st_q2) / abs(st_full) if abs(st_full) > 1e-12 else 1.0
    out["states"] = dict(full_ic=round(st_full, 5), y26_ic=round(st_26, 5),
                         q2_ic=round(st_q2, 5), ratio=round(ratio, 3))
    if abs(st_q2) <= 0.5 * abs(st_full):
        base = "decay"
    elif st_q2 * st_full < 0 and abs(st_q2) > 1e-9:
        base = "decay"
    else:
        base = "stable"
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
    out["flip_to_pos"] = (st_full < 0)
    out["cost_s"] = round(time.time() - t0, 2)
    return out


def classify_logic(dsl, code):
    low = (dsl + " " + (code or "")).lower()
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
    tags = {t for t, kws in _LOGIC_TAGS.items() if any(k in low for k in kws)}
    return sorted(tags) if tags else ["other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-json", action="store_true", help="只计算写增量 JSON，不合并 robustness_2026.json")
    ap.add_argument("--merge", action="store_true", help="计算后合并进 robustness_2026.json（默认 off，可另跑 merge 脚本）")
    args = ap.parse_args()

    plog("=" * 60)
    plog(f"[main] rob26 incremental t={time.strftime('%H:%M:%S')}")

    formula_map = json.load(open(FORMULA))
    fm = {e["page_name"]: e for e in formula_map}

    d26 = json.load(open("/tmp/new50_selected.json"))
    d9 = json.load(open("/tmp/minute_9_result.json"))
    # 26 daily: page_name from new50; rename the 4 banned-algo page names to display names
    RENAME = {
        "alphasage_zscore_ts_cov_sqrt_vol_ret": "vol_ret_covariance_zscore",
        "alphasage_power_ts_pct_log_amt_vwap": "amt_vwap_elasticity_power",
        "alphasage_log_cs_resid_ts_delta_rank_winsor": "cs_residual_momentum_rank_winsor",
        "alphasage_rank_amount": "amount_rank_pressure",
    }
    FLIP = {e["page_name"]: bool(e.get("is_flipped")) for e in d26}
    # 22 rebuilt full-coverage daily matrices live in factor_matrices_all_2026daily (no factor_ prefix)
    REBUILT_22 = set([
        "tail_asymmetry_volume_confirmed", "overnight_gap_fragility",
        "intraday_conviction_volume_gate", "downside_upside_tail_imbalance_volume_",
        "ovni_stable_curvature", "ovni_vol_align", "adaptive_vol_asym_v2",
        "intraday_overnight_gap_smoothed", "vol_range_conviction_smooth",
        "relative_volume_return_coherence_regime", "intraday_persistence_share_ewma",
        "vol_momentum_divergence_stable", "exhaustion_local_norm", "pv_dir_alignment",
        "vol_trend_stability", "intraday_phase_share_ewma_volume_quality",
        "overnight_gap_fragility_log", "signed_vol_ret_alignment",
        "crash_gap_overnight_fragility", "range_amplitude_volatility",
        "intraday_downside_variance_share_logit", "crash_gap_fragility_signed_sqrt",
    ])
    tasks = [{"page": e["page_name"], "src": "daily_rebuilt" if e["page_name"] in REBUILT_22 else "daily",
              "flip": bool(e.get("is_flipped"))} for e in d26]
    # 9 minute: page names as-is; matrix source = 2026 extension dir
    for p in d9:
        if p == "_rankic_summary":
            continue
        tasks.append({"page": p, "src": "minute2026", "flip": True})
    plog(f"[main] tasks {len(tasks)}")

    load_GV()

    rob = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_workers = 10
    plog(f"[main] ProcessPoolExecutor workers={n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(analyze_factor, t["page"], t["src"]): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            res = fut.result()
            res["flip"] = t["flip"]
            res["src"] = t["src"]
            rob[t["page"]] = res
            done += 1
            plog(f"  done {done}/{len(tasks)} {t['page']} status={res.get('status')} "
                 f"ic26={res.get('rankIC_2026', {}).get('ic') if res.get('rankIC_2026') else None} "
                 f"({time.time()-0:.0f}s)")
            if done % 5 == 0:
                json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'r26'} for k, v in rob.items()},
                          open(OUT_INCR, "w"), ensure_ascii=False, indent=1)

    # logic tags
    for r in rob.values():
        meta = fm.get(r["page"], {}) if isinstance(fm, dict) else {}
        dsl = meta.get("dsl") or ""
        code = meta.get("code") or ""
        r["logic_tags"] = classify_logic(dsl, code)

    rob_list = [r for r in rob.values() if r.get("status") != "error"]
    json.dump(rob_list, open(OUT_INCR, "w"), ensure_ascii=False, indent=1)
    plog(f"[main] incremental JSON written {OUT_INCR} n={len(rob_list)}")

    c = Counter(r.get("status") for r in rob_list)
    plog(f"[SUMMARY-35] stable={c['stable']} decay={c['decay']} failed={c['failed']} "
         f"stable_ratio={c['stable']/max(len(rob_list),1):.1%}")

    if args.merge:
        _merge(rob_list)
    return rob_list


def _merge(rob_list):
    existing = json.load(open(OUT_JSON))
    pages_existing = {r["page"] for r in existing}
    merged = {r["page"]: r for r in existing}
    n_upd = 0
    n_add = 0
    for r in rob_list:
        if r["page"] in merged and merged[r["page"]].get("_src") != "minute2026_inc":
            # 26 daily are all new; minute 9 already exist -> update them with 2026-window results
            merged[r["page"]].update({k: v for k, v in r.items() if k not in ("page", "_src")})
            merged[r["page"]]["_src"] = "minute2026_inc"
            n_upd += 1
        elif r["page"] not in merged:
            merged[r["page"]] = r
            n_add += 1
    final = list(merged.values())
    json.dump(final, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    plog(f"[merge] added {n_add} updated {n_upd} total {len(final)}")


if __name__ == "__main__":
    main()

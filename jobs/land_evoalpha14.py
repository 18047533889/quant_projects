#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Land & evaluate the EvoAlpha top-14 factors with the library hard rules.

Stack
-----
- Data          : data_access (ashare_stock_daily) with admin COS mirror root
                  ASHARE_PARQUET_ROOT=$HOME/cos_data (already fully mirrored).
- Factor values : factor_engine runtime (pandas cleaned-operators backend) with
                  all 14 formulas expressed in factor_engine DSL.  `returns`
                  is expressed as StockDailyBar.Return (SourceRef) so the
                  engine reads it as a physical column, NOT a python-side
                  reimplementation.
- Evaluation    : quant_evaluator RankIC / ICIR (vwap-to-vwap) + 2026 + deciles
                  + 2026 long-short drawdown / status (mirrors
                  analyze_2026_robustness semantics).

Outputs
-------
- weekly_backtest_output/evoalpha14/<name>.parquet  (date-symbol wide matrices,
  index 2019-01-02..2026-08-24 aligned to the 456-factor panel)
- weekly_backtest_output/evoalpha14/evoalpha14_meta.json

Environment notes (all set inside this script)
-----------------------------------------------
- DATA_ACCESS_SKIP_COS_MIRROR=1   (local mirror is authoritative; no COS cp)
- DATA_ACCESS_RUN_MODE=interactive_research  (research; keeps mirror lax)
- FACTOR_ENGINE_RUN_MODE=research
Note: DataAccessSource.run_mode is derived from the ENGINE run_mode; the
temporary auth issue in mirror does NOT apply here because the store is
bound with the runtime mode identity by prepare_read.  We set both envs.

Smoke:  python jobs/land_evoalpha14.py --smoke
Full :  python jobs/land_evoalpha14.py
"""
import os, sys, json, time, argparse, warnings, resource
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.join(os.path.expanduser("~"), "cos_data"))
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")

PROJECT = Path("/home/sunhaiwei/quant_projects")
for _p in (str(PROJECT), str(PROJECT / "quant_evaluator"),
           str(PROJECT / "factor_preprocess"), str(PROJECT / "vectorbt_qs"),
           str(PROJECT / "data_access"), str(PROJECT / "jobs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# NOTE: deliberately do NOT add factor_engine/ to sys.path — the repo's
# factor_engine/factor_engine/ nesting would shadow the module; the canonical
# package resolves via the repo-root package directory.

import factor_engine.cleaned_operators  # noqa: F401  (boot the operator registry first; see ParamRole metaclass import-order quirk)
from factor_engine.api.columns import col, field
from factor_engine.api.factor import Factor
from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.backend.factory import build_backend
from factor_engine.storage.factory import build_data_source
from factor_engine.runtime.engine import FactorEngine
from data_access.runtime.mode_identity import set_runtime_mode_identity, reset_runtime_mode_identity

# --------------------------------------------------------------------------- config
FV_OUT = PROJECT / "weekly_backtest_output" / "evoalpha14"
META_PATH = FV_OUT / "evoalpha14_meta.json"
VWAP_ADJ = PROJECT / "lightgbm_qs" / "data" / "build" / "ohlcv_adj_wide" / "Vwap_adj.parquet"
FACTOR_DATES = PROJECT / "weekly_backtest_output" / "factor_dates.parquet"
FULL_START = "2019-01-02"
FULL_END = "2026-08-24"
Y26_START = "2026-01-01"
Y26_END = "2026-08-24"
DECILE_WINDOW = 25
RANK_MIN = 15
LS_MIN_N = 40

NAMES = json.load(open("/tmp/evoalpha_top14.json"))

# --------------------------------------------------------------------------- formula builders (factor_engine DSL)
def _ops():
    o = {}
    for nm in ["rank", "zscore", "ts_mean", "ts_std", "ts_corr", "ts_rank", "ts_delta",
               "ts_sum", "log", "abs", "sign", "clip", "delay", "where", "and_",
               "eq", "lt"]:
        o[nm] = make_cleaned_call_factory(nm)
    return o


def _finite(x):
    """Finite-mask (1/NaN) built from library operators only.

    ``isnan`` is not registered; a finite check is expressed as
    ``(x==x) & (abs(x) < 1e300)`` using eq/and_/abs/lt. Comparison ops
    (``scalar_compare``) flesh invalid cells to NaN, and ``and_`` keeps the
    panel truthiness. Returns 1.0 where finite, NaN where not.
    """
    o = _ops()
    return o["where"](o["and_"](o["eq"](x, x), o["lt"](o["abs"](x), 1e300)), 1.0, 0.0)


def _san(x):
    """Replace +-inf with NaN so cross-sectional stats don't poison whole rows.

    The engine's zscore/rank treat +-inf as valid (finite-only masking is only
    applied in the cs_* statistics kernels), so a single inf in a long window
    makes the row's std NaN.  We sanitize the inf-producing subexpressions:
    within a multivariate product / log / return ratio, inf cells become NaN.
    """
    o = _ops()
    return o["where"](_finite(x), x, 0.0)


def build_formulas():
    o = _ops()
    C = col("AdjClose"); V = col("Volume"); A = col("AdjAmount"); H = col("AdjHigh")
    ret = col("Return")     # raw physical Return column (bp); ranking-invariant scale
    vwap = col("AdjVwap")   # 后复权 AdjVwap（StockDailyBarAdj），复权前数据禁用
    ratio1 = (C / (o["delay"](C, 1) + 1e-6) - 1)
    ratio_clip = o["clip"]((C - o["delay"](C, 1)) / (o["delay"](C, 1) + 1e-6), 0, 2)
    logV = o["log"](V)
    amtm3 = o["ts_sum"](A, 3) + 1e-6
    amtm5 = o["ts_mean"](A, 5); amtm10 = o["ts_mean"](A, 10); amtm20 = o["ts_mean"](A, 20)
    clm5 = o["ts_mean"](C, 5); clm10 = o["ts_mean"](C, 10)
    v5 = o["ts_mean"](V, 5); v10 = o["ts_mean"](V, 10)
    rstd20 = o["ts_std"](ret, 20) + 1e-6
    rstd10 = o["ts_std"](ret, 10) + 1e-6

    F = {
        "volume_price_divergence_momentum_v2":
            o["zscore"](o["ts_corr"](clm10, amtm10, 20)
                        * o["ts_rank"]((0 - 1) * _san(ratio1), 5)
                        * o["ts_delta"](_san(logV), 10)),
        "pv_v7_native_a_00451_hybrid_value_proxy_synthesis":
            o["zscore"](o["ts_corr"](clm5, amtm5, 10)
                        * (o["ts_rank"](ret, 20) + 1e-6)
                        * o["ts_delta"](_san(logV), 10) / (amtm3)
                        + o["ts_rank"](ratio_clip, 20)),
        "pv_v7_native_a_00451_mutated":
            o["zscore"](o["ts_rank"](o["ts_delta"](C, 5) / (rstd20), 3)),
        "volatility_adjusted_hybrid_volume_price_momentum":
            o["rank"](o["ts_corr"](clm10, v10, 20)
                      * (o["ts_rank"](ret, 20) + 1e-6) / (rstd20)),
        "pv_v7_native_a_00451_mutated_reversal":
            o["zscore"](o["ts_corr"](o["ts_mean"](ret, 10), amtm10, 20)
                        / (rstd20)
                        * o["ts_rank"]((0 - 1) * ret, 5)),
        "pv_v7_native_a_00451_mutated_490b74":
            o["zscore"](o["ts_delta"](C, 3) / (amtm3)),
        "pv_v4_a_0062_momentum":
            o["rank"](o["ts_delta"](C, 5) / (o["ts_mean"](ret, 10) + 0.001)),
        "hybrid_factor_momentum":
            o["rank"](o["ts_corr"](o["ts_delta"](C, 1), v5, 10)
                      / (o["ts_std"](ret, 20) + 0.001)),
        "pv_vol_liquidity_momentum_hybrid":
            o["zscore"](o["ts_rank"](o["ts_corr"](C, V, 10) / (amtm20), 3)),
        "Alpha158_Volume_Price_Diversity_Enhanced_Mutated":
            o["rank"](o["ts_corr"](H, V, 10) / (o["ts_std"](ret, 20) + 0.001))
            * o["ts_mean"](o["sign"](ret), 5),
        "Alpha158_VOLATILITY_RANK":
            o["rank"](o["ts_std"](ret, 10) / (o["ts_mean"](ret, 20) + 0.001)),
        "pv_vol_liquidity_divergence":
            o["ts_rank"](o["ts_corr"](C, V, 10) / (o["ts_mean"](A, 5) + 1e-6), 20)
            * o["zscore"](o["ts_delta"](C, 5) / (rstd10)),
        "pv_v7_native_a_00451_fusion_hybrid_value_proxy":
            o["zscore"](o["ts_corr"](clm5, amtm5, 10)
                        * o["ts_rank"](o["abs"](ret), 10) / (o["ts_mean"](vwap, 3) + 1e-6)),
        "pv_v7_native_a_00451_hybrid_value_proxy_fusion":
            o["zscore"](o["ts_corr"](clm5, amtm5, 10)
                        * (o["ts_rank"](ratio_clip, 20)
                           / (amtm3))),
    }
    return F


# --------------------------------------------------------------------------- evaluation (vwap-to-vwap)
def load_vwap_ret():
    v = pd.read_parquet(VWAP_ADJ)
    idx = v.index
    assert idx[0] == pd.Timestamp("2016-01-04") and idx[-1] == pd.Timestamp("2026-08-24")
    rv = v / v.shift(1).replace(0, np.nan) - 1.0
    ret = rv.shift(-2)                    # vwap-to-vwap 后复权（t+1成交→t+2卖出，企业级，对齐TargetVwapReturnH01）
    return v.astype("float32"), ret


def _rank_ic_series(r, ret, v):
    cnt = v.sum(1)
    R = np.where(v, r, 0.0)
    Ret = np.where(v, ret, 0.0)
    n = np.where(cnt == 0, 1, cnt)
    mR = np.nansum(np.where(v, R, np.nan), 1) / n
    mRet = np.nansum(np.where(v, Ret, np.nan), 1) / n
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
                icir_y=(m / s * np.sqrt(245)) if s > 1e-12 else 0.0)


def _window_quantiles(x, window, pct_lo, pct_hi, min_n=LS_MIN_N):
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


def analyze_mat(mat, vwap_adj, ret_full, idx, name):
    fac = mat.reindex(index=idx, columns=vwap_adj.columns).to_numpy(dtype="float32", na_value=np.nan)
    order = np.argsort(vwap_adj.to_numpy(), axis=1)
    r = np.empty_like(fac); v = ~np.isnan(fac)
    r[v] = order[v]
    r = np.where(v, r / (r.shape[1] - 1), np.nan)
    ret_a = ret_full.to_numpy()

    m_full = (idx >= FULL_START) & (idx <= FULL_END)
    m26 = (idx >= Y26_START) & (idx <= Y26_END)
    m26q2 = (idx >= pd.Timestamp("2026-04-01")) & (idx <= Y26_END)

    ic_full = _rank_ic_series(r, ret_a, v)[m_full]
    ic_26 = _rank_ic_series(r, ret_a, v)[m26]
    ic_26q2 = _rank_ic_series(r, ret_a, v)[m26q2]
    s_full = _period_stats(ic_full); s_26 = _period_stats(ic_26); s_q2 = _period_stats(ic_26q2)

    sel26 = (idx >= pd.Timestamp(Y26_START)) & (idx <= pd.Timestamp(Y26_END))
    r26 = r[sel26]; ret26 = ret_a[sel26]
    top, bot = _window_quantiles(r26, DECILE_WINDOW, 0.10, 0.90)
    ls = np.zeros(top.shape[0]); rr26 = pd.DataFrame(ret26)
    for d in range(top.shape[0]):
        tt = np.nan_to_num(rr26.values[d][top[d]], nan=0.0)
        bb = np.nan_to_num(rr26.values[d][bot[d]], nan=0.0)
        if len(tt) and len(bb):
            ls[d] = float(np.nanmean(tt)) - float(np.nanmean(bb))
    nav = (1 + ls).cumprod()
    cm = np.maximum.accumulate(nav)
    final = float(nav[-1] - 1)
    mdd = float((nav / cm - 1).min())
    peak = int(cm.argmax()); trough = int((nav / cm - 1).argmin())
    d26 = idx[sel26]
    dd_len = int((d26[trough] - d26[peak]).days) if trough >= peak else 0

    st_full = s_full["ic"] if s_full else 0.0
    st_26 = s_26["ic"] if s_26 else 0.0
    st_q2 = s_q2["ic"] if s_q2 else 0.0
    base = "decay" if abs(st_q2) <= 0.5 * abs(st_full) else "stable"
    if st_q2 * st_full < 0 and abs(st_q2) > 1e-9:
        base = "decay"
    if final < 0 and mdd < -0.15:
        base = "failed"
    elif final < 0 and base == "stable":
        base = "decay"
    elif mdd < -0.30 and base != "failed":
        base = "decay"
    if base == "decay" and mdd <= -0.35:
        base = "failed"

    return {
        "name": name,
        "rankIC_full": s_full, "rankIC_2026": s_26, "rankIC_2026_Q2": s_q2,
        "ls2026": dict(cumret=final, maxdd=mdd, maxdd_dur_days=dd_len,
                       last_d=d26[-1].strftime("%Y-%m-%d")),
        "status": base,
        "panel": fac.shape,
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="", help="one factor only")
    args = ap.parse_args()
    smoke = args.smoke
    FV_OUT.mkdir(parents=True, exist_ok=True)

    formulas = build_formulas()
    t0 = time.time()
    vwap_adj, ret_full = load_vwap_ret()
    idx = vwap_adj.index
    print(f"[init] vwap {vwap_adj.shape}  loaded in {time.time()-t0:.1f}s", flush=True)

    tok = set_runtime_mode_identity("interactive_research", source="land_evo14")
    meta = {}
    for nm, ex in formulas.items():
        if args.name and nm != args.name:
            continue
        if smoke and nm != "volume_price_divergence_momentum_v2":
            continue
        t1 = time.time()
        try:
            ds = build_data_source({
                "type": "data_access",
                "dataset": "ashare_stock_daily_adj",  # 后复权表（2026-08-28 硬性）；col(Close/Vwap) → AdjClose/AdjVwap
                "start_date": FULL_START,
                "end_date": FULL_END,
                # leave instrument_filter=None -> whole universe (mirror is complete).
            })
            eng = FactorEngine(build_backend("pandas"), ds, run_mode="research")
            f = Factor(name=nm, expr=ex)
            out = eng.run(f, market="ashare")
            res = out["result"]                      # MultiIndex Series
            mat = res.unstack(level=res.index.names[-1])
            mat = mat.reindex(index=idx).astype("float32")
            # trim pre-2019 warmup rows
            mat = mat.loc[mat.index >= pd.Timestamp(FULL_START)]
            nonnull = float(mat.notna().mean().mean()) if mat.size else 0.0
            print(f"[run] {nm} {mat.shape} nonnull={nonnull:.3f} t={time.time()-t1:.1f}s", flush=True)
            if not smoke and mat.shape[1] and mat.notna().any().any():
                mat.to_parquet(FV_OUT / f"{nm}.parquet")
            meta[nm] = analyze_mat(mat, vwap_adj, ret_full, idx, nm)
            meta[nm]["shape"] = list(mat.shape)
            meta[nm]["nonnull"] = round(nonnull, 4)
        except Exception as e:
            print(f"[fail] {nm}: {type(e).__name__} {str(e)[:200]}", flush=True)
            meta[nm] = {"name": nm, "error": f"{type(e).__name__}: {str(e)[:180]}"}

    reset_runtime_mode_identity(tok)

    if not smoke:
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"\n[done] {sum(1 for v in meta.values() if 'error' not in v)} ok / {len(formulas)} "
          f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    for nm, m in meta.items():
        if "error" not in m:
            s = m.get("rankIC_full") or {}
            s26 = m.get("rankIC_2026") or {}
            print(f"  {nm[:44]:46s} IC {s.get('ic', float('nan')):+.4f} IR {s.get('icir', 0):+.2f} "
                  f"2026IC {s26.get('ic', float('nan')):+.4f} status {m.get('status','?')}", flush=True)


if __name__ == "__main__":
    main()
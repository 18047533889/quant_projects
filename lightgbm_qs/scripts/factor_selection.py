# -*- coding: utf-8 -*-
"""Walk-forward factor selection (fixes P0-B full-sample selection leakage).

ORIGINAL BUG (P0-B problem 1): the old factor_selection.py computed rank_IC over ALL
dates for every factor, then kept rank_ic > 0.015, writing data/build/selected_factors.csv.
That full-sample list fed the downstream walk-forward LightGBM training, so features
were selected using FUTURE OOS labels (selection leakage).

THIS REWORK: selection is itself walk-forward. For each fold (cut date) the rank-IC /
coverage stats are computed ONLY on the train/validation portion (dates < cut), and the
last PURGE_TRADING_DAYS of that selection set are also dropped so no selection label
window reaches into the OOS block. That fold's factor set is frozen for that fold. A
guard asserts the OOS block is never seen when computing a fold's selection stats.

CLI
---
  python factor_selection.py                          # legacy full-sample list (back-compat)
  python factor_selection.py --fold 2019-04-01        # per-fold walk-forward list
  python factor_selection.py --folds-from-train       # same cuts as scripts/train_final.py

Only walk-forward (--fold / --cuts / --folds-from-train) mode is the research-correct one.
"""
import glob
import os
import argparse
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")

# 10-day forward label (fwd_ret10 = Vwap.pct_change().shift(-1).cumprod-1 style):
# a selection/train sample at date t sees Vwap at trading index t+10. To never let a
# label window reach into the OOS block starting at the cut, drop the last
# PURGE_TRADING_DAYS trading days before the cut from BOTH selection and train.
PURGE_TRADING_DAYS = 10
EMBARGO_TRADING_DAYS = 0       # layered on top of the purge; default 0

# Per-fold selection handoff: data/build/walkforward_selection.json
WF_SELECTION_JSON = os.path.join(BUILD, "walkforward_selection.json")

# ---- selection gates ---------------------------------------------------------
RANK_IC_THRESHOLD = 0.015      # rank-IC keep gate (applied per fold, train only)
MIN_N_DATES = 1500             # legacy full-sample dense-coverage floor
MIN_FOLD_N_DATES = 200         # per-fold minimum #selection dates for a factor
MIN_N_DATES_FRAC = 0.5         # per-fold minimum = frac * (#selection dates)

trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad.parquet").columns)
fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")  # DatetimeIndex 'date'


# ---------------------------------------------------------------------- loading
def load_full(pool):
    """Return list of (factor_name, long df) for factors with full coverage in a pool."""
    out = []
    for f in glob.glob(f"{ROOT}/data/factor_pools/{pool}/*.parquet"):
        try:
            df = pq.read_table(f, columns=["datetime", "asset", "factor_value"]).to_pandas()
        except Exception:
            continue
        dmin, dmax = str(df.datetime.min())[:10], str(df.datetime.max())[:10]
        if not (dmin <= "2016-03-01" and dmax >= "2024-01-01"):
            continue
        name = os.path.basename(f).replace("_neu.parquet", "").replace(".parquet", "")
        out.append((name, df[df.asset.isin(trad)]))
    return out


def fwd_long():
    """Long form of fwd_ret10 -> DataFrame(date, asset, fwd) with datetime dates."""
    f = fwd.stack().rename("fwd").reset_index()
    f.columns = ["date", "asset", "fwd"]
    f["date"] = pd.to_datetime(f["date"])
    return f


def _factor_long(df):
    """Normalise a factor long df to DataFrame(date, asset, fv) with datetime dates.

    Accepts the on-disk schema (datetime, asset, factor_value) and the
    minimal schema (date, asset, fv) used by tests.
    """
    d = df.copy()
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"])
        d["date"] = pd.to_datetime(d["datetime"].dt.date)
    else:
        d["date"] = pd.to_datetime(d["date"])
    if "factor_value" in d.columns:
        d = d.rename(columns={"factor_value": "fv"})
    return d[["date", "asset", "fv"]]


# ------------------------------------------------------------------ selection
def selection_dates_for_cut(all_dates, cut, purge=PURGE_TRADING_DAYS,
                            embargo=EMBARGO_TRADING_DAYS):
    """Train/validation selection dates for a fold, purged of the label-window tail.

    Returns the sorted set of dates < cut minus the last (purge + embargo) trading
    days, so that no selection label (10d fwd) reaches into the OOS block (>= cut).
    Accepts a Series or a DatetimeIndex of unique sorted dates.
    """
    all_dates = pd.Series(pd.to_datetime(pd.Index(all_dates))).sort_values().reset_index(drop=True)
    sel = all_dates[all_dates < cut]
    drop = purge + embargo
    # drop exactly (purge + embargo) trading dates immediately before the cut:
    # the label window of the last kept sample ends at position cut_pos - 1,
    # strictly before the OOS block starting at cut_pos (label horizon == purge).
    if len(sel) <= drop:
        return []
    return sel.iloc[:-drop]


def rank_ic_series(fv, fwd_dt, sel_dates):
    """Cross-sectional rank IC per date, computed ONLY on dates in sel_dates.

    Returns (IC Series indexed by date, n_dates_used).
    """
    m = fv.merge(fwd_dt[["date", "asset", "fwd"]], on=["date", "asset"], how="inner")
    m = m[m["date"].isin(sel_dates)]          # <-- the anti-leak filter
    if m["date"].nunique() < 2 or len(m) < 1000:
        return pd.Series(dtype=float), 0
    ic = m.groupby("date").apply(
        lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False
    )
    ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return ic, len(ic)


def select_factors_walk_forward(factors, fwd_dt, cuts,
                                rank_ic_thr=RANK_IC_THRESHOLD,
                                purge=PURGE_TRADING_DAYS,
                                min_fold_n_dates=MIN_FOLD_N_DATES,
                                min_dates_frac=MIN_N_DATES_FRAC):
    """Walk-forward factor selection (anti-leak).

    For each fold the selection statistics use ONLY dates in [train start, cut - purge]
    (never the OOS block, and never the purge tail whose labels would look into OOS).

    Returns (fold_factors, report):
      fold_factors : {pd.Timestamp cut: [factor_name, ...]}  frozen per-fold list.
      report       : DataFrame(cut, factor, rank_ic, n_dates) on selection-set only.
    """
    all_dates = pd.Series(pd.to_datetime(fwd_dt["date"].unique())).sort_values().reset_index(drop=True)
    fold_factors = {}
    report = []
    for cut in cuts:
        cut = pd.Timestamp(cut)
        sel_dates = selection_dates_for_cut(all_dates, cut)
        if len(sel_dates) == 0:
            fold_factors[cut] = []
            continue

        # ---- GUARD: selection must NEVER see OOS (>= cut) or the purge tail -------
        assert bool((sel_dates < cut).all()), \
            f"[fold {cut.date()}] selection leaks into OOS block (>= cut)"
        # the last selection date's label window ends at its index + purge, which
        # must be strictly before the cut (guard catches an over-long label)
        max_sel_idx = all_dates[all_dates.isin(sel_dates)].index.max()
        assert max_sel_idx + purge < len(all_dates) and all_dates.iloc[max_sel_idx + purge] < cut, \
            f"[fold {cut.date()}] last selection label window reaches OOS (>= cut)"

        sel_set = set(sel_dates)
        min_dates = max(min_fold_n_dates, int(min_dates_frac * len(sel_dates)))
        chosen = []
        for name, df in factors:
            fv = _factor_long(df)
            ic, n = rank_ic_series(fv, fwd_dt, sel_set)
            if n < min_dates:
                continue
            rank_ic = float(ic.mean()) if len(ic) else float("nan")
            if not np.isfinite(rank_ic):
                continue
            report.append({"cut": cut.date(), "factor": name,
                           "rank_ic": rank_ic, "n_dates": n})
            if rank_ic > rank_ic_thr:
                chosen.append(name)
        fold_factors[cut] = chosen
    rep = pd.DataFrame(report)
    return fold_factors, rep


# ------------------------------------------------------- per-fold selection manifest
def write_selection_manifest(fold_factors, cuts, purge=PURGE_TRADING_DAYS,
                             embargo=EMBARGO_TRADING_DAYS,
                             label_basis="vwap_to_vwap_fwd10",
                             rank_ic_thr=RANK_IC_THRESHOLD,
                             min_fold_n_dates=MIN_FOLD_N_DATES,
                             min_dates_frac=MIN_N_DATES_FRAC,
                             all_dates=None, path=None):
    """Write the per-fold selection handoff consumed by feature builders + trainers.

    The manifest is the ONLY thing downstream feature-build / training scripts are
    allowed to read for choosing features per fold — the full-sample
    selected_factors*.csv lists are leakage-carrying diagnostics, never inputs.

    Structure of the JSON:
      { purge_trading_days, embargo_trading_days, label_basis, rank_ic_threshold,
        cuts: { "<cut iso date>": { sel_start, sel_end, n_sel_dates, factors: [...] } } }

    all_dates: optional sorted trading-date index; defaults to the fwd_ret10 calendar.
    """
    import json
    cuts = [pd.Timestamp(c) for c in cuts]
    if all_dates is None:
        all_dates = pd.Series(pd.to_datetime(pd.Index(fwd.index))).sort_values().reset_index(drop=True)
    else:
        all_dates = pd.Series(pd.to_datetime(pd.Index(all_dates))).sort_values().reset_index(drop=True)
    payload = {
        "purge_trading_days": int(purge),
        "embargo_trading_days": int(embargo),
        "label_basis": label_basis,
        "rank_ic_threshold": float(rank_ic_thr),
        "min_fold_n_dates": int(min_fold_n_dates),
        "min_dates_frac": float(min_dates_frac),
        "cuts": {},
    }
    for cut in cuts:
        lst = list(fold_factors.get(cut, []))
        sel_dates = selection_dates_for_cut(all_dates, cut, purge=purge,
                                            embargo=embargo) if lst else []
        if len(sel_dates):
            sel_start = str(pd.Timestamp(sel_dates.iloc[0]).date())
            sel_end = str(pd.Timestamp(sel_dates.iloc[-1]).date())
            n_sel = int(len(sel_dates))
        else:
            sel_start = sel_end = None
            n_sel = 0
        payload["cuts"][str(cut.date())] = {
            "sel_start": sel_start,
            "sel_end": sel_end,
            "n_sel_dates": n_sel,
            "factors": lst,
        }
    path = path or WF_SELECTION_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[wf] SAVED selection manifest {path} ({len(cuts)} cuts)")
    return path


def load_selection_manifest(path=None, cuts=None):
    """Read the per-fold selection manifest written by write_selection_manifest.

    Returns ({cut_iso_date: [factor, ...]}, meta dict). Raises if the file is absent,
    if its purge/label-basis don't match the current constants, or if a requested cut
    has an empty factor list. Downstream must never fall back to the full-sample list.
    """
    import json
    path = path or WF_SELECTION_JSON
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python factor_selection.py --folds-from-train` first; "
            "the full-sample selected_factors.csv is NOT a valid substitute (selection leakage)")
    with open(path) as f:
        payload = json.load(f)
    if payload.get("purge_trading_days") != PURGE_TRADING_DAYS:
        raise ValueError(
            f"selection manifest purge={payload.get('purge_trading_days')} != PURGE_TRADING_DAYS="
            f"{PURGE_TRADING_DAYS} — regenerate the manifest with the current label horizon")
    if payload.get("label_basis") != "vwap_to_vwap_fwd10":
        raise ValueError(
            f"selection manifest label_basis={payload.get('label_basis')!r} is not the mandated "
            "vwap-to-vwap basis (Vwap.pct_change() 10d forward) — refusing to consume it")
    folds = {k: list(v.get("factors", [])) for k, v in payload.get("cuts", {}).items()}
    if cuts is not None:
        missing = [str(pd.Timestamp(c).date()) for c in cuts if str(pd.Timestamp(c).date()) not in folds]
        if missing:
            raise KeyError(f"selection manifest is missing cuts {missing} — regenerate it")
    empty = [k for k, v in folds.items() if not v]
    if empty:
        raise ValueError(f"selection manifest has empty factor lists for cuts {empty} — "
                         "selection gates produced nothing; fix selection before training")
    return folds, {k: v for k, v in payload.items() if k != "cuts"}


def select_factors_full(factors, fwd_dt, rank_ic_thr=RANK_IC_THRESHOLD,
                        min_n_dates=MIN_N_DATES):
    """Legacy full-sample selection (backward compatibility only; has leakage)."""
    res = []
    for name, df in factors:
        fv = _factor_long(df)
        ic, n = rank_ic_series(fv, fwd_dt, set(fwd_dt["date"]))
        if n < min_n_dates:
            continue
        rank_ic = float(ic.mean()) if len(ic) else float("nan")
        res.append({"factor": name, "rank_ic": rank_ic, "n_dates": n})
    dfres = pd.DataFrame(res)
    sel = dfres[dfres["rank_ic"] > rank_ic_thr].sort_values("rank_ic", ascending=False)
    return dfres, sel


# --------------------------------------------------------------------- CLI
def _train_cuts_from_train_final():
    """Replicate the expanding-window cut schedule from scripts/train_final.py."""
    start = pd.Timestamp(pd.to_datetime(fwd.index).min())
    first_cut = start + pd.DateOffset(months=36)
    oos_end = pd.Timestamp(pd.to_datetime(fwd.index).max())
    cuts, c = [], first_cut
    while c <= oos_end:
        cuts.append(c)
        c += pd.DateOffset(months=3)
    return cuts


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Walk-forward LightGBM factor selection")
    ap.add_argument("--fold", type=str, default=None,
                    help="single cut date YYYY-MM-DD -> per-fold walk-forward selection")
    ap.add_argument("--cuts", type=str, default=None,
                    help="comma-separated cut dates -> per-fold walk-forward selection")
    ap.add_argument("--folds-from-train", action="store_true",
                    help="use the same cuts as scripts/train_final.py (36m init, 3m roll)")
    ap.add_argument("--out", type=str, default=None,
                    help="output CSV path (default data/build/selected_factors*.csv)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(BUILD, exist_ok=True)

    fm247 = load_full("fm247")
    fmqa = load_full("fmqa")
    print(f"full-coverage: fm247={len(fm247)} fmqa={len(fmqa)}")

    seen, factors = set(), []
    for name, df in fm247 + fmqa:
        if name in seen:
            continue
        seen.add(name)
        factors.append((name, df))
    print("unique factors after cross-pool dedup:", len(factors))

    fwd_dt = fwd_long()

    walkforward = args.fold or args.cuts or args.folds_from_train
    if args.fold:
        cuts = [pd.Timestamp(args.fold)]
    elif args.cuts:
        cuts = [pd.Timestamp(c.strip()) for c in args.cuts.split(",") if c.strip()]
    elif args.folds_from_train:
        cuts = _train_cuts_from_train_final()
    else:
        cuts = None

    if cuts is not None:
        fold_factors, rep = select_factors_walk_forward(factors, fwd_dt, cuts)
        for cut in cuts:
            cut = pd.Timestamp(cut)
            print(f"[wf] cut={cut.date()} selected={len(fold_factors[cut])}")
        for cut, lst in fold_factors.items():
            csv = os.path.join(BUILD, f"selected_factors_fold_{cut.date()}.csv")
            pd.DataFrame({"factor": lst}).to_csv(csv, index=False)
            print(f"[wf] SAVED {csv}")
        # the manifest is the authoritative per-fold handoff for feature builders +
        # trainers (full-sample selected_factors.csv is a leakage-carrying diagnostic)
        write_selection_manifest(fold_factors, cuts, all_dates=fwd_dt["date"].unique())
        rep_csv = os.path.join(BUILD, "rankic_walkforward_report.csv")
        if len(rep):
            rep.to_csv(rep_csv, index=False)
            print(f"[wf] SAVED on-selection-set report {rep_csv}")
        return

    # legacy full-sample (kept for compatibility only)
    full, sel = select_factors_full(factors, fwd_dt)
    print("factors with full panel coverage (n_dates>=%d):" % MIN_N_DATES, len(full))
    print("selected (rank_ic>%g):" % RANK_IC_THRESHOLD, len(sel))
    out = args.out or os.path.join(BUILD, "selected_factors.csv")
    sel.to_csv(out, index=False)
    print("top selected:")
    print(sel.head(15).to_string())


if __name__ == "__main__":
    main()

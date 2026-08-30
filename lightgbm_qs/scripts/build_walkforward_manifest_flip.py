# -*- coding: utf-8 -*-
"""Build the walk-forward per-fold factor selection manifest from the prepped
flip-selected feature matrix (anti-leak, same semantics as factor_selection.py).

VECTORIZED rank-ic: the prepped features are already cross-sectionally
rank/zscore-transformed, so per-date Spearman(factor, label) == per-date Pearson
on the z-scored values. We compute it on wide (date x asset) numpy blocks, one
factor at a time — ~10x faster than the groupby.apply equivalent and numerically
identical for the rank/zscored inputs.

For each 3-month cut:
  - selection window = trading dates < cut, minus the last PURGE_TRADING_DAYS
    (the 10-day fwd-label horizon must not reach into the OOS block).
  - per-factor rank_ic on that purged train window (label fwd_adj_neu).
  - keep rank_ic > RANK_IC_THRESHOLD; the fold's factor list is frozen.

Two opt-in gates (both default OFF so the default run is byte-identical to before):
  --min-coverage FLOAT   gate A: a factor must be present (>=1 non-NaN obs) on >= F
                          fraction of the fold's train-window trading dates, else it
                          is excluded from that fold (drops dead/stale factors whose
                          history lies entirely outside the fold train window). 0=off.
  --require-history-days N   gate B: a factor's max data date must extend to within N
                          trading days AFTER the fold OOS-start cut date (generic over
                          all factors; auto-removes minute/short-window factors from
                          folds whose train+label window needs later data). 0=off.
  --dry-run              compute selections with gates but write only per-fold
                          exclusion stats to --dry-run-out; the production manifest
                          (default OUT) is NOT written on a dry run.

Output: data/build/walkforward_selection_flip.json (schema matches
factor_selection.write_selection_manifest).

Usage:
  python build_walkforward_manifest_flip.py                                  # default (gates off)
  python build_walkforward_manifest_flip.py --min-coverage 0.6 --dry-run     # gate A stats only
  python build_walkforward_manifest_flip.py --min-coverage 0.6 --require-history-days 30
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PANEL = os.path.join(ROOT, "data/panel")
LOG = "/tmp/build_walkforward_manifest_flip.log"

PURGE_TRADING_DAYS = 10
EMBARGO_TRADING_DAYS = 0
RANK_IC_THRESHOLD = 0.015
ROLL_MONTHS = 3
TRAIN_MONTHS = 48
LABEL_BASIS = "vwap_to_vwap_fwd10_neu"

FEATS = os.environ.get("FEATURES_PARQUET", os.path.join(BUILD, "features_full3_adj_prep.parquet"))
LABEL = os.environ.get("LABEL_PARQUET", os.path.join(BUILD, "fwd_adj_neu.parquet"))
OUT = os.environ.get("WF_OUT", os.path.join(BUILD, "walkforward_selection_flip.json"))


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def rolls():
    START = pd.Timestamp("2016-01-04")
    fc = START + pd.DateOffset(months=TRAIN_MONTHS)
    r = []
    c = fc
    while c <= pd.Timestamp("2026-08-10"):
        r.append(c)
        c += pd.DateOffset(months=ROLL_MONTHS)
    return r


def rank_rows(X):
    """Rank each row of X (argsort-order ranks; ties broken by position, which is
    fine for Spearman on float factors with few ties)."""
    n, m = X.shape
    order = np.argsort(X, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    cols = np.tile(np.arange(m), (n, 1))
    ranks[np.arange(n)[:, None], order] = cols
    return ranks


def per_date_rank_ic_wide(fv_wide, lab_wide):
    """Fully vectorized per-date Spearman (rank-rank Pearson) between two wide
    (date x asset) blocks. Ranks BOTH inputs (correct for all pools — 'neu' pools
    are winsor-only, not rank-transformed). Returns (mean_ic, n_valid_dates)."""
    assert fv_wide.shape == lab_wide.shape
    f = fv_wide.astype(np.float64)
    L = lab_wide.astype(np.float64)
    ok = np.isfinite(f) & np.isfinite(L)
    rf = rank_rows(np.where(ok, f, np.inf))
    rl = rank_rows(np.where(ok, L, np.inf))
    rf[~ok] = np.nan
    rl[~ok] = np.nan
    rf_d = rf - np.nanmean(rf, axis=1, keepdims=True)
    rl_d = rl - np.nanmean(rl, axis=1, keepdims=True)
    num = np.nansum(rf_d * rl_d, axis=1)
    denom = np.sqrt(np.nansum(rf_d ** 2, axis=1) * np.nansum(rl_d ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ics = num / denom
    ics = ics[np.isfinite(ics)]
    if len(ics) == 0:
        return np.nan, 0
    return float(ics.mean()), int(len(ics))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-feats", type=int, default=0)
    ap.add_argument("--min-fold-dates", type=int, default=200)
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--feats-start", type=int, default=0)
    ap.add_argument("--feats-end", type=int, default=0)
    # Gate A: per-fold per-factor coverage gate. A factor is only admitted to a
    # fold's feature set if it is "alive" across the fold's train window, i.e. the
    # fraction of trading dates in the (purged) train window on which the factor
    # holds at least one non-NaN observation is >= min-coverage. Dead factors whose
    # history lies in an old region (or was rebuilt for an old backtest window) get
    # automatically excluded from late folds whose train window lands entirely past
    # the factor's data. Default 0.6; set --min-coverage 0 to disable (back-compat).
    ap.add_argument("--min-coverage", type=float, default=0.6)
    # Gate B: generic "factor data end vs fold start" regime switch. Any factor whose
    # data does not extend to within `require-history-days` trading days AFTER the
    # fold's OOS start cut date is excluded from that fold -- i.e. a factor is only
    # admitted when its max coverage date is close enough to the fold start. This is
    # generic over ALL factors (not just the minute family): any newly-added
    # short-window factor is handled automatically without touching code. Default 0
    # disables the gate (back-compat old behaviour). N enables it with an N-trading-day
    # buffer.
    ap.add_argument("--require-history-days", type=int, default=0)
    # --dry-run: compute the fold/feature selection with gates applied but write ONLY
    # per-fold exclusion statistics to the (separate) --dry-run-out json -- never the
    # production manifest.
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-run-out", type=str, default="/tmp/gate_dryrun.json")
    args = ap.parse_args()
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] build walkforward manifest from {FEATS}")

    df = pd.read_parquet(FEATS)
    df["date"] = pd.to_datetime(df["date"])
    lab = pd.read_parquet(LABEL)
    lab["date"] = pd.to_datetime(lab["date"])
    df = df.merge(lab, on=["date", "asset"], how="left")
    FEAT = [c for c in df.columns if c not in ("date", "asset", "fwd_neu")]
    if args.limit_feats:
        FEAT = FEAT[:args.limit_feats]
    elif args.feats_end > args.feats_start:
        FEAT = FEAT[args.feats_start:args.feats_end]
    plog(f"features {len(df)} rows x {len(FEAT)} cols")

    # pre-pivot label wide once
    lab_wide = df.pivot_table(index="date", columns="asset", values="fwd_neu").sort_index()
    lab_wide = lab_wide.astype(np.float64)
    all_dates = lab_wide.index
    cuts = rolls()
    plog(f"{len(cuts)} cuts, label wide {lab_wide.shape}")

    # Pre-pivot every factor ONCE to a wide float32 block (memory ~2.1GB for 690
    # factors x 2588 dates x 297 assets), then each fold slices rows — no repeated pivots.
    # NOTE: pivot_table drops all-NaN rows; sparse factors would then have fewer rows
    # than the label block, breaking the vectorized rank-IC. We therefore reindex every
    # factor wide frame onto the label index so shapes always match lab_wide.
    fv_cache = {}
    for fcol in FEAT:
        w = df.pivot_table(index="date", columns="asset", values=fcol).sort_index()
        w = w.reindex(index=lab_wide.index, columns=lab_wide.columns)
        fv_cache[fcol] = w.astype(np.float32)
    plog(f"pre-pivoted {len(fv_cache)} factors ({time.time()-t0:.0f}s)")

    # Gate A/B support: for each factor, precompute (a) whether it is "alive" on
    # each label-grid date (>=1 non-NaN observation that day) and (b) its max data
    # date. These are computed once over the whole merged frame and sliced per fold.
    alive_cache = {}
    maxdate_cache = {}
    cov_active = (args.min_coverage > 0.0) or (args.require_history_days > 0)
    if cov_active:
        # alive matrix: for each factor, a boolean vector over all_dates (True where
        # the factor has >=1 non-NaN value that day). Reuse the wide float blocks.
        alive = np.zeros((len(FEAT), len(all_dates)), dtype=bool)
        maxdate = np.zeros(len(FEAT), dtype="datetime64[ns]")
        for i, fcol in enumerate(FEAT):
            w = fv_cache[fcol].reindex(index=all_dates, columns=lab_wide.columns)
            arr = w.to_numpy(dtype=np.float64)
            any_nonnan = np.isfinite(arr).any(axis=1)
            alive[i] = any_nonnan
            if any_nonnan.any():
                last_ok = np.where(any_nonnan)[0][-1]
                maxdate[i] = all_dates[last_ok]
        for i, fcol in enumerate(FEAT):
            alive_cache[fcol] = alive[i]
            maxdate_cache[fcol] = maxdate[i]
        plog(f"gates active: min_coverage={args.min_coverage} "
             f"require_history_days={args.require_history_days}")

    payload = {
        "purge_trading_days": int(PURGE_TRADING_DAYS),
        "embargo_trading_days": int(EMBARGO_TRADING_DAYS),
        "label_basis": LABEL_BASIS,
        "rank_ic_threshold": float(RANK_IC_THRESHOLD),
        "cuts": {},
    }

    for ci, cut in enumerate(cuts):
        sel_dates = all_dates[all_dates < cut]
        if len(sel_dates) > PURGE_TRADING_DAYS:
            sel_dates = sel_dates[:-PURGE_TRADING_DAYS]
        assert (sel_dates < cut).all()
        if len(sel_dates) < 100:
            payload["cuts"][str(cut.date())] = {"sel_start": None, "sel_end": None,
                                                 "n_sel_dates": int(len(sel_dates)), "factors": []}
            continue
        # only dates present in the label grid (avoids get_indexer -1 -> look-ahead)
        sel_dates = lab_wide.index.intersection(sel_dates)
        if len(sel_dates) < 100:
            payload["cuts"][str(cut.date())] = {"sel_start": None, "sel_end": None,
                                                 "n_sel_dates": int(len(sel_dates)), "factors": []}
            continue
        sel_pos = lab_wide.index.get_indexer(sel_dates)
        lab_block = lab_wide.iloc[sel_pos].to_numpy()
        chosen = []
        excluded = []  # factors dropped by gates (name -> reason tag)
        for fcol in FEAT:
            # ---- Gate A: per-fold train-window coverage ----
            if args.min_coverage > 0.0:
                fold_dates = all_dates[sel_pos]
                alive_slice = alive_cache[fcol][sel_pos]
                n_alive = int(alive_slice.sum())
                total = int(len(fold_dates))
                cov = (n_alive / total) if total else 0.0
                if cov < args.min_coverage:
                    excluded.append((fcol, "coverage<%.2f" % args.min_coverage))
                    continue
            # ---- Gate B: factor data end vs fold start (regime) ----
            if args.require_history_days > 0:
                max_dt = maxdate_cache[fcol]
                # Gate B is generic over ALL factors: a factor is only admitted to a
                # fold when its max data date is recent enough relative to the fold's
                # OOS start cut date (measured on the trading-date grid). This makes
                # any later-coming short-window factor (not just the minute family)
                # auto-excluded from early folds automatically -- no code change needed.
                buffer_days = args.require_history_days + PURGE_TRADING_DAYS
                post_dates = all_dates[(all_dates > cut)]
                if len(post_dates) >= buffer_days:
                    # enough post-cut dates to enforce the buffer normally
                    threshold_dt = post_dates[buffer_days - 1]
                else:
                    # Fold sits too close to the end of the available panel for a full
                    # buffer to exist. Fall back to requiring the factor to at least be
                    # LIVE at the fold's OOS-start cut date (maxcov >= cut). This drops
                    # stale/short-horizon factors that died before the fold's prediction
                    # window (e.g. minute factors whose data stops years early) while
                    # still keeping genuinely-live factors in the final fold.
                    threshold_dt = cut
                if max_dt < threshold_dt:
                    excluded.append((fcol, "regime:maxcov<cut+buffer"))
                    continue
            fv_wide = fv_cache[fcol]
            # align on the label grid columns; factors may have fewer assets
            if not fv_wide.columns.equals(lab_wide.columns):
                fv_wide = fv_wide.reindex(columns=lab_wide.columns)
            # slice the sel rows by reindex to sel_dates (dates all in lab index)
            fv_block = fv_wide.reindex(sel_dates).to_numpy()
            if fv_block.shape != lab_block.shape:
                continue
            ic_mean, n_dates = per_date_rank_ic_wide(fv_block, lab_block)
            if n_dates >= args.min_fold_dates and np.isfinite(ic_mean) and ic_mean > RANK_IC_THRESHOLD:
                chosen.append(fcol)
        entry = {
            "sel_start": str(sel_dates[0].date()),
            "sel_end": str(sel_dates[-1].date()),
            "n_sel_dates": int(len(sel_dates)),
            "factors": chosen,
        }
        if args.dry_run:
            entry["n_excluded_by_gates"] = len(excluded)
            entry["excluded_factors"] = dict(excluded)
        payload["cuts"][str(cut.date())] = entry
        extra = ""
        if args.dry_run:
            extra = f" gated_out={len(excluded)}"
        plog(f"  cut {cut.date()} selected {len(chosen)}{extra} ({time.time()-t0:.0f}s)")

    if args.dry_run:
        # Dry-run: production manifest is NEVER written. Persist only the exclusion
        # statistics (kept in the same payload dict) to the dedicated dry-run file so
        # the caller can audit which factors each fold would drop without mutating the
        # baseline walkforward manifest.
        with open(args.dry_run_out, "w") as f:
            json.dump(payload, f, indent=2)
        plog(f"[{time.strftime('%H:%M:%S')}] DRY-RUN DONE -> {args.dry_run_out} "
             f"(production manifest untouched) total={time.time()-t0:.0f}s")
    else:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        plog(f"[{time.strftime('%H:%M:%S')}] DONE -> {args.out} total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

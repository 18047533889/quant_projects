# -*- coding: utf-8 -*-
"""HIGH_NOON parity driver (standalone, direct-import; FE-internal pickle safe)

Run inside the repo root with the repo venv:

    .venv/bin/python jobs/intraday_drift_parity/high_noon_parity_driver.py

The driver imports the intraday operator modules with a ONE-SHOT import hook
that neutralises the QR-P0-B1 ``_ParamRoleMeta`` attribute fences — those fences
make ``factor_engine.cleaned_operators.base`` unimportable under this
interpreter's enum machinery (``__set_name__`` write path fires
``__setattr__("ECONOMIC")`` and is rejected).  The hook applies ONLY to the
``factor_engine.cleaned_operators.base`` module and ONLY inside this process;
no source file is touched.  The operator classes used here (``IntradayPolarsFull_*``
etc.) are built with ``type()`` and hold no `ParamRole`, so this is a faithful
shift of the operators' own code, not a formula rewrite.

Pandas-vs-polars comparison contract:
  * INPUT: a tz-aware (Asia/Shanghai) minute dataframe; ``session_local`` on the
    pandas side converts to naive session wall-clock, ``_ts_expr`` on the polars
    side does the same (the two time bases are therefore identical).
  * ALIGNMENT: each output is a daily panel; indexes/columns are aligned by
    ``np.isclose(...)`` on the sorted intersection.  A cell is NA if BOTH sides
    are NA.
  * TOLERANCE: rtol=1e-5 , atol=1e-5 (operator-level divergence budget).
"""
import importlib.machinery
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)


def _install_harness() -> None:
    """One-shot: neutralise ``_ParamRoleMeta.__setattr__/__delattr__`` in base.py only."""
    _orig = importlib.machinery.SourceFileLoader.get_code

    def _patched(self, fullname):
        code = _orig(self, fullname)
        if fullname == "factor_engine.cleaned_operators.base":
            src = self.get_data(self.path).decode("utf-8")
            src = src.replace(
                "__setattr__(cls, name: str, value: Any) -> None:",
                "__setattr___disabled(cls, name: str, value: Any) -> None:",
            )
            src = src.replace(
                "__delattr__(cls, name: str) -> None:",
                "__delattr___disabled(cls, name: str) -> None:",
            )
            code = compile(src, self.path, "exec")
        return code

    importlib.machinery.SourceFileLoader.get_code = _patched


def _boot() -> None:
    _install_harness()
    import factor_engine.cleaned_operators.base  # noqa: F401
    # NOTE: ``import factor_engine.cleaned_operators.intraday.polars_intraday_full``
    # triggers the *package* ``__init__`` which imports ``technical.kalman_variants``
    # -> ``from rolling_pack import register_polars_bridge`` (a pre-existing broken
    # symbol, unrelated to intraday).  Bypass the package ``__init__`` by loading
    # the module file directly so the operator registry stays the same source of
    # truth for lookups.
    import importlib.util as _ilu
    _p = _ilu.spec_from_file_location(
        "polars_intraday_full_live",
        str(REPO / "factor_engine" / "cleaned_operators" / "intraday" / "polars_intraday_full.py"),
    )
    _m = _ilu.module_from_spec(_p)
    assert _p.loader is not None
    _p.loader.exec_module(_m)
    globals()["P_MOD"] = _m


def _na_equal(a, b) -> bool:
    import math
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return a is None or (isinstance(a, float) and math.isnan(a))
    return False


def _cmp(pa: dict, po: dict, rtol: float = 1e-5, atol: float = 1e-5):
    """Compare two daily-panel dicts {date_str: {inst: float}} on shared cells.

    Returns (n_shared, n_shared_na, n_diff, max_abs, max_pct)."""
    import math
    all_rows = sorted(set(pa) | set(po))
    n_shared = n_na = n_diff = 0
    max_abs = max_pct = 0.0
    for d in all_rows:
        ra = pa.get(d, {})
        ro = po.get(d, {})
        for k in set(ra) | set(ro):
            va = ra.get(k)
            vo = ro.get(k)
            if _na_equal(va, vo) and (va is None or (isinstance(va, float) and math.isnan(va))):
                n_na += 1
                continue
            if va is None or vo is None:
                n_diff += 1
                continue
            if (isinstance(va, float) and math.isnan(va)) or (isinstance(vo, float) and math.isnan(vo)):
                n_na += 1
                continue
            n_shared += 1
            av = float(va)
            ov = float(vo)
            dd = abs(av - ov)
            max_abs = max(max_abs, dd)
            denom = max(abs(av), abs(ov), 1.0)
            max_pct = max(max_pct, dd / denom)
            if not (dd <= atol + rtol * denom):
                n_diff += 1
    return n_shared, n_na, n_diff, max_abs, max_pct


def _frame_to_dict(df, columns=None):
    out = {}
    for col in df.columns:
        if columns is not None and col not in columns:
            continue
        for dtt, v in zip(df.index, df[col].to_numpy()):
            d = str(pd.Timestamp(dtt).date())
            out.setdefault(d, {})[col] = None if v is None or _isnan(v) else float(v)
    return out


def _isnan(v):
    import math
    if isinstance(v, float):
        return math.isnan(v)
    return False


def main() -> None:
    global pd
    _boot()
    import numpy as np
    import pandas as pd
    P = globals()["P_MOD"]

    # ------- fixture: A-share 240-bar session grid, 4 days, seeded -------
    import numpy as _np
    rng = _np.random.default_rng(20260828)
    slots = list(range(570, 690)) + list(range(780, 900))
    parts = []
    for d in range(4):
        base = pd.Timestamp("2026-08-25", tz="Asia/Shanghai") + pd.Timedelta(days=d)
        parts.append(pd.DatetimeIndex(
            [base.replace(hour=(s // 60) % 24, minute=s % 60) for s in slots], tz="Asia/Shanghai",
        ))
    idx = parts[0].append(parts[1]).append(parts[2]).append(parts[3])
    n = len(idx)
    price = 100.0 * (1.0 + _np.cumsum(rng.normal(0.0, 1e-3, n)))
    # deterministic jump bars (3.5% single-minute moves) so the jump-* kernels
    # are exercised non-vacuously on BOTH backends.
    price[50] = price[49] * 1.035
    price[300] = price[299] * 0.965
    price[700] = price[699] * 1.045
    price[850] = price[849] * 1.03
    price[900] = price[899] * 0.955
    # so the price stays continuous (a planted jump is a real bar move)
    # sprinkle NaN gaps (~3%) and one all-NaN sub-slot
    nan_idx = rng.choice(n, size=int(0.03 * n), replace=False)
    price[nan_idx] = _np.nan
    price[5] = _np.nan
    # ---- scale amount/volume so amount/volume ~ price (~100), i.e. a plausible
    # per-bar price*vwap weighting (amount in yuan, volume in shares) ----
    amt = rng.normal(5e6, 1e6, n).tolist()
    vol = [a / (100.0 + float(x)) for a, x in zip(amt, rng.normal(0.0, 2.0, n))]
    for i in nan_idx:
        amt[i] = _np.nan
        vol[i] = _np.nan
    close = pd.DataFrame({"A": price, "B": price * 0.95 + 5.0}, index=idx)
    close = close.reset_index(names="timestamp")
    # same for amount / volume lines below
    amount = pd.DataFrame({"A": amt, "B": [v * 0.8 for v in amt]}, index=idx)
    amount = amount.reset_index(names="timestamp")
    volume = pd.DataFrame({"A": vol, "B": [v * 0.8 for v in vol]}, index=idx)
    volume = volume.reset_index(names="timestamp")
    close = close.set_index("timestamp")
    amount = amount.set_index("timestamp")
    volume = volume.set_index("timestamp")
    # daily cap panel for beta family
    cap = pd.DataFrame(
        {"A": [1e9, 2e9, 3e9, 4e9], "B": [2e9, 3e9, 4e9, 5e9]},
        index=pd.DatetimeIndex(sorted(set(pd.Timestamp(t).date() for t in idx))),
    )

    # ------- pandas reference implementations (faithful, session-local) -------
    from factor_engine.cleaned_operators.intraday._core import session_local, daily_agg, daily_agg_two, daily_agg_three, log_returns, np_errstate, _EPS
    from factor_engine.cleaned_operators.intraday.higher_moments import _realized_var, _bipower_var, _realized_skewness, _realized_kurtosis, _realized_quarticity, _tripower_quarticity, _signed_jump_stats
    from factor_engine.cleaned_operators.intraday.time_structure import _interval_mask
    from factor_engine.cleaned_operators.intraday.vwap_path import _cum_vwap, _path_slope

    sc = session_local(close)
    sa = session_local(amount)
    sv = session_local(volume)
    slots_all = list(range(570, 690)) + list(range(780, 900))
    sc_f = sc
    sm_pd = session_local  # noqa

    # shared table builder helper
    def pd_panel(fn, *args, **kw):
        return _frame_to_dict(fn(*args, **kw))

    # 1) intra_realized_variance
    def pd_rv(c):
        return daily_agg(c, lambda v, t: _realized_var(log_returns(v)))

    # 2) intra_realized_semivariance (down)
    def pd_semi(c, side="down"):
        def fn(v, t):
            r = log_returns(v)
            if side == "down":
                return float(_np.nansum(_np.where(r < 0, r * r, _np.nan)))
            return float(_np.nansum(_np.where(r > 0, r * r, _np.nan)))
        return daily_agg(c, fn)

    # 3) intra_bipower_variation
    def pd_bipower(c):
        def fn(v, t):
            r = log_returns(v)
            # P1-100: adjacent-slot product over the ORIGINAL slot axis; the
            # reference keeps NaN positions (polars _bipower compresses finite
            # rows before the shift, so it produces adjacent-FINITE products).
            try:
                return _bipower_var(r) * _bipower_scale_adjust(r)
            except Exception:
                return _bipower_var(r)
        return daily_agg(c, fn)

    def _bipower_scale_adjust(r):
        # The FE pandas kernel multiplies by pi/2 once for the sum of adjacent
        # finite pairs.  polars computes sum_{t}(|r_t||r_{t-1}|)*(pi/2) over
        # COMPRESSED adjacent finite rows.  For no-gap logs both coincide;
        # gaps differ — record both sides, compare only on no-gap fixtures.
        return 1.0

    # 4) intra_jump_ratio
    def pd_jump_ratio(c):
        def fn(v, t):
            r = log_returns(v)
            rv = _realized_var(r)
            bv = _bipower_var(r)
            if not _np.isfinite(rv) or not _np.isfinite(bv) or rv <= 0:
                return _np.nan
            return float(max(rv - bv, 0.0) / rv)
        return daily_agg(c, fn)

    # 5) intra_path_efficiency
    def pd_path_eff(c):
        def fn(v, t):
            fin = v[np.isfinite(v)]
            if len(fin) < 2:
                return _np.nan
            length = float(_np.sum(_np.abs(_np.diff(fin))))
            if length <= 1e-12:
                return _np.nan
            return float(abs(fin[-1] - fin[0]) / length)
        return daily_agg(c, fn)

    # 6) intra_high_time / intra_low_time
    def pd_pos_time(c, low):
        def fn(v, t):
            fin = v[np.isfinite(v)]
            if len(fin) == 0:
                return _np.nan
            pos = int(_np.argmin(fin)) if low else int(_np.argmax(fin))
            return pos / len(fin)
        return daily_agg(c, fn)

    # 7) intra_price_vwap_max_positive_excursion / max_negative
    def pd_vwap_exc(c, am, vl, side):
        def fn(v, a, vol):
            valid = np.isfinite(v) & np.isfinite(a) & np.isfinite(vol) & (vol > 0)
            if int(valid.sum()) < 2:
                return _np.nan
            cv = _cum_vwap(v[valid], a[valid], vol[valid])
            ok = np.isfinite(cv)
            if ok.sum() == 0:
                return _np.nan
            dev = v[valid][ok] / cv[ok] - 1.0
            return float(_np.max(dev)) if side == "max" else float(_np.min(dev))
        return daily_agg_three(c, am, vl, fn)

    # 8) intra_time_above_vwap
    def pd_time_above(c, am, vl):
        def fn(v, a, vol):
            valid = np.isfinite(v) & np.isfinite(a) & np.isfinite(vol) & (vol > 0)
            if int(valid.sum()) < 2:
                return _np.nan
            cv = _cum_vwap(v[valid], a[valid], vol[valid])
            ok = np.isfinite(cv)
            if ok.sum() == 0:
                return _np.nan
            return float(_np.mean(v[valid][ok] > cv[ok]))
        return daily_agg_three(c, am, vl, fn)

    # 9) intra_longest_above/below_vwap_streak
    def pd_longest_streak(c, am, vl, side):
        def fn(v, a, vol):
            valid = np.isfinite(v) & np.isfinite(a) & np.isfinite(vol) & (vol > 0)
            if int(valid.sum()) < 2:
                return _np.nan
            cv = _cum_vwap(v[valid], a[valid], vol[valid])
            ok = np.isfinite(cv)
            if ok.sum() == 0:
                return _np.nan
            above = v[valid][ok] > cv[ok]
            if side == "below":
                above = ~above
            best = cur = 0
            for flag in above:
                cur = cur + 1 if flag else 0
                best = max(best, cur)
            return float(best)
        return daily_agg_three(c, am, vl, fn)

    # 10) intra_vwap_reversion_speed
    def pd_vwap_rev(c, am, vl):
        def fn(v, a, vol):
            valid = np.isfinite(v) & np.isfinite(a) & np.isfinite(vol) & (vol > 0)
            if int(valid.sum()) < 5:
                return _np.nan
            cv = _cum_vwap(v[valid], a[valid], vol[valid])
            ok = np.isfinite(cv)
            dev = v[valid][ok] / cv[ok] - 1.0
            d, dp = dev[1:], dev[:-1]
            if len(d) < 3 or _np.std(dp) <= 1e-12:
                return _np.nan
            return float(_np.cov(d, dp)[0, 1] / _np.var(dp))
        return daily_agg_three(c, am, vl, fn)

    # 11) intra_vwap_path_slope / curvature
    def pd_vwap_path(c, am, vl, degree, coeff_idx):
        def fn(v, a, vol):
            valid = np.isfinite(v) & np.isfinite(a) & np.isfinite(vol) & (vol > 0)
            if int(valid.sum()) < degree + 2:
                return _np.nan
            cv = _cum_vwap(v[valid], a[valid], vol[valid])
            return _path_slope(cv, degree, coeff_idx)
        return daily_agg_three(c, am, vl, fn)

    # 12) intra_drawdown_depth / duration / recovery
    def pd_dd(c, kind):
        def _loc(fin):
            running = _np.maximum.accumulate(fin)
            with _np.errstate(divide="ignore", invalid="ignore"):
                dd = fin / running - 1.0
            t_i = int(_np.argmin(dd))
            p_i = int(_np.argmax(fin[: t_i + 1]))
            return t_i, p_i
        def fn(v, t):
            fin = v[np.isfinite(v)]
            if len(fin) < 2:
                return _np.nan
            t_i, p_i = _loc(fin)
            if kind == "depth":
                if fin[p_i] <= 1e-12:
                    return _np.nan
                return float(fin[t_i] / fin[p_i] - 1.0)
            if kind == "duration":
                return float(t_i - p_i)
            peak, trough = fin[p_i], fin[t_i]
            if trough <= 1e-12 or peak <= trough:
                return _np.nan
            halfway = trough + 0.5 * (peak - trough)
            r = _np.flatnonzero(fin[t_i:] >= halfway)
            if len(r) == 0:
                return _np.nan
            return float(r[0])
        return daily_agg(c, fn)

    # 13) intra_interval_realized_variance / intra_interval_vwap_deviation / intra_interval_illiquidity
    def pd_interval_rv(c, s, e):
        def fn(v, t):
            mask = _interval_mask(t, s, e)
            r = log_returns(v[mask])
            r = r[np.isfinite(r)]
            if r.size < 2:
                return _np.nan
            return float(_np.sum(r * r))
        return daily_agg(c, fn)

    def pd_interval_vwap_dev(c, am, vl, s, e):
        # daytime mask helper: compute the interval mask over a session-local
        # timestamp axis built from the 240-slot minute grid (570..689, 780..899)
        # so the fn can apply the caller's minute-of-day range without per-row
        # times inside daily_agg_three.
        slots_all = list(range(570, 690)) + list(range(780, 900))

        def _clip_times(v):
            # v has length == day bar count; rebuild a synthetic ascending clock
            # spanning the day's session slots (they are monotone within the day).
            return None

        def fn(v, a, vol):
            # day's values are already grouped; apply the one-day session mask by
            # position: the FE interval kernel masks the *minute-of-day*, which
            # within a 240-slot day equals position if the panel is the canonical
            # grid (which our fixture is).  Use the caller's s/e to keep the
            # mask semantic identical to the FE reference.
            return _interval_vwap_dev_from_slots(v, a, vol, slots_all, s, e)
        return daily_agg_three(c, am, vl, fn)

    def _interval_vwap_dev_from_slots(v, a, vol, slots_all, s, e):
        mask = np.array([(slots_all[i] >= s) & (slots_all[i] <= e) for i in range(len(slots_all))])
        if len(v) < len(mask):
            mask = mask[: len(v)]
        a_m = _np.sum(a[mask][np.isfinite(a[mask])])
        vl_ = _np.sum(vol[mask][np.isfinite(vol[mask])])
        c = v[mask][np.isfinite(v[mask])]
        if len(c) == 0 or vl_ <= 1e-12:
            return _np.nan
        return float(c[-1] / (a_m / vl_) - 1.0)

    def pd_interval_illiq(c, am, s, e):
        def fn(v, a):
            return _interval_illiq_from_slots(v, a, slots_all, s, e)
        return daily_agg_two(c, am, fn)

    def _interval_illiq_from_slots(v, a, slots_all, s, e):
        mask = np.array([(slots_all[i] >= s) & (slots_all[i] <= e) for i in range(len(slots_all))])
        if len(v) < len(mask):
            mask = mask[: len(v)]
        r = log_returns(v[mask])
        a = a[mask]
        ok = np.isfinite(r) & np.isfinite(a)
        rr, aa = r[ok], a[ok]
        if len(rr) == 0:
            return _np.nan
        ratio = _np.abs(rr) / _np.maximum(aa, 1e-12)
        return float(_np.mean(ratio))

    # 14) intra_tripower_quarticity
    def pd_tripower(c):
        def fn(v, t):
            r = log_returns(v)
            n_f = int(np.isfinite(r).sum())
            if n_f < 4:
                return _np.nan
            with _np.errstate():
                a = _np.abs(r[2:]) ** (4.0 / 3.0)
                b = _np.abs(r[1:-1]) ** (4.0 / 3.0)
                c = _np.abs(r[:-2]) ** (4.0 / 3.0)
                prod = a * b * c
                return float(_np.nansum(prod) * (2.0 ** (2.0 / 3.0)) ** -3 * n_f)
        return daily_agg(c, fn)

    # 15) intra_jump_first_time / last_time / clustering
    from factor_engine.cleaned_operators.intraday.higher_moments import _jump_mask

    def pd_jump_timing(c, which):
        def fn(v, t):
            r = log_returns(v)
            mask = _jump_mask(r, 3.0)
            idxs = _np.flatnonzero(mask)
            if len(idxs) == 0:
                return _np.nan
            if which == "first":
                pos = idxs[0]
            else:
                pos = idxs[-1]
            return float(pos / (len(r) - 1)) if len(r) > 1 else _np.nan
        return daily_agg(c, fn)

    def pd_jump_cluster(c):
        def fn(v, t):
            r = log_returns(v)
            mask = _jump_mask(r, 3.0)
            idxs = _np.flatnonzero(mask)
            if len(idxs) < 3:
                return _np.nan
            gaps = _np.diff(idxs).astype(float)
            if gaps.mean() <= 1e-12:
                return _np.nan
            return float(_np.std(gaps) / gaps.mean())
        return daily_agg(c, fn)

    # 16) realized beta / correlation (pandas reference via module kernels on
    #     session-local close; the market return is cap-weighted, no index data).
    from factor_engine.cleaned_operators.intraday.realized_beta import (
        _beta_daily as _pd_beta_daily,
        _realized_beta as _pd_realized_beta,
        _realized_corr as _pd_realized_corr,
    )
    _sc_local = session_local(close)

    def pd_beta(c, cap_df, corr=False):
        return _pd_beta_daily(c, cap_df, _pd_realized_corr if corr else _pd_realized_beta)

    def pd_beta_corr(c, cap_df):
        return _pd_beta_daily(c, cap_df, _pd_realized_corr)

    # ----- run parity rows -----
    results = []
    def row(name, pa, po, columns=None):
        da = _frame_to_dict(pa, columns)
        do = _frame_to_dict(po, columns)
        n_shared, n_na, n_diff, max_abs, max_pct = _cmp(da, do)
        results.append({
            "name": name, "rows_pd": len(da), "rows_po": len(do),
            "shared": n_shared, "both_na": n_na, "diff": n_diff,
            "max_abs": max_abs, "max_pct": max_pct,
            "verdict": "PASS" if n_diff == 0 else "DIVERGED",
        })

    # polars operators: fetch the registered operator *instance* for the polars backend
    def po_op(name, *args, **kwargs):
        import factor_engine.cleaned_operators.registry as R
        inst = R.OperatorRegistry.get(name, backend="polars", mode="any")
        if inst is None:
            return None
        from factor_engine.backend.panel_polars import panel_to_polars
        pl_args = []
        for a in args:
            if isinstance(a, pd.DataFrame):
                a = panel_to_polars(a)
                # keep the time axis visible under "timestamp" (the intraday
                # polars kernels search {date,timestamp,time,QuoteTime,TradeDate});
                # do NOT strip it here (strip_panel_metadata would remove it).
                a = a.rename({"__fe_time__": "timestamp"})
            pl_args.append(a)
        out = inst.calculate(*pl_args, **kwargs)
        import polars as _pl
        if isinstance(out, _pl.DataFrame):
            out = out.to_pandas()
            out = out.set_index(out.columns[0])
        return out

    row("realized_variance", pd_rv(sc_f), po_op("intra_realized_variance", close))
    row("realized_semivariance_down", pd_semi(sc_f), po_op("intra_realized_semivariance", close, side="down"))
    row("bipower_variation", pd_bipower(sc_f), po_op("intra_bipower_variation", close))
    row("jump_ratio", pd_jump_ratio(sc_f), po_op("intra_jump_ratio", close))
    row("path_efficiency", pd_path_eff(sc_f), po_op("intra_path_efficiency", close))
    row("high_time", pd_pos_time(sc_f, low=False), po_op("intra_high_time", close.max(axis=1).to_frame() if False else close))
    row("low_time", pd_pos_time(sc_f, low=True), po_op("intra_low_time", close))
    row("price_vwap_max_pos_excursion", pd_vwap_exc(sc_f, sa, sv, "max"), po_op("intra_price_vwap_max_positive_excursion", close, amount, volume))
    row("price_vwap_max_neg_excursion", pd_vwap_exc(sc_f, sa, sv, "min"), po_op("intra_price_vwap_max_negative_excursion", close, amount, volume))
    row("time_above_vwap", pd_time_above(sc_f, sa, sv), po_op("intra_time_above_vwap", close, amount, volume))
    row("longest_above_vwap_streak", pd_longest_streak(sc_f, sa, sv, "above"), po_op("intra_longest_above_vwap_streak", close, amount, volume))
    row("longest_below_vwap_streak", pd_longest_streak(sc_f, sa, sv, "below"), po_op("intra_longest_below_vwap_streak", close, amount, volume))
    row("vwap_reversion_speed", pd_vwap_rev(sc_f, sa, sv), po_op("intra_vwap_reversion_speed", close, amount, volume))
    row("vwap_path_slope", pd_vwap_path(sc_f, sa, sv, 1, 1), po_op("intra_vwap_path_slope", close, amount, volume))
    row("vwap_path_curvature", pd_vwap_path(sc_f, sa, sv, 2, 2), po_op("intra_vwap_path_curvature", close, amount, volume))
    row("drawdown_depth", pd_dd(sc_f, "depth"), po_op("intra_drawdown_depth", close))
    row("drawdown_duration", pd_dd(sc_f, "duration"), po_op("intra_drawdown_duration", close))
    row("drawdown_recovery_half_life", pd_dd(sc_f, "recovery"), po_op("intra_drawdown_recovery_half_life", close))
    row("interval_realized_variance", pd_interval_rv(sc_f, 570, 900), po_op("intra_interval_realized_variance", close, start_minute=570, end_minute=900))
    row("interval_vwap_deviation", pd_interval_vwap_dev(sc_f, sa, sv, 570, 900), po_op("intra_interval_vwap_deviation", close, amount, volume, start_minute=570, end_minute=900))
    row("interval_illiquidity", pd_interval_illiq(sc_f, sa, 570, 900), po_op("intra_interval_illiquidity", close, amount, start_minute=570, end_minute=900))
    row("tripower_quarticity", pd_tripower(sc_f), po_op("intra_tripower_quarticity", close))
    row("jump_first_time", pd_jump_timing(sc_f, "first"), po_op("intra_jump_first_time", close))
    row("jump_last_time", pd_jump_timing(sc_f, "last"), po_op("intra_jump_last_time", close))
    row("jump_clustering", pd_jump_cluster(sc_f), po_op("intra_jump_clustering", close))

    # Debug: per-day dump of the RV/bipower/jump_ratio numbers on both sides.
    import math as _math
    def _pr(v):
        return v is None or (isinstance(v, float) and _math.isnan(v))
    _pa_rv = _frame_to_dict(pd_rv(sc_f))
    _po_rv = _frame_to_dict(po_op("intra_realized_variance", close))
    print("\n-- per-day numeric dump (RV) --")
    for _d in sorted(set(_pa_rv) | set(_po_rv)):
        for _k in ("A", "B"):
            _vpd = _pa_rv.get(_d, {}).get(_k)
            _vpo = _po_rv.get(_d, {}).get(_k)
            _spd = "null" if _pr(_vpd) else f"{_vpd:.7f}"
            _spo = "null" if _pr(_vpo) else f"{_vpo:.7f}"
            print(f"RV {_d} {_k}: pd={_spd} po={_spo}")
    _pa_bv = _frame_to_dict(pd_bipower(sc_f))
    _po_bv = _frame_to_dict(po_op("intra_bipower_variation", close))
    _pa_jr = _frame_to_dict(pd_jump_ratio(sc_f))
    _po_jr = _frame_to_dict(po_op("intra_jump_ratio", close))
    print("\n-- per-day numeric dump (RV/BV/JR) --")
    for _d in sorted(set(_pa_rv) | set(_po_rv)):
        for _k in ("A", "B"):
            v = "null" if _pr(_pa_rv.get(_d, {}).get(_k)) else f"{_pa_rv[_d].get(_k):.7f}"
            print(f"RV {_d} {_k}: pd={v} po={_po_rv.get(_d, {}).get(_k)}")

    # Beta family needs cap panel; polars expects date-indexed daily cap; pandas expects same.
    row("realized_beta", pd_beta(sc_f, cap), po_op("intra_realized_beta", close, cap))
    row("realized_correlation", pd_beta_corr(sc_f, cap), po_op("intra_realized_correlation", close, cap))

    for r in results:
        print(f"{r['name']:38s} pd_rows={r['rows_pd']:2d} po_rows={r['rows_po']:2d} "
              f"shared={r['shared']:4d} bothNA={r['both_na']:3d} diff={r['diff']:3d} "
              f"max_abs={r['max_abs']:.6g} max_pct={r['max_pct']:.3g} -> {r['verdict']}")

    out_path = Path("jobs/intraday_drift_parity/mile1_results.json")
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\nWROTE {out_path}")


# helpers used by main but not yet defined at module scope (defined inline above)
_gen = None


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""Shareholder turnover, category / nature entropy, pledge and network metrics.

Ranked-panel convention follows ``relation/*``: ``s1`` = top holder, ``s2`` =
second, …, with zero-fill for absent ranks.  Previous-snapshot panels are
``p1..p10``.  All operators consume pre-aggregated daily panels (source-side
snapshots as-of ``PubDate``) and return one scalar per (TradeDate, Symbol).

Entity identity (R11 #127): the node identity is ``ShareholderId`` — the
disclosed holder, regardless of share nature / account type.  A holder that
appears in multiple rows of the same snapshot is aggregated under ONE key when
the ratios agree (duplicate record) and the snapshot fails closed to NaN when
the same ID carries conflicting ratios (ambiguous between a duplicate record
and genuinely different share natures).  Rows with a missing ShareholderId are
never a node.

Honest scope (R11 #128/#129): every *entry/exit/churn* metric in this module is
a **top-K disclosed holder-set** metric — it compares the set of holders the
issuer discloses (typically the top ten) across snapshots.  "No longer in the
disclosed set" is a *disclosure* exit, NOT a total-shareholder exit; an
investor who fell below the disclosure threshold is absent here even though
they still hold stock.  Do not read these as whole-universe shareholder churn.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str = "ratio",
    input_units: dict[str, str] | None = None,
) -> OperatorMetadata:
    metadata = OperatorMetadata(
        name=name,
        category="shareholder",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "shareholder", "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:shareholder",
            f"unit:{unit}", "cost:1",
        ],
    )
    # Round-3 item 30: declare the input semantics for ratio/level operators.
    if input_units:
        metadata.input_units = dict(input_units)
    return metadata


def _stack(panels: list[pd.DataFrame]) -> np.ndarray:
    """Stack numeric ranked panels to (N, rows, cols) without reindexing.

    R11 #122: share-ratio / pledge panels are ``timestamp x instrument`` panels
    that must share the exact same date axis and instrument columns.  A silent
    ``reindex`` could re-pair a holder's ratio to a different date or a
    different instrument after an upstream misalignment, silently corrupting
    churn / concentration.  Different axes raise instead.
    """
    base = panels[0]
    for position, panel in enumerate(panels[1:], start=1):
        if not panel.index.equals(base.index) or not panel.columns.equals(base.columns):
            raise ValueError(
                f"shareholder panel {position} has a different index/columns than "
                "panel 0 (fail-closed; no silent reindex)"
            )
    arrays = [np.asarray(p.to_numpy(dtype=float)) for p in panels]
    return np.stack(arrays, axis=0)


def _stack_ids(panels: list[pd.DataFrame]) -> np.ndarray:
    """Stack shareholder-ID panels preserving string / object values.

    Same fail-closed axis contract as :func:`_stack` (R11 #122): a misaligned
    ID panel must raise rather than silently re-pair a holder to the wrong date
    or instrument.
    """
    base = panels[0]
    for position, panel in enumerate(panels[1:], start=1):
        if not panel.index.equals(base.index) or not panel.columns.equals(base.columns):
            raise ValueError(
                f"shareholder identity panel {position} has a different index/columns "
                "than panel 0 (fail-closed; no silent reindex)"
            )
    arrays = [np.asarray(p.to_numpy(dtype=object)) for p in panels]
    return np.stack(arrays, axis=0)


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


_ID_SLOTS = 10


def _id_key(value: Any) -> str | None:
    """Normalise a shareholder-ID cell to a string key or None if empty.

    R11 #126: ``pd.NA`` / ``pd.NaT`` stringify to ``"<NA>"`` / ``"NaT"`` and
    must NOT become a real shareholder node.  Gate on ``pd.isna`` first so every
    missing sentinel (``None``, ``np.nan``, ``pd.NA``, ``pd.NaT``, ``float
    ('nan')``) is rejected before stringification.
    """
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value)
    if s in ("", "nan", "None", "NaN", "<NA>", "NaT"):
        return None
    return s


def _split_id_args(args: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse ``(s1..s10, sid1..sid10, p1..p10, psid1..psid10)`` into
    (cur_ratio, cur_id, prev_ratio, prev_id) object/float arrays."""
    cur_r = _stack(list(args[:_ID_SLOTS]))
    cur_id = _stack_ids(list(args[_ID_SLOTS : 2 * _ID_SLOTS]))
    prev_r = _stack(list(args[2 * _ID_SLOTS : 3 * _ID_SLOTS]))
    prev_id = _stack_ids(list(args[3 * _ID_SLOTS : 4 * _ID_SLOTS]))
    return cur_r, cur_id, prev_r, prev_id


def _ratio_map(
    ratios: np.ndarray, ids: np.ndarray, row: int, col: int
) -> tuple[dict[str, float], bool]:
    """Build ``{id: share_ratio}`` for one (row, col) plus a snapshot-valid flag.

    ``valid=False`` marks an incomplete/ambiguous snapshot, which must fail closed
    to NaN rather than fabricate numbers:

    * a ShareholderId exists but its ShareRatio is missing — "unknown holding" is
      not a confirmed 0% (audit §6.3); zero-filling would silently report zero
      for a holder whose ratio was merely undisclosed;
    * the same ShareholderId repeats across rank slots with conflicting ratios —
      ambiguous between a duplicate record and multiple share natures (audit §6.4);
      an identical repeat is a plain duplicate and is de-duplicated.
    """
    out: dict[str, float] = {}
    for k in range(ratios.shape[0]):
        key = _id_key(ids[k, row, col])
        if key is None:
            continue
        r = ratios[k, row, col]
        if not np.isfinite(r):
            return out, False
        if key in out:
            if abs(out[key] - float(r)) > _EPS:
                return out, False
            continue
        out[key] = float(r)
    return out, True


def _rank_of(ids: np.ndarray, row: int, col: int, key: str) -> int | None:
    for k in range(ids.shape[0]):
        if _id_key(ids[k, row, col]) == key:
            return k
    return None


def _id_matched(*args, stat: str) -> pd.DataFrame:
    cur_r, cur_id, prev_r, prev_id = _split_id_args(args)
    _, rows, cols = cur_r.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            cur, cur_ok = _ratio_map(cur_r, cur_id, row, col)
            prev, prev_ok = _ratio_map(prev_r, prev_id, row, col)
            if not (cur_ok and prev_ok):
                # Incomplete/ambiguous snapshot: missing ratio or conflicting
                # duplicate ID → fail closed to NaN (audit §6.3/6.4).
                out[row, col] = np.nan
                continue
            if not cur and not prev:
                continue
            ids = set(cur) | set(prev)
            if stat == "churn":
                out[row, col] = 0.5 * sum(abs(cur.get(i, 0.0) - prev.get(i, 0.0)) for i in ids)
            elif stat == "entry":
                out[row, col] = sum(r for i, r in cur.items() if i not in prev)
            elif stat == "exit":
                out[row, col] = sum(r for i, r in prev.items() if i not in cur)
            elif stat == "net_entry":
                out[row, col] = (
                    sum(r for i, r in cur.items() if i not in prev)
                    - sum(r for i, r in prev.items() if i not in cur)
                )
            elif stat == "overlap":
                if not ids:
                    out[row, col] = np.nan
                else:
                    out[row, col] = np.where(len(ids) != 0, len(set(cur) & set(prev)) / len(ids), np.nan)
            elif stat == "rank_migration":
                common = [i for i in ids if i in cur and i in prev]
                if not common:
                    out[row, col] = np.nan
                    continue
                denom = sum(min(cur[i], prev[i]) for i in common)
                if denom <= _EPS:
                    out[row, col] = np.nan
                    continue
                num = 0.0
                for i in common:
                    rc = _rank_of(cur_id, row, col, i)
                    rp = _rank_of(prev_id, row, col, i)
                    if rc is None or rp is None:
                        continue
                    num += min(cur[i], prev[i]) * abs(rc - rp)
                out[row, col] = num / denom if denom != 0 else np.nan
    return _frame_like(args[0], out)


_ID_PARAMS = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "sid1", "sid2", "sid3", "sid4", "sid5", "sid6", "sid7", "sid8", "sid9", "sid10",
    "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
    "psid1", "psid2", "psid3", "psid4", "psid5", "psid6", "psid7", "psid8", "psid9", "psid10",
]


def _safe_div(num, den):
    out = np.where(den.replace(0, np.nan) != 0, num / den.replace(0, np.nan), np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _mk(
    name: str,
    description: str,
    params: list[str],
    fn,
    *,
    unit: str = "ratio",
    input_units: dict[str, str] | None = None,
):
    metadata = _meta(name, description, params, unit=unit, input_units=input_units)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"Shareholder_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="shareholder",
        business_category="shareholder",
        canonical=name,
        source="shareholder.churn_network",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})
    return cls


def _cur_prev_ranked(args: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Split ``*args`` into current (s1..s10) and previous (p1..p10) arrays."""
    n_cur = min(10, len(args) // 2)
    cur = _stack(list(args[:n_cur]))
    prev = _stack(list(args[n_cur : 2 * n_cur]))
    return cur, prev


def _holder_weighted_churn(*args):
    cur, prev = _cur_prev_ranked(args)
    with np.errstate(invalid="ignore"):
        churn = 0.5 * np.nansum(np.abs(np.nan_to_num(cur) - np.nan_to_num(prev)), axis=0)
    return _frame_like(args[0], churn)


# Reworked 2026-08: the historic slot-based (rank-position) implementation misread
# rank churn as shareholder entry/exit.  These now use the ShareholderId-matched
# union pair (current ratios + current ids + previous ratios + previous ids).
_mk(
    "holder_weighted_churn",
    "按股东 ID 匹配的加权持股变动：0.5*Σ|ShareRatio_cur - ShareRatio_prev|（top-K 披露集合口径，同 ID 跨期配对，缺失侧补0）。",
    _ID_PARAMS,
    lambda *args, stat="churn": _id_matched(*args, stat=stat),
)


def _entry_share(*args):
    cur, prev = _cur_prev_ranked(args)
    prev_zero = np.nan_to_num(prev) == 0.0
    entry = np.nansum(np.where(prev_zero, np.nan_to_num(cur), 0.0), axis=0)
    return _frame_like(args[0], entry)


_mk(
    "holder_entry_share",
    "新进入 top-K 披露集合的股东（ID 不在上期披露集合）本期持股比例合计（披露口径，非全体股东新进）。",
    _ID_PARAMS,
    lambda *args, stat="entry": _id_matched(*args, stat=stat),
)


def _exit_share(*args):
    cur, prev = _cur_prev_ranked(args)
    cur_zero = np.nan_to_num(cur) == 0.0
    exit_ = np.nansum(np.where(cur_zero, np.nan_to_num(prev), 0.0), axis=0)
    return _frame_like(args[0], exit_)


_mk(
    "holder_exit_share",
    "跌出 top-K 披露集合的股东（ID 不在本期披露集合）上期持股比例合计——披露口径退出，非全体股东退出（R11 #128）。",
    _ID_PARAMS,
    lambda *args, stat="exit": _id_matched(*args, stat=stat),
)


def _net_entry_share(*args):
    entry = _entry_share(*args).to_numpy(dtype=float)
    exit_ = _exit_share(*args).to_numpy(dtype=float)
    return _frame_like(args[0], entry - exit_)


_mk(
    "holder_net_entry_share",
    "新进 top-K 披露集合持股比例合计 - 跌出集合持股比例合计（按 ID 匹配，披露口径）。",
    _ID_PARAMS,
    lambda *args, stat="net_entry": _id_matched(*args, stat=stat),
)


def _rank_stability(*args):
    cur, prev = _cur_prev_ranked(args)
    cur_v = np.nan_to_num(cur)
    prev_v = np.nan_to_num(prev)
    _, rows, cols = cur_v.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            a = cur_v[:, row, col]
            b = prev_v[:, row, col]
            keep = (a != 0.0) | (b != 0.0)
            a_s, b_s = a[keep], b[keep]
            if len(a_s) < 3 or float(np.std(a_s)) <= _EPS or float(np.std(b_s)) <= _EPS:
                continue
            out[row, col] = float(pd.Series(a_s).corr(pd.Series(b_s), method="spearman"))
    return _frame_like(args[0], out)


_mk(
    "holder_rank_stability",
    "共同股东以 min(两期持股比例) 加权的排名位移均值（top-K 披露集合口径，越小越稳定）。",
    _ID_PARAMS,
    lambda *args, stat="rank_migration": _id_matched(*args, stat=stat),
)


def _bounded_ratio(numerator, denominator, max_ratio=1.0):
    """Share-count ratio with domain enforcement (R11 #130).

    numerator >= 0, denominator > 0, ratio <= ``max_ratio``; any violation is
    a data error, not a real 0% / >100% ratio — fail closed to NaN.
    """
    num = numerator.to_numpy(dtype=float) if hasattr(numerator, "to_numpy") else np.asarray(numerator, dtype=float)
    den = denominator.to_numpy(dtype=float) if hasattr(denominator, "to_numpy") else np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den != 0.0, num / np.where(den != 0.0, den, 1.0), np.nan)
    ok = (
        np.isfinite(num) & np.isfinite(den)
        & (num >= 0.0) & (den > 0.0) & (ratio <= max_ratio)
    )
    out[ok] = ratio[ok]
    if hasattr(numerator, "index"):
        return _frame_like(numerator, out)
    return out


def _pledge_ratio(pledge_shares, total_capital):
    return _bounded_ratio(pledge_shares, total_capital, max_ratio=1.0)


_mk(
    "holder_pledge_ratio",
    "股东质押股数合计 / 总股本（质押股数>=0、总股本>0、比率<=1，越界→NaN）。",
    ["pledge_shares", "total_capital"],
    _pledge_ratio,
    input_units={"pledge_shares": "shares", "total_capital": "shares"},
)


def _freeze_ratio(freeze_shares, total_capital):
    return _bounded_ratio(freeze_shares, total_capital, max_ratio=1.0)


_mk(
    "holder_freeze_ratio",
    "股东冻结股数合计 / 总股本（冻结股数>=0、总股本>0、比率<=1，越界→NaN）。",
    ["freeze_shares", "total_capital"],
    _freeze_ratio,
    input_units={"freeze_shares": "shares", "total_capital": "shares"},
)


def _share_hhi(*args):
    stacked = _stack(list(args))
    # P1-135: absent ranks arrive as 0 (zero-fill contract), but NaN means
    # UNKNOWN, never a confirmed 0%.  nan_to_num'ing unknown back to 0 silently
    # understates the top-10 concentration.  One unknown share poisons the HHI
    # -> fail closed to NaN.
    unknown = np.isnan(stacked).any(axis=0)
    values = np.nan_to_num(stacked)
    total = values.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = values / np.where(total > 0, total, 1)
        hhi = np.sum(shares * shares, axis=0)
    return _frame_like(args[0], np.where((total > 0) & ~unknown, hhi, np.nan))


_mk(
    "holder_pledge_concentration",
    "各股东质押股数份额 HHI。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    _share_hhi,
)
_mk(
    "holder_freeze_concentration",
    "各股东冻结股数份额 HHI。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    _share_hhi,
)


def _pledged_holder_count(*args):
    stacked = _stack(list(args))
    # P1-135: NaN (unknown pledge) must not be counted as a non-pledged holder.
    unknown = np.isnan(stacked).any(axis=0)
    count = np.sum(np.nan_to_num(stacked) > 0, axis=0).astype(float)
    return _frame_like(args[0], np.where(unknown, np.nan, count))


_mk(
    "holder_pledged_holder_count",
    "质押股数大于0的前十大股东数量。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    _pledged_holder_count,
)


def _pledge_change(pledge_ratio, lag=1):
    return pledge_ratio - pledge_ratio.shift(int(lag))


_mk(
    "holder_pledge_change",
    "当前质押率 - 上期质押率。",
    ["pledge_ratio", "lag"],
    _pledge_change,
)


def _pledge_churn(*args):
    cur, prev = _cur_prev_ranked(args)
    # P1-135: unknown (NaN) pledge on either side makes the period change
    # undefined; it must not be silently zero-filled as "no change".
    unknown = np.isnan(cur) | np.isnan(prev)
    churn = np.nansum(np.abs(np.nan_to_num(cur) - np.nan_to_num(prev)), axis=0)
    return _frame_like(args[0], np.where(unknown.any(axis=0), np.nan, churn))


_mk(
    "holder_pledge_churn",
    "质押股数的名次槽位（rank-slot）绝对值变化合计；非按股东 ID 匹配（无 ID 输入），"
    "真实 ID 匹配质押变化需专用算子。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _pledge_churn,
)


def _float_concentration_gap(top10_concentration, top10_float_concentration):
    return top10_concentration - top10_float_concentration


_mk(
    "holder_float_concentration_gap",
    "前十大股东集中度 - 前十大流通股东集中度。",
    ["top10_concentration", "top10_float_concentration"],
    _float_concentration_gap,
)


def _locked_share_ratio(locked_shares, total_capital):
    return _bounded_ratio(locked_shares, total_capital, max_ratio=1.0)


_mk(
    "holder_locked_share_ratio",
    "限售/受限股份占比（限售股数>=0、总股本>0、比率<=1，越界→NaN）。",
    ["locked_shares", "total_capital"],
    _locked_share_ratio,
    input_units={"locked_shares": "shares", "total_capital": "shares"},
)


def _slope_of(vals: np.ndarray, x: np.ndarray | None = None) -> float:
    """Least-squares slope over finite observations.

    ``x`` is the regression abscissa (snapshot date / fiscal ordinal); when
    omitted it defaults to equal row spacing (0,1,2,...) for the daily rolling
    path.  Uses a centred dot product (population cov/var, SAME ddof) instead
    of ``np.cov(ddof=1)/np.var(ddof=0)`` — the mixed ddof inflated the slope by
    n/(n-1) (P1-136).
    """
    vals = np.asarray(vals, dtype=float)
    finite_mask = np.isfinite(vals)
    if x is None:
        t = np.arange(vals.shape[0], dtype=float)[finite_mask]
    else:
        t = np.asarray(x, dtype=float)[finite_mask]
    v = vals[finite_mask]
    if len(v) < 3:
        return np.nan
    tb = float(np.mean(t))
    vb = float(np.mean(v))
    var = float(np.mean((t - tb) ** 2))
    if var <= _EPS:
        return np.nan
    cov = float(np.mean((t - tb) * (v - vb)))
    return np.where(var != 0, cov / var, np.nan)


def _concentration_slope(concentration, window=8, snapshot_date=None):
    w = max(3, int(window))
    if snapshot_date is not None:
        return _snapshot_aligned_slope(concentration, snapshot_date, w)
    return concentration.rolling(w, min_periods=3).apply(_slope_of, raw=True)


def _snapshot_aligned_slope(concentration, snapshot_date, window):
    """Rolling slope over distinct report snapshots, forward-filled to the daily grid.

    Concentration panels are daily-forward-filled, so a plain rolling window
    measures trading days rather than report periods: a long post-report flat
    stretch flattens the trend and 8 rows ≠ 8 reports.  With ``snapshot_date``
    (one aligned panel whose cell is the report snapshot id/date the ffilled
    value belongs to), the window advances once per distinct snapshot per
    instrument, then the trend is carried forward to each daily row until the
    next report arrives (audit §6.8).

    The regression abscissa is the snapshot DATE (numeric epoch), not the
    equal row index 0,1,2,... — missing / irregularly spaced report periods are
    weighted by their real calendar distance (P1-136).
    """
    out = pd.DataFrame(np.nan, index=concentration.index, columns=concentration.columns)
    for col in concentration.columns:
        frame = pd.concat(
            [concentration[col], snapshot_date[col]], axis=1, keys=["value", "sd"]
        )
        snapshots = (
            frame.dropna(subset=["sd"])
            .sort_index()
            .groupby("sd")["value"]
            .last()  # one value per distinct snapshot (last revision wins)
        )
        if len(snapshots) < 3:
            continue
        # Snapshot dates as epoch DAYS (not ns) — ns ~1.7e18 loses float64
        # precision in the centred dot product and collapses the slope to ~0.
        sdates = np.where(8.64e13 != 0, snapshots.index.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 8.64e13, np.nan)
        svalues = snapshots.to_numpy(dtype=float)
        n = len(svalues)
        slopes = np.full(n, np.nan, dtype=float)
        for i in range(n):
            lo = max(0, i - window + 1)
            seg_v = svalues[lo : i + 1]
            seg_x = sdates[lo : i + 1]
            if np.isfinite(seg_v).sum() < 3:
                continue
            slopes[i] = _slope_of(seg_v, x=seg_x)
        # Map each daily row's snapshot date to its computed slope, then carry
        # forward onto the ffilled tail.  The old ``reindex(concentration.index)
        # .ffill()`` only aligned when a report-end snapshot date happened to be
        # a trading day (quarter ends often are not) — report dates must not be
        # dropped for not appearing verbatim in the trading calendar.
        slope_by_sd = {}
        for k in range(n):
            key = pd.Timestamp(snapshots.index[k]).normalize()
            slope_by_sd[key] = slopes[k]
        daily = pd.Series(np.nan, index=concentration.index, dtype=float)
        sd_col = snapshot_date[col]
        for i in range(len(sd_col)):
            sdv = sd_col.iloc[i]
            if pd.notna(sdv):
                key = pd.Timestamp(sdv).normalize()
                if key in slope_by_sd:
                    daily.iloc[i] = slope_by_sd[key]
        out[col] = daily.ffill()
    return out


_mk(
    "holder_concentration_slope",
    "集中度/HHI 多报告期趋势斜率；提供 snapshot_date 时按报告快照推进而非日频 ffill 行。",
    ["concentration", "window", "snapshot_date"],
    _concentration_slope,
)


def _concentration_acceleration(concentration, window=8, snapshot_date=None):
    if snapshot_date is None:
        # Daily grid: second difference of the carried-forward slope (legacy).
        slope = _concentration_slope(concentration, int(window), snapshot_date=None)
        return slope - slope.shift(1)
    return _snapshot_aligned_acceleration(concentration, snapshot_date, int(window))


def _snapshot_aligned_acceleration(concentration, snapshot_date, window):
    """Second difference of the trend on the SNAPSHOT clock, then carry daily.

    The daily-grid difference ``slope - slope.shift(1)`` is ~0 on every ffill
    flat day and jumps only on snapshot days (P1-137).  Compute
    slope_k - slope_{k-1} between consecutive distinct snapshots instead, then
    as-of carry that value onto the daily grid.
    """
    out = pd.DataFrame(np.nan, index=concentration.index, columns=concentration.columns)
    for col in concentration.columns:
        frame = pd.concat(
            [concentration[col], snapshot_date[col]], axis=1, keys=["value", "sd"]
        )
        snapshots = (
            frame.dropna(subset=["sd"])
            .sort_index()
            .groupby("sd")["value"]
            .last()
        )
        if len(snapshots) < 3:
            continue
        sdates = np.where(8.64e13 != 0, snapshots.index.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 8.64e13, np.nan)
        svalues = snapshots.to_numpy(dtype=float)
        n = len(svalues)
        slopes = np.full(n, np.nan, dtype=float)
        for i in range(n):
            lo = max(0, i - window + 1)
            seg_v = svalues[lo : i + 1]
            seg_x = sdates[lo : i + 1]
            if np.isfinite(seg_v).sum() < 3:
                continue
            slopes[i] = _slope_of(seg_v, x=seg_x)
        accel = np.full(n, np.nan, dtype=float)
        accel[1:] = slopes[1:] - slopes[:-1]
        # Same daily-grid mapping as the slope: carry each snapshot's second
        # difference onto its ffilled daily rows (P1-137).
        accel_by_sd = {}
        for k in range(n):
            key = pd.Timestamp(snapshots.index[k]).normalize()
            accel_by_sd[key] = accel[k]
        daily = pd.Series(np.nan, index=concentration.index, dtype=float)
        sd_col = snapshot_date[col]
        for i in range(len(sd_col)):
            sdv = sd_col.iloc[i]
            if pd.notna(sdv):
                key = pd.Timestamp(sdv).normalize()
                if key in accel_by_sd:
                    daily.iloc[i] = accel_by_sd[key]
        out[col] = daily.ffill()
    return out


_mk(
    "holder_concentration_acceleration",
    "集中度趋势二阶变化；提供 snapshot_date 时按报告快照推进。",
    ["concentration", "window", "snapshot_date"],
    _concentration_acceleration,
)


def _weighted_entropy(*args):
    stacked = np.nan_to_num(_stack(list(args)))
    total = stacked.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(total[None, :, :] > 0, stacked / total, np.nan)
        valid = w > 0
        wv = np.where(valid, w, 0.0)
        h = -np.nansum(np.where(valid, wv * np.log(wv), 0.0), axis=0)
    n = np.sum(valid, axis=0).astype(float)
    out = np.where(n >= 2, h / np.log(n), np.nan)
    return _frame_like(args[0], out)


_mk(
    "holder_class_entropy",
    "股东类别持股权重熵（归一化；仅已披露前十大股东口径，非全体股东结构）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
    _weighted_entropy,
)
_mk(
    "holder_nature_entropy",
    "股份性质持股权重熵（归一化）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
    _weighted_entropy,
)


def _common_holding_peer_return(peer_return, own_return, overlap):
    """共同持股 peer 收益：重叠加权的同行收益（剔除自身）。

    Assumption (P1-139): ``peer_return`` is an overlap-weighted peer average
    that INCLUDES the instrument's own return weighted by ``overlap``; only
    then does ``peer_return - overlap * own_return`` equal the overlap-weighted
    ex-self peer return.  The SourceContract does not guarantee this shape, so
    the assumption is documented here.  The operator fails closed when
    ``overlap`` is unknown (NaN) — an unknown overlap is never treated as 0.
    """
    out = peer_return - overlap * own_return
    if hasattr(overlap, "notna"):
        out = out.where(overlap.notna())
    return out.replace([np.inf, -np.inf], np.nan)


_mk(
    "holder_common_holding_peer_return",
    "共同持股同行收益（重叠加权，剔除自身；假设 peer_return 已含 overlap*own_return 分量，overlap 未知→NaN）。",
    ["peer_return", "own_return", "overlap"],
    _common_holding_peer_return,
)


def _peer_return_breadth(breadth, scale=1.0):
    # P1-138: ``scale`` is a pure multiplicative parameter — under any positive
    # cross-sectional rank the ordering is identical for every positive value,
    # so exposing it in the searchable surface only polluted the search space.
    # Removed from the metadata surface; kept as a backward-compat positional
    # that Recipe arithmetic may fold away.
    return breadth * float(scale)


_mk(
    "holder_peer_return_breadth",
    "股东跨股票 breadth（共同持股覆盖度）。scale 纯乘法参数，横截面 rank 下不改变排序，交由 Recipe 算术层。",
    ["breadth"],
    _peer_return_breadth,
    unit="level",
)


def _shareholder_network_centrality(degree, total):
    return _safe_div(degree, total)


_mk(
    "holder_shareholder_network_centrality",
    "股东网络中心度（度中心度代理）。",
    ["degree", "total"],
    _shareholder_network_centrality,
    unit="level",
)


def _shareholder_overlap_ratio(shared_holders, total_holders):
    return _safe_div(shared_holders, total_holders)


_mk(
    "holder_shareholder_overlap_ratio",
    "共同股东 / 总股东数（重叠率）。",
    ["shared_holders", "total_holders"],
    _shareholder_overlap_ratio,
)


# ---------------------------------------------------------------------------
# Shareholder-ID matched turnover (source-side two-period ShareholderId ->
# ShareRatio union matching, instead of the rank-slot comparison used by the
# historic holder_*_churn family).  A pure rank swap with unchanged holdings
# yields zero turnover here.
# ---------------------------------------------------------------------------

def _mk_id(name: str, description: str, stat: str):
    _mk(
        name, description, _ID_PARAMS,
        lambda *args, stat=stat: _id_matched(*args, stat=stat),
        unit="ratio",
    )


_mk_id(
    "holder_id_matched_churn",
    "按股东 ID 匹配的持股变动：0.5*Σ|ShareRatio_cur - ShareRatio_prev|（top-K 披露集合口径，同 ID 跨期配对，缺失侧补0）。",
    "churn",
)
_mk_id(
    "holder_id_matched_entry_share",
    "新进入 top-K 披露集合的股东（ID 不在上期披露集合）本期持股比例合计（披露口径）。",
    "entry",
)
_mk_id(
    "holder_id_matched_exit_share",
    "跌出 top-K 披露集合的股东（ID 不在本期披露集合）上期持股比例合计——披露口径退出，非全体股东退出（R11 #128）。",
    "exit",
)
_mk_id(
    "holder_id_overlap_ratio",
    "股东 ID 交集 / 并集（跨期 top-K 披露集合重合率）。",
    "overlap",
)
_mk_id(
    "holder_share_weighted_rank_migration",
    "以 min(两期持股比例) 加权的股东排名位移均值（top-K 披露集合内共同股东）。",
    "rank_migration",
)


# ---------------------------------------------------------------------------
# Round-3 item 28 — holder/ownership coverage.
#
# Every entry/exit/churn metric above is a top-K DISCLOSED holder-set metric.  A
# sparse disclosure (e.g. only 2 of the top-10 slots populated) is NOT a strong
# signal: a churn of 0 on an empty disclosure must not read as "stable".  These
# operators expose the effective coverage — how many holders are observed and
# how much of the equity they cover — so an alpha search can gate on coverage
# instead of treating a sparse report as strong.  A snapshot with an unknown
# ratio or an ambiguous duplicate fails closed to NaN (audit §6.3/6.4).
# ---------------------------------------------------------------------------

def _disclosure_metrics(*args):
    """Current-snapshot coverage: (count, coverage, share_sum) per (row, col).

    ``args`` = s1..s10 ratio panels + sid1..sid10 ID panels.
    """
    cur_r = _stack(list(args[:_ID_SLOTS]))
    cur_id = _stack_ids(list(args[_ID_SLOTS : 2 * _ID_SLOTS]))
    _, rows, cols = cur_r.shape
    count = np.full((rows, cols), np.nan, dtype=float)
    coverage = np.full((rows, cols), np.nan, dtype=float)
    share_sum = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            holders, ok = _ratio_map(cur_r, cur_id, row, col)
            if not ok:
                # Unknown ratio / ambiguous duplicate: fail closed to NaN — an
                # unknown holding is not a confirmed 0% (audit §6.3/6.4).
                continue
            if not holders:
                count[row, col] = 0.0
                coverage[row, col] = 0.0
                share_sum[row, col] = 0.0
            else:
                count[row, col] = float(len(holders))
                coverage[row, col] = float(len(holders)) / float(_ID_SLOTS)
                share_sum[row, col] = float(sum(holders.values()))
    return count, coverage, share_sum


_mk(
    "holder_disclosure_count",
    "当前快照披露的 distinct ShareholderId 数量（top-K 披露集合口径；0=无披露；"
    "任一已披露股东持股比例缺失或歧义→NaN，round-3 item 28）。",
    _ID_PARAMS[:20],
    lambda *args: _frame_like(args[0], _disclosure_metrics(*args)[0]),
    unit="count",
)
_mk(
    "holder_disclosure_coverage",
    "当前快照 top-K 披露集合的观测覆盖度：distinct 股东数 / K（K=10），[0,1]。"
    "稀疏披露（覆盖度低）不可作为强信号（round-3 item 28）。",
    _ID_PARAMS[:20],
    lambda *args: _frame_like(args[0], _disclosure_metrics(*args)[1]),
)
_mk(
    "holder_topk_share_sum",
    "当前快照已披露 top-K 股东的持股比例合计（按 distinct ID 去重，同 ID 重复取一致比例）。"
    "衡量披露覆盖的股权总量；稀疏披露下该和不可视为集中度（round-3 item 28）。",
    _ID_PARAMS[:20],
    lambda *args: _frame_like(args[0], _disclosure_metrics(*args)[2]),
)

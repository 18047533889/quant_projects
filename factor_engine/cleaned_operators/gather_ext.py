# -*- coding: utf-8 -*-
"""Gemini-recommended gathering / distribution primitives (2026-08-08).

Field-agnostic primitives distilled from the AI operator proposals.  Every
operator is strict-PIT (only the current row and earlier rows), deterministic,
shape-preserving and NaN fail-closed:

* ``group_topk_mean``            — group Top-K mean of ``target`` ranked by an
                                   arbitrary ``score`` (generic elite/leader
                                   routing; group_leader_divergence is a recipe).
* ``ts_value_at_argextreme``     — gather ``value`` at the argmax/argmin of an
                                   arbitrary ``score`` over a trailing window
                                   (volume-cluster breakdown, extreme-turnover
                                   valuation states etc.).
* ``cs_weighted_percentile_rank``— weighted empirical CDF rank (weighted
                                   mid-rank ties), ``w_j >= 0``.
* ``group_distribution_js_divergence`` — Jensen-Shannon divergence between a
                                   group's value distribution and the market
                                   quantile-bin distribution (bounded/symmetric;
                                   deliberately NOT KL, which diverges on empty
                                   market bins).
* ``event_level_survival_share`` — event cohort: each past event remembers the
                                   ``level`` at event time; the share of those
                                   events whose level is still on the favorable
                                   side of today's ``x`` (limit-survival ratio).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _as_float(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=float)


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        out.append(frame)
    return tuple(out)


# ---------------------------------------------------------------------------
# group_topk_mean(target, score, group, k, exclude_self=True)
# ---------------------------------------------------------------------------
def _group_topk_mean(
    target: pd.DataFrame,
    score: pd.DataFrame,
    group: pd.DataFrame,
    k: int = 3,
    exclude_self: bool = True,
) -> pd.DataFrame:
    target, score, group = _align(target, score, group)
    kk = int(k)
    if kk < 1:
        raise ValueError("group_topk_mean requires k >= 1")
    tv = _as_float(target)
    sv = _as_float(score)
    gv = group.to_numpy()
    rows, cols = tv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        g_row = gv[r]
        positions: dict[Any, list[int]] = {}
        for i in range(cols):
            lab = g_row[i]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            positions.setdefault(lab, []).append(i)
        for i in range(cols):
            lab = g_row[i]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            peers = positions.get(lab)
            if not peers:
                continue
            if exclude_self:
                peers = [p for p in peers if p != i]
            if len(peers) < kk:
                continue
            scores = sv[r, peers]
            targets = tv[r, peers]
            finite = np.isfinite(scores) & np.isfinite(targets)
            if int(finite.sum()) < kk:
                continue
            order = np.argsort(-scores[finite], kind="stable")
            top = targets[finite][order[:kk]]
            out[r, i] = float(np.mean(top))
    return frame_like(target, out)


# ---------------------------------------------------------------------------
# ts_value_at_argextreme(value, score, window, mode, include_current)
# ---------------------------------------------------------------------------
def _ts_value_at_argextreme(
    value: pd.DataFrame,
    score: pd.DataFrame,
    window: int = 20,
    mode: str = "max",
    include_current: bool = False,
) -> pd.DataFrame:
    value, score = _align(value, score)
    w = int(window)
    if w < 2:
        raise ValueError("ts_value_at_argextreme requires window >= 2")
    mode_s = str(mode).lower()
    if mode_s not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")
    vv = _as_float(value)
    sv = _as_float(score)
    rows, cols = vv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    off = 0 if include_current else 1
    for c in range(cols):
        for r in range(rows):
            end = r + 1 - off
            if end <= 0:
                continue
            start = max(0, end - w)
            sseg = sv[start:end, c]
            finite = np.isfinite(sseg)
            if not finite.any():
                continue
            if mode_s == "max":
                pos = int(np.argmax(sseg[finite]))
            else:
                pos = int(np.argmin(sseg[finite]))
            # translate back to absolute row (first occurrence among ties)
            f_idx = np.flatnonzero(finite)[pos]
            val = vv[start + f_idx, c]
            if np.isfinite(val):
                out[r, c] = float(val)
    return frame_like(value, out)


# ---------------------------------------------------------------------------
# cs_weighted_percentile_rank(x, weight)
# ---------------------------------------------------------------------------
def _cs_weighted_percentile_rank(x: pd.DataFrame, weight: pd.DataFrame) -> pd.DataFrame:
    x, weight = _align(x, weight)
    xv = _as_float(x)
    wv = _as_float(weight)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        xr = xv[r]
        wr = wv[r]
        valid = np.isfinite(xr) & np.isfinite(wr) & (wr >= 0.0)
        valid_idx = np.flatnonzero(valid)
        total = float(wr[valid].sum())
        if total <= _EPS:
            continue
        # weighted mid-rank for ties
        order = np.argsort(xr[valid], kind="stable")
        xs = xr[valid][order]
        ws = wr[valid][order]
        n = xs.size
        lower_cum = 0.0
        rank_out = np.full(n, np.nan)
        i = 0
        while i < n:
            j = i
            tie_w = 0.0
            while j < n and xs[j] == xs[i]:
                tie_w += ws[j]
                j += 1
            for t in range(i, j):
                rank_out[t] = (lower_cum + 0.5 * tie_w) / total
            lower_cum += tie_w
            i = j
        pos = np.zeros(n, dtype=int)
        pos[order] = np.arange(n)
        for t in range(n):
            orig = int(valid_idx[pos[t]])
            out[r, orig] = float(rank_out[t])
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# group_distribution_js_divergence(x, group, bins, min_group_size)
# ---------------------------------------------------------------------------
def _group_distribution_js_divergence(
    x: pd.DataFrame,
    group: pd.DataFrame,
    bins: int = 10,
    min_group_size: int = 5,
) -> pd.DataFrame:
    x, group = _align(x, group)
    nb = int(bins)
    if nb < 2:
        raise ValueError("group_distribution_js_divergence requires bins >= 2")
    mg = int(min_group_size)
    if mg < 1:
        raise ValueError("min_group_size must be >= 1")
    xv = _as_float(x)
    gv = group.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)

    def _js(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        eps = _EPS
        kl = 0.0
        for a, b in ((p, m), (q, m)):
            for pi, mi in zip(a, b):
                if pi <= 0.0:
                    continue
                kl += pi * (np.log(pi + eps) - np.log(mi + eps))
        return float(0.5 * kl)

    for r in range(rows):
        xr = xv[r]
        g_row = gv[r]
        market_mask = np.isfinite(xr)
        if int(market_mask.sum()) < 2:
            continue
        market = xr[market_mask]
        # market quantile-bin edges (dedupe ties so each bin has positive width)
        edges = np.quantile(market, np.linspace(0.0, 1.0, nb + 1))
        edges = np.unique(edges)
        n_bins = int(edges.size - 1)
        if n_bins < 1:
            continue
        market_bin = np.histogram(market, bins=edges)[0].astype(float)
        market_p = market_bin / market_bin.sum()
        positions: dict[Any, list[int]] = {}
        for i in range(cols):
            lab = g_row[i]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            if not np.isfinite(xr[i]):
                continue
            positions.setdefault(lab, []).append(i)
        for i in range(cols):
            lab = g_row[i]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            members = positions.get(lab)
            if not members or len(members) < mg:
                continue
            gvals = xr[members]
            g_bin = np.histogram(gvals, bins=edges)[0].astype(float)
            gp = g_bin / g_bin.sum()
            out[r, i] = _js(gp, market_p)
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# event_level_survival_share(event, level, x, history_window, direction)
# ---------------------------------------------------------------------------
def _event_level_survival_share(
    event: pd.DataFrame,
    level: pd.DataFrame,
    x: pd.DataFrame,
    history_window: int = 60,
    direction: str = "up",
) -> pd.DataFrame:
    event, level, x = _align(event, level, x)
    w = int(history_window)
    if w < 2:
        raise ValueError("event_level_survival_share requires history_window >= 2")
    direction_s = str(direction).lower()
    if direction_s not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    ev = _as_float(event)
    lv = _as_float(level)
    xv = _as_float(x)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        cohort: list[tuple[int, float]] = []  # (event_row, level) past events
        for r in range(rows):
            # incorporate the event at r-1 into the cohort
            if r >= 1:
                prev = r - 1
                if np.isfinite(ev[prev, c]) and ev[prev, c] != 0.0 and np.isfinite(lv[prev, c]):
                    cohort.append((prev, float(lv[prev, c])))
            # drop events older than the window
            while cohort and (r - cohort[0][0]) > w:
                cohort.pop(0)
            if not cohort:
                continue
            cur = xv[r, c]
            if not np.isfinite(cur):
                continue
            survived = 0.0
            for _ev_row, lev in cohort:
                if direction_s == "up":
                    if cur > lev:
                        survived += 1.0
                else:
                    if cur < lev:
                        survived += 1.0
            out[r, c] = float(survived / len(cohort))
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# Registration (pandas + polars, daily_panel pattern)
# ---------------------------------------------------------------------------
_DAILY_CANONICALS: tuple[str, ...] = (
    "group_topk_mean",
    "ts_value_at_argextreme",
    "cs_weighted_percentile_rank",
    "group_distribution_js_divergence",
    "event_level_survival_share",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "group_topk_mean": _group_topk_mean,
    "ts_value_at_argextreme": _ts_value_at_argextreme,
    "cs_weighted_percentile_rank": _cs_weighted_percentile_rank,
    "group_distribution_js_divergence": _group_distribution_js_divergence,
    "event_level_survival_share": _event_level_survival_share,
}

_PARAMS: dict[str, list[str]] = {
    "group_topk_mean": ["target", "score", "group", "k", "exclude_self"],
    "ts_value_at_argextreme": ["value", "score", "window", "mode", "include_current"],
    "cs_weighted_percentile_rank": ["x", "weight"],
    "group_distribution_js_divergence": ["x", "group", "bins", "min_group_size"],
    "event_level_survival_share": ["event", "level", "x", "history_window", "direction"],
}

_CATEGORIES: dict[str, str] = {
    "group_topk_mean": "cross_sectional",
    "ts_value_at_argextreme": "time_series_order",
    "cs_weighted_percentile_rank": "cross_sectional",
    "group_distribution_js_divergence": "cross_sectional",
    "event_level_survival_share": "time_series_event",
}


def _register() -> None:
    from cleaned_operators.base import Operator as PandasOperator
    from cleaned_operators.base import OperatorMetadata as PandasMetadata

    for canonical, fn in _KERNELS.items():
        params = _PARAMS[canonical]
        category = _CATEGORIES[canonical]

        class _PandasOp(PandasOperator):
            metadata = PandasMetadata(
                name=canonical,
                category=category,
                description=canonical,
                examples=[],
                param_names=params,
                return_type="series",
                tags=["daily", "panel", "pit_safe", "causal", "deterministic",
                      f"signature:{','.join(params)}->series",
                      "domain:cross_section", "unit:same_as:target", "cost:3"],
            )

            def calculate(self, *args, _fn=fn, **kwargs):
                return _fn(*args, **kwargs)

        OperatorRegistry.register(
            _PandasOp(), canonical=canonical, backend="pandas_numpy",
            source="gather_ext", backend_explicit=True,
        )

        class _PolarsOp(PolarsSeriesOperator):
            metadata = PolarsMetadata(name=canonical, category="gather_ext", param_names=[])

            def _calculate_series(self, *frames, _fn=fn, **params):
                import polars as pl  # noqa: F401
                pdfs = [f.select([c for c in f.columns if c not in _SKIP]).to_pandas() for f in frames]
                out = _fn(*pdfs, **params)
                base = frames[0]
                cols = [c for c in base.columns if c not in _SKIP]
                return base.with_columns(
                    [pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols]
                )

        OperatorRegistry.register(
            _PolarsOp(), canonical=canonical, backend="polars",
            source="gather_ext_polars", backend_explicit=True,
        )

    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_DAILY_CANONICALS)
    )


_SKIP = frozenset({"date", "stock_code"})
_register()

# -*- coding: utf-8 -*-
"""Gemini-recommended gathering / distribution primitives (2026-08-08).

Field-agnostic primitives distilled from the AI operator proposals.  Every
operator is strict-PIT (only the current row and earlier rows), deterministic,
shape-preserving and NaN fail-closed:

* ``group_topk_mean``            — group Top-K mean of ``target`` ranked by an
                                   arbitrary ``score`` (generic elite/leader
                                   routing; group_leader_divergence is a recipe).
* ``ts_value_at_argextreme``     — gather ``value`` at the argmax/argmin of an
                                   arbitrary ``score`` over a trailing window;
                                   ties resolve to the *latest* occurrence
                                   (volume-cluster breakdown, extreme-turnover
                                   valuation states etc.).
* ``cs_weighted_percentile_rank``— weighted empirical CDF rank (weighted
                                   mid-rank ties), ``w_j >= 0``.
* ``group_distribution_js_divergence`` — Jensen-Shannon divergence between a
                                   group's value distribution and a quantile-bin
                                   reference distribution (bounded/symmetric;
                                   deliberately NOT KL, which diverges on empty
                                   bins).  With ``exclude_group_from_reference``
                                   the reference *and its bin edges* are built
                                   from the ex-group cross-section only, so the
                                   group cannot contaminate the bins it is
                                   measured against.
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
            # P1-01: a genuinely misaligned panel must fail loudly, not be
            # silently reindexed to the base.  Reindexing misaligned panels pairs
            # e.g. x_t with score_{t'} or target with a shifted peer — a silent
            # cross-sectional/point-in-time corruption the operator contract
            # forbids.  Callers are expected to pass aligned panels.
            raise ValueError(
                "multi-panel inputs must share identical index/columns; "
                "reindexing misaligned panels is not allowed"
            )
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
            if pd.isna(lab):
                continue
            positions.setdefault(lab, []).append(i)
        for i in range(cols):
            lab = g_row[i]
            if pd.isna(lab):
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
            # Routing universe = peers with a *finite score* ONLY.  A peer whose
            # target is NaN must still compete for a leader slot by its score —
            # filtering on score AND target together silently reroutes the group
            # (a high-score leader with a missing target would be dropped and
            # the top-k mean would describe a different peer set).
            score_finite = np.isfinite(scores)
            if int(score_finite.sum()) < kk:
                continue
            s = scores[score_finite]
            t = targets[score_finite]
            # Top-K with an explicit tie policy at the cutoff: all peers strictly
            # above the k-th score are included in full; the tied group at the
            # cutoff enters with *fractional* weight so the total weight is k.
            # A stable argsort alone would break ties by stock-column order and
            # make the factor change under column permutation.
            order = np.argsort(-s, kind="mergesort")
            kth = float(s[order][kk - 1])
            above = np.flatnonzero(s > kth)
            n_above = int(above.size)
            eq_idx = np.flatnonzero(s == kth)
            n_eq = int(eq_idx.size)
            n_take_eq = min(max(kk - n_above, 0), n_eq)
            # Strict missing policy: any *chosen* leader (fully-above peers plus
            # the portion of the cutoff tie that enters) with a missing target
            # fail-closes the row — never substitute a lower-score peer.
            chosen = above if n_take_eq == 0 else np.concatenate([above, eq_idx[:n_take_eq]])
            if not np.all(np.isfinite(t[chosen])):
                continue
            frac = (n_take_eq / n_eq) if n_eq > 0 else 0.0
            mean_val = (
                float(np.sum(t[above])) + frac * float(np.sum(t[eq_idx]))
            ) / kk
            out[r, i] = float(mean_val)
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
            sub = sseg[finite]
            # Tie policy = LATEST occurrence: argmax/argmin on the reversed
            # slice gives the last row attaining the extreme.  A forward argmax
            # would pick the first (earliest) tied row, so the gathered ``value``
            # would silently depend on the underlying arg-extreme convention
            # instead of an explicit, stable "most recent extreme" (review
            # P1-62a).
            if mode_s == "max":
                pos = int(sub.size - 1 - np.argmax(sub[::-1]))
            else:
                pos = int(sub.size - 1 - np.argmin(sub[::-1]))
            # translate back to absolute row (last occurrence among ties)
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
        # A finite NEGATIVE weight is an invalid state (NonNegativeWeight): it
        # fails the WHOLE row closed (NaN) rather than being silently dropped and
        # the percentile re-normalized over the survivors.  NaN weights stay
        # "missing" and are excluded pairwise (they carry no weight information),
        # but a finite negative weight is a contract violation (P1-02).
        if np.any(np.isfinite(wr) & (wr < 0.0)):
            continue
        valid = np.isfinite(xr) & np.isfinite(wr)
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
        # Correct unsort: rank_out[i] belongs to the i-th *sorted* position, so
        # it must be written back to original index order[i] (not order[rank]).
        unsorted = np.empty(n, dtype=float)
        unsorted[order] = rank_out
        out[r, valid_idx] = unsorted
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# group_distribution_js_divergence(x, group, bins, min_group_size)
# ---------------------------------------------------------------------------
def _group_distribution_js_divergence(
    x: pd.DataFrame,
    group: pd.DataFrame,
    bins: int = 10,
    min_group_size: int = 5,
    exclude_group_from_reference: bool = True,
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
        # Group membership covers ALL labelled members — the group-level JS state
        # is broadcast even to a member whose own ``x`` is missing (the *statistical
        # sample* still requires finite ``x`` below).
        positions: dict[Any, list[int]] = {}
        for i in range(cols):
            lab = g_row[i]
            if pd.isna(lab):
                continue
            positions.setdefault(lab, []).append(i)
        market_mask = np.isfinite(xr)
        if int(market_mask.sum()) < 2:
            continue
        market = xr[market_mask]
        if exclude_group_from_reference:
            # Reference = the market distribution EXCLUDING the current group,
            # and the quantile-bin edges are built from that SAME ex-group
            # reference.  Otherwise a large group (banks / electronics) is
            # mechanically "close to the market" simply because it IS most of
            # the market (self-inclusion bias), and full-market edges would let
            # the group define the very bins it is compared against (reference
            # self-contamination, review P1-62b).  A degenerate reference
            # (< 2 finite rows) fail-closes.
            for lab, members in positions.items():
                gvals = xr[members]
                gvals = gvals[np.isfinite(gvals)]
                if gvals.size < mg:
                    continue
                other = np.ones(cols, dtype=bool)
                other[members] = False
                ref = xr[other & market_mask]
                if ref.size < 2:
                    continue
                ref_edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, nb + 1)))
                if ref_edges.size < 2:
                    continue
                g_bin = np.histogram(gvals, bins=ref_edges)[0].astype(float)
                if g_bin.sum() <= 0:
                    # All group values fall outside the ex-group bin range; the
                    # distribution on these bins is undefined -> fail closed.
                    continue
                gp = g_bin / g_bin.sum()
                r_bin = np.histogram(ref, bins=ref_edges)[0].astype(float)
                rp = r_bin / r_bin.sum()
                val = _js(gp, rp)
                for i in members:
                    out[r, i] = val
        else:
            # Legacy / explicit mode: compare against the whole cross-section
            # (includes the group itself) on full-market edges.
            edges = np.unique(np.quantile(market, np.linspace(0.0, 1.0, nb + 1)))
            if int(edges.size - 1) < 1:
                continue
            market_bin = np.histogram(market, bins=edges)[0].astype(float)
            market_p = market_bin / market_bin.sum()
            for lab, members in positions.items():
                gvals = xr[members]
                gvals = gvals[np.isfinite(gvals)]
                if gvals.size < mg:
                    continue
                g_bin = np.histogram(gvals, bins=edges)[0].astype(float)
                if g_bin.sum() <= 0:
                    continue
                gp = g_bin / g_bin.sum()
                val = _js(gp, market_p)
                for i in members:
                    out[r, i] = val
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
    tolerance: float = 0.0,
) -> pd.DataFrame:
    event, level, x = _align(event, level, x)
    w = int(history_window)
    if w < 2:
        raise ValueError("event_level_survival_share requires history_window >= 2")
    direction_s = str(direction).lower()
    if direction_s not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    tol = float(tolerance)
    if tol < 0.0:
        raise ValueError("tolerance must be >= 0")
    ev = _as_float(event)
    lv = _as_float(level)
    xv = _as_float(x)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        # (event_row, level, running_extreme) — the running extremum tracks
        # min (up) / max (down) of x from the event row (exclusive) to today,
        # so "survived" means the *whole path since the event* stayed on the
        # favorable side, not merely that today's x happens to be there (a dip
        # below the level and recovery would otherwise count as "survived").
        cohort: list[tuple[int, float, float]] = []
        for r in range(rows):
            if r >= 1:
                prev = r - 1
                if np.isfinite(ev[prev, c]) and ev[prev, c] != 0.0 and np.isfinite(lv[prev, c]):
                    if direction_s == "up":
                        cohort.append((prev, float(lv[prev, c]), np.inf))
                    else:
                        cohort.append((prev, float(lv[prev, c]), -np.inf))
            # drop events older than the window
            while cohort and (r - cohort[0][0]) > w:
                cohort.pop(0)
            if not cohort:
                continue
            cur = xv[r, c]
            if not np.isfinite(cur):
                continue
            survived = 0.0
            for k in range(len(cohort)):
                e_row, lev, ext = cohort[k]
                if direction_s == "up":
                    ext = min(ext, cur)
                else:
                    ext = max(ext, cur)
                cohort[k] = (e_row, lev, ext)
                # Inclusive survival: x_t == level (e.g. a stock that touches the
                # limit price) is still "alive" in A-share limit semantics; a
                # small tick tolerance avoids floating-point boundary flips.
                if direction_s == "up":
                    if ext >= lev - tol:
                        survived += 1.0
                else:
                    if ext <= lev + tol:
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
    "group_distribution_js_divergence": ["x", "group", "bins", "min_group_size", "exclude_group_from_reference"],
    "event_level_survival_share": ["event", "level", "x", "history_window", "direction", "tolerance"],
}

_CATEGORIES: dict[str, str] = {
    "group_topk_mean": "cross_sectional",
    "ts_value_at_argextreme": "time_series_order",
    "cs_weighted_percentile_rank": "cross_sectional",
    "group_distribution_js_divergence": "cross_sectional",
    "event_level_survival_share": "time_series_event",
}

# Output unit per operator: only group_topk_mean / ts_value_at_argextreme inherit
# the target's unit; rank, divergence (nats) and probability must not be tagged
# ``same_as:target`` or typed search / unit algebra will be polluted.
_UNITS: dict[str, str] = {
    "group_topk_mean": "same_as:target",
    "ts_value_at_argextreme": "same_as:target",
    "cs_weighted_percentile_rank": "dimensionless",
    "group_distribution_js_divergence": "dimensionless",
    "event_level_survival_share": "probability",
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
                      "domain:cross_section", f"unit:{_UNITS[canonical]}", "cost:3"],
            )

            _HANDLES_CALL_CONTRACT = True  # R5-02: routes through validate_operator_call

            def calculate(self, *args, _fn=fn, **kwargs):
                # R5-02: this module registered ``calculate`` directly without
                # routing through the central validator, so integer / panel-axis
                # / unknown-kwarg checks were bypassed for every gather_* op.
                from cleaned_operators.base import validate_operator_call

                processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
                return _fn(*processed_args, **processed_kwargs)

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

    _surface.extend_extended_only(set(_DAILY_CANONICALS))


_SKIP = frozenset({"date", "stock_code"})
_register()

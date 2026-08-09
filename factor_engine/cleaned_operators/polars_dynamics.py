# -*- coding: utf-8 -*-
"""Genuine Polars UDF backends for the 2026-08 V2/V3 dynamics families.

The kernels are per-instrument sequential (ordinal patterns, state transitions,
first-passage stopping times, event response, energy distance, Hill tail), which
plain ``pl.Expr`` window functions cannot express.  These backends run the shared
numpy kernel per instrument column via Polars-native column UDFs
(``Series.to_numpy()`` → kernel → ``pl.Series``) — no pandas DataFrame
round-trip, no per-row Python loop in the data plumbing, and exact
pandas-reference parity by construction.  Cross-sectional (cs_knn_*) and group
(group_corr_*) and minute→daily (volume_clock / session_recovery) operators stay
pandas_numpy-only: their aggregation is cross-column, not per-column.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.markov_dynamics import (
    _state_dynamics_series,
    _observed_reachable,
    _MIN_COUNT_RELATIONAL,
    _MIN_PERIODS_RELATIONAL,
)
from cleaned_operators.state_geometry import (
    _state_density_series,
    _irreversibility_series,
)
from cleaned_operators.first_passage import _first_passage_series, _fp_stats_series
from cleaned_operators.event_response import _horizon_response, _response_curve_stats_series
from cleaned_operators.distribution_break import _joint_shift_cell, _robust_z
from cleaned_operators.extreme_tail import (
    _gpd_shape_pwm_series,
    _hill_series,
    _mean_excess_slope_series,
    _extremal_index_series,
)
from cleaned_operators.report_timing import _delay_surprise_series
from cleaned_operators.advanced_information import _te_peak_window

_SKIP_PANEL = frozenset({"date", "stock_code"})
_EPS = 1e-12


def _cols(fr: pl.DataFrame) -> list[str]:
    return [c for c in fr.columns if c not in _SKIP_PANEL]


def _col(fr: pl.DataFrame, name: str) -> np.ndarray:
    return fr.select(name).to_series().to_numpy().astype(np.float64)


def _rebuild(base: pl.DataFrame, data: dict[str, np.ndarray]) -> pl.DataFrame:
    return base.with_columns(
        [pl.Series(name=c, values=np.asarray(v, dtype=np.float64)) for c, v in data.items()]
    )


def _mk(canonical: str, description: str, params: list[str], fn):
    metadata = OperatorMetadata(
        name=canonical,
        category="state_dynamics",
        description=description,
        param_names=params,
        return_type="series",
        tags=["polars", "daily", "native_udf", "typed_v2"],
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"PolarsDynamics_{canonical}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=canonical,
        category="state_dynamics",
        business_category="state_dynamics",
        canonical=canonical,
        source="polars_dynamics",
    )(cls)
    return cls


# ---------------------------------------------------------------------------
# Local Markov dynamics (shared DiscreteStateDynamicsKernel).
# ---------------------------------------------------------------------------

def _markov_series(series: np.ndarray, window: int, bins: int, lag: int, min_count: int, kind: str, target: str = "upper") -> np.ndarray:
    res = _state_dynamics_series(series, int(window), int(bins), int(lag), int(min_count))
    B = res["P"].shape[1]
    n = series.shape[0]
    out = np.full(n, np.nan)
    log_b = np.log(max(2, B))
    for t in range(n):
        k = res["state"][t]
        if not np.isfinite(k):
            continue
        k = int(k)
        if kind in {"persistence", "state_entropy"}:
            # R11 round-2 P0 (TASK 2, mirrors markov_dynamics): a window that
            # observed only a single state has no estimated transition
            # distribution — persistence ~1 / entropy ~0 there are degenerate.
            if res["n_states_obs"][t] < 2:
                continue
            if res["counts"][t, k] < min_count:
                continue
            if kind == "persistence":
                # TASK 1 reverse: never-entered state -> no persistence to report.
                if res["N_obs"][t, :, k].sum() <= 0:
                    continue
                p = res["P"][t, k, k]
                if np.isfinite(p):
                    out[t] = float(p)
            else:
                row = res["P"][t, k]
                # TASK 1 reverse: only destinations with observed incoming
                # transitions are real outcomes (mirrors markov_dynamics).
                col_support = res["N_obs"][t].sum(axis=0) > 0
                row = row[np.isfinite(row) & col_support]
                # Renormalise over the observed support and require at least two
                # estimable destinations before calling it a distribution.
                if row.size < 2:
                    continue
                s = float(row.sum())
                if not np.isfinite(s) or s <= _EPS:
                    continue
                row = row / s
                h = -float(np.sum(row * np.log(row)))
                out[t] = float(h / log_b)
        elif kind == "surprisal":
            edges = res["edges"][t]
            if not np.isfinite(edges).any() or t - lag < 0:
                continue
            i_val = series[t - lag]
            if not np.isfinite(i_val):
                continue
            i = int(np.clip(np.searchsorted(edges, i_val, side="left") - 1, 0, B - 1))
            # R11 round-2 P0 (TASK 2, mirrors markov_dynamics).
            if res["n_states_obs"][t] < 2:
                continue
            if res["counts"][t, i] < min_count:
                continue
            p = res["P"][t, i, k]
            if np.isfinite(p) and p > 0.0:
                out[t] = float(-np.log(p))
        elif kind == "local_stability":
            if not (0 < k < B - 1):
                continue
            d1 = res["D1"][t]
            c = res["centers"][t]
            if (
                not np.isfinite(d1[k - 1]) or not np.isfinite(d1[k]) or not np.isfinite(d1[k + 1])
                or not np.isfinite(c[k - 1]) or not np.isfinite(c[k + 1])
            ):
                continue
            if res["counts"][t, k] < min_count:
                continue
            denom = c[k + 1] - c[k - 1]
            if abs(denom) <= _EPS:
                continue
            out[t] = float(-(d1[k + 1] - d1[k - 1]) / denom)
        elif kind == "committor":
            # R11 round-2 P0 (TASK 2, mirrors markov_dynamics).
            if res["n_states_obs"][t] < 2:
                continue
            if res["counts"][t, k] < min_count:
                continue
            P = res["P"][t]
            if not np.all(np.isfinite(P)):
                continue
            # audit P1 empirical-support gate (mirrors markov_dynamics): the
            # upper target must be visited, reachable via observed edges, and
            # the window must hold a minimum observed edge count.
            N_obs = res["N_obs"][t]
            if int(np.count_nonzero(N_obs)) < 2:
                continue
            if int(res["counts"][t, B - 1]) < 1:
                continue
            if not _observed_reachable(N_obs, k, B - 1):
                continue
            if B == 2:
                out[t] = float(k)
                continue
            interior = list(range(1, B - 1))
            if not interior:
                continue
            A = np.eye(len(interior)) - P[np.ix_(interior, interior)]
            try:
                q = np.linalg.solve(A, P[interior, B - 1])
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(q)):
                continue
            if k == 0:
                out[t] = 0.0
            elif k == B - 1:
                out[t] = 1.0
            else:
                out[t] = float(np.clip(q[k - 1], 0.0, 1.0))
        elif kind == "mfpt":
            # R11 round-2 P0 (TASK 2, mirrors markov_dynamics).
            if res["n_states_obs"][t] < 2:
                continue
            if res["counts"][t, k] < min_count:
                continue
            P = res["P"][t]
            if not np.all(np.isfinite(P)):
                continue
            if target == "upper":
                A = {B - 1}
            elif target == "lower":
                A = {0}
            elif target == "extreme":
                A = {0, B - 1}
            else:
                raise ValueError(f"unknown target {target!r}")
            if k in A:
                out[t] = 0.0
                continue
            # audit P1 empirical-support gate (mirrors markov_dynamics): minimum
            # observed edge count, at least one target visited, and an observed-
            # support path from the current state to some target.
            N_obs = res["N_obs"][t]
            if int(np.count_nonzero(N_obs)) < 2:
                continue
            if not any(int(res["counts"][t, a]) >= 1 for a in A):
                continue
            if not any(_observed_reachable(N_obs, k, a) for a in A):
                continue
            nonA = [i for i in range(B) if i not in A]
            if not nonA:
                continue
            M = np.eye(len(nonA)) - P[np.ix_(nonA, nonA)]
            try:
                m = np.linalg.solve(M, np.ones(len(nonA)))
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(m)):
                continue
            out[t] = float(max(m[nonA.index(k)], 0.0)) * lag
        elif kind == "spectral_gap":
            # R11 round-2 P0 (TASK 2, mirrors markov_dynamics).
            if res["n_states_obs"][t] < 2:
                continue
            if res["total_trans"][t] < min_count:
                continue
            P = res["P"][t]
            if not np.all(np.isfinite(P)):
                continue
            try:
                ev = np.linalg.eigvals(P)
            except np.linalg.LinAlgError:
                continue
            if ev.size == 0 or not np.all(np.isfinite(ev)):
                continue
            mag = np.abs(ev)
            order = np.argsort(mag)[::-1]
            second = float(mag[order[1]]) if ev.size >= 2 else 0.0
            out[t] = float(np.clip(1.0 - second, 0.0, 1.0))
        elif kind == "stationary_surprisal":
            # R11 round-2 P0 (TASK 2 + TASK 1 reverse, mirrors markov_dynamics).
            if res["n_states_obs"][t] < 2:
                continue
            if res["counts"][t, k] < min_count:
                continue
            if res["N_obs"][t, :, k].sum() <= 0:
                continue
            pi = res["pi"][t]
            if not np.all(np.isfinite(pi)):
                continue
            out[t] = float(-np.log(float(pi[k]) + _EPS))
        elif kind == "equilibrium":
            if res["counts"][t, k] < min_count:
                continue
            d1 = res["D1"][t]
            c = res["centers"][t]
            if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(c)):
                continue
            xstar = None
            for m in range(B - 1):
                a, b = d1[m], d1[m + 1]
                if np.isnan(a) or np.isnan(b):
                    continue
                if a > 0.0 and b < 0.0:
                    denom = a - b
                    xstar = c[m] + (c[m + 1] - c[m]) * (a / denom) if abs(denom) > _EPS else c[m]
                    break
            if xstar is None:
                continue
            lo = max(0, t - int(window))
            past = series[lo:t]
            finite = past[np.isfinite(past)]
            mad = 1.0
            if finite.size >= 4:
                med = np.median(finite)
                madv = np.median(np.abs(finite - med))
                if np.isfinite(madv) and madv > _EPS:
                    mad = madv
                else:
                    span = float(np.nanmax(finite) - np.nanmin(finite))
                    if np.isfinite(span) and span > _EPS:
                        mad = span
            xt = series[t]
            if not np.isfinite(xt):
                continue
            out[t] = float((xt - xstar) / mad)
        elif kind == "diffusion_gradient":
            if not (0 < k < B - 1):
                continue
            d2 = res["D2"][t]
            c = res["centers"][t]
            if (
                not np.isfinite(d2[k - 1]) or not np.isfinite(d2[k + 1])
                or not np.isfinite(c[k - 1]) or not np.isfinite(c[k + 1])
            ):
                continue
            if res["counts"][t, k] < min_count:
                continue
            denom = c[k + 1] - c[k - 1]
            if abs(denom) <= _EPS:
                continue
            out[t] = float((d2[k + 1] - d2[k - 1]) / denom)
        elif kind == "quasipotential":
            if res["counts"][t, k] < min_count:
                continue
            d1 = res["D1"][t]
            d2 = res["D2"][t]
            c = res["centers"][t]
            if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(d2)) or not np.all(np.isfinite(c)):
                continue
            if B < 3:
                continue
            dx = np.diff(c)
            U = np.zeros(B)
            for m in range(1, B):
                U[m] = U[m - 1] - (d1[m - 1] / (d2[m - 1] + _EPS)) * dx[m - 1]
            mins = [
                m for m in range(B)
                if (m == 0 or U[m] <= U[m - 1]) and (m == B - 1 or U[m] <= U[m + 1])
            ]
            maxs = [
                m for m in range(B)
                if (m == 0 or U[m] >= U[m - 1]) and (m == B - 1 or U[m] >= U[m + 1])
            ]
            if not mins:
                continue
            well = min(mins, key=lambda m: abs(m - k))
            left = [m for m in maxs if m < well]
            right = [m for m in maxs if m > well]
            levels: list[float] = []
            if left:
                levels.append(float(max(U[m] for m in left)))
            if right:
                levels.append(float(max(U[m] for m in right)))
            if not levels:
                continue
            depth = float(min(levels)) - float(U[well])
            if np.isfinite(depth) and depth > _EPS:
                out[t] = depth
    return out


def _markov_family(frame, window, bins, lag, min_count, kind, target="upper"):
    cols = _cols(frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = _markov_series(_col(frame, c), window, bins, lag, min_count, kind, target)
    return _rebuild(frame, out)


for _name, _desc, _kind, _bins in (
    ("ts_markov_persistence", "当前状态的历史保持概率 P_kk（Polars）。", "persistence", 3),
    ("ts_markov_state_entropy", "当前状态转移行熵 H_k/log(B)（Polars）。", "state_entropy", 3),
    ("ts_markov_transition_surprisal", "当前状态跳变罕见度 -log P_ij（Polars）。", "surprisal", 3),
    ("ts_kramers_moyal_local_stability", "当前状态局部稳定性 -D'(x_k)（Polars）。", "local_stability", 5),
):
    _cls = _mk(
        _name, _desc, ["x", "window", "bins", "lag", "min_count"],
        lambda frame, window=60, bins=_bins, lag=1, min_count=3, _kind=_kind: _markov_family(
            frame, window, bins, lag, min_count, _kind
        ),
    )
    # R11 round-2 P0 (TASK 3): binding-time relational feasibility mirror of the
    # pandas metadata (min_count <= window - lag).
    _cls.metadata.relational_specs = list(_MIN_COUNT_RELATIONAL)

for _name, _desc, _kind, _bins in (
    ("ts_markov_committor", "当前状态先达上边界概率 q_k（Polars）。", "committor", 3),
    ("ts_markov_mean_first_passage_time", "当前状态到 target 集平均首达时间（Polars）。", "mfpt", 3),
    ("ts_markov_spectral_gap", "转移矩阵谱隙 1-|λ2|（Polars）。", "spectral_gap", 3),
    ("ts_markov_stationary_surprisal", "当前状态长期稀有度 -log π_k（Polars）。", "stationary_surprisal", 3),
    ("ts_km_equilibrium_distance", "当前值相对 D1 吸引子距离（Polars）。", "equilibrium", 5),
    ("ts_km_diffusion_gradient", "当前状态扩散梯度 dD2/dx（Polars）。", "diffusion_gradient", 5),
    ("ts_km_quasipotential_depth", "当前状态势阱深度（Polars）。", "quasipotential", 5),
):
    if _kind == "mfpt":
        _cls = _mk(
            _name, _desc, ["x", "window", "bins", "lag", "min_count", "target"],
            lambda frame, window=60, bins=_bins, lag=1, min_count=3, target="upper": _markov_family(
                frame, window, bins, lag, min_count, "mfpt", target
            ),
        )
        _cls.metadata.relational_specs = list(_MIN_COUNT_RELATIONAL)
    elif _kind == "spectral_gap":
        # Pandas ts_markov_spectral_gap guards on min_periods (default 5), not
        # min_count; keep the Polars parameter parity exact.
        _cls = _mk(
            _name, _desc, ["x", "window", "bins", "lag", "min_periods"],
            lambda frame, window=120, bins=_bins, lag=1, min_periods=5: _markov_family(
                frame, window, bins, lag, min_periods, "spectral_gap"
            ),
        )
        _cls.metadata.relational_specs = list(_MIN_PERIODS_RELATIONAL)
    elif _kind == "quasipotential":
        # Pandas ts_km_quasipotential_depth defaults to window=120.
        _cls = _mk(
            _name, _desc, ["x", "window", "bins", "lag", "min_count"],
            lambda frame, window=120, bins=_bins, lag=1, min_count=3: _markov_family(
                frame, window, bins, lag, min_count, "quasipotential"
            ),
        )
        _cls.metadata.relational_specs = list(_MIN_COUNT_RELATIONAL)
    else:
        _cls = _mk(
            _name, _desc, ["x", "window", "bins", "lag", "min_count"],
            lambda frame, window=60, bins=_bins, lag=1, min_count=3, _kind=_kind: _markov_family(
                frame, window, bins, lag, min_count, _kind
            ),
        )
        _cls.metadata.relational_specs = list(_MIN_COUNT_RELATIONAL)


# ---------------------------------------------------------------------------
# State geometry.
# ---------------------------------------------------------------------------

def _single_kernel(frame, kernel_fn):
    cols = _cols(frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = kernel_fn(_col(frame, c))
    return _rebuild(frame, out)


_mk(
    "ts_state_density",
    "当前状态的历史 Epanechnikov 密度（Polars）。",
    ["x", "window", "bandwidth", "min_periods"],
    lambda frame, window=60, bandwidth=1.0, min_periods=5: _single_kernel(
        frame, lambda s: _state_density_series(s, window, bandwidth, min_periods)
    ),
)
_mk(
    "ts_ordinal_irreversibility",
    "序数模式时间不可逆性 sqrt(JS/ln2)（Polars）。",
    ["x", "window", "order", "delay", "min_patterns"],
    lambda frame, window=60, order=3, delay=1, min_patterns=5: _single_kernel(
        frame, lambda s: _irreversibility_series(s, window, order, delay, min_patterns)
    ),
)


# ---------------------------------------------------------------------------
# First-passage bias (x + scale pair).
# ---------------------------------------------------------------------------

def _first_passage_pair(x_frame, scale_frame, window, barrier, horizon, min_anchors):
    cols = _cols(x_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = _first_passage_series(
            _col(x_frame, c), _col(scale_frame, c), window, barrier, horizon, min_anchors
        )
    return _rebuild(x_frame, out)


_mk(
    "ts_first_passage_bias",
    "首达偏向 mean(d_s*w_s)（Polars）。",
    ["x", "scale", "window", "barrier", "horizon", "min_anchors"],
    lambda x, scale, window=120, barrier=1.0, horizon=10, min_anchors=3: _first_passage_pair(
        x, scale, window, barrier, horizon, min_anchors
    ),
)


# ---------------------------------------------------------------------------
# Historical event response.
# ---------------------------------------------------------------------------

def _event_response_pair(response_frame, event_frame, history_window, horizon, mode, min_events, sign_balance, require_full_horizon=True, refractory=0):
    cols = _cols(response_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = _horizon_response(
            _col(response_frame, c), _col(event_frame, c),
            history_window, horizon, mode, min_events, sign_balance,
            require_full_horizon=bool(require_full_horizon),
            refractory=refractory,
        )
    return _rebuild(response_frame, out)


_mk(
    "event_historical_response_mean",
    "历史事件平均 horizon 响应（Polars）。",
    ["response", "event", "history_window", "horizon", "mode", "min_events", "require_full_horizon", "refractory"],
    lambda response, event, history_window=120, horizon=5, mode="mean", min_events=5, require_full_horizon=True, refractory=0: _event_response_pair(
        response, event, history_window, horizon, mode, min_events, False, require_full_horizon, refractory,
    ),
)
_mk(
    "event_historical_response_sign_balance",
    "历史事件响应符号平衡（Polars）。",
    ["response", "event", "history_window", "horizon", "min_events", "require_full_horizon", "refractory"],
    lambda response, event, history_window=120, horizon=5, min_events=5, require_full_horizon=True, refractory=0: _event_response_pair(
        response, event, history_window, horizon, "mean", min_events, True, require_full_horizon, refractory,
    ),
)


# ---------------------------------------------------------------------------
# Multivariate distribution break.
# ---------------------------------------------------------------------------

def _joint_energy_family(f1, f2, f3, recent, prior, break_score, window):
    cols = _cols(f1)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = _col(f1, c)
        y = _col(f2, c)
        z = _col(f3, c)
        block = np.stack([x, y, z], axis=1)          # (n, 3)
        n = block.shape[0]
        shifts = np.full(n, np.nan)
        for t in range(n):
            lo = max(0, t - (recent + prior) + 1)
            if t - lo + 1 < recent + prior:
                continue
            shifts[t] = _joint_shift_cell(block[lo : t + 1], recent, prior)
        if break_score:
            shifts = _robust_z(shifts, window, 3)
        out[c] = shifts
    return _rebuild(f1, out)


_mk(
    "ts_joint_energy_shift",
    "多变量 Energy distance 联合分布漂移（Polars）。",
    ["f1", "f2", "f3", "recent_window", "prior_window"],
    lambda f1, f2, f3, recent_window=20, prior_window=60: _joint_energy_family(
        f1, f2, f3, recent_window, prior_window, False, 20
    ),
)
_mk(
    "ts_energy_break_score",
    "联合分布 break 得分（Polars）。",
    ["f1", "f2", "f3", "window", "recent_window", "prior_window"],
    lambda f1, f2, f3, window=60, recent_window=20, prior_window=60: _joint_energy_family(
        f1, f2, f3, recent_window, prior_window, True, window
    ),
)


# ---------------------------------------------------------------------------
# Extreme tail + report timing.
# ---------------------------------------------------------------------------

_mk(
    "ts_hill_tail_index",
    "Hill 尾部指数 ξ（Polars）。",
    ["x", "window", "side", "tail_fraction", "min_tail_count"],
    lambda frame, window=120, side="upper", tail_fraction=0.2, min_tail_count=10: _single_kernel(
        frame, lambda s: _hill_series(s, window, side, tail_fraction, min_tail_count)
    ),
)
def _delay_surprise_pair(delay_frame, event_frame, window, min_periods):
    cols = _cols(delay_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        ev = None if event_frame is None else _col(event_frame, c)
        out[c] = _delay_surprise_series(_col(delay_frame, c), window, min_periods, ev)
    return _rebuild(delay_frame, out)


_mk(
    "report_filing_delay_surprise",
    "披露延迟稳健 z（Polars）。",
    ["delay", "filing_event", "window", "min_periods"],
    lambda frame, filing_event=None, window=8, min_periods=3: _delay_surprise_pair(
        frame, filing_event, window, min_periods
    ),
)


# ---------------------------------------------------------------------------
# First-passage decomposition (hit probability / conditional time).
# ---------------------------------------------------------------------------

def _fp_stats_pair(x_frame, scale_frame, window, barrier, horizon, min_anchors, which):
    cols = _cols(x_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        up, dn, up_ct, dn_ct = _fp_stats_series(
            _col(x_frame, c), _col(scale_frame, c), window, barrier, horizon, min_anchors
        )
        out[c] = (up, dn, up_ct, dn_ct)[which]
    return _rebuild(x_frame, out)


_mk(
    "ts_first_passage_hit_probability",
    "历史首达命中概率（Polars）。",
    ["x", "scale", "window", "barrier", "horizon", "min_anchors", "side"],
    lambda x, scale, window=120, barrier=1.0, horizon=10, min_anchors=3, side="upper": _fp_stats_pair(
        x, scale, window, barrier, horizon, min_anchors, 0 if str(side).lower() == "upper" else 1
    ),
)
_mk(
    "ts_first_passage_conditional_time",
    "命中条件下的平均首达时间（Polars）。",
    ["x", "scale", "window", "barrier", "horizon", "min_anchors", "side"],
    lambda x, scale, window=120, barrier=1.0, horizon=10, min_anchors=3, side="upper": _fp_stats_pair(
        x, scale, window, barrier, horizon, min_anchors, 2 if str(side).lower() == "upper" else 3
    ),
)


# ---------------------------------------------------------------------------
# Event-response curve shape (peak lag / decay / dispersion / reversal).
# ---------------------------------------------------------------------------

def _response_curve_pair(response_frame, event_frame, history_window, horizon, min_events, which, refractory=0):
    cols = _cols(response_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        pl_, dec, disp, rev = _response_curve_stats_series(
            _col(response_frame, c), _col(event_frame, c), history_window, horizon, min_events,
            refractory=refractory,
        )
        out[c] = (pl_, dec, disp, rev)[which]
    return _rebuild(response_frame, out)


_mk(
    "event_response_peak_lag",
    "平均响应曲线峰值时滞（Polars）。",
    ["response", "event", "history_window", "horizon", "min_events"],
    lambda response, event, history_window=120, horizon=10, min_events=3: _response_curve_pair(
        response, event, history_window, horizon, min_events, 0
    ),
)
_mk(
    "event_response_decay_rate",
    "平均响应曲线衰减速率（Polars）。",
    ["response", "event", "history_window", "horizon", "min_events"],
    lambda response, event, history_window=120, horizon=10, min_events=3: _response_curve_pair(
        response, event, history_window, horizon, min_events, 1
    ),
)
_mk(
    "event_response_dispersion",
    "事件响应分布离散度（Polars）。",
    ["response", "event", "history_window", "horizon", "min_events", "refractory"],
    lambda response, event, history_window=120, horizon=10, min_events=3, refractory=0: _response_curve_pair(
        response, event, history_window, horizon, min_events, 2, refractory
    ),
)
_mk(
    "event_response_reversal_strength",
    "响应曲线首尾反转强度（Polars）。",
    ["response", "event", "history_window", "horizon", "min_events"],
    lambda response, event, history_window=120, horizon=10, min_events=3: _response_curve_pair(
        response, event, history_window, horizon, min_events, 3
    ),
)


# ---------------------------------------------------------------------------
# Transfer-entropy peak (fused).
# ---------------------------------------------------------------------------

def _te_peak_pair(target_frame, source_frame, window, bins, min_transitions, min_cells_ratio, which):
    w = max(2, int(window))
    cols = _cols(target_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        tv = _col(target_frame, c)
        sv = _col(source_frame, c)
        n = tv.shape[0]
        res = np.full(n, np.nan)
        for r in range(n):
            lo = max(0, r - w + 1)
            peak, lag = _te_peak_window(
                tv[lo : r + 1], sv[lo : r + 1], bins, min_transitions, min_cells_ratio
            )
            res[r] = peak if which == 0 else lag
        out[c] = res
    return _rebuild(target_frame, out)


_mk(
    "ts_transfer_entropy_peak_strength",
    "TE 在 lag∈{1,2,3,5,10} 的峰值（Polars）。",
    ["target", "source", "window", "bins", "min_transitions", "min_cells_ratio"],
    lambda target, source, window=60, bins=3, min_transitions=None, min_cells_ratio=1.0: _te_peak_pair(
        target, source, window, bins, 30 if min_transitions is None else int(min_transitions), min_cells_ratio, 0
    ),
)
_mk(
    "ts_transfer_entropy_peak_lag",
    "TE 峰值时滞（Polars）。",
    ["target", "source", "window", "bins", "min_transitions", "min_cells_ratio"],
    lambda target, source, window=60, bins=3, min_transitions=None, min_cells_ratio=1.0: _te_peak_pair(
        target, source, window, bins, 30 if min_transitions is None else int(min_transitions), min_cells_ratio, 1
    ),
)


# ---------------------------------------------------------------------------
# Extreme-value tail shape (extremal index / mean-excess slope / GPD-PWM).
# ---------------------------------------------------------------------------

_mk(
    "ts_extremal_index",
    "极值指数 θ = 簇数/超阈次数（Polars）。",
    ["x", "window", "side", "q", "min_exceed", "run_length"],
    lambda frame, window=120, side="upper", q=0.9, min_exceed=3, run_length=1: _single_kernel(
        frame, lambda s: _extremal_index_series(s, window, side, q, min_exceed, run_length)
    ),
)
_mk(
    "ts_mean_excess_slope",
    "mean-excess 图斜率（Polars）。",
    ["x", "window", "side", "min_tail_count"],
    lambda frame, window=120, side="upper", min_tail_count=5: _single_kernel(
        frame, lambda s: _mean_excess_slope_series(s, window, side, min_tail_count)
    ),
)
_mk(
    "ts_gpd_shape_pwm",
    "GPD 形状参数 ξ（PWM，Polars）。",
    ["x", "window", "side", "tail_fraction", "min_tail_count"],
    lambda frame, window=120, side="upper", tail_fraction=0.2, min_tail_count=10: _single_kernel(
        frame, lambda s: _gpd_shape_pwm_series(s, window, side, tail_fraction, min_tail_count)
    ),
)

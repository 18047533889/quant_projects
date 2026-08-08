# -*- coding: utf-8 -*-
"""State-episode excursion operators (2026-08 geometry/math expansion).

The family measures, within each *episode* of a state panel, how a price series
``x`` travels relative to the episode entry price.  A state value ``s`` is
mapped to a direction ``+1`` when ``s > 1e-9``, ``-1`` when ``s < -1e-9``, and
to a **break** when ``|s| <= 1e-9`` or ``s`` is non-finite.  An *episode* is a
maximal run of rows with the same non-zero direction; it starts at entry index
``e`` with entry price ``x_e``, and any break/zero row ends it (the output is
``NaN`` on those rows).

For every row ``t`` inside an episode with direction ``dir``:

* ``P_t = dir * (x_t - x_e)``
* ``MFE_t = max_{k<=t} P_k``   (maximum favourable excursion)
* ``MAE_t = max_{k<=t} -P_k``  (maximum adverse excursion)

The five operators normalise these building blocks into unit-free ratios
(MFE/MAE scale to an entry ``scale``; efficiency, retrace and balance are pure
ratios).  All operators are per-column, prefix-causal, deterministic and
NaN-safe; rows outside an episode are ``NaN``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="state_episode_excursion",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "state_episode_excursion", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:state_episode",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared episode kernel
# ---------------------------------------------------------------------------
def _episode_map(x: np.ndarray, state: np.ndarray):
    """Per-column episode decomposition.

    Returns ``(sign, P, MFE, MAE, path, entry)``:
    ``sign`` (+1/-1/0), ``P`` (signed excursion), ``MFE`` / ``MAE`` (running
    favourable / adverse max within the episode), ``path`` (cumulative
    ``|x_k - x_{k-1}|`` from the entry row) and ``entry`` (episode entry index;
    ``-1`` outside any episode).  Rows that are breaks, have a non-finite price,
    or belong to an episode whose entry price is non-finite are ``NaN``/``-1``.

    A non-finite price also *breaks* the episode: the row is unmeasurable (NaN)
    and the next finite price starts a fresh episode — an episode never
    continues across a data gap, so MFE/MAE/path can not span missing prices
    (R4-06).
    """
    n = len(x)
    sign = np.zeros(n, dtype=np.int8)
    finite = np.isfinite(state)
    sign[finite & (state > 1e-9)] = 1
    sign[finite & (state < -1e-9)] = -1

    P = np.full(n, np.nan, dtype=float)
    MFE = np.full(n, np.nan, dtype=float)
    MAE = np.full(n, np.nan, dtype=float)
    path = np.full(n, np.nan, dtype=float)
    entry = np.full(n, -1, dtype=int)

    cur_dir = 0
    cur_entry = -1
    run_max = 0.0
    run_max_neg = 0.0
    cum_path = 0.0
    last_fin = np.nan  # last finite price inside the current episode
    broken = False
    for t in range(n):
        if sign[t] == 0:
            cur_dir = 0
            cur_entry = -1
            run_max = 0.0
            run_max_neg = 0.0
            cum_path = 0.0
            last_fin = np.nan
            broken = False
            continue
        if sign[t] != cur_dir or broken:
            # New direction, or the previous row broke the episode (non-finite
            # price): start a fresh episode here instead of continuing across a
            # data gap (R4-06).
            cur_dir = sign[t]
            cur_entry = t
            run_max = 0.0
            run_max_neg = 0.0
            cum_path = 0.0
            last_fin = np.nan
            broken = not np.isfinite(x[t])
        e = cur_entry
        entry[t] = e
        if not np.isfinite(x[t]):
            # Non-finite price at the entry or mid-episode: the row is NaN and
            # the episode is broken — the next finite price restarts it.
            broken = True
            continue
        if broken:
            continue
        # Value-contiguous arc length: since a non-finite price breaks the
        # episode, every interior step here is a consecutive finite price, so
        # ``path >= |x_t - x_e|`` always (triangle inequality) and efficiency
        # can never exceed 1 — P0-16.
        if t > e and np.isfinite(last_fin):
            cum_path += abs(x[t] - last_fin)
        last_fin = x[t]
        Pt = cur_dir * (x[t] - x[e])
        P[t] = Pt
        run_max = run_max if run_max >= Pt else Pt
        run_max_neg = run_max_neg if run_max_neg >= -Pt else -Pt
        MFE[t] = run_max
        MAE[t] = run_max_neg
        path[t] = cum_path
    return sign, P, MFE, MAE, path, entry


def _episode_scale_entry(scale: np.ndarray, e: int) -> float:
    """Entry-row scale only.

    An episode whose entry scale is missing/invalid has an undefined
    entry-normalised excursion — there is no later fallback (P0-15).  The
    callers test ``np.isfinite`` on the result and emit NaN.
    """
    if e < 0:
        return np.nan
    se = scale[e]
    return float(se) if np.isfinite(se) else np.nan


# ---------------------------------------------------------------------------
# per-operator kernels
# ---------------------------------------------------------------------------
def _mfe_series(x2d: np.ndarray, s2d: np.ndarray, scale2d: np.ndarray) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        sign, _P, MFE, _MAE, _path, entry = _episode_map(x2d[:, c], s2d[:, c])
        sc = scale2d[:, c]
        for t in range(rows):
            if sign[t] == 0:
                continue
            se = _episode_scale_entry(sc, entry[t])
            if not np.isfinite(se):
                continue
            out[t, c] = MFE[t] / (se + _EPS)
    return out


def _mae_series(x2d: np.ndarray, s2d: np.ndarray, scale2d: np.ndarray) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        sign, _P, _MFE, MAE, _path, entry = _episode_map(x2d[:, c], s2d[:, c])
        sc = scale2d[:, c]
        for t in range(rows):
            if sign[t] == 0:
                continue
            se = _episode_scale_entry(sc, entry[t])
            if not np.isfinite(se):
                continue
            out[t, c] = MAE[t] / (se + _EPS)
    return out


def _efficiency_series(x2d: np.ndarray, s2d: np.ndarray) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        x = x2d[:, c]
        sign, _P, _MFE, _MAE, path, entry = _episode_map(x, s2d[:, c])
        for t in range(rows):
            if sign[t] == 0:
                continue
            e = entry[t]
            out[t, c] = abs(x[t] - x[e]) / (path[t] + _EPS)
    return out


def _retrace_series(x2d: np.ndarray, s2d: np.ndarray) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        sign, P, MFE, _MAE, _path, _entry = _episode_map(x2d[:, c], s2d[:, c])
        for t in range(rows):
            if sign[t] == 0:
                continue
            if not (np.isfinite(MFE[t]) and np.isfinite(P[t])):
                continue
            # (MFE-P)/(MFE+eps) is unbounded when the episode's favourable
            # excursion is ~0 while the position is already adverse.  Clamp for
            # numeric hygiene: values > 1 already mean "past best into loss".
            out[t, c] = min((MFE[t] - P[t]) / (MFE[t] + _EPS), 100.0)
    return out


def _balance_series(x2d: np.ndarray, s2d: np.ndarray) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        sign, _P, MFE, MAE, _path, _entry = _episode_map(x2d[:, c], s2d[:, c])
        for t in range(rows):
            if sign[t] == 0:
                continue
            if not (np.isfinite(MFE[t]) and np.isfinite(MAE[t])):
                continue
            out[t, c] = (MFE[t] - MAE[t]) / (MFE[t] + MAE[t] + _EPS)
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="state_episode_mfe",
    category="state_episode_excursion",
    business_category="state_episode_excursion",
    canonical="state_episode_mfe",
    source="state_episode_excursion",
)
class StateEpisodeMfe(SeriesOperator):
    """状态片段内最大有利偏移 MFE, 按片段入场 scale 归一。 >=0。 P1。
    """

    metadata = _metadata(
        "state_episode_mfe",
        "状态片段内最大有利偏移(MFE)按入场 scale 归一。",
        ["x", "state", "scale"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, state: pd.DataFrame, scale: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        return frame_like(
            x,
            _mfe_series(
                x.to_numpy(dtype=float), state.to_numpy(dtype=float), scale.to_numpy(dtype=float)
            ),
        )


@register_operator(
    name="state_episode_mae",
    category="state_episode_excursion",
    business_category="state_episode_excursion",
    canonical="state_episode_mae",
    source="state_episode_excursion",
)
class StateEpisodeMae(SeriesOperator):
    """状态片段内最大不利偏移 MAE, 按入场 scale 归一。 >=0。 P1。
    """

    metadata = _metadata(
        "state_episode_mae",
        "状态片段内最大不利偏移(MAE)按入场 scale 归一。",
        ["x", "state", "scale"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, state: pd.DataFrame, scale: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        return frame_like(
            x,
            _mae_series(
                x.to_numpy(dtype=float), state.to_numpy(dtype=float), scale.to_numpy(dtype=float)
            ),
        )


@register_operator(
    name="state_episode_efficiency",
    category="state_episode_excursion",
    business_category="state_episode_excursion",
    canonical="state_episode_efficiency",
    source="state_episode_excursion",
)
class StateEpisodeEfficiency(SeriesOperator):
    """片段路径效率: 净位移 |x_t-x_e| / 片段内已行走路径和。 [0,1]。 P1。
    """

    metadata = _metadata(
        "state_episode_efficiency",
        "状态片段净位移与已行走路径之和的比值(路径效率)。",
        ["x", "state"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, state: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        return frame_like(
            x,
            _efficiency_series(x.to_numpy(dtype=float), state.to_numpy(dtype=float)),
        )


@register_operator(
    name="state_episode_retrace_ratio",
    category="state_episode_excursion",
    business_category="state_episode_excursion",
    canonical="state_episode_retrace_ratio",
    source="state_episode_excursion",
)
class StateEpisodeRetraceRatio(SeriesOperator):
    """片段回撤比: (MFE-P_t)/MFE。 当前离入场越近/回撤越大 → 越接近 1。 >=0。 P1。
    """

    metadata = _metadata(
        "state_episode_retrace_ratio",
        "状态片段内相对最大有利偏移的回撤比例。",
        ["x", "state"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, state: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        return frame_like(
            x,
            _retrace_series(x.to_numpy(dtype=float), state.to_numpy(dtype=float)),
        )


@register_operator(
    name="state_episode_excursion_balance",
    category="state_episode_excursion",
    business_category="state_episode_excursion",
    canonical="state_episode_excursion_balance",
    source="state_episode_excursion",
)
class StateEpisodeExcursionBalance(SeriesOperator):
    """片段偏移平衡: (MFE-MAE)/(MFE+MAE+eps)。 [-1,1]。 P1。

    ``scale`` was a dead public parameter — nothing used it — so it has been
    removed (P0-14); every declared searchable parameter must change the output.
    """

    metadata = _metadata(
        "state_episode_excursion_balance",
        "状态片段内 MFE 与 MAE 的归一化平衡。",
        ["x", "state"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, state: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        return frame_like(
            x,
            _balance_series(x.to_numpy(dtype=float), state.to_numpy(dtype=float)),
        )


_NEW_CANONICALS = (
    "state_episode_mfe",
    "state_episode_mae",
    "state_episode_efficiency",
    "state_episode_retrace_ratio",
    "state_episode_excursion_balance",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()

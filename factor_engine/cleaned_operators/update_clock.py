# -*- coding: utf-8 -*-
"""Update-clock operators for sparse non-price fields (2026-08 market language, P1).

Report / shareholder / capital / index-weight fields are forward-filled daily,
so a raw ``ts_path_efficiency`` on the filled series sees ``0,0,0,0,update,...``
and measures the fill, not the economics.  These operators run the same
path/anomaly machinery on the **real update nodes only** (``update_event`` marks
true update days) and return a result as-of the current day:

* ``update_path_efficiency``         — net change / total path over the last
  ``n_updates`` real updates (persistent direction vs repeated revision).
* ``update_acceleration``            — last delta vs its own recent spread
  (improvement *speeding up*?).
* ``update_surprise``                — last delta vs the median/MAD of past
  deltas (innovation magnitude of this particular update).
* ``update_direction_persistence``   — magnitude-weighted fraction of same-sign
  deltas.

``update_event`` is the user's own "true update day" boolean (e.g. the raw
field is not a fill).  All kernels are prefix-causal and fail closed to NaN when
the window holds too few real updates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_udf

_EPS = 1e-12
# Real update fields (fundamentals / index weights / shareholding) update at
# quarterly-to-semiannual frequency — a 5-update window is NOT a 10-day window.
# ``2*n_updates`` days was a data-understanding error: on quarterly A-share
# reports (~1.5% of trading days) it could never accumulate n_updates=5, so the
# whole family returned all-NaN on real data.  Scan the trailing 3 trading years
# (~756 days, ~12 quarterly updates) so the last ``n_updates`` *real* update nodes
# are actually reachable; NaN only when even that horizon holds too few updates.
# The kernel is O(rows): per-column update indices are precomputed once and each
# row does a binary search (no per-row window scan).
_UPDATE_LOOKBACK_DAYS = 756


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="update_clock",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "update_clock", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", "domain:fundamental",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _update_kernel(canonical: str, min_updates: int, fn) -> SeriesOperator:
    def _calculate_series(
        self,
        x: pd.DataFrame,
        update_event: pd.DataFrame,
        n_updates: int = 5,
        **_: Any,
    ) -> pd.DataFrame:
        n = int(n_updates)
        if n < min_updates:
            raise ValueError(f"{canonical} requires n_updates >= {min_updates}")

        xv = x.to_numpy(dtype=float)
        ev = update_event.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            # Precompute absolute update-node indices once (O(rows) per column);
            # each row binary-searches for the last ``n`` real updates within the
            # trailing ``_UPDATE_LOOKBACK_DAYS`` horizon (PIT: only updates <= r).
            upd_idx = np.flatnonzero((ev[:, c] >= 0.5) & np.isfinite(ev[:, c]))
            if upd_idx.size < n:
                continue
            for r in range(rows):
                lo = r - _UPDATE_LOOKBACK_DAYS + 1
                j0 = int(np.searchsorted(upd_idx, lo, side="left"))
                j1 = int(np.searchsorted(upd_idx, r + 1, side="left"))
                if j1 - j0 < n:
                    continue
                vals = xv[upd_idx[j1 - n : j1], c]
                if not np.all(np.isfinite(vals)):
                    continue
                out[r, c] = fn(vals)
        return frame_like(x, out)

    return register_operator(
        name=canonical,
        category="update_clock",
        business_category="update_clock",
        canonical=canonical,
        source="update_clock",
    )(
        type(
            canonical.replace("_", " ").title().replace(" ", "") + "Op",
            (SeriesOperator,),
            {"metadata": _metadata(canonical, _DESCRIPTIONS[canonical], ["x", "update_event", "n_updates"], unit=_UNITS[canonical], cost=4), "_calculate_series": _calculate_series, "__module__": __name__},
        )
    )


_DESCRIPTIONS = {
    "update_path_efficiency": "真实 update 节点上的路径效率 |Δ_K|/Σ|Δ_j|（方向一致性）。",
    "update_acceleration": "最近两次 update 差的归一化 (d_K - d_{K-1})/MAD(d)（改善加速）。",
    "update_surprise": "本次 update 相对其自身历史的中位数 z 分（innovation）。",
    "update_direction_persistence": "幅度加权的方向保持率 |Σ w_j·sign(Δ_j)|/Σ w_j。",
}
_UNITS = {
    "update_path_efficiency": "ratio",
    "update_acceleration": "ratio",
    "update_surprise": "zscore",
    "update_direction_persistence": "probability",
}


def _path_eff(vals: np.ndarray) -> float:
    d = np.diff(vals)
    total = float(np.sum(np.abs(d)))
    if total <= _EPS:
        return np.nan
    return float(abs(vals[-1] - vals[0]) / total)


def _acceleration(vals: np.ndarray) -> float:
    d = np.diff(vals)
    if d.size < 3:
        return np.nan
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad <= _EPS:
        return np.nan
    return float((d[-1] - d[-2]) / mad)


def _surprise(vals: np.ndarray) -> float:
    d = np.diff(vals)
    if d.size < 4:
        return np.nan
    past = d[:-1]
    med = float(np.median(past))
    mad = float(np.median(np.abs(past - med)))
    if mad <= _EPS:
        return np.nan
    # MAD-normalised innovation (1.4826 puts MAD on the sigma scale).
    return float((d[-1] - med) / (1.4826 * mad))


def _direction_persist(vals: np.ndarray) -> float:
    d = np.diff(vals)
    if d.size < 2:
        return np.nan
    w = np.abs(d)
    s = np.sign(d)
    denom = float(np.sum(w))
    if denom <= _EPS:
        return np.nan
    return float(abs(np.sum(w * s)) / denom)


TsUpdatePathEfficiency = _update_kernel("update_path_efficiency", 3, _path_eff)
TsUpdateAcceleration = _update_kernel("update_acceleration", 4, _acceleration)
TsUpdateSurprise = _update_kernel("update_surprise", 5, _surprise)
TsUpdateDirectionPersistence = _update_kernel("update_direction_persistence", 3, _direction_persist)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "update_path_efficiency",
            "update_acceleration",
            "update_surprise",
            "update_direction_persistence",
        }
    )
    for _canon in (
        "update_path_efficiency",
        "update_acceleration",
        "update_surprise",
        "update_direction_persistence",
    ):
        register_polars_udf(_canon)


_register_surface()

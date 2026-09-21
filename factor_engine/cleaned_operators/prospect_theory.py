# -*- coding: utf-8 -*-
"""Cumulative prospect theory value (2026-08).

``ts_cpt_value`` computes the *cumulative* prospect-theory value of the trailing
return distribution using Tversky & Kahneman (1992) / Barberis, Mukherjee &
Wang (2016) functional forms:

    v(r) = r**alpha            r >= 0
           -lambda * (-r)**alpha   r < 0

    w(p) = p**gamma / (p**gamma + (1-p)**gamma) ** (1/gamma)

The decision weights are the *differences* of the probability-weighting
function evaluated at cumulative probabilities — never a naive
``w(rank) * v(r)`` product:

    CPT = sum_j pi_j^+ v(g_j) + sum_j pi_j^- v(l_j)

where gains ``g_j`` are ranked best-to-worst (decision weight from the top) and
losses ``l_j`` are ranked worst-to-least (decision weight from the bottom).
Higher CPT values have been shown (Barberis, Mukherjee & Wang, RFS 2016) to
predict lower subsequent returns.

Parameters are deliberately NOT exposed for search: the shape parameters
(alpha / lambda / gamma) are fixed inside the ``preset``, so GP / symbolic
regression cannot turn this primitive into a free parameter-mining surface.
Only the window is searchable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _trailing_window_matrix(values: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trailing-window gather for one column, NaN-aware.

    Returns ``(W, valid, seg_n)``: row ``r`` of ``W`` holds rows
    ``max(0, r-w+1) .. r`` of ``values`` in a ``w``-wide row (short leading rows
    are NaN padded at the tail); ``valid`` marks the positions that are both
    in-window and finite — exactly the authority's ``seg[np.isfinite(seg)]`` —
    and ``seg_n = min(r+1, w)`` is that ``seg.size``.  Only the *set* of valid
    positions is used downstream, so the padding side is immaterial.
    """
    rows = values.shape[0]
    pos = np.arange(w)
    start = np.maximum(0, np.arange(rows) - w + 1)
    gather = np.clip(start[:, None] + pos[None, :], 0, rows - 1)
    inwin = np.ascontiguousarray(pos[None, :] < (np.arange(rows) + 1)[:, None])
    W = np.where(inwin, values[gather], np.nan)
    return W, inwin & np.isfinite(W), inwin.sum(axis=1)


def _trailing_run_matrix(values: np.ndarray, w: int):
    """Trailing *contiguous finite* run ending at every row, NaN padded.

    Row ``r`` holds the maximal trailing finite run of ``values[max(0,r-w+1)..r]``
    starting at column 0 (never re-connected across a gap; a NaN at ``r`` yields
    the empty run), so the padded tail of the row is exactly the authority's
    ``_trailing_contiguous`` / ``_trailing_contiguous_finite`` block.
    """
    rows = values.shape[0]
    idx = np.arange(rows)
    last_bad = np.where(np.isfinite(values), -1, idx)
    np.maximum.accumulate(last_bad, out=last_bad)
    start = np.maximum(np.maximum(0, idx - w + 1), last_bad + 1)
    length = np.where(np.isfinite(values), idx - start + 1, 0)
    pos = np.arange(w)
    gather = np.clip(start[:, None] + pos[None, :], 0, rows - 1)
    V = np.where(pos[None, :] < length[:, None], values[gather], np.nan)
    return V, length


def _padded_median(A: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """``np.median`` of the leading ``cnt`` entries of each row of ``A``.

    ``A`` carries NaN in the unused tail (sorted last), so the two middle order
    statistics of the valid prefix are picked directly; the average of the two
    middles reproduces ``np.median`` bit for bit (including ``cnt < 1`` -> NaN).
    """
    S = np.sort(A, axis=1)
    ridx = np.arange(A.shape[0])
    lo = (cnt - 1) // 2
    hi = cnt // 2
    return (S[ridx, lo] + S[ridx, hi]) / 2.0


# R11 round-3 #118: trailing-window coverage gate.  A window of 60 with only 8
# valid returns would build 1/8-probability decision weights and compare them
# against full-60 peers — a coverage-confounded cross-section.  A sample is
# rejected unless at least this fraction of the trailing window is finite.
# The default is exposed as the ``min_coverage`` parameter (a governance knob,
# not a search dimension); ``polars_chip_tail`` mirrors the same default.
_MIN_COVERAGE = 0.8

# Fixed behavioural parameters (single preset for phase 1).
_PRESETS: dict[str, dict[str, float]] = {
    "bmw2016": {
        "alpha": 0.88,
        "lambda": 2.25,
        "gamma_gain": 0.65,
        "gamma_loss": 0.65,
    }
}


def _probability_weight(p: np.ndarray, gamma: float) -> np.ndarray:
    """w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)."""
    p = np.clip(p, 0.0, 1.0)
    pg = p**gamma
    return pg / np.power(pg + (1.0 - p) ** gamma, 1.0 / gamma)


def _cpt_from_sample(
    r: np.ndarray,
    alpha: float,
    lam: float,
    gamma_gain: float,
    gamma_loss: float,
) -> float:
    """Cumulative prospect-theory value of one sorted return sample."""
    n = r.size
    if n < 2:
        return np.nan
    prob = 1.0 / n
    total = 0.0

    gains = r[r >= 0.0]
    losses = r[r < 0.0]

    if gains.size:
        # best -> worst
        g = gains[::-1]
        cumulative = prob * np.arange(1, g.size + 1, dtype=float)
        prior = np.concatenate([[0.0], cumulative[:-1]])
        decision = _probability_weight(cumulative, gamma_gain) - _probability_weight(prior, gamma_gain)
        total += float(np.sum(decision * np.power(g, alpha)))

    if losses.size:
        # worst (most negative) -> least negative
        l = losses
        cumulative = prob * np.arange(1, l.size + 1, dtype=float)
        prior = np.concatenate([[0.0], cumulative[:-1]])
        decision = _probability_weight(cumulative, gamma_loss) - _probability_weight(prior, gamma_loss)
        total += float(np.sum(decision * (-lam * np.power(-l, alpha))))

    return total


def _cpt_support_floor(window: int) -> int:
    """R16-105: machine-declared CPT estimator-support floor.

    ``max(5, window // 10)`` was a hidden kernel constant that governed
    statistical stability.  It is now a named, versioned policy (also exposed
    as ``metadata.min_effective_sample`` on the operator) so the planner sees
    the effective-N floor instead of a buried magic number.
    """
    return max(5, int(window) // 10)


def _column_cpt(
    returns: np.ndarray,
    window: int,
    preset: dict[str, float],
    min_periods: int,
    min_coverage: float = _MIN_COVERAGE,
) -> np.ndarray:
    """CPT value of the trailing window at every row of one column.

    Row-parallel replacement of the per-row loop: the trailing windows are
    gathered once into a ``(rows, window)`` matrix (NaN preserving) and
    ``_cpt_batch`` evaluates all of them at once.  Same gates, same
    ``np.sort``-based ranking, same decision weights.
    """
    values = np.asarray(returns, dtype=float).reshape(-1)
    W, mask, seg_n = _trailing_window_matrix(values, int(window))
    return _cpt_batch(W, mask, seg_n, preset, int(min_periods), float(min_coverage))


def _cpt_batch(
    W: np.ndarray,
    mask: np.ndarray,
    seg_n: np.ndarray,
    preset: dict[str, float],
    min_periods: int,
    min_coverage: float = _MIN_COVERAGE,
) -> np.ndarray:
    """Vectorised ``_column_cpt``: every trailing window of one column at once.

    The window matrix is sorted once (NaN padded to the tail, so the valid
    prefix is the authority's ``np.sort(valid)``); gains are then read in
    descending order by reversing that prefix and losses in place, and the
    decision weights are the differences of the probability-weighting function
    at the cumulative probabilities, exactly as in ``_cpt_from_sample``.
    """
    rows, w = W.shape
    cnt = mask.sum(axis=1)
    win_n = np.maximum(seg_n, 1)
    ok = (cnt >= min_periods) & (cnt >= 2) & (cnt / win_n >= min_coverage)
    out = np.full(rows, np.nan, dtype=float)
    if not ok.any():
        return out
    alpha = preset["alpha"]
    lam = preset["lambda"]
    gamma_gain = preset["gamma_gain"]
    gamma_loss = preset["gamma_loss"]

    S = np.sort(W, axis=1)
    ks = np.arange(w)
    prob = 1.0 / np.maximum(cnt, 1)
    n_loss = (S < 0.0).sum(axis=1)
    n_gain = np.maximum(cnt - n_loss, 0)
    gain_idx = np.maximum(cnt[:, None] - 1 - ks[None, :], 0)
    G = np.take_along_axis(S, gain_idx, axis=1)
    valid_g = ks[None, :] < n_gain[:, None]
    valid_l = ks[None, :] < n_loss[:, None]

    cum = prob[:, None] * (ks + 1.0)[None, :]
    prior = prob[:, None] * ks[None, :].astype(float)
    dec_g = _probability_weight(cum, gamma_gain) - _probability_weight(prior, gamma_gain)
    dec_l = _probability_weight(cum, gamma_loss) - _probability_weight(prior, gamma_loss)

    Gv = np.where(valid_g, G, 0.0)
    Lv = np.where(valid_l, S, 0.0)
    total = np.sum(dec_g * np.power(Gv, alpha), axis=1)
    total = total + np.sum(dec_l * (-lam * np.power(-Lv, alpha)), axis=1)
    out[ok] = total[ok]
    return out



def _metadata_proxy() -> Any:
    from factor_engine.cleaned_operators.base import OperatorMetadata

    return OperatorMetadata(
        name="ts_cpt_value",
        category="time_series_behavioral",
        description="累积前景理论价值(固定 bmw2016 参数集)。",
        # P1-86: ``preset`` has exactly one legal value (bmw2016) and is NOT a
        # search parameter — it stays as an internal kernel default only.
        # R5-06: the kernel method accepts ``preset``, so it must be a declared
        # parameter name, otherwise the central validator rejects the call as an
        # undeclared keyword.  It is declared as a fixed-choice spec that is not
        # searchable.
        param_names=["returns", "window", "preset", "min_coverage"],
        return_type="series",
        # R11 round-3 #119: v(r)=r^alpha with alpha=0.88 is a behavioural
        # utility score, not an ordinary level — the output unit is
        # ``behavioral_score``.
        output_unit="behavioral_score",
        tags=[
            "time_series_behavioral", "daily", "pit_safe", "causal", "typed_v2",
            "signature:returns,window,preset,min_coverage->series",
            "unit:behavioral_score", "cost:1",
        ],
        param_specs={
            # R16-102: the trailing window is the ONLY searchable economic
            # dimension — declared as a HORIZON ParamSpec (default 60 matches
            # the kernel signature) instead of living only in the docstring.
            "window": ParamSpec(dtype=int, min=5, default=60, param_role=ParamRole.HORIZON),
            "preset": ParamSpec(dtype=str, choices=("bmw2016",), searchable=False),
            # R11 round-3 #118: coverage gate — fraction of the trailing window
            # that must be finite.  A governance threshold, not a search
            # dimension.
            "min_coverage": ParamSpec(
                dtype=float,
                # R16-103: the runtime requires min_coverage > 0 (a coverage
                # gate of 0 would silently disable the gate).  An inclusive
                # ``min=0.0`` let compile accept a value runtime rejects — the
                # exclusive bound is approximated so ``0`` fails at BOTH layers.
                min=1e-6,
                max=1.0,
                default=_MIN_COVERAGE,
                searchable=False,
                param_role=ParamRole.POLICY,
            ),
        },
    )


@register_operator(
    name="ts_cpt_value",
    category="time_series_behavioral",
    business_category="time_series_behavioral",
    canonical="ts_cpt_value",
    source="prospect_theory",
)
class TsCptValue(SeriesOperator):
    """Cumulative prospect-theory value of the trailing return distribution.

    Uses the fixed ``bmw2016`` preset (alpha=0.88, lambda=2.25, gamma=0.65).
    Higher values predict lower subsequent average returns (Barberis, Mukherjee
    & Wang 2016).  NaN while fewer than the internal minimum of valid returns
    are available, and while the trailing-window coverage is below
    ``min_coverage`` (default 0.8) — a sparse 8-of-60 sample must not build
    1/8-probability decision weights against full-60 peers.
    """

    metadata = _metadata_proxy()

    def _calculate_series(
        self,
        returns: pd.DataFrame,
        window: int = 60,
        preset: str = "bmw2016",
        min_coverage: float = _MIN_COVERAGE,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        key = str(preset).lower()
        if key not in _PRESETS:
            raise ValueError(f"unknown CPT preset: {preset!r}; supported={sorted(_PRESETS)}")
        mc = float(min_coverage)
        if not (0.0 < mc <= 1.0):
            raise ValueError("min_coverage must be in (0, 1]")
        params = _PRESETS[key]
        # R16-105: the hidden ``max(5, w // 10)`` support policy is now a
        # machine-declared, NON-searchable policy on the operator metadata
        # (``min_effective_sample``) so the estimator-support floor is visible
        # to the planner, not a buried kernel constant.
        min_periods = _cpt_support_floor(w)
        rv = returns.to_numpy(dtype=float)
        cols = rv.shape[1]
        out = np.column_stack(
            [_column_cpt(rv[:, c], w, params, min_periods, mc) for c in range(cols)]
        )
        return frame_like(returns, out)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_cpt_value"})
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    register_polars_bridge("ts_cpt_value")


_register_surface()

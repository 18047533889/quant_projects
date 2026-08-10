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

from cleaned_operators.base import ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12

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
    rows = returns.shape[0]
    out = np.full(rows, np.nan)
    for t in range(rows):
        lo = max(0, t - window + 1)
        seg = returns[lo : t + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size < min_periods:
            continue
        # R11 round-3 #118: coverage gate.  A window of 60 with only 8 valid
        # returns builds 1/8-probability decision weights and compares them
        # against full-60 peers — a coverage-confounded cross-section.  Require
        # at least ``min_coverage`` (default 0.8) of the trailing window to be
        # finite, else NaN.
        if valid.size / max(1, seg.size) < min_coverage:
            continue
        out[t] = _cpt_from_sample(
            np.sort(valid),
            preset["alpha"],
            preset["lambda"],
            preset["gamma_gain"],
            preset["gamma_loss"],
        )
    return out


def _metadata_proxy() -> Any:
    from cleaned_operators.base import OperatorMetadata

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
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_cpt_value"})
    from cleaned_operators.rolling_pack import register_polars_bridge

    register_polars_bridge("ts_cpt_value")


_register_surface()

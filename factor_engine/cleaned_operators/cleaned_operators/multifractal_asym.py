# -*- coding: utf-8 -*-
"""Multifractal spectrum asymmetry (2026-08-08 Gemini V2 round).

``ts_multifractal_asymmetry`` — the left-right asymmetry of the multifractal
spectrum, computed from the generalised Hurst exponents at matched positive /
negative moments:

    asym = mean_{q in {1,2,4}} ( h(-q) - h(q) )

``h(-q)`` reflects the scaling of *small* fluctuations (low |x| structure) and
``h(+q)`` reflects *large* fluctuations, so:

* ``asym > 0`` — small-fluctuation (negative-moment) scaling dominates;
* ``asym < 0`` — large-fluctuation scaling dominates;
* ``≈ 0``      — symmetric spectrum.

It reuses the sibling module's dyadic-lag generalised-Hurst kernel
(``multifractal._hurst_generalized``) which already enforces the minimum
lag / R² scaling-fit guards, so a window that cannot produce a clean scaling
law fails closed.  P1 / extended.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)
from factor_engine.cleaned_operators.multifractal import _hurst_generalized

_QS = np.array([1.0, 2.0, 4.0])
_EPS = 1e-12


def _spectrum_asymmetry(v: np.ndarray) -> float:
    # P1-34: the negative moments h(-q) are dominated by the SMALLEST increment
    # (``|dx|^q``, q < 0), which is exactly where A-share data is degenerate
    # (0-return / suspended / limit-duplicate increments).  The sibling kernel
    # already fail-closes on these without any EPS added into the moment:
    #   * an exact-zero increment makes ``0^-q = inf`` and the structure
    #     function returns NaN;
    #   * near-zero increments make the negative-moment scaling law noisy, and
    #     the log-log fit's R² >= 0.9 gate rejects the window.
    # So the honest treatment is the kernel's own fail-closed behaviour — no
    # extra ad-hoc floor is imposed (an aggressive min/median guard would also
    # reject healthy series, since the smallest of ~hundreds of increments is
    # almost always << its median).
    hp = np.asarray([_hurst_generalized(v, q_) for q_ in _QS], dtype=float)
    hn = np.asarray([_hurst_generalized(v, -q_) for q_ in _QS], dtype=float)
    if not (np.all(np.isfinite(hp)) and np.all(np.isfinite(hn))):
        return np.nan
    return float(np.mean(hn - hp))


def _ts_multifractal_asymmetry(x: pd.DataFrame, window: int = 120, min_periods: int = 30) -> pd.DataFrame:
    if int(window) < 20:
        raise ValueError("ts_multifractal_asymmetry requires window >= 20")
    rows, cols = x.shape
    w = int(window)
    mp = max(20, int(min_periods))
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            # R4-94: ``window`` is a real horizon — the trailing window must be
            # fully accumulated before an asymmetry estimate is emitted.
            # ``min_periods`` still acts as the finite-contiguous floor for the
            # (possibly gapped) full window.
            if r + 1 < w:
                continue
            lo = r - w + 1
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < mp:
                continue
            val = _spectrum_asymmetry(v)
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


def _register() -> None:
    register_dual(
        "ts_multifractal_asymmetry",
        _ts_multifractal_asymmetry,
        ["x", "window", "min_periods"],
        category="multifractal",
        domain="scaling",
        unit="level",
        cost=6,
        source="multifractal_asym",
        tags_extra=[],
        output_unit="level",
        # R4-95: the trailing ``window`` is a real horizon (R4-94); at most
        # ``window`` rows are consumed, gapped rows reduce the finite
        # contiguous run below the ``min_periods`` floor.
        window_semantics="max_rows",
    )
    # R6-218 (A-share coverage audit): negative q-moments of the generalised
    # Hurst exponent are undefined at any exact-zero increment (A-share 停牌 /
    # 一字板 / 零收益 are common), and the kernel fails closed -> the whole
    # window is NaN.  This operator stays EXTENDED-only with a documented
    # caveat: before promotion to daily, run a real A-share coverage audit; if
    # daily valid coverage < 70-80%, demote to research-only.  Do NOT add an
    # arbitrary EPS floor — that would make the negative-moment estimate
    # entirely determined by the EPS.
    union_extended("ts_multifractal_asymmetry")


_register()

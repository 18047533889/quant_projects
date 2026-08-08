# -*- coding: utf-8 -*-
"""Research-surface transform primitives (2026-08-08 Gemini round).

All three operators live on the Research surface only (never the default
AlphaProbe/AlphaMiner grammar) and are excluded from default production mining
via ``default_search_weight = 0``:

* ``ts_wavelet_lowpass_reconstruct`` — strict trailing causal Haar low-pass
  reconstruction; only the ``level`` coarsest detail levels survive, the rest
  are zeroed before inversion.  Transform primitive.
* ``ts_signature_mahalanobis_anomaly`` — 3-field level-2 truncated path
  signature vector; Mahalanobis anomaly vs. the trailing ``history_window`` of
  signature vectors under shrinkage covariance.  ``depth`` fixed at 2.
* ``ts_persistence_birth_dispersion`` — persistence-diagram birth-time
  dispersion (how strongly structure bifurcates across filtration scales),
  reuse of the certified Rips H1 kernel.  ``cost = 10``.  (Renamed from
  ``ts_betti_crocker_bifurcation_score`` — it is a birth-time dispersion proxy,
  not a true CROCKER; the old name resolves as an alias.)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.ts_model._rolling_core import trailing_contiguous_finite

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="research_transform",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "research", "daily", "pit_safe", "causal", "deterministic",
            f"signature:{','.join(params)}->series", "domain:research_transform",
            "unit:level", "cost:10", "default_search_weight:0",
        ],
    )


# ---------------------------------------------------------------------------
# ts_wavelet_lowpass_reconstruct(x, window, level)
# ---------------------------------------------------------------------------
def _haar_lowpass_current(vals: np.ndarray, window: int, level: int) -> float:
    """Return the trailing-causal Haar low-pass reconstruction at the last row."""
    seg = vals[-int(window):]
    keep = int(level)
    if keep < 0:
        raise ValueError("level must be >= 0")
    # Nearest TRAILING contiguous finite segment (P1-010): the largest power-of-two
    # must come from the most recent unbroken run, not from the oldest ``finite[:m]``
    # prefix.  Compressing out missing rows would shift the wavelet's time origin
    # and mix samples from different calendar times.
    bad = np.flatnonzero(~np.isfinite(seg))
    start = int(bad[-1] + 1) if bad.size else 0
    contig = seg[start:]
    n = contig.size
    if n < 2 ** (keep + 2):
        return np.nan
    m = 2 ** int(np.floor(np.log2(n)))
    # P2-1: take the MOST RECENT power-of-two block of the contiguous run.  When
    # the run length is not a power of two, ``contig[:m]`` would discard the
    # newest observations while claiming to report the *current* low-frequency
    # reconstruction.
    x = contig[-m:].copy()
    details: list[np.ndarray] = []
    approx = x
    while len(approx) >= 2:
        a = (approx[::2] + approx[1::2]) / np.sqrt(2.0)
        d = (approx[::2] - approx[1::2]) / np.sqrt(2.0)
        details.append(d)
        approx = a
    # details[k] gets finer as k grows toward the start; the last `keep` entries
    # (lowest-frequency) are kept, all finer detail levels are zeroed.
    rec = approx
    for k in range(len(details) - 1, -1, -1):
        d = details[k] if (len(details) - k) <= keep else np.zeros_like(details[k])
        inter = np.empty(rec.size * 2, dtype=float)
        inter[0::2] = (rec + d) / np.sqrt(2.0)
        inter[1::2] = (rec - d) / np.sqrt(2.0)
        rec = inter
    return float(rec[-1]) if rec.size else np.nan


@register_operator(
    name="ts_wavelet_lowpass_reconstruct",
    category="research_transform",
    business_category="research_transform",
    canonical="ts_wavelet_lowpass_reconstruct",
    source="research_transform",
    status="experimental",
)
class TsWaveletLowpassReconstruct(SeriesOperator):
    """Haar 低频重建信号（严格 trailing/causal，level 个最粗细节层保留）。"""

    metadata = _metadata(
        "ts_wavelet_lowpass_reconstruct",
        "Haar 小波低频重建（仅保留 level 个最粗细节层后反变换）。",
        ["x", "window", "level"],
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 128, level: int = 2, **_: Any) -> pd.DataFrame:
        w = int(window)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            for r in range(rows):
                out[r, c] = _haar_lowpass_current(xv[: r + 1, c], w, int(level))
        return frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_signature_mahalanobis_anomaly(f1, f2, f3, path_window, history_window, depth)
# ---------------------------------------------------------------------------
def _sig_vector(seg: np.ndarray) -> np.ndarray | None:
    """Level-2 truncated signature of a 3-channel path (3 + 9 = 12 comps).

    P1-010: the path must be FULLY finite — missing rows are never dropped and
    the remaining rows stitched together (that would change the path's time
    origin and sequence of increments).
    """
    n = seg.shape[0]
    if n < 4:
        return None
    if not np.all(np.isfinite(seg)):
        return None
    s = seg
    d = np.diff(s, axis=0)
    # Standard level-2 truncated signature (P2-3): for a piecewise-linear path
    # the level-2 coefficient is the iterated integral
    #   S^(2)_{jk} = sum_i [ run_j(i) * dS_k(i) + 0.5 * dS_j(i) * dS_k(i) ]
    # where ``run`` is the running level-1 value before segment i.  The earlier
    # code dropped the 0.5 * dS_j * dS_k correction term, so it was not the
    # genuine path signature.
    comps: list[float] = [float(d[:, j].sum()) for j in range(3)]
    n2 = np.zeros((3, 3), dtype=float)
    run = np.zeros(3, dtype=float)
    for i in range(d.shape[0]):
        di = d[i]
        for j in range(3):
            for k in range(3):
                n2[j, k] += run[j] * di[k] + 0.5 * di[j] * di[k]
        run += di
    for j in range(3):
        for k in range(3):
            comps.append(float(n2[j, k]))
    return np.asarray(comps, dtype=float)


def _sig_mahalanobis_series(fs: list[np.ndarray], path_window: int, history_window: int) -> np.ndarray:
    rows = fs[0].shape[0]
    out = np.full((rows,), np.nan, dtype=float)
    pw = int(path_window)
    hw = int(history_window)
    for r in range(rows):
        start = max(0, r - pw + 1)
        seg = np.column_stack([f[start : r + 1] for f in fs])
        cur = _sig_vector(seg)
        if cur is None:
            continue
        hist: list[np.ndarray] = []
        for q in range(max(0, r - hw), r):
            qseg = np.column_stack([f[max(0, q - pw + 1) : q + 1] for f in fs])
            sv = _sig_vector(qseg)
            if sv is not None:
                hist.append(sv)
        if len(hist) < 5:
            continue
        H = np.stack(hist, axis=0)
        mu = H.mean(axis=0)
        centered = H - mu
        n, dim = H.shape
        # Covariance needs a robust sample: >= max(20, 2*dim) history vectors
        # (dim=12 -> at least 24) before the 12-dim Mahalanobis distance is stable.
        if n < max(20, 2 * dim):
            continue
        cov = (centered.T @ centered) / (n - 1.0)
        shrink = 0.1 * np.trace(cov) / dim * np.eye(dim) + cov
        try:
            inv = np.linalg.pinv(shrink)
        except np.linalg.LinAlgError:
            continue
        diff = cur - mu
        d = float(np.sqrt(diff @ inv @ diff))
        out[r] = d
    return out


@register_operator(
    name="ts_signature_mahalanobis_anomaly",
    category="research_transform",
    business_category="research_transform",
    canonical="ts_signature_mahalanobis_anomaly",
    source="research_transform",
    status="experimental",
)
class TsSignatureMahalanobisAnomaly(SeriesOperator):
    """三字段 level-2 路径签名向量的 shrinkage-Mahalanobis 异常度。"""

    metadata = _metadata(
        "ts_signature_mahalanobis_anomaly",
        "三字段路径签名 Mahalanobis 异常（depth=2，收缩协方差）。",
        ["f1", "f2", "f3", "path_window", "history_window", "depth"],
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        path_window: int = 20,
        history_window: int = 60,
        depth: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        if int(depth) != 2:
            raise ValueError("ts_signature_mahalanobis_anomaly fixes depth = 2")
        base = f1
        # P1-010: reindex f2/f3 into the base index/columns and REBIND them (the
        # previous loop reindexed into a loop-local variable that was discarded,
        # so f2/f3 stayed misaligned for the column extraction below).
        if not f2.index.equals(base.index) or not f2.columns.equals(base.columns):
            f2 = f2.reindex(index=base.index, columns=base.columns)
        if not f3.index.equals(base.index) or not f3.columns.equals(base.columns):
            f3 = f3.reindex(index=base.index, columns=base.columns)
        cols = base.columns
        out = np.full(base.shape, np.nan, dtype=float)
        for ci, c in enumerate(cols):
            fs = [f[c].to_numpy(dtype=float) for f in (f1, f2, f3)]
            out[:, ci] = _sig_mahalanobis_series(fs, int(path_window), int(history_window))
        return frame_like(base, out)


# ---------------------------------------------------------------------------
# ts_betti_crocker_bifurcation_score(x, window, tau, dim)
# ---------------------------------------------------------------------------
def _bifurcation_score(chunk: np.ndarray, tau: int, dim: int) -> float:
    try:
        from cleaned_operators.topology_ext import _persistence_pairs
    except Exception:  # pragma: no cover - optional topology backend
        return np.nan
    pairs = _persistence_pairs(chunk, int(tau), int(dim))
    if not pairs:
        return np.nan
    births = np.asarray([p[0] for p in pairs], dtype=float)
    deaths = np.asarray([p[1] for p in pairs], dtype=float)
    finite = np.isfinite(births) & np.isfinite(deaths)
    births = births[finite]
    if births.size < 2:
        return np.nan
    span = float(np.max(deaths[finite]) - np.min(births))
    if span <= _EPS:
        return np.nan
    return float(np.std(births) / span)


@register_operator(
    name="ts_persistence_birth_dispersion",
    category="research_transform",
    business_category="research_transform",
    canonical="ts_persistence_birth_dispersion",
    source="research_transform",
    status="experimental",
)
class TsBettiCrockerBifurcationScore(SeriesOperator):
    """持久图特征出生时间的散布度（bifurcation proxy，仅 Research）。

    注意命名（P1-010）：这是持久图 birth-time 散布代理 ``std(births)/span``，
    不是真正的 CROCKER 图（CROCKER 需沿 filtration 参数追踪 Hk 的变化）。
    """

    metadata = _metadata(
        "ts_persistence_birth_dispersion",
        "持久图特征出生时间散布（bifurcation proxy，cost=10，仅 Research）。",
        ["x", "window", "tau", "dim"],
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, tau: int = 5, dim: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            for r in range(rows):
                start = max(0, r - w + 1)
                out[r, c] = _bifurcation_score(xv[start : r + 1, c], int(tau), int(dim))
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {
            "ts_wavelet_lowpass_reconstruct",
            "ts_signature_mahalanobis_anomaly",
            "ts_persistence_birth_dispersion",
        }
    )
    from cleaned_operators.registry import OperatorRegistry

    # P1-010: renamed to the honest _dispersion name; old name stays as an alias.
    OperatorRegistry.register_alias(
        "ts_betti_crocker_bifurcation_score", "ts_persistence_birth_dispersion"
    )
    for _canon in (
        "ts_wavelet_lowpass_reconstruct",
        "ts_signature_mahalanobis_anomaly",
        "ts_persistence_birth_dispersion",
    ):
        from cleaned_operators.rolling_pack import register_polars_udf

        register_polars_udf(_canon)


_register_surface()

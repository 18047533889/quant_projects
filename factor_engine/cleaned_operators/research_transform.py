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

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
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
# R6-101: the Haar cascade only ever uses the largest power-of-two ``m`` that
# fits inside the trailing contiguous run, so ``window=65/80/100/127`` all
# reduce to the same 64-point transform — a large block of equivalent search
# parameters (different AST, identical output).  Fix (review option A): the
# window itself is restricted to powers of two (32/64/128/256), so every
# declared parameter value maps to a distinct cascade depth.
_WAVELET_WINDOW_CHOICES = (32, 64, 128, 256)


def _haar_lowpass_current(vals: np.ndarray, window: int, level: int) -> float:
    """Return the trailing-causal Haar low-pass reconstruction at the last row."""
    seg = vals[-int(window):]
    keep = int(level)
    if keep < 0:
        raise ValueError("level must be >= 0")
    # R6-101: reject a window that is not a power of two — the kernel must never
    # silently coerce 65/80/100/127 to the same 64-point transform.
    w = int(window)
    if w <= 0 or (w & (w - 1)) != 0:
        raise ValueError(
            "ts_wavelet_lowpass_reconstruct requires window to be a power of two "
            f"(32/64/128/256), got {w}"
        )
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
    research_only=True,
)
class TsWaveletLowpassReconstruct(SeriesOperator):
    """Haar 低频重建信号（严格 trailing/causal，level 个最粗细节层保留）。"""

    metadata = _metadata(
        "ts_wavelet_lowpass_reconstruct",
        "Haar 小波低频重建（仅保留 level 个最粗细节层后反变换）。",
        ["x", "window", "level"],
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, choices=_WAVELET_WINDOW_CHOICES),
        # R6-102: level is bounded by the cascade — a contiguous run must have
        # n >= 2^(level+2) for `level` detail layers + the final approximation.
        # Declared as a RelationalParamSpec so search never emits a
        # guaranteed-NaN (window=32, level=5) combination.
        "level": ParamSpec(dtype=int, min=0, max=6),
    }
    metadata.relational_specs = [
        RelationalParamSpec(
            "2 ** (level + 2) <= window",
            "level and window are infeasible: need 2^(level+2) <= window "
            "(level={level}, window={window})",
        )
    ]

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
    # R6-104: overlapping path signatures share pw-1 samples, so ``len(hist)``
    # massively overstates the effective sample size (24 vectors of a 12-dim
    # signature are far from 24 independent observations).  Sample the history
    # with a stride proportional to the path length and require the effective
    # count to reach 2*dim — the covariance is only then stable.
    stride = max(1, pw // 4)
    for r in range(rows):
        start = max(0, r - pw + 1)
        if r - start + 1 < pw:
            # R6-103: no signature at all before the first COMPLETE path_window
            # — a 4-bar signature in the history is not the same random variable
            # as a 20-bar one, so it must not pollute the covariance sample.
            continue
        seg = np.column_stack([f[start : r + 1] for f in fs])
        cur = _sig_vector(seg)
        if cur is None:
            continue
        hist: list[np.ndarray] = []
        # R6-103: a history signature is only admitted if its own path window is
        # complete (q - pw + 1 >= 0) — same random variable as ``cur``.
        hist_first = max(pw - 1, r - hw)
        for q in range(hist_first, r, stride):
            qseg = np.column_stack([f[q - pw + 1 : q + 1] for f in fs])
            sv = _sig_vector(qseg)
            if sv is not None:
                hist.append(sv)
        if len(hist) < 5:
            continue
        H = np.stack(hist, axis=0)
        # P0-26: the three path channels carry different units (a CNY price vs a
        # return vs a ratio), so the 12 signature components sit on wildly
        # different scales.  Robust-standardise each component on the HISTORY
        # sample first (median/MAD -> std fallback -> 1.0), then Mahalanobis on
        # the standardised space; ``cur`` is standardised with the same history
        # statistics (strict-PIT, never the current row's own scale).
        med = np.median(H, axis=0)
        mad = 1.4826 * np.median(np.abs(H - med), axis=0)
        scale = np.where(mad > _EPS, mad, np.std(H, axis=0))
        scale = np.where(np.isfinite(scale) & (scale > _EPS), scale, 1.0)
        Hs = (H - med) / scale
        cur_s = (cur - med) / scale
        mu = Hs.mean(axis=0)
        centered = Hs - mu
        n, dim = Hs.shape
        # Covariance needs a robust sample: >= max(20, 2*dim) history vectors
        # (dim=12 -> at least 24) before the 12-dim Mahalanobis distance is stable.
        # R6-104: ``n`` is now the STRIDED (effective) history count, so this is
        # an effective-sample-size gate rather than a nominal-count gate.
        if n < max(20, 2 * dim):
            continue
        cov = (centered.T @ centered) / (n - 1.0)
        # R6-105: this is ridge (diagonal) loading, NOT convex shrinkage — the
        # Ledoit-Wolf / oracle form is (1-lambda)Sigma + lambda*mu*I with a
        # convex weight.  Renamed to match what the math actually does: a fixed
        # 10% ridge on the trace-normalised diagonal.
        ridge = 0.1 * np.trace(cov) / dim * np.eye(dim) + cov
        try:
            inv = np.linalg.pinv(ridge)
        except np.linalg.LinAlgError:
            continue
        diff = cur_s - mu
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
    research_only=True,
)
class TsSignatureMahalanobisAnomaly(SeriesOperator):
    """三字段 level-2 路径签名向量的 shrinkage-Mahalanobis 异常度。"""

    metadata = _metadata(
        "ts_signature_mahalanobis_anomaly",
        "三字段路径签名 Mahalanobis 异常（depth=2，收缩协方差，通道先稳健标准化）。",
        ["f1", "f2", "f3", "path_window", "history_window"],
    )
    # R9-OP-010 (default infeasibility): history is sampled with stride
    # ``path_window//4``, so the effective history count is ~
    # ``history_window / (path_window//4)``; the covariance gate needs >= 24
    # effective vectors.  The old default ``history_window=60`` with
    # ``path_window=20`` gave only ~12 effective vectors — the canonical was
    # guaranteed NaN for every row at its own default.  The default is now
    # feasible AND the declared contract prunes the guaranteed-NaN region
    # (path_window >= 4 so ``path_window//4 >= 1`` and the relation is sound).
    metadata.param_specs = {
        "path_window": ParamSpec(dtype=int, min=4),
        "history_window": ParamSpec(dtype=int, min=24),
    }
    metadata.relational_specs = [
        RelationalParamSpec(
            "history_window >= 24 * (path_window // 4)",
            "ts_signature_mahalanobis_anomaly needs >= 24 strided history "
            "signatures: history_window >= 24*(path_window//4) "
            "(path_window={path_window}, history_window={history_window})",
        )
    ]

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        path_window: int = 20,
        history_window: int = 120,
        depth: int = 2,  # P0-26: fixed at 2 — not a search parameter (removed from the signature)
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
    # P0-05 review: ``_persistence_pairs`` requires the explicit ``h0`` flag
    # (H0 loops / H1 filtration are computed differently); the old call omitted
    # it and would TypeError on every evaluation.  H1-only for the bifurcation
    # proxy (consistent with the certified Rips H1 kernel this operator reuses).
    pairs = _persistence_pairs(chunk, int(tau), int(dim), h0=False)
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
    research_only=True,
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
    # R6-106: the operator computes Rips H1, which requires a >= 2-dimensional
    # Takens embedding — dim=1 produces a 1D point cloud with no loops, so the
    # default must be dim>=2 and the search space must not offer dim=1.
    # R6-107: with dim=1 the delay-embedding lag tau*(dim-1) is always zero
    # (tau is a dead parameter); restricting dim>=2 keeps tau meaningful.
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=20),
        "tau": ParamSpec(dtype=int, min=1),
        "dim": ParamSpec(dtype=int, min=2, default=2),
    }

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, tau: int = 5, dim: int = 2, **_: Any) -> pd.DataFrame:
        if int(dim) < 2:
            raise ValueError(
                "ts_persistence_birth_dispersion requires dim >= 2: Rips H1 needs "
                "a >= 2-dimensional Takens embedding (dim=1 has no loop structure)"
            )
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

    # R5-50: live extend mutator, never a frozenset reassignment.
    _surface.extend_research_only({
        "ts_wavelet_lowpass_reconstruct",
        "ts_signature_mahalanobis_anomaly",
        "ts_persistence_birth_dispersion",
    })
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

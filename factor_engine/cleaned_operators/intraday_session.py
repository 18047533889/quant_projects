# -*- coding: utf-8 -*-
"""Intraday session-shape operators (2026-08 geometry/math expansion).

Minute panels (rows = minutes, columns = symbols) are segmented per symbol by an
integer ``session_id`` panel that tags every row with its trading day.  Each
consecutive run of equal ``session_id`` is one *session*.  This module compares
the shape of the current session against the recent history of completed
sessions:

* ``intraday_session_shape_novelty`` — Euclidean distance of the current
  session's normalised price shape to the nearest historical session's shape
  (shapes are standardised to zero mean / unit std and resampled to the current
  session's length on a common grid by linear interpolation).
* ``intraday_profile_pca_residual`` — residual of the current session's shape
  after projecting it on the principal components fitted on past sessions only.

Both operators are per-symbol and strictly prefix-causal / PIT-safe: a session's
shape is only known once the session has completed, so the value is emitted at
the final minute of the session (earlier minutes of that session are NaN).  Only
sessions that ended strictly before the current one are used as history, and
``history_days`` bounds how many of them are consulted.  Degenerate inputs
(fewer than two finite points, zero session std) emit NaN.
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
        category="intraday_session",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday_session", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:intraday_session",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared session kernels (per column; 2D panel = Minute x Symbol)
# ---------------------------------------------------------------------------
def _session_runs(x_col: np.ndarray, sid_col: np.ndarray) -> list[dict]:
    """Split one symbol's minute column into sessions of equal consecutive id.

    Returns a list of ``{"sid", "vals", "start", "end"}`` where ``vals`` is the
    finite-validated raw column slice and ``start``/``end`` are inclusive row
    indices of the session.  A non-finite ``session_id`` breaks the run.
    """
    runs: list[dict] = []
    cur_sid: int | None = None
    cur_vals: list[float] = []
    cur_start: int | None = None
    n = x_col.shape[0]
    for i in range(n):
        s = sid_col[i]
        if not np.isfinite(s):
            if cur_sid is not None:
                runs.append({"sid": cur_sid, "vals": np.asarray(cur_vals, dtype=float),
                             "start": cur_start, "end": i - 1})
            cur_sid, cur_vals, cur_start = None, [], None
            continue
        sid = int(s)
        if cur_sid is None:
            cur_sid, cur_start = sid, i
        elif sid != cur_sid:
            runs.append({"sid": cur_sid, "vals": np.asarray(cur_vals, dtype=float),
                         "start": cur_start, "end": i - 1})
            cur_sid, cur_start, cur_vals = sid, i, []
        cur_vals.append(x_col[i])
    if cur_sid is not None:
        runs.append({"sid": cur_sid, "vals": np.asarray(cur_vals, dtype=float),
                     "start": cur_start, "end": n - 1})
    return runs


def _normalized_shape(vals: np.ndarray) -> np.ndarray | None:
    """Standardised shape vector (x - mean)/std; None when degenerate."""
    v = vals.astype(float)
    if v.size < 2 or not np.all(np.isfinite(v)):
        return None
    sd = v.std()
    if sd <= _EPS:
        return None
    return (v - v.mean()) / sd


def _resample_shape(shape: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interpolate a shape onto the common grid of length ``target_len``."""
    n = shape.size
    if target_len == n:
        return shape.astype(float)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, shape)


def _shape_novelty_series(x2d: np.ndarray, sid2d: np.ndarray, history_days: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    hd = int(history_days)
    if hd < 1:
        raise ValueError("history_days must be >= 1")
    for c in range(cols):
        runs = _session_runs(x2d[:, c], sid2d[:, c])
        for i, run in enumerate(runs):
            shape = _normalized_shape(run["vals"])
            if shape is None:
                continue
            m = shape.size
            best = np.inf
            seen = 0
            for j in range(max(0, i - hd), i):  # only sessions ending before current
                hshape = _normalized_shape(runs[j]["vals"])
                if hshape is None:
                    continue
                hr = _resample_shape(hshape, m)
                d = float(np.linalg.norm(hr - shape))
                seen += 1
                if d < best:
                    best = d
            if seen < 1:
                continue  # NaN if < 1 usable historical session
            out[run["end"], c] = float(best)
    return out


def _pca_residual_series(
    x2d: np.ndarray, sid2d: np.ndarray, history_days: int, n_components: int,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    hd = int(history_days)
    nc = int(n_components)
    if hd < 1:
        raise ValueError("history_days must be >= 1")
    if nc < 1:
        raise ValueError("n_components must be >= 1")
    for c in range(cols):
        runs = _session_runs(x2d[:, c], sid2d[:, c])
        for i, run in enumerate(runs):
            shape = _normalized_shape(run["vals"])
            if shape is None:
                continue
            m = shape.size
            hist: list[np.ndarray] = []
            for j in range(max(0, i - hd), i):  # past sessions only
                hshape = _normalized_shape(runs[j]["vals"])
                if hshape is None:
                    continue
                hist.append(_resample_shape(hshape, m))
            if len(hist) < nc + 1:
                continue  # too few past sessions for an nc-component fit
            M = np.stack(hist, axis=0)  # (n_past, m)
            mean_m = M.mean(axis=0)
            Mc = M - mean_m
            _U, _S, Vt = np.linalg.svd(Mc, full_matrices=False)
            n_used = min(nc, Vt.shape[0])
            comps = Vt[:n_used]  # (n_used, m)
            coeff = (shape - mean_m) @ comps.T  # (n_used,)
            z_hat = mean_m + coeff @ comps
            num = float(np.linalg.norm(shape - z_hat))
            den = float(np.linalg.norm(shape)) + _EPS
            out[run["end"], c] = num / den
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="intraday_session_shape_novelty",
    category="intraday_session",
    business_category="intraday_session",
    canonical="intraday_session_shape_novelty",
    source="intraday_session",
)
class IntradaySessionShapeNovelty(SeriesOperator):
    """日内分时形态新颖度：当前交易日分时形态相对最近历史交易日的最近距离。

    每个 session 的形态先标准化 ``(x-mean)/std``，再把历史 session 形态线性插值
    重采样到当前 session 长度后求欧氏距离，取最小值。少于 1 个历史 session → NaN。
    在 session 最后一分钟输出（完成后才知道完整形态，保持 PIT 安全）。P1。
    """

    metadata = _metadata(
        "intraday_session_shape_novelty",
        "当前 session 标准化分时形态到最近历史 session 形态的欧氏距离。",
        ["x", "session_id", "history_days"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self, x: pd.DataFrame, session_id: pd.DataFrame, history_days: int = 20, **_: Any,
    ) -> pd.DataFrame:
        arr = _shape_novelty_series(
            x.to_numpy(dtype=float), session_id.to_numpy(dtype=float), history_days,
        )
        return frame_like(x, arr)


@register_operator(
    name="intraday_profile_pca_residual",
    category="intraday_session",
    business_category="intraday_session",
    canonical="intraday_profile_pca_residual",
    source="intraday_session",
)
class IntradayProfilePcaResidual(SeriesOperator):
    """日内分时 PCA 残差：当前 session 形态相对仅用历史 session 拟合的主成分重建的残差。

    把最近 history_days 个过去 session 的标准化形态重采样到当前长度组成矩阵，
    只在这些过去 session 上拟合 PCA；残差 =
    ``|Z - Ẑ| / (|Z|+eps)``，Ẑ 为 n_components 主成分重建。过去 session 少于
    ``n_components+1`` 个 → NaN。在 session 最后一分钟输出（PIT 安全）。P1。
    """

    metadata = _metadata(
        "intraday_profile_pca_residual",
        "当前 session 形态相对历史 PCA 主成分重建的归一化残差。",
        ["x", "session_id", "history_days", "n_components"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(
        self, x: pd.DataFrame, session_id: pd.DataFrame, history_days: int = 20,
        n_components: int = 3, **_: Any,
    ) -> pd.DataFrame:
        arr = _pca_residual_series(
            x.to_numpy(dtype=float), session_id.to_numpy(dtype=float),
            history_days, n_components,
        )
        return frame_like(x, arr)


_NEW_CANONICALS = (
    "intraday_session_shape_novelty",
    "intraday_profile_pca_residual",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()

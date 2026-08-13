# -*- coding: utf-8 -*-
"""Rolling panel model operators (P2, experimental).

Rolling PCA residuals / loadings, PCR / PLS / ElasticNet forecasts, regime-
conditioned and mixture-of-experts forecasts, and a linear (PCA) autoencoder
reconstruction-error baseline.

M-030 (per-symbol, not pooled): despite the ``panel`` prefix, every operator in
this module is a PER-SYMBOL rolling time-series model.  The cross-section is
used only to share a PCA fit across stocks on the same date (the same date's
cross-section is fit once and broadcast to member stocks); it is never pooled
into a single stock×date regression.  A genuinely pooled panel model would be a
separate canonical.

R35-P0-M10 (doc drift fix): the caller does NOT need to pre-lag / pre-embargo a
label panel.  ``_forecast_loop`` / ``_regime_forecast`` / ``_moe_forecast``
enforce ``fit_lag>=1`` AND ``label_horizon`` maturity internally: a forward-H
label anchored at row ``s`` is only usable at ``s + label_horizon``, so the
last ``label_horizon`` training rows are always excluded.  The operator is the
single authority for its own PIT boundary — re-lagging the label in the caller
would double-shift it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.ts_model._rolling_core import fit_linear_model_checked

_EPS = 1e-12
# R35-P0-M07: PCA min-history is an EXPLICIT public contract, not a hidden
# hardcode.  A stock is ACTIVE for the PCA fit only when it has at least
# ``min_obs`` finite observations inside the training window, where
# ``min_obs = max(PCA_MIN_HISTORY, ceil(window * PCA_MIN_COVERAGE))``.
# A stock with 2 days of data in a 120-day window used to enter the fit with
# 118 mean-imputed days and distort the covariance — far too loose.  The values
# are module-public so the R35 gate can assert them and the parameter-domain
# matrix can explore around them (see ``tests/operators/r35/``).
# Note: this is a per-stock coverage gate inside the fit window.  A separate
# "no output before the window has enough rows" floor is enforced by
# ``_rolling_pca`` / ``_industry_pca_loading`` in full-history mode
# (``warmup_policy="full"``, the default): the first scored row is the one
# whose training window holds a FULL ``window`` rows (``fit_end >= window - 1``).
# Expanding warmup (fitting on fewer than ``window`` rows) is only available to
# a caller that explicitly opts in via ``warmup_policy="expanding"`` (M-020).
PCA_MIN_HISTORY = 2
PCA_MIN_COVERAGE = 0.7
_ABSOLUTE_MIN_OBS = PCA_MIN_HISTORY
_MIN_COVERAGE_RATIO = PCA_MIN_COVERAGE
_CANONICALS: list[str] = []

#: M-036: internal fit-quality telemetry for the supervised regime / MoE
#: forecasts.  Populated on every (col, row) fit that reaches the design stage;
#: the module-level ``last_fit_telemetry()`` accessor exposes the most recent
#: snapshot to audit probes.  This is a DIAGNOSTIC accessor (mirroring
#: ``ts_model.state_space.numba_dispatch_stats``), NOT a public operator
#: canonical — no new operator surface is registered from it.
_LAST_FIT_TELEMETRY: dict[str, Any] = {}


def last_fit_telemetry() -> dict[str, Any]:
    """Fit-quality telemetry from the most recent regime / MoE supervised fit.

    Keys captured (M-036): ``model``, ``effective_train_obs``,
    ``effective_regime_obs`` (regime) / ``active_expert_count`` (MoE),
    ``condition_number``, ``convergence``, ``gate_entropy`` and (MoE)
    ``expert_weight_max``.  Returns a defensive copy.
    """
    return dict(_LAST_FIT_TELEMETRY)


def _design_cond(design: np.ndarray) -> float:
    """Condition number of a design matrix (``inf`` on a degenerate design)."""
    try:
        return float(np.linalg.cond(design))
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")


# R10-P0-009: panel-model integer knobs must declare ``ParamSpec(dtype=int)``.
# ``component`` / ``label_horizon`` / ``n_regimes`` / ``n_experts`` are NOT in
# the central legacy integer whitelist (``_INTEGER_PARAM_NAMES``), so without
# a spec a fractional search candidate (``n_experts=3.9``) silently truncated
# to ``int(3.9) == 3`` inside the kernel.  A declared spec makes the strict
# call gate reject the fractional value at the boundary instead.
_INT_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2),
    "n_components": ParamSpec(dtype=int, min=1),
    "component": ParamSpec(dtype=int, min=0),
    "label_horizon": ParamSpec(dtype=int, min=1),
    "n_regimes": ParamSpec(dtype=int, min=2),
    "n_experts": ParamSpec(dtype=int, min=2),
}
_FLOAT_PARAM_SPECS: dict[str, ParamSpec] = {
    "alpha": ParamSpec(dtype=float, min=1e-8),
    "l1_ratio": ParamSpec(dtype=float, min=0.0, max=1.0),
}


def _meta(name: str, description: str, params: list[str], *, unit: str = "level", pit_safe: bool = True) -> OperatorMetadata:
    tags = [
        "panel_model", "daily", "causal", "typed_v2",
        f"signature:{','.join(params)}->series", "domain:panel_model",
        f"unit:{unit}", "cost:10",
    ]
    if pit_safe:
        tags.append("pit_safe")
    else:
        # Supervised panel forecasts carry a label panel whose training window
        # is *not* re-lagged by the operator; they must not be advertised as
        # point-in-time safe for default mining.
        tags.append("supervised_model")
        tags.append("not_pit_certified")
    param_specs = {
        name: _INT_PARAM_SPECS[name]
        for name in params
        if name in _INT_PARAM_SPECS
    }
    param_specs.update(
        {
            name: _FLOAT_PARAM_SPECS[name]
            for name in params
            if name in _FLOAT_PARAM_SPECS
        }
    )
    return OperatorMetadata(
        name=name,
        category="panel_model",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
        param_specs=param_specs,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "level", pit_safe: bool = True):
    metadata = _meta(name, description, params, unit=unit, pit_safe=pit_safe)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"PanelModel_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="panel_model",
        business_category="panel_model",
        canonical=name,
        source="cross_section.panel_model",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({name})
    return cls


def _pca_svd(X: np.ndarray, n_components: int, *, rank_policy: str = "reconstruction"):
    """Standardised SVD PCA; returns the model in the *active sub-space* only.

    R38 P0-060（§23）：唯一 authoritative 数学在 ``pca_state`` 模块——canonical
    与 shared ``PCABlock`` 共用同一 fit（coverage-gated active / 标准化 SVD /
    rank cap / missing imputation），不再在 backend 复制第二套。

    ``rank_policy`` (M-021): ``"reconstruction"`` (default) caps ``k <= p-1``
    so the reconstruction residual is never trivially zero; ``"regression"``
    allows ``k <= p`` so PCR can use every component.  The PCR forecast path
    (``_model_predict``) calls this with ``rank_policy="regression"``.
    """
    from cleaned_operators.cross_section.pca_state import PCAState

    state = PCAState.from_window(
        X, n_components,
        min_coverage_ratio=_MIN_COVERAGE_RATIO,
        absolute_min_obs=_ABSOLUTE_MIN_OBS,
        rank_policy=rank_policy,
    )
    return state.to_dict() if state is not None else None


def _pca_transform(pca, row: np.ndarray) -> np.ndarray:
    """Project ``row`` (full universe) onto the active sub-space's components.

    R10-P0-006: a stock that is missing TODAY must not poison the common
    component scores of every other stock.  Its standardized value is pinned
    to the training mean (``0.0``) so the shared ``loadings @ z`` stays finite;
    the missing stock's OWN output cell is written as NaN by the caller.
    """
    row_active = row[pca["active"]]
    current_valid = np.isfinite(row_active)
    z = np.where(current_valid, (row_active - pca["mu"]) / pca["sd"], 0.0)
    return pca["loadings"] @ z


def _rolling_pca(ret: pd.DataFrame, window: int, fn, *, fit_lag: int = 1, warmup_policy: str = "full") -> pd.DataFrame:
    """Rolling PCA evaluated row by row.

    ``fit_lag>=1`` (default) trains the PCA on rows ``[start, row-1]`` and
    evaluates the *current* row against that historical model, so the current
    observation never enters its own training set (no in-sample projection).
    ``fit_lag=0`` reproduces the legacy in-sample behaviour.

    ``warmup_policy`` (M-020): ``"full"`` (default) enforces a full-history
    floor — the output stays NaN until ``fit_end >= window - 1``, i.e. the
    training window is a complete ``window`` rows.  ``"expanding"`` opts in to
    the legacy expanding warmup that fits on partial windows.
    """
    rv = ret.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    lag = max(0, int(fit_lag))
    w = int(window)
    for row in range(rows):
        fit_end = row - lag
        if fit_end < 0:
            continue
        if warmup_policy == "full" and fit_end < w - 1:
            continue
        start = max(0, fit_end - w + 1)
        X = rv[start : fit_end + 1]
        out[row] = fn(X, rv[row])
    return _frame_like(ret, out)


def _pca_loading(X: np.ndarray, cur: np.ndarray, component: int, column_ids: tuple[Any, ...] | None = None) -> np.ndarray:
    pca = _pca_svd(X, component + 1)
    if pca is None or component >= pca["k"]:
        return np.full(len(cur), np.nan)
    # P0-14: loadings are active-space (k, n_active); map back to the full
    # universe, leaving inactive columns NaN (fail-closed).
    loading = np.full(len(cur), np.nan)
    loading[pca["active"]] = pca["loadings"][component]
    # R10-P0-005: STATELESS sign orientation.  The SVD eigenvector sign is
    # arbitrary per window; aligning to the PREVIOUS window's loading made this
    # a hidden stateful operator — full-run, chunked-run and mid-series starts
    # could yield opposite signs for the same window.  Instead each window is
    # normalised deterministically on its own: the largest |loading| position
    # is forced positive.  R35-P0-M08: when two active loadings tie in |value|,
    # the winner is decided by STABLE INSTRUMENT IDENTITY (the panel's column
    # label, lexicographically) instead of column position — a universe reorder
    # must not flip the sign of the whole eigenvector.
    fin = np.flatnonzero(np.isfinite(loading))
    if fin.size == 0:
        return loading
    abs_vals = np.abs(loading[fin])
    best = int(np.argmax(abs_vals))
    max_abs = float(abs_vals[best])
    tied = fin[abs_vals >= max_abs - 1e-12]
    if tied.size > 1 and column_ids is not None:
        ids = [column_ids[int(i)] for i in tied]
        k = tied[int(min(range(len(ids)), key=lambda i: str(ids[i])))]
    else:
        k = fin[best]
    if loading[k] < 0:
        loading = -loading
    return loading


def _pca_loading_series(ret: pd.DataFrame, window: int, component: int) -> pd.DataFrame:
    ids = tuple(ret.columns)
    return _rolling_pca(
        ret, int(window), lambda X, cur: _pca_loading(X, cur, int(component), ids)
    )


_mk(
    "panel_rolling_pca_loading",
    "过去窗口收益矩阵 PCA 的指定主成分载荷（逐窗确定性 sign：最大 |载荷| 位置强制为正，无跨窗记忆）。",
    ["ret", "window", "component"],
    lambda ret, window=120, component=0: _pca_loading_series(ret, int(window), int(component)),
    unit="loading",
)


def _pca_resid(X: np.ndarray, cur: np.ndarray, n_components: int) -> np.ndarray:
    pca = _pca_svd(X, min(int(n_components), X.shape[1]))
    if pca is None:
        return np.full(len(cur), np.nan)
    # R10-P0-006: ``score`` is computed with the current row's missing stocks
    # pinned to the training mean (0 standardized), so the shared components and
    # the reconstructed values stay FINITE for every active stock.  A stock that
    # is missing TODAY gets its own residual NaN (``cur_active`` is NaN there),
    # while its peers keep a valid residual — one halted stock no longer blanks
    # the whole cross-section.
    score = _pca_transform(pca, cur)
    cur_active = cur[pca["active"]]
    recon = pca["mu"] + pca["sd"] * (pca["loadings"].T @ score)
    out = np.full(len(cur), np.nan)
    # P0-14: write residuals only for the active sub-space; inactive stocks stay
    # NaN (fail-closed) and can no longer poison their peers.
    out[pca["active"]] = cur_active - recon
    return out


_mk(
    "panel_rolling_pca_resid",
    "当前收益相对前 K 主成分模型的残差。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _rolling_pca(ret, int(window), lambda X, c: _pca_resid(X, c, int(n_components))),
)


def _pca_commonality(X: np.ndarray, cur: np.ndarray, n_components: int) -> np.ndarray:
    """Per-stock commonality ``1 - Var(resid_i)/Var(ret_i)`` over the training window.

    R38 P0-060（§23）：唯一公式在 ``pca_state.pca_commonality``——canonical 与
    shared ``PCABlock.commonality`` 共用，不再两套数学漂移。
    """
    from cleaned_operators.cross_section.pca_state import PCAState, pca_commonality

    state = PCAState.from_window(
        X, min(int(n_components), X.shape[1]),
        min_coverage_ratio=_MIN_COVERAGE_RATIO,
        absolute_min_obs=_ABSOLUTE_MIN_OBS,
    )
    if state is None:
        return np.full(len(cur), np.nan)
    return pca_commonality(X, state.to_dict())


_mk(
    "panel_rolling_pca_explained_ratio",
    "每股被共同因子解释的比例（1 - 残差方差/总方差，每股独立）。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _rolling_pca(ret, int(window), lambda X, c: _pca_commonality(X, c, int(n_components))),
)


def _industry_pca_loading(ret, group, window, component):
    rv = ret.to_numpy(dtype=float)
    gv = group.to_numpy()
    col_ids = tuple(ret.columns)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for row in range(rows):
        fit_end = row - 1
        if fit_end < 0:
            continue
        # M-020: full-history floor — no expanding warmup for industry PCA.
        if fit_end < w - 1:
            continue
        start = max(0, fit_end - w + 1)
        # M-026: fit ONCE per (date, unique industry) and broadcast the loading
        # to every member stock — every member shares the same fit, so the
        # per-(row, col) SVD refit was wasted work.
        for g in np.unique(gv[row]):
            mask = gv[row] == g
            members = np.flatnonzero(mask)
            if len(members) < 4:
                continue
            X = rv[start : fit_end + 1][:, members]
            # R10-P0-004: the loading kernel is STATELESS (3 args) — the old
            # call omitted the fourth ``prev`` argument and crashed with a
            # TypeError as soon as any industry had >= 4 members.
            member_ids = tuple(col_ids[int(i)] for i in members)
            loading = _pca_loading(X, rv[row][members], int(component), member_ids)
            # ``local`` = position of each member inside the industry members;
            # correct even when the industry's columns are not contiguous.
            for local, col in enumerate(members):
                out[row, col] = loading[local] if np.isfinite(loading).any() else np.nan
    return _frame_like(ret, out)


def _pca_resid_vol(ret: pd.DataFrame, window: int, n_components: int) -> pd.DataFrame:
    resid = _rolling_pca(ret, int(window), lambda X, c: _pca_resid(X, c, int(n_components)))
    return resid.rolling(int(window), min_periods=10).std()


_mk(
    "panel_rolling_pca_resid_vol",
    "PCA 残差波动率（过去窗口）。两级成熟（M-027）：先需完整 window 行 PCA 训练窗（full-history floor，输出在第 window 行才开始），再需 ≥10 行残差滚动统计 —— 首次输出在第 (window-1)+10 行，不是短窗口统计。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _pca_resid_vol(ret, int(window), int(n_components)),
    unit="volatility",
)


def _pca_resid_momentum(ret: pd.DataFrame, window: int, n_components: int) -> pd.DataFrame:
    resid = _rolling_pca(ret, int(window), lambda X, c: _pca_resid(X, c, int(n_components)))
    return resid.rolling(int(window), min_periods=10).sum()


_mk(
    "panel_rolling_pca_resid_momentum",
    "PCA 残差累计收益。两级成熟（M-027）：先需完整 window 行 PCA 训练窗（full-history floor），再需 ≥10 行残差滚动累计 —— 首次输出在第 (window-1)+10 行，不是短窗口统计。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _pca_resid_momentum(ret, int(window), int(n_components)),
)


_mk(
    "industry_rolling_pca_loading",
    "行业内部滚动 PCA 载荷（训练窗口截至前一日，局部索引按成员定位）。",
    ["ret", "group", "window", "component"],
    lambda ret, group, window=120, component=0: _industry_pca_loading(ret, group, int(window), int(component)),
    unit="loading",
)


# ---------------------------------------------------------------------------
# § Model forecasts (features up to 4 panels + label panel)
# ---------------------------------------------------------------------------

def _gather(args: tuple[Any, ...], count: int, y: Any) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feats = [a.to_numpy(dtype=float) for a in args[:count] if a is not None]
    yv = y.to_numpy(dtype=float)
    n_rows, n_cols = yv.shape
    return feats, yv, y


def _forecast_loop(feats, yv, window, fn_train_predict, fit_lag: int = 1, label_horizon: int = 1) -> np.ndarray:
    """Walk-forward supervised forecast loop.

    ``fit_lag>=1`` (default) trains each model on rows strictly before the
    current row and predicts the *current* row's features, so the current
    label/features never enter the model that produces the current prediction
    (no same-row leakage even when the caller hands in an un-lagged label).

    ``label_horizon`` (default 1) is the P0-32/P0-33 maturity rule: a label
    anchored at row ``s`` is only *mature* at ``s + label_horizon``, so the
    last ``label_horizon`` rows of the training window are excluded from the
    fit.  ``fit_lag=1`` alone is not enough — with a forward-H label, even
    ``y_{t-1}`` needs prices through ``t-1+H`` to be knowable.  The operator
    enforces maturity itself instead of trusting the caller to lag/embargo.
    """
    n_rows, n_cols = yv.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    lag = max(0, int(fit_lag))
    h = max(1, int(label_horizon))
    for col in range(n_cols):
        Xstock = np.column_stack([f[:, col] for f in feats])
        ystock = yv[:, col]
        for row in range(n_rows):
            fit_end = row - lag
            if fit_end < 0:
                continue
            start = max(0, fit_end - int(window) + 1)
            train_end = fit_end - h
            if train_end < start:
                continue
            X = Xstock[start : train_end + 1]
            y = ystock[start : train_end + 1]
            out[row, col] = fn_train_predict(X, y, Xstock[row])
    return out


def _model_predict(X: np.ndarray, y: np.ndarray, x_cur: np.ndarray, method: str, n_components: int, alpha: float, l1_ratio: float) -> float:
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if valid.sum() < 10:
        return np.nan
    Xv, yv = X[valid], y[valid]
    mu = Xv.mean(axis=0)
    sd = np.std(Xv, axis=0)
    sd = np.where(sd > _EPS, sd, 1.0)
    Xs = (Xv - mu) / sd
    if method == "pcr":
        # M-022: a single predictor (p=1) cannot be PCA-compressed —
        # ``PCAState`` requires >= 2 active features.  Degrade to a legal
        # standardized linear regression (still respects ``fit_lag`` and
        # ``label_horizon`` maturity via ``_forecast_loop``'s slicing).
        if Xs.shape[1] == 1:
            design = np.column_stack([np.ones(len(Xs)), Xs])
            beta = fit_linear_model_checked(design, yv)
            if beta is None:
                return np.nan
            z = (x_cur - mu) / sd
            return float(beta[0] + beta[1:] @ z)
        # M-021: PCR is a REGRESSION — allow k <= p (every component).
        pca = _pca_svd(Xs, min(int(n_components), Xs.shape[1]), rank_policy="regression")
        if pca is None:
            return np.nan
        # P0-14: components live in the active sub-space only.
        Xa = Xs[:, pca["active"]]
        score = pca["loadings"] @ Xa.T
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(score.shape[1]), score.T]), yv, rcond=None)
        za = (x_cur - mu)[pca["active"]] / sd[pca["active"]]
        s = pca["loadings"] @ za
        return float(beta[0] + beta[1:] @ s)
    if method == "pls":
        return _pls1_predict(Xs, yv, int(n_components), (x_cur - mu) / sd)
    if method == "enet":
        return _enet_predict(Xs, yv, float(alpha), float(l1_ratio), (x_cur - mu) / sd)
    raise ValueError(f"unknown model: {method}")


def _pls1_predict(X: np.ndarray, y: np.ndarray, n_components: int, x_new: np.ndarray) -> float:
    """PLS1 with genuine multi-latent-component NIPALS deflation.

    Audit M05: ``n_components`` must extract that many latent components
    (deflate X and y per component), not silently ignore the extra ones.
    Round-6 P0-37: the NEW sample must be deflated in lockstep with the
    training X/y deflation.  Reusing the original ``x_new`` for every
    component's score ``dot(x_new, w)`` is algebraically wrong for 2+ latent
    components — it is correct only for PLS1's first component.  Each component
    scores the *residual* new sample ``x_new -= t_new * p`` after the previous
    component's loading ``p`` has been removed, which is exactly the NIPALS
    prediction rule and matches sklearn's ``PLSRegression.predict`` to numerical
    precision.
    """
    n = min(n_components, X.shape[1], X.shape[0] - 1)
    if n < 1:
        return np.nan
    Xc = X.copy()
    yc = y - y.mean()
    pred = y.mean()
    x_resid = np.asarray(x_new, dtype=float).copy()
    for _ in range(n):
        w = Xc.T @ yc
        nw = np.linalg.norm(w)
        if nw <= _EPS:
            break
        w = w / nw
        t = Xc @ w
        tt = float(np.dot(t, t))
        if tt <= _EPS:
            break
        p = (Xc.T @ t) / tt
        q = float(np.dot(t, yc)) / tt
        t_new = float(np.dot(x_resid, w))
        pred += t_new * q
        # P0-37: deflate the new sample with the same loading ``p`` used to
        # deflate the training X, so component 2+ scores the residual and the
        # multi-component regression coefficient is consistent.
        x_resid = x_resid - t_new * p
        Xc = Xc - np.outer(t, p)
        yc = yc - q * t
    return float(pred)


def _enet_predict(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float, x_new: np.ndarray) -> float:
    """Coordinate-descent ElasticNet with intercept (standardised features).

    Audit M06: a convergence tolerance + max_iter cap, and a non-converged fit
    fails closed (NaN) instead of returning the solver's last iterate.
    """
    n, p = X.shape
    if alpha <= 0 or l1_ratio < 0 or l1_ratio > 1:
        return np.nan
    beta = np.zeros(p)
    intercept = float(np.mean(y))
    yc = y - intercept
    max_iter = 500
    tol = 1e-8
    converged = False
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            rj = yc - X @ beta + beta[j] * X[:, j]
            rho = float(X[:, j] @ rj) / n
            z = np.sign(rho) * max(abs(rho) - float(alpha) * float(l1_ratio), 0.0) / (
                1.0 + float(alpha) * (1.0 - float(l1_ratio))
            )
            beta[j] = z
        if float(np.max(np.abs(beta - beta_old)) <= tol:
            converged = True
            break
    if not converged:
        return np.nan
    return float(intercept + float(x_new @ beta)) if p else np.nan


def _mk_forecast(name: str, method: str, description: str, params: list[str]):
    _mk(
        name, description, params, unit="forecast", pit_safe=False,
        fn=lambda y, x1=None, x2=None, x3=None, x4=None, window=120, n_components=5, alpha=0.01, l1_ratio=0.5, label_horizon=1: _forecast_generic(
            y, (x1, x2, x3, x4), int(window), method, int(n_components), float(alpha), float(l1_ratio), int(label_horizon),
        ),
    )


def _forecast_generic(y, feats, window, method, n_components, alpha, l1_ratio, label_horizon=1):
    yv = y.to_numpy(dtype=float)
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    # R35-P0-M05: a supervised panel forecast with zero feature panels is a
    # contract violation, not a runnable model — fail closed at the call
    # boundary instead of reaching ``np.column_stack([])`` in _forecast_loop.
    if not collected:
        raise ValueError(
            f"panel_rolling_{method}_forecast: at least one feature panel "
            "(x1..x4) is required (required_feature_count_min=1)"
        )
    return _frame_like(y, _forecast_loop(collected, yv, window,
                                         lambda X, yy, xc: _model_predict(X, yy, xc, method, n_components, alpha, l1_ratio),
                                         label_horizon=int(label_horizon)))


_mk_forecast("panel_rolling_pcr_forecast", "pcr", "滚动 PCR 预测（label_horizon 内训练标签自动排除，未成熟标签不参与拟合）。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。", ["y", "x1", "x2", "x3", "x4", "window", "n_components", "label_horizon"])
_mk_forecast("panel_rolling_pls_forecast", "pls", "滚动 PLS 预测（label_horizon 内未成熟标签自动排除）。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。", ["y", "x1", "x2", "x3", "x4", "window", "n_components", "label_horizon"])
_mk_forecast("panel_rolling_elastic_net_forecast", "enet", "滚动 ElasticNet 预测（label_horizon 内未成熟标签自动排除）。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。", ["y", "x1", "x2", "x3", "x4", "window", "alpha", "l1_ratio", "label_horizon"])


def _regime_forecast(y, feats, market_state, window, n_regimes, label_horizon=1):
    # R35-P0-M06: market_state is a REQUIRED panel for regime-conditioned
    # forecasts — the kernel dereferences it immediately, so a None default
    # would crash with an AttributeError instead of a contract error.
    if market_state is None:
        raise ValueError(
            "panel_regime_conditioned_forecast: market_state is a required "
            "panel (required panel param, cannot be None)"
        )
    yv = y.to_numpy(dtype=float)
    mv = market_state.to_numpy(dtype=float)
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    # M-033: market_state is a routing/state variable — it does NOT satisfy the
    # predictor requirement.  Fail closed with a clear message instead of
    # reaching ``np.column_stack([])`` and raising an opaque
    # "need at least one array to concatenate".
    if not collected:
        raise ValueError(
            "panel_regime_conditioned_forecast requires at least one predictor "
            "feature (market_state alone does not count)"
        )

    # per-stock, per-row regime assignment from market_state quantiles.  Both
    # the regime quantile edges and the training rows use the window *ending at
    # the previous row* (``start:row``), so the current market state is never
    # used to place itself into a regime and the current label never trains the
    # model that predicts it.  Round-6 P0-40: ``label_horizon`` additionally
    # excludes the last ``label_horizon`` rows from training — a forward-H label
    # is not mature at ``fit_end`` (P0-33), so un-matured labels must not fit.
    n_rows, n_cols = yv.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    nr = max(2, int(n_regimes))
    h = max(1, int(label_horizon))
    _LAST_FIT_TELEMETRY.clear()
    for col in range(n_cols):
        ms = mv[:, col]
        for row in range(n_rows):
            if row < 1 or not np.isfinite(ms[row]):
                continue
            start = max(0, row - int(window))
            hi = row - h
            if hi <= start:
                continue
            win_ms = ms[start:row]
            win_valid = np.isfinite(win_ms)
            if win_valid.sum() < nr * 5:
                continue
            edges = np.quantile(win_ms[win_valid], np.linspace(0, 1, nr + 1)[1:-1])
            reg = int(np.digitize(ms[row], edges))
            win_ms_tr = win_ms[: hi - start]
            win_valid_tr = win_valid[: hi - start]
            Xc = np.column_stack([f[start:hi, col] for f in collected])
            yc = yv[start:hi, col]
            mask = (
                np.all(np.isfinite(Xc), axis=1)
                & np.isfinite(yc)
                & win_valid_tr
                & (np.digitize(win_ms_tr, edges) == reg)
            )
            if mask.sum() < 8:
                continue
            Xm, ym = Xc[mask], yc[mask]
            mu, sd = Xm.mean(axis=0), np.std(Xm, axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            Xs = (Xm - mu) / sd
            # ModelDesignGate: a rank-deficient / ill-conditioned regime design
            # fails closed (NaN) instead of returning a meaningless coefficient.
            design = np.column_stack([np.ones(len(ym)), Xs])
            beta = fit_linear_model_checked(design, ym)
            # M-036: fit-quality telemetry (module-level diagnostic accessor).
            reg_tr = np.digitize(win_ms_tr[win_valid_tr], edges)
            reg_counts = np.bincount(reg_tr, minlength=nr)
            reg_p = reg_counts / max(int(reg_counts.sum()), 1)
            gate_entropy = float(-np.sum(reg_p[reg_p > 0] * np.log(reg_p[reg_p > 0])))
            _LAST_FIT_TELEMETRY.update({
                "model": "regime",
                "effective_train_obs": int(mask.sum()),
                "effective_regime_obs": int(mask.sum()),
                "condition_number": _design_cond(design),
                "convergence": bool(beta is not None),
                "gate_entropy": gate_entropy,
                "regime_index": int(reg),
                "n_regimes": int(nr),
            })
            if beta is None:
                continue
            x_cur = np.array([f[row, col] for f in collected], dtype=float)
            z = (x_cur - mu) / sd
            out[row, col] = float(beta[0] + beta[1:] @ z)
    return _frame_like(y, out)


_mk(
    "panel_regime_conditioned_forecast",
    "按市场状态分 regime 训练的条件线性预测（market_state_t 选择当前 regime；历史 regime 边界与专家参数严格截至 t-1；label_horizon 内未成熟标签自动排除）。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。",
    ["y", "x1", "x2", "x3", "x4", "market_state", "window", "n_regimes", "label_horizon"],
    lambda y, x1=None, x2=None, x3=None, x4=None, market_state=None, window=120, n_regimes=3, label_horizon=1: _regime_forecast(
        y, (x1, x2, x3, x4), market_state, int(window), int(n_regimes), int(label_horizon)),
    unit="forecast",
    pit_safe=False,
)


def _moe_forecast(y, feats, market_state, window, n_experts, label_horizon=1):
    # R35-P0-M06: market_state is a REQUIRED panel for mixture-of-experts.
    if market_state is None:
        raise ValueError(
            "panel_mixture_of_experts_score: market_state is a required "
            "panel (required panel param, cannot be None)"
        )
    yv = y.to_numpy(dtype=float)
    mv = market_state.to_numpy(dtype=float)
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    # M-033: market_state is a routing/state variable — it does NOT satisfy the
    # predictor requirement.  Fail closed with a clear message instead of
    # reaching ``np.column_stack([])`` in the (now dead) ``Xc.shape[1] == 0``
    # guard below.
    if not collected:
        raise ValueError(
            "panel_mixture_of_experts_score requires at least one predictor "
            "feature (market_state alone does not count)"
        )
    n_rows, n_cols = yv.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    ne = max(2, int(n_experts))
    h = max(1, int(label_horizon))
    _LAST_FIT_TELEMETRY.clear()
    for col in range(n_cols):
        ms = mv[:, col]
        for row in range(n_rows):
            if row < 1 or not np.isfinite(ms[row]):
                continue
            start = max(0, row - int(window))
            hi = row - h
            if hi <= start:
                continue
            win_ms = ms[start:row]
            win_valid = np.isfinite(win_ms)
            if win_valid.sum() < ne * 6:
                continue
            # Expert centers and gating scale are computed causally from the
            # window ending at the previous row (the current market state never
            # places itself into an expert bin).  Round-6 P0-40: training rows
            # stop at ``hi = row - label_horizon`` so un-matured forward-H
            # labels never fit the experts.
            raw_edges = np.quantile(win_ms[win_valid], np.linspace(0, 1, ne + 1)[1:-1])
            centers = np.unique(raw_edges)
            # R10-P0-030: a degenerate market_state (heavy ties / near-constant)
            # can collapse several requested expert bins into one — ``n_experts``
            # would silently become a smaller, nearly-identical model.  Fail
            # closed (NaN cell) instead of running with fewer real experts.
            if len(centers) + 1 < ne:
                continue
            win_ms_tr = win_ms[: hi - start]
            win_valid_tr = win_valid[: hi - start]
            Xc = np.column_stack([f[start:hi, col] for f in collected])
            yc = yv[start:hi, col]
            if Xc.shape[1] == 0 or len(yc) < 15:
                continue
            x_cur = np.array([f[row, col] for f in collected], dtype=float)
            preds = []
            row_cond_max = 0.0
            for e in range(ne):
                # A NaN market_state must never place a training sample into an
                # expert: win_valid_tr excludes missing-regime rows (np.digitize
                # would otherwise still assign a bin).
                if e < len(centers):
                    mask = (np.digitize(win_ms_tr, centers) == e) & np.all(np.isfinite(Xc), axis=1) & np.isfinite(yc) & win_valid_tr
                else:
                    mask = (np.digitize(win_ms_tr, centers) == ne - 1) & np.all(np.isfinite(Xc), axis=1) & np.isfinite(yc) & win_valid_tr
                if mask.sum() < 5:
                    preds.append(np.nan)
                    continue
                Xm, ym = Xc[mask], yc[mask]
                mu, sd = Xm.mean(axis=0), np.std(Xm, axis=0)
                sd = np.where(sd > _EPS, sd, 1.0)
                # ModelDesignGate: an ill-conditioned expert design fails closed.
                design = np.column_stack([np.ones(len(ym)), (Xm - mu) / sd])
                beta = fit_linear_model_checked(design, ym)
                if beta is None:
                    preds.append(np.nan)
                    continue
                row_cond_max = max(row_cond_max, _design_cond(design))
                z = (x_cur - mu) / sd
                preds.append(float(beta[0] + beta[1:] @ z))
            valid_preds = [p for p in preds if np.isfinite(p)]
            if not valid_preds:
                continue
            # softmax gating by distance to current market state.  ``centers``
            # holds the ne-1 internal quantile edges that define ne digitize
            # bins; each bin gets the midpoint of its edges as a representative
            # center so gates align 1:1 with the ne per-expert predictions.
            bin_edges = np.concatenate(([float(win_ms[win_valid].min())], centers, [float(win_ms[win_valid].max())]))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            dist = np.abs(bin_centers - ms[row])
            dist = np.where(np.isfinite(dist), dist, np.inf)
            if not np.all(np.isinf(dist)):
                gates = np.exp(-dist / max(float(np.std(win_ms[win_valid])), 1e-6))
                gates = gates / max(gates.sum(), _EPS)
                pred_arr = np.array(preds)
                # Audit M04: a NaN expert prediction must not silently lose its
                # gate mass.  Renormalize the gates over the FINITE experts only
                # and weight by that conditional distribution.  The gate vector
                # is indexed by the boolean mask FIRST (shape (n_finite,)) so the
                # subsequent multiply / dot never mixes shape(ne) x shape(n_finite).
                finite_experts = np.isfinite(pred_arr)
                if not np.any(finite_experts):
                    continue
                g = gates[finite_experts]
                p = pred_arr[finite_experts]
                g = g / g.sum()
                out[row, col] = float(np.dot(g, p))
                # M-036: fit-quality telemetry (module-level diagnostic accessor).
                eff_train = int(np.sum(
                    np.all(np.isfinite(Xc), axis=1) & np.isfinite(yc) & win_valid_tr
                ))
                _LAST_FIT_TELEMETRY.update({
                    "model": "moe",
                    "effective_train_obs": eff_train,
                    "active_expert_count": int(finite_experts.sum()),
                    "condition_number": row_cond_max,
                    "convergence": bool(len(valid_preds) == ne),
                    "gate_entropy": float(-np.sum(g * np.log(np.clip(g, 1e-12, None)))),
                    "expert_weight_max": float(g.max()),
                    "n_experts": int(ne),
                })
    return _frame_like(y, out)


_mk(
    "panel_mixture_of_experts_score",
    "基于市场状态的 Mixture-of-Experts 加权预测（训练窗口截至前一日，market_state_t 选择当前 gating；历史专家边界/参数严格截至 t-1；label_horizon 内未成熟标签自动排除）。逐股票独立滚动时序模型，非 pooled stock×date 模型（M-030）。",
    ["y", "x1", "x2", "x3", "x4", "market_state", "window", "n_experts", "label_horizon"],
    lambda y, x1=None, x2=None, x3=None, x4=None, market_state=None, window=120, n_experts=3, label_horizon=1: _moe_forecast(
        y, (x1, x2, x3, x4), market_state, int(window), int(n_experts), int(label_horizon)),
    unit="forecast",
    pit_safe=False,
)


def _autoencoder_error(feats, window, n_components: int = 2):
    """Per-stock feature PCA reconstruction error.

    Each input panel is a (timestamp x instrument) cross-section; the feature
    vector for one instrument is built by stacking that instrument's value
    across the supplied panels, so reconstruction error is per-instrument and
    preserves the input panel shape.  The number of retained principal
    components is ``n_components`` (must be strictly less than the feature
    count) so the reconstruction is a genuine low-rank compression instead of a
    near-perfect copy.

    Round-6 P0-38: the rank bound is the FEATURE count ``len(collected)``, not
    the instrument count ``collected[0].shape[1]``.  P0-39: features that are
    all-NaN / constant / sub-coverage within the training window are dropped
    from the active feature space before ``nanmean``/SVD (a degenerate feature
    must not inject NaN into the shared decomposition).
    """
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    # R10-P0-029: the operator is only meaningful with >= 2 features.  With a
    # single feature ``p=1`` the reconstruction is trivially perfect (rank
    # forced to 0 -> every output NaN), so a ``n_features == 1`` call was a
    # legal-but-always-NaN search dead end.  Fail closed at the call boundary
    # instead of emitting a full-NaN panel.
    if len(collected) < 2:
        raise ValueError(
            "ts_feature_pca_reconstruction_error requires at least 2 feature "
            f"panels (got {len(collected)}); a single feature cannot be "
            "low-rank compressed"
        )
    n_features = len(collected)
    rank = max(1, min(int(n_components), n_features - 1))
    n_rows, n_cols = collected[0].shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    for col in range(n_cols):
        X = np.column_stack([c[:, col] for c in collected])
        for row in range(n_rows):
            start = max(0, row - int(window) + 1)
            Xw = X[start:row]
            if Xw.shape[0] < 10:
                continue
            # P0-39 coverage gate: a feature with < 2 finite rows in the window
            # is INACTIVE (all-NaN column would otherwise leak NaN into the
            # SVD); a constant feature is kept with unit scale.  Fail closed
            # when fewer than 2 features survive.
            finite_count = np.sum(np.isfinite(Xw), axis=0)
            active = finite_count >= 2
            n_active = int(active.sum())
            if n_active < 2:
                continue
            sub = Xw[:, active]
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            Xc = np.where(np.isfinite(sub), sub, mu)
            Xs = (Xc - mu) / sd
            _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
            r = min(rank, n_active - 1, Vt.shape[0])
            if r < 1:
                continue
            row_active = X[row][active]
            z = (row_active - mu) / sd
            if not np.all(np.isfinite(z)):
                continue
            recon = Vt[:r].T @ (Vt[:r] @ z)
            out[row, col] = float(np.sqrt(np.sum((z - recon) ** 2)))
    return _frame_like(feats[0], out)


_mk(
    "ts_feature_pca_reconstruction_error",
    "每股多特征时序 PCA 低秩重构误差（n_components < 特征数）。",
    ["f1", "f2", "f3", "f4", "window", "n_components"],
    lambda f1, f2=None, f3=None, f4=None, window=120, n_components=2: _autoencoder_error(
        (f1, f2, f3, f4), int(window), int(n_components)),
    unit="distance",
)


# Historic name for the same kernel; kept registered for backward
# compatibility.  The honest name is ``ts_feature_pca_reconstruction_error``.
_mk(
    "cs_autoencoder_reconstruction_error",
    "线性自编码器（PCA 基准）重构误差（deprecated，请用 ts_feature_pca_reconstruction_error）。",
    ["f1", "f2", "f3", "f4", "window"],
    lambda f1, f2=None, f3=None, f4=None, window=120: _autoencoder_error((f1, f2, f3, f4), int(window)),
    unit="distance",
)

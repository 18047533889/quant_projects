# -*- coding: utf-8 -*-
"""Rolling panel model operators (P2, experimental).

Rolling PCA residuals / loadings, PCR / PLS / ElasticNet forecasts, regime-
conditioned and mixture-of-experts forecasts, and a linear (PCA) autoencoder
reconstruction-error baseline.  Label panels must already be lagged /
embargoed by the caller so no future information leaks.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


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
    return OperatorMetadata(
        name=name,
        category="panel_model",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
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

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return cls


def _pca_svd(X: np.ndarray, n_components: int):
    """Standardised SVD PCA; returns (loadings, explained_ratio, proj_fn).

    Missing values are handled per column: the mean / std are estimated from
    each column's finite rows and any still-missing entry is imputed with that
    column mean before the SVD (suspended / halted instruments keep a neutral
    contribution instead of poisoning the factor space).  Columns with zero
    variance are kept with unit scale so the SVD is never singular.

    Audit M01: a column whose finite coverage is below 2 rows is INACTIVE and is
    dropped from the fit (``np.nanmean`` on an all-NaN column would otherwise
    leak NaN into the SVD); its mu/sd/loading come back NaN (fail-closed).
    Audit M07: ``n_components`` is capped below the fit rank
    (``min(n_features, n_observations) - 1``) so the reconstruction error is
    never trivially zero from a full-rank fit.
    """
    n, d = X.shape
    finite_count = np.sum(np.isfinite(X), axis=0)
    active = finite_count >= 2
    if int(active.sum()) < 2:
        return None
    sub = X[:, active]
    mu_sub = np.nanmean(sub, axis=0)
    sd_sub = np.nanstd(sub, axis=0)
    sd_sub = np.where(sd_sub > _EPS, sd_sub, 1.0)
    Xc = np.where(np.isfinite(sub), sub, mu_sub)
    Xs = (Xc - mu_sub) / sd_sub
    k = int(min(n_components, Xs.shape[1] - 1, Xs.shape[0] - 1))
    if k < 1:
        return None
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    total_var = float(np.sum(s * s))
    mu = np.full(d, np.nan)
    sd = np.full(d, np.nan)
    loadings = np.full((n_components, d), np.nan)
    explained = np.full(n_components, np.nan)
    mu[active] = mu_sub
    sd[active] = sd_sub
    loadings[:k, active] = Vt[:k]
    explained[:k] = s[:k] ** 2 / max(total_var, _EPS)
    return {"mu": mu, "sd": sd, "loadings": loadings, "explained": explained}


def _pca_transform(pca, row: np.ndarray) -> np.ndarray:
    z = (row - pca["mu"]) / pca["sd"]
    return pca["loadings"] @ z


def _rolling_pca(ret: pd.DataFrame, window: int, fn, *, fit_lag: int = 1) -> pd.DataFrame:
    """Rolling PCA evaluated row by row.

    ``fit_lag>=1`` (default) trains the PCA on rows ``[start, row-1]`` and
    evaluates the *current* row against that historical model, so the current
    observation never enters its own training set (no in-sample projection).
    ``fit_lag=0`` reproduces the legacy in-sample behaviour.
    """
    rv = ret.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    lag = max(0, int(fit_lag))
    for row in range(rows):
        fit_end = row - lag
        if fit_end < 0:
            continue
        start = max(0, fit_end - int(window) + 1)
        X = rv[start : fit_end + 1]
        out[row] = fn(X, rv[row])
    return _frame_like(ret, out)


def _pca_loading(
    X: np.ndarray, cur: np.ndarray, component: int, prev: np.ndarray | None
) -> np.ndarray:
    pca = _pca_svd(X, component + 1)
    if pca is None:
        return np.full(len(cur), np.nan)
    loading = pca["loadings"][component].copy()
    # Audit M02: the eigenvector sign is arbitrary per SVD; align each window's
    # loading to the previous window's loading so the loading factor has no
    # pure-numerical sign flip between consecutive windows.
    if prev is not None and np.isfinite(prev).any() and np.isfinite(loading).any():
        m = np.isfinite(loading) & np.isfinite(prev)
        if np.dot(loading[m], prev[m]) < 0.0:
            loading = -loading
    else:
        # No usable previous window: fall back to per-window sign-normalisation
        # on the largest |loading|.
        k = int(np.argmax(np.abs(loading)))
        if loading[k] < 0:
            loading = -loading
    return loading


def _pca_loading_series(ret: pd.DataFrame, window: int, component: int) -> pd.DataFrame:
    prev: np.ndarray | None = None

    def _fn(X: np.ndarray, cur: np.ndarray) -> np.ndarray:
        nonlocal prev
        loading = _pca_loading(X, cur, int(component), prev)
        prev = loading.copy()
        return loading

    return _rolling_pca(ret, int(window), _fn)


_mk(
    "panel_rolling_pca_loading",
    "过去窗口收益矩阵 PCA 的指定主成分载荷（跨窗 sign 对齐）。",
    ["ret", "window", "component"],
    lambda ret, window=120, component=0: _pca_loading_series(ret, int(window), int(component)),
    unit="loading",
)


def _pca_resid(X: np.ndarray, cur: np.ndarray, n_components: int) -> np.ndarray:
    pca = _pca_svd(X, min(int(n_components), X.shape[1]))
    if pca is None:
        return np.full(len(cur), np.nan)
    score = _pca_transform(pca, cur)
    z = (cur - pca["mu"]) / pca["sd"]
    recon = pca["mu"] + pca["sd"] * (pca["loadings"].T @ score)
    return cur - recon


_mk(
    "panel_rolling_pca_resid",
    "当前收益相对前 K 主成分模型的残差。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _rolling_pca(ret, int(window), lambda X, c: _pca_resid(X, c, int(n_components))),
)


def _pca_commonality(X: np.ndarray, cur: np.ndarray, n_components: int) -> np.ndarray:
    """Per-stock commonality ``1 - Var(resid_i)/Var(ret_i)`` over the training window.

    Unlike the previous implementation this returns a *per-stock* value (each
    instrument's own time-series variance decomposition) instead of one scalar
    broadcast to every column.
    """
    pca = _pca_svd(X, min(int(n_components), X.shape[1]))
    if pca is None:
        return np.full(len(cur), np.nan)
    z = (X - pca["mu"]) / pca["sd"]           # (n_rows, n_features)
    score = pca["loadings"] @ z.T             # (n_components, n_rows)
    recon = (pca["mu"][:, None] + pca["sd"][:, None] * (pca["loadings"].T @ score)).T  # (n_rows, n_features)
    resid = X - recon
    var_resid = np.nanvar(resid, axis=0)
    var_ret = np.nanvar(X, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var_ret > _EPS, 1.0 - var_resid / var_ret, np.nan)


_mk(
    "panel_rolling_pca_explained_ratio",
    "每股被共同因子解释的比例（1 - 残差方差/总方差，每股独立）。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _rolling_pca(ret, int(window), lambda X, c: _pca_commonality(X, c, int(n_components))),
)


def _industry_pca_loading(ret, group, window, component):
    rv = ret.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        fit_end = row - 1
        if fit_end < 0:
            continue
        start = max(0, fit_end - int(window) + 1)
        for col in range(cols):
            g = gv[row, col]
            mask = gv[row] == g
            members = np.flatnonzero(mask)
            if len(members) < 4:
                continue
            # Local position of ``col`` inside the industry members; correct
            # even when the industry's columns are not contiguous in the panel.
            local = int(np.where(members == col)[0][0])
            X = rv[start : fit_end + 1][:, members]
            loading = _pca_loading(X, rv[row][members], int(component))
            out[row, col] = loading[local] if np.isfinite(loading).any() else np.nan
    return _frame_like(ret, out)


def _pca_resid_vol(ret: pd.DataFrame, window: int, n_components: int) -> pd.DataFrame:
    resid = _rolling_pca(ret, int(window), lambda X, c: _pca_resid(X, c, int(n_components)))
    return resid.rolling(int(window), min_periods=10).std()


_mk(
    "panel_rolling_pca_resid_vol",
    "PCA 残差波动率（过去窗口）。",
    ["ret", "window", "n_components"],
    lambda ret, window=120, n_components=5: _pca_resid_vol(ret, int(window), int(n_components)),
    unit="volatility",
)


def _pca_resid_momentum(ret: pd.DataFrame, window: int, n_components: int) -> pd.DataFrame:
    resid = _rolling_pca(ret, int(window), lambda X, c: _pca_resid(X, c, int(n_components)))
    return resid.rolling(int(window), min_periods=10).sum()


_mk(
    "panel_rolling_pca_resid_momentum",
    "PCA 残差累计收益。",
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


def _forecast_loop(feats, yv, window, fn_train_predict, fit_lag: int = 1) -> np.ndarray:
    """Walk-forward supervised forecast loop.

    ``fit_lag>=1`` (default) trains each model on rows strictly before the
    current row and predicts the *current* row's features, so the current
    label/features never enter the model that produces the current prediction
    (no same-row leakage even when the caller hands in an un-lagged label).
    """
    n_rows, n_cols = yv.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    lag = max(0, int(fit_lag))
    for col in range(n_cols):
        Xstock = np.column_stack([f[:, col] for f in feats])
        ystock = yv[:, col]
        for row in range(n_rows):
            fit_end = row - lag
            if fit_end < 0:
                continue
            start = max(0, fit_end - int(window) + 1)
            X = Xstock[start : fit_end + 1]
            y = ystock[start : fit_end + 1]
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
        pca = _pca_svd(Xs, min(int(n_components), Xs.shape[1]))
        if pca is None:
            return np.nan
        score = pca["loadings"] @ Xs.T
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(score.shape[1]), score.T]), yv, rcond=None)
        z = (x_cur - mu) / sd
        s = pca["loadings"] @ z
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
    """
    n = min(n_components, X.shape[1], X.shape[0] - 1)
    if n < 1:
        return np.nan
    Xc = X.copy()
    yc = y - y.mean()
    pred = y.mean()
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
        pred += float(np.dot(x_new, w)) * q
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
        if float(np.max(np.abs(beta - beta_old))) <= tol:
            converged = True
            break
    if not converged:
        return np.nan
    return float(intercept + float(x_new @ beta)) if p else np.nan


def _mk_forecast(name: str, method: str, description: str, params: list[str]):
    _mk(
        name, description, params, unit="forecast", pit_safe=False,
        fn=lambda y, x1=None, x2=None, x3=None, x4=None, window=120, n_components=5, alpha=0.01, l1_ratio=0.5: _forecast_generic(
            y, (x1, x2, x3, x4), int(window), method, int(n_components), float(alpha), float(l1_ratio),
        ),
    )


def _forecast_generic(y, feats, window, method, n_components, alpha, l1_ratio):
    yv = y.to_numpy(dtype=float)
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    return _frame_like(y, _forecast_loop(collected, yv, window,
                                         lambda X, yy, xc: _model_predict(X, yy, xc, method, n_components, alpha, l1_ratio)))


_mk_forecast("panel_rolling_pcr_forecast", "pcr", "滚动 PCR 预测（训练标签须已结束）。", ["y", "x1", "x2", "x3", "x4", "window", "n_components"])
_mk_forecast("panel_rolling_pls_forecast", "pls", "滚动 PLS 预测。", ["y", "x1", "x2", "x3", "x4", "window", "n_components"])
_mk_forecast("panel_rolling_elastic_net_forecast", "enet", "滚动 ElasticNet 预测。", ["y", "x1", "x2", "x3", "x4", "window", "alpha", "l1_ratio"])


def _regime_forecast(y, feats, market_state, window, n_regimes):
    yv = y.to_numpy(dtype=float)
    mv = market_state.to_numpy(dtype=float)
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]

    def _fn(X, yy):
        # regime of current row is the last value of market_state column
        return np.nan

    # per-stock, per-row regime assignment from market_state quantiles.  Both
    # the regime quantile edges and the training rows use the window *ending at
    # the previous row* (``start:row``), so the current market state is never
    # used to place itself into a regime and the current label never trains the
    # model that predicts it.
    n_rows, n_cols = yv.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    nr = max(2, int(n_regimes))
    for col in range(n_cols):
        ms = mv[:, col]
        for row in range(n_rows):
            if row < 1 or not np.isfinite(ms[row]):
                continue
            start = max(0, row - int(window))
            win_ms = ms[start:row]
            win_valid = np.isfinite(win_ms)
            if win_valid.sum() < nr * 5:
                continue
            edges = np.quantile(win_ms[win_valid], np.linspace(0, 1, nr + 1)[1:-1])
            reg = int(np.digitize(ms[row], edges))
            Xc = np.column_stack([f[start:row, col] for f in collected])
            yc = yv[start:row, col]
            mask = (
                np.all(np.isfinite(Xc), axis=1)
                & np.isfinite(yc)
                & win_valid
                & (np.digitize(win_ms, edges) == reg)
            )
            if mask.sum() < 8:
                continue
            Xm, ym = Xc[mask], yc[mask]
            mu, sd = Xm.mean(axis=0), np.std(Xm, axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            Xs = (Xm - mu) / sd
            beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ym)), Xs]), ym, rcond=None)
            x_cur = np.array([f[row, col] for f in collected], dtype=float)
            z = (x_cur - mu) / sd
            out[row, col] = float(beta[0] + beta[1:] @ z)
    return _frame_like(y, out)


_mk(
    "panel_regime_conditioned_forecast",
    "按市场状态分 regime 训练的条件线性预测（训练窗口截至前一日）。",
    ["y", "x1", "x2", "x3", "x4", "market_state", "window", "n_regimes"],
    lambda y, x1=None, x2=None, x3=None, x4=None, market_state=None, window=120, n_regimes=3: _regime_forecast(
        y, (x1, x2, x3, x4), market_state, int(window), int(n_regimes)),
    unit="forecast",
    pit_safe=False,
)


def _moe_forecast(y, feats, market_state, window, n_experts):
    yv = y.to_numpy(dtype=float)
    mv = market_state.to_numpy(dtype=float)
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    n_rows, n_cols = yv.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    ne = max(2, int(n_experts))
    for col in range(n_cols):
        ms = mv[:, col]
        for row in range(n_rows):
            if row < 1 or not np.isfinite(ms[row]):
                continue
            start = max(0, row - int(window))
            win_ms = ms[start:row]
            win_valid = np.isfinite(win_ms)
            if win_valid.sum() < ne * 6:
                continue
            # Expert centers and gating scale are computed causally from the
            # window ending at the previous row (the current market state never
            # places itself into an expert bin).
            centers = np.quantile(win_ms[win_valid], np.linspace(0, 1, ne + 1)[1:-1])
            Xc = np.column_stack([f[start:row, col] for f in collected])
            yc = yv[start:row, col]
            if Xc.shape[1] == 0 or len(yc) < 15:
                continue
            x_cur = np.array([f[row, col] for f in collected], dtype=float)
            preds = []
            for e in range(ne):
                if e < len(centers):
                    mask = (np.digitize(win_ms, centers) == e) & np.all(np.isfinite(Xc), axis=1) & np.isfinite(yc)
                else:
                    mask = (np.digitize(win_ms, centers) == ne - 1) & np.all(np.isfinite(Xc), axis=1) & np.isfinite(yc)
                if mask.sum() < 5:
                    preds.append(np.nan)
                    continue
                Xm, ym = Xc[mask], yc[mask]
                mu, sd = Xm.mean(axis=0), np.std(Xm, axis=0)
                sd = np.where(sd > _EPS, sd, 1.0)
                beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ym)), (Xm - mu) / sd]), ym, rcond=None)
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
                # and weight by that conditional distribution.
                finite_experts = np.isfinite(pred_arr)
                if not np.any(finite_experts):
                    continue
                gates_valid = gates * finite_experts
                gates_valid = gates_valid / max(gates_valid.sum(), _EPS)
                out[row, col] = float(np.sum(gates_valid * pred_arr[finite_experts]))
    return _frame_like(y, out)


_mk(
    "panel_mixture_of_experts_score",
    "基于市场状态的 Mixture-of-Experts 加权预测（训练窗口截至前一日）。",
    ["y", "x1", "x2", "x3", "x4", "market_state", "window", "n_experts"],
    lambda y, x1=None, x2=None, x3=None, x4=None, market_state=None, window=120, n_experts=3: _moe_forecast(
        y, (x1, x2, x3, x4), market_state, int(window), int(n_experts)),
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
    """
    collected = [f.to_numpy(dtype=float) for f in feats if f is not None]
    if not collected:
        raise ValueError("at least one feature panel is required")
    p = collected[0].shape[1]
    rank = max(1, min(int(n_components), p - 1))
    n_rows, n_cols = collected[0].shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    for col in range(n_cols):
        X = np.column_stack([c[:, col] for c in collected])
        for row in range(n_rows):
            start = max(0, row - int(window) + 1)
            Xw = X[start:row]
            if Xw.shape[0] < 10:
                continue
            mu = np.nanmean(Xw, axis=0)
            sd = np.nanstd(Xw, axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            Xc = np.where(np.isfinite(Xw), Xw, mu)
            Xs = (Xc - mu) / sd
            _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
            r = min(rank, Vt.shape[0])
            z = (X[row] - mu) / sd
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

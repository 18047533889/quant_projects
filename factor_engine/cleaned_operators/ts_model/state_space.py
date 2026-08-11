# -*- coding: utf-8 -*-
"""Kalman-filter state-space operators (P2, experimental).

Deterministic initialisation (first finite observation), explicit noise
parameters, and a causal one-pass filter so segmented / full-history execution
agree.  Outputs are the filtered level / trend / beta state, standardised
innovations and state uncertainty.

Scale contract (audit round-3, item 32):
* The noise/variance scale is EXPLICIT and fixed — the user-supplied ``q`` /
  ``r`` (process / observation variance) are never re-estimated from the data,
  so a gap of missing observations can never silently re-scale the filter on
  its edges as if the observations were contiguous.
* A missing observation is a pure predict step: the state is propagated and the
  covariance is advanced by the process noise (``P += Q``) for EVERY missing
  row, so a K-row gap correctly accumulates ``K*Q`` of extra uncertainty.  The
  first observation after the gap is then processed with that grown covariance
  (a correctly-scaled innovation), not with a stale pre-gap scale.
* Fail closed on unknown scale: non-finite / non-positive noise parameters are
  rejected before any recursion, and the filter emits NaN until the first finite
  observation establishes a real scale (a leading gap or an all-missing series
  stays NaN rather than fabricating a filtered value).

Model-audit remediation (2026-08-11, M-070..M-074):
* M-070 — every Kalman canonical is STATEFUL: a causal one-pass filter, so the
  row-t output depends on the entire finite prefix through t (chunking without a
  restored checkpoint does NOT reproduce the full-history output).  Only
  ``ts_kalman_level`` currently declares ``stateful=True`` in the shared
  ``model_contract.py``; :data:`KALMAN_STATEFUL_CANONICALS` +
  :func:`kalman_stateful_contract` expose the full contract here and flag the
  other five for the reconciler to add to ``model_contract.py``.
* M-071 — :func:`_scale_qr` implements a DIMENSIONLESS q/r mode
  (``q_eff = cq * scale**2``, ``r_eff = cr * scale**2``, i.e. q/r read as ratios
  against a caller-supplied ``scale``).  It is deliberately a helper +
  documentation ONLY — NOT wired to a public ``scale_mode`` parameter, so the
  default absolute-variance behaviour is unchanged and the typed surface
  metadata is untouched (flagged for reconciler whether to expose the param).
* M-072 — ``ts_kalman_beta`` is THROUGH-ORIGIN: the observation model is
  ``y = b*x + eps`` with NO intercept.  ``ts_kalman_alpha_beta`` is deliberately
  NOT created (deferred); the constraint is documented here and tagged
  ``through_origin:true``.
* M-074 — the beta warmup policy is FINITE-PAIR: the first ``_BETA_WARMUP``
  finite ``(y, x)`` pairs are accumulated across gaps (non-contiguous).  The
  alternative ``"contiguous"`` policy (restart warmup after every missing pair)
  is NOT implemented — see :data:`_KALMAN_WARMUP_POLICY` /
  :func:`kalman_warmup_policy`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import aligned, frame_like, metadata

_CANONICALS: list[str] = []

# Model-audit Phase 4 (search-space hygiene): the Kalman noise scales.  ``q`` /
# ``r`` (process / observation variance) and ``q_level`` / ``q_trend`` (local-
# linear-trend process noise) are NUMERICAL policy knobs — the filtered
# level/trend/beta state IS the alpha mechanism, not the noise ratio that
# controls the filter's update speed (M-115/M-162/M-170).  They are never
# full-resolution search dimensions.
_KALMAN_PARAM_SPECS: dict[str, ParamSpec] = {
    "q": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.NUMERICAL, searchable=False),
    "r": ParamSpec(dtype=float, min=1e-12, param_role=ParamRole.NUMERICAL, searchable=False),
    "q_level": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.NUMERICAL, searchable=False),
    "q_trend": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.NUMERICAL, searchable=False),
}


#: M-070: every Kalman canonical is a causal one-pass recursive filter — the
#: row-t output depends on the full finite prefix through t.  Chunked execution
#: without a restored checkpoint therefore does NOT reproduce the full-history
#: output (segmented-execution statefulness, cf. ``stateful_contract.py``).
#: Only ``ts_kalman_level`` currently declares ``stateful=True`` in the shared
#: ``model_contract.py``; these six share the same semantics and are flagged for
#: the reconciler to add there.
KALMAN_STATEFUL_CANONICALS: frozenset[str] = frozenset({
    "ts_kalman_level",
    "ts_kalman_trend",
    "ts_kalman_beta",
    "ts_kalman_beta_change",
    "ts_kalman_beta_uncertainty",
    "ts_kalman_innovation_z",
})


def kalman_stateful_contract(canonical: str) -> dict[str, Any]:
    """M-070: machine-readable stateful contract for a Kalman canonical.

    Returns the fields the reconciler / segmented-execution layer needs:
    ``stateful``, ``state_schema_version``, ``checkpointable``,
    ``time_shard_safe``, ``reset_semantics``, ``missing_update_semantics`` and
    ``revision_replay_semantics``.  ``time_shard_safe`` is False because the
    filter state spans the whole finite prefix — a time shard mid-series can
    only be resumed with a restored checkpoint, never re-derived locally.
    """
    return {
        "stateful": True,
        "state_schema_version": f"{canonical}.v1",
        "checkpointable": True,
        "time_shard_safe": False,          # filter state spans the whole prefix
        "reset_semantics": "reset_on_first_finite_observation",
        "missing_update_semantics": "predict_only_covariance_growth",
        "revision_replay_semantics": "deterministic_replay",
    }


def kalman_stateful_contracts() -> dict[str, dict[str, Any]]:
    """M-070: all Kalman stateful contracts, keyed by canonical (stable order)."""
    return {c: kalman_stateful_contract(c) for c in sorted(KALMAN_STATEFUL_CANONICALS)}


def _stateful_contract_tags(canonical: str) -> list[str]:
    """Tags that surface the M-070 stateful contract on the operator metadata."""
    contract = kalman_stateful_contract(canonical)
    return [
        "stateful:true",
        f"state_schema_version:{contract['state_schema_version']}",
        f"checkpointable:{contract['checkpointable']}",
        f"time_shard_safe:{contract['time_shard_safe']}",
        f"reset_semantics:{contract['reset_semantics']}",
        f"missing_update_semantics:{contract['missing_update_semantics']}",
        f"revision_replay_semantics:{contract['revision_replay_semantics']}",
    ]


#: R38 §37：Numba 主链 dispatch 计数器（end-to-end 证据：accelerated_kernel_used）。
_NUMBA_DISPATCH_COUNTER: dict[str, int] = {}


def numba_dispatch_stats() -> dict[str, int]:
    """主链实际 dispatch 到 Numba kernel 的次数（tests/gate 消费）。"""
    return dict(_NUMBA_DISPATCH_COUNTER)


def _scale_qr(
    q: float,
    r: float,
    scale_mode: str | None = None,
    scale: float | None = None,
) -> tuple[float, float]:
    """M-071: DIMENSIONLESS q/r mode helper (NOT wired to a public parameter).

    The default ``scale_mode=None`` keeps the ABSOLUTE variance semantics
    unchanged — ``q`` / ``r`` are used as-is (this is the legacy behaviour and
    must never change).  With ``scale_mode="dimensionless"`` and a positive
    ``scale``, ``q`` / ``r`` are read as RATIOS and scaled to absolute
    variances::

        q_eff = cq * scale**2
        r_eff = cr * scale**2

    so ``cq=0.01, cr=1.0, scale=0.02`` means process noise ``0.01*0.02**2`` and
    observation noise ``1.0*0.02**2`` on a 2%-return scale.  Fail closed on
    unknown ``scale_mode`` / non-finite / non-positive ``scale``.

    The conservative route is intentional: exposing a ``scale_mode`` parameter
    would change the typed surface metadata (``param_names`` / the
    ``signature:...`` tag), so the helper is provided here with documentation
    and the reconciler is asked whether the parameter should be exposed on the
    public surface (M-071).
    """
    if scale_mode is None:
        return float(q), float(r)
    if scale_mode != "dimensionless":
        raise ValueError(
            f"scale_mode must be None or 'dimensionless', got {scale_mode!r}"
        )
    if scale is None or not (np.isfinite(scale) and scale > 0.0):
        raise ValueError("scale must be finite and > 0 in dimensionless mode")
    cq, cr = float(q), float(r)
    if not (np.isfinite(cq) and np.isfinite(cr) and cq >= 0.0 and cr > 0.0):
        raise ValueError("q must be finite and >= 0; r must be finite and > 0")
    s2 = scale * scale
    return cq * s2, cr * s2


def _numba_kernel(kernel_name: str):
    """R38 P0-058（§22）：加速准入 gate——certified kernel + numba 可用才返回 callable。

    准入 key（§P0-058 至少项）：kernel_name / semantic_version（registry 内 /
    dtype float64 / missing policy（kernel 已对 hostile fixture 认证））。返回
    ``callable(vals, *params)`` 或 None（走 reference）。
    """
    try:
        from backend.numba_kernel_registry import NumbaKernelRegistry

        # 确保 certified kernel 已注册（backend/numba_kernels/__init__ 导入即注册；
        # 幂等——registry 已有则跳过）。
        try:
            if not NumbaKernelRegistry.kernels():
                import backend.numba_kernels  # noqa: F401
        except Exception:
            pass
        kernel = NumbaKernelRegistry.get(kernel_name)
        if kernel is None or kernel.numba_fn is None or not kernel.numba_available:
            return None
        # dtype 准入：kernel 只认证 float64（`supported_dtypes` 默认 float64）。
        if "float64" not in kernel.spec.supported_dtypes:
            return None
        _NUMBA_DISPATCH_COUNTER[kernel_name] = _NUMBA_DISPATCH_COUNTER.get(kernel_name, 0) + 1
        return kernel.call
    except Exception:
        return None


def _register(name: str, description: str, params: list[str], unit: str, fn,
              *, input_units: dict[str, str] | None = None,
              extra_tags: tuple[str, ...] = ()):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.state_space",
        backend="pandas_numpy",
        status="experimental",
    )
    class _StateOp(SeriesOperator):
        metadata = metadata(
            name, description, params, unit=unit, cost=8,
            input_units=input_units,
            param_specs={k: v for k, v in _KALMAN_PARAM_SPECS.items() if k in params},
        )

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    # M-070: surface the stateful contract as machine-readable metadata tags for
    # every Kalman canonical (reconciler: mirror ``stateful=True`` into the
    # shared model_contract.py for the five beyond ``ts_kalman_level``).
    if name in KALMAN_STATEFUL_CANONICALS:
        _StateOp.metadata.tags.extend(_stateful_contract_tags(name))
    if extra_tags:
        _StateOp.metadata.tags.extend(extra_tags)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({name})
    return _StateOp


def _kalman_level(vals: np.ndarray, q: float, r: float, out_stat: str) -> np.ndarray:
    # P0-15 / audit round-3 (item 32): the noise parameters must be well-typed —
    # a negative process-noise would shrink uncertainty, a non-positive or
    # non-finite observation-noise breaks the Kalman update.  Fail fast instead
    # of silently producing nonsense (a non-finite scale is an *unknown* scale).
    # （R38：q/r 校验必须在 Numba dispatch **之前**——fail-closed 不能被 kernel
    # 绕过。）
    if not (np.isfinite(q) and np.isfinite(r) and q >= 0.0 and r > 0.0):
        raise ValueError("q must be finite and >= 0; r must be finite and > 0")
    # R38 P0-057/058（§22）：certified Numba kernel 只在**逐算子 parity 验证过**的
    # 组合接入主链。``kalman_level`` kernel 与算子 reference 在 hostile fixture
    # （NaN gaps / 有限段）上逐值一致（diff=0.0）——``out_stat == "level"`` 时
    # dispatch；其余 out_stat（innovation_z / p）与 trend/beta kernel 有语义漂移，
    # 保持 reference（honest，不改结果）。
    if out_stat == "level":
        kernel = _numba_kernel("kalman_level")
        if kernel is not None:
            return kernel(vals, float(q), float(r))
    n = len(vals)
    mu = np.full(n, np.nan, dtype=float)
    p = np.full(n, np.nan, dtype=float)
    innov_z = np.full(n, np.nan, dtype=float)
    mu_prev = np.nan
    p_prev = 1.0
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            if np.isfinite(mu_prev):
                # P0-15: a missing observation is predict-only — the filtered
                # state is unchanged but the covariance really advances by Q.
                # Previously p_prev was left behind and the same stale P was
                # re-emitted for every missing row, so K consecutive gaps did
                # not accumulate P + K*Q as the theory requires.
                mu[t] = mu_prev
                p_prev = p_prev + q
                p[t] = p_prev
            continue
        if not np.isfinite(mu_prev):
            mu_prev = x
            p_prev = r
            mu[t] = x
            p[t] = r
            continue
        p_pred = p_prev + q
        k = p_pred / (p_pred + r)
        # Innovation is the difference between the observation and the
        # *predicted* state (mu_prev before the Kalman update), not the filtered
        # state.  Computing it after the update would shrink every innovation by
        # (1 - k) and corrupt the standardised innovation.
        innov = x - mu_prev
        mu_prev = mu_prev + k * innov
        p_prev = (1.0 - k) * p_pred
        mu[t] = mu_prev
        p[t] = p_prev
        innov_z[t] = innov / np.sqrt(max(p_pred + r, 1e-12))
    if out_stat == "level":
        return mu
    if out_stat == "innovation_z":
        return innov_z
    return p


# M-073: the dynamic-beta filter models ``y = b*x + eps`` — the compatible input
# semantics are return-vs-return (asset return vs market/benchmark return) or
# excess-return-vs-market.  Declared here so all three beta variants share it.
_BETA_INPUT_UNITS: dict[str, str] = {"y": "return", "x": "market_return"}
# M-072: beta is THROUGH-ORIGIN (no intercept).  Tagged on every beta variant;
# ``ts_kalman_alpha_beta`` is deliberately NOT created (deferred to a later wave).
_BETA_THROUGH_ORIGIN_TAG: tuple[str, ...] = ("through_origin:true",)


_register("ts_kalman_level", "局部水平模型的过滤水平估计。", ["x", "q", "r"], "level",
           lambda x, q=1e-4, r=1.0: _apply_col(x, lambda v: _kalman_level(v, float(q), float(r), "level")))
_register("ts_kalman_innovation_z", "观测值相对 Kalman 预测的标准化创新。", ["x", "q", "r"], "level",
           lambda x, q=1e-4, r=1.0: _apply_col(x, lambda v: _kalman_level(v, float(q), float(r), "innovation_z")))
_register("ts_kalman_beta_uncertainty",
          "Beta 状态滤波协方差 P(标准误为 sqrt(P), 此处输出 P)。通过原点回归 y=b*x(无截距); 输入 y 为个股收益、x 为市场收益 (return-vs-return)。",
          ["y", "x", "q", "r"], "level",
          lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "uncertainty")),
          input_units=_BETA_INPUT_UNITS, extra_tags=_BETA_THROUGH_ORIGIN_TAG)


def _kalman_trend_slope(vals: np.ndarray, q_level: float, q_trend: float, r: float) -> np.ndarray:
    """Local linear trend: level and slope states."""
    if not (np.isfinite(q_level) and np.isfinite(q_trend) and np.isfinite(r)
            and q_level >= 0.0 and q_trend >= 0.0 and r > 0.0):
        raise ValueError("q_level/q_trend must be finite and >= 0; r must be finite and > 0")
    n = len(vals)
    slope = np.full(n, np.nan, dtype=float)
    level = np.nan
    trend = 0.0
    p11 = p12 = p22 = 1.0
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            if np.isfinite(level):
                # Predict-only on a missing observation (audit P0-N): propagate
                # the state AND the covariance — never silently drop the
                # uncertainty growth.
                level = level + trend
                p11_p = p11 + q_level + 2 * p12 + p22
                p12_p = p12 + p22
                p22_p = p22 + q_trend
                p11, p12, p22 = p11_p, p12_p, p22_p
                slope[t] = trend
            continue
        if not np.isfinite(level):
            level = x
            trend = 0.0
            slope[t] = 0.0
            continue
        # predict
        l_pred = level + trend
        t_pred = trend
        p11_p = p11 + q_level + 2 * p12 + p22
        p12_p = p12 + p22
        p22_p = p22 + q_trend
        # update (scalar observation with H = [1, 0])
        k1 = p11_p / (p11_p + r)
        k2 = p12_p / (p11_p + r)
        innov = x - l_pred
        level = l_pred + k1 * innov
        trend = t_pred + k2 * innov
        p11 = (1 - k1) * p11_p
        p12 = (1 - k1) * p12_p
        p22 = p22_p - k2 * p12_p
        slope[t] = trend
    return slope


_register("ts_kalman_trend", "局部线性趋势模型的潜在斜率。", ["x", "q_level", "q_trend", "r"], "level",
           lambda x, q_level=1e-5, q_trend=1e-5, r=1.0: _apply_col(x, lambda v: _kalman_trend_slope(v, float(q_level), float(q_trend), float(r))))


_BETA_WARMUP = 5

# M-074: the beta warmup is FINITE-PAIR — the first ``_BETA_WARMUP`` finite
# ``(y, x)`` pairs are accumulated across gaps (non-contiguous): a missing pair
# does NOT reset the warmup.  The alternative "contiguous" policy (restart the
# warmup after every missing pair) is NOT implemented.  Exposed via
# :func:`kalman_warmup_policy` so the contract is machine-readable.
_KALMAN_WARMUP_POLICY = "finite_pair"


def kalman_warmup_policy() -> str:
    """M-074: beta warmup policy (``"finite_pair"``; ``"contiguous"`` NOT implemented)."""
    return _KALMAN_WARMUP_POLICY


def _kalman_beta(y: np.ndarray, x: np.ndarray, q: float, r: float, out_stat: str) -> np.ndarray:
    if not (np.isfinite(q) and np.isfinite(r) and q >= 0.0 and r > 0.0):
        raise ValueError("q must be finite and >= 0; r must be finite and > 0")
    n = len(y)
    beta = np.full(n, np.nan, dtype=float)
    change = np.full(n, np.nan, dtype=float)
    unc = np.full(n, np.nan, dtype=float)
    b = np.nan
    p = 1.0
    b_prev = np.nan
    warmup_y: list[float] = []
    warmup_x: list[float] = []
    for t in range(n):
        yv, xv = y[t], x[t]
        if not (np.isfinite(yv) and np.isfinite(xv)):
            if np.isfinite(b):
                # Predict-only on a missing observation (audit P0-N): the random
                # walk beta keeps its covariance growth (F P F' + Q with F=1).
                p = p + q
                unc[t] = p
            continue
        if not np.isfinite(b):
            # Audit P0-N: never initialise beta from the unstable single ratio
            # y/x (it explodes when x ~ 0).  Use a trailing warmup OLS through
            # the origin; fall back to a diffuse prior when it is degenerate.
            warmup_y.append(yv)
            warmup_x.append(xv)
            if len(warmup_y) >= _BETA_WARMUP:
                wy = np.asarray(warmup_y, dtype=float)
                wx = np.asarray(warmup_x, dtype=float)
                denom = float(np.dot(wx, wx))
                if denom > 1e-12:
                    b = float(np.dot(wx, wy) / denom)
                    p = r / max(denom, 1e-12)
                else:
                    b = 0.0  # diffuse prior fallback
                    p = 1e3
                b_prev = b
                beta[t] = b
                unc[t] = p
            continue
        p_pred = p + q
        denom = p_pred * xv * xv + r
        if denom <= 0.0:
            continue
        k = p_pred * xv / denom
        innov = yv - b * xv
        b_new = b + k * innov
        p = (1.0 - k * xv) * p_pred
        beta[t] = b_new
        change[t] = b_new - b_prev
        unc[t] = p
        b_prev = b_new
        b = b_new
    if out_stat == "beta":
        return beta
    if out_stat == "beta_change":
        return change
    return unc


_register("ts_kalman_beta",
          "动态市场 Beta 状态。通过原点回归 y=b*x(无截距); warmup 为 finite-pair(跨缺口累积前 5 个有限 y/x 对)。输入 y 为个股收益、x 为市场收益 (return-vs-return)。",
          ["y", "x", "q", "r"], "level",
          lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "beta")),
          input_units=_BETA_INPUT_UNITS, extra_tags=_BETA_THROUGH_ORIGIN_TAG)
_register("ts_kalman_beta_change",
          "动态 Beta 变化。通过原点回归 y=b*x(无截距)。输入 y 为个股收益、x 为市场收益 (return-vs-return)。",
          ["y", "x", "q", "r"], "level",
          lambda y, x, q=1e-3, r=1.0: _apply_two(y, x, lambda a, b: _kalman_beta(a, b, float(q), float(r), "beta_change")),
          input_units=_BETA_INPUT_UNITS, extra_tags=_BETA_THROUGH_ORIGIN_TAG)


def _apply_col(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = fn(xv[:, col])
    return frame_like(x, out)


def _apply_two(y: pd.DataFrame, x: pd.DataFrame, fn) -> pd.DataFrame:
    # P0-041: model inputs must be index/column aligned before the per-column
    # recursion, so a reordered ``x`` panel can never silently mispair stocks.
    y, x = aligned(y, x)
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = fn(yv[:, col], xv[:, col])
    return frame_like(y, out)


# M-008: Kalman is the single-authority stateful declaration.  Every Kalman
# canonical is a causal one-pass filter — stateful, checkpointable, and legal to
# time-shard ONLY with state handoff (see KALMAN_STATEFUL_CANONICALS +
# kalman_stateful_contract()).  Declared here (its own module) per the
# Round-11 #12 rule: never edit a central list.  The ModelOperatorContract
# stateful flags in model_contract.py mirror this.
for _kalman_canonical in sorted(KALMAN_STATEFUL_CANONICALS):
    try:
        from runtime.execution_contract import declare_stateful

        declare_stateful(
            _kalman_canonical,
            state_model="recursive",
            chunking="checkpoint",
            checkpoint_schema="kalman_v1",
            minimum_history=1,
        )
    except Exception:  # pragma: no cover - import/registry ordering guard
        pass

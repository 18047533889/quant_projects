# -*- coding: utf-8 -*-
"""Filter layer: hysteresis and turnover control operators (2026-08 P0).

This family provides three turnover-control primitives for signal filtering:

* ``state_adaptive_deadband`` — adaptive deadband using rolling volatility of
  changes (MAD or STD of delta) to automatically scale the threshold;
* ``state_rank_deadband`` — rank-space deadband (cross-sectional percentile
  hysteresis): only update when the rank percentile moves more than band_pct;
* ``state_quantile_hysteresis`` — dual-threshold entry/exit hysteresis in
  cross-sectional quantile space (enter at 90%, exit at 80%).

All three are stateful, checkpointable, and prefix-causal (PIT-safe).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.common.daily_panel import _aligned
from factor_engine.cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _cs_rank_pct(x: np.ndarray, group: np.ndarray | None) -> np.ndarray:
    """Cross-sectional rank percentile [0, 1], NaN-aware.

    FL-P0-014: Vectorized implementation O(T×N log N) instead of O(T×N²).
    Uses scipy.stats.rankdata for efficient ranking with proper tie handling.
    """
    rows, cols = x.shape
    out = np.full((rows, cols), np.nan, dtype=float)

    # FL-P0-014: Vectorized per-row ranking (much faster than nested loops)
    for r in range(rows):
        row_vals = x[r, :]

        if group is not None:
            g_row = group[r, :]
            unique_groups = np.unique(g_row[np.isfinite(g_row)])

            # Rank within each group
            for g in unique_groups:
                mask = (g_row == g) & np.isfinite(row_vals)
                if not np.any(mask):
                    continue

                group_vals = row_vals[mask]
                group_indices = np.where(mask)[0]

                if len(group_vals) == 0:
                    continue

                # Use scipy rankdata for O(N log N) ranking with average tie method
                try:
                    from scipy.stats import rankdata
                    ranks = rankdata(group_vals, method='average')
                    # Convert to percentile [0, 1]
                    percentiles = np.where(len(ranks) != 0, ranks / len(ranks), np.nan)
                    out[r, group_indices] = percentiles
                except ImportError:
                    # Fallback to O(N²) if scipy unavailable
                    for i, idx in enumerate(group_indices):
                        v = group_vals[i]
                        rank = np.sum(group_vals < v) + 0.5 * np.sum(group_vals == v)
                        out[r, idx] = np.where(len(group_vals) != 0, rank / len(group_vals), np.nan)
        else:
            # No grouping: rank entire row
            finite_mask = np.isfinite(row_vals)
            if not np.any(finite_mask):
                continue

            finite_vals = row_vals[finite_mask]
            finite_indices = np.where(finite_mask)[0]

            if len(finite_vals) == 0:
                continue

            # FL-P0-014: Vectorized ranking O(N log N)
            try:
                from scipy.stats import rankdata
                ranks = rankdata(finite_vals, method='average')
                # Convert to percentile [0, 1]
                percentiles = np.where(len(ranks) != 0, ranks / len(ranks), np.nan)
                out[r, finite_indices] = percentiles
            except ImportError:
                # Fallback to O(N²) if scipy unavailable
                for i, idx in enumerate(finite_indices):
                    v = finite_vals[i]
                    rank = np.sum(finite_vals < v) + 0.5 * np.sum(finite_vals == v)
                    out[r, idx] = np.where(len(finite_vals) != 0, rank / len(finite_vals), np.nan)

    return out


@register_operator(
    name="state_adaptive_deadband",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_adaptive_deadband",
    source="filter_hysteresis",
)
class StateAdaptiveDeadband(SeriesOperator):
    """Adaptive deadband using rolling volatility of changes.

    The band adapts to recent signal volatility:
        scale_t = MAD(delta(x)_{t-window:t-1}) * 1.4826  (or STD)
        band_t = band_mult * scale_t

    Only updates output when |x_t - y_{t-1}| > band_t; otherwise holds y_{t-1}.
    """

    metadata = OperatorMetadata(
        name="state_adaptive_deadband",
        category="signal_filter",
        description="自适应死区: band 随信号变化波动率动态调整, 减少换手。",
        param_names=["x", "band_mult", "scale_window", "scale_method"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:hysteresis",
            "signature:x,band_mult,scale_window,scale_method->series",
            "domain:signal_processing", "unit:level", "cost:2",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "band_mult": ParamSpec(
                dtype=float, min=0.0, default=1.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
            "scale_window": ParamSpec(
                dtype=int, min=2, default=20, searchable=False,
                param_role=ParamRole.NUMERICAL,
                history_semantics="max_rows",
            ),
            "scale_method": ParamSpec(
                dtype=str, default="mad_delta", choices=("mad_delta", "std_delta"),
                searchable=True, param_role=ParamRole.POLICY,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        band_mult: float = 1.0,
        scale_window: int = 20,
        scale_method: str = "mad_delta",
        **_: Any,
    ) -> pd.DataFrame:
        if scale_window < 2:
            raise ValueError("state_adaptive_deadband requires scale_window >= 2")
        if band_mult < 0.0:
            raise ValueError("state_adaptive_deadband requires band_mult >= 0")
        if scale_method not in ("mad_delta", "std_delta"):
            raise ValueError("scale_method must be 'mad_delta' or 'std_delta'")

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            y_prev = np.nan
            delta_history = []

            for row in range(rows):
                curr = xv[row, col]
                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    delta_history.clear()
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # FL-P0-001: Compute adaptive scale BEFORE adding current delta
                # The threshold for bar t must NOT include delta_t itself (strict causality)
                # Current implementation is CORRECT: delta_history only updated AFTER scale computation

                # Compute adaptive scale from PAST deltas only
                if len(delta_history) >= 2:
                    delta_arr = np.array(delta_history)
                    if scale_method == "mad_delta":
                        scale_t = 1.4826 * np.median(np.abs(delta_arr - np.median(delta_arr)))
                    else:  # std_delta
                        scale_t = np.std(delta_arr, ddof=1)

                    # FL-P0-043: zero scale policy - avoid band=0 from float noise
                    MIN_BAND = 1e-12  # Numerical floor to prevent spurious updates from float precision
                    band_t = max(band_mult * scale_t, MIN_BAND)
                else:
                    band_t = 0.0

                # Deadband logic
                diff = curr - y_prev
                if np.abs(diff) > band_t + _EPS:
                    y_prev = curr
                out[row, col] = y_prev

                # FL-P0-001: Update delta history AFTER computing scale_t and making decision
                # This ensures delta_t does not participate in its own threshold
                delta = curr - xv[row - 1, col] if row > 0 and np.isfinite(xv[row - 1, col]) else np.nan
                if np.isfinite(delta):
                    delta_history.append(delta)
                    if len(delta_history) > scale_window:
                        delta_history.pop(0)

        return frame_like(x, out)


@register_operator(
    name="state_rank_deadband",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_rank_deadband",
    source="filter_hysteresis",
)
class StateRankDeadband(SeriesOperator):
    """Rank-space deadband: only update when rank percentile moves > band_pct.

    FL-P0-010: Output is the HELD RANK PERCENTILE (always in [0,1]),
    not the held raw value.

    Operates in cross-sectional rank space: computes rank_pct(x_t) and only
    updates the output when |rank_t - rank_prev| > band_pct.
    """

    metadata = OperatorMetadata(
        name="state_rank_deadband",
        category="signal_filter",
        description="横截面排名空间死区: 只有排名百分位变化超过 band_pct 才更新。",
        param_names=["x", "band_pct", "group"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:hysteresis",
            "signature:x,band_pct,group->series",
            "domain:signal_processing", "unit:level", "cost:3",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "band_pct": ParamSpec(
                dtype=float, min=0.0, max=1.0, default=0.05, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
            "group": ParamSpec(dtype=None, searchable=False, param_role=ParamRole.POLICY),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        band_pct: float = 0.05,
        group: pd.DataFrame | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        if not (0.0 <= band_pct <= 1.0):
            raise ValueError("state_rank_deadband requires 0 <= band_pct <= 1")

        if group is not None:
            x, group = _aligned(x, group)
            gv = group.to_numpy(dtype=float)
        else:
            gv = None

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        # Compute all rank percentiles first
        rank_pct = _cs_rank_pct(xv, gv)

        for col in range(cols):
            rank_held = np.nan

            for row in range(rows):
                curr = xv[row, col]
                rank_t = rank_pct[row, col]

                if not np.isfinite(curr) or not np.isfinite(rank_t):
                    out[row, col] = np.nan
                    rank_held = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(rank_held):
                    rank_held = rank_t
                    out[row, col] = rank_t  # FL-P0-010: output rank percentile
                    continue

                # Check rank change
                if np.abs(rank_t - rank_held) > band_pct + _EPS:
                    rank_held = rank_t

                out[row, col] = rank_held  # FL-P0-010: output held rank percentile

        return frame_like(x, out)


@register_operator(
    name="state_quantile_hysteresis",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_quantile_hysteresis",
    source="filter_hysteresis",
)
class StateQuantileHysteresis(SeriesOperator):
    """Dual-threshold entry/exit hysteresis in cross-sectional quantile space.

    State machine with different entry and exit thresholds:
    - Enter (OUT->IN) when rank_pct >= enter_quantile
    - Exit (IN->OUT) when rank_pct < exit_quantile
    Returns 1 when IN, 0 when OUT.
    """

    metadata = OperatorMetadata(
        name="state_quantile_hysteresis",
        category="signal_filter",
        description="横截面分位数双阈值迟滞: 进入和退出采用不同阈值, 减少状态抖动。",
        param_names=["x", "enter_quantile", "exit_quantile", "group"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:hysteresis",
            "signature:x,enter_quantile,exit_quantile,group->series",
            "domain:signal_processing", "unit:state", "cost:3",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "enter_quantile": ParamSpec(
                dtype=float, min=0.0, max=1.0, default=0.9, searchable=True,
                param_role=ParamRole.STATE_THRESHOLD,
            ),
            "exit_quantile": ParamSpec(
                dtype=float, min=0.0, max=1.0, default=0.8, searchable=True,
                param_role=ParamRole.STATE_THRESHOLD,
            ),
            "group": ParamSpec(dtype=None, searchable=False, param_role=ParamRole.POLICY),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        enter_quantile: float = 0.9,
        exit_quantile: float = 0.8,
        group: pd.DataFrame | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        if not (0.0 <= enter_quantile <= 1.0):
            raise ValueError("state_quantile_hysteresis requires 0 <= enter_quantile <= 1")
        if not (0.0 <= exit_quantile <= 1.0):
            raise ValueError("state_quantile_hysteresis requires 0 <= exit_quantile <= 1")
        if exit_quantile > enter_quantile:
            raise ValueError("state_quantile_hysteresis requires exit_quantile <= enter_quantile")

        if group is not None:
            x, group = _aligned(x, group)
            gv = group.to_numpy(dtype=float)
        else:
            gv = None

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        # Compute all rank percentiles
        rank_pct = _cs_rank_pct(xv, gv)

        for col in range(cols):
            state = 0  # OUT

            for row in range(rows):
                q_t = rank_pct[row, col]

                if not np.isfinite(q_t):
                    out[row, col] = np.nan
                    state = 0
                    continue

                # State machine
                if state == 0:  # OUT
                    if q_t >= enter_quantile - _EPS:
                        state = 1
                else:  # IN
                    if q_t < exit_quantile - _EPS:
                        state = 0

                out[row, col] = float(state)

        return frame_like(x, out)


@register_operator(
    name="state_adaptive_slew_limit",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_adaptive_slew_limit",
    source="filter_hysteresis",
)
class StateAdaptiveSlewLimit(SeriesOperator):
    """自适应Slew Rate Limiter：限制信号单步变化幅度为delta的自适应尺度倍数。

    数学（参考文档 §9）：
        # 自适应scale估计（严格因果：使用过去窗口）
        if scale_method == "mad_delta":
            delta_past = x_{t-window:t-1} - x_{t-window-1:t-2}
            scale_t = 1.4826 * median(|delta_past|)
        elif scale_method == "std_delta":
            delta_past = x_{t-window:t-1} - x_{t-window-1:t-2}
            scale_t = std(delta_past)

        # Slew limiting
        limit_t = slew_mult * scale_t
        delta_t = x_t - y_{t-1}
        delta_clamped = clip(delta_t, -limit_t, limit_t)
        y_t = y_{t-1} + delta_clamped

    认证参数：
        slew_mult: SCALAR_FLOAT，可搜索（控制slew强度）
        scale_window: NUMERICAL_POLICY，不可搜索
        scale_method: SCALAR_STR，可选 "mad_delta" 或 "std_delta"

    Parameters:
        x: 输入信号
        slew_mult: slew限制倍数（默认1.0）
        scale_window: 估计scale的回溯窗口（默认20）
        scale_method: scale估计方法，"mad_delta" 或 "std_delta"（默认"mad_delta"）

    Returns:
        slew限制后的信号（单步变化被自适应scale约束）
    """

    metadata = OperatorMetadata(
        name="state_adaptive_slew_limit",
        category="signal_filter",
        description="自适应Slew Rate Limiter：限制单步变化幅度为delta自适应尺度倍数",
        param_names=["x", "slew_mult", "scale_window", "scale_method"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:rate_limit",
            "signature:x,slew_mult,scale_window,scale_method->series",
            "domain:signal_processing", "unit:dimensionless", "cost:2",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "slew_mult": ParamSpec(
                dtype=float, min=0.0, default=1.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
            "scale_window": ParamSpec(
                dtype=int, min=2, default=20, searchable=False,
                param_role=ParamRole.NUMERICAL,
                history_semantics="max_rows",
            ),
            "scale_method": ParamSpec(
                dtype=str, default="mad_delta", choices=("mad_delta", "std_delta"),
                searchable=True, param_role=ParamRole.POLICY,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        slew_mult: float = 1.0,
        scale_window: int = 20,
        scale_method: str = "mad_delta",
        **_: Any,
    ) -> pd.DataFrame:
        if scale_window < 2:
            raise ValueError("state_adaptive_slew_limit requires scale_window >= 2")
        if slew_mult <= 0:
            raise ValueError("state_adaptive_slew_limit requires slew_mult > 0")
        if scale_method not in ("mad_delta", "std_delta"):
            raise ValueError("scale_method must be 'mad_delta' or 'std_delta'")

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        MAD_SCALE = 1.4826
        MIN_SCALE = 1e-10

        for col in range(cols):
            prev_output = np.nan

            for row in range(rows):
                curr = xv[row, col]

                # First row: initialize
                if row == 0:
                    if np.isfinite(curr):
                        out[row, col] = curr
                        prev_output = curr
                    else:
                        out[row, col] = np.nan
                        prev_output = np.nan
                    continue

                # Current invalid: propagate NaN
                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    continue

                # Previous output invalid: reinitialize
                if not np.isfinite(prev_output):
                    out[row, col] = curr
                    prev_output = curr
                    continue

                # Warmup: need scale_window+1 points for scale_window deltas
                if row < scale_window + 1:
                    out[row, col] = curr
                    prev_output = curr
                    continue

                # Compute past deltas (causal)
                past_x = xv[row - scale_window : row, col]
                past_x_lagged = xv[row - scale_window - 1 : row - 1, col]
                finite_mask = np.isfinite(past_x) & np.isfinite(past_x_lagged)

                if not np.any(finite_mask):
                    out[row, col] = curr
                    prev_output = curr
                    continue

                delta_past = past_x[finite_mask] - past_x_lagged[finite_mask]

                # Estimate scale
                if scale_method == "mad_delta":
                    mad_raw = float(np.median(np.abs(delta_past - np.median(delta_past))))
                    scale_t = max(MAD_SCALE * mad_raw, MIN_SCALE)
                else:  # std_delta
                    if len(delta_past) < 2:
                        out[row, col] = curr
                        prev_output = curr
                        continue
                    scale_t = max(float(np.std(delta_past, ddof=1)), MIN_SCALE)

                # Slew limiting
                limit_t = slew_mult * scale_t
                delta_t = curr - prev_output
                delta_clamped = float(np.clip(delta_t, -limit_t, limit_t))

                new_output = prev_output + delta_clamped
                out[row, col] = new_output
                prev_output = new_output

        return frame_like(x, out)


@register_operator(
    name="state_cost_aware_deadband",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_cost_aware_deadband",
    source="filter_hysteresis",
)
class StateCostAwareDeadband(SeriesOperator):
    """Cost-aware deadband: band scales with trading cost proxy.

    The deadband threshold adapts to expected trading costs:
        band_t = cost_mult * cost_proxy_t

    Only updates output when |x_t - y_{t-1}| > band_t; otherwise holds y_{t-1}.
    This directly ties turnover control to real trading costs.
    """

    metadata = OperatorMetadata(
        name="state_cost_aware_deadband",
        category="signal_filter",
        description="成本感知死区: band 与交易成本代理成比例, 直接面向真实交易成本。",
        param_names=["x", "cost_proxy", "cost_mult"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:rate_limit",
            "signature:x,cost_proxy,cost_mult->series",
            "domain:signal_processing", "unit:level", "cost:2",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "cost_proxy": ParamSpec(
                dtype=None, searchable=False,
                param_role=ParamRole.POLICY,
            ),
            "cost_mult": ParamSpec(
                dtype=float, min=0.0, default=2.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        cost_proxy: pd.DataFrame,
        cost_mult: float = 2.0,
        **_: Any,
    ) -> pd.DataFrame:
        if cost_mult < 0.0:
            raise ValueError("state_cost_aware_deadband requires cost_mult >= 0")

        x, cost_proxy = _aligned(x, cost_proxy)
        xv = x.to_numpy(dtype=float)
        cv = cost_proxy.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            y_prev = np.nan

            for row in range(rows):
                curr = xv[row, col]
                cost_t = cv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # Compute cost-aware band (FL-P0-009: invalid cost → fail-closed)
                if np.isfinite(cost_t) and cost_t > 0:
                    band_t = cost_mult * cost_t
                else:
                    # Invalid/missing cost: fail-closed (output NaN, don't update state)
                    out[row, col] = np.nan
                    continue

                # Deadband logic
                diff = curr - y_prev
                if np.abs(diff) > band_t + _EPS:
                    y_prev = curr
                out[row, col] = y_prev

        return frame_like(x, out)


@register_operator(
    name="state_cost_aware_slew",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_cost_aware_slew",
    source="filter_hysteresis",
)
class StateCostAwareSlew(SeriesOperator):
    """Cost-aware slew rate limiter: change rate inversely proportional to cost.

    FL-P0-008: Dimensionally valid formula with bounded limit:
        cost_norm_t = cost_t / typical_cost  (assume typical_cost ≈ 1 for normalized input)
        limit_t = base_limit / (1 + k * cost_norm_t)

    Where:
        - base_limit = slew_mult (max allowed change when cost=0)
        - k = 1.0 (sensitivity to cost)
        - cost=0 → limit = base_limit (full freedom)
        - cost→∞ → limit → 0 (tight constraint)

    FL-P0-009: Invalid cost (NaN/negative/missing) → fail-closed (output NaN).

    Slew limiting:
        delta_t = x_t - y_{t-1}
        delta_clamped = clip(delta_t, -limit_t, limit_t)
        y_t = y_{t-1} + delta_clamped
    """

    metadata = OperatorMetadata(
        name="state_cost_aware_slew",
        category="signal_filter",
        description="成本感知slew: 流动性差时允许变化小, 流动性好时允许快速更新。",
        param_names=["x", "cost_proxy", "slew_mult"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:rate_limit",
            "signature:x,cost_proxy,slew_mult->series",
            "domain:signal_processing", "unit:level", "cost:2",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "cost_proxy": ParamSpec(
                dtype=None, searchable=False,
                param_role=ParamRole.POLICY,
            ),
            "slew_mult": ParamSpec(
                dtype=float, min=0.0, default=3.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        cost_proxy: pd.DataFrame,
        slew_mult: float = 3.0,
        **_: Any,
    ) -> pd.DataFrame:
        if slew_mult <= 0.0:
            raise ValueError("state_cost_aware_slew requires slew_mult > 0")

        x, cost_proxy = _aligned(x, cost_proxy)
        xv = x.to_numpy(dtype=float)
        cv = cost_proxy.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        EPSILON = 1e-10

        for col in range(cols):
            y_prev = np.nan

            for row in range(rows):
                curr = xv[row, col]
                cost_t = cv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # FL-P0-008: Compute cost-aware slew limit (bounded formula)
                # FL-P0-009: Invalid cost → fail-closed
                if np.isfinite(cost_t) and cost_t >= 0:
                    # limit_t = base_limit / (1 + k * cost_norm)
                    # Assume cost is already normalized (typical_cost ≈ 1)
                    cost_norm = cost_t
                    k = 1.0
                    limit_t = np.where((1.0 + k * cost_norm) != 0, slew_mult / (1.0 + k * cost_norm), np.nan)
                else:
                    # Invalid/missing cost: fail-closed (output NaN, don't update state)
                    out[row, col] = np.nan
                    continue

                # Slew limiting
                delta_t = curr - y_prev
                delta_clamped = float(np.clip(delta_t, -limit_t, limit_t))
                y_prev = y_prev + delta_clamped
                out[row, col] = y_prev

        return frame_like(x, out)


@register_operator(
    name="state_confidence_weighted_ema",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_confidence_weighted_ema",
    source="filter_hysteresis",
)
class StateConfidenceWeightedEma(SeriesOperator):
    """置信度调节 EMA：alpha 随模型置信度动态调整。

    高置信度时快速更新，低置信度时更多平滑。适合模型输出层。

    数学（参考文档 §13.J1）：
        alpha_t = alpha_min + confidence_t * (alpha_max - alpha_min)
        y_t = y_{t-1} + alpha_t * (x_t - y_{t-1})

    认证参数：
        alpha_min: [0.02, 0.05, 0.1]
        alpha_max: [0.3, 0.5, 0.7]

    Parameters:
        x: 输入信号
        confidence: 置信度序列 [0, 1]
        alpha_min: 最小 alpha（低置信度时，默认0.05）
        alpha_max: 最大 alpha（高置信度时，默认0.5）

    Returns:
        置信度调节后的平滑信号
    """

    metadata = OperatorMetadata(
        name="state_confidence_weighted_ema",
        category="signal_filter",
        description="置信度调节 EMA：alpha 随模型置信度动态调整，高置信快速更新。",
        param_names=["x", "confidence", "alpha_min", "alpha_max"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:adaptive_low_pass",
            "signature:x,confidence,alpha_min,alpha_max->series",
            "domain:signal_processing", "unit:dimensionless", "cost:2",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "confidence": ParamSpec(dtype=None, searchable=False, param_role=ParamRole.POLICY),
            "alpha_min": ParamSpec(
                dtype=float, min=0.0, max=1.0, default=0.05, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
            "alpha_max": ParamSpec(
                dtype=float, min=0.0, max=1.0, default=0.5, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        confidence: pd.DataFrame,
        alpha_min: float = 0.05,
        alpha_max: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        if not (0.0 <= alpha_min <= 1.0):
            raise ValueError("state_confidence_weighted_ema requires 0 <= alpha_min <= 1")
        if not (0.0 <= alpha_max <= 1.0):
            raise ValueError("state_confidence_weighted_ema requires 0 <= alpha_max <= 1")
        if alpha_min > alpha_max:
            raise ValueError("state_confidence_weighted_ema requires alpha_min <= alpha_max")

        x, confidence = _aligned(x, confidence)
        xv = x.to_numpy(dtype=float)
        cv = confidence.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            y_prev = np.nan

            for row in range(rows):
                curr = xv[row, col]
                conf = cv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # FL-P0-033: confidence must be in [0, 1]; out-of-range is fail-closed
                if not np.isfinite(conf):
                    alpha_t = alpha_min
                elif conf < 0.0 or conf > 1.0:
                    # FL-P0-033: Out-of-range confidence → fail-closed (output NaN)
                    # Don't silently clamp; reject invalid confidence values
                    out[row, col] = np.nan
                    continue
                else:
                    alpha_t = alpha_min + conf * (alpha_max - alpha_min)

                # EMA update
                y_prev = y_prev + alpha_t * (curr - y_prev)
                out[row, col] = y_prev

        return frame_like(x, out)


@register_operator(
    name="state_uncertainty_deadband",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_uncertainty_deadband",
    source="filter_hysteresis",
)
class StateUncertaintyDeadband(SeriesOperator):
    """不确定度阈值滤波器：变化未超过预测不确定度时不更新。

    适合模型集成输出或带不确定度估计的预测器。

    数学（参考文档 §13.J2）：
        if |x_t - y_{t-1}| > k_sigma * uncertainty_t:
            y_t = x_t
        else:
            y_t = y_{t-1}

    认证参数：
        k_sigma: [1.0, 1.5, 2.0]

    Parameters:
        x: 输入信号
        uncertainty: 不确定度序列（标准差尺度，> 0）
        k_sigma: 阈值倍数（默认1.0）

    Returns:
        不确定度阈值滤波后的信号
    """

    metadata = OperatorMetadata(
        name="state_uncertainty_deadband",
        category="signal_filter",
        description="不确定度阈值滤波器：变化未超过预测不确定度时不更新。",
        param_names=["x", "uncertainty", "k_sigma"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:hysteresis",
            "signature:x,uncertainty,k_sigma->series",
            "domain:signal_processing", "unit:dimensionless", "cost:2",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "uncertainty": ParamSpec(dtype=None, searchable=False, param_role=ParamRole.POLICY),
            "k_sigma": ParamSpec(
                dtype=float, min=0.0, default=1.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        uncertainty: pd.DataFrame,
        k_sigma: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        if k_sigma < 0.0:
            raise ValueError("state_uncertainty_deadband requires k_sigma >= 0")

        x, uncertainty = _aligned(x, uncertainty)
        xv = x.to_numpy(dtype=float)
        uv = uncertainty.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            y_prev = np.nan

            for row in range(rows):
                curr = xv[row, col]
                unc = uv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # FL-P0-034: Uncertainty invalid or non-positive → fail-closed (output NaN)
                # Don't treat uncertainty<=0 as "no constraint" (unconditional update)
                # This would make the filter vulnerable to invalid uncertainty estimates
                if not np.isfinite(unc) or unc <= 0.0:
                    out[row, col] = np.nan
                    continue

                # Check threshold
                threshold = k_sigma * unc
                if np.abs(curr - y_prev) > threshold + _EPS:
                    y_prev = curr

                out[row, col] = y_prev

        return frame_like(x, out)


@register_operator(
    name="state_l1_turnover_prox",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_l1_turnover_prox",
    source="filter_hysteresis",
)
class StateL1TurnoverProx(SeriesOperator):
    """L1 turnover proximal operator (soft-threshold / no-trade band).

    Solves the optimization problem:
        y_t = argmin_y  0.5*(y - x_t)^2 + lambda_turnover * |y - y_{t-1}|

    Analytical solution (soft-threshold):
        delta = x_t - y_{t-1}
        if |delta| <= lambda_turnover:
            y_t = y_{t-1}  (no trade)
        elif delta > lambda_turnover:
            y_t = y_{t-1} + (delta - lambda_turnover)
        else:  # delta < -lambda_turnover
            y_t = y_{t-1} + (delta + lambda_turnover)

    This creates a no-trade band around the previous state, where small changes
    are suppressed to reduce turnover. The optimization formulation makes the
    tradeoff between tracking the signal and turnover explicit.
    """

    metadata = OperatorMetadata(
        name="state_l1_turnover_prox",
        category="signal_filter",
        description="L1 换手近端算子: 围绕旧状态的 soft-threshold/no-trade band, 明确建模信号跟踪与换手的权衡。",
        param_names=["x", "lambda_turnover"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:rate_limit",
            "signature:x,lambda_turnover->series",
            "domain:signal_processing", "unit:level", "cost:1",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "lambda_turnover": ParamSpec(
                dtype=float, min=0.0, default=1.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lambda_turnover: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        if lambda_turnover < 0.0:
            raise ValueError("state_l1_turnover_prox requires lambda_turnover >= 0")

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            y_prev = np.nan

            for row in range(rows):
                curr = xv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # Soft-threshold (L1 proximal operator)
                delta = curr - y_prev
                if np.abs(delta) <= lambda_turnover + _EPS:
                    # No trade: stay at previous state
                    y_new = y_prev
                elif delta > lambda_turnover:
                    # Positive move exceeds threshold
                    y_new = y_prev + (delta - lambda_turnover)
                else:  # delta < -lambda_turnover
                    # Negative move exceeds threshold
                    y_new = y_prev + (delta + lambda_turnover)

                out[row, col] = y_new
                y_prev = y_new

        return frame_like(x, out)


@register_operator(
    name="state_l2_partial_adjustment",
    category="signal_filter",
    business_category="signal_filter",
    canonical="state_l2_partial_adjustment",
    source="filter_hysteresis",
)
class StateL2PartialAdjustment(SeriesOperator):
    """L2 partial adjustment filter (smooth turnover control).

    Equivalent to partial adjustment model:
        y_t = (x_t + lambda_smooth * y_{t-1}) / (1 + lambda_smooth)

    Or equivalently:
        y_t = y_{t-1} + (1/(1+lambda_smooth)) * (x_t - y_{t-1})

    Higher lambda_smooth means slower adjustment (more smoothing, less turnover).
    This provides smoother turnover control than absolute slew limits and is more
    aligned with portfolio optimization semantics.
    """

    metadata = OperatorMetadata(
        name="state_l2_partial_adjustment",
        category="signal_filter",
        description="L2 部分调整滤波: 平滑换手控制, 比绝对 slew 更符合组合优化语义。",
        param_names=["x", "lambda_smooth"],
        return_type="series",
        tags=[
            "signal_filter", "daily", "pit_safe", "causal", "stateful",
            "checkpointable", "typed_v2", "filter_role:rate_limit",
            "signature:x,lambda_smooth->series",
            "domain:signal_processing", "unit:level", "cost:1",
        ],
        param_specs={
            "x": ParamSpec(dtype=None, searchable=False),
            "lambda_smooth": ParamSpec(
                dtype=float, min=0.0, default=1.0, searchable=True,
                param_role=ParamRole.ECONOMIC,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lambda_smooth: float = 1.0,
        **_: Any,
    ) -> pd.DataFrame:
        if lambda_smooth < 0.0:
            raise ValueError("state_l2_partial_adjustment requires lambda_smooth >= 0")

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            y_prev = np.nan

            for row in range(rows):
                curr = xv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    y_prev = np.nan
                    continue

                # First valid observation initializes
                if not np.isfinite(y_prev):
                    y_prev = curr
                    out[row, col] = curr
                    continue

                # Partial adjustment: y_t = (x_t + lambda * y_{t-1}) / (1 + lambda)
                y_new = np.where((1.0 + lambda_smooth) != 0, (curr + lambda_smooth * y_prev) / (1.0 + lambda_smooth), np.nan)

                out[row, col] = y_new
                y_prev = y_new

        return frame_like(x, out)




def _register_surface() -> None:
    """注册到 extended surface 并添加 Polars 后端支持。"""
    import factor_engine.cleaned_operators.operator_surface as _surface
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    _CANONICALS = {
        "state_adaptive_deadband",
        "state_rank_deadband",
        "state_quantile_hysteresis",
        "state_adaptive_slew_limit",
        "state_cost_aware_deadband",
        "state_cost_aware_slew",
        "state_confidence_weighted_ema",
        "state_uncertainty_deadband",
        "state_l1_turnover_prox",
        "state_l2_partial_adjustment",
    }

    _surface.extend_extended_only(_CANONICALS)

    # Polars 后端：委托 pandas reference
    for _canon in _CANONICALS:
        register_polars_bridge(_canon)


_register_surface()

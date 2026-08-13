# -*- coding: utf-8 -*-
"""Downside risk and lag-reaction operators.

Includes downside/upside deviation, drawdown duration, time-under-water,
best lag correlation and a price-delay proxy.  All operators are causal
daily-panel transforms.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _check_int(value: Any, name: str, minimum: int) -> int:
    """R11 #159: strict integer contract — reject bools and non-integer floats so
    ``5.9`` never silently truncates to ``5`` and compiles to the same factor as
    ``5.0`` (a false search space)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    fv = float(value)
    if not np.isfinite(fv) or fv != float(int(fv)):
        raise ValueError(f"{name} must be an integer")
    iv = int(fv)
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return iv


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    output_unit: str | None = None,
    input_units: dict[str, str] | None = None,
    compatible_units: dict[str, tuple[str, ...]] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_risk",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_risk", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
        output_unit=output_unit,
        input_units=dict(input_units or {}),
        compatible_units=dict(compatible_units or {}),
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _rolling_apply_2d(values: np.ndarray, window: int, fn: Any, min_periods: int = 1) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = values[start : row + 1, col]
            out[row, col] = fn(chunk)
    return out


@register_operator(
    name="ts_downside_deviation",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_downside_deviation",
    source="downside_risk",
    status="experimental",
)
class TsDownsideDeviation(SeriesOperator):
    """下行偏离：sqrt(mean(min(x - target, 0)^2))。

    x 是收益/带符号序列，target 是相同单位的最小可接受收益 MAR（minimal
    acceptable return）。对收益相对目标收益的偏离才有意义；对原始正价格
    target=0 会输出约 0（死因子）——现在由类型契约在 typed 层拒绝，而不是
    静默产生无意义信号。单位继承 x (R5 P1-36(c))：输出与 x 同单位，不是
    无量纲比值。
    """

    metadata = _metadata(
        "ts_downside_deviation",
        "下行偏离 sqrt(mean(min(x-target,0)^2))：x 为收益/MAR 目标（同单位），输出继承 x 单位。",
        ["x", "window", "target", "min_periods"],
        domain="price_volume",
        unit="same_as:x",
        output_unit="same_as:x",
        # R11 #158: the deviation is dimensionally unit(x); ``target`` is a
        # MAR threshold scalar in the SAME unit as x — never the output
        # reference.  x is a RETURN / signed series measured against a target
        # return, NOT a raw price: ``price`` with target=0 would be ~0 for
        # positive prices — a dead factor.  The typed layer reads these
        # declarations, so a price-only input is a contract violation
        # (rejected) instead of a silent ~0 factor.
        input_units={"x": "return_or_same_unit_numeric", "target": "same_unit_as:x"},
        compatible_units={"x": ("return", "same_unit_as:x"), "target": ("same_unit_as:x",)},
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, target: float = 0.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        # R11 #159: strict integer + domain validation — ``2 <= min_periods <= window``.
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        tgt = float(target)
        if not np.isfinite(tgt):
            raise ValueError("target must be finite")

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            below = np.minimum(valid - tgt, 0.0)
            return float(np.sqrt(np.mean(below * below)))

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


@register_operator(
    name="ts_upside_deviation",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_upside_deviation",
    source="downside_risk",
    status="experimental",
)
class TsUpsideDeviation(SeriesOperator):
    """上行偏离：sqrt(mean(max(x - target, 0)^2))。

    x 是收益/带符号序列，target 是相同单位的最小可接受收益 MAR（minimal
    acceptable return）。对收益相对目标收益的偏离才有意义；对原始正价格
    target=0 会输出约 0（死因子）——现在由类型契约在 typed 层拒绝，而不是
    静默产生无意义信号。单位继承 x (R5 P1-36(c))：输出与 x 同单位，不是
    无量纲比值。
    """

    metadata = _metadata(
        "ts_upside_deviation",
        "上行偏离 sqrt(mean(max(x-target,0)^2))：x 为收益/MAR 目标（同单位），输出继承 x 单位。",
        ["x", "window", "target", "min_periods"],
        domain="price_volume",
        unit="same_as:x",
        output_unit="same_as:x",
        # R11 #158: the deviation is dimensionally unit(x); ``target`` is a
        # MAR threshold scalar in the SAME unit as x — never the output
        # reference.  x is a RETURN / signed series measured against a target
        # return, NOT a raw price: ``price`` with target=0 would be ~0 for
        # positive prices — a dead factor.  The typed layer reads these
        # declarations, so a price-only input is a contract violation
        # (rejected) instead of a silent ~0 factor.
        input_units={"x": "return_or_same_unit_numeric", "target": "same_unit_as:x"},
        compatible_units={"x": ("return", "same_unit_as:x"), "target": ("same_unit_as:x",)},
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, target: float = 0.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        # R11 #159: strict integer + domain validation — ``2 <= min_periods <= window``.
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        tgt = float(target)
        if not np.isfinite(tgt):
            raise ValueError("target must be finite")

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            above = np.maximum(valid - tgt, 0.0)
            return float(np.sqrt(np.mean(above * above)))

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


@register_operator(
    name="ts_current_drawdown_duration",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_current_drawdown_duration",
    source="downside_risk",
    status="experimental",
)
class TsCurrentDrawdownDuration(SeriesOperator):
    """当前连续处于回撤（低于窗口运行最高价）的交易行数。"""

    metadata = _metadata(
        "ts_current_drawdown_duration",
        "当前连续低于窗口运行最高价的行数。",
        ["x", "window"],
        domain="price_volume",
        unit="count",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                chunk = xv[start : row + 1, col]
                valid_mask = np.isfinite(chunk)
                # R5 P1-36(a): the *current* observation is missing — the "current"
                # drawdown duration is unknowable.  Fail-close to NaN instead of
                # silently reporting the stale duration from the older peers.
                if not valid_mask.any() or not valid_mask[-1]:
                    continue
                # P1-07: the running peak must NOT carry across a missing row — a
                # value after a NaN gap is compared only against the post-gap
                # segment, otherwise a disconnected earlier peak makes a fresh
                # value look like it is still in drawdown.  Reset the running peak
                # at every gap so the next valid observation starts a fresh
                # reference (drawdown duration restarts from the current level).
                running_peak = np.full(len(chunk), np.nan)
                peak = -np.inf
                for k in range(len(chunk)):
                    if not valid_mask[k]:
                        peak = -np.inf
                        continue
                    if chunk[k] > peak:
                        peak = chunk[k]
                    running_peak[k] = peak
                streak = 0
                for back in range(len(chunk) - 1, -1, -1):
                    if not valid_mask[back]:
                        # NaN is a hard episode boundary: the streak must never
                        # re-link across the gap.  Stop counting instead of
                        # resetting-and-scanning so bars on the far side of a NaN
                        # never count toward the trailing episode (P0).
                        break
                    if chunk[back] < running_peak[back]:
                        streak += 1
                    else:
                        break
                out[row, col] = float(streak)
        return _frame_like(x, out)


@register_operator(
    name="ts_time_under_water",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_time_under_water",
    source="downside_risk",
    status="experimental",
)
class TsTimeUnderWater(SeriesOperator):
    """窗口内价格低于此前运行最高价的日期比例。"""

    metadata = _metadata(
        "ts_time_under_water",
        "窗口内低于此前运行最高价的比例。",
        ["x", "window"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                chunk = xv[start : row + 1, col]
                valid = chunk[np.isfinite(chunk)]
                # R5 P1-36(b): a missing *current* observation must fail-close to
                # NaN rather than emitting the historical under-water ratio.
                if valid.size == 0 or not np.isfinite(chunk[-1]):
                    continue
                # R11 #160: the running peak must NOT carry across a missing row —
                # exactly the ``ts_current_drawdown_duration`` gap policy.  The old
                # ``np.maximum.accumulate(where(NaN, -inf, x))`` kept the pre-gap
                # peak (``-inf`` never lowers a running max), so a value after a
                # NaN gap was still measured against a disconnected earlier peak.
                # Reset the running peak at every gap so the next valid observation
                # starts a fresh reference.
                running_peak = np.full(len(chunk), np.nan)
                peak = -np.inf
                for k in range(len(chunk)):
                    if not np.isfinite(chunk[k]):
                        peak = -np.inf
                        continue
                    if chunk[k] > peak:
                        peak = chunk[k]
                    running_peak[k] = peak
                under = np.sum((chunk < running_peak) & np.isfinite(chunk))
                out[row, col] = under / valid.size
        return _frame_like(x, out)


def _best_lag_corr_raw(x: np.ndarray, y: np.ndarray, row: int, window: int, max_lag: int) -> float:
    """Clean primitive: max over lags 0..max_lag of |Pearson corr(y[t], x[t-lag])|.

    No multiple-lag selection correction (R11 round-2): the Fisher-z heuristic
    that subtracted ~sqrt(2 ln(2K))/sqrt(n-3) treated the K lag-correlations as
    independent tests, which they are not (highly correlated lags).  The clean
    math primitive is just the peak magnitude.

    NaN means "no lag was computable"; a *finite* 0.0 is a legitimate result
    (all valid-lag correlations were exactly zero) and must survive as 0.0, not
    be coerced to NaN — P0-30.  The argmax lag is tracked internally; the peak
    magnitude is the operator output (the lag is exposed by
    ``ts_lag_of_peak_corr``).
    """
    best = -1.0
    best_lag = -1
    for lag in range(0, max_lag + 1):
        end = row + 1 - lag
        start = max(0, end - window)
        if end - start < 2:
            continue
        xs = x[start:end]
        ys = y[start + lag : row + 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 2:
            continue
        if np.std(xs[valid]) > 0 and np.std(ys[valid]) > 0:
            value = abs(float(np.corrcoef(xs[valid], ys[valid])[0, 1]))
            if value > best:
                best = value
                best_lag = lag
    if best_lag < 0:
        return np.nan
    return float(best)


# Backward-compatible helper name: the historical ``_best_lag_corr`` now carries
# the RAW (uncorrected) semantics — the multiple-lag selection correction was
# split out of the canonical into ``ts_best_lag_corr_excess``.
_best_lag_corr = _best_lag_corr_raw


def _circular_block_permute(arr: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic circular block permutation (permutation null).

    Rotates ``arr`` by a random offset, splits the rotated sequence into
    contiguous blocks of size ``block`` (last block partial) and shuffles the
    block order.  The result is a genuine permutation of ``arr`` (multiset
    preserved) whose short-range autocorrelation is largely retained — a block-
    permutation null for the source series.
    """
    n = len(arr)
    if n <= 1:
        return arr.copy()
    b = max(1, min(int(block), n))
    off = int(rng.integers(0, n))
    rotated = np.concatenate([arr[off:], arr[:off]])
    blocks = [rotated[i * b : (i + 1) * b] for i in range(int(np.ceil(n / b)))]
    order = list(range(len(blocks)))
    rng.shuffle(order)
    return np.concatenate([blocks[i] for i in order])[:n]


def _best_lag_corr_excess(
    x: np.ndarray,
    y: np.ndarray,
    row: int,
    window: int,
    max_lag: int,
    *,
    n_surrogates: int = 20,
    seed: int = 42,
    block_frac: float = 0.1,
) -> float:
    """max_real - E[max_surrogate] under a circular-block-permutation null.

    ``x`` is the leading/source series, ``y`` the trailing/target series.  The
    surrogate source is the longest trailing window of ``x`` any lag touches
    (``max_lag + window`` bars); each surrogate is a deterministic circular
    block permutation of it (``block = ceil(0.1 * n)``, ``n_surrogates=20``,
    per-row ``seed``).  Independent x,y -> excess ~ 0; a true lag dependence ->
    the real peak survives the permutation while surrogate peaks do not ->
    excess > 0.
    """
    real = _best_lag_corr_raw(x, y, row, window, max_lag)
    if not np.isfinite(real):
        return np.nan
    src_lo = max(0, row + 1 - max_lag - window)
    x_source = x[src_lo : row + 1]
    n = len(x_source)
    block = max(1, int(math.ceil(block_frac * n)))
    rng = np.random.default_rng(seed + row)
    surr_maxes = np.empty(n_surrogates, dtype=float)
    for j in range(n_surrogates):
        surr = _circular_block_permute(x_source, block, rng)
        best = -1.0
        for lag in range(0, max_lag + 1):
            end = row + 1 - lag
            start = max(0, end - window)
            if end - start < 2:
                continue
            offset = start - src_lo
            length = end - start
            xs = surr[offset : offset + length]
            ys = y[start + lag : row + 1]
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() < 2:
                continue
            if np.std(xs[valid]) > 0 and np.std(ys[valid]) > 0:
                value = abs(float(np.corrcoef(xs[valid], ys[valid])[0, 1]))
                if value > best:
                    best = value
        surr_maxes[j] = best if best >= 0 else np.nan
    finite = surr_maxes[np.isfinite(surr_maxes)]
    if finite.size == 0:
        return np.nan
    return real - float(np.mean(finite))


@register_operator(
    name="ts_best_lag_corr_raw",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_best_lag_corr_raw",
    source="downside_risk",
    status="experimental",
)
class TsBestLagCorrRaw(SeriesOperator):
    """source x 领先 target y 的绝对相关强度（原始峰值，无选择校正）。

    方向约定 (R5 P1-36(d))：计算 ``corr(y[t], x[t-lag])`` 在 lag=0..max_lag 上的
    最大绝对值，即 ``x``（source/领先序列）过去的值对 ``y``（target/跟随序列）
    当前的预测能力。x 领先 y，方向是 source -> target。

    R11 round-2 (命名诚实)：这是干净数学原语 —— max over lags of |Pearson corr|，
    不施加任何多重滞后选择校正。历史上 ``ts_best_lag_corr`` 的 Fisher-z 校正
    假定 K 个滞后相关是独立检验，实际上它们高度相关，故拆分为
    ``ts_best_lag_corr_excess``（置换零模型）。历史拼写 ``ts_best_lag_corr``
    保留为本算子的废弃别名。
    """

    metadata = _metadata(
        "ts_best_lag_corr_raw",
        "source x 领先 target y：max_lag 内 |corr(y[t], x[t-lag])| 的最大值（原始原语，无选择校正）。",
        ["y", "x", "window", "max_lag"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, max_lag: int = 5, **_: Any) -> pd.DataFrame:
        # R11 #159: strict integer + domain validation (``max_lag < window``).
        w = _check_int(window, "window", 2)
        ml = _check_int(max_lag, "max_lag", 0)
        if ml >= w:
            raise ValueError("max_lag must be < window")
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                value = _best_lag_corr_raw(xv[:, col], yv[:, col], row, w, ml)
                # NaN (no computable lag) propagates; a genuine 0.0 stays 0.0.
                out[row, col] = value
        return _frame_like(y, out)


@register_operator(
    name="ts_best_lag_corr_excess",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_best_lag_corr_excess",
    source="downside_risk",
    status="experimental",
)
class TsBestLagCorrExcess(SeriesOperator):
    """source x 领先 target y：原始峰值减去循环块置换零模型期望。

    ``excess = max_real - E[max_surrogate]``，其中零模型由对 source 序列 x 做
    循环块置换（固定 seed=42，n_surrogates=20，block=ceil(0.1*n)）估计。
    独立 x,y -> excess ≈ 0；真实单滞后依赖 -> excess > 0。
    """

    metadata = _metadata(
        "ts_best_lag_corr_excess",
        "source x 领先 target y：|corr| 峰值减去循环块置换零模型 E[max_surrogate]（seed=42, n_surrogates=20, block=ceil(0.1n)）。",
        ["y", "x", "window", "max_lag"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, max_lag: int = 5, **_: Any) -> pd.DataFrame:
        # R11 #159: strict integer + domain validation (``max_lag < window``).
        w = _check_int(window, "window", 2)
        ml = _check_int(max_lag, "max_lag", 0)
        if ml >= w:
            raise ValueError("max_lag must be < window")
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                out[row, col] = _best_lag_corr_excess(xv[:, col], yv[:, col], row, w, ml)
        return _frame_like(y, out)


def _price_delay_model(
    stock: np.ndarray,
    bench: np.ndarray,
    end: int,
    window: int,
    max_lag: int,
    min_periods: int,
) -> float:
    """Hou–Moskowitz style price delay for row ``end`` (causal).

    Restricted model: y_t = a + b0 * bench_t.
    Full model:       y_t = a + b0 * bench_t + … + bK * bench_{t-K}.
    Returns 1 - R²_restricted / R²_full.
    """
    lo = max(0, end - window + 1 - max_lag)
    if lo + max_lag >= end:
        return np.nan
    ts = np.arange(max(lo + max_lag, end - window + 1), end + 1)
    if ts.size < max_lag + 2:
        return np.nan
    y = stock[ts]
    bench_values = bench[ts]
    lagged = np.column_stack([bench[ts - lag] for lag in range(max_lag + 1)])
    X_full = np.column_stack([np.ones(len(ts)), lagged])
    X_restricted = np.column_stack([np.ones(len(ts)), bench_values])
    valid = np.isfinite(y) & np.all(np.isfinite(X_full), axis=1)
    # R11 #162: the full model has ~max_lag+2 parameters; with N ~ P the in-sample
    # R² is mechanically inflated.  Require a real DOF margin (at least 2x the
    # parameter count) and a well-conditioned design before trusting the fit.
    n_params = max_lag + 2
    required = max(int(min_periods), 2 * n_params)
    if valid.sum() < required:
        return np.nan
    yv = y[valid]
    xr = X_restricted[valid]
    xf = X_full[valid]
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    if sst <= 0.0:
        return np.nan
    # Condition-number gate: a near-singular lagged design makes R²_full
    # meaningless (exact collinearity between the restricted and lagged columns).
    try:
        cond = float(np.linalg.cond(xf))
    except Exception:
        cond = float("inf")
    if not np.isfinite(cond) or cond > 1e10:
        return np.nan
    beta_r, *_ = np.linalg.lstsq(xr, yv, rcond=None)
    rss_r = float(np.sum((yv - xr @ beta_r) ** 2))
    r2_r = 1.0 - rss_r / sst
    beta_f, *_ = np.linalg.lstsq(xf, yv, rcond=None)
    rss_f = float(np.sum((yv - xf @ beta_f) ** 2))
    r2_f = 1.0 - rss_f / sst
    if r2_f <= 0.0:
        return np.nan
    return max(0.0, 1.0 - r2_r / r2_f)


@register_operator(
    name="ts_price_delay",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_price_delay",
    source="downside_risk",
    status="experimental",
)
class TsPriceDelay(SeriesOperator):
    """价格延迟代理（Hou–Moskowitz 式）：1 - R²_restricted / R²_full。

    受限模型仅含当日基准收益；完整模型叠加滞后基准收益。数值越高，价格对
    基准信息反应越滞后。
    """

    metadata = _metadata(
        "ts_price_delay",
        "价格延迟代理 1-R2_restricted/R2_full。",
        ["stock_return", "benchmark_return", "window", "max_lag", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(
        self,
        stock_return: pd.DataFrame,
        benchmark_return: pd.DataFrame,
        window: int = 20,
        max_lag: int = 5,
        min_periods: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        # R11 #159: strict integer + domain validation.
        w = _check_int(window, "window", 2)
        ml = _check_int(max_lag, "max_lag", 1)
        mp = _check_int(min_periods, "min_periods", 3)
        if ml >= w:
            raise ValueError("max_lag must be < window")
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = stock_return.to_numpy(dtype=float)
        bv = benchmark_return.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                out[row, col] = _price_delay_model(sv[:, col], bv[:, col], row, w, ml, mp)
        return _frame_like(stock_return, out)


def _register_best_lag_corr_split() -> None:
    """R11 round-2: split the corrected ``ts_best_lag_corr`` into two honest
    canonicals.

    ``ts_best_lag_corr_raw`` is the clean primitive (max |corr| over lags, no
    heuristic correction); ``ts_best_lag_corr_excess`` subtracts a circular-
    block-permutation null.  The historical spelling ``ts_best_lag_corr``
    becomes a deprecated alias of the raw version.
    ``rename_canonical``/``register_alias`` preserve the old DSL name.  The
    static surface partition is updated in sync (new canonicals -> extended,
    retired spelling -> removed) so ``layer_governance``'s exact-classification
    check still holds at finalize.
    """
    from cleaned_operators.registry import OperatorRegistry

    if "ts_best_lag_corr" in OperatorRegistry._operators:
        # A backend registered the historical spelling directly; migrate it.
        OperatorRegistry.rename_canonical("ts_best_lag_corr", "ts_best_lag_corr_raw")
    elif "ts_best_lag_corr_raw" in OperatorRegistry._operators:
        OperatorRegistry.register_alias("ts_best_lag_corr", "ts_best_lag_corr_raw")
    try:
        from cleaned_operators.operator_surface import (
            extend_extended_only,
            retract_extended_only,
        )

        extend_extended_only(["ts_best_lag_corr_raw", "ts_best_lag_corr_excess"])
        retract_extended_only(["ts_best_lag_corr"])
    except ImportError:  # pragma: no cover - surface module always present in-tree
        pass


_register_best_lag_corr_split()

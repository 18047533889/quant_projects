# -*- coding: utf-8 -*-
"""Direction ratios and temporal concentration operators.

These measure the fraction of positive/negative/zero observations in a rolling
window and how concentrated absolute changes are across time (HHI / entropy).
All operators are causal daily-panel transforms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str, output_unit: str | None = None) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
        output_unit=output_unit,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _strict_bool(value: Any, name: str = "normalize") -> bool:
    """Strict boolean: only True/False, 1/0 or 'true'/'false'/'1'/'0' accepted.

    Anything else (e.g. the string ``"false"`` reaching a bare ``bool()`` and
    silently becoming ``True``) raises ``ValueError``.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    elif isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1"}:
            return True
        if s in {"false", "0"}:
            return False
    raise ValueError(f"{name} must be a strict bool (true/false/1/0)")


def _rolling_apply_2d(values: np.ndarray, window: int, fn: Any, min_periods: int = 1) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = values[start : row + 1, col]
            out[row, col] = fn(chunk)
    return out


def _sign_ratio(values: np.ndarray, sign: int, threshold: float, min_periods: int) -> float:
    # R5 P1-35(a): a negative or non-finite threshold is a caller bug — reject it
    # instead of silently producing a meaningless ratio.
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    finite = values[np.isfinite(values)]
    if finite.size < min_periods:
        return np.nan
    if sign > 0:
        hits = float(np.sum(finite > threshold))
    elif sign < 0:
        hits = float(np.sum(finite < threshold))
    else:
        hits = float(np.sum(np.abs(finite) <= threshold))
    return hits / finite.size


@register_operator(
    name="ts_positive_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_positive_ratio",
    source="direction_concentration",
    status="experimental",
)
class TsPositiveRatio(SeriesOperator):
    """窗口内 x > threshold 的有效观测比例。"""

    metadata = _metadata(
        "ts_positive_ratio",
        "窗口内 x > threshold 的有效观测比例。",
        ["x", "window", "threshold", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, threshold: float = 0.0, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        thr = float(threshold)
        mp = max(1, int(min_periods))
        return _frame_like(
            x,
            _rolling_apply_2d(
                x.to_numpy(dtype=float), w, lambda c: _sign_ratio(c, 1, thr, mp), mp
            ),
        )


@register_operator(
    name="ts_negative_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_negative_ratio",
    source="direction_concentration",
    status="experimental",
)
class TsNegativeRatio(SeriesOperator):
    """窗口内 x < threshold 的有效观测比例。"""

    metadata = _metadata(
        "ts_negative_ratio",
        "窗口内 x < threshold 的有效观测比例。",
        ["x", "window", "threshold", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, threshold: float = 0.0, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        thr = float(threshold)
        mp = max(1, int(min_periods))
        return _frame_like(
            x,
            _rolling_apply_2d(
                x.to_numpy(dtype=float), w, lambda c: _sign_ratio(c, -1, thr, mp), mp
            ),
        )


@register_operator(
    name="ts_zero_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_zero_ratio",
    source="direction_concentration",
    status="experimental",
)
class TsZeroRatio(SeriesOperator):
    """窗口内 |x| <= tolerance 的有效观测比例。"""

    metadata = _metadata(
        "ts_zero_ratio",
        "窗口内 |x| <= tolerance 的有效观测比例。",
        ["x", "window", "tolerance", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, tolerance: float = 0.0, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        tol = float(tolerance)
        mp = max(1, int(min_periods))
        return _frame_like(
            x,
            _rolling_apply_2d(
                x.to_numpy(dtype=float), w, lambda c: _sign_ratio(c, 0, tol, mp), mp
            ),
        )


def _abs_concentration(values: np.ndarray, min_periods: int) -> float:
    abs_values = np.abs(values[np.isfinite(values)])
    if abs_values.size < min_periods:
        return np.nan
    total = float(abs_values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.nan
    shares = abs_values / total
    return float(np.sum(shares * shares))


@register_operator(
    name="ts_abs_concentration",
    category="time_series",
    business_category="time_series",
    canonical="ts_abs_concentration",
    source="direction_concentration",
    status="experimental",
)
class TsAbsConcentration(SeriesOperator):
    """窗口内 |x| 份额的 HHI（原始 Herfindahl = sum(share_i^2)，范围 [1/N, 1]）。

    注意 (R5 P1-35(d))：原始 HHI 带有随窗口长度变化的机械 baseline —— 完全均匀
    分布时 HHI = 1/N，N 越小 baseline 越高，跨不同窗口长度直接比较会失真。需要
    与样本长度无关的度量时，请使用 ``ts_mass_concentration``
    （alpha_language_shape），它输出归一化 excess-HHI
    ``(HHI - 1/N) / (1 - 1/N)``，范围 [0, 1]。
    """

    metadata = _metadata(
        "ts_abs_concentration",
        "窗口内 |x| 份额的原始 HHI（有 1/N 机械 baseline；归一化 excess-HHI 见 ts_mass_concentration）。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        output_unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(1, int(min_periods))
        return _frame_like(
            x, _rolling_apply_2d(x.to_numpy(dtype=float), w, lambda c: _abs_concentration(c, mp), mp)
        )


def _abs_entropy(values: np.ndarray, min_periods: int, normalize: bool) -> float:
    abs_values = np.abs(values[np.isfinite(values)])
    if abs_values.size < min_periods:
        return np.nan
    total = float(abs_values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.nan
    shares = abs_values / total
    entropy = float(-np.sum(shares * np.log(shares + 1e-300)))
    if normalize and shares.size > 1:
        entropy = entropy / np.log(shares.size)
    return entropy


@register_operator(
    name="ts_abs_entropy",
    category="time_series",
    business_category="time_series",
    canonical="ts_abs_entropy",
    source="direction_concentration",
    status="experimental",
)
class TsAbsEntropy(SeriesOperator):
    """窗口内 |x| 份额的（归一化）熵：接近 1 表示分布均匀，接近 0 表示集中。

    单位随 ``normalize`` 变化 (R5 P1-35(c))：``normalize=True`` 时输出无量纲
    [0, 1]（除以 log(N)）；``normalize=False`` 时输出自然对数单位 (nats)，
    两者量纲不同，不可直接混用比较。
    """

    metadata = _metadata(
        "ts_abs_entropy",
        "窗口内 |x| 份额的熵；normalize=True 无量纲 [0,1]，normalize=False 为 nats。",
        ["x", "window", "normalize", "min_periods"],
        domain="price_volume",
        unit="ratio",
        output_unit="normalized:dimensionless / raw:nats",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, normalize: bool = True, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(1, int(min_periods))
        # R5 P1-35(b): strict bool — a bare ``bool("false")`` would silently be
        # True; reject anything that is not a genuine true/false/1/0 value.
        norm = _strict_bool(normalize, "normalize")
        return _frame_like(
            x, _rolling_apply_2d(x.to_numpy(dtype=float), w, lambda c: _abs_entropy(c, mp, norm), mp)
        )

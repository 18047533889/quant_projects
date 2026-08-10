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
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
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
            *extra_tags,
        ],
        output_unit=output_unit,
        input_units=dict(input_units or {}),
        compatible_units={k: tuple(v) for k, v in (compatible_units or {}).items()},
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _strict_bool(value: Any, name: str = "normalize") -> bool:
    """R16-101: THE single strict boolean authority.

    The old LOCAL parser accepted strings (``'true'`` / ``'false'`` / ``'1'``),
    inconsistent with the global policy that rejects string coercion entirely.
    Now it delegates to ``cleaned_operators.base.strict_bool_param`` — only
    ``type(x) is bool`` passes; a string reaching a kernel means the parameter
    was not declared boolean at the DSL/binder layer.
    """
    from cleaned_operators.base import strict_bool_param

    return strict_bool_param(value, name)


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
    """窗口内 x > threshold 的有效观测比例。

    R11 round-3 P1-I-124: 阈值相对算子 —— 语义随输入单位剧烈变化 (Return>0 有
    意义, Volume>0 / Price>0 几乎恒真)。输入应是有符号收益/比率/同比等零阈值有
    经济含义的序列, 而非原始价格/成交量水平 (metadata 层通过 ``input_semantics`` /
    ``compatible_units`` 声明)。
    """

    metadata = _metadata(
        "ts_positive_ratio",
        "窗口内 x > threshold 的有效观测比例 (输入应为有符号收益/比率, 非原始价格/量)。",
        ["x", "window", "threshold", "min_periods"],
        domain="price_volume",
        unit="ratio",
        input_units={"x": "return_or_ratio_or_signed_numeric"},
        compatible_units={"x": ("return", "ratio", "rate", "signed_numeric")},
        extra_tags=("input_semantics:return_or_ratio",),
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
    """窗口内 x < threshold 的有效观测比例。

    R11 round-3 P1-I-124: 阈值相对算子 —— 输入应是有符号收益/比率 (Return<0 有
    意义, Volume<0 无意义)。原始价格/成交量水平会被 ``input_semantics`` /
    ``compatible_units`` 在 metadata 层拒绝。
    """

    metadata = _metadata(
        "ts_negative_ratio",
        "窗口内 x < threshold 的有效观测比例 (输入应为有符号收益/比率, 非原始价格/量)。",
        ["x", "window", "threshold", "min_periods"],
        domain="price_volume",
        unit="ratio",
        input_units={"x": "return_or_ratio_or_signed_numeric"},
        compatible_units={"x": ("return", "ratio", "rate", "signed_numeric")},
        extra_tags=("input_semantics:return_or_ratio",),
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
    """窗口内 |x| <= tolerance 的有效观测比例。

    R11 round-3 P1-I-124: 阈值相对算子 —— 对零附近容忍区间的占比。原始价格
    (>0 恒真, 永不接近 0) / 原始成交量语义退化, metadata 层经 ``input_semantics`` /
    ``compatible_units`` 声明输入应为有符号收益/比率。
    """

    metadata = _metadata(
        "ts_zero_ratio",
        "窗口内 |x| <= tolerance 的有效观测比例 (输入应为有符号收益/比率, 非原始价格/量)。",
        ["x", "window", "tolerance", "min_periods"],
        domain="price_volume",
        unit="ratio",
        input_units={"x": "return_or_ratio_or_signed_numeric"},
        compatible_units={"x": ("return", "ratio", "rate", "signed_numeric")},
        extra_tags=("input_semantics:return_or_ratio",),
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

    R11 round-3 P1-I-126：原始 HHI 与归一化 excess-HHI 高度相关；默认挖矿表面
    使用归一化 ``ts_mass_concentration``，本算子（原始 HHI）保留在 extended 表面。
    """

    metadata = _metadata(
        "ts_abs_concentration",
        "窗口内 |x| 份额的原始 HHI（有 1/N 机械 baseline；默认表面用归一化 ts_mass_concentration）。",
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

    R11 round-3 P1-I-125：输出单位不得通过布尔切换 —— 旧签名里 ``normalize=False``
    输出 nats、``normalize=True`` 输出无量纲，量纲不同不可混用。本算子已拆分：
      * ``ts_abs_entropy_normalized`` —— 固定无量纲 [0,1]（除以 log(N)）；
      * ``ts_abs_entropy_nats`` —— 固定自然对数单位 (nats)。
    为向后兼容保留 ``normalize`` 参数，但仅接受 ``True``；请求 ``False`` 是调用
    错误，引导用户改用 ``ts_abs_entropy_nats``。
    """

    metadata = _metadata(
        "ts_abs_entropy",
        "窗口内 |x| 份额的熵（已拆分；normalize 仅接受 True，nats 用 ts_abs_entropy_nats）。",
        ["x", "window", "normalize", "min_periods"],
        domain="price_volume",
        unit="ratio",
        output_unit="dimensionless",
        extra_tags=("deprecated:split_into_ts_abs_entropy_normalized_ts_abs_entropy_nats",),
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, normalize: bool = True, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(1, int(min_periods))
        # R5 P1-35(b): strict bool — a bare ``bool("false")`` would silently be
        # True; reject anything that is not a genuine true/false/1/0 value.
        norm = _strict_bool(normalize, "normalize")
        # R11 round-3 P1-I-125: the output UNIT must never switch via the boolean.
        # ``normalize=False`` (nats) is a separate canonical now; requesting it
        # here is a caller bug, not a supported unit switch.
        if not norm:
            raise ValueError(
                "ts_abs_entropy: normalize=False (nats output) is no longer "
                "supported on this operator (R11 round-3 P1-I-125).  Use "
                "ts_abs_entropy_nats for nats units or ts_abs_entropy_normalized "
                "for dimensionless [0,1] units."
            )
        return _frame_like(
            x, _rolling_apply_2d(x.to_numpy(dtype=float), w, lambda c: _abs_entropy(c, mp, True), mp)
        )


@register_operator(
    name="ts_abs_entropy_normalized",
    category="time_series",
    business_category="time_series",
    canonical="ts_abs_entropy_normalized",
    source="direction_concentration",
    status="experimental",
)
class TsAbsEntropyNormalized(SeriesOperator):
    """窗口内 |x| 份额的归一化熵：无量纲 [0,1]，1=均匀分布，0=集中。

    R11 round-3 P1-I-125: ``ts_abs_entropy`` 的 normalize 布尔曾同时切换输出单位
    (True 无量纲 / False nats)。本算子为固定无量纲版本 —— 输出单位固定，不再随
    布尔变化。nats 版本见 ``ts_abs_entropy_nats``。
    """

    metadata = _metadata(
        "ts_abs_entropy_normalized",
        "窗口内 |x| 份额的归一化熵（无量纲 [0,1]，固定单位，无 normalize 布尔）。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        output_unit="dimensionless",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(1, int(min_periods))
        return _frame_like(
            x,
            _rolling_apply_2d(x.to_numpy(dtype=float), w, lambda c: _abs_entropy(c, mp, True), mp),
        )


@register_operator(
    name="ts_abs_entropy_nats",
    category="time_series",
    business_category="time_series",
    canonical="ts_abs_entropy_nats",
    source="direction_concentration",
    status="experimental",
)
class TsAbsEntropyNats(SeriesOperator):
    """窗口内 |x| 份额的熵：自然对数单位 (nats)。

    R11 round-3 P1-I-125: 固定 nats 版本 —— 输出单位固定，不再随布尔变化。
    无量纲 [0,1] 版本见 ``ts_abs_entropy_normalized``。
    """

    metadata = _metadata(
        "ts_abs_entropy_nats",
        "窗口内 |x| 份额的熵（nats，固定单位，无 normalize 布尔）。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="nats",
        output_unit="nats",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(1, int(min_periods))
        return _frame_like(
            x,
            _rolling_apply_2d(x.to_numpy(dtype=float), w, lambda c: _abs_entropy(c, mp, False), mp),
        )


def _r11_round3_split_and_surfaces() -> None:
    """R11 round-3 (P1-I-124/125/126) surface + split finalization.

    * ``ts_abs_entropy_normalized`` is the default-surface successor of the
      unit-switching ``ts_abs_entropy`` — register it on the daily surface AND in
      EXTENDED_ONLY (layer_governance's static partition check requires every
      daily-migrated canonical to also live in a partition set; DAILY_FACTOR_MIGRATED
      is not one of them).
    * ``ts_abs_entropy_nats`` is a niche fixed-unit variant — extended only.
    * ``ts_abs_concentration`` (raw HHI) is demoted to extended: the default
      mining surface prefers normalized excess-HHI (``ts_mass_concentration``).
      Both ``REVIEWED_MIGRATION_MANIFEST`` and the legacy ``DAILY_FACTOR_MIGRATED``
      frozenset are updated together so the surface-orthogonal invariant
      (``len(manifest) == len(DAILY_FACTOR_MIGRATED)``) is preserved.
    """
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(
        {"ts_abs_entropy_normalized", "ts_abs_entropy_nats"}
    )
    _surface.register_daily_migration(
        "ts_abs_entropy_normalized",
        review_id="r11-round3-p1-i-125-abs-entropy-split",
    )
    # Raw HHI stays on extended; normalized excess-HHI is the default surface.
    if "ts_abs_concentration" in _surface.REVIEWED_MIGRATION_MANIFEST:
        _surface.REVIEWED_MIGRATION_MANIFEST.pop("ts_abs_concentration")
    _surface.DAILY_FACTOR_MIGRATED = frozenset(
        c for c in _surface.DAILY_FACTOR_MIGRATED if c != "ts_abs_concentration"
    )


_r11_round3_split_and_surfaces()

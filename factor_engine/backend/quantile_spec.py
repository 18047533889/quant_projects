# -*- coding: utf-8
"""分位数/百分位/winsorize 命名与语义拆分契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QuantileRole = Literal["value", "bucket", "below_indicator", "winsorize_clip"]
Interpolation = Literal["linear", "lower", "higher", "nearest", "midpoint"]


@dataclass(frozen=True)
class QuantileOpSpec:
    """单个 quantile 类算子的角色与参数域。"""

    role: QuantileRole
    p_domain: tuple[float, float] = (0.0, 1.0)
    interpolation: Interpolation = "linear"
    null_input_is_null: bool = True
    small_sample_is_null: bool = False


# 当前 canonical 名称 → 公开语义（长期可拆分为 cs_quantile_value 等）
QUANTILE_OP_SPECS: dict[str, QuantileOpSpec] = {
    "quantile": QuantileOpSpec(role="value"),
    "cs_quantile": QuantileOpSpec(role="value"),
    "ts_quantile": QuantileOpSpec(role="value"),
    "c_percentile": QuantileOpSpec(role="value"),
    "group_quantile_value": QuantileOpSpec(role="value"),
    "group_percentile": QuantileOpSpec(
        role="below_indicator",
        p_domain=(0.0, 1.0),
        null_input_is_null=True,
    ),
    "winsorize": QuantileOpSpec(role="winsorize_clip"),
    "group_winsorize": QuantileOpSpec(role="winsorize_clip"),
}


def quantile_spec_for(canon: str) -> QuantileOpSpec:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return QUANTILE_OP_SPECS.get(name, QuantileOpSpec(role="value"))


def group_percentile_is_indicator() -> bool:
    """``group_percentile`` 返回 0/1：rank/n <= p，非分位数值。"""
    return quantile_spec_for("group_percentile").role == "below_indicator"


def winsorize_is_clip() -> bool:
    return quantile_spec_for("winsorize").role == "winsorize_clip"

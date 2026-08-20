# -*- coding: utf-8 -*-
"""PriceBasis 契约（R40 #228）。

涨跌停算子 / return 算子 / limit 价必须知道自己输入的**价格基准**：

* ``RAW`` — 未复权 OHLC（涨跌停系列要求，官方涨停价也是 RAW 基准）；
* ``RAW_OFFICIAL_LIMIT`` — 交易所官方涨跌停价（与 RAW 兼容）；
* ``ADJUSTED`` — 后复权 / 前复权价格（不可与 RAW 混用）；
* ``CONTINUOUS`` — 连续合约（期货）等连续序列。

#228 的硬规则：operator 声明 ``PriceBasis=RAW``（OHLC）+ ``RAW_OFFICIAL_LIMIT``
（limit 价）时，compile-time 检查输入实际 basis；RAW 与
ADJUSTED/CONTINUOUS 混用 → ``PriceBasisMismatchError`` hard fail。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class PriceBasis(str, enum.Enum):
    RAW = "raw"
    RAW_OFFICIAL_LIMIT = "raw_official_limit"
    ADJUSTED = "adjusted"
    CONTINUOUS = "continuous"


#: 与 ``PriceBasis.RAW`` 兼容的 basis 集合（官方涨跌停价本身是 RAW 基准）。
_RAW_COMPATIBLE = frozenset({PriceBasis.RAW, PriceBasis.RAW_OFFICIAL_LIMIT})


class PriceBasisMismatchError(ValueError):
    """R40 #228：输入价格 basis 与算子声明不兼容（compile-time hard fail）。"""


@dataclass(frozen=True)
class PriceBasisContract:
    """算子声明：每个参数位置要求的价格基准。

    ``declared`` 映射参数名 -> 要求的 PriceBasis。``validate`` 对每个实际
    basis 做兼容性检查：

    - 要求 RAW 的参数不接受 ADJUSTED / CONTINUOUS；
    - 要求 ADJUSTED 的参数接受 ADJUSTED / CONTINUOUS（后复权与连续同族）；
    - 未知实际 basis（``None``）默认按 RAW 放行（调用方没传 = 未声明，
      research 兼容；production 检查应显式传入 basis）。
    """

    declared: dict[str, PriceBasis]
    canonical: str = ""

    def validate(self, actual_basis: dict[str, PriceBasis | None]) -> None:
        for param, required in self.declared.items():
            actual = actual_basis.get(param)
            if actual is None:
                continue
            if required == PriceBasis.RAW and actual not in _RAW_COMPATIBLE:
                raise PriceBasisMismatchError(
                    f"{self.canonical or 'operator'} requires {param!r} to be "
                    f"PriceBasis.RAW (or RAW_OFFICIAL_LIMIT), got {actual.value!r} "
                    "— adjusted / continuous prices must never feed raw "
                    "limit-distance or touch logic"
                )
            if required in (PriceBasis.ADJUSTED, PriceBasis.CONTINUOUS):
                if actual not in (PriceBasis.ADJUSTED, PriceBasis.CONTINUOUS):
                    raise PriceBasisMismatchError(
                        f"{self.canonical or 'operator'} requires {param!r} to be "
                        f"{required.value!r}, got {actual.value!r}"
                    )


#: 涨跌停系列算子的标准契约：#228 —— OHLC 必须是 RAW，limit 价必须是
#: RAW_OFFICIAL_LIMIT。
LIMIT_OPS_PRICE_BASIS = PriceBasisContract(
    canonical="ashare_limit_*",
    declared={
        "open": PriceBasis.RAW,
        "high": PriceBasis.RAW,
        "low": PriceBasis.RAW,
        "close": PriceBasis.RAW,
        "upper_limit": PriceBasis.RAW_OFFICIAL_LIMIT,
        "lower_limit": PriceBasis.RAW_OFFICIAL_LIMIT,
    },
)


def validate_limit_ops_price_basis(basis: dict[str, PriceBasis | None]) -> None:
    """compile-time 入口：涨跌停算子调 ``_calculate_series`` 前调用。"""
    LIMIT_OPS_PRICE_BASIS.validate(basis)


__all__ = [
    "LIMIT_OPS_PRICE_BASIS",
    "PriceBasis",
    "PriceBasisContract",
    "PriceBasisMismatchError",
    "validate_limit_ops_price_basis",
]

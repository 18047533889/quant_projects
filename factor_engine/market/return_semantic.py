# -*- coding: utf-8 -*-
"""ReturnSemantic —— return 算子语义身份（R40 #229）。

所有 return 算子过去混为一类，无法区分「前复权 close-to-close」「后复权
overnight」「总回报（含分红再投）」。R40 #229 引入显式语义 dataclass 并折叠
进 semantic identity：same expr + 不同 return 语义 => 不同 factor 身份
（不同 cache/checkpoint/evidence）。

与 :mod:`market.adjustment_policy` 联动：``corporate_action_adjustment`` 区分
``RAW``（完全不复权）/ ``SPLIT_DIV_ADJUSTED``（拆分+现金分红复权）/
``TOTAL_RETURN``（总回报，含再投资）。``RETROSPECTIVE_ADJUSTED`` 的复权因子
本身有 PIT 语义（见 AdjustmentVintage），所以 return 的语义 hash 必须绑定
复权 vintage 的 knowledge_time。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from factor_engine.market.price_basis import PriceBasis


class ReturnInterval(str, enum.Enum):
    CLOSE_TO_CLOSE = "close_to_close"
    OPEN_TO_CLOSE = "open_to_close"
    OVERNIGHT = "overnight"


class CorporateActionAdjustment(str, enum.Enum):
    RAW = "raw"
    SPLIT_DIV_ADJUSTED = "split_div_adjusted"
    TOTAL_RETURN = "total_return"


@dataclass(frozen=True)
class ReturnSemantic:
    """return 算子的完整数值语义。

    ``source_price_basis`` 是基础价格基准；``interval`` 描述收益的时间跨度；
    ``corporate_action_adjustment`` 描述复权 / 总回报口径；
    ``adjustment_knowledge_time`` 是复权因子可及时点（PIT，R40 #230）——复权
    因子的后验修正（如除权除息日之后才公布）必须绑定到这里，避免未来事件
    泄漏进历史 return。
    """

    source_price_basis: PriceBasis = PriceBasis.ADJUSTED
    interval: ReturnInterval = ReturnInterval.CLOSE_TO_CLOSE
    corporate_action_adjustment: CorporateActionAdjustment = CorporateActionAdjustment.SPLIT_DIV_ADJUSTED
    adjustment_knowledge_time: str = ""

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source_price_basis": self.source_price_basis.value,
            "interval": self.interval.value,
            "corporate_action_adjustment": self.corporate_action_adjustment.value,
            "adjustment_knowledge_time": self.adjustment_knowledge_time,
        }

    def identity_hash(self) -> str:
        """跨进程稳定的语义 hash（进 factor semantic identity）。"""
        import hashlib

        raw = ",".join(
            f"{k}={v}"
            for k, v in sorted(self.identity_payload().items())
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CorporateActionAdjustment",
    "ReturnInterval",
    "ReturnSemantic",
]

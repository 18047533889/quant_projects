# -*- coding: utf-8 -*-
"""P11+P12: MaterializationTier —— 不是所有因子都永久物化全历史。

100k 因子 × 5000 股 × float32 全历史不可行。因子按用途分四档，agent mining 默认
TIER0（绝不自动把生成的全部因子永久物化到全历史）：

    TIER0_EPHEMERAL_CANDIDATE   compute → FactorEvaluator → 丢弃值；只保留
                                definition / fingerprint / evaluation。
    TIER1_RESEARCH              只存研究需要的历史窗口（短窗）/ 已初筛因子。
    TIER2_PRODUCTION            全历史 + 每日增量 + checkpoint。

各 tier 语义：

  * TIER0：值即刻丢弃。只保留因子定义（definition）、fingerprint（内容指纹）与
    评估结果（evaluation）。不落任何 FeatureBlock 值文件。
  * TIER1：只保留 ``history_window`` 天窗口内最近的值（短窗），用于研究/回看，
    不存全历史。窗口由 ``history_days`` 指定，超出窗口的旧值裁剪。
  * TIER2：全历史。写入 FeatureBlockWriter（分组块），支持每日增量 append 与
    checkpoint（manifest 即 checkpoint）。

策略由 ``resolve_tier`` 决策：显式 tier 优先，否则按是否生产/候选默认。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- tier 常量 --------------------------------------------------------------
TIER0 = "TIER0_EPHEMERAL_CANDIDATE"
TIER1 = "TIER1_RESEARCH"
TIER2 = "TIER2_PRODUCTION"

_TIER_ORDER = {TIER0: 0, TIER1: 1, TIER2: 2}

#: 各 tier 是否物理落值（是否写 FeatureBlock 值文件）。
TIER_PERSISTS_VALUES = {TIER0: False, TIER1: True, TIER2: True}

#: 各 tier 默认历史窗口（天）。TIER1 短窗、TIER2 全历史（None=不裁剪）。
DEFAULT_HISTORY_DAYS = {TIER0: 0, TIER1: 250, TIER2: None}

#: tier 标签（用于 manifest / 元数据）。
TIER_LABEL = {
    TIER0: "ephemeral_candidate",
    TIER1: "research",
    TIER2: "production",
}

#: 反向查找标签 → tier。
_LABEL_TO_TIER = {v: k for k, v in TIER_LABEL.items()}


def tier_from_label(label: str) -> str | None:
    """由标签（manifest 存的是标签）还原 tier。"""
    return _LABEL_TO_TIER.get(str(label))


@dataclass
class FactorMaterialization:
    """单个因子的物化档位决策 + 结果载体。"""

    factor_id: str
    tier: str = TIER0
    definition: Any | None = None      # 因子定义（表达式/参数）
    fingerprint: str | None = None     # 内容指纹（TIER0 保留，值丢弃）
    evaluation: dict[str, Any] = field(default_factory=dict)  # FactorEvaluator 结果
    history_days: int | None = None    # TIER1 窗口；TIER2 为 None（全历史）
    values: Any | None = None          # TIER1/TIER2 实际落盘值；TIER0 恒 None
    block_id: str | None = None        # TIER1/TIER2 写入的 FeatureBlock
    column: int | None = None          # 块内列号
    persisted: bool = False            # 是否已物理落值

    def persist(self, values: Any, *, block_id: str | None = None, column: int | None = None) -> None:
        """按 tier 落值。TIER0 抛错：ephemeral 永不物理落值。"""
        if self.tier == TIER0:
            raise ValueError(
                f"{self.factor_id}: TIER0 是 ephemeral candidate，值即刻丢弃，不允许 persist"
            )
        self.values = values
        self.block_id = block_id
        self.column = column
        self.persisted = True

    def keep_values(self) -> bool:
        return TIER_PERSISTS_VALUES.get(self.tier, False)

    def to_metadata(self) -> dict[str, Any]:
        """写进 manifest 的元数据（与值分离）。"""
        return {
            "factor_id": self.factor_id,
            "tier": TIER_LABEL.get(self.tier, self.tier),
            "history_days": self.history_days,
            "block_id": self.block_id,
            "column": self.column,
            "persisted": self.persisted,
            "fingerprint": self.fingerprint,
            "evaluation": dict(self.evaluation),
        }


def resolve_tier(
    *,
    explicit: str | None = None,
    is_candidate: bool = True,
    is_production: bool = False,
    research_window: int | None = None,
) -> str:
    """决策 tier：显式 tier 优先，否则按生产/候选默认。

    规则：
      * ``explicit`` 给定 → 直接返回（校验合法）。
      * 否则：候选（刚生成的、未短筛的）→ TIER0（默认，绝不自动全历史）。
      * 生产模式且非候选 → TIER2。
      * 研究窗口给定 → TIER1（短窗）。
      * 兜底 → TIER0。
    """
    if explicit is not None:
        if explicit not in _TIER_ORDER:
            raise ValueError(f"未知 tier: {explicit}")
        return explicit
    if is_candidate:
        # agent mining 默认：新生成因子先不落全历史。
        return TIER0
    if research_window is not None and research_window > 0:
        return TIER1
    if is_production:
        return TIER2
    return TIER0


def history_days_for(tier: str, explicit_days: int | None = None) -> int | None:
    """某 tier 的历史窗口（天）。TIER2/显式覆盖优先。"""
    if tier == TIER2:
        return None  # 全历史，不裁剪
    if explicit_days is not None and explicit_days > 0:
        return int(explicit_days)
    return DEFAULT_HISTORY_DAYS.get(tier)


def trim_to_window(values: Any, history_days: int | None) -> Any:
    """TIER1：裁剪到最近 history_days 天窗口（值在后，丢弃窗口前的旧值）。

    TIER2（history_days is None）原样返回（全历史）。``history_days<=0``（TIER0）
    返回空——值即丢弃。
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if history_days is None:
        return arr  # TIER2 全历史
    if history_days <= 0:
        return arr[:0].copy()  # TIER0：值丢弃
    return arr[-int(history_days):].copy()

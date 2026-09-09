"""A 股成本情景接口（gross 默认 + 1x/2x/3x 成本情景留存）。

成本按每笔单边（入场 + 出场各一次）的比例收取，扣在真实 cohort 组合的
每日 PnL 上。默认 per_side_cost=0 = gross 语义（与规范 §7 一致：先看
gross signal quality，再看成本敏感性）。

参考 A 股费率（每笔单边，买卖合计约 0.1%-0.15%）：
- 佣金 ~0.025%（最低 5 元，此处按比例口径）
- 印花税 0.05%（卖出单边）
- 过户费 ~0.001%
"""

from __future__ import annotations

from typing import Dict, Optional
from types import MappingProxyType
from dataclasses import dataclass
import numpy as np

# 1x/2x/3x 成本情景（per-side 比例）：0 = gross，1x = 标准 A 股估算，
# 2x/3x 用于成本压力测试。留存接口，供外部调用方批量跑情景。
COST_SCENARIOS = MappingProxyType({
    "gross": 0.0,
    "1x": 0.0012,
    "2x": 0.0024,
    "3x": 0.0036,
})

@dataclass(frozen=True)
class CostEvidence:
    """Typed cost result; heuristic scans cannot satisfy formal net metrics."""
    values: np.ndarray
    evidence_level: str
    formal_net_eligible: bool
    components: MappingProxyType


def compute_transaction_costs(traded_notional, aum, *, commission_rate=0.0,
                              slippage_rate=0.0, financing_balance=None,
                              financing_rate=0.0, borrow_balance=None,
                              borrow_rate=0.0):
    """Exact linear costs from traded notional, with carry kept separate."""
    traded = np.asarray(traded_notional, dtype=float)
    if traded.ndim != 1 or not np.isfinite(traded).all() or np.any(traded < 0):
        raise ValueError("traded_notional must be a finite nonnegative 1D trajectory")
    if isinstance(aum, bool) or not np.isfinite(aum) or aum <= 0:
        raise ValueError("aum must be positive and finite")
    rates = (commission_rate, slippage_rate, financing_rate, borrow_rate)
    if any(isinstance(x, bool) or not np.isfinite(x) or x < 0 for x in rates):
        raise ValueError("cost rates must be finite and nonnegative")
    def balance(value, name):
        if value is None: return np.zeros_like(traded)
        out=np.asarray(value,dtype=float)
        if out.shape!=traded.shape or not np.isfinite(out).all() or np.any(out<0): raise ValueError(f"{name} must align and be nonnegative")
        return out
    financing=balance(financing_balance,"financing_balance")*financing_rate/aum
    borrow=balance(borrow_balance,"borrow_balance")*borrow_rate/aum
    commission=traded*commission_rate/aum; slippage=traded*slippage_rate/aum
    components=MappingProxyType({"commission":commission.copy(),"slippage":slippage.copy(),"financing":financing.copy(),"borrow":borrow.copy()})
    return CostEvidence(commission+slippage+financing+borrow,"TRADE_AND_CARRY_LINEAR_V1",True,components)


def cost_scenario(name: Optional[str]) -> float:
    """按情景名取 per-side cost 比例。

    Args:
        name: "gross" | "1x" | "2x" | "3x" | None（= "gross"）。
              也接受任意 float（直接作为 per-side cost）。

    Returns:
        per-side cost 比例。
    """
    if name is None:
        return COST_SCENARIOS["gross"]
    if isinstance(name, (int, float, np.floating)):
        value = float(name)
        if not (0.0 <= value < 1.0):
            raise ValueError(f"per-side cost 必须在 [0, 1)，got {value}")
        return value
    key = str(name)
    if key not in COST_SCENARIOS:
        raise ValueError(
            f"未知成本情景 {name!r}；可选 {sorted(COST_SCENARIOS.keys())} "
            "或直接传 float"
        )
    return COST_SCENARIOS[key]


def apply_per_side_costs(
    pnl_net: np.ndarray,
    cost_multiplier: float = 1.0,
    base_cost: float = 0.0012,
    *, research_only: bool = False,
) -> np.ndarray:
    """按成本情景倍数近似折算净 PnL（二次线性缩放，仅用于快速敏感性）。

    .. warning::
        这是近似口径：真实成本应通过 compute_cohort_pnl(per_side_cost=...)
        重算，本函数只用于成本情景的快速扫描（gross 结果 × 成本倍数缩放）。
        精确口径请重新跑 cohort。

    Args:
        pnl_net : (T,) 某成本下的每日 PnL（通常是 gross）。
        cost_multiplier : 成本倍数（1x/2x/3x）。
        base_cost : 基准单边成本（默认 0.0012）。

    Returns:
        (T,) 近似净 PnL。
    """
    if research_only is not True:
        raise ValueError("HEURISTIC_COST_SCAN requires explicit research_only=True; not formal net evidence")
    if not np.isfinite(cost_multiplier) or cost_multiplier < 0 or not np.isfinite(base_cost) or base_cost < 0:
        raise ValueError("cost inputs must be finite and nonnegative")
    pnl_net = np.asarray(pnl_net, dtype=np.float64)
    # 近似：净 PnL ≈ gross PnL - 成本拖累；成本拖累正比于双边名义额。
    # 简化：假设双边名义额 = 2（多空各 1），每日摊还 = 2 * base / H。
    # 该近似只用于快速敏感性演示，精确值必须以 cohort 重算为准。
    drag = 2.0 * base_cost * cost_multiplier / 20.0
    return pnl_net - drag


def heuristic_cost_scan(pnl_gross, **kwargs):
    """Typed wrapper for the explicitly non-qualifying approximation."""
    values=apply_per_side_costs(pnl_gross,research_only=True,**kwargs)
    return CostEvidence(values,"HEURISTIC_COST_SCAN",False,MappingProxyType({}))

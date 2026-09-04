"""FactorFitness V2.1 —— confidence shrinkage（plan Task 8.2 / Part G #20）。

对 utility U ∈ [0, 1] 做有效样本置信收缩：

    reliability = sqrt(n_eff / (n_eff + k))
    U_conf      = 0.5 + reliability * (U - 0.5)

- n_eff=0（无有效样本）→ reliability=0 → U_conf=0.5（中性，绝不当最好）。
- n_eff→∞ → reliability→1 → U_conf→U（不改变排序）。
- k 默认中性默认值（Part J3：不宣称最优），可配置。

k 语义：达到 reliability=sqrt(0.5)≈0.707 所需的有效样本数。
"""

from __future__ import annotations

import math
from typing import Any

#: 默认收缩强度 k（Part J3 中性默认值，不宣称最优；生产可经校准后覆盖）。
DEFAULT_SHRINKAGE_K = 64.0


def reliability_of(n_eff: int | float | None, *, k: float = DEFAULT_SHRINKAGE_K) -> float:
    """reliability = sqrt(n_eff / (n_eff + k))，∈ [0, 1)。"""
    n = float(n_eff) if n_eff is not None else 0.0
    n = max(0.0, n)
    k = max(float(k), 1e-9)
    if not math.isfinite(n):
        n = 0.0
    return math.sqrt(n / (n + k))


#: 旧名兼容（V3.1 R61 早期 draft 用 ``reliability`` 作为函数名，与行内局部
#: 变量同名导致 shadowing bug；改名后保留别名避免既有调用方/测试断链）。
def reliability(n_eff: int | float | None, *, k: float = DEFAULT_SHRINKAGE_K) -> float:
    """``reliability_of`` 的向后兼容别名。"""
    return reliability_of(n_eff, k=k)
    n = float(n_eff) if n_eff is not None else 0.0
    n = max(0.0, n)
    k = max(float(k), 1e-9)
    if not math.isfinite(n):
        n = 0.0
    return math.sqrt(n / (n + k))


def shrink_utility(u: float, *, reliability: float | None = None, n_eff: int | float | None = None,
                   k: float = DEFAULT_SHRINKAGE_K) -> float:
    """U_conf = 0.5 + reliability * (U - 0.5)。U 须已 clip 到 [0, 1]。"""
    u = max(0.0, min(1.0, float(u)))
    if reliability is None:
        reliability = reliability_of(n_eff, k=k)
    r = max(0.0, min(1.0, float(reliability)))
    return 0.5 + r * (u - 0.5)


def shrink_dimension(
    dimension_utility: float,
    *,
    n_eff: int | float | None,
    k: float = DEFAULT_SHRINKAGE_K,
    requirement: Any = None,
) -> float:
    """对单个维度总分的置信收缩。

    - 显式 ``n_eff`` 给定 → 标准公式（n_eff=0 → 中性 0.5）。
    - 显式 ``n_eff`` 为 None：
      * requirement 为 ``optional`` → 0.5（OPTIONAL 缺失的维度即中性）；
      * 否则（diagnostic / required 缺省）→ 不做收缩原样返回（DIAGNOSTIC
        缺失无分影响；REQUIRED 缺失由 gate 层负责，不在此惩罚）。
    """
    if n_eff is not None:
        return shrink_utility(dimension_utility, n_eff=n_eff, k=k)
    req_name = getattr(requirement, "name", None)
    if req_name == "OPTIONAL":
        return 0.5
    return max(0.0, min(1.0, float(dimension_utility)))

"""算子/机制生存统计（任务书 §47 / Phase 10）。

对 (FactorDNA, SurvivalLabel) 列表做置信度收缩归因：每个 operator 与机制
输出 support_count / raw_survival_rate / shrunk_survival_rate /
confidence_interval(Wilson) / recent_retention。

§47.1：小样本不下强结论——shrunk 向总体 prior 收缩（复用 fitness 的
survival_opportunity_shrunk 思想）。输出带 support_count，绝不把低样本
现象直接标成规则（§47 / §81）。
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from alphaprobe.contracts import FactorDNA, SurvivalLabel
from alphaprobe.survival.dna import default_operator_allowlist

__all__ = [
    "SurvivalStat",
    "operator_survival_stats",
    "mechanism_survival_stats",
    "wilson_interval",
    "SURVIVAL_LABEL_ORDER",
    "DEFAULT_SHRINKAGE_STRENGTH",
]

#: 存活标签（按存活强度降序；未列为存活）
SURVIVAL_LABEL_ORDER: tuple[SurvivalLabel, ...] = (
    SurvivalLabel.PERSISTENT_ALPHA,
    SurvivalLabel.HEALTHY,
    SurvivalLabel.RECOVERED,
)
#: §47.1 收缩强度（与 fitness.survival_opportunity_shrunk 默认一致）
DEFAULT_SHRINKAGE_STRENGTH = 20.0


@dataclass
class SurvivalStat:
    """§47.1 单个 operator/机制的输出卡片。"""

    key: str
    kind: str  # "operator" | "mechanism"
    support_count: int = 0
    raw_survival_rate: float = 0.0
    shrunk_survival_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 1.0
    recent_retention: float | None = None
    label_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "support_count": self.support_count,
            "raw_survival_rate": self.raw_survival_rate,
            "shrunk_survival_rate": self.shrunk_survival_rate,
            "confidence_interval": [self.ci_low, self.ci_high],
            "recent_retention": self.recent_retention,
            "label_distribution": dict(self.label_distribution),
        }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """标准 Wilson score interval（小样本不依赖正态近似，§47.1）。

    边界精确化：0/total 下界 0，total/total 上界 1（对称性质，
    与连续性修正无关；下界仍由标准公式给出）。
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    lo = max(0.0, (centre - half) / denom)
    hi = min(1.0, (centre + half) / denom)
    if successes == 0:
        lo = 0.0
    if successes == total:
        hi = 1.0
        # 单侧 95% 下界：z^2/(n+z^2)（Wilson 单侧，symmetric 不适用）
        lo = max(lo, (1.6449 * 1.6449) / (total + 1.6449 * 1.6449))
    return (lo, hi)


def _is_survival(label: SurvivalLabel) -> bool:
    return label in SURVIVAL_LABEL_ORDER


def _recent_retention_of(profile: Any) -> float | None:
    if profile is None:
        return None
    v = getattr(profile, "recent_retention", None)
    if v is None or not math.isfinite(float(v)):
        return None
    return max(0.0, min(1.0, float(v)))


def operator_survival_stats(
    factors: Sequence[tuple[FactorDNA, SurvivalLabel]],
    *,
    prior: float | None = None,
    shrinkage_strength: float = DEFAULT_SHRINKAGE_STRENGTH,
    recent_retention_fn: Any | None = None,
) -> list[SurvivalStat]:
    """§47.1：算子生存统计（置信度收缩）。

    Parameters
    ----------
    factors : list[(FactorDNA, SurvivalLabel)]
        已分类的因子 DNA + 标签列表。
    prior : float, optional
        总体存活率（未给时用样本内总存活率；样本为空时 0.5）。
    shrinkage_strength : float
        收缩强度（越大越保守）。
    recent_retention_fn : callable, optional
        factor_id → recent_retention 的函数（缺省不输出 recent_retention）。

    Returns
    -------
    list[SurvivalStat]
        每个出现过的 operator 一行，按 support_count 降序。
    """
    totals: Counter[str] = Counter()
    survivals: Counter[str] = Counter()
    label_dist: dict[str, Counter[str]] = defaultdict(Counter)
    retention_vals: dict[str, list[float]] = defaultdict(list)

    total_factors = len(factors)
    total_survival = sum(1 for _, lbl in factors if _is_survival(lbl))
    effective_prior = (
        (total_survival / total_factors) if (prior is None and total_factors > 0) else (prior if prior is not None else 0.5)
    )

    allow = default_operator_allowlist()
    for dna, label in factors:
        ops = getattr(dna, "operators", None) or []
        for op in ops:
            if allow is not None and op not in allow:
                continue
            totals[op] += 1
            if _is_survival(label):
                survivals[op] += 1
            label_dist[op][label.value] += 1
            if recent_retention_fn is not None:
                r = _recent_retention_of(recent_retention_fn(getattr(dna, "factor_id", "")))
                if r is not None:
                    retention_vals[op].append(r)

    out: list[SurvivalStat] = []
    for op in totals:
        support = totals[op]
        raw = survivals[op] / support
        shrunk = _shrunk_rate(raw, support, effective_prior, shrinkage_strength)
        lo, hi = wilson_interval(survivals[op], support)
        stat = SurvivalStat(
            key=op,
            kind="operator",
            support_count=support,
            raw_survival_rate=raw,
            shrunk_survival_rate=shrunk,
            ci_low=lo,
            ci_high=hi,
            label_distribution=dict(label_dist[op]),
        )
        if retention_vals[op]:
            stat.recent_retention = float(statistics.fmean(retention_vals[op]))
        out.append(stat)

    out.sort(key=lambda s: (-s.support_count, -s.raw_survival_rate))
    return out


def mechanism_survival_stats(
    factors: Sequence[tuple[FactorDNA, SurvivalLabel]],
    *,
    prior: float | None = None,
    shrinkage_strength: float = DEFAULT_SHRINKAGE_STRENGTH,
) -> list[SurvivalStat]:
    """§47.1：机制标签生存统计（与 operator 同构）。"""
    totals: Counter[str] = Counter()
    survivals: Counter[str] = Counter()
    label_dist: dict[str, Counter[str]] = defaultdict(Counter)

    total_factors = len(factors)
    total_survival = sum(1 for _, lbl in factors if _is_survival(lbl))
    effective_prior = (
        (total_survival / total_factors) if (prior is None and total_factors > 0) else (prior if prior is not None else 0.5)
    )

    for dna, label in factors:
        mechs = getattr(dna, "mechanisms", None) or []
        for m in mechs:
            totals[m] += 1
            if _is_survival(label):
                survivals[m] += 1
            label_dist[m][label.value] += 1

    out: list[SurvivalStat] = []
    for m in totals:
        support = totals[m]
        raw = survivals[m] / support
        shrunk = _shrunk_rate(raw, support, effective_prior, shrinkage_strength)
        lo, hi = wilson_interval(survivals[m], support)
        out.append(
            SurvivalStat(
                key=m,
                kind="mechanism",
                support_count=support,
                raw_survival_rate=raw,
                shrunk_survival_rate=shrunk,
                ci_low=lo,
                ci_high=hi,
                label_distribution=dict(label_dist[m]),
            )
        )
    out.sort(key=lambda s: (-s.support_count, -s.raw_survival_rate))
    return out


def _shrunk_rate(
    raw: float, support: int, prior: float, strength: float
) -> float:
    """置信度收缩：shrunk = (raw·k + prior·s) / (k + s)。"""
    k = max(0.0, float(support))
    return (raw * k + prior * strength) / (k + strength)

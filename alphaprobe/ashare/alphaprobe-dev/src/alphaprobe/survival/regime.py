"""Market Regime / Event Memory（任务书 §50 / §44 / Phase 10）。

- 注册 2026-06..2026-07 大规模衰减事件（decay_2026_jun_jul）；
- 读取侧必须过 research_protocol.survival.assert_survival_visible 的 cutoff
  闸门（§44：vN 搜索期禁用 2026 信息，只有冻结后 vN+1 可用）；
- regime_response：描述统计（哪些 family 衰减/幸存），不做朴素因果归因
  ——输出带 support_count 与 shrinkage（§47 / §81）。

纯统计/规则代码；不触碰行情数据。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from alphaprobe.contracts import FactorDNA, SurvivalLabel

__all__ = [
    "DECAY_2026_EVENT_ID",
    "Decay2026Window",
    "register_2026_event",
    "assert_regime_visible",
    "regime_response",
    "regime_family_response",
]

#: 2026 大规模衰减事件（§50：不只作为日期标签）
DECAY_2026_EVENT_ID = "decay_2026_jun_jul"
#: 事件窗口（闭区间）
Decay2026Window: tuple[str, str] = ("2026-06-01", "2026-07-31")


def register_2026_event(store: Any) -> bool:
    """向 store 注册 2026-06..2026-07 衰减事件（幂等）。

    若 store 已有 decay_2026_jun_jul 事件则跳过。store 必须提供
    ``add_regime_event(*, event_id, start_date, end_date, description, payload)``
    （即 memory.GlobalMemoryStore）。

    Returns
    -------
    bool
        True=本次新注册；False=已存在（幂等跳过）。
    """
    if hasattr(store, "regime_event_exists"):
        try:
            if store.regime_event_exists(DECAY_2026_EVENT_ID):
                return False
        except (AttributeError, NotImplementedError):
            pass
    # 兜底：GlobalMemoryStore 无查询方法 → 直接 upsert（INSERT OR REPLACE 幂等）
    store.add_regime_event(
        event_id=DECAY_2026_EVENT_ID,
        start_date=Decay2026Window[0],
        end_date=Decay2026Window[1],
        description="2026 年 6-7 月 A 股因子大规模衰减（历史已知市场结果，vN+1 才可用）",
        payload={
            "kind": "massive_decay",
            "window": list(Decay2026Window),
            "reference": "taskbook §43/§50",
        },
    )
    return True


def assert_regime_visible(*, event_date: str, cutoff: Any) -> None:
    """§44.1 闸门：event_date > cutoff 的 regime/survival 记忆禁止读取。

    直接委托 research_protocol.survival.assert_survival_visible（代码防线）。
    """
    from alphaprobe.research_protocol.survival import assert_survival_visible

    assert_survival_visible(event_date=event_date, cutoff=cutoff)


# ---------------------------------------------------------------------------
# §50 / §47 regime 响应：描述统计，不做朴素因果归因
# ---------------------------------------------------------------------------

#: 衰减标签（regime 内 broken/degrading 视为失效）
DECAYED_LABELS = (SurvivalLabel.BROKEN, SurvivalLabel.DEGRADING)


@dataclass
class RegimeFamilyResponse:
    """某个 family 在 regime 窗口内的描述统计。"""

    family: str
    total: int = 0
    decayed: int = 0
    survived: int = 0
    raw_decay_rate: float = 0.0
    shrunk_decay_rate: float = 0.0
    support_count: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "total": self.total,
            "decayed": self.decayed,
            "survived": self.survived,
            "raw_decay_rate": self.raw_decay_rate,
            "shrunk_decay_rate": self.shrunk_decay_rate,
            "support_count": self.support_count,
            "label_distribution": dict(self.label_distribution),
        }


def _shrunk_rate(raw: float, support: int, prior: float, strength: float = 20.0) -> float:
    k = max(0.0, float(support))
    return (raw * k + prior * strength) / (k + strength)


def regime_family_response(
    factors_period_perf: Sequence[tuple[FactorDNA, SurvivalLabel]],
    *,
    prior_decay_rate: float | None = None,
    shrinkage_strength: float = 20.0,
) -> list[RegimeFamilyResponse]:
    """按 family 聚合 regime 窗口内的存活/衰减描述统计。

    Parameters
    ----------
    factors_period_perf : list[(FactorDNA, SurvivalLabel)]
        regime 窗口内每个因子的 DNA + 生存标签。
    prior_decay_rate : float, optional
        总体衰减率先验（未给时用样本内总体衰减率）。
    shrinkage_strength : float
        收缩强度。

    Returns
    -------
    list[RegimeFamilyResponse]
        每个出现过的 family 一行，按 support_count 降序。
    """
    totals: Counter[str] = Counter()
    decayed: Counter[str] = Counter()
    label_dist: dict[str, Counter[str]] = defaultdict(Counter)

    total_factors = len(factors_period_perf)
    total_decayed = sum(1 for _, lbl in factors_period_perf if lbl in DECAYED_LABELS)
    prior = (
        (total_decayed / total_factors) if (prior_decay_rate is None and total_factors > 0) else (prior_decay_rate if prior_decay_rate is not None else 0.5)
    )

    for dna, label in factors_period_perf:
        families = getattr(dna, "field_families", None) or []
        for fam in families:
            totals[fam] += 1
            if label in DECAYED_LABELS:
                decayed[fam] += 1
            label_dist[fam][label.value] += 1

    out: list[RegimeFamilyResponse] = []
    for fam in totals:
        support = totals[fam]
        raw = decayed[fam] / support
        shrunk = _shrunk_rate(raw, support, prior, shrinkage_strength)
        out.append(
            RegimeFamilyResponse(
                family=fam,
                total=support,
                decayed=decayed[fam],
                survived=support - decayed[fam],
                raw_decay_rate=raw,
                shrunk_decay_rate=shrunk,
                support_count=support,
                label_distribution=dict(label_dist[fam]),
            )
        )
    out.sort(key=lambda r: (-r.support_count, -r.raw_decay_rate))
    return out


def regime_response(
    event: dict[str, Any],
    factors_period_perf: Sequence[tuple[FactorDNA, SurvivalLabel]],
    *,
    cutoff: Any = None,
    prior_decay_rate: float | None = None,
    shrinkage_strength: float = 20.0,
) -> dict[str, Any]:
    """§50：regime 窗口内哪些 family 衰减/幸存（描述统计，不做因果归因）。

    Parameters
    ----------
    event : dict
        事件描述（含 start_date / end_date / event_id，来自 store 或直接构造）。
    factors_period_perf : list[(FactorDNA, SurvivalLabel)]
        regime 窗口内的因子表现（DNA + 标签）。
    cutoff : optional
        知识截止日。若给定且 event 窗口落在 cutoff 之后，直接抛
        SealedTestViolation（读取侧必须过闸门，§44.1）。

    Returns
    -------
    dict
        - event_id / window；
        - families：按 family 的 RegimeFamilyResponse 描述统计；
        - summary：总体 support / 衰减率 / 收缩衰减率 / 幸存率；
        - causal_note：明确「描述统计，不做朴素因果归因」。
    """
    start = str(event.get("start_date") or event.get("window", [None, None])[0])
    end = str(event.get("end_date") or event.get("window", [None, None])[1])
    if cutoff is not None:
        if not start or not end:
            raise ValueError("event missing start_date/end_date; cannot gate")
        # 事件窗口内任一日期 > cutoff 即整体不可见
        assert_regime_visible(event_date=end, cutoff=cutoff)

    families = regime_family_response(
        factors_period_perf,
        prior_decay_rate=prior_decay_rate,
        shrinkage_strength=shrinkage_strength,
    )

    total = len(factors_period_perf)
    decayed = sum(1 for _, lbl in factors_period_perf if lbl in DECAYED_LABELS)
    survived = total - decayed
    raw_rate = (decayed / total) if total else 0.0
    prior = (
        raw_rate
        if prior_decay_rate is None
        else prior_decay_rate
    )
    shrunk = _shrunk_rate(raw_rate, total, prior, shrinkage_strength)

    return {
        "event_id": event.get("event_id", "?"),
        "window": [start, end],
        "families": [f.to_dict() for f in families],
        "summary": {
            "total_support": total,
            "decayed": decayed,
            "survived": survived,
            "raw_decay_rate": raw_rate,
            "shrunk_decay_rate": shrunk,
            "survival_rate": 1.0 - raw_rate,
        },
        "causal_note": (
            "描述统计：输出带 support_count 与 shrinkage，不做朴素因果归因"
            "（§47/§81：禁止把低样本现象直接标成规则）。"
        ),
    }

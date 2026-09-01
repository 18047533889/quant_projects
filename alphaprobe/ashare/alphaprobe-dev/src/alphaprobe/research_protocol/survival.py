"""§88.4 / §44.1：survival memory cutoff 闸门（代码防线，不靠 prompt 自觉）。"""

from __future__ import annotations

from datetime import date

from alphaprobe.research_protocol import SealedTestViolation


def assert_survival_visible(*, event_date: str, cutoff) -> None:
    """event_date > cutoff 的 survival/regime 记忆对当前 run 不可见。

    cutoff 为 ExperimentContext.knowledge_cutoff_date 或 survival_memory_cutoff
    （date）。由代码强制（§44.1），不靠 prompt 自觉。
    """
    if cutoff is None:
        # 未配置 cutoff：视为全不可信，拒绝读取 survival 记忆
        raise SealedTestViolation(
            "survival read denied: survival_cutoff/knowledge_cutoff not configured"
        )
    from datetime import datetime

    if isinstance(cutoff, datetime):
        cutoff = cutoff.date()
    if isinstance(cutoff, date):
        cut = cutoff.isoformat()
    else:
        cut = str(cutoff)[:10]
    ev = str(event_date)[:10]
    if ev > cut:
        raise SealedTestViolation(
            f"survival memory {ev} is post-cutoff ({cut}): forbidden for current run"
        )


def filter_survival_rows(rows: list[dict], cutoff) -> list[dict]:
    """按 cutoff 过滤 survival/period-performance 行（event_date 字段）。"""
    out = []
    for r in rows:
        ev = r.get("event_date") or r.get("date") or r.get("period_end")
        try:
            assert_survival_visible(event_date=ev, cutoff=cutoff)
        except SealedTestViolation:
            continue
        out.append(r)
    return out
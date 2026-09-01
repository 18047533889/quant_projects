"""冷启动抽样：结构去参、结构多样性。"""

from __future__ import annotations

import random
import re
from typing import Protocol, Sequence, TypeVar

_NUM_RE = re.compile(r"\b\d+\.?\d*\b")


class _ExprEntry(Protocol):
    expr: str


T = TypeVar("T", bound=_ExprEntry)


def norm_expr_structure(expr: str) -> str:
    """将数字参数归一为 {P}，用于判断「公式形状」是否相同。"""
    e = " ".join(str(expr).split())
    return _NUM_RE.sub("{P}", e)


def group_by_structure(entries: Sequence[T]) -> dict[str, list[T]]:
    groups: dict[str, list[T]] = {}
    for e in entries:
        key = norm_expr_structure(e.expr)
        groups.setdefault(key, []).append(e)
    return groups


def sample_structurally_diverse_entries(
    entries: Sequence[T],
    n: int,
    *,
    seed: int | None = None,
) -> list[T]:
    """每批抽样中，同一公式形状（仅换数字参数）只取一条。"""
    if n <= 0:
        return []
    pool = list(entries)
    if n >= len(pool):
        return pool

    rng = random.Random(seed)
    groups = group_by_structure(pool)
    keys = list(groups.keys())
    rng.shuffle(keys)

    chosen: list[T] = []
    chosen_exprs: set[str] = set()

    for key in keys:
        if len(chosen) >= n:
            break
        candidates = [c for c in groups[key] if c.expr not in chosen_exprs]
        if not candidates:
            continue
        pick = rng.choice(candidates)
        chosen.append(pick)
        chosen_exprs.add(pick.expr)

    if len(chosen) < n:
        remaining = [e for e in pool if e.expr not in chosen_exprs]
        rng.shuffle(remaining)
        for e in remaining:
            if len(chosen) >= n:
                break
            chosen.append(e)
            chosen_exprs.add(e.expr)

    return chosen[:n]

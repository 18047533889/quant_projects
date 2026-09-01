"""Train-only orientation（任务书 §10）。

Train RankIC >= 0 → oriented = f（orientation=+1）
Train RankIC < 0 → oriented = -f（orientation=-1）
Valid/Test 禁止重新翻方向：负 RankIC 是 instability，不是再乘 -1。
f 与 -f 同一 SignalEquivalenceID，orientation 不同。
"""

from __future__ import annotations

import math


def orientation_from_train_rankic(train_rankic: float | None) -> int:
    """§10：只允许 Train 决定方向。None/NaN → +1（不翻转）。"""
    if train_rankic is None or not math.isfinite(train_rankic):
        return 1
    return 1 if train_rankic >= 0 else -1


def apply_orientation(value: float | None, orientation: int) -> float | None:
    """raw metric × orientation → oriented metric（§10.2 统一 raw+orientation+oriented）。"""
    if value is None:
        return None
    return value * (1 if orientation >= 0 else -1)


def forbid_reorientation(valid_or_test_rankic: float | None) -> None:
    """Valid/Test 出现负 RankIC 时禁止翻方向的显式断言点。

    返回 None；调用方若想翻方向必须改为记录 instability。
    保留此函数作为 grep 目标：代码中出现 orientation=-1 的合法来源
    只允许 orientation_from_train_rankic。
    """
    return None


def signal_equivalence_key(canonical_ast_hash: str) -> str:
    """f 与 -f、0-f、(-1)*f 同一 SignalEquivalenceID（sign-invariant，§11.2）。

    canonical 层在 dedup 模块完成 -x/(-1)*x 归一后 hash 即 sign 归一；
    此处再兜底剥离一层前导负号标记（canonical_ast_hash 已 sign-normalized
    时两者本来就相同）。
    """
    h = canonical_ast_hash
    # dedup.canonical 生成时 -x 与 x 已归一到相同 hash；此处仅防御旧格式
    if h.startswith("neg:"):
        h = h[4:]
    return h
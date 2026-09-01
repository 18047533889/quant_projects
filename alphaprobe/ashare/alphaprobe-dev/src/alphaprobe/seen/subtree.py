"""subtree_stats（§15/§40）。

subtree_hash PK + canonical_subtree + appearance_count + unique_factor_count +
successful_factor_count + elite_factor_count + survival_rate(惰性计算列存分子分母) +
last_seen_at。提供 saturation(subtree_hash) -> float（soft guidance 值，不做 hard reject）。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.seen.store import SeenStore


class SubtreeStats:
    """§15/§40 子树统计管理。"""

    def __init__(self, store: SeenStore) -> None:
        self._store = store

    def record(
        self,
        subtree_hash: str,
        canonical_subtree: str | None,
        factor_pk: int,
        *,
        success: bool = False,
        elite: bool = False,
    ) -> None:
        import time
        self._store.upsert_subtree_stat(
            subtree_hash, canonical_subtree, factor_pk,
            success=success, elite=elite, now=time.time(),
        )

    def get(self, subtree_hash: str) -> dict[str, Any] | None:
        return self._store.get_subtree_stat(subtree_hash)

    def saturation(self, subtree_hash: str) -> float:
        """§40 saturation = 1 - (successful / max(unique, 1))，soft guidance。

        返回 0~1 的浮点数：0=没饱和，1=完全饱和。
        不做 hard reject。
        """
        stat = self.get(subtree_hash)
        if stat is None:
            return 0.0
        total = int(stat.get("unique_factor_count", 0))
        if total == 0:
            return 0.0
        successful = int(stat.get("successful_factor_count", 0))
        # 分母选 unique_factor_count 而非 appearance_count：同因子多次出现不算
        return 1.0 - successful / total

    def survival_rate(self, subtree_hash: str) -> float | None:
        """惰性计算 survival_rate = successful / success_denominator。

        返回 None 当分母为 0。
        """
        stat = self.get(subtree_hash)
        if stat is None:
            return None
        denom = int(stat.get("success_denominator", 0))
        if denom == 0:
            return None
        successful = int(stat.get("successful_factor_count", 0))
        return successful / denom
"""factor_families / factor_family_members（§20）。

family_id PK + family_template + tested_parameter_ranges(json) + member_count +
best_factor_id + best_search_fitness + historical_success_rate + saturation。
register_evaluation 时若 metrics 里有 search_fitness 则更新 best_*。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.seen.store import SeenStore


class FamilyRegistry:
    """§20 参数族注册表。"""

    def __init__(self, store: SeenStore) -> None:
        self._store = store

    def register_member(
        self,
        family_id: str,
        *,
        factor_id: str,
        factor_pk: int | None = None,
        family_template: str | None = None,
        parameter_fingerprint: str | None = None,
        search_fitness: float | None = None,
        outcome: str | None = None,
    ) -> None:
        if factor_pk is None:
            factor_pk = self._store.factor_pk_by_id(factor_id)
        if factor_pk is None:
            raise ValueError(f"factor_id not found: {factor_id}")
        self._store.upsert_family(
            family_id=family_id,
            family_template=family_template,
            factor_pk=factor_pk,
            factor_id=factor_id,
            parameter_fingerprint=parameter_fingerprint,
            search_fitness=search_fitness,
            outcome=outcome,
        )

    def get(self, family_id: str) -> dict[str, Any] | None:
        return self._store.get_family(family_id)

    def saturation(self, family_id: str) -> float:
        """§20 族饱和建议值（soft guidance，不做 hard reject）。

        member_count 越多、成功率越低 → 越饱和。
        """
        fam = self.get(family_id)
        if fam is None:
            return 0.0
        member_count = int(fam.get("member_count", 0))
        if member_count == 0:
            return 0.0
        success_rate = fam.get("historical_success_rate")
        if success_rate is None:
            return min(member_count / 50.0, 1.0)
        return min(max(1.0 - float(success_rate), 0.0) + min(member_count / 100.0, 0.5), 1.0)

    def best_member(self, family_id: str) -> dict[str, Any] | None:
        fam = self.get(family_id)
        if fam is None:
            return None
        return {
            "best_factor_id": fam.get("best_factor_id"),
            "best_search_fitness": fam.get("best_search_fitness"),
            "member_count": fam.get("member_count"),
            "historical_success_rate": fam.get("historical_success_rate"),
            "saturation": self.saturation(family_id),
        }
"""Generator scorecard: aggregate quality / cost metrics per generator.

Pure standard-library module (``dataclasses``, ``typing``).

A :class:`GeneratorScorecard` aggregates every candidate a generator produced
through the firewall + admission pipeline and summarizes:

- volume & pass rates per firewall stage,
- exact- vs behavior-duplicate rates,
- admission rate, novelty, incremental IC,
- compute / wall-clock / API-token cost per admitted factor,
- family diversity.

The scorecard is fed incrementally via :meth:`record` and summarized lazily by
:meth:`summary`. It is generator-agnostic and depends only on this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .firewall import FirewallResult


@dataclass
class GeneratorScorecard:
    """Running aggregate for one generator across many candidates."""

    generator_type: str = ""
    generator_version: str = ""

    # --- raw accumulation ---
    proposal_count: int = 0
    admitted_count: int = 0
    _stage_pass: Dict[str, int] = field(default_factory=dict)
    _stage_total: Dict[str, int] = field(default_factory=dict)
    exact_duplicate_count: int = 0
    behavior_duplicate_count: int = 0
    novelty_scores: List[float] = field(default_factory=list)
    incremental_ics: List[float] = field(default_factory=list)
    compute_costs: List[float] = field(default_factory=list)
    wall_clock_ms: List[float] = field(default_factory=list)
    api_token_costs: List[float] = field(default_factory=list)
    families: Dict[str, int] = field(default_factory=dict)
    l0_l3: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        results: Optional[List[FirewallResult]] = None,
        passed: bool = False,
        admitted: bool = False,
        novelty: Optional[float] = None,
        incremental_ic: Optional[float] = None,
        compute_cost: Optional[float] = None,
        wall_clock_ms: Optional[float] = None,
        api_token_cost: Optional[float] = None,
        family: Optional[str] = None,
        stage_names: Optional[List[str]] = None,
    ) -> None:
        """Record one candidate's passage through the firewall + admission."""
        self.proposal_count += 1
        stages = stage_names or (
            [r.stage_name for r in results] if results else []
        )
        for name in stages:
            self._stage_total[name] = self._stage_total.get(name, 0) + 1
        if results:
            for r in results:
                if r.passed:
                    self._stage_pass[r.stage_name] = (
                        self._stage_pass.get(r.stage_name, 0) + 1
                    )

        # L0..L3 pass counts (bucket all stages into coarse levels).
        for name in stages:
            self.l0_l3[name] = self.l0_l3.get(name, 0) + 1

        if passed:
            for name in stages:
                self.l0_l3[name] = self.l0_l3.get(name, 0) + 1
        if novelty is not None:            self.novelty_scores.append(novelty)
        if incremental_ic is not None:
            self.incremental_ics.append(incremental_ic)
        if compute_cost is not None:
            self.compute_costs.append(compute_cost)
        if wall_clock_ms is not None:
            self.wall_clock_ms.append(wall_clock_ms)
        if api_token_cost is not None:
            self.api_token_costs.append(api_token_cost)
        if family:
            self.families[family] = self.families.get(family, 0) + 1
        if admitted:
            self.admitted_count += 1

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        n = self.proposal_count or 1
        admitted = self.admitted_count

        def rate(dividend: int) -> float:
            return dividend / n if n else 0.0

        return {
            "generator_type": self.generator_type,
            "proposal_count": self.proposal_count,
            "legality_pass_rate": self._stage_pass_rate("grammar_legality"),
            "compile_pass_rate": self._stage_pass_rate("fe_compilation"),
            "exact_duplicate_rate": self.exact_duplicate_count / n
            if n else 0.0,
            "behavior_duplicate_rate": self.behavior_duplicate_count / n
            if n else 0.0,
            "l0_l3_pass_rate": {
                k: (self.l0_l3.get(k, 0) / n) if n else 0.0
                for k in ("l0", "l1", "l2", "l3")
            },
            "final_admission_rate": admitted / n if n else 0.0,
            "mean_novelty": _mean(self.novelty_scores),
            "mean_incremental_ic": _mean(self.incremental_ics),
            "compute_cost_per_admitted": (
                (sum(self.compute_costs) / admitted)
                if admitted else 0.0
            ),
            "wall_clock_cost": _mean(self.wall_clock_ms),
            "api_token_cost": _mean(self.api_token_costs),
            "family_diversity": len(self.families),
        }

    def _stage_pass_rate(self, stage: str) -> float:
        total = self._stage_total.get(stage, 0)
        if not total:
            return 0.0
        return self._stage_pass.get(stage, 0) / total


def _mean(xs: List[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0

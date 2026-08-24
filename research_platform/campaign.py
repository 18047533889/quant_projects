"""Research campaign bookkeeping and multiple-testing ledger.

Pure standard-library module (``dataclasses``, ``hashlib``, ``typing``,
``datetime``).

Purpose
-------
A *research campaign* is a batch of candidate factors generated under one
research question with one generator policy. The ``TrialLedger`` records every
trial (candidate) and lets us compute the multiple-testing burden: how many
candidates were tried, how many were admitted, and how the admissions break
down by generator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .artifacts import Artifact, _dt_to_str, _normalize_dt, utcnow


@dataclass(frozen=True)
class ResearchCampaignArtifact(Artifact):
    """A named research campaign over candidate factors."""

    campaign_id: str = ""
    research_question: str = ""
    generator_policy: str = ""
    candidate_count: int = 0
    all_trial_ids: tuple = ()
    failed_trials: tuple = ()
    admitted_trial_ids: tuple = ()
    created_at: Optional[datetime] = None

    @property
    def admission_rate(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        return len(self.admitted_trial_ids) / self.candidate_count


@dataclass
class TrialLedger:
    """Mutable running ledger of all trials across one or more campaigns.

    The ledger owns the *counts*; a frozen :class:`ResearchCampaignArtifact`
    is a snapshot export of a subset of those counts.
    """

    # trial_id -> (generator_type, admitted: bool, stage: str)
    trials: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # campaign_id -> ResearchCampaignArtifact (last exported snapshot)
    campaigns: Dict[str, ResearchCampaignArtifact] = field(default_factory=dict)

    def append(
        self,
        trial_id: str,
        generator_type: str,
        admitted: bool,
        stage: str = "",
    ) -> None:
        """Record a single trial outcome."""
        if trial_id in self.trials:
            raise ValueError(f"trial {trial_id!r} already recorded")
        self.trials[trial_id] = {
            "generator_type": generator_type,
            "admitted": admitted,
            "stage": stage,
        }

    def export_campaign(self, campaign_id: str) -> ResearchCampaignArtifact:
        """Snapshot current ledger state into a campaign artifact."""
        ids = list(self.trials.keys())
        failed = [
            tid for tid, r in self.trials.items() if not r["admitted"]
        ]
        admitted = [
            tid for tid, r in self.trials.items() if r["admitted"]
        ]
        art = ResearchCampaignArtifact(
            artifact_id=f"campaign-{campaign_id}",
            artifact_type="research_campaign",
            campaign_id=campaign_id,
            research_question="",
            generator_policy="",
            candidate_count=len(ids),
            all_trial_ids=tuple(ids),
            failed_trials=tuple(failed),
            admitted_trial_ids=tuple(admitted),
            created_at=utcnow(),
        ).with_hash()
        self.campaigns[campaign_id] = art
        return art

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @property
    def multiple_testing_count(self) -> int:
        """Number of trials, i.e. the multiple-testing multiplicity."""
        return len(self.trials)

    @property
    def admission_rate(self) -> float:
        if not self.trials:
            return 0.0
        admitted = sum(1 for r in self.trials.values() if r["admitted"])
        return admitted / len(self.trials)

    def generator_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Per-generator trial/admit counts."""
        out: Dict[str, Dict[str, int]] = {}
        for r in self.trials.values():
            g = r["generator_type"]
            bucket = out.setdefault(g, {"trials": 0, "admitted": 0})
            bucket["trials"] += 1
            if r["admitted"]:
                bucket["admitted"] += 1
        return out

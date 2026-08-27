"""
Campaign coordinator for optimization lifecycle management.

Manages campaign state, budget tracking, and plateau-based stopping.

DLIB-FA-002: the search-control fields (``max_evaluations``, ``cost_budget``,
``plateau_patience``, ``target_metric``, ``should_stop``) belong to FO
SearchSession/Budget or QuantPlatform Workflow — NOT to FA. FA's campaign is a
RESEARCH-campaign governance-metadata record only (campaign id / generator /
factor assets / admission outcomes / library outcomes / statistics refs).

The legacy ``CampaignCoordinator`` / ``CampaignConfig`` / ``CampaignBudget`` /
``Campaign`` types below are retained for compatibility (§108) and are
RESEARCH_ONLY. New code should use :class:`ResearchCampaignGovernance`, which
carries governance metadata only and never search control.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CampaignState(Enum):
    """Campaign lifecycle states."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ResearchCampaignGovernance:
    """RESEARCH-campaign governance metadata only (DLIB-FA-002).

    Carries the campaign identity and the governance/outcome references that
    FA legitimately owns. It deliberately has NO search-control fields
    (no max_evaluations / cost_budget / plateau_patience / target_metric /
    should_stop) — those belong to FO SearchSession/Budget or QuantPlatform
    Workflow. FA records the campaign as a governance artifact, not as an
    optimization search controller.
    """

    campaign_id: str
    generator: Optional[str] = None          # e.g. "llm", "search", "manual"
    factor_asset_refs: tuple[str, ...] = ()  # factor assets produced/consumed
    admission_outcome_refs: tuple[str, ...] = ()
    library_outcome_refs: tuple[str, ...] = ()
    statistics_refs: tuple[str, ...] = ()
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        object.__setattr__(self, "factor_asset_refs", tuple(self.factor_asset_refs))
        object.__setattr__(self, "admission_outcome_refs", tuple(self.admission_outcome_refs))
        object.__setattr__(self, "library_outcome_refs", tuple(self.library_outcome_refs))
        object.__setattr__(self, "statistics_refs", tuple(self.statistics_refs))
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "generator": self.generator,
            "factor_asset_refs": list(self.factor_asset_refs),
            "admission_outcome_refs": list(self.admission_outcome_refs),
            "library_outcome_refs": list(self.library_outcome_refs),
            "statistics_refs": list(self.statistics_refs),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class CampaignConfig:
    """
    Campaign configuration.

    RESEARCH_ONLY (DLIB-FA-002): the search-control fields below belong to FO
    SearchSession/Budget. Retained for compatibility only.

    Attributes:
        campaign_id: Unique campaign identifier
        objective: Optimization objective
        max_evaluations: Maximum number of evaluations
        max_duration_s: Maximum duration in seconds
        initial_population: Initial population size
        plateau_patience: Generations without improvement before stopping
        target_metric_value: Target metric value for early stopping
    """

    campaign_id: str
    objective: str
    max_evaluations: int = 1000
    max_duration_s: int = 86400
    initial_population: int = 20
    plateau_patience: int = 10
    target_metric_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "objective": self.objective,
            "max_evaluations": self.max_evaluations,
            "max_duration_s": self.max_duration_s,
            "initial_population": self.initial_population,
            "plateau_patience": self.plateau_patience,
            "target_metric_value": self.target_metric_value,
        }


@dataclass
class CampaignBudget:
    """
    Campaign budget tracking.

    Attributes:
        max_evaluations: Maximum evaluations allowed
        evaluations_used: Evaluations consumed
        max_duration_s: Maximum duration in seconds
        elapsed_s: Elapsed time in seconds
        cost_budget: Cost budget (in L0 units)
        cost_used: Cost consumed
    """

    max_evaluations: int
    evaluations_used: int = 0
    max_duration_s: int = 86400
    elapsed_s: int = 0
    cost_budget: float = float('inf')
    cost_used: float = 0.0

    def is_exhausted(self) -> bool:
        """Check if any budget constraint is exhausted."""
        return (
            self.evaluations_used >= self.max_evaluations
            or self.elapsed_s >= self.max_duration_s
            or self.cost_used >= self.cost_budget
        )

    def remaining_evaluations(self) -> int:
        """Get remaining evaluation budget."""
        return max(0, self.max_evaluations - self.evaluations_used)

    def remaining_cost(self) -> float:
        """Get remaining cost budget."""
        return max(0.0, self.cost_budget - self.cost_used)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_evaluations": self.max_evaluations,
            "evaluations_used": self.evaluations_used,
            "max_duration_s": self.max_duration_s,
            "elapsed_s": self.elapsed_s,
            "cost_budget": self.cost_budget,
            "cost_used": self.cost_used,
        }


@dataclass
class Campaign:
    """
    Optimization campaign.

    Attributes:
        config: Campaign configuration
        state: Current state
        budget: Budget tracking
        best_metric: Best metric value observed
        generations_without_improvement: Generations without improvement
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        metadata: Additional metadata
    """

    config: CampaignConfig
    state: CampaignState = CampaignState.CREATED
    budget: Optional[CampaignBudget] = None
    best_metric: float = float('-inf')
    generations_without_improvement: int = 0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.budget is None:
            self.budget = CampaignBudget(
                max_evaluations=self.config.max_evaluations,
                max_duration_s=self.config.max_duration_s,
            )
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": self.config.to_dict(),
            "state": self.state.value,
            "budget": self.budget.to_dict() if self.budget else None,
            "best_metric": self.best_metric,
            "generations_without_improvement": self.generations_without_improvement,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": dict(self.metadata),
        }


class CampaignCoordinator:
    """
    Coordinate campaign lifecycle.

    Manages state transitions, budget tracking, and stopping criteria.
    """

    def __init__(self):
        self.campaigns: Dict[str, Campaign] = {}

    def create_campaign(self, config: CampaignConfig) -> Campaign:
        """Create a new campaign."""
        if config.campaign_id in self.campaigns:
            raise ValueError(f"Campaign {config.campaign_id} already exists")

        campaign = Campaign(config=config)
        self.campaigns[config.campaign_id] = campaign
        return campaign

    def start_campaign(self, campaign_id: str) -> Campaign:
        """Start a campaign."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.state != CampaignState.CREATED:
            raise ValueError(f"Campaign {campaign_id} cannot be started from state {campaign.state.value}")

        campaign.state = CampaignState.ACTIVE
        campaign.started_at = datetime.utcnow()
        return campaign

    def pause_campaign(self, campaign_id: str) -> Campaign:
        """Pause a campaign."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.state != CampaignState.ACTIVE:
            raise ValueError(f"Campaign {campaign_id} cannot be paused from state {campaign.state.value}")

        campaign.state = CampaignState.PAUSED
        return campaign

    def resume_campaign(self, campaign_id: str) -> Campaign:
        """Resume a paused campaign."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.state != CampaignState.PAUSED:
            raise ValueError(f"Campaign {campaign_id} cannot be resumed from state {campaign.state.value}")

        campaign.state = CampaignState.ACTIVE
        return campaign

    def complete_campaign(self, campaign_id: str, reason: str = "") -> Campaign:
        """Complete a campaign."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign.state = CampaignState.COMPLETED
        campaign.completed_at = datetime.utcnow()
        campaign.metadata["completion_reason"] = reason
        return campaign

    def cancel_campaign(self, campaign_id: str, reason: str = "") -> Campaign:
        """Cancel a campaign."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign.state = CampaignState.CANCELLED
        campaign.completed_at = datetime.utcnow()
        campaign.metadata["cancellation_reason"] = reason
        return campaign

    def update_metric(self, campaign_id: str, metric_value: float) -> bool:
        """
        Update campaign with new metric value.

        Returns True if improvement detected.
        """
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if metric_value > campaign.best_metric:
            campaign.best_metric = metric_value
            campaign.generations_without_improvement = 0
            return True
        else:
            campaign.generations_without_improvement += 1
            return False

    def should_stop(self, campaign_id: str) -> bool:
        """Check if campaign should stop based on criteria."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Check budget exhaustion
        if campaign.budget and campaign.budget.is_exhausted():
            return True

        # Check plateau patience
        if campaign.generations_without_improvement >= campaign.config.plateau_patience:
            return True

        # Check target metric reached
        if (campaign.config.target_metric_value is not None
                and campaign.best_metric >= campaign.config.target_metric_value):
            return True

        return False

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID."""
        return self.campaigns.get(campaign_id)

    def list_campaigns(self, state: Optional[CampaignState] = None) -> List[Campaign]:
        """List campaigns, optionally filtered by state."""
        campaigns = list(self.campaigns.values())
        if state:
            campaigns = [c for c in campaigns if c.state == state]
        return campaigns

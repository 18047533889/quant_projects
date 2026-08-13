"""Immutable event types for research control ledger."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass(frozen=True)
class CampaignEvent:
    """Represents a research campaign lifecycle event."""

    event_id: str = field(default_factory=lambda: uuid4().hex)
    campaign_id: str = field(default="")
    event_type: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        if not self.event_type:
            raise ValueError("event_type is required")
        if self.event_type not in {"created", "started", "paused", "resumed", "completed", "cancelled"}:
            raise ValueError(f"Invalid campaign event_type: {self.event_type}")


@dataclass(frozen=True)
class TrialEvent:
    """Represents a trial execution event."""

    event_id: str = field(default_factory=lambda: uuid4().hex)
    trial_id: str = field(default="")
    campaign_id: str = field(default="")
    event_type: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.trial_id:
            raise ValueError("trial_id is required")
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        if not self.event_type:
            raise ValueError("event_type is required")
        if self.event_type not in {"submitted", "running", "completed", "failed", "timeout"}:
            raise ValueError(f"Invalid trial event_type: {self.event_type}")


@dataclass(frozen=True)
class DecisionEvent:
    """Represents a decision point in research workflow."""

    event_id: str = field(default_factory=lambda: uuid4().hex)
    decision_id: str = field(default="")
    campaign_id: str = field(default="")
    decision_type: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    outcome: Optional[str] = None
    rationale: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.decision_id:
            raise ValueError("decision_id is required")
        if not self.campaign_id:
            raise ValueError("campaign_id is required")
        if not self.decision_type:
            raise ValueError("decision_type is required")
        if self.decision_type not in {"early_stop", "parameter_adjust", "resource_allocation", "promote_to_production"}:
            raise ValueError(f"Invalid decision decision_type: {self.decision_type}")
        if self.outcome and self.outcome not in {"approved", "rejected", "deferred"}:
            raise ValueError(f"Invalid outcome: {self.outcome}")

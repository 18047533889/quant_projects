"""Tests for research_control event types."""

import pytest
from datetime import datetime
from research_control.events import CampaignEvent, TrialEvent, DecisionEvent


class TestCampaignEvent:
    def test_create_valid_campaign_event(self):
        event = CampaignEvent(
            campaign_id="camp_001",
            event_type="created",
            metadata={"user": "researcher_a"}
        )
        assert event.campaign_id == "camp_001"
        assert event.event_type == "created"
        assert event.event_id
        assert isinstance(event.timestamp, datetime)
        assert event.metadata["user"] == "researcher_a"

    def test_immutable_campaign_event(self):
        event = CampaignEvent(campaign_id="camp_001", event_type="created")
        with pytest.raises(Exception):
            event.campaign_id = "camp_002"

    def test_campaign_event_requires_campaign_id(self):
        with pytest.raises(ValueError, match="campaign_id is required"):
            CampaignEvent(event_type="created")

    def test_campaign_event_requires_event_type(self):
        with pytest.raises(ValueError, match="event_type is required"):
            CampaignEvent(campaign_id="camp_001")

    def test_campaign_event_validates_event_type(self):
        with pytest.raises(ValueError, match="Invalid campaign event_type"):
            CampaignEvent(campaign_id="camp_001", event_type="invalid")

    def test_campaign_event_all_valid_types(self):
        valid_types = ["created", "started", "paused", "resumed", "completed", "cancelled"]
        for event_type in valid_types:
            event = CampaignEvent(campaign_id="camp_001", event_type=event_type)
            assert event.event_type == event_type

    def test_unique_event_ids(self):
        e1 = CampaignEvent(campaign_id="camp_001", event_type="created")
        e2 = CampaignEvent(campaign_id="camp_001", event_type="started")
        assert e1.event_id != e2.event_id


class TestTrialEvent:
    def test_create_valid_trial_event(self):
        event = TrialEvent(
            trial_id="trial_001",
            campaign_id="camp_001",
            event_type="submitted",
            parameters={"alpha": 0.01, "lookback": 20},
            metrics={"sharpe": 1.5, "returns": 0.12}
        )
        assert event.trial_id == "trial_001"
        assert event.campaign_id == "camp_001"
        assert event.event_type == "submitted"
        assert event.parameters["alpha"] == 0.01
        assert event.metrics["sharpe"] == 1.5

    def test_trial_event_requires_trial_id(self):
        with pytest.raises(ValueError, match="trial_id is required"):
            TrialEvent(campaign_id="camp_001", event_type="submitted")

    def test_trial_event_requires_campaign_id(self):
        with pytest.raises(ValueError, match="campaign_id is required"):
            TrialEvent(trial_id="trial_001", event_type="submitted")

    def test_trial_event_validates_event_type(self):
        with pytest.raises(ValueError, match="Invalid trial event_type"):
            TrialEvent(
                trial_id="trial_001",
                campaign_id="camp_001",
                event_type="invalid"
            )

    def test_trial_event_all_valid_types(self):
        valid_types = ["submitted", "running", "completed", "failed", "timeout"]
        for event_type in valid_types:
            event = TrialEvent(
                trial_id="trial_001",
                campaign_id="camp_001",
                event_type=event_type
            )
            assert event.event_type == event_type


class TestDecisionEvent:
    def test_create_valid_decision_event(self):
        event = DecisionEvent(
            decision_id="dec_001",
            campaign_id="camp_001",
            decision_type="early_stop",
            outcome="approved",
            rationale="Poor performance across all trials",
            context={"trials_completed": 50, "best_sharpe": 0.3}
        )
        assert event.decision_id == "dec_001"
        assert event.campaign_id == "camp_001"
        assert event.decision_type == "early_stop"
        assert event.outcome == "approved"
        assert event.rationale == "Poor performance across all trials"

    def test_decision_event_requires_decision_id(self):
        with pytest.raises(ValueError, match="decision_id is required"):
            DecisionEvent(campaign_id="camp_001", decision_type="early_stop")

    def test_decision_event_requires_campaign_id(self):
        with pytest.raises(ValueError, match="campaign_id is required"):
            DecisionEvent(decision_id="dec_001", decision_type="early_stop")

    def test_decision_event_validates_decision_type(self):
        with pytest.raises(ValueError, match="Invalid decision decision_type"):
            DecisionEvent(
                decision_id="dec_001",
                campaign_id="camp_001",
                decision_type="invalid"
            )

    def test_decision_event_validates_outcome(self):
        with pytest.raises(ValueError, match="Invalid outcome"):
            DecisionEvent(
                decision_id="dec_001",
                campaign_id="camp_001",
                decision_type="early_stop",
                outcome="invalid"
            )

    def test_decision_event_all_valid_types(self):
        valid_types = ["early_stop", "parameter_adjust", "resource_allocation", "promote_to_production"]
        for decision_type in valid_types:
            event = DecisionEvent(
                decision_id="dec_001",
                campaign_id="camp_001",
                decision_type=decision_type
            )
            assert event.decision_type == decision_type

    def test_decision_event_all_valid_outcomes(self):
        valid_outcomes = ["approved", "rejected", "deferred"]
        for outcome in valid_outcomes:
            event = DecisionEvent(
                decision_id="dec_001",
                campaign_id="camp_001",
                decision_type="early_stop",
                outcome=outcome
            )
            assert event.outcome == outcome

    def test_decision_event_optional_outcome(self):
        event = DecisionEvent(
            decision_id="dec_001",
            campaign_id="camp_001",
            decision_type="early_stop"
        )
        assert event.outcome is None
        assert event.rationale is None

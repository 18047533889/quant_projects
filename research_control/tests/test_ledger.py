"""Tests for research_control event ledger."""

import pytest
from datetime import datetime, timedelta
from research_control.events import CampaignEvent, TrialEvent, DecisionEvent
from research_control.ledger import EventLedger


class TestEventLedger:
    def test_append_and_get_all(self):
        ledger = EventLedger()
        e1 = CampaignEvent(campaign_id="camp_001", event_type="created")
        e2 = CampaignEvent(campaign_id="camp_001", event_type="started")

        assert ledger.append(e1) is True
        assert ledger.append(e2) is True

        events = ledger.get_all()
        assert len(events) == 2
        assert events[0] == e1
        assert events[1] == e2

    def test_append_duplicate_event_id(self):
        ledger = EventLedger()
        e1 = CampaignEvent(
            event_id="evt_123",
            campaign_id="camp_001",
            event_type="created"
        )
        e2 = CampaignEvent(
            event_id="evt_123",
            campaign_id="camp_001",
            event_type="started"
        )

        assert ledger.append(e1) is True
        assert ledger.append(e2) is False  # Duplicate ignored
        assert ledger.count() == 1

    def test_get_by_campaign(self):
        ledger = EventLedger()

        c1 = CampaignEvent(campaign_id="camp_001", event_type="created")
        c2 = CampaignEvent(campaign_id="camp_002", event_type="created")
        t1 = TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="submitted")
        t2 = TrialEvent(trial_id="t2", campaign_id="camp_002", event_type="submitted")

        ledger.append(c1)
        ledger.append(c2)
        ledger.append(t1)
        ledger.append(t2)

        camp_001_events = ledger.get_by_campaign("camp_001")
        assert len(camp_001_events) == 2
        assert c1 in camp_001_events
        assert t1 in camp_001_events

        camp_002_events = ledger.get_by_campaign("camp_002")
        assert len(camp_002_events) == 2

    def test_get_by_trial(self):
        ledger = EventLedger()

        t1_submit = TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="submitted")
        t1_run = TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="running")
        t1_complete = TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="completed")
        t2_submit = TrialEvent(trial_id="t2", campaign_id="camp_001", event_type="submitted")

        ledger.append(t1_submit)
        ledger.append(t1_run)
        ledger.append(t1_complete)
        ledger.append(t2_submit)

        t1_events = ledger.get_by_trial("t1")
        assert len(t1_events) == 3
        assert t1_submit in t1_events
        assert t1_run in t1_events
        assert t1_complete in t1_events

    def test_get_by_decision(self):
        ledger = EventLedger()

        d1 = DecisionEvent(decision_id="dec_001", campaign_id="camp_001", decision_type="early_stop")
        d2 = DecisionEvent(decision_id="dec_002", campaign_id="camp_001", decision_type="early_stop")

        ledger.append(d1)
        ledger.append(d2)

        dec_events = ledger.get_by_decision("dec_001")
        assert len(dec_events) == 1
        assert dec_events[0] == d1

    def test_get_by_time_range(self):
        ledger = EventLedger()

        base_time = datetime(2026, 8, 13, 10, 0, 0)

        e1 = CampaignEvent(
            campaign_id="camp_001",
            event_type="created",
            timestamp=base_time
        )
        e2 = CampaignEvent(
            campaign_id="camp_001",
            event_type="started",
            timestamp=base_time + timedelta(hours=1)
        )
        e3 = CampaignEvent(
            campaign_id="camp_001",
            event_type="completed",
            timestamp=base_time + timedelta(hours=2)
        )

        ledger.append(e1)
        ledger.append(e2)
        ledger.append(e3)

        # Query middle hour
        events = ledger.get_by_time_range(
            start=base_time + timedelta(minutes=30),
            end=base_time + timedelta(hours=1, minutes=30)
        )
        assert len(events) == 1
        assert events[0] == e2

        # Query all
        events = ledger.get_by_time_range(start=base_time)
        assert len(events) == 3

        # Query with end only
        events = ledger.get_by_time_range(end=base_time + timedelta(hours=1))
        assert len(events) == 2

    def test_get_by_event_id(self):
        ledger = EventLedger()

        e1 = CampaignEvent(
            event_id="evt_123",
            campaign_id="camp_001",
            event_type="created"
        )
        ledger.append(e1)

        found = ledger.get_by_event_id("evt_123")
        assert found == e1

        not_found = ledger.get_by_event_id("evt_999")
        assert not_found is None

    def test_count(self):
        ledger = EventLedger()
        assert ledger.count() == 0

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="created"))
        assert ledger.count() == 1

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="started"))
        assert ledger.count() == 2

    def test_count_by_campaign(self):
        ledger = EventLedger()

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="created"))
        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="started"))
        ledger.append(CampaignEvent(campaign_id="camp_002", event_type="created"))

        assert ledger.count_by_campaign("camp_001") == 2
        assert ledger.count_by_campaign("camp_002") == 1
        assert ledger.count_by_campaign("camp_999") == 0

    def test_append_preserves_order(self):
        ledger = EventLedger()

        events = []
        for i in range(10):
            e = CampaignEvent(
                campaign_id=f"camp_{i:03d}",
                event_type="created"
            )
            events.append(e)
            ledger.append(e)

        retrieved = ledger.get_all()
        assert retrieved == events

    def test_mixed_event_types(self):
        ledger = EventLedger()

        c = CampaignEvent(campaign_id="camp_001", event_type="created")
        t = TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="submitted")
        d = DecisionEvent(decision_id="dec_001", campaign_id="camp_001", decision_type="early_stop")

        ledger.append(c)
        ledger.append(t)
        ledger.append(d)

        all_events = ledger.get_all()
        assert len(all_events) == 3
        assert isinstance(all_events[0], CampaignEvent)
        assert isinstance(all_events[1], TrialEvent)
        assert isinstance(all_events[2], DecisionEvent)

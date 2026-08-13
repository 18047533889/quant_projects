"""Tests for research_control idempotent sync."""

import pytest
from research_control.events import CampaignEvent, TrialEvent, DecisionEvent
from research_control.ledger import EventLedger
from research_control.sync import IdempotentSync


class TestIdempotentSync:
    def test_sync_events_all_new(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        events = [
            CampaignEvent(campaign_id="camp_001", event_type="created"),
            CampaignEvent(campaign_id="camp_001", event_type="started"),
            TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="submitted"),
        ]

        result = sync.sync_events(events)

        assert result["added"] == 3
        assert result["duplicates"] == 0
        assert result["failed"] == 0
        assert result["total_processed"] == 3
        assert ledger.count() == 3

    def test_sync_events_with_duplicates(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        e1 = CampaignEvent(event_id="evt_001", campaign_id="camp_001", event_type="created")
        e2 = CampaignEvent(event_id="evt_002", campaign_id="camp_001", event_type="started")
        e3 = CampaignEvent(event_id="evt_001", campaign_id="camp_001", event_type="started")  # Duplicate ID

        ledger.append(e1)

        result = sync.sync_events([e2, e3])

        assert result["added"] == 1
        assert result["duplicates"] == 1
        assert result["failed"] == 0
        assert result["total_processed"] == 2
        assert ledger.count() == 2

    def test_sync_events_empty_list(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        result = sync.sync_events([])

        assert result["added"] == 0
        assert result["duplicates"] == 0
        assert result["failed"] == 0
        assert result["total_processed"] == 0

    def test_get_missing_event_ids(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        e1 = CampaignEvent(event_id="evt_001", campaign_id="camp_001", event_type="created")
        e2 = CampaignEvent(event_id="evt_002", campaign_id="camp_001", event_type="started")

        ledger.append(e1)

        missing = sync.get_missing_event_ids(["evt_001", "evt_002", "evt_003"])

        assert "evt_001" not in missing
        assert "evt_002" in missing
        assert "evt_003" in missing
        assert len(missing) == 2

    def test_get_missing_event_ids_all_present(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        e1 = CampaignEvent(event_id="evt_001", campaign_id="camp_001", event_type="created")
        e2 = CampaignEvent(event_id="evt_002", campaign_id="camp_001", event_type="started")

        ledger.append(e1)
        ledger.append(e2)

        missing = sync.get_missing_event_ids(["evt_001", "evt_002"])

        assert len(missing) == 0

    def test_verify_campaign_consistency_valid(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="created"))
        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="started"))
        ledger.append(TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="submitted"))
        ledger.append(TrialEvent(trial_id="t1", campaign_id="camp_001", event_type="completed"))
        ledger.append(DecisionEvent(decision_id="dec_001", campaign_id="camp_001", decision_type="early_stop"))
        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="completed"))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["campaign_id"] == "camp_001"
        assert result["total_events"] == 6
        assert result["campaign_events"] == 3
        assert result["trial_events"] == 2
        assert result["decision_events"] == 1
        assert result["campaign_states"] == ["created", "started", "completed"]
        assert result["is_consistent"] is True
        assert len(result["anomalies"]) == 0

    def test_verify_campaign_consistency_missing_created(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="started"))
        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="completed"))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is False
        assert len(result["anomalies"]) == 2
        assert any("first event" in a for a in result["anomalies"])
        assert any("started without being created" in a for a in result["anomalies"])

    def test_verify_campaign_consistency_multiple_terminal_states(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="created"))
        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="completed"))
        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="cancelled"))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is False
        assert any("multiple terminal states" in a for a in result["anomalies"])

    def test_verify_campaign_consistency_empty_campaign(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        result = sync.verify_campaign_consistency("camp_999")

        assert result["campaign_id"] == "camp_999"
        assert result["total_events"] == 0
        assert result["campaign_events"] == 0
        assert result["is_consistent"] is True

    def test_verify_campaign_consistency_started_without_created(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        ledger.append(CampaignEvent(campaign_id="camp_001", event_type="started"))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is False
        assert any("started without being created" in a for a in result["anomalies"])
        assert any("first event" in a for a in result["anomalies"])

    def test_idempotent_sync_multiple_calls(self):
        ledger = EventLedger()
        sync = IdempotentSync(ledger)

        events = [
            CampaignEvent(event_id="evt_001", campaign_id="camp_001", event_type="created"),
            CampaignEvent(event_id="evt_002", campaign_id="camp_001", event_type="started"),
        ]

        # First sync
        result1 = sync.sync_events(events)
        assert result1["added"] == 2

        # Second sync with same events
        result2 = sync.sync_events(events)
        assert result2["added"] == 0
        assert result2["duplicates"] == 2

        # Ledger should still have 2 events
        assert ledger.count() == 2

"""Tests for IdempotentSyncEngine with fail-open and fail-atomic modes."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from research_control.ledger.campaign import CampaignLedger
from research_control.ledger.trial import TrialLedger
from research_control.sync.idempotency import IdempotentSyncEngine


class TestIdempotentSyncEngine:
    def test_sync_campaign_events_fail_open_all_valid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {
                "event_id": "evt_001",
                "campaign_id": "camp_001",
                "state": "created",
                "timestamp": base_ts,
                "metadata": {"user": "alice"},
            },
            {
                "event_id": "evt_002",
                "campaign_id": "camp_001",
                "state": "started",
                "timestamp": base_ts + timedelta(hours=1),
            },
        ]

        result = sync.sync_campaign_events(events, mode="fail_open")

        assert result["added"] == 2
        assert result["duplicates"] == 0
        assert result["failed"] == 0
        assert result["total_processed"] == 2
        assert len(result["failed_events"]) == 0

    def test_sync_campaign_events_fail_open_with_duplicates(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        # Add one event directly
        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)

        # Sync batch with duplicate
        events = [
            {"event_id": "evt_001", "campaign_id": "camp_001", "state": "created", "timestamp": base_ts},
            {"event_id": "evt_002", "campaign_id": "camp_001", "state": "started", "timestamp": base_ts + timedelta(hours=1)},
        ]

        result = sync.sync_campaign_events(events, mode="fail_open")

        assert result["added"] == 1
        assert result["duplicates"] == 1
        assert result["failed"] == 0

    def test_sync_campaign_events_fail_open_with_invalid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {"event_id": "evt_001", "campaign_id": "camp_001", "state": "created", "timestamp": base_ts},
            {"event_id": "evt_002", "campaign_id": "camp_001", "state": "invalid_state", "timestamp": base_ts + timedelta(hours=1)},
            {"event_id": "evt_003", "campaign_id": "camp_001", "state": "started", "timestamp": base_ts + timedelta(hours=2)},
        ]

        result = sync.sync_campaign_events(events, mode="fail_open")

        assert result["added"] == 2
        assert result["duplicates"] == 0
        assert result["failed"] == 1
        assert len(result["failed_events"]) == 1
        assert "invalid_state" in result["failed_events"][0]["error"]

    def test_sync_campaign_events_fail_atomic_all_valid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {"event_id": "evt_001", "campaign_id": "camp_001", "state": "created", "timestamp": base_ts},
            {"event_id": "evt_002", "campaign_id": "camp_001", "state": "started", "timestamp": base_ts + timedelta(hours=1)},
        ]

        result = sync.sync_campaign_events(events, mode="fail_atomic")

        assert result["added"] == 2
        assert result["duplicates"] == 0
        assert result["failed"] == 0
        assert "error" not in result

    def test_sync_campaign_events_fail_atomic_with_invalid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {"event_id": "evt_001", "campaign_id": "camp_001", "state": "created", "timestamp": base_ts},
            {"event_id": "evt_002", "campaign_id": "camp_001", "state": "invalid_state", "timestamp": base_ts + timedelta(hours=1)},
            {"event_id": "evt_003", "campaign_id": "camp_001", "state": "started", "timestamp": base_ts + timedelta(hours=2)},
        ]

        result = sync.sync_campaign_events(events, mode="fail_atomic")

        assert result["added"] == 0
        assert result["duplicates"] == 0
        assert result["failed"] == 3
        assert "error" in result
        assert "Validation failed" in result["error"]

        # No events should have been added
        assert campaign_ledger.count_events() == 0

    def test_sync_trial_events_fail_open_all_valid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {
                "event_id": "evt_001",
                "trial_id": "trial_001",
                "campaign_id": "camp_001",
                "state": "submitted",
                "timestamp": base_ts,
                "parameters": {"alpha": 0.01},
                "metrics": {"sharpe": 1.5},
            },
            {
                "event_id": "evt_002",
                "trial_id": "trial_001",
                "campaign_id": "camp_001",
                "state": "running",
                "timestamp": base_ts + timedelta(minutes=5),
            },
        ]

        result = sync.sync_trial_events(events, mode="fail_open")

        assert result["added"] == 2
        assert result["duplicates"] == 0
        assert result["failed"] == 0

    def test_sync_trial_events_fail_open_with_duplicates(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        # Add one event directly
        trial_ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)

        events = [
            {"event_id": "evt_001", "trial_id": "trial_001", "campaign_id": "camp_001", "state": "submitted", "timestamp": base_ts},
            {"event_id": "evt_002", "trial_id": "trial_001", "campaign_id": "camp_001", "state": "running", "timestamp": base_ts + timedelta(minutes=5)},
        ]

        result = sync.sync_trial_events(events, mode="fail_open")

        assert result["added"] == 1
        assert result["duplicates"] == 1
        assert result["failed"] == 0

    def test_sync_trial_events_fail_atomic_with_invalid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {"event_id": "evt_001", "trial_id": "trial_001", "campaign_id": "camp_001", "state": "submitted", "timestamp": base_ts},
            {"event_id": "evt_002", "trial_id": "trial_001", "campaign_id": "camp_001", "state": "invalid_state", "timestamp": base_ts + timedelta(minutes=5)},
        ]

        result = sync.sync_trial_events(events, mode="fail_atomic")

        assert result["added"] == 0
        assert result["failed"] == 2
        assert "error" in result
        assert trial_ledger.count_events() == 0

    def test_get_missing_event_ids_campaigns(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_003", "camp_001", "started", base_ts + timedelta(hours=1))

        missing = sync.get_missing_event_ids(["evt_001", "evt_002", "evt_003", "evt_004"], entity_type="campaign")

        assert "evt_001" not in missing
        assert "evt_002" in missing
        assert "evt_003" not in missing
        assert "evt_004" in missing

    def test_get_missing_event_ids_trials(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        trial_ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)
        trial_ledger.append("evt_003", "trial_001", "camp_001", "running", base_ts + timedelta(minutes=5))

        missing = sync.get_missing_event_ids(["evt_001", "evt_002", "evt_003"], entity_type="trial")

        assert "evt_001" not in missing
        assert "evt_002" in missing
        assert "evt_003" not in missing

    def test_verify_campaign_consistency_valid(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_003", "camp_001", "completed", base_ts + timedelta(hours=2))

        trial_ledger.append("evt_004", "trial_001", "camp_001", "submitted", base_ts + timedelta(minutes=5))
        trial_ledger.append("evt_005", "trial_002", "camp_001", "submitted", base_ts + timedelta(minutes=10))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is True
        assert len(result["anomalies"]) == 0
        assert result["state_sequence"] == ["created", "started", "completed"]
        assert result["trial_count"] == 2

    def test_verify_campaign_consistency_missing_created(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "started", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "completed", base_ts + timedelta(hours=1))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is False
        assert len(result["anomalies"]) >= 2
        assert any("First state" in a for a in result["anomalies"])
        assert any("started without being created" in a for a in result["anomalies"])

    def test_verify_campaign_consistency_multiple_terminal(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "completed", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_003", "camp_001", "cancelled", base_ts + timedelta(hours=2))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is False
        assert any("Multiple terminal states" in a for a in result["anomalies"])

    def test_verify_campaign_consistency_transition_after_terminal(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "completed", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_003", "camp_001", "started", base_ts + timedelta(hours=2))

        result = sync.verify_campaign_consistency("camp_001")

        assert result["is_consistent"] is False
        assert any("after terminal state" in a for a in result["anomalies"])

    def test_invalid_mode_raises_error(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        with pytest.raises(ValueError, match="Invalid mode"):
            sync.sync_campaign_events([], mode="invalid_mode")

        with pytest.raises(ValueError, match="Invalid mode"):
            sync.sync_trial_events([], mode="invalid_mode")

    def test_timestamp_parsing_datetime_object(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {"event_id": "evt_001", "campaign_id": "camp_001", "state": "created", "timestamp": base_ts},
        ]

        result = sync.sync_campaign_events(events, mode="fail_open")

        assert result["added"] == 1
        assert result["failed"] == 0

    def test_timestamp_parsing_iso_string(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)
        events = [
            {"event_id": "evt_001", "campaign_id": "camp_001", "state": "created", "timestamp": base_ts.isoformat()},
        ]

        result = sync.sync_campaign_events(events, mode="fail_open")

        assert result["added"] == 1
        assert result["failed"] == 0

    def test_empty_event_list(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        sync = IdempotentSyncEngine(campaign_ledger, trial_ledger)

        result = sync.sync_campaign_events([], mode="fail_open")
        assert result["added"] == 0
        assert result["total_processed"] == 0

        result = sync.sync_trial_events([], mode="fail_atomic")
        assert result["added"] == 0
        assert result["total_processed"] == 0

"""Tests for LedgerQuery temporal queries and lineage reconstruction."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from research_control.ledger.campaign import CampaignLedger
from research_control.ledger.trial import TrialLedger
from research_control.ledger.query import LedgerQuery


class TestLedgerQuery:
    def test_query_time_range_campaigns(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_003", "camp_001", "completed", base_ts + timedelta(hours=3))

        result = query.query_time_range(
            start=base_ts + timedelta(minutes=30),
            end=base_ts + timedelta(hours=2),
            entity_type="campaign",
        )

        assert len(result["campaigns"]) == 1
        assert result["campaigns"][0]["event_id"] == "evt_002"

    def test_query_time_range_trials(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        trial_ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)
        trial_ledger.append("evt_002", "trial_001", "camp_001", "running", base_ts + timedelta(minutes=5))
        trial_ledger.append("evt_003", "trial_001", "camp_001", "completed", base_ts + timedelta(minutes=15))

        result = query.query_time_range(
            start=base_ts + timedelta(minutes=3),
            end=base_ts + timedelta(minutes=10),
            entity_type="trial",
        )

        assert len(result["trials"]) == 1
        assert result["trials"][0]["event_id"] == "evt_002"

    def test_query_time_range_all(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        trial_ledger.append("evt_002", "trial_001", "camp_001", "submitted", base_ts + timedelta(minutes=5))
        campaign_ledger.append("evt_003", "camp_001", "started", base_ts + timedelta(minutes=10))

        result = query.query_time_range(
            start=base_ts,
            end=base_ts + timedelta(minutes=15),
            entity_type="all",
        )

        assert len(result["campaigns"]) == 2
        assert len(result["trials"]) == 1

    def test_reconstruct_lineage(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        # Campaign events
        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(minutes=5))

        # Trial events
        trial_ledger.append("evt_003", "trial_001", "camp_001", "submitted", base_ts + timedelta(minutes=10))
        trial_ledger.append("evt_004", "trial_001", "camp_001", "completed", base_ts + timedelta(minutes=15))
        trial_ledger.append("evt_005", "trial_002", "camp_001", "submitted", base_ts + timedelta(minutes=20))

        lineage = query.reconstruct_lineage("camp_001")

        assert lineage["campaign_id"] == "camp_001"
        assert lineage["current_state"] == "started"
        assert len(lineage["state_history"]) == 2
        assert len(lineage["trials"]) == 2
        assert "trial_001" in lineage["trials"]
        assert "trial_002" in lineage["trials"]
        assert len(lineage["trials"]["trial_001"]) == 2
        assert len(lineage["trials"]["trial_002"]) == 1

    def test_reconstruct_lineage_timeline(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        trial_ledger.append("evt_002", "trial_001", "camp_001", "submitted", base_ts + timedelta(minutes=5))
        campaign_ledger.append("evt_003", "camp_001", "started", base_ts + timedelta(minutes=10))

        lineage = query.reconstruct_lineage("camp_001")
        timeline = lineage["timeline"]

        assert len(timeline) == 3
        assert timeline[0]["type"] == "campaign_state"
        assert timeline[0]["state"] == "created"
        assert timeline[1]["type"] == "trial_event"
        assert timeline[1]["entity_id"] == "trial_001"
        assert timeline[2]["type"] == "campaign_state"
        assert timeline[2]["state"] == "started"

    def test_get_state_at(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_003", "camp_001", "completed", base_ts + timedelta(hours=2))

        # Before creation
        state = query.get_state_at("camp_001", base_ts - timedelta(minutes=1))
        assert state is None

        # Right at creation
        state = query.get_state_at("camp_001", base_ts)
        assert state == "created"

        # Between started and completed
        state = query.get_state_at("camp_001", base_ts + timedelta(hours=1, minutes=30))
        assert state == "started"

        # After completion
        state = query.get_state_at("camp_001", base_ts + timedelta(hours=3))
        assert state == "completed"

    def test_get_history_in_range(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_003", "camp_001", "paused", base_ts + timedelta(hours=2))
        campaign_ledger.append("evt_004", "camp_001", "resumed", base_ts + timedelta(hours=3))

        history = query.get_history_in_range(
            "camp_001",
            base_ts + timedelta(minutes=30),
            base_ts + timedelta(hours=2, minutes=30),
        )

        assert len(history) == 2
        assert history[0]["state"] == "started"
        assert history[1]["state"] == "paused"

    def test_find_campaigns_active_during(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        # Campaign 1: created at 10:00, completed at 12:00
        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "completed", base_ts + timedelta(hours=2))

        # Campaign 2: created at 11:00, still active
        campaign_ledger.append("evt_003", "camp_002", "created", base_ts + timedelta(hours=1))
        campaign_ledger.append("evt_004", "camp_002", "started", base_ts + timedelta(hours=1, minutes=5))

        # Campaign 3: created at 13:00, completed at 14:00
        campaign_ledger.append("evt_005", "camp_003", "created", base_ts + timedelta(hours=3))
        campaign_ledger.append("evt_006", "camp_003", "completed", base_ts + timedelta(hours=4))

        # Query 10:30 - 11:30 (should find camp_001 and camp_002)
        active = query.find_campaigns_active_during(
            base_ts + timedelta(minutes=30),
            base_ts + timedelta(hours=1, minutes=30),
        )

        campaign_ids = [c["campaign_id"] for c in active]
        assert "camp_001" in campaign_ids
        assert "camp_002" in campaign_ids
        assert "camp_003" not in campaign_ids

    def test_count_events_in_range(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
        campaign_ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))

        trial_ledger.append("evt_003", "trial_001", "camp_001", "submitted", base_ts + timedelta(minutes=5))
        trial_ledger.append("evt_004", "trial_002", "camp_001", "submitted", base_ts + timedelta(minutes=10))
        trial_ledger.append("evt_005", "trial_001", "camp_001", "completed", base_ts + timedelta(hours=2))

        counts = query.count_events_in_range(
            base_ts,
            base_ts + timedelta(hours=1, minutes=30),
        )

        assert counts["campaigns"] == 2
        assert counts["trials"] == 2

    def test_query_with_shared_db_path(self):
        # Use same database for both ledgers
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "shared.db")

            campaign_ledger = CampaignLedger(db_path=db_path)
            # Note: This creates separate tables in the same database
            trial_ledger = TrialLedger(db_path=db_path)
            query = LedgerQuery(campaign_ledger, trial_ledger)

            base_ts = datetime(2026, 8, 13, 10, 0, 0)

            campaign_ledger.append("evt_001", "camp_001", "created", base_ts)
            trial_ledger.append("evt_002", "trial_001", "camp_001", "submitted", base_ts + timedelta(minutes=5))

            lineage = query.reconstruct_lineage("camp_001")
            assert len(lineage["state_history"]) == 1
            assert len(lineage["trials"]) == 1

    def test_empty_lineage(self):
        campaign_ledger = CampaignLedger()
        trial_ledger = TrialLedger()
        query = LedgerQuery(campaign_ledger, trial_ledger)

        lineage = query.reconstruct_lineage("camp_999")

        assert lineage["campaign_id"] == "camp_999"
        assert lineage["current_state"] is None
        assert len(lineage["state_history"]) == 0
        assert len(lineage["trials"]) == 0
        assert len(lineage["timeline"]) == 0

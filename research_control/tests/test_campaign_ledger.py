"""Tests for CampaignLedger SQLite implementation."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from research_control.ledger.campaign import CampaignLedger


class TestCampaignLedger:
    def test_init_in_memory(self):
        ledger = CampaignLedger()
        assert ledger.count_events() == 0

    def test_init_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "campaign.db")
            ledger = CampaignLedger(db_path=db_path)
            assert ledger.count_events() == 0
            assert os.path.exists(db_path)

    def test_append_valid_event(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        result = ledger.append(
            event_id="evt_001",
            campaign_id="camp_001",
            state="created",
            timestamp=ts,
            metadata={"user": "researcher_a"},
        )

        assert result is True
        assert ledger.count_events() == 1

    def test_append_duplicate_event_id(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "camp_001", "created", ts)
        result = ledger.append("evt_001", "camp_002", "created", ts)

        assert result is False  # Idempotent
        assert ledger.count_events() == 1

    def test_append_invalid_state(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        with pytest.raises(ValueError, match="Invalid campaign state"):
            ledger.append("evt_001", "camp_001", "invalid_state", ts)

    def test_append_all_valid_states(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        valid_states = ["created", "started", "paused", "resumed", "completed", "cancelled"]
        for i, state in enumerate(valid_states):
            result = ledger.append(f"evt_{i:03d}", "camp_001", state, ts + timedelta(minutes=i))
            assert result is True

        assert ledger.count_events() == len(valid_states)

    def test_get_campaign_history(self):
        ledger = CampaignLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "camp_001", "created", base_ts)
        ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))
        ledger.append("evt_003", "camp_002", "created", base_ts + timedelta(hours=2))
        ledger.append("evt_004", "camp_001", "completed", base_ts + timedelta(hours=3))

        history = ledger.get_campaign_history("camp_001")
        assert len(history) == 3
        assert history[0]["state"] == "created"
        assert history[1]["state"] == "started"
        assert history[2]["state"] == "completed"

    def test_get_current_state(self):
        ledger = CampaignLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "camp_001", "created", base_ts)
        ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))
        ledger.append("evt_003", "camp_001", "completed", base_ts + timedelta(hours=2))

        state = ledger.get_current_state("camp_001")
        assert state == "completed"

    def test_get_current_state_nonexistent(self):
        ledger = CampaignLedger()
        state = ledger.get_current_state("camp_999")
        assert state is None

    def test_get_all_campaigns(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "camp_001", "created", ts)
        ledger.append("evt_002", "camp_002", "created", ts)
        ledger.append("evt_003", "camp_001", "started", ts)
        ledger.append("evt_004", "camp_003", "created", ts)

        campaigns = ledger.get_all_campaigns()
        assert len(campaigns) == 3
        assert "camp_001" in campaigns
        assert "camp_002" in campaigns
        assert "camp_003" in campaigns

    def test_get_active_campaigns(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "camp_001", "created", ts)
        ledger.append("evt_002", "camp_001", "started", ts + timedelta(hours=1))

        ledger.append("evt_003", "camp_002", "created", ts)
        ledger.append("evt_004", "camp_002", "completed", ts + timedelta(hours=1))

        ledger.append("evt_005", "camp_003", "created", ts)
        ledger.append("evt_006", "camp_003", "cancelled", ts + timedelta(hours=1))

        active = ledger.get_active_campaigns()
        assert len(active) == 1
        assert "camp_001" in active
        assert "camp_002" not in active
        assert "camp_003" not in active

    def test_count_events(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        assert ledger.count_events() == 0

        ledger.append("evt_001", "camp_001", "created", ts)
        assert ledger.count_events() == 1

        ledger.append("evt_002", "camp_002", "created", ts)
        assert ledger.count_events() == 2

        assert ledger.count_events(campaign_id="camp_001") == 1
        assert ledger.count_events(campaign_id="camp_002") == 1
        assert ledger.count_events(campaign_id="camp_999") == 0

    def test_metadata_persistence(self):
        ledger = CampaignLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        metadata = {
            "user": "alice",
            "reason": "initial exploration",
            "tags": ["momentum", "mean_reversion"],
        }

        ledger.append("evt_001", "camp_001", "created", ts, metadata=metadata)

        history = ledger.get_campaign_history("camp_001")
        assert len(history) == 1
        assert history[0]["metadata"] == metadata

    def test_chronological_ordering(self):
        ledger = CampaignLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        # Insert out of order
        ledger.append("evt_003", "camp_001", "completed", base_ts + timedelta(hours=3))
        ledger.append("evt_001", "camp_001", "created", base_ts)
        ledger.append("evt_002", "camp_001", "started", base_ts + timedelta(hours=1))

        history = ledger.get_campaign_history("camp_001")
        assert history[0]["state"] == "created"
        assert history[1]["state"] == "started"
        assert history[2]["state"] == "completed"

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "campaign.db")
            ts = datetime(2026, 8, 13, 10, 0, 0)

            # First instance
            ledger1 = CampaignLedger(db_path=db_path)
            ledger1.append("evt_001", "camp_001", "created", ts)
            ledger1.append("evt_002", "camp_001", "started", ts + timedelta(hours=1))

            # Second instance (same db)
            ledger2 = CampaignLedger(db_path=db_path)
            assert ledger2.count_events() == 2
            assert ledger2.get_current_state("camp_001") == "started"

            history = ledger2.get_campaign_history("camp_001")
            assert len(history) == 2

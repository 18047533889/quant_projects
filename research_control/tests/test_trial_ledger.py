"""Tests for TrialLedger SQLite implementation."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from research_control.ledger.trial import TrialLedger


class TestTrialLedger:
    def test_init_in_memory(self):
        ledger = TrialLedger()
        assert ledger.count_events() == 0

    def test_init_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trial.db")
            ledger = TrialLedger(db_path=db_path)
            assert ledger.count_events() == 0
            assert os.path.exists(db_path)

    def test_append_valid_event(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        result = ledger.append(
            event_id="evt_001",
            trial_id="trial_001",
            campaign_id="camp_001",
            state="submitted",
            timestamp=ts,
            parameters={"alpha": 0.01},
            metrics={"sharpe": 1.5},
        )

        assert result is True
        assert ledger.count_events() == 1

    def test_append_duplicate_event_id(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", ts)
        result = ledger.append("evt_001", "trial_002", "camp_001", "submitted", ts)

        assert result is False  # Idempotent
        assert ledger.count_events() == 1

    def test_append_invalid_state(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        with pytest.raises(ValueError, match="Invalid trial state"):
            ledger.append("evt_001", "trial_001", "camp_001", "invalid_state", ts)

    def test_append_all_valid_states(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        valid_states = ["submitted", "running", "completed", "failed", "timeout"]
        for i, state in enumerate(valid_states):
            result = ledger.append(
                f"evt_{i:03d}", "trial_001", "camp_001", state, ts + timedelta(minutes=i)
            )
            assert result is True

        assert ledger.count_events() == len(valid_states)

    def test_get_trial_history(self):
        ledger = TrialLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)
        ledger.append("evt_002", "trial_001", "camp_001", "running", base_ts + timedelta(minutes=5))
        ledger.append("evt_003", "trial_002", "camp_001", "submitted", base_ts + timedelta(minutes=10))
        ledger.append("evt_004", "trial_001", "camp_001", "completed", base_ts + timedelta(minutes=15))

        history = ledger.get_trial_history("trial_001")
        assert len(history) == 3
        assert history[0]["state"] == "submitted"
        assert history[1]["state"] == "running"
        assert history[2]["state"] == "completed"

    def test_get_campaign_trials(self):
        ledger = TrialLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)
        ledger.append("evt_002", "trial_002", "camp_001", "submitted", base_ts + timedelta(minutes=5))
        ledger.append("evt_003", "trial_003", "camp_002", "submitted", base_ts + timedelta(minutes=10))
        ledger.append("evt_004", "trial_001", "camp_001", "completed", base_ts + timedelta(minutes=15))

        camp_001_trials = ledger.get_campaign_trials("camp_001")
        assert len(camp_001_trials) == 3
        trial_ids = {evt["trial_id"] for evt in camp_001_trials}
        assert trial_ids == {"trial_001", "trial_002"}

    def test_get_current_state(self):
        ledger = TrialLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)
        ledger.append("evt_002", "trial_001", "camp_001", "running", base_ts + timedelta(minutes=5))
        ledger.append("evt_003", "trial_001", "camp_001", "completed", base_ts + timedelta(minutes=10))

        state = ledger.get_current_state("trial_001")
        assert state == "completed"

    def test_get_current_state_nonexistent(self):
        ledger = TrialLedger()
        state = ledger.get_current_state("trial_999")
        assert state is None

    def test_get_all_trials(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", ts)
        ledger.append("evt_002", "trial_002", "camp_001", "submitted", ts)
        ledger.append("evt_003", "trial_003", "camp_002", "submitted", ts)
        ledger.append("evt_004", "trial_001", "camp_001", "running", ts)

        all_trials = ledger.get_all_trials()
        assert len(all_trials) == 3
        assert "trial_001" in all_trials
        assert "trial_002" in all_trials
        assert "trial_003" in all_trials

        camp_001_trials = ledger.get_all_trials(campaign_id="camp_001")
        assert len(camp_001_trials) == 2
        assert "trial_001" in camp_001_trials
        assert "trial_002" in camp_001_trials

    def test_get_completed_trials(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", ts)
        ledger.append("evt_002", "trial_001", "camp_001", "completed", ts + timedelta(minutes=5))

        ledger.append("evt_003", "trial_002", "camp_001", "submitted", ts)
        ledger.append("evt_004", "trial_002", "camp_001", "failed", ts + timedelta(minutes=5))

        ledger.append("evt_005", "trial_003", "camp_001", "submitted", ts)
        ledger.append("evt_006", "trial_003", "camp_001", "running", ts + timedelta(minutes=5))

        completed = ledger.get_completed_trials("camp_001")
        assert len(completed) == 1
        assert "trial_001" in completed

    def test_get_trial_metrics(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        metrics = {"sharpe": 1.5, "returns": 0.12, "max_drawdown": -0.08}

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", ts)
        ledger.append(
            "evt_002", "trial_001", "camp_001", "completed", ts + timedelta(minutes=5), metrics=metrics
        )

        retrieved_metrics = ledger.get_trial_metrics("trial_001")
        assert retrieved_metrics == metrics

    def test_get_trial_metrics_none(self):
        ledger = TrialLedger()
        metrics = ledger.get_trial_metrics("trial_999")
        assert metrics is None

    def test_count_events(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        assert ledger.count_events() == 0

        ledger.append("evt_001", "trial_001", "camp_001", "submitted", ts)
        assert ledger.count_events() == 1

        ledger.append("evt_002", "trial_002", "camp_001", "submitted", ts)
        assert ledger.count_events() == 2

        assert ledger.count_events(trial_id="trial_001") == 1
        assert ledger.count_events(campaign_id="camp_001") == 2
        assert ledger.count_events(trial_id="trial_999") == 0

    def test_parameters_and_metrics_persistence(self):
        ledger = TrialLedger()
        ts = datetime(2026, 8, 13, 10, 0, 0)

        parameters = {"alpha": 0.01, "lookback": 20, "threshold": 0.05}
        metrics = {"sharpe": 1.5, "returns": 0.12, "volatility": 0.08}
        metadata = {"gpu_used": True, "runtime_seconds": 120}

        ledger.append(
            "evt_001",
            "trial_001",
            "camp_001",
            "completed",
            ts,
            parameters=parameters,
            metrics=metrics,
            metadata=metadata,
        )

        history = ledger.get_trial_history("trial_001")
        assert len(history) == 1
        assert history[0]["parameters"] == parameters
        assert history[0]["metrics"] == metrics
        assert history[0]["metadata"] == metadata

    def test_chronological_ordering(self):
        ledger = TrialLedger()
        base_ts = datetime(2026, 8, 13, 10, 0, 0)

        # Insert out of order
        ledger.append("evt_003", "trial_001", "camp_001", "completed", base_ts + timedelta(minutes=15))
        ledger.append("evt_001", "trial_001", "camp_001", "submitted", base_ts)
        ledger.append("evt_002", "trial_001", "camp_001", "running", base_ts + timedelta(minutes=5))

        history = ledger.get_trial_history("trial_001")
        assert history[0]["state"] == "submitted"
        assert history[1]["state"] == "running"
        assert history[2]["state"] == "completed"

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trial.db")
            ts = datetime(2026, 8, 13, 10, 0, 0)

            # First instance
            ledger1 = TrialLedger(db_path=db_path)
            ledger1.append("evt_001", "trial_001", "camp_001", "submitted", ts)
            ledger1.append("evt_002", "trial_001", "camp_001", "running", ts + timedelta(minutes=5))

            # Second instance (same db)
            ledger2 = TrialLedger(db_path=db_path)
            assert ledger2.count_events() == 2
            assert ledger2.get_current_state("trial_001") == "running"

            history = ledger2.get_trial_history("trial_001")
            assert len(history) == 2

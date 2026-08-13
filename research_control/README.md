# Research Control

Lightweight append-only ledger for tracking research campaigns, trials, and decisions.

## Features

- **Immutable Events**: CampaignEvent, TrialEvent, DecisionEvent with frozen dataclasses
- **Append-Only Ledger**: In-memory store with indexed queries by campaign, trial, decision, time range
- **Idempotent Sync**: Event deduplication via event IDs, batch sync operations
- **Consistency Verification**: Campaign lifecycle validation

## Installation

```bash
pip install -e .
```

## Usage

```python
from research_control import (
    CampaignEvent,
    TrialEvent,
    DecisionEvent,
    EventLedger,
    IdempotentSync,
)

# Create ledger
ledger = EventLedger()

# Append events
campaign_event = CampaignEvent(
    campaign_id="camp_001",
    event_type="created",
    metadata={"researcher": "alice"}
)
ledger.append(campaign_event)

trial_event = TrialEvent(
    trial_id="trial_001",
    campaign_id="camp_001",
    event_type="submitted",
    parameters={"alpha": 0.01},
    metrics={"sharpe": 1.5}
)
ledger.append(trial_event)

# Query events
all_events = ledger.get_all()
campaign_events = ledger.get_by_campaign("camp_001")
trial_events = ledger.get_by_trial("trial_001")

# Idempotent sync
sync = IdempotentSync(ledger)
result = sync.sync_events([campaign_event, trial_event])  # Duplicates ignored

# Verify consistency
consistency = sync.verify_campaign_consistency("camp_001")
print(consistency["is_consistent"])
```

## Development

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

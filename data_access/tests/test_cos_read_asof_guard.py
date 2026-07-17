from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_runtime import _guarded_read_asof


def test_legacy_read_asof_cannot_claim_event_pit():
    original = lambda dataset, **kwargs: {"dataset": dataset}
    with pytest.raises(ValidationError, match="read_cos_events_asof"):
        _guarded_read_asof(
            original,
            "ashare_stock_balance",
            {"as_of": "2024-01-01"},
        )
    result = _guarded_read_asof(
        original,
        "ashare_stock_daily",
        {"as_of": "2024-01-01"},
    )
    assert result["dataset"] == "ashare_stock_daily"

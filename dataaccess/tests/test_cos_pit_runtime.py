from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.cos_contract import require_cos_contract, validate_event_filters
from data_access.cos_runtime import _read_cos_events, _read_cos_events_asof


class _Dataset:
    def __init__(self, columns):
        self.schema = {column: "double" for column in columns}


class _Registry:
    def __init__(self, tables):
        self._tables = tables

    def get(self, dataset):
        return _Dataset(self._tables[dataset].columns)


class FakeEventStore:
    def __init__(self, tables):
        self.tables = tables
        self._registry = _Registry(tables)
        self.view_columns = []

    def sql(self, query, **kwargs):
        dataset = kwargs["read_datasets"][0]
        requested = kwargs["view_columns"][dataset]
        self.view_columns.append(requested)
        frame = self.tables[dataset]
        # A real view exposes exactly the authorized physical projection.
        # Missing declared columns are null in this bounded fake fixture.
        projected = frame.reindex(columns=requested)
        return pa.Table.from_pandas(projected, preserve_index=False)


def test_latest_period_does_not_roll_back_after_old_period_restatement():
    events = pd.DataFrame({
        "ticker": ["A", "A", "A", "B"],
        "filing_date": ["2024-05-01", "2024-08-01", "2024-09-01", "2024-05-03"],
        "period_end": ["2024-03-31", "2024-06-30", "2023-12-31", "2024-03-31"],
        "timeframe": ["quarterly"] * 4,
        "value": [10.0, 20.0, 999.0, 30.0],
    })
    decisions = pd.DataFrame({
        "instrument": ["A", "B", "A"],
        "decision_timestamp": ["2024-05-10", "2024-05-10", "2024-09-10"],
    })
    result = _read_cos_events_asof(
        FakeEventStore({"us_stock_balance": events}),
        "us_stock_balance",
        decisions,
        columns=["value"],
        event_filters={"timeframe": "quarterly"},
    )
    assert result["value"].tolist() == [10.0, 30.0, 20.0]
    assert result["instrument"].tolist() == ["A", "B", "A"]


def test_future_filings_hidden_and_stale_values_blank():
    events = pd.DataFrame({
        "ticker": ["A", "A"],
        "filing_date": ["2024-05-01", "2024-08-01"],
        "period_end": ["2024-03-31", "2024-06-30"],
        "timeframe": ["quarterly", "quarterly"],
        "value": [10.0, 20.0],
    })
    decisions = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "decision_timestamp": ["2024-04-15", "2024-05-03", "2025-12-31"],
    })
    result = _read_cos_events_asof(
        FakeEventStore({"us_stock_balance": events}),
        "us_stock_balance",
        decisions,
        columns=["value"],
        event_filters={"timeframe": "quarterly"},
        max_age_days=180,
    )
    assert pd.isna(result.loc[0, "value"])
    assert result.loc[1, "value"] == 10.0
    assert pd.isna(result.loc[2, "value"])
    assert result.loc[2, "fundamental_staleness_days"] > 180


def test_full_payload_and_required_timeframe():
    events = pd.DataFrame({
        "Symbol": ["000001.SZ"],
        "PubDate": ["2024-04-30"],
        "ReportPeriodEndDate": ["2024-03-31"],
        "UpdateTime": ["2024-04-30T08:00:00Z"],
        "TotalAssets": [100.0],
    })
    result = _read_cos_events(
        FakeEventStore({"ashare_stock_balance": events}),
        "ashare_stock_balance",
    )
    assert "TotalAssets" in result.columns
    with pytest.raises(ValidationError, match="timeframe"):
        validate_event_filters(require_cos_contract("us_stock_balance"), None)


def test_effective_event_rejects_lossy_one_row_asof():
    # us_stock_dividend 已升级为 strict-PIT；用仍为 effective_time_only 的 split
    # 验证「effective-only 事件禁止 generic latest-asof」的语义。
    events = pd.DataFrame({
        "ticker": ["A"],
        "execution_date": ["2024-01-01"],
        "id": ["x"],
    })
    decisions = pd.DataFrame({
        "instrument": ["A"],
        "decision_timestamp": ["2024-02-01"],
    })
    with pytest.raises(ValidationError, match="显式聚合"):
        _read_cos_events_asof(
            FakeEventStore({"us_stock_capital_split": events}),
            "us_stock_capital_split",
            decisions,
            allow_effective_time=True,
        )

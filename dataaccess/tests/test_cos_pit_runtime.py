from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import duckdb

from data_access.core.exceptions import ValidationError
from data_access.cos_contract import require_cos_contract, validate_event_filters
from data_access.cos_runtime import _read_cos_events, _read_cos_events_asof
from store import _period_selection_sql


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


def test_latest_period_empty_revision_order_preserves_pit_history_and_fails_closed_on_ambiguity(tmp_path):
    parquet = tmp_path / "us_financial_vintages.parquet"
    frame = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "filing_date": ["2024-05-01", "2024-08-01", "2024-08-01"],
        "period_end": ["2024-03-31", "2024-06-30", "2024-06-30"],
        "value": [10, 20, 21],
    })
    rows = f"SELECT * FROM read_parquet('{parquet.as_posix()}')"

    pq.write_table(
        pa.Table.from_pandas(frame.iloc[:2], preserve_index=False), parquet
    )
    query, _ = _period_selection_sql(
        rows,
        inst_col="ticker",
        period_col="period_end",
        knowledge_col="filing_date",
        selection="latest_period",
        period_is_text=True,
    )
    result = duckdb.sql(
        f"SELECT ticker, filing_date, period_end, value FROM ({query}) "
        "ORDER BY filing_date"
    ).fetchall()
    assert result == [
        ("A", "2024-05-01", "2024-03-31", 10),
        ("A", "2024-08-01", "2024-06-30", 20),
    ]

    asof = duckdb.sql(
        "WITH selected AS (" + query + "), decisions(ticker, decision_time) AS "
        "(VALUES ('A', DATE '2024-06-01'), ('A', DATE '2024-09-01')) "
        "SELECT d.decision_time, s.value FROM decisions AS d "
        "ASOF LEFT JOIN selected AS s ON d.ticker = s.ticker "
        "AND d.decision_time >= CAST(s.filing_date AS DATE) "
        "ORDER BY d.decision_time"
    ).fetchall()
    assert [row[1] for row in asof] == [10, 20]

    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet)
    query, _ = _period_selection_sql(
        rows,
        inst_col="ticker",
        period_col="period_end",
        knowledge_col="filing_date",
        selection="latest_period",
        period_is_text=True,
    )
    with pytest.raises(duckdb.Error, match="ambiguous latest PIT vintage"):
        duckdb.sql(query).fetchall()



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

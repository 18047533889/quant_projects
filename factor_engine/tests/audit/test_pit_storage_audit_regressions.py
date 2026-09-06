from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from factor_engine.pit_contract import AvailabilityPrecision, _market_visible_shift, _period_mask, select_visible_row_bundles
from factor_engine.storage.sources.data_access_source import HistoricalCoverageContract
from factor_engine.storage.sources.financial import FinancialFieldContract, MARKET_FUNDAMENTAL_FIELD_CONTRACTS, load_financial_row_bundle
from factor_engine.storage.sources.parquet_source import ParquetSource


def events():
    return pd.DataFrame({
        "instrument": ["A"] * 4,
        "period_end": ["2024-12-31", "2025-03-31", "2024-12-31", "2025-03-31"],
        "available_at": ["2025-02-01 10:00Z", "2025-05-01 10:00Z", "2025-06-01 10:00Z", "2025-07-01 10:00Z"],
        "revision_id": [1, 1, 2, 2],
        "value": [10.0, 20.0, 11.0, 21.0],
    })


def test_production_same_day_unknown_precision_rejected_for_noon_values():
    decisions = pd.DataFrame({"instrument": ["A"], "decision_timestamp": ["2025-02-02 12:00Z"]})
    with pytest.raises(ValueError, match="DECLARED"):
        select_visible_row_bundles(decisions, events().iloc[:1], available_policy="same_day", production=True, precision=AvailabilityPrecision.UNKNOWN)


def test_local_session_labels_localized_before_utc_conversion():
    shifted = _market_visible_shift(
        pd.Series([pd.Timestamp("2026-03-02 16:00", tz="America/New_York")]),
        pd.DatetimeIndex(["2026-03-02", "2026-03-03"]),
        market_timezone="America/New_York",
    )
    assert shifted.iloc[0] == pd.Timestamp("2026-03-03 05:00Z")


def test_quarter_mask_accepts_null_and_rejects_fractional_quarters():
    mask = _period_mask(pd.Series(pd.to_datetime(["2025-03-31"] * 4)), "quarterly_only", fiscal_quarter=pd.Series([1.0, float("nan"), 1.9, 4.0]))
    assert mask.tolist() == [True, False, False, True]


def test_empty_decision_grid_preserves_schema():
    decisions = pd.DataFrame({"instrument": pd.Series(dtype=str), "decision_timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
    out = select_visible_row_bundles(decisions, events())
    assert out.empty
    assert list(out.columns[:2]) == list(decisions.columns)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_all_coverage_evidence_must_be_finite_unit_interval(bad):
    assert HistoricalCoverageContract(field="x", coverage_ratio=bad).violations()


def test_requested_universe_is_stock_coverage_denominator():
    contract = HistoricalCoverageContract(field="x", coverage_ratio=1.0, coverage_by_stock={"A": 1.0}, min_stock_coverage_threshold=0.9, min_stock_coverage_quantile=0.9)
    problems = contract.violations(universe_stocks={"A", "B"})
    assert problems and "1/2" in problems[-1]


def test_parquet_conflicting_duplicate_key_is_fail_closed():
    source = ParquetSource("/unused", timestamp_column="ts", instrument_column="symbol")
    frame = pd.DataFrame({"ts": ["2025-01-01", "2025-01-01"], "symbol": ["A", "A"], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        source._split_into_series(frame, needed=["x"], actual_names=["x"])


def test_parquet_public_load_does_not_turn_manifest_failure_into_partial(monkeypatch):
    source = ParquetSource("/unused", timestamp_column="ts", instrument_column="symbol")
    monkeypatch.setattr(source, "_selected_files", lambda: [Path("good.parquet"), Path("bad.parquet")])
    monkeypatch.setattr(source, "_ensure_columns_resolved", lambda _path: None)
    monkeypatch.setattr(source, "_duckdb_read_batch", lambda *_args: (_ for _ in ()).throw(OSError("corrupt footer")))
    monkeypatch.setattr(source, "_duckdb_read_per_file", lambda *_args: (_ for _ in ()).throw(RuntimeError("manifest is incomplete")))
    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        source.load_columns(["x"])


def test_latest_period_state_does_not_regress_on_old_period_restatement():
    decisions = pd.DataFrame({"instrument": ["A"] * 4, "decision_timestamp": ["2025-03-01", "2025-05-02", "2025-06-02", "2025-07-02"]})
    out = select_visible_row_bundles(decisions, events(), available_policy="same_day", precision=AvailabilityPrecision.TIMESTAMP_SECOND)
    assert out["value"].tolist() == [10.0, 20.0, 20.0, 21.0]


def test_state_sweep_matches_independent_pointwise_latest_period_oracle():
    source = _events = events().sample(frac=1.0, random_state=42).reset_index(drop=True)
    decisions = pd.DataFrame({
        "instrument": ["A"] * 8,
        "decision_timestamp": pd.date_range("2025-02-01", "2025-07-02", periods=8, tz="UTC"),
    })
    actual = select_visible_row_bundles(
        decisions, source, available_policy="same_day",
        precision=AvailabilityPrecision.TIMESTAMP_SECOND,
    )
    expected = []
    normalized = source.assign(
        available_at=pd.to_datetime(source["available_at"], utc=True),
        period_end=pd.to_datetime(source["period_end"], utc=True),
    )
    for decision_at in decisions["decision_timestamp"]:
        visible = normalized.loc[normalized["available_at"] <= decision_at]
        if visible.empty:
            expected.append(pd.NA)
            continue
        latest_period = visible["period_end"].max()
        vintage = visible.loc[visible["period_end"] == latest_period].sort_values(
            ["available_at", "revision_id"], kind="stable"
        ).iloc[-1]
        expected.append(vintage["value"])
    assert actual["value"].tolist() == expected


def test_same_visibility_matches_canonical_revision_tie_break():
    source = pd.DataFrame({
        "instrument": ["A", "A"],
        "period_end": ["2025-03-31", "2025-03-31"],
        "available_at": ["2025-05-01 09:00Z", "2025-05-01 18:00Z"],
        "revision_id": [99, 1],
        "value": [9.0, 18.0],
    })
    decisions = pd.DataFrame({
        "instrument": ["A"],
        "decision_timestamp": ["2025-05-02 12:00Z"],
    })
    actual = select_visible_row_bundles(
        decisions, source, available_policy="next_trading_day",
        market_calendar=pd.DatetimeIndex(["2025-05-01", "2025-05-02"]),
        market_timezone="UTC",
    )
    # Reference algorithm first replaces available_at with market_visible_at,
    # selects latest period, then sorts same-session candidates by
    # (available_at, revision_id).  Both visible times are equal, so rev=99 wins.
    visible = source.copy()
    visible["available_at"] = pd.Timestamp("2025-05-02", tz="UTC")
    expected = visible.sort_values(
        ["available_at", "revision_id"], kind="stable"
    ).iloc[-1]["value"]
    assert expected == 9.0
    assert actual.loc[0, "value"] == expected


def test_nat_decision_is_empty_and_does_not_reorder_or_leak_future_state():
    decisions = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "decision_timestamp": [pd.Timestamp("2025-07-02", tz="UTC"), pd.NaT, pd.Timestamp("2025-03-01", tz="UTC")],
    })
    actual = select_visible_row_bundles(
        decisions, events().sample(frac=1.0, random_state=7),
        available_policy="same_day",
        precision=AvailabilityPrecision.TIMESTAMP_SECOND,
    )
    assert actual.loc[0, "value"] == 21.0
    assert pd.isna(actual.loc[1, "value"])
    assert actual.loc[2, "value"] == 10.0


def test_market_contract_authoritative_and_unknown_production_rejects(monkeypatch):
    monkeypatch.setitem(MARKET_FUNDAMENTAL_FIELD_CONTRACTS, ("us", "value"), FinancialFieldContract("StockIncome", "value", "annual_only", True))
    decisions = pd.DataFrame({"instrument": ["A"], "decision_timestamp": ["2025-07-02"]})
    with pytest.raises(ValueError, match="annual_only"):
        load_financial_row_bundle(decisions, events(), ["value"], market="us", selector="quarterly_only")
    with pytest.raises(ValueError, match="unknown.*contract"):
        load_financial_row_bundle(decisions, events(), ["value"], market="unknown", production=True)


def test_production_requires_market_even_when_legacy_ashare_contract_matches(monkeypatch):
    from factor_engine.storage.sources.financial import FUNDAMENTAL_FIELD_CONTRACTS
    monkeypatch.setitem(FUNDAMENTAL_FIELD_CONTRACTS, "value", FinancialFieldContract("StockIncome", "value"))
    decisions = pd.DataFrame({"instrument": ["A"], "decision_timestamp": ["2025-07-02"]})
    with pytest.raises(ValueError, match="explicit non-empty market"):
        load_financial_row_bundle(decisions, events(), ["value"], production=True)

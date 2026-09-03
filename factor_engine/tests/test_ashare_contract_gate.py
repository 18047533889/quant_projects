# -*- coding: utf-8 -*-
"""P0#10: A-share data-contract gate tests (GO prompt §110 item 10)."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.ashare_contract_gate import (
    AShareContractViolation,
    assert_contract,
    check_membership_pit,
    check_minute_sessions,
    check_pit_no_lookahead,
    check_return_units,
    validate_ashare_return_unit_contracts,
    validate_contract,
)


def _minute_index(*times: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("Asia/Shanghai")


# ---------------------------------------------------------------------------
# (a) return-unit mislabel
# ---------------------------------------------------------------------------
def test_return_unit_decimal_labeled_bp_fails() -> None:
    # A decimal return 0.05 labeled bp is a hard failure.
    verdict = check_return_units(
        {"return_fields": [{"name": "ret_10d", "unit": "basis_point", "value": 0.05}]}
    )
    assert verdict.failures
    assert "basis_point" in verdict.failures[0]


def test_return_unit_percent_labeled_ratio_fails() -> None:
    # A percent return 5.0 labeled ratio is a hard failure.
    verdict = check_return_units({"return_fields": {"ret_5d": {"unit": "ratio", "value": 5.0}}})
    assert verdict.failures


def test_return_unit_correct_bp_passes() -> None:
    # Adj-table Return column in basis points is the repo convention.
    verdict = check_return_units({"return_fields": {"Return": "basis_point"}})
    assert verdict.ok


def test_return_unit_unlabeled_warns() -> None:
    verdict = check_return_units({"return_fields": [{"name": "momentum", "unit": None}]})
    assert not verdict.failures
    assert verdict.warnings


# ---------------------------------------------------------------------------
# (b) minute / session contract
# ---------------------------------------------------------------------------
def test_minute_panel_with_lunch_gap_passes() -> None:
    # 09:30-11:30 then 13:00-15:00 — no lunch-gap bars.
    idx = _minute_index(
        "2026-09-01 09:30:00", "2026-09-01 10:00:00", "2026-09-01 11:30:00",
        "2026-09-01 13:00:00", "2026-09-01 14:00:00", "2026-09-01 15:00:00",
    )
    verdict = check_minute_sessions({"minute_index": idx})
    assert verdict.ok


def test_minute_panel_with_lunch_gap_bar_fails() -> None:
    # A 12:00 bar bridges the lunch gap.
    idx = _minute_index("2026-09-01 09:30:00", "2026-09-01 12:00:00")
    verdict = check_minute_sessions({"minute_index": idx})
    assert verdict.failures
    assert "outside the A-share session" in verdict.failures[0]


def test_minute_panel_with_weekend_date_fails() -> None:
    # 2026-09-05 is a Saturday.
    idx = _minute_index("2026-09-05 10:00:00")
    verdict = check_minute_sessions({"minute_index": idx})
    assert verdict.failures
    assert "weekend" in verdict.failures[0]


def test_minute_panel_overnight_bar_fails() -> None:
    # 16:00 is after the afternoon close.
    idx = _minute_index("2026-09-01 16:00:00")
    verdict = check_minute_sessions({"minute_index": idx})
    assert verdict.failures


def test_minute_panel_off_boundary_warns() -> None:
    # 09:31 is inside a session but not on a boundary (240-bar mode first bar).
    idx = _minute_index("2026-09-01 09:31:00")
    verdict = check_minute_sessions({"minute_index": idx})
    assert not verdict.failures
    assert verdict.warnings


# ---------------------------------------------------------------------------
# (c) PIT no-lookahead
# ---------------------------------------------------------------------------
def test_pit_violation_future_join_caught() -> None:
    # Announcement dated AFTER the decision timestamp is a future join.
    verdict = check_pit_no_lookahead(
        {
            "pit_columns": [
                {
                    "name": "eps",
                    "asof": pd.Timestamp("2026-09-10"),
                    "decision": pd.Timestamp("2026-09-01"),
                }
            ]
        }
    )
    assert verdict.failures
    assert "PIT violation" in verdict.failures[0]


def test_pit_ok_when_asof_le_decision() -> None:
    verdict = check_pit_no_lookahead(
        {
            "pit_columns": [
                {
                    "name": "eps",
                    "asof": pd.Timestamp("2026-08-20"),
                    "decision": pd.Timestamp("2026-09-01"),
                }
            ]
        }
    )
    assert verdict.ok


def test_pit_null_timestamp_fails_closed() -> None:
    verdict = check_pit_no_lookahead(
        {
            "pit_columns": [
                {"name": "eps", "asof": pd.NaT, "decision": pd.Timestamp("2026-09-01")}
            ]
        }
    )
    assert verdict.failures


# ---------------------------------------------------------------------------
# (d) membership / relation PIT
# ---------------------------------------------------------------------------
def test_current_membership_join_caught() -> None:
    # Industry join with no PIT key uses current membership.
    verdict = check_membership_pit({"membership_joins": {"industry": False}})
    assert verdict.failures
    assert "point-in-time" in verdict.failures[0]


def test_membership_pit_key_present_passes() -> None:
    verdict = check_membership_pit({"membership_joins": {"industry": True}})
    assert verdict.ok


def test_membership_pit_key_not_join_key_warns() -> None:
    verdict = check_membership_pit(
        {"membership_joins": [{"name": "index", "has_pit_key": True, "pit_key_is_join_key": False}]}
    )
    assert not verdict.failures
    assert verdict.warnings


# ---------------------------------------------------------------------------
# aggregate entry points
# ---------------------------------------------------------------------------
def test_validate_contract_aggregates_failures() -> None:
    verdict = validate_contract(
        "op_a",
        {
            "return_fields": {"ret_10d": "basis_point"},
            "minute_index": _minute_index("2026-09-01 12:00:00"),
            "pit_columns": [
                {"name": "eps", "asof": pd.Timestamp("2026-09-10"), "decision": pd.Timestamp("2026-09-01")}
            ],
            "membership_joins": {"industry": False},
        },
    )
    assert verdict.failures
    # Every failure is prefixed with the operator name.
    assert all(f.startswith("op_a:") for f in verdict.failures)


def test_assert_contract_raises_on_failure() -> None:
    with pytest.raises(AShareContractViolation):
        assert_contract("op_b", {"return_fields": {"ret": {"unit": "dimensionless", "value": 0.05}}})


def test_assert_contract_returns_verdict_when_ok() -> None:
    verdict = assert_contract("op_c", {"return_fields": {"Return": "basis_point"}})
    assert verdict.ok


# ---------------------------------------------------------------------------
# Analyzer compile-path hook (P0#10 non-bypassable gate)
# ---------------------------------------------------------------------------
def test_analyzer_hook_rejects_mislabeled_return_field() -> None:
    from factor_engine.fields.spec import FieldSpec

    # A return-named field labeled dimensionless is a hard failure.
    spec = FieldSpec(
        name="ret_10d",
        table="StockDailyBarAdj",
        source_name="ret_10d",
        unit="dimensionless",
    )
    errors = validate_ashare_return_unit_contracts({"ret_10d": spec})
    assert errors
    assert "ret_10d" in errors[0]


def test_analyzer_hook_accepts_bp_return_field() -> None:
    from factor_engine.fields.spec import FieldSpec

    spec = FieldSpec(
        name="Return",
        table="StockDailyBarAdj",
        source_name="Return",
        unit="basis_point",
    )
    errors = validate_ashare_return_unit_contracts({"Return": spec})
    assert errors == []

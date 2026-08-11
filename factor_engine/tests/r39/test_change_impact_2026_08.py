# -*- coding: utf-8 -*-
"""R39 P0 fixes for Change Impact DAG + HistoryTransform fail-closed.

Covers:
  - P0-045: shared DAG node synthetic node-id stability (same object -> same id).
  - P0-046: source-identity column matching (dataset/field/market/grain/
    price_basis/revision_identity) — a revision to dataset B's ``close`` must
    NOT dirty dataset A's ``close``.
  - P0-047: calendar-based forward window (CNY/US holidays skipped) via a
    frozen :class:`CalendarSnapshot`, no ``pandas.offsets.BDay`` guess.
  - P0-048: contract failure -> unbounded full recompute, no operator-name
    heuristic fallback (ts_/ewm prefix guessing).
  - P0-049: ``HistoryTransform.kind`` invariant — unknown kind fails closed
    (never a silent 0 lookback).
"""
from __future__ import annotations

import pandas as pd
import pytest

from planner.logical_plan import PlanNode


# ---------------------------------------------------------------------------
# P0-045: shared-node synthetic id stability
# ---------------------------------------------------------------------------
def test_shared_node_gets_one_stable_id():
    from runtime.change_impact import _walk_nodes

    shared_col = PlanNode("column", attrs={"name": "close"})
    left = PlanNode("ts_mean", [shared_col], attrs={"window": 5})
    right = PlanNode("ts_delay", [shared_col], attrs={"n": 2})
    root = PlanNode("add", [left, right])

    nodes = _walk_nodes(root)
    ids = [nid for nid, *_ in nodes]
    # Every object appears exactly once with a single assigned id (no duplicates).
    assert len(ids) == len(set(ids)), f"duplicate node ids assigned: {ids}"

    col_nodes = [n for n in nodes if n[1] == "column"]
    assert len(col_nodes) == 1, "shared column must be walked exactly once"
    col_id = col_nodes[0][0]

    by_op = {n[1]: n for n in nodes}
    # Both parents reference the SAME canonical id for the shared child.
    assert by_op["ts_mean"][3] == (col_id,)
    assert by_op["ts_delay"][3] == (col_id,)


def test_shared_column_impact_reaches_both_branches():
    """End-to-end: the dropped-edge bug hid ts_delay's affected window."""
    from runtime.change_impact import compute_change_impact

    shared_col = PlanNode("column", attrs={"name": "close"})
    left = PlanNode("ts_mean", [shared_col], attrs={"window": 5})
    right = PlanNode("ts_delay", [shared_col], attrs={"n": 2})
    root = PlanNode("add", [left, right])

    affected = compute_change_impact(root, field="close", changed_start="2024-01-10")
    by_op = {w.op: w for w in affected}
    assert "add" in by_op, "root must be affected"
    assert "ts_delay" in by_op, "shared child's change must reach BOTH branches"
    assert by_op["ts_mean"].end == "2024-01-16"   # window-1 = 4 bdays
    assert by_op["ts_delay"].end == "2024-01-12"  # lag n=2 -> 2 bdays
    assert by_op["add"].end == "2024-01-16"       # max of both branches


def test_explicit_node_id_is_preserved_for_shared_node():
    from runtime.change_impact import _walk_nodes

    shared_col = PlanNode(
        "column", attrs={"name": "close"}, node_id="col:close"
    )
    root = PlanNode(
        "add",
        [
            PlanNode("ts_mean", [shared_col], attrs={"window": 5}),
            PlanNode("ts_delay", [shared_col], attrs={"n": 2}),
        ],
    )
    nodes = _walk_nodes(root)
    col_nodes = [n for n in nodes if n[1] == "column"]
    assert len(col_nodes) == 1
    assert col_nodes[0][0] == "col:close"
    by_op = {n[1]: n for n in nodes}
    assert by_op["ts_mean"][3] == ("col:close",)
    assert by_op["ts_delay"][3] == ("col:close",)


# ---------------------------------------------------------------------------
# P0-046: source-identity column matching
# ---------------------------------------------------------------------------
def test_revision_of_dataset_b_close_does_not_dirty_dataset_a():
    from runtime.change_impact import ColumnIdentity, compute_change_impact

    col_a = PlanNode("column", attrs={"name": "close", "source_table": "dataset_a"})
    col_b = PlanNode("column", attrs={"name": "close", "source_table": "dataset_b"})
    plan = PlanNode("add", [col_a, col_b])

    # Revision of dataset B's close -> the add root IS affected.
    affected = compute_change_impact(
        plan,
        field="close",
        changed_start="2024-01-10",
        column_identity=ColumnIdentity(dataset="dataset_b", field="close"),
    )
    assert len(affected) == 1
    assert affected[0].op == "add"
    assert affected[0].start == affected[0].end == "2024-01-10"

    # A change to dataset A's field "ret" (which this plan never reads) dirtied nothing.
    affected2 = compute_change_impact(
        plan,
        field="ret",
        changed_start="2024-01-10",
        column_identity=ColumnIdentity(dataset="dataset_a", field="ret"),
    )
    assert affected2 == []


def test_dataset_a_only_plan_unaffected_by_dataset_b_change():
    from runtime.change_impact import ColumnIdentity, compute_change_impact

    col_a = PlanNode("column", attrs={"name": "close", "source_table": "dataset_a"})
    plan = PlanNode("ts_mean", [col_a], attrs={"window": 5})

    # Change to dataset B's close must not touch a plan that reads only A.
    affected = compute_change_impact(
        plan,
        field="close",
        changed_start="2024-01-10",
        column_identity=ColumnIdentity(dataset="dataset_b", field="close"),
    )
    assert affected == []

    # Legacy name-only matching (no column_identity) still dirties it.
    legacy = compute_change_impact(plan, field="close", changed_start="2024-01-10")
    assert len(legacy) == 1
    assert legacy[0].op == "ts_mean"


def test_full_identity_includes_grain_price_basis_revision():
    from runtime.change_impact import ColumnIdentity, compute_change_impact

    col = PlanNode(
        "column",
        attrs={"name": "close", "source_table": "us_daily", "price_basis": "close"},
        semantic_attrs={"market": "US", "grain": ("day",), "source_vintage": "rev-2"},
    )
    plan = PlanNode("ts_mean", [col], attrs={"window": 5})

    change = ColumnIdentity(
        dataset="us_daily",
        field="close",
        market="US",
        grain="day",
        price_basis="close",
        revision_identity="rev-2",
    )
    affected = compute_change_impact(
        plan, field="close", changed_start="2024-01-10", column_identity=change
    )
    assert len(affected) == 1
    assert affected[0].op == "ts_mean"

    # Wrong revision -> no match (revision identity is load-bearing).
    wrong_rev = ColumnIdentity(
        dataset="us_daily",
        field="close",
        market="US",
        grain="day",
        price_basis="close",
        revision_identity="rev-3",
    )
    affected2 = compute_change_impact(
        plan, field="close", changed_start="2024-01-10", column_identity=wrong_rev
    )
    assert affected2 == []


def test_column_identity_accepts_dict():
    from runtime.change_impact import compute_change_impact

    col_b = PlanNode("column", attrs={"name": "close", "source_table": "dataset_b"})
    plan = PlanNode("ts_mean", [col_b], attrs={"window": 5})
    affected = compute_change_impact(
        plan,
        field="close",
        changed_start="2024-01-10",
        column_identity={"dataset": "dataset_b", "field": "close"},
    )
    assert len(affected) == 1
    with pytest.raises(TypeError):
        compute_change_impact(
            plan,
            field="close",
            changed_start="2024-01-10",
            column_identity={"bogus_field": "x"},
        )


def test_source_ref_encoded_column_matches_table_identity():
    from api.source_ref import make_source_ref, encode_source_ref
    from runtime.change_impact import ColumnIdentity, compute_change_impact

    ref_a = encode_source_ref(make_source_ref("StockDailyBar", "close", market="US"))
    ref_b = encode_source_ref(
        make_source_ref("StockDailyBar", "close", market="US", dataset="cn_bar")
    )
    col_a = PlanNode("column", attrs={"name": ref_a})
    col_b = PlanNode("column", attrs={"name": ref_b})
    plan = PlanNode("add", [col_a, col_b])

    affected = compute_change_impact(
        plan,
        field="close",
        changed_start="2024-01-10",
        column_identity=ColumnIdentity(dataset="cn_bar", field="close"),
    )
    # Only the cn_bar column matches -> add root is affected, nothing else.
    assert len(affected) == 1
    assert affected[0].op == "add"


# ---------------------------------------------------------------------------
# P0-047: calendar-based forward window (real sessions, holidays skipped)
# ---------------------------------------------------------------------------
def _holiday_calendar():
    from storage.trading_calendar import TradingCalendar

    days = list(pd.bdate_range("2024-01-02", "2024-01-31"))
    # MLK Day 2024-01-15 (Mon) is NOT a US trading session.
    days = [d for d in days if pd.Timestamp(d).normalize() != pd.Timestamp("2024-01-15")]
    return TradingCalendar(days, anchor_policy="exact_trade_day")


def test_calendar_forward_window_skips_holiday():
    from runtime.change_impact import CalendarSnapshot, compute_change_impact

    plan = PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 5})
    snapshot = CalendarSnapshot(_holiday_calendar())

    affected = compute_change_impact(
        plan, field="close", changed_start="2024-01-10", calendar=snapshot
    )
    # window=5 -> impact 4 real sessions: 01-10, 11, 12, 16, 17 -> 2024-01-17.
    assert affected[0].end == "2024-01-17"

    # The pandas BDay fallback (which treats the MLK holiday as a work day)
    # gives 2024-01-16 — proving the calendar is load-bearing.
    fallback = compute_change_impact(plan, field="close", changed_start="2024-01-10")
    assert fallback[0].end == "2024-01-16"


def test_calendar_forward_window_skips_cny_holiday():
    from runtime.change_impact import CalendarSnapshot, compute_change_impact

    days = list(pd.bdate_range("2024-02-05", "2024-02-23"))
    # Spring Festival 2024: no sessions 2024-02-09 .. 2024-02-16.
    holiday = {
        pd.Timestamp(d).normalize()
        for d in pd.bdate_range("2024-02-09", "2024-02-16")
    }
    days = [d for d in days if pd.Timestamp(d).normalize() not in holiday]
    snapshot = CalendarSnapshot(
        type(
            "TradingCalendar",
            (),
            {"days": days, "offset": None},
        )()
    )
    plan = PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 6})
    # impact = 5 sessions starting 2024-02-05: 05, 06, 07, 08, 19, 20 -> 2024-02-20.
    affected = compute_change_impact(
        plan, field="close", changed_start="2024-02-05", calendar=snapshot
    )
    assert affected[0].end == "2024-02-20"


def test_calendar_snapshot_accepts_raw_trading_calendar():
    from runtime.change_impact import compute_change_impact

    plan = PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 5})
    # A raw TradingCalendar is coerced to a CalendarSnapshot internally.
    affected = compute_change_impact(
        plan, field="close", changed_start="2024-01-10", calendar=_holiday_calendar()
    )
    assert affected[0].end == "2024-01-17"


# ---------------------------------------------------------------------------
# P0-048: contract failure -> unbounded, no name-guess heuristic
# ---------------------------------------------------------------------------
def test_contract_failure_is_unbounded_not_name_guess(monkeypatch):
    import runtime.execution_contract as ec
    from runtime import change_impact as ci

    def _boom(op, params=None, *, production=False):
        raise RuntimeError("contract resolver down")

    monkeypatch.setattr(ec, "forward_impact", _boom)
    # ts_mean + window=5 would have been guessed as window-1 = 4 (finite) by the
    # old name heuristic.  On contract failure we must get UNBOUNDED full recompute.
    plan = PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 5})
    affected = ci.compute_change_impact(plan, field="close", changed_start="2024-01-10")
    assert len(affected) == 1
    w = affected[0]
    assert w.op == "ts_mean"
    assert w.unbounded is True
    assert w.end is None


def test_unresolvable_window_is_unbounded_not_truncated():
    from runtime.change_impact import compute_change_impact

    # A fractional window is UNKNOWN history — forward impact must be unbounded,
    # never silently floored to int(5.9)-1 = 4.
    plan = PlanNode("ts_mean", [PlanNode("column", attrs={"name": "close"})], attrs={"window": 5.9})
    affected = compute_change_impact(plan, field="close", changed_start="2024-01-10")
    assert len(affected) == 1
    assert affected[0].unbounded is True
    assert affected[0].end is None


def test_own_impact_returns_none_on_contract_error(monkeypatch):
    import runtime.execution_contract as ec
    from runtime.change_impact import _own_impact

    def _boom(op, params=None, *, production=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "forward_impact", _boom)
    assert _own_impact("ts_mean", {"window": 5}) is None  # unbounded


# ---------------------------------------------------------------------------
# P0-049: HistoryTransform kind invariant
# ---------------------------------------------------------------------------
def test_history_transform_unknown_kind_raises_at_init():
    from runtime.execution_contract import HistoryTransform

    with pytest.raises(ValueError):
        HistoryTransform(kind="windows")  # typo of "window"
    with pytest.raises(ValueError):
        HistoryTransform(kind="lagg")
    # Valid kinds still construct.
    HistoryTransform(kind="identity")
    HistoryTransform(kind="window", params=("window",))
    HistoryTransform(kind="lag", fixed=1)
    HistoryTransform(kind="compound", fn=lambda c, p: 0)


def test_history_transform_compound_requires_fn():
    from runtime.execution_contract import HistoryTransform

    with pytest.raises(ValueError):
        HistoryTransform(kind="compound")


def test_apply_transform_unknown_kind_fails_closed():
    from runtime.execution_contract import HistoryTransform, _apply_transform

    # Bypass __post_init__ to simulate a deserialized/bad transform.
    bad = HistoryTransform.__new__(HistoryTransform)
    object.__setattr__(bad, "kind", "bogus")
    object.__setattr__(bad, "params", ())
    object.__setattr__(bad, "fixed", None)
    object.__setattr__(bad, "fn", None)
    with pytest.raises(Exception):
        _apply_transform(bad, "ts_fake", {})

    # A valid window transform still applies normally.
    good = HistoryTransform(kind="window", params=("window",))
    assert _apply_transform(good, "ts_mean", {"window": 5}) == 4

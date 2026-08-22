# -*- coding: utf-8 -*-
"""Review item R9-P0-010 — ClockContract regression tests (2026-08).

ClockSemantics alone cannot express operators that *change* cadence between
inputs and output (e.g. ``event_historical_response_mean``: event inputs,
daily trading-bar output; ``fin_announcement_lag``: filing-clock inputs,
daily evaluation).  Production clock resolution must come from an explicit
per-canonical ClockContract registry, NEVER from ``name.startswith('fin_' /
'event_' / 'ts_')`` prefix guessing.

Permanent regression tests:
* explicit contract for ``event_historical_response_mean`` (input != output);
* explicit contract for the financial as-of operator ``fin_announcement_lag``
  (input update clock = filing, output evaluation clock = daily);
* unlisted canonicals get a conservative same-clock default, with no
  input/output split guessed from a ``fin_``/``event_`` prefix;
* legacy single-clock API (``ClockSemantics`` / ``clock_for`` / ``declare_clock``
  / ``clock_registry``) remains intact for backward compatibility.
"""
from __future__ import annotations

import pytest

from planner.clock_semantics import (
    ANY_INPUT,
    ClockContract,
    ClockSemantics,
    clock_contract_for,
    clock_for,
    clock_registry,
    contract_registry,
    declare_clock,
    declare_contract,
)


# ---------------------------------------------------------------------------
# Explicit event -> daily contract
# ---------------------------------------------------------------------------


def test_clock_contract_event_historical_response_mean_event_input_daily_output():
    contract = clock_contract_for("event_historical_response_mean")

    assert isinstance(contract, ClockContract)
    # event inputs sit on the EVENT clock ...
    assert contract.input_clocks["response"] == ClockSemantics.EVENT
    assert contract.input_clocks["event"] == ClockSemantics.EVENT
    # ... but the output is evaluated once per daily trading bar.
    assert contract.output_clock == ClockSemantics.TRADING_BAR
    # the whole point: input clock != output clock.
    assert contract.output_clock != contract.input_clocks["event"]
    assert contract.grain_transform == "event->daily"
    assert contract.update_trigger == "event_detect"


def test_clock_contract_input_clock_helper():
    contract = clock_contract_for("event_historical_response_mean")
    assert contract.input_clock("event") == ClockSemantics.EVENT
    assert contract.input_clock("response") == ClockSemantics.EVENT
    # unknown input names conservatively fall back to the output clock.
    assert contract.input_clock("unlisted_input") == ClockSemantics.TRADING_BAR


# ---------------------------------------------------------------------------
# Explicit financial as-of contract (filing input, daily output)
# ---------------------------------------------------------------------------


def test_clock_contract_fin_announcement_lag_filing_input_daily_output():
    contract = clock_contract_for("fin_announcement_lag")

    assert isinstance(contract, ClockContract)
    # inputs update only on publication -> fiscal/filing clock.
    assert contract.input_clocks["period_end_date"] == ClockSemantics.FISCAL_PERIOD
    assert contract.input_clocks["pub_date"] == ClockSemantics.FISCAL_PERIOD
    # the output lag is evaluated on the daily trading-bar clock.
    assert contract.output_clock == ClockSemantics.TRADING_BAR
    # input update clock != output evaluation clock.
    assert contract.output_clock != contract.input_clocks["pub_date"]
    assert contract.grain_transform == "filing->daily"
    assert contract.update_trigger == "filing"


# ---------------------------------------------------------------------------
# Unlisted canonicals: conservative same-clock default, no prefix guessing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "fin_future_unclassified_operator",
        "event_future_unclassified_operator",
        "ts_future_unclassified_operator",
        "totally_unknown_operator",
    ],
)
def test_clock_contract_unlisted_is_conservative_same_clock_no_prefix_guess(name):
    contract = clock_contract_for(name)

    assert isinstance(contract, ClockContract)
    # No input/output split is ever guessed from a fin_/event_/ts_ prefix:
    # an unlisted fin_* operator must NOT come back with filing->daily.
    assert contract.grain_transform == "same"
    assert contract.update_trigger == "unknown"
    assert contract.output_clock == ClockSemantics.UNKNOWN
    # the "*" wildcard records that every input shares the (unknown) clock.
    assert contract.input_clocks[ANY_INPUT] == ClockSemantics.UNKNOWN
    # same-clock: every input clock equals the output clock.
    assert all(
        clock == contract.output_clock for clock in contract.input_clocks.values()
    )
    # the helper resolves any name to that same conservative clock.
    assert contract.input_clock("x") == contract.output_clock


# ---------------------------------------------------------------------------
# Declaration / registry surface + legacy backward compatibility
# ---------------------------------------------------------------------------


def test_clock_contract_declare_and_registry_view():
    contract = ClockContract(
        input_clocks={"a": ClockSemantics.EVENT},
        output_clock=ClockSemantics.TRADING_BAR,
        grain_transform="event->daily",
        update_trigger="event_detect",
    )
    declare_contract("r9_test_my_contract_op", contract)
    assert clock_contract_for("r9_test_my_contract_op").output_clock == ClockSemantics.TRADING_BAR
    assert clock_contract_for("r9_test_my_contract_op").input_clocks["a"] == ClockSemantics.EVENT

    view = contract_registry()
    assert view["event_historical_response_mean"]["output_clock"] == "trading_bar"
    assert view["event_historical_response_mean"]["input_clocks"]["event"] == "event"
    assert view["fin_announcement_lag"]["grain_transform"] == "filing->daily"
    assert view["fin_announcement_lag"]["update_trigger"] == "filing"
    # every explicit contract is round-trippable through the resolver.
    for name in view:
        assert clock_contract_for(name).output_clock.value == view[name]["output_clock"]

    with pytest.raises(TypeError):
        declare_contract("r9_test_bad", "not-a-contract")


def test_clock_contract_backward_compat_legacy_single_clock_api():
    # The legacy single-clock API must keep its exact prior behaviour.
    assert clock_for("ts_mean") == ClockSemantics.TRADING_BAR
    assert clock_for("event_response_effective_events") == ClockSemantics.EVENT
    assert clock_for("fin_revision_count") == ClockSemantics.FISCAL_PERIOD
    assert clock_for("rank") == ClockSemantics.OBSERVATION

    declare_clock("ts_my_event_op", ClockSemantics.EVENT)
    assert clock_for("ts_my_event_op") == ClockSemantics.EVENT

    reg = clock_registry()
    assert reg["ts_mean"] == "trading_bar"
    assert reg["intraday_slot"] == "session_slot"
    for name, value in reg.items():
        assert clock_for(name).value == value

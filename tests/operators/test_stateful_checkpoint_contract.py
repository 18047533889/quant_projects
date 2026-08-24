# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.stateful_contract import (
    StateCheckpoint,
    StatefulCheckpointRegistry,
    StatefulContractError,
)


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def _ema_checkpoint():
    identity = {"dataset": "prices", "adjustment": "split", "frequency": "1d"}
    checkpoint = StatefulCheckpointRegistry.create_checkpoint(
        "ts_ema",
        instrument="AAPL",
        as_of=datetime(2026, 7, 16, 20, 0, tzinfo=timezone.utc),
        # R5-09: ts_ema checkpoint carries the pandas-EWM state tuple.
        state={"ema": {"weighted_avg": 203.5, "old_wt": 1.0, "valid_count": 1},
               "last_timestamp": "2026-07-16T20:00:00+00:00"},
        input_identity=identity,
    )
    return checkpoint, identity


def test_segmented_stateful_execution_requires_checkpoint() -> None:
    with pytest.raises(StatefulContractError, match="requires a validated checkpoint"):
        StatefulCheckpointRegistry.require_for_segment(
            "ts_ema",
            starts_at_dataset_origin=False,
            checkpoint=None,
        )
    StatefulCheckpointRegistry.require_for_segment(
        "ts_ema",
        starts_at_dataset_origin=True,
        checkpoint=None,
    )


def test_checkpoint_roundtrip_and_fingerprint_validation() -> None:
    checkpoint, identity = _ema_checkpoint()
    restored = StateCheckpoint.from_json(checkpoint.to_json())
    assert restored == checkpoint
    StatefulCheckpointRegistry.require_for_segment(
        "ts_ema",
        starts_at_dataset_origin=False,
        checkpoint=restored,
        input_identity=identity,
        expected_instrument="AAPL",
    )
    with pytest.raises(StatefulContractError, match="fingerprint"):
        StatefulCheckpointRegistry.validate(
            restored,
            input_identity={"dataset": "different"},
        )


def test_checkpoint_semantic_and_schema_versions_are_strict() -> None:
    checkpoint, identity = _ema_checkpoint()
    wrong_schema = StateCheckpoint(
        **{**checkpoint.__dict__, "state_schema_version": "ema_state.v0"}
    )
    with pytest.raises(StatefulContractError, match="schema mismatch"):
        StatefulCheckpointRegistry.validate(wrong_schema, input_identity=identity)

    wrong_semantics = StateCheckpoint(
        **{**checkpoint.__dict__, "semantic_version": "0.9"}
    )
    with pytest.raises(StatefulContractError, match="semantic version mismatch"):
        StatefulCheckpointRegistry.validate(wrong_semantics, input_identity=identity)


def test_checkpoint_contracts_are_attached_to_runtime_catalog() -> None:
    for canonical in (
        "ts_ema",
        "RSI_WILDER",
        "ATR_WILDER",
        "ADX",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
    ):
        spec = StatefulCheckpointRegistry.get(canonical)
        assert spec is not None
        catalog = OperatorRegistry.catalog()[canonical]
        assert catalog["stateful"] is True
        assert catalog["segmented_execution_requires_checkpoint"] is True
        assert catalog["checkpoint_contract"]["state_schema_version"] == spec.state_schema_version


def test_checkpoint_missing_state_is_rejected() -> None:
    with pytest.raises(StatefulContractError, match="missing fields"):
        StatefulCheckpointRegistry.create_checkpoint(
            "ATR_WILDER",
            instrument="AAPL",
            as_of="2026-07-16T20:00:00Z",
            state={"atr": 2.0},
            input_identity={"dataset": "prices"},
        )

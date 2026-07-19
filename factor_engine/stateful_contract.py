# -*- coding: utf-8 -*-
"""Checkpoint contracts for stateful operators and segmented execution."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import math


class StatefulContractError(ValueError):
    pass


@dataclass(frozen=True)
class StatefulOperatorSpec:
    canonical: str
    state_schema_version: str
    semantic_version: str
    minimum_history: int
    checkpoint_fields: tuple[str, ...]
    missing_policy: str
    dependencies: tuple[str, ...] = ()
    checkpoint_required_for_segmented: bool = True
    segmented_execution_supported: bool = True


@dataclass(frozen=True)
class StateCheckpoint:
    operator: str
    instrument: str
    as_of: str
    state_schema_version: str
    semantic_version: str
    input_fingerprint: str
    state: Mapping[str, Any]

    def to_json(self) -> str:
        def standard_json(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, Mapping):
                return {str(k): standard_json(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [standard_json(v) for v in value]
            return value

        return json.dumps(
            standard_json(asdict(self)), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "StateCheckpoint":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise StatefulContractError("checkpoint payload must be an object")
        raw["state"] = dict(raw.get("state") or {})
        return cls(**raw)


class StatefulCheckpointRegistry:
    _specs: dict[str, StatefulOperatorSpec] = {}

    @classmethod
    def register(cls, spec: StatefulOperatorSpec) -> None:
        if spec.canonical in cls._specs:
            raise StatefulContractError(f"duplicate stateful spec: {spec.canonical}")
        cls._specs[spec.canonical] = spec

    @classmethod
    def get(cls, canonical: str) -> StatefulOperatorSpec | None:
        return cls._specs.get(canonical)

    @classmethod
    def catalog(cls) -> dict[str, dict[str, Any]]:
        return {name: asdict(spec) for name, spec in sorted(cls._specs.items())}

    @staticmethod
    def fingerprint(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def create_checkpoint(
        cls,
        canonical: str,
        *,
        instrument: str,
        as_of: str | datetime,
        state: Mapping[str, Any],
        input_identity: Mapping[str, Any],
    ) -> StateCheckpoint:
        spec = cls.get(canonical)
        if spec is None:
            raise StatefulContractError(f"operator is not checkpoint-managed: {canonical}")
        missing = sorted(set(spec.checkpoint_fields) - set(state))
        if missing:
            raise StatefulContractError(f"checkpoint state missing fields for {canonical}: {missing}")
        timestamp = as_of
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            timestamp = timestamp.astimezone(timezone.utc).isoformat()
        else:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = parsed.astimezone(timezone.utc).isoformat()
        checkpoint = StateCheckpoint(
            operator=canonical,
            instrument=str(instrument),
            as_of=str(timestamp),
            state_schema_version=spec.state_schema_version,
            semantic_version=spec.semantic_version,
            input_fingerprint=cls.fingerprint(input_identity),
            state=dict(state),
        )
        cls.validate(checkpoint, input_identity=input_identity)
        return checkpoint

    @classmethod
    def validate(
        cls,
        checkpoint: StateCheckpoint,
        *,
        input_identity: Mapping[str, Any] | None = None,
        expected_instrument: str | None = None,
    ) -> None:
        spec = cls.get(checkpoint.operator)
        if spec is None:
            raise StatefulContractError(f"unknown checkpoint operator: {checkpoint.operator}")
        if checkpoint.state_schema_version != spec.state_schema_version:
            raise StatefulContractError(
                f"checkpoint state schema mismatch for {checkpoint.operator}: "
                f"{checkpoint.state_schema_version} != {spec.state_schema_version}"
            )
        if checkpoint.semantic_version != spec.semantic_version:
            raise StatefulContractError(
                f"checkpoint semantic version mismatch for {checkpoint.operator}: "
                f"{checkpoint.semantic_version} != {spec.semantic_version}"
            )
        missing = sorted(set(spec.checkpoint_fields) - set(checkpoint.state))
        if missing:
            raise StatefulContractError(f"checkpoint missing fields: {missing}")
        if expected_instrument is not None and checkpoint.instrument != expected_instrument:
            raise StatefulContractError("checkpoint instrument mismatch")
        if input_identity is not None:
            expected = cls.fingerprint(input_identity)
            if checkpoint.input_fingerprint != expected:
                raise StatefulContractError("checkpoint input fingerprint mismatch")
        try:
            parsed = datetime.fromisoformat(checkpoint.as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            raise StatefulContractError("checkpoint as_of is invalid") from exc
        if parsed.tzinfo is None:
            raise StatefulContractError("checkpoint as_of must be timezone-aware")

    @classmethod
    def require_for_segment(
        cls,
        canonical: str,
        *,
        starts_at_dataset_origin: bool,
        checkpoint: StateCheckpoint | None,
        input_identity: Mapping[str, Any] | None = None,
        expected_instrument: str | None = None,
    ) -> None:
        spec = cls.get(canonical)
        if spec is not None and not spec.segmented_execution_supported and not starts_at_dataset_origin:
            raise StatefulContractError(
                f"segmented execution of {canonical} is unsupported without a native checkpoint"
            )
        if spec is None or not spec.checkpoint_required_for_segmented:
            return
        if starts_at_dataset_origin:
            return
        if checkpoint is None:
            raise StatefulContractError(
                f"segmented execution of {canonical} requires a validated checkpoint"
            )
        if checkpoint.operator != canonical:
            raise StatefulContractError(
                f"checkpoint operator mismatch: {checkpoint.operator} != {canonical}"
            )
        cls.validate(
            checkpoint,
            input_identity=input_identity,
            expected_instrument=expected_instrument,
        )


for _spec in (
    StatefulOperatorSpec(
        canonical="trade_when",
        state_schema_version="trade_when_state.v1",
        semantic_version="1.0",
        minimum_history=1,
        checkpoint_fields=("last_value", "last_timestamp"),
        missing_policy="carry_state_emit_null",
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_std", state_schema_version="unsupported.v1",
        semantic_version="2.0", minimum_history=2, checkpoint_fields=(),
        missing_policy="recursive_state", checkpoint_required_for_segmented=False,
        segmented_execution_supported=False,
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_var", state_schema_version="unsupported.v1",
        semantic_version="2.0", minimum_history=2, checkpoint_fields=(),
        missing_policy="recursive_state", checkpoint_required_for_segmented=False,
        segmented_execution_supported=False,
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_cov", state_schema_version="unsupported.v1",
        semantic_version="2.0", minimum_history=2, checkpoint_fields=(),
        missing_policy="recursive_state", checkpoint_required_for_segmented=False,
        segmented_execution_supported=False,
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_corr", state_schema_version="unsupported.v1",
        semantic_version="2.0", minimum_history=2, checkpoint_fields=(),
        missing_policy="recursive_state", checkpoint_required_for_segmented=False,
        segmented_execution_supported=False,
    ),
    StatefulOperatorSpec(
        canonical="ts_ema",
        state_schema_version="ema_state.v1",
        semantic_version="1.0",
        minimum_history=1,
        checkpoint_fields=("last_ema", "last_timestamp"),
        missing_policy="carry_state_emit_null",
    ),
    StatefulOperatorSpec(
        canonical="RSI_WILDER",
        state_schema_version="rsi_wilder_state.v1",
        semantic_version="2.0",
        minimum_history=14,
        checkpoint_fields=("avg_gain", "avg_loss", "last_close", "last_timestamp"),
        missing_policy="carry_state_emit_null",
    ),
    StatefulOperatorSpec(
        canonical="ATR_WILDER",
        state_schema_version="atr_wilder_state.v1",
        semantic_version="2.0",
        minimum_history=14,
        checkpoint_fields=("atr", "last_close", "last_timestamp"),
        missing_policy="carry_state_emit_null",
    ),
    StatefulOperatorSpec(
        canonical="ADX",
        state_schema_version="adx_state.v1",
        semantic_version="2.0",
        minimum_history=28,
        checkpoint_fields=("atr", "plus_dm", "minus_dm", "adx", "last_high", "last_low", "last_close", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        dependencies=("ATR_WILDER",),
    ),
    StatefulOperatorSpec(
        canonical="MACD_line",
        state_schema_version="macd_line_state.v1",
        semantic_version="2.0",
        minimum_history=1,
        checkpoint_fields=("fast_ema", "slow_ema", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        dependencies=("ts_ema",),
    ),
    StatefulOperatorSpec(
        canonical="MACD_signal",
        state_schema_version="macd_signal_state.v1",
        semantic_version="2.0",
        minimum_history=1,
        checkpoint_fields=("fast_ema", "slow_ema", "signal_ema", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        dependencies=("ts_ema", "MACD_line"),
    ),
    StatefulOperatorSpec(
        canonical="MACD_hist",
        state_schema_version="macd_hist_state.v1",
        semantic_version="2.0",
        minimum_history=1,
        checkpoint_fields=("fast_ema", "slow_ema", "signal_ema", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        dependencies=("ts_ema", "MACD_signal"),
    ),
):
    StatefulCheckpointRegistry.register(_spec)

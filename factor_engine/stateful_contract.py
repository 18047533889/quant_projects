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


class CheckpointSerializationError(StatefulContractError):
    """Raised when a checkpoint's state cannot be serialized without silently
    corrupting it (a non-finite float in a field that requires finite values).
    Callers must fail closed (fall back to full-history replay), never persist a
    lossy null in place of an ``Inf``/``NaN`` value."""


@dataclass(frozen=True)
class CheckpointFieldSpec:
    """Per-field type / value-domain contract for a checkpoint's state.

    ``dtype`` is one of ``float|int|str|bool|object``; ``object`` means no type
    constraint beyond the surrounding checks.  ``finite=True`` fails closed when
    a numeric value is ``NaN``/``Inf`` (non-finite), ``finite=False`` permits a
    ``NaN`` "missing" marker that the kernels treat identically to ``null``.
    ``min``/``max`` bound numeric values (inclusive).  ``nested_schema`` lists
    the required keys of an object-valued field (e.g. an ``EwmState`` dict); the
    field's ``finite`` then applies to the nested numeric values.
    ``nested_fields`` (NEW-P1-32) is a RECURSIVE typed schema for an object-valued
    field: each ``CheckpointFieldSpec`` declares the sub-field's own
    dtype / finite / min / max, so validation on read checks sub-field TYPES and
    RANGES — not just required key names.  When present it supersedes the
    key-name-only ``nested_schema`` check (``nested_schema`` is kept for legacy
    readers / diagnostics).
    """

    name: str
    dtype: str = "float"          # float|int|str|bool|object
    nullable: bool = True
    finite: bool = True           # 非有限时 fail closed
    min: float | None = None
    max: float | None = None
    nested_schema: tuple[str, ...] = ()
    nested_fields: tuple["CheckpointFieldSpec", ...] = ()


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
    checkpoint_field_specs: tuple[CheckpointFieldSpec, ...] = ()


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
        # Audit #381: never silently null a non-finite float.  A non-finite value
        # in a field whose spec requires ``finite=True`` (or in any un-spec'd
        # state field) fails closed with an explicit error — the old behaviour
        # mapped ``Inf``/``NaN`` to ``None``, which corrupts a real ``Inf`` state.
        # Fields explicitly declared ``finite=False`` may legitimately hold a
        # ``NaN`` missing-marker that the kernels treat identically to null; only
        # those are persisted as null.
        spec = StatefulCheckpointRegistry.get(self.operator)
        field_specs = (
            {fs.name: fs for fs in spec.checkpoint_field_specs}
            if spec is not None else {}
        )

        def standard_json(value: Any, *, state_field: str | None = None) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                fs = field_specs.get(state_field) if state_field is not None else None
                if fs is not None and not fs.finite:
                    return None  # lossless NaN "missing" marker
                raise CheckpointSerializationError(
                    "checkpoint state contains non-finite value; refusing to silently null it"
                )
            if isinstance(value, Mapping):
                return {str(k): standard_json(v, state_field=state_field) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [standard_json(v, state_field=state_field) for v in value]
            return value

        payload = asdict(self)
        state = payload.get("state")
        if isinstance(state, Mapping):
            payload = dict(payload)
            payload["state"] = {
                str(k): standard_json(v, state_field=str(k)) for k, v in state.items()
            }
            serialized = payload
        else:
            serialized = standard_json(payload)
        return json.dumps(
            serialized, ensure_ascii=False, sort_keys=True,
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
        if spec.checkpoint_field_specs:
            cls._validate_field_specs(spec.checkpoint_field_specs, checkpoint.state)
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

    @staticmethod
    def _validate_field_specs(
        field_specs: tuple[CheckpointFieldSpec, ...],
        state: Mapping[str, Any],
    ) -> None:
        """Audit #380: enforce the per-field dtype / finite / range contract."""
        for fs in field_specs:
            if fs.name not in state:
                raise StatefulContractError(f"checkpoint state missing field for spec: {fs.name}")
            value = state[fs.name]
            # NEW-P1-32: recursive typed schema — validate every sub-field's own
            # dtype / finite / range, not just the required key names.
            if fs.nested_fields:
                if isinstance(value, Mapping):
                    for sub in fs.nested_fields:
                        if sub.name not in value:
                            raise StatefulContractError(
                                f"checkpoint field {fs.name} missing nested key: {sub.name}"
                            )
                        # Recurse so a sub-field's own nested_schema / nested_fields
                        # (and its dtype/finite/min/max) are enforced.
                        StatefulCheckpointRegistry._validate_field_specs(
                            (sub,), {sub.name: value[sub.name]}
                        )
                    continue
                # A non-object value in an object-schema field falls through to the
                # scalar dtype check below (tolerates legacy scalar states while
                # still validating them as numeric when the dtype demands it).
            if fs.nested_schema:
                if isinstance(value, Mapping):
                    missing_nested = sorted(set(fs.nested_schema) - set(value))
                    if missing_nested:
                        raise StatefulContractError(
                            f"checkpoint field {fs.name} missing nested keys: {missing_nested}"
                        )
                    if fs.finite:
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, float) and not math.isfinite(sub_value):
                                raise StatefulContractError(
                                    f"checkpoint field {fs.name}.{sub_key} is non-finite"
                                )
                    continue
                # A non-object value in an object-schema field falls through to the
                # scalar dtype check below (tolerates legacy scalar states while
                # still validating them as numeric when the dtype demands it).

            if value is None:
                if not fs.nullable:
                    raise StatefulContractError(f"checkpoint field {fs.name} must not be null")
                continue
            if fs.dtype == "float":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise StatefulContractError(f"checkpoint field {fs.name} must be numeric")
                numeric = float(value)
                if fs.finite and not math.isfinite(numeric):
                    raise StatefulContractError(f"checkpoint field {fs.name} must be finite")
                if fs.min is not None and numeric < fs.min:
                    raise StatefulContractError(
                        f"checkpoint field {fs.name} below min {fs.min}"
                    )
                if fs.max is not None and numeric > fs.max:
                    raise StatefulContractError(
                        f"checkpoint field {fs.name} above max {fs.max}"
                    )
            elif fs.dtype == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise StatefulContractError(f"checkpoint field {fs.name} must be an integer")
                if fs.min is not None and value < fs.min:
                    raise StatefulContractError(
                        f"checkpoint field {fs.name} below min {fs.min}"
                    )
                if fs.max is not None and value > fs.max:
                    raise StatefulContractError(
                        f"checkpoint field {fs.name} above max {fs.max}"
                    )
            elif fs.dtype == "str":
                if not isinstance(value, str):
                    raise StatefulContractError(f"checkpoint field {fs.name} must be a string")
            elif fs.dtype == "bool":
                if not isinstance(value, bool):
                    raise StatefulContractError(f"checkpoint field {fs.name} must be a boolean")
            # dtype == "object": presence is the only contract (no type gate).

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
        if spec is None and not starts_at_dataset_origin:
            # A recursive operator can honestly declare required_full_history
            # without pretending that a serializer/restore kernel exists in
            # this registry. Such a declaration must still fail closed at a
            # segment boundary: absence from the registry is not permission to
            # restart recursive state from the middle.
            from factor_engine.runtime.execution_contract import execution_contract

            contract = execution_contract(canonical)
            if contract.is_stateful or contract.requires_full_history:
                raise StatefulContractError(
                    f"segmented execution of {canonical} is unsupported without "
                    "a native checkpoint"
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


# Audit #380: per-field type / value-domain specs for every registered operator.
# EwmState-backed fields are nested object dicts ``{weighted_avg, old_wt,
# valid_count}``; ``*_last_*``/``mean_*`` fields may hold a ``NaN`` missing
# marker (declared ``finite=False`` — persisted as null, semantically identical
# to None on resume), every other numeric field fails closed on non-finite.
def _nested_ewm(name: str) -> CheckpointFieldSpec:
    # The production state is an ``EwmState`` dict; a scalar float is tolerated
    # for legacy/foreign checkpoints and validated as numeric.
    # NEW-P1-32: declare the recursive typed schema so read-time validation checks
    # the sub-fields' dtype / finite / range (weighted_avg/old_wt may hold a NaN
    # "missing" marker; valid_count must be a non-negative integer), not just the
    # required key names.
    return CheckpointFieldSpec(
        name=name,
        dtype="float",
        nullable=False,
        finite=True,
        nested_schema=("weighted_avg", "old_wt", "valid_count"),
        nested_fields=(
            CheckpointFieldSpec(
                name="weighted_avg", dtype="float", nullable=True, finite=False
            ),
            CheckpointFieldSpec(
                name="old_wt", dtype="float", nullable=True, finite=False
            ),
            CheckpointFieldSpec(
                name="valid_count", dtype="int", nullable=True, min=0
            ),
        ),
    )


def _ts(name: str) -> CheckpointFieldSpec:
    return CheckpointFieldSpec(name=name, dtype="str", nullable=False)


def _f(name: str, *, finite: bool = True, nullable: bool = True,
       min: float | None = None, max: float | None = None) -> CheckpointFieldSpec:
    return CheckpointFieldSpec(
        name=name, dtype="float", nullable=nullable, finite=finite, min=min, max=max
    )


def _i(name: str, *, min: int | None = None, max: int | None = None) -> CheckpointFieldSpec:
    return CheckpointFieldSpec(name=name, dtype="int", nullable=True, min=min, max=max)


for _spec in (
    StatefulOperatorSpec(
        canonical="trade_when",
        state_schema_version="trade_when_state.v1",
        semantic_version="1.0",
        minimum_history=1,
        checkpoint_fields=("last_value", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        # ``stateful_runtime.execute_stateful_segment`` has no branch for
        # ``trade_when`` yet, so checkpoint-restore must fail closed.  It remains
        # a full-history-replay operator until a real restore implementation is
        # added alongside ``stateful_runtime``.
        segmented_execution_supported=False,
        checkpoint_field_specs=(
            CheckpointFieldSpec(name="last_value", dtype="object", nullable=True, finite=False),
            _ts("last_timestamp"),
        ),
    ),
    # Recursive technical operators documented as stateful (checkpoint schemas in
    # production_hardening.STATEFUL_CHECKPOINTS) but with no segmented restore
    # implementation; checkpoint-restore fails closed, they stay full-replay.
    StatefulOperatorSpec(
        canonical="KAMA",
        state_schema_version="kama_state.v1",
        semantic_version="1.0",
        minimum_history=2,
        checkpoint_fields=("last_value", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        segmented_execution_supported=False,
        checkpoint_field_specs=(
            _f("last_value"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="Supertrend",
        state_schema_version="supertrend_state.v1",
        semantic_version="1.0",
        minimum_history=10,
        checkpoint_fields=("final_upper", "final_lower", "direction", "last_close", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        segmented_execution_supported=False,
        dependencies=("ATR_WILDER",),
        checkpoint_field_specs=(
            _f("final_upper"),
            _f("final_lower"),
            _i("direction", min=-1, max=1),
            _f("last_close", finite=False),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="SupertrendDirection",
        state_schema_version="supertrend_direction_state.v1",
        semantic_version="1.0",
        minimum_history=10,
        checkpoint_fields=("final_upper", "final_lower", "direction", "last_close", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        segmented_execution_supported=False,
        dependencies=("Supertrend",),
        checkpoint_field_specs=(
            _f("final_upper"),
            _f("final_lower"),
            _i("direction", min=-1, max=1),
            _f("last_close", finite=False),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="PSAR",
        state_schema_version="psar_state.v1",
        semantic_version="1.0",
        minimum_history=2,
        checkpoint_fields=("sar", "direction", "extreme_point", "acceleration_factor", "prev_high", "prev_low", "last_timestamp"),
        missing_policy="carry_state_emit_null",
        segmented_execution_supported=False,
        checkpoint_field_specs=(
            _f("sar"),
            _i("direction", min=-1, max=1),
            _f("extreme_point"),
            _f("acceleration_factor", min=0.0),
            _f("prev_high"),
            _f("prev_low"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_std", state_schema_version="ewm_moment_state.v1",
        semantic_version="3.0", minimum_history=2,
        checkpoint_fields=("effective_weight", "weight_sum", "squared_weight_sum", "observation_count", "mean_x", "mean_y", "second_moment_x", "second_moment_y", "cross_moment", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _f("effective_weight"),
            _f("weight_sum"),
            _f("squared_weight_sum"),
            _i("observation_count", min=0),
            _f("mean_x", finite=False),
            _f("mean_y", finite=False),
            _f("second_moment_x"),
            _f("second_moment_y"),
            _f("cross_moment"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_var", state_schema_version="ewm_moment_state.v1",
        semantic_version="3.0", minimum_history=2,
        checkpoint_fields=("effective_weight", "weight_sum", "squared_weight_sum", "observation_count", "mean_x", "mean_y", "second_moment_x", "second_moment_y", "cross_moment", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _f("effective_weight"),
            _f("weight_sum"),
            _f("squared_weight_sum"),
            _i("observation_count", min=0),
            _f("mean_x", finite=False),
            _f("mean_y", finite=False),
            _f("second_moment_x"),
            _f("second_moment_y"),
            _f("cross_moment"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_cov", state_schema_version="ewm_moment_state.v1",
        semantic_version="3.0", minimum_history=2,
        checkpoint_fields=("effective_weight", "weight_sum", "squared_weight_sum", "observation_count", "mean_x", "mean_y", "second_moment_x", "second_moment_y", "cross_moment", "last_timestamp"),
        missing_policy="pairwise_pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _f("effective_weight"),
            _f("weight_sum"),
            _f("squared_weight_sum"),
            _i("observation_count", min=0),
            _f("mean_x", finite=False),
            _f("mean_y", finite=False),
            _f("second_moment_x"),
            _f("second_moment_y"),
            _f("cross_moment"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ts_ewm_corr", state_schema_version="ewm_moment_state.v1",
        semantic_version="3.0", minimum_history=2,
        checkpoint_fields=("effective_weight", "weight_sum", "squared_weight_sum", "observation_count", "mean_x", "mean_y", "second_moment_x", "second_moment_y", "cross_moment", "last_timestamp"),
        missing_policy="pairwise_pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _f("effective_weight"),
            _f("weight_sum"),
            _f("squared_weight_sum"),
            _i("observation_count", min=0),
            _f("mean_x", finite=False),
            _f("mean_y", finite=False),
            _f("second_moment_x"),
            _f("second_moment_y"),
            _f("cross_moment"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ts_ema",
        state_schema_version="ema_state.v2",
        semantic_version="2.0",
        minimum_history=1,
        # R5-09: checkpoint carries the full pandas-EWM state ``(weighted_avg,
        # old_wt, valid_count)`` (schema .v1 stored only ``last_ema``, so a NaN
        # gap did not decay the weight and full/incremental diverged).
        checkpoint_fields=("ema", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _nested_ewm("ema"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="RSI_WILDER",
        state_schema_version="rsi_wilder_state.v2",
        semantic_version="3.0",
        minimum_history=14,
        # R5-10: seeds are the *recursive* EWM gains/losses (``gain/loss`` EwmState
        # dicts), not SMA averages of the first window.
        checkpoint_fields=("gain", "loss", "last_close", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _nested_ewm("gain"),
            _nested_ewm("loss"),
            _f("last_close", finite=False),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ATR_WILDER",
        state_schema_version="atr_wilder_state.v2",
        semantic_version="3.0",
        minimum_history=14,
        # R5-11: first true-range is NaN (pandas ``np.maximum`` propagation), and
        # the smoothing is the pandas EWM (``tr`` EwmState dict), not a seeded mean.
        checkpoint_fields=("tr", "last_close", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        checkpoint_field_specs=(
            _nested_ewm("tr"),
            _f("last_close", finite=False),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="ADX",
        state_schema_version="adx_state.v2",
        semantic_version="3.0",
        minimum_history=28,
        # R5-10: every stage (TR/+DM/-DM/DX) is a ``min_periods=1`` pandas EWM —
        # there is no SMA seeding anywhere.
        checkpoint_fields=("tr", "plus_dm", "minus_dm", "dx", "last_high", "last_low", "last_close", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        dependencies=("ATR_WILDER",),
        checkpoint_field_specs=(
            _nested_ewm("tr"),
            _nested_ewm("plus_dm"),
            _nested_ewm("minus_dm"),
            _nested_ewm("dx"),
            _f("last_high", finite=False),
            _f("last_low", finite=False),
            _f("last_close", finite=False),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="MACD_line",
        state_schema_version="macd_line_state.v2",
        semantic_version="3.0",
        minimum_history=1,
        checkpoint_fields=("fast_ema", "slow_ema", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        dependencies=("ts_ema",),
        checkpoint_field_specs=(
            _nested_ewm("fast_ema"),
            _nested_ewm("slow_ema"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="MACD_signal",
        state_schema_version="macd_signal_state.v2",
        semantic_version="3.0",
        minimum_history=1,
        checkpoint_fields=("fast_ema", "slow_ema", "signal_ema", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        dependencies=("ts_ema", "MACD_line"),
        checkpoint_field_specs=(
            _nested_ewm("fast_ema"),
            _nested_ewm("slow_ema"),
            _nested_ewm("signal_ema"),
            _ts("last_timestamp"),
        ),
    ),
    StatefulOperatorSpec(
        canonical="MACD_hist",
        state_schema_version="macd_hist_state.v2",
        semantic_version="3.0",
        minimum_history=1,
        checkpoint_fields=("fast_ema", "slow_ema", "signal_ema", "last_timestamp"),
        missing_policy="pandas_adjust_false_ignore_na_false",
        dependencies=("ts_ema", "MACD_signal"),
        checkpoint_field_specs=(
            _nested_ewm("fast_ema"),
            _nested_ewm("slow_ema"),
            _nested_ewm("signal_ema"),
            _ts("last_timestamp"),
        ),
    ),
):
    StatefulCheckpointRegistry.register(_spec)


# ---------------------------------------------------------------------------
# R34 P0-029: stateful behavior detection（不依赖手工 canonical 列表）
# ---------------------------------------------------------------------------

def detect_stateful_behavior(
    fn,
    x,
    *,
    split: int | None = None,
    equal_nan: bool = True,
) -> bool:
    """行为检测：``full(x)`` vs ``run(x[:s]) + run(x[s:])``（无 restore）。

    无 restore 时两者不同 → stateful = True。这是对"新算子漏登记 stateful"
    的自动化兜底——不再只靠手工 canonical 集合分类。

    Args:
        fn: 一元序列函数 ``fn(series) -> series``（生产算子的单序列执行）。
        x: 输入序列（支持切片即可，numpy array 或 pandas Series）。
        split: 分段点（默认 ``len(x)//2``）。
        equal_nan: NaN 是否视为相等（``np.allclose`` 参数）。

    Returns:
        True 若 full 与 chunked 结果不同（stateful）；False 若一致（bounded）。
    """
    import numpy as np

    n = len(x)
    if split is None:
        split = max(1, n // 2)
    split = max(1, min(n - 1, int(split)))
    full = fn(x)
    head = fn(x[:split])
    try:
        tail = fn(x[split:])
    except Exception:
        # chunk 触发非法窗口等 → 保守判 stateful（fail-closed 分类）
        return True
    # chunked 拼接（无 restore 语义）：head 尾值 续 tail。
    try:
        combined = np.concatenate([head, tail]) if hasattr(head, "shape") else list(head) + list(tail)
    except Exception:
        return True
    f = np.asarray(full, dtype=float)
    c = np.asarray(combined, dtype=float)
    if f.shape != c.shape:
        return True
    # 只比较尾部 1/4：chunk 边界处 bounded 算子因缺历史产生的差异是 artifact，
    # 但 tail 区段已充分 warmup；真 stateful 会因丢状态而在尾部仍不同。
    trail = max(1, n // 4)
    f_t = f[-trail:]
    c_t = c[-trail:]
    mask = np.isfinite(f_t) & np.isfinite(c_t)
    if mask.any():
        if not np.allclose(f_t[mask], c_t[mask], rtol=1e-8, atol=1e-10, equal_nan=equal_nan):
            return True
    # 非有限区也要对齐（NaN/Inf 位置不同 = 语义不同）
    if not np.array_equal(np.isnan(f_t), np.isnan(c_t)):
        return True
    if not np.array_equal(np.isinf(f_t), np.isinf(c_t)):
        return True
    return False

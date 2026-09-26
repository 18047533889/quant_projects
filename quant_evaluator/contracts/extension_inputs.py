"""Typed sidecar inputs for the QE extension metric families (plan §5.1, §8.2).

All dataclasses are ``frozen``; numpy payloads adopt the existing
``_freeze_array`` immutable-ownership contract (copied and marked read-only, so
caller aliases cannot mutate the artifact afterwards).  Metadata uses
:class:`FrozenMapping`.  Construction validates fully; runtime upload never
re-scans the payloads.

Content identity (plan §5.4): ``content_ref`` is computed *from the actual
content* via ``stable_content_hex`` — a caller can never supply a trusted ref
string.  Changing any payload byte, axis coordinate or ref field changes the
digest.

Serialization (plan §5.1 "序列化约定"): ``to_dict()`` stores only durable
refs + schema version + content hash; large arrays are NOT embedded.
``from_dict()`` restores an *unbound* ref record and never auto-loads data;
missing arrays are never treated as empty panels.  Computation must call
``require()`` / ``require_bound()``, which fail closed with
``MissingInputError`` (maps to UNAVAILABLE / SOURCE_ARTIFACT_MISSING, plan
§5.3) — a serialized ref alone can never fabricate qualified evidence
(plan §1.3 item 5).
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import (
    InvalidContractError,
    MissingInputError,
    SchemaVersionError,
)
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.metric_artifacts import FrozenMapping, _freeze_array
from quant_evaluator.contracts.metric_instance import EvaluationScenario

__all__ = [
    "SCHEMA_VERSION",
    "PairedReturnInput",
    "OOSPredictionInput",
    "TrialFamilyInput",
    "ControlPanel",
    "FactorReturnControlInput",
    "TradePanel",
    "TradeMarketInput",
    "ScenarioGridInput",
    "EventPanel",
    "PredictionDistributionInput",
    "RiskImplementationInput",
    "ContributionInput",
    "SpecificationInput",
    "QuantileInferenceInput",
    "ExtensionInputs",
    "UnboundInputRef",
    "UnboundExtensionInputs",
]

#: Frozen schema version for all extension input contracts (plan §8.2).
SCHEMA_VERSION = "1.0.0"

_UNAVAILABLE_REASON = "SOURCE_ARTIFACT_MISSING"

_COMPARISON_MANIFEST_KEYS = frozenset({
    "baseline_portfolio_ref",
    "candidate_portfolio_ref",
    "cost_policy_ref",
    "risk_policy_ref",
    "comparison_kind",
})

_LEDGER_MANIFEST_KEYS = frozenset({
    "loss_formula_ref",
    "window",
    "direction",
    "universe_ref",
    "model_ref",
    "parameters",
})


# ---------------------------------------------------------------------------
# validation helpers (fail-closed; plan §1.1 precision/dtype clauses)
# ---------------------------------------------------------------------------

def _real_ndarray(name: str, value: Any, ndim: int) -> np.ndarray:
    """Require a real numeric ndarray (bool/complex/object rejected), frozen.

    FP32 (and integer) payloads are stored at their original dtype (plan §1.1:
    "输入已有 FP32 时可按原 dtype 存储"); statistical reduction to FP64 is the
    primitives' responsibility, not the contract's.
    """
    if not isinstance(value, np.ndarray):
        raise InvalidContractError(
            f"{name} must be a numpy ndarray, got {type(value).__name__}"
        )
    if value.dtype.kind not in "fiu":
        raise InvalidContractError(
            f"{name} must have a real numeric dtype "
            f"(bool/complex/object are rejected), got {value.dtype}"
        )
    if value.ndim != ndim:
        raise InvalidContractError(
            f"{name} must be rank-{ndim}, got shape {value.shape}"
        )
    return _freeze_array(value, name)


def _int_ndarray(name: str, value: Any, ndim: int) -> np.ndarray:
    array = _real_ndarray(name, value, ndim)
    if array.dtype.kind not in "iu":
        raise InvalidContractError(f"{name} must have an integer dtype, got {array.dtype}")
    return array


def _bool_ndarray(name: str, value: Any, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype.kind != "b":
        got = getattr(value, "dtype", type(value))
        raise InvalidContractError(f"{name} must be a bool ndarray, got {got}")
    if value.ndim != ndim:
        raise InvalidContractError(f"{name} must be rank-{ndim}, got shape {value.shape}")
    return _freeze_array(value, name)


def _axis(name: str, axis: Any, size: int) -> AxisRef:
    """Require an explicit-coordinate AxisRef sized to the payload (plan §6.1.1)."""
    if not isinstance(axis, AxisRef):
        raise InvalidContractError(
            f"{name} must be an AxisRef, got {type(axis).__name__}"
        )
    if axis.values is None:
        raise InvalidContractError(f"{name} requires explicit coordinates (no bare-size axis)")
    if axis.size != size:
        raise InvalidContractError(
            f"{name} declares size {axis.size} but the payload axis has {size}"
        )
    return axis


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidContractError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip() or any(ord(ch) < 32 for ch in value):
        raise InvalidContractError(f"{name} must be a non-empty plain string")
    return value


def _str_tuple(name: str, value: Any, length: int | None = None,
               *, unique: bool = False) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise InvalidContractError(f"{name} must be a tuple of strings")
    items = tuple(_text(name, item) for item in value)
    if length is not None and len(items) != length:
        raise InvalidContractError(f"{name} must have length {length}, got {len(items)}")
    if unique and len(set(items)) != len(items):
        raise InvalidContractError(f"{name} entries must be unique")
    return items


def _mapping(name: str, value: Any) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise InvalidContractError(f"{name} must be a Mapping, got {type(value).__name__}")
    return FrozenMapping(value)


def _optional_mapping(name: str, value: Any) -> FrozenMapping | None:
    return None if value is None else _mapping(name, value)


def _optional_real(name: str, value: Any, ndim: int) -> np.ndarray | None:
    return None if value is None else _real_ndarray(name, value, ndim)


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidContractError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise InvalidContractError(f"{name} must be finite")
    return value


def _axis_hash(name: str, axis: AxisRef) -> dict:
    return {"name": axis.name, "dtype": axis.dtype, "size": axis.size,
            "values": np.asarray(axis.values)}


def _require_keys(name: str, manifest: Mapping, keys: frozenset) -> None:
    missing = sorted(keys - set(manifest))
    if missing:
        raise InvalidContractError(
            f"{name} is missing required entries: {missing} "
            "(a bare ref string cannot fabricate qualified evidence)"
        )


class _ExtensionInput:
    """Shared content-identity and refs-only serialization mixin (plan §5.1/§5.4)."""

    def _hash_fields(self) -> dict:
        raise NotImplementedError

    @property
    def content_ref(self) -> str:
        return stable_content_hex(
            tag=f"{type(self).__name__}.{SCHEMA_VERSION}", fields=self._hash_fields()
        )

    def to_dict(self) -> dict:
        """Serialize as durable ref only: schema version + content hash (no arrays)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "input_type": type(self).__name__,
            "content_ref": self.content_ref,
            "bound": True,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "UnboundInputRef":
        """Restore an UNBOUND ref record; never auto-loads payloads (plan §5.1)."""
        if not isinstance(data, Mapping):
            raise InvalidContractError("extension input payload must be a Mapping")
        allowed = {"schema_version", "input_type", "content_ref", "bound"}
        unknown = set(data) - allowed
        if unknown:
            raise TypeError(f"unknown serialized keys for {cls.__name__}: {sorted(unknown)}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported extension input schema {data.get('schema_version')!r}"
            )
        if data.get("input_type") != cls.__name__:
            raise InvalidContractError(
                f"serialized input_type {data.get('input_type')!r} does not match {cls.__name__}"
            )
        ref = data.get("content_ref")
        if not isinstance(ref, str) or not ref.strip():
            raise InvalidContractError("serialized content_ref must be a non-empty string")
        return UnboundInputRef(
            input_type=cls.__name__, content_ref=ref, schema_version=SCHEMA_VERSION
        )


@dataclass(frozen=True)
class UnboundInputRef:
    """A serialized-but-unbound extension input (refs only, no arrays)."""

    input_type: str
    content_ref: str
    schema_version: str

    @property
    def bound(self) -> bool:
        return False

    def require(self) -> None:
        raise MissingInputError(
            f"UNAVAILABLE/{_UNAVAILABLE_REASON}: {self.input_type} is serialized as a "
            "ref only and is not bound to in-memory arrays; reload the durable "
            "payload before computation (never treat a ref string as evidence)"
        )


# ---------------------------------------------------------------------------
# typed inputs (field lists frozen by plan §8.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedReturnInput(_ExtensionInput):
    """Paired net-NAV returns: candidate (T,F) vs baseline (T,1)|(T,F).

    ``baseline_map[i]`` names the baseline column compared against candidate
    factor i (no positional guessing, plan §5.1).  ``rf_daily`` must be an
    explicit float64 daily series; an annualized rate is never silently
    converted here.  The comparison protocol (risk normalization, constraints,
    cost policy, baseline ids) lives in ``comparison_manifest``.
    """

    candidate: np.ndarray
    baseline: np.ndarray
    baseline_map: Tuple[str, ...]
    rf_daily: np.ndarray
    time_axis: AxisRef
    comparison_manifest: Mapping

    def __post_init__(self) -> None:
        candidate = _real_ndarray("candidate", self.candidate, 2)
        baseline = _real_ndarray("baseline", self.baseline, 2)
        rf_daily = _real_ndarray("rf_daily", self.rf_daily, 1)
        if rf_daily.dtype != np.dtype(np.float64):
            raise InvalidContractError(
                f"rf_daily must be float64, got {rf_daily.dtype}"
            )
        t, f = candidate.shape
        if baseline.shape[0] != t:
            raise InvalidContractError(
                f"baseline has T={baseline.shape[0]} but candidate has T={t}"
            )
        if baseline.shape[1] not in (1, f):
            raise InvalidContractError(
                "baseline must be (T,1) shared or (T,F) per-factor, got "
                f"{baseline.shape} for F={f}"
            )
        baseline_map = _str_tuple("baseline_map", self.baseline_map, length=f)
        _axis("time_axis", self.time_axis, t)
        manifest = _mapping("comparison_manifest", self.comparison_manifest)
        _require_keys("comparison_manifest", manifest, _COMPARISON_MANIFEST_KEYS)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "baseline_map", baseline_map)
        object.__setattr__(self, "rf_daily", rf_daily)
        object.__setattr__(self, "comparison_manifest", manifest)

    def _hash_fields(self) -> dict:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "baseline_map": self.baseline_map,
            "rf_daily": self.rf_daily,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "comparison_manifest": self.comparison_manifest,
        }


@dataclass(frozen=True)
class OOSPredictionInput(_ExtensionInput):
    """Out-of-sample prediction evidence (plan §5.1 / §8.2).

    ``reference_prediction`` is mandatory: a missing reference is never patched
    with a test mean or zeros.  ``fit_manifests[n]`` documents the model,
    training window and fold identity for candidate n; training-fit values can
    never masquerade as OOS through this contract because the fit manifest and
    ``prediction_available_at`` clock are part of the content hash.
    """

    baseline: np.ndarray            # (T,N)
    candidates: np.ndarray          # (T,N,F)
    reference_prediction: np.ndarray  # (T,N)
    time_axis: AxisRef
    asset_axis: AxisRef
    candidate_ids: Tuple[str, ...]
    validity: np.ndarray            # bool (T,N,F)
    fit_manifests: Tuple[Mapping, ...]
    prediction_available_at: np.ndarray  # int64 (T,) or (T,N), UTC ns
    labels_ref: str

    def __post_init__(self) -> None:
        baseline = _real_ndarray("baseline", self.baseline, 2)
        candidates = _real_ndarray("candidates", self.candidates, 3)
        reference = _real_ndarray("reference_prediction", self.reference_prediction, 2)
        t, n = baseline.shape
        if candidates.shape[:2] != (t, n):
            raise InvalidContractError(
                f"candidates shape {candidates.shape} does not match baseline (T,N)=({t},{n})"
            )
        if reference.shape != (t, n):
            raise InvalidContractError(
                f"reference_prediction shape {reference.shape} does not match (T,N)=({t},{n})"
            )
        f = candidates.shape[2]
        candidate_ids = _str_tuple("candidate_ids", self.candidate_ids, length=f, unique=True)
        validity = _bool_ndarray("validity", self.validity, 3)
        if validity.shape != candidates.shape:
            raise InvalidContractError(
                f"validity shape {validity.shape} must equal candidates shape {candidates.shape}"
            )
        manifests = tuple(
            _mapping(f"fit_manifests[{i}]", item) for i, item in enumerate(self.fit_manifests)
        )
        if len(manifests) != f:
            raise InvalidContractError(
                f"fit_manifests must have one entry per candidate (F={f}), got {len(manifests)}"
            )
        available = _int_ndarray("prediction_available_at", self.prediction_available_at,
                                 np.asarray(self.prediction_available_at).ndim)
        if available.shape[0] != t or available.ndim == 2 and available.shape[1] != n:
            raise InvalidContractError(
                f"prediction_available_at shape {available.shape} does not align with T={t}"
            )
        _axis("time_axis", self.time_axis, t)
        _axis("asset_axis", self.asset_axis, n)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "reference_prediction", reference)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "fit_manifests", manifests)
        object.__setattr__(self, "prediction_available_at", available)
        object.__setattr__(self, "labels_ref", _text("labels_ref", self.labels_ref))
    def _hash_fields(self) -> dict:
        return {
            "baseline": self.baseline,
            "candidates": self.candidates,
            "reference_prediction": self.reference_prediction,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "asset_axis": _axis_hash("asset_axis", self.asset_axis),
            "candidate_ids": self.candidate_ids,
            "validity": self.validity,
            "fit_manifests": self.fit_manifests,
            "prediction_available_at": self.prediction_available_at,
            "labels_ref": self.labels_ref,
        }


@dataclass(frozen=True)
class TrialFamilyInput(_ExtensionInput):
    """Complete trial-family evidence (plan §5.1 / §8.2).

    ``differentials`` are d = baseline_loss - candidate_loss on the family's
    common grid; ``returns`` are only needed for PBO/DSR.  Storing the winner
    alone is structurally impossible: C is frozen by ``trial_ids`` and at least
    one full (T,C)/(C,) payload is required.  The ledger manifest must declare
    formula, window, direction, universe, model and parameters (plan §5.1).
    """

    trial_ids: Tuple[str, ...]
    time_axis: AxisRef
    ledger_manifest: Mapping
    loss_kind: str
    cost_ref: str
    portfolio_ref: str
    differentials: np.ndarray | None = None   # (T,C)
    returns: np.ndarray | None = None         # (T,C)
    family_sharpes: np.ndarray | None = None  # (C,)
    sampling_scope: str = ""

    def __post_init__(self) -> None:
        trial_ids = _str_tuple("trial_ids", self.trial_ids, unique=True)
        c = len(trial_ids)
        if c == 0:
            raise InvalidContractError("trial family requires at least one trial id")
        differentials = _optional_real("differentials", self.differentials, 2)
        returns = _optional_real("returns", self.returns, 2)
        family_sharpes = _optional_real("family_sharpes", self.family_sharpes, 1)
        if differentials is None and returns is None and family_sharpes is None:
            raise InvalidContractError(
                "TrialFamilyInput requires at least one numeric payload "
                "(differentials, returns or family_sharpes)"
            )
        t = None
        for name, payload in (("differentials", differentials), ("returns", returns)):
            if payload is not None:
                if payload.shape[1] != c:
                    raise InvalidContractError(
                        f"{name} has C={payload.shape[1]} but trial_ids has {c}"
                    )
                t = payload.shape[0] if t is None else t
                if payload.shape[0] != t:
                    raise InvalidContractError(
                        f"{name} and differentials/returns disagree on T"
                    )
        if family_sharpes is not None and family_sharpes.shape != (c,):
            raise InvalidContractError(
                f"family_sharpes has C={family_sharpes.shape[0]} but trial_ids has {c}"
            )
        _axis("time_axis", self.time_axis, t if t is not None else self.time_axis.size)
        manifest = _mapping("ledger_manifest", self.ledger_manifest)
        _require_keys("ledger_manifest", manifest, _LEDGER_MANIFEST_KEYS)
        object.__setattr__(self, "trial_ids", trial_ids)
        object.__setattr__(self, "differentials", differentials)
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "family_sharpes", family_sharpes)
        object.__setattr__(self, "ledger_manifest", manifest)
        object.__setattr__(self, "loss_kind", _text("loss_kind", self.loss_kind))
        object.__setattr__(self, "cost_ref", _text("cost_ref", self.cost_ref))
        object.__setattr__(self, "portfolio_ref", _text("portfolio_ref", self.portfolio_ref))
        object.__setattr__(self, "sampling_scope", _text("sampling_scope", self.sampling_scope))

    def _hash_fields(self) -> dict:
        return {
            "trial_ids": self.trial_ids,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "ledger_manifest": self.ledger_manifest,
            "loss_kind": self.loss_kind,
            "cost_ref": self.cost_ref,
            "portfolio_ref": self.portfolio_ref,
            "differentials": self.differentials,
            "returns": self.returns,
            "family_sharpes": self.family_sharpes,
            "sampling_scope": self.sampling_scope,
        }


@dataclass(frozen=True)
class ControlPanel(_ExtensionInput):
    """Regression/conditional control panel (T,N,K) aligned to the factor axis."""

    values: np.ndarray            # (T,N,K)
    time_axis: AxisRef
    asset_axis: AxisRef
    control_ids: Tuple[str, ...]
    weights: np.ndarray           # (T,N) strictly positive
    available_at: np.ndarray      # int64 (T,) | (T,N) | (T,N,K)
    source_ref: str

    def __post_init__(self) -> None:
        values = _real_ndarray("values", self.values, 3)
        t, n, k = values.shape
        control_ids = _str_tuple("control_ids", self.control_ids, length=k, unique=True)
        weights = _real_ndarray("weights", self.weights, 2)
        if weights.shape != (t, n):
            raise InvalidContractError(
                f"weights shape {weights.shape} does not match (T,N)=({t},{n})"
            )
        if not np.all(weights > 0):
            raise InvalidContractError("weights must be strictly positive (NaN/zero rejected)")
        available = np.asarray(self.available_at)
        available = _int_ndarray("available_at", available, available.ndim)
        if available.shape[0] != t or not (available.ndim == 1 and available.shape == (t,)
                                           or available.ndim == 2 and available.shape == (t, n)
                                           or available.ndim == 3 and available.shape == (t, n, k)):
            raise InvalidContractError(
                f"available_at shape {available.shape} must be (T,), (T,N) or (T,N,K) with T={t}"
            )
        _axis("time_axis", self.time_axis, t)
        _axis("asset_axis", self.asset_axis, n)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "control_ids", control_ids)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "source_ref", _text("source_ref", self.source_ref))

    def _hash_fields(self) -> dict:
        return {
            "values": self.values,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "asset_axis": _axis_hash("asset_axis", self.asset_axis),
            "control_ids": self.control_ids,
            "weights": self.weights,
            "available_at": self.available_at,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class FactorReturnControlInput(_ExtensionInput):
    """Existing strategy returns (T,K) used as factor-return controls."""

    values: np.ndarray            # (T,K)
    time_axis: AxisRef
    control_ids: Tuple[str, ...]
    return_basis: str
    cost_ref: str
    portfolio_ref: str
    available_at: np.ndarray      # int64 (T,) | (T,K)
    source_ref: str

    def __post_init__(self) -> None:
        values = _real_ndarray("values", self.values, 2)
        t, k = values.shape
        control_ids = _str_tuple("control_ids", self.control_ids, length=k, unique=True)
        available = np.asarray(self.available_at)
        available = _int_ndarray("available_at", available, available.ndim)
        if available.shape[0] != t or not (available.ndim == 1 and available.shape == (t,)
                                           or available.ndim == 2 and available.shape == (t, k)):
            raise InvalidContractError(
                f"available_at shape {available.shape} must be (T,) or (T,K) with T={t}"
            )
        _axis("time_axis", self.time_axis, t)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "control_ids", control_ids)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "return_basis", _text("return_basis", self.return_basis))
        object.__setattr__(self, "cost_ref", _text("cost_ref", self.cost_ref))
        object.__setattr__(self, "portfolio_ref", _text("portfolio_ref", self.portfolio_ref))
        object.__setattr__(self, "source_ref", _text("source_ref", self.source_ref))

    def _hash_fields(self) -> dict:
        return {
            "values": self.values,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "control_ids": self.control_ids,
            "return_basis": self.return_basis,
            "cost_ref": self.cost_ref,
            "portfolio_ref": self.portfolio_ref,
            "available_at": self.available_at,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class TradePanel(_ExtensionInput):
    """Columnar/CSR trade panel; intraday trades that cannot net share a netting key."""

    date_row: np.ndarray            # int64 (M,)
    asset_col: np.ndarray           # int64 (M,)
    candidate_col: np.ndarray       # int64 (M,)
    netting_key: Tuple[str, ...]
    signed_notional: np.ndarray     # (M,)
    aum_pretrade: np.ndarray        # (T,F)
    time_axis: AxisRef
    asset_axis: AxisRef
    candidate_ids: Tuple[str, ...]
    scope: str
    source_ref: str
    gross_return: np.ndarray | None = None        # (T,F)
    carry_return: np.ndarray | None = None        # (T,F)
    positions_notional: np.ndarray | None = None  # (T,N,F)
    order_details: Mapping | None = None

    def __post_init__(self) -> None:
        date_row = _int_ndarray("date_row", self.date_row, 1)
        asset_col = _int_ndarray("asset_col", self.asset_col, 1)
        candidate_col = _int_ndarray("candidate_col", self.candidate_col, 1)
        m = date_row.shape[0]
        if asset_col.shape != (m,) or candidate_col.shape != (m,):
            raise InvalidContractError("date_row/asset_col/candidate_col must share length M")
        signed_notional = _real_ndarray("signed_notional", self.signed_notional, 1)
        if signed_notional.shape != (m,):
            raise InvalidContractError("signed_notional must have length M")
        netting_key = _str_tuple("netting_key", self.netting_key)
        if not netting_key:
            raise InvalidContractError("netting_key must be a non-empty tuple of column names")
        aum = _real_ndarray("aum_pretrade", self.aum_pretrade, 2)
        t, f = aum.shape
        if m and (int(date_row.min()) < 0 or int(date_row.max()) >= t):
            raise InvalidContractError("date_row out of range for the time axis")
        if m and (int(asset_col.min()) < 0 or int(candidate_col.min()) < 0):
            raise InvalidContractError("trade columns must be non-negative")
        candidate_ids = _str_tuple("candidate_ids", self.candidate_ids, length=f, unique=True)
        _axis("time_axis", self.time_axis, t)
        _axis("asset_axis", self.asset_axis, self.asset_axis.size)
        gross = _optional_real("gross_return", self.gross_return, 2)
        carry = _optional_real("carry_return", self.carry_return, 2)
        positions = _optional_real("positions_notional", self.positions_notional, 3)
        if positions is not None and positions.shape[:2] != (t, self.asset_axis.size):
            raise InvalidContractError(
                f"positions_notional leading axes {positions.shape[:2]} do not match (T,N)=({t},{self.asset_axis.size})"
            )
        for name, payload in (("gross_return", gross), ("carry_return", carry)):
            if payload is not None and payload.shape != (t, f):
                raise InvalidContractError(f"{name} shape {payload.shape} must be (T,F)=({t},{f})")
        object.__setattr__(self, "date_row", date_row)
        object.__setattr__(self, "asset_col", asset_col)
        object.__setattr__(self, "candidate_col", candidate_col)
        object.__setattr__(self, "netting_key", netting_key)
        object.__setattr__(self, "signed_notional", signed_notional)
        object.__setattr__(self, "aum_pretrade", aum)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "gross_return", gross)
        object.__setattr__(self, "carry_return", carry)
        object.__setattr__(self, "positions_notional", positions)
        object.__setattr__(self, "order_details", _optional_mapping("order_details", self.order_details))
        object.__setattr__(self, "scope", _text("scope", self.scope))
        object.__setattr__(self, "source_ref", _text("source_ref", self.source_ref))

    def _hash_fields(self) -> dict:
        return {
            "date_row": self.date_row,
            "asset_col": self.asset_col,
            "candidate_col": self.candidate_col,
            "netting_key": self.netting_key,
            "signed_notional": self.signed_notional,
            "aum_pretrade": self.aum_pretrade,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "asset_axis": _axis_hash("asset_axis", self.asset_axis),
            "candidate_ids": self.candidate_ids,
            "scope": self.scope,
            "source_ref": self.source_ref,
            "gross_return": self.gross_return,
            "carry_return": self.carry_return,
            "positions_notional": self.positions_notional,
            "order_details": self.order_details,
        }


@dataclass(frozen=True)
class TradeMarketInput(_ExtensionInput):
    """Market capacity/cost inputs.  Missing market data must omit the whole
    input; NaN placeholders are rejected so silence can never be read as zero
    cost (plan §5.1: "缺失不能填0")."""

    adv_cash: np.ndarray           # (T,N)
    vol_daily: np.ndarray          # (T,N)
    buy_linear_rate: np.ndarray    # (T,N)
    sell_linear_rate: np.ndarray   # (T,N)
    available_at: np.ndarray       # int64 (T,) | (T,N)
    currency: str
    source_ref: str
    calibration_ref: str
    impact_eta: np.ndarray | None = None          # (T,N)
    realized_volume_cash: np.ndarray | None = None  # (T,N)

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("adv_cash", "vol_daily", "buy_linear_rate", "sell_linear_rate"):
            array = _real_ndarray(name, getattr(self, name), 2)
            if not np.isfinite(array).all():
                raise InvalidContractError(
                    f"{name} must be finite; omit the whole input when market data is missing "
                    "(NaN/zero placeholders are forbidden)"
                )
            arrays[name] = array
        t, n = arrays["adv_cash"].shape
        for name, array in arrays.items():
            if array.shape != (t, n):
                raise InvalidContractError(f"{name} shape {array.shape} does not match (T,N)=({t},{n})")
        if np.any(arrays["vol_daily"] < 0):
            raise InvalidContractError("vol_daily must be non-negative")
        available = np.asarray(self.available_at)
        available = _int_ndarray("available_at", available, available.ndim)
        if available.shape[0] != t or not (available.ndim == 1 and available.shape == (t,)
                                           or available.ndim == 2 and available.shape == (t, n)):
            raise InvalidContractError(
                f"available_at shape {available.shape} must be (T,) or (T,N) with T={t}"
            )
        for name in ("impact_eta", "realized_volume_cash"):
            payload = _optional_real(name, getattr(self, name), 2)
            if payload is not None and payload.shape != (t, n):
                raise InvalidContractError(f"{name} shape {payload.shape} must be (T,N)=({t},{n})")
            object.__setattr__(self, name, payload)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "currency", _text("currency", self.currency))
        object.__setattr__(self, "source_ref", _text("source_ref", self.source_ref))
        object.__setattr__(self, "calibration_ref", _text("calibration_ref", self.calibration_ref))

    def _hash_fields(self) -> dict:
        return {
            "adv_cash": self.adv_cash,
            "vol_daily": self.vol_daily,
            "buy_linear_rate": self.buy_linear_rate,
            "sell_linear_rate": self.sell_linear_rate,
            "available_at": self.available_at,
            "currency": self.currency,
            "source_ref": self.source_ref,
            "calibration_ref": self.calibration_ref,
            "impact_eta": self.impact_eta,
            "realized_volume_cash": self.realized_volume_cash,
        }


@dataclass(frozen=True)
class ScenarioGridInput(_ExtensionInput):
    """Full (delay, horizon) scenario grid; every cell binds real labels/trajectories.

    Plan §8.2 requires a "完整冻结配置ref": the bound :class:`EvaluationScenario`
    payloads are runtime objects, so their durable identity rides
    ``scenario_config_ref`` while the content hash pins the coordinates and the
    ref itself.  No cell may be synthesized from strings.
    """

    ordered_coordinates: Tuple[Tuple[str, str], ...]
    scenarios: Mapping
    reference_coordinate: Tuple[str, str]
    sample_policy: str
    scenario_config_ref: str

    def __post_init__(self) -> None:
        coords = self.ordered_coordinates
        if not isinstance(coords, tuple) or not coords:
            raise InvalidContractError("ordered_coordinates must be a non-empty tuple")
        normalized = []
        for coord in coords:
            if (isinstance(coord, tuple) and len(coord) == 2):
                normalized.append((_text("delay_id", coord[0]), _text("horizon_id", coord[1])))
            else:
                raise InvalidContractError(
                    "each coordinate must be a (delay_id, horizon_id) string pair"
                )
        if len(set(normalized)) != len(normalized):
            raise InvalidContractError("ordered_coordinates entries must be unique")
        if not isinstance(self.scenarios, Mapping):
            raise InvalidContractError("scenarios must be a Mapping keyed by coordinate")
        if set(self.scenarios) != set(normalized):
            raise InvalidContractError(
                "scenarios keys must equal the ordered coordinate set exactly "
                "(no partial grid, no string-fabricated cells)"
            )
        for key, scenario in self.scenarios.items():
            if not isinstance(scenario, EvaluationScenario):
                raise InvalidContractError(
                    f"scenarios[{key!r}] must be an EvaluationScenario, got {type(scenario).__name__}"
                )
        if not isinstance(self.reference_coordinate, tuple) or len(self.reference_coordinate) != 2:
            raise InvalidContractError("reference_coordinate must be a (delay_id, horizon_id) pair")
        reference = (_text("reference delay_id", self.reference_coordinate[0]),
                     _text("reference horizon_id", self.reference_coordinate[1]))
        if reference not in normalized:
            raise InvalidContractError("reference_coordinate must be one of ordered_coordinates")
        # EvaluationScenario instances are themselves frozen runtime bindings;
        # deep-freezing them through FrozenMapping would reject the dataclass,
        # so the mapping is shallow-frozen (keys fixed, values already frozen).
        scenarios = MappingProxyType({key: self.scenarios[key] for key in normalized})
        object.__setattr__(self, "ordered_coordinates", tuple(normalized))
        object.__setattr__(self, "reference_coordinate", reference)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "sample_policy", _text("sample_policy", self.sample_policy))
        object.__setattr__(self, "scenario_config_ref", _text("scenario_config_ref", self.scenario_config_ref))

    def _hash_fields(self) -> dict:
        return {
            "ordered_coordinates": self.ordered_coordinates,
            "scenario_keys": [list(key) for key in self.ordered_coordinates],
            "reference_coordinate": self.reference_coordinate,
            "sample_policy": self.sample_policy,
            "scenario_config_ref": self.scenario_config_ref,
        }


@dataclass(frozen=True)
class EventPanel(_ExtensionInput):
    """Columnar event evidence with explicit availability clock and versions."""

    event_id: Tuple[str, ...]
    issuer_id: Tuple[str, ...]
    version_id: Tuple[str, ...]
    source_id: Tuple[str, ...]
    cluster_id: Tuple[str, ...]
    available_at: np.ndarray          # int64 (M,) UTC ns
    observation_sessions: np.ndarray  # int64 (M,)
    value: np.ndarray                 # (M,)
    decision_bindings: Mapping        # sparse cell -> event/version mapping
    calendar_ref: str
    factor_mapping: Mapping
    source_ref: str
    event_returns: np.ndarray | None = None  # (M,)

    def __post_init__(self) -> None:
        columns = {}
        for name in ("event_id", "issuer_id", "version_id", "source_id", "cluster_id"):
            columns[name] = _str_tuple(name, getattr(self, name))
        m = len(columns["event_id"])
        for name, column in columns.items():
            if len(column) != m:
                raise InvalidContractError(f"{name} must have the same length M as event_id")
        pairs = list(zip(columns["event_id"], columns["version_id"]))
        if len(set(pairs)) != len(pairs):
            raise InvalidContractError("(event_id, version_id) pairs must be unique (dedup rule)")
        available_at = _int_ndarray("available_at", self.available_at, 1)
        observation_sessions = _int_ndarray("observation_sessions", self.observation_sessions, 1)
        value = _real_ndarray("value", self.value, 1)
        if not (available_at.shape == observation_sessions.shape == value.shape == (m,)):
            raise InvalidContractError("available_at/observation_sessions/value must share length M")
        event_returns = _optional_real("event_returns", self.event_returns, 1)
        if event_returns is not None and event_returns.shape != (m,):
            raise InvalidContractError(f"event_returns shape {event_returns.shape} must be (M,)=({m},)")
        object.__setattr__(self, "event_id", columns["event_id"])
        object.__setattr__(self, "issuer_id", columns["issuer_id"])
        object.__setattr__(self, "version_id", columns["version_id"])
        object.__setattr__(self, "source_id", columns["source_id"])
        object.__setattr__(self, "cluster_id", columns["cluster_id"])
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "observation_sessions", observation_sessions)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "event_returns", event_returns)
        object.__setattr__(self, "decision_bindings", _mapping("decision_bindings", self.decision_bindings))
        object.__setattr__(self, "factor_mapping", _mapping("factor_mapping", self.factor_mapping))
        object.__setattr__(self, "calendar_ref", _text("calendar_ref", self.calendar_ref))
        object.__setattr__(self, "source_ref", _text("source_ref", self.source_ref))

    def _hash_fields(self) -> dict:
        return {
            "event_id": self.event_id,
            "issuer_id": self.issuer_id,
            "version_id": self.version_id,
            "source_id": self.source_id,
            "cluster_id": self.cluster_id,
            "available_at": self.available_at,
            "observation_sessions": self.observation_sessions,
            "value": self.value,
            "decision_bindings": self.decision_bindings,
            "calendar_ref": self.calendar_ref,
            "factor_mapping": self.factor_mapping,
            "source_ref": self.source_ref,
            "event_returns": self.event_returns,
        }


@dataclass(frozen=True)
class PredictionDistributionInput(_ExtensionInput):
    """Typed distributional predictions; ``output_kind`` is exclusive (plan §5.1).

    ``binary``/``variance`` payloads are (T,N,F) with one id per candidate;
    ``quantiles``/``ensemble`` payloads are (T,N,M) for a single model, so
    ``candidate_ids`` must be empty there and the model identity rides
    ``model_ref``/``ensemble_ref``.  Probabilities stay in [0,1]; quantile
    levels are strictly increasing in (0,1); variances are non-negative.
    """

    output_kind: str
    prediction: np.ndarray          # (T,N,F) | (T,N,M)
    target: np.ndarray              # (T,N)
    validity: np.ndarray            # bool, same shape as prediction
    time_axis: AxisRef
    asset_axis: AxisRef
    candidate_ids: Tuple[str, ...]
    available_at: np.ndarray        # int64 (T,) | (T,N)
    model_ref: str
    quantile_levels: np.ndarray | None = None  # (M,)
    ensemble_ref: str | None = None

    _KINDS = ("binary", "quantiles", "ensemble", "variance")

    def __post_init__(self) -> None:
        if self.output_kind not in self._KINDS:
            raise InvalidContractError(
                f"output_kind must be one of {self._KINDS}, got {self.output_kind!r}"
            )
        prediction = _real_ndarray("prediction", self.prediction, 3)
        target = _real_ndarray("target", self.target, 2)
        if target.shape != prediction.shape[:2]:
            raise InvalidContractError(
                f"target shape {target.shape} must match prediction leading axes {prediction.shape[:2]}"
            )
        validity = _bool_ndarray("validity", self.validity, 3)
        if validity.shape != prediction.shape:
            raise InvalidContractError(
                f"validity shape {validity.shape} must equal prediction shape {prediction.shape}"
            )
        t, n, last = prediction.shape
        candidate_ids = _str_tuple("candidate_ids", self.candidate_ids, unique=True)
        if self.output_kind in ("binary", "variance"):
            if len(candidate_ids) != last:
                raise InvalidContractError(
                    f"candidate_ids must have length F={last} for {self.output_kind}"
                )
        else:
            if candidate_ids:
                raise InvalidContractError(
                    "candidate_ids must be empty for quantiles/ensemble payloads "
                    "(single model; identity rides model_ref/ensemble_ref)"
                )
        if self.output_kind == "binary" and (
            not np.isfinite(prediction).all() or np.any(prediction < 0) or np.any(prediction > 1)
        ):
            raise InvalidContractError("binary probabilities must be finite within [0,1]")
        if self.output_kind == "variance" and (
            not np.isfinite(prediction).all() or np.any(prediction < 0)
        ):
            raise InvalidContractError("variance predictions must be finite and non-negative")
        quantile_levels = _optional_real("quantile_levels", self.quantile_levels, 1)
        if self.output_kind == "quantiles":
            if quantile_levels is None or quantile_levels.shape != (last,):
                raise InvalidContractError(
                    f"quantile_levels must have length M={last} for quantiles output"
                )
            if not np.all(quantile_levels > 0) or not np.all(quantile_levels < 1) \
                    or not np.all(np.diff(quantile_levels) > 0):
                raise InvalidContractError("quantile_levels must be strictly increasing in (0,1)")
        available = np.asarray(self.available_at)
        available = _int_ndarray("available_at", available, available.ndim)
        if available.shape[0] != t or not (available.ndim == 1 and available.shape == (t,)
                                           or available.ndim == 2 and available.shape == (t, n)):
            raise InvalidContractError(
                f"available_at shape {available.shape} must be (T,) or (T,N) with T={t}"
            )
        _axis("time_axis", self.time_axis, t)
        _axis("asset_axis", self.asset_axis, n)
        ensemble_ref = self.ensemble_ref
        if ensemble_ref is not None:
            ensemble_ref = _text("ensemble_ref", ensemble_ref)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "quantile_levels", quantile_levels)
        object.__setattr__(self, "ensemble_ref", ensemble_ref)
        object.__setattr__(self, "model_ref", _text("model_ref", self.model_ref))

    def _hash_fields(self) -> dict:
        return {
            "output_kind": self.output_kind,
            "prediction": self.prediction,
            "target": self.target,
            "validity": self.validity,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "asset_axis": _axis_hash("asset_axis", self.asset_axis),
            "candidate_ids": self.candidate_ids,
            "available_at": self.available_at,
            "model_ref": self.model_ref,
            "quantile_levels": self.quantile_levels,
            "ensemble_ref": self.ensemble_ref,
        }


@dataclass(frozen=True)
class RiskImplementationInput(_ExtensionInput):
    """Alpha/weights plus a full covariance XOR a factor risk model (mutually exclusive)."""

    alpha: np.ndarray              # (T,N)
    weights: np.ndarray            # (T,N)
    time_axis: AxisRef
    asset_axis: AxisRef
    risk_available_at: np.ndarray  # int64 (T,) | (T,N)
    calibration_ref: str
    portfolio_ref: str
    risk_ref: str
    space_ref: str
    candidate_id: str
    covariance: np.ndarray | None = None        # (T,N,N) full model
    loadings: np.ndarray | None = None          # (T,N,K) factor model
    factor_covariance: np.ndarray | None = None  # (T,K,K)
    specific_variance: np.ndarray | None = None  # (T,N)

    def __post_init__(self) -> None:
        alpha = _real_ndarray("alpha", self.alpha, 2)
        weights = _real_ndarray("weights", self.weights, 2)
        t, n = alpha.shape
        if weights.shape != (t, n):
            raise InvalidContractError(
                f"weights shape {weights.shape} does not match (T,N)=({t},{n})"
            )
        available = np.asarray(self.risk_available_at)
        available = _int_ndarray("risk_available_at", available, available.ndim)
        if available.shape[0] != t or not (available.ndim == 1 and available.shape == (t,)
                                           or available.ndim == 2 and available.shape == (t, n)):
            raise InvalidContractError(
                f"risk_available_at shape {available.shape} must be (T,) or (T,N) with T={t}"
            )
        covariance = _optional_real("covariance", self.covariance, 3)
        loadings = _optional_real("loadings", self.loadings, 3)
        factor_cov = _optional_real("factor_covariance", self.factor_covariance, 3)
        specific = _optional_real("specific_variance", self.specific_variance, 2)
        if covariance is not None:
            if loadings is not None or factor_cov is not None or specific is not None:
                raise InvalidContractError(
                    "full (covariance) and factor risk models are mutually exclusive"
                )
            if covariance.shape != (t, n, n):
                raise InvalidContractError(
                    f"covariance shape {covariance.shape} must be (T,N,N)=({t},{n},{n})"
                )
        else:
            if loadings is None or factor_cov is None:
                raise InvalidContractError(
                    "risk model required: covariance, or loadings together with factor_covariance"
                )
            if loadings.shape[:2] != (t, n):
                raise InvalidContractError(
                    f"loadings leading axes {loadings.shape[:2]} must be (T,N)=({t},{n})"
                )
            k = loadings.shape[2]
            if factor_cov.shape != (t, k, k):
                raise InvalidContractError(
                    f"factor_covariance shape {factor_cov.shape} must be (T,K,K) with K={k}"
                )
            if specific is not None and specific.shape != (t, n):
                raise InvalidContractError(
                    f"specific_variance shape {specific.shape} must be (T,N)=({t},{n})"
                )
        _axis("time_axis", self.time_axis, t)
        _axis("asset_axis", self.asset_axis, n)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "risk_available_at", available)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "factor_covariance", factor_cov)
        object.__setattr__(self, "specific_variance", specific)
        for name in ("calibration_ref", "portfolio_ref", "risk_ref", "space_ref"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "candidate_id", _text("candidate_id", self.candidate_id))

    def _hash_fields(self) -> dict:
        return {
            "alpha": self.alpha,
            "weights": self.weights,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "asset_axis": _axis_hash("asset_axis", self.asset_axis),
            "risk_available_at": self.risk_available_at,
            "calibration_ref": self.calibration_ref,
            "portfolio_ref": self.portfolio_ref,
            "risk_ref": self.risk_ref,
            "space_ref": self.space_ref,
            "candidate_id": self.candidate_id,
            "covariance": self.covariance,
            "loadings": self.loadings,
            "factor_covariance": self.factor_covariance,
            "specific_variance": self.specific_variance,
        }


@dataclass(frozen=True)
class ContributionInput(_ExtensionInput):
    """Return-source concentration payloads: dense (T,G) values or a sparse table."""

    values: np.ndarray | None      # (T,G)
    sparse_table: Mapping | None
    time_axis: AxisRef
    group_ids: Tuple[str, ...]
    candidate_ids: Tuple[str, ...]
    total_pnl: float
    reconciliation_residual: float
    unit: str                      # currency | normalized_return
    group_kind: str                # asset | industry | event | day
    source_ref: str
    comparison_manifest: Mapping

    _UNITS = ("currency", "normalized_return")
    _GROUP_KINDS = ("asset", "industry", "event", "day")

    def __post_init__(self) -> None:
        values = _optional_real("values", self.values, 2)
        sparse = _optional_mapping("sparse_table", self.sparse_table)
        if values is None and sparse is None:
            raise InvalidContractError("ContributionInput requires values or a sparse_table")
        group_ids = _str_tuple("group_ids", self.group_ids, unique=True)
        if not group_ids:
            raise InvalidContractError("group_ids must be a non-empty unique tuple")
        if values is not None and values.shape[1] != len(group_ids):
            raise InvalidContractError(
                f"values has G={values.shape[1]} but group_ids has {len(group_ids)}"
            )
        if self.unit not in self._UNITS:
            raise InvalidContractError(f"unit must be one of {self._UNITS}")
        if self.group_kind not in self._GROUP_KINDS:
            raise InvalidContractError(f"group_kind must be one of {self._GROUP_KINDS}")
        _axis("time_axis", self.time_axis, self.time_axis.size if values is None else values.shape[0])
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sparse_table", sparse)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "candidate_ids", _str_tuple("candidate_ids", self.candidate_ids))
        object.__setattr__(self, "total_pnl", _finite_float("total_pnl", self.total_pnl))
        object.__setattr__(self, "reconciliation_residual",
                           _finite_float("reconciliation_residual", self.reconciliation_residual))
        object.__setattr__(self, "source_ref", _text("source_ref", self.source_ref))
        object.__setattr__(self, "comparison_manifest",
                           _mapping("comparison_manifest", self.comparison_manifest))

    def _hash_fields(self) -> dict:
        return {
            "values": self.values,
            "sparse_table": self.sparse_table,
            "time_axis": _axis_hash("time_axis", self.time_axis),
            "group_ids": self.group_ids,
            "candidate_ids": self.candidate_ids,
            "total_pnl": self.total_pnl,
            "reconciliation_residual": self.reconciliation_residual,
            "unit": self.unit,
            "group_kind": self.group_kind,
            "source_ref": self.source_ref,
            "comparison_manifest": self.comparison_manifest,
        }


@dataclass(frozen=True)
class SpecificationInput(_ExtensionInput):
    """Specification/parameter-neighborhood grid (S specs x F candidates)."""

    values: np.ndarray               # (S,F)
    states: np.ndarray               # (S,F) same shape
    spec_ids: Tuple[str, ...]
    candidate_ids: Tuple[str, ...]
    required_spec_mask: np.ndarray   # bool (S,)
    orientation: np.ndarray          # (F,)
    neighborhood_edges: np.ndarray   # int64 (E,2)
    center_spec_ids: Tuple[str, ...]
    thresholds: Mapping
    spec_manifest: Mapping

    def __post_init__(self) -> None:
        values = _real_ndarray("values", self.values, 2)
        states = _real_ndarray("states", self.states, 2)
        s, f = values.shape
        if states.shape != (s, f):
            raise InvalidContractError(f"states shape {states.shape} must equal values shape {(s, f)}")
        spec_ids = _str_tuple("spec_ids", self.spec_ids, length=s, unique=True)
        candidate_ids = _str_tuple("candidate_ids", self.candidate_ids, length=f, unique=True)
        mask = _bool_ndarray("required_spec_mask", self.required_spec_mask, 1)
        if mask.shape != (s,):
            raise InvalidContractError(f"required_spec_mask shape {mask.shape} must be (S,)=({s},)")
        orientation = _real_ndarray("orientation", self.orientation, 1)
        if orientation.shape != (f,):
            raise InvalidContractError(f"orientation shape {orientation.shape} must be (F,)=({f},)")
        edges = _int_ndarray("neighborhood_edges", self.neighborhood_edges, 2)
        if edges.ndim == 2 and edges.shape[1] != 2:
            raise InvalidContractError("neighborhood_edges must be (E,2)")
        if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= s):
            raise InvalidContractError("neighborhood_edges indices out of range for S specs")
        center = _str_tuple("center_spec_ids", self.center_spec_ids, unique=True)
        unknown = set(center) - set(spec_ids)
        if unknown:
            raise InvalidContractError(f"center_spec_ids not in spec_ids: {sorted(unknown)}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "spec_ids", spec_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "required_spec_mask", mask)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "neighborhood_edges", edges)
        object.__setattr__(self, "center_spec_ids", center)
        object.__setattr__(self, "thresholds", _mapping("thresholds", self.thresholds))
        object.__setattr__(self, "spec_manifest", _mapping("spec_manifest", self.spec_manifest))

    def _hash_fields(self) -> dict:
        return {
            "values": self.values,
            "states": self.states,
            "spec_ids": self.spec_ids,
            "candidate_ids": self.candidate_ids,
            "required_spec_mask": self.required_spec_mask,
            "orientation": self.orientation,
            "neighborhood_edges": self.neighborhood_edges,
            "center_spec_ids": self.center_spec_ids,
            "thresholds": self.thresholds,
            "spec_manifest": self.spec_manifest,
        }


@dataclass(frozen=True)
class QuantileInferenceInput(_ExtensionInput):
    """Binding to the existing daily quantile artifact and counts (no second bucketing)."""

    quantile_returns: np.ndarray   # (...,F) existing daily quantile return payload
    counts: np.ndarray             # int, shape == quantile_returns.shape[:-1]
    binding_label_ref: str
    bucket_policy_ref: str
    sample_ref: str

    def __post_init__(self) -> None:
        values = _real_ndarray("quantile_returns", self.quantile_returns, 2) \
            if np.asarray(self.quantile_returns).ndim == 2 \
            else _real_ndarray("quantile_returns", self.quantile_returns, 3)
        counts = _int_ndarray("counts", self.counts, values.ndim - 1)
        if counts.shape != values.shape[:-1]:
            raise InvalidContractError(
                f"counts shape {counts.shape} must equal quantile_returns leading shape {values.shape[:-1]}"
            )
        object.__setattr__(self, "quantile_returns", values)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "binding_label_ref", _text("binding_label_ref", self.binding_label_ref))
        object.__setattr__(self, "bucket_policy_ref", _text("bucket_policy_ref", self.bucket_policy_ref))
        object.__setattr__(self, "sample_ref", _text("sample_ref", self.sample_ref))

    def _hash_fields(self) -> dict:
        return {
            "quantile_returns": self.quantile_returns,
            "counts": self.counts,
            "binding_label_ref": self.binding_label_ref,
            "bucket_policy_ref": self.bucket_policy_ref,
            "sample_ref": self.sample_ref,
        }


# ---------------------------------------------------------------------------
# ExtensionInputs container (plan §5.1: exactly 14 fixed fields)
# ---------------------------------------------------------------------------

_EXTENSION_INPUT_FIELD_TYPES = {
    "paired_returns": PairedReturnInput,
    "oos_predictions": OOSPredictionInput,
    "trial_family": TrialFamilyInput,
    "controls": ControlPanel,
    "factor_return_controls": FactorReturnControlInput,
    "trade_panel": TradePanel,
    "trade_market": TradeMarketInput,
    "scenario_grid": ScenarioGridInput,
    "events": EventPanel,
    "distribution_predictions": PredictionDistributionInput,
    "risk_implementation": RiskImplementationInput,
    "contributions": ContributionInput,
    "specifications": SpecificationInput,
    "quantile_evidence": QuantileInferenceInput,
}

_EXTENSION_INPUT_TYPES_BY_NAME = {
    cls.__name__: cls for cls in _EXTENSION_INPUT_FIELD_TYPES.values()
}


@dataclass(frozen=True)
class ExtensionInputs:
    """The fixed 14-field sidecar container appended to requests (plan §5.1).

    Every field is optional at construction (a request may carry only the
    sidecars it needs), but any metric depending on a sidecar must obtain it
    through :meth:`require`, which fails closed on ``None`` — absence is
    UNAVAILABLE/SOURCE_ARTIFACT_MISSING, never an empty panel.  Unknown fields
    are a :class:`TypeError`.
    """

    paired_returns: PairedReturnInput | None = None
    oos_predictions: OOSPredictionInput | None = None
    trial_family: TrialFamilyInput | None = None
    controls: ControlPanel | None = None
    factor_return_controls: FactorReturnControlInput | None = None
    trade_panel: TradePanel | None = None
    trade_market: TradeMarketInput | None = None
    scenario_grid: ScenarioGridInput | None = None
    events: EventPanel | None = None
    distribution_predictions: PredictionDistributionInput | None = None
    risk_implementation: RiskImplementationInput | None = None
    contributions: ContributionInput | None = None
    specifications: SpecificationInput | None = None
    quantile_evidence: QuantileInferenceInput | None = None

    def __post_init__(self) -> None:
        for name, expected in _EXTENSION_INPUT_FIELD_TYPES.items():
            value = getattr(self, name)
            if value is not None and type(value) is not expected:
                raise InvalidContractError(
                    f"ExtensionInputs.{name} must be {expected.__name__} or None, "
                    f"got {type(value).__name__}"
                )

    def require(self, *names: str) -> Tuple[_ExtensionInput, ...]:
        """Fail-closed accessor: returns the bound inputs or raises (plan §1.3.5)."""
        unknown = [name for name in names if name not in _EXTENSION_INPUT_FIELD_TYPES]
        if unknown:
            raise TypeError(f"unknown ExtensionInputs fields: {unknown}")
        values = tuple(getattr(self, name) for name in names)
        missing = [name for name, value in zip(names, values) if value is None]
        if missing:
            raise MissingInputError(
                f"UNAVAILABLE/{_UNAVAILABLE_REASON}: required extension inputs {missing} "
                "are absent; a ref string cannot fabricate qualified evidence"
            )
        return values

    @property
    def content_ref(self) -> str:
        return stable_content_hex(
            tag=f"ExtensionInputs.{SCHEMA_VERSION}",
            fields={name: (None if getattr(self, name) is None else getattr(self, name).content_ref)
                    for name in _EXTENSION_INPUT_FIELD_TYPES},
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "content_ref": self.content_ref,
            "inputs": {
                name: (None if getattr(self, name) is None else getattr(self, name).to_dict())
                for name in _EXTENSION_INPUT_FIELD_TYPES
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "UnboundExtensionInputs":
        """Restore refs only; the result is unbound and cannot compute (plan §5.1)."""
        if not isinstance(data, Mapping):
            raise InvalidContractError("ExtensionInputs payload must be a Mapping")
        unknown = set(data) - {"schema_version", "content_ref", "inputs"}
        if unknown:
            raise TypeError(f"unknown ExtensionInputs serialized keys: {sorted(unknown)}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported ExtensionInputs schema {data.get('schema_version')!r}"
            )
        if not isinstance(data.get("content_ref"), str) or not data["content_ref"].strip():
            raise InvalidContractError("ExtensionInputs payload requires a content_ref")
        inputs = data.get("inputs")
        if not isinstance(inputs, Mapping):
            raise InvalidContractError("ExtensionInputs payload requires an 'inputs' mapping")
        unknown_inputs = set(inputs) - set(_EXTENSION_INPUT_FIELD_TYPES)
        if unknown_inputs:
            raise TypeError(f"unknown ExtensionInputs fields: {sorted(unknown_inputs)}")
        refs = {}
        for name, payload in inputs.items():
            if payload is None:
                refs[name] = None
                continue
            if not isinstance(payload, Mapping):
                raise InvalidContractError(f"inputs[{name!r}] must be a Mapping or None")
            input_type = payload.get("input_type")
            if input_type not in _EXTENSION_INPUT_TYPES_BY_NAME:
                raise InvalidContractError(f"unknown input_type {input_type!r} for field {name!r}")
            expected = _EXTENSION_INPUT_FIELD_TYPES[name]
            if payload.get("input_type") != expected.__name__:
                raise InvalidContractError(
                    f"inputs[{name!r}] declares {input_type!r} but the field requires "
                    f"{expected.__name__}"
                )
            refs[name] = expected.from_dict(payload)
        # UnboundInputRef records are frozen dataclasses but not canonical
        # values; the ref map is shallow-frozen (keys fixed) instead of deep
        # FrozenMapping.
        refs = MappingProxyType(dict(refs))
        return UnboundExtensionInputs(
            refs=refs,
            schema_version=SCHEMA_VERSION,
            content_ref=data["content_ref"],
        )


@dataclass(frozen=True)
class UnboundExtensionInputs:
    """Refs-only restoration of :class:`ExtensionInputs` (plan §5.1)."""

    refs: Mapping
    schema_version: str
    content_ref: str

    @property
    def bound(self) -> bool:
        return False

    def require_bound(self, *names: str) -> None:
        missing = [name for name in names if self.refs.get(name) is None]
        raise MissingInputError(
            f"UNAVAILABLE/{_UNAVAILABLE_REASON}: serialized extension inputs "
            f"{sorted(names)} are not bound to in-memory arrays "
            f"(missing refs: {missing or 'requested set'}); reload the durable "
            "payload before computation"
        )

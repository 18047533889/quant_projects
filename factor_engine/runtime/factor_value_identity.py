# -*- coding: utf-8 -*-
"""Wave1-F factor-lake value identity — FactorValueIdentity + FeatureSetIdentity.

Mission: the model layer must consume identities rather than bare factor names.
A block writer emits these identities (per factor, and per feature set) into the
block manifest; the reader verifies them on load; a model-layer helper returns
the ``FeatureSetIdentity`` for a given backtest factor set.

Design rules (kept local and low-risk):
  * PURE stdlib + numpy. No new hard dependencies.
  * Deterministic: identity_hash is a sha256 over canonical, length-prefixed
    semantic fields (same canonical discipline as the platform contracts layer,
    without importing it).
  * ``block_content_hash`` is COMPUTED from the block's serialized column values
    (not stored raw), so a block whose manifest identity no longer matches its
    actual content is rejected by the reader (non-vacuous verification).
  * This module is self-contained so it can be wired into
    ``factor_engine.runtime.feature_block`` /
    ``factor_engine.runtime.remote_factor_block_writer`` without touching owned
    streaming code.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FactorValueIdentity",
    "FeatureSetIdentity",
    "UNKNOWN_SNAPSHOT",
    "build_factor_value_identity_from_block",
    "compute_block_content_hash",
    "factor_value_identity_fields",
    "feature_set_identity_for_backtest",
    "verify_manifest_factors_against_content",
    "universe_stable_id",
    "_canonical_value",
]


#: Sentinel for snapshot/universe fields genuinely unspecified at write time.
#: It participates in the identity like a concrete id, so "explicitly
#: unspecified" still renders a distinct identity from "snapshot X".
UNKNOWN_SNAPSHOT = "__unknown__"


# ---------------------------------------------------------------------------
# Canonical field codec (deterministic sha256)
# ---------------------------------------------------------------------------

def _canonical_value(value: Any) -> str:
    """Deterministic canonical string of a scalar / mapping / sequence.

    Mirrors ``quant_platform.app.contracts._contenthash`` WITHOUT importing the
    platform contracts layer (FactorEngine does not depend on it).  Datetimes
    must be tz-aware (naive raise, fail closed); ``-0.0`` normalized to ``0.0``;
    floats must be finite; mapping keys sorted canonically.
    """
    if isinstance(value, enum.Enum):
        return _canonical_value(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "naive datetime not allowed in factor identity; use tz-aware UTC"
            )
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: _canonical_value(kv[0]))
        return "{" + ",".join(
            f"{_canonical_value(k)}={_canonical_value(v)}" for k, v in items
        ) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_canonical_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("NaN/Inf not allowed in factor identity hash")
        if value == 0.0:
            return "0.0"
        return repr(value)
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    raise TypeError(
        f"unsupported type for factor identity hash: {type(value).__name__!r}"
    )


def _canonical_fields(mapping: Mapping[str, Any]) -> dict[str, str]:
    """Map canonical field-name -> canonical value, sorted by name."""
    return {name: _canonical_value(value) for name, value in sorted(mapping.items())}


def _identity_hash(mapping: Mapping[str, Any], *extra: Any) -> str:
    """sha256 over length-prefixed canonical fields (order-independent fields).

    ``extra`` values are length-prefixed after the mapping so ordering nuances
    (e.g. an ordered member tuple) can be folded in explicitly.
    """
    digest = hashlib.sha256()
    for name, encoded in _canonical_fields(mapping).items():
        payload = f"{name}={encoded}".encode("utf-8")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b":")
        digest.update(payload)
    for value in extra:
        encoded = _canonical_value(value).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Identity value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorValueIdentity:
    """Immutable semantic identity of one factor-value snapshot (lake block).

    Carries, per mission::

        factor_definition_hash, factor_implementation_hash,
        treatment_preprocess_hash, market, frequency, partition_time,
        universe_snapshot_id, ordered_instrument_hash, data_snapshot_id,
        calendar_snapshot_id, decision_time_policy, numeric_policy, dtype,
        build_sha, block_content_hash.

    ``identity_hash`` covers all semantic fields EXCEPT ``block_content_hash``:
    the content hash is a *verification* bound on stored bytes and must be able
    to change independently (e.g. silent byte drift) while the semantic identity
    stays the same.  Reader verification compares the recorded content hash to a
    fresh recomputation over the actual block bytes.
    """

    factor_id: str
    factor_definition_hash: str
    factor_implementation_hash: str
    treatment_preprocess_hash: str
    market: str
    frequency: str
    partition_time: str
    universe_snapshot_id: str
    ordered_instrument_hash: str
    data_snapshot_id: str
    calendar_snapshot_id: str
    decision_time_policy: str
    numeric_policy: str
    dtype: str
    build_sha: str
    block_content_hash: str = ""

    def identity_hash(self) -> str:
        """Semantic hash over all fields except block_content_hash."""
        return _identity_hash(
            {
                "factor_id": self.factor_id,
                "factor_definition_hash": self.factor_definition_hash,
                "factor_implementation_hash": self.factor_implementation_hash,
                "treatment_preprocess_hash": self.treatment_preprocess_hash,
                "market": self.market,
                "frequency": self.frequency,
                "partition_time": self.partition_time,
                "universe_snapshot_id": self.universe_snapshot_id,
                "ordered_instrument_hash": self.ordered_instrument_hash,
                "data_snapshot_id": self.data_snapshot_id,
                "calendar_snapshot_id": self.calendar_snapshot_id,
                "decision_time_policy": self.decision_time_policy,
                "numeric_policy": self.numeric_policy,
                "dtype": self.dtype,
                "build_sha": self.build_sha,
            }
        )

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {
            f.name: getattr(self, f.name) for f in fields(self)
        }
        out["identity_hash"] = self.identity_hash()
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FactorValueIdentity":
        def _v(name: str, default: str = "") -> str:
            val = d.get(name, default)
            return default if val is None else str(val)

        return cls(
            factor_id=_v("factor_id", ""),
            factor_definition_hash=_v("factor_definition_hash"),
            factor_implementation_hash=_v("factor_implementation_hash"),
            treatment_preprocess_hash=_v("treatment_preprocess_hash"),
            market=_v("market"),
            frequency=_v("frequency"),
            partition_time=_v("partition_time"),
            universe_snapshot_id=_v("universe_snapshot_id", UNKNOWN_SNAPSHOT),
            ordered_instrument_hash=_v("ordered_instrument_hash"),
            data_snapshot_id=_v("data_snapshot_id", UNKNOWN_SNAPSHOT),
            calendar_snapshot_id=_v("calendar_snapshot_id", UNKNOWN_SNAPSHOT),
            decision_time_policy=_v("decision_time_policy"),
            numeric_policy=_v("numeric_policy", "float32"),
            dtype=_v("dtype", "float32"),
            build_sha=_v("build_sha"),
            block_content_hash=_v("block_content_hash"),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FactorValueIdentity) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.identity_hash())


@dataclass(frozen=True)
class FeatureSetIdentity:
    """Identity of an ordered set of factor values feeding a model/backtest.

    ``feature_set_content_hash`` covers ``feature_set_version_id`` + the ordered
    ``FactorValueIdentity.identity_hash`` list (order-sensitive) + each member's
    ``block_content_hash`` — so a membership, numeric-policy/universe change, or
    block content drift all change the identity (non-vacuous).
    """

    feature_set_version_id: str
    feature_set_content_hash: str
    factors: tuple[FactorValueIdentity, ...] = ()

    @classmethod
    def build(
        cls,
        feature_set_version_id: str,
        factors: Iterable[FactorValueIdentity],
    ) -> "FeatureSetIdentity":
        ordered = tuple(factors)
        factor_levels = tuple(
            (f.identity_hash(), f.block_content_hash) for f in ordered
        )
        content = _identity_hash(
            {"feature_set_version_id": feature_set_version_id},
            factor_levels,
        )
        return cls(
            feature_set_version_id=feature_set_version_id,
            feature_set_content_hash=content,
            factors=ordered,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set_version_id": self.feature_set_version_id,
            "feature_set_content_hash": self.feature_set_content_hash,
            "factors": [f.to_dict() for f in self.factors],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureSetIdentity":
        return cls(
            feature_set_version_id=str(d.get("feature_set_version_id", "")),
            feature_set_content_hash=str(d.get("feature_set_content_hash", "")),
            factors=tuple(
                FactorValueIdentity.from_dict(f) for f in d.get("factors", ())
            ),
        )


# ---------------------------------------------------------------------------
# block_content_hash — written by the block writer, verified by the reader
# ---------------------------------------------------------------------------

def compute_block_content_hash(
    column_values: np.ndarray,
    *,
    factor_ids: Sequence[str] | None = None,
    numeric_policy: str = "",
) -> str:
    """Byte-level content hash over the block's float32 column values.

    The writer records this hash in the manifest identity; the reader recomputes
    it from the stored bytes and rejects the block on mismatch.
    """
    arr = np.ascontiguousarray(column_values, dtype=np.dtype("float32"))
    digest = hashlib.sha256(arr.tobytes(order="C"))
    if factor_ids is not None:
        for fid in factor_ids:
            digest.update(str(fid).encode("utf-8"))
    if numeric_policy:
        digest.update(str(numeric_policy).encode("utf-8"))
    return digest.hexdigest()


def build_factor_value_identity_from_block(
    *,
    factor_id: str,
    block_values: np.ndarray,
    factor_ids: Sequence[str],
    factor_definition_hash: str = "",
    factor_implementation_hash: str = "",
    treatment_preprocess_hash: str = "",
    market: str = "",
    frequency: str = "",
    partition_time: str = "",
    universe_snapshot_id: str = UNKNOWN_SNAPSHOT,
    ordered_instrument_hash: str = "",
    data_snapshot_id: str = UNKNOWN_SNAPSHOT,
    calendar_snapshot_id: str = UNKNOWN_SNAPSHOT,
    decision_time_policy: str = "",
    numeric_policy: str = "float32",
    dtype: str = "float32",
    build_sha: str = "",
    block_content_hash: str | None = None,
) -> FactorValueIdentity:
    """Build a factor identity from a block's column values.

    ``block_content_hash`` is computed over the WHOLE block (all columns) by
    default — the identity is bound to the block, not just one column; pass an
    explicit hash to override the computed value.
    """
    if factor_id not in tuple(factor_ids):
        raise ValueError(f"factor {factor_id!r} not in factor_ids {list(factor_ids)}")
    content = block_content_hash or compute_block_content_hash(
        block_values, factor_ids=list(factor_ids), numeric_policy=numeric_policy
    )
    return FactorValueIdentity(
        factor_id=factor_id,
        factor_definition_hash=factor_definition_hash,
        factor_implementation_hash=factor_implementation_hash,
        treatment_preprocess_hash=treatment_preprocess_hash,
        market=market,
        frequency=frequency,
        partition_time=partition_time,
        universe_snapshot_id=universe_snapshot_id,
        ordered_instrument_hash=ordered_instrument_hash,
        data_snapshot_id=data_snapshot_id,
        calendar_snapshot_id=calendar_snapshot_id,
        decision_time_policy=decision_time_policy,
        numeric_policy=numeric_policy,
        dtype=dtype,
        build_sha=build_sha,
        block_content_hash=content,
    )


def factor_value_identity_fields(
    *,
    factor_id: str,
    factor_definition_hash: str = "",
    factor_implementation_hash: str = "",
    treatment_preprocess_hash: str = "",
    market: str = "",
    frequency: str = "",
    partition_time: str = "",
    universe_snapshot_id: str = UNKNOWN_SNAPSHOT,
    ordered_instrument_hash: str = "",
    data_snapshot_id: str = UNKNOWN_SNAPSHOT,
    calendar_snapshot_id: str = UNKNOWN_SNAPSHOT,
    decision_time_policy: str = "",
    numeric_policy: str = "float32",
    dtype: str = "float32",
    build_sha: str = "",
    block_content_hash: str = "",
) -> dict[str, str]:
    """Keyword builder for the ``FactorValueIdentity`` field dict.

    Kept in this module (used by the writers' ``_build_missing_identities`` to
    synthesize stable factor identities without importing the DTO).
    """
    return {
        "factor_id": str(factor_id),
        "factor_definition_hash": str(factor_definition_hash),
        "factor_implementation_hash": str(factor_implementation_hash),
        "treatment_preprocess_hash": str(treatment_preprocess_hash),
        "market": str(market),
        "frequency": str(frequency),
        "partition_time": str(partition_time),
        "universe_snapshot_id": str(universe_snapshot_id),
        "ordered_instrument_hash": str(ordered_instrument_hash),
        "data_snapshot_id": str(data_snapshot_id),
        "calendar_snapshot_id": str(calendar_snapshot_id),
        "decision_time_policy": str(decision_time_policy),
        "numeric_policy": str(numeric_policy),
        "dtype": str(dtype),
        "build_sha": str(build_sha),
        "block_content_hash": str(block_content_hash),
    }


# ---------------------------------------------------------------------------
# Reader-side verification
# ---------------------------------------------------------------------------

def verify_manifest_factors_against_content(
    manifest: Any,
    *,
    reader: Any | None = None,
    load_block: Any | None = None,
) -> list[str]:
    """Verify every manifest factor identity against the block's real content.

    Returns the list of factor ids whose recorded ``block_content_hash`` does NOT
    match a recomputation over the stored block values.  A non-empty result means
    the caller must reject the block (identity/content mismatch).

    ``load_block``: callable(block_id) -> np.ndarray (defaults to ``reader``).
    """
    mismatches: list[str] = []
    identities = getattr(manifest, "identities", None) or {}
    blocks = getattr(manifest, "blocks", {})
    for factor_id, ident in identities.items():
        block_content = _identity_field(ident, "block_content_hash")
        if not block_content:
            continue
        resolved = manifest.resolve(factor_id)
        if not resolved:
            mismatches.append(factor_id)
            continue
        _partition, block_spec, _column = resolved[0]
        block_arr = None
        if load_block is not None:
            try:
                block_arr = load_block(block_spec.block_id)
            except Exception:
                block_arr = None
        elif reader is not None:
            block_arr = reader.read_block(block_spec.block_id)
        if block_arr is None:
            mismatches.append(factor_id)
            continue
        block_factor_ids = getattr(block_spec, "factor_ids", None) or ()
        policy = _identity_field(ident, "numeric_policy") or ""
        recomputed = compute_block_content_hash(
            block_arr,
            factor_ids=list(block_factor_ids),
            numeric_policy=str(policy),
        )
        if recomputed != block_content:
            mismatches.append(factor_id)
    return mismatches


def _identity_field(ident: Any, name: str) -> Any:
    """Read a field from a FactorValueIdentity OR its dict form."""
    if ident is None:
        return None
    if isinstance(ident, Mapping):
        return ident.get(name)
    return getattr(ident, name, None)


# ---------------------------------------------------------------------------
# Model-layer helper: FeatureSetIdentity for a backtest's factor set
# ---------------------------------------------------------------------------

def universe_stable_id(universe: str) -> str:
    if not universe:
        return UNKNOWN_SNAPSHOT
    return _identity_hash({"universe": str(universe)})


def feature_set_identity_for_backtest(
    backtest_id: str,
    factor_ids: Iterable[str],
    *,
    identities: Mapping[str, Any],
    version: str = "v1",
    universe: str = "",
    frequency: str = "",
    build_sha: str = "",
    numeric_policy: str = "float32",
    dtype: str = "float32",
) -> FeatureSetIdentity:
    """Resolve the ordered ``FeatureSetIdentity`` for a backtest factor set.

    The model layer consumes identities rather than bare names: it passes the
    backtest's ordered ``factor_ids`` plus a ``{factor_id: FactorValueIdentity |
    dict}`` mapping (from the block manifest); the helper constructs the
    feature-set identity whose content hash binds the ordered member identities
    AND their block content hashes.

    A factor that has no manifest identity gets a synthesized identity from the
    supplied context — it NEVER silently degrades to a bare factor name.

    Args:
        backtest_id: stable backtest id (forms the ``feature_set_version_id``).
        factor_ids: the backtest's ordered factor set.
        identities: ``{factor_id: FactorValueIdentity | dict}`` from the manifest.
        version: feature-set version label.
        universe/frequency/build_sha/numeric_policy/dtype: fallback context for
            factors lacking an explicit identity.

    Returns:
        FeatureSetIdentity (immutable).
    """
    ordered: list[FactorValueIdentity] = []
    for fid in factor_ids:
        ident = identities.get(str(fid))
        if isinstance(ident, Mapping):
            ordered.append(FactorValueIdentity.from_dict(ident))
        elif ident is not None:
            ordered.append(ident)
        else:
            # No explicit identity in the manifest: synthesize with explicit
            # context, never a bare factor name.
            ordered.append(
                FactorValueIdentity(
                    factor_id=str(fid),
                    factor_definition_hash=_identity_hash(
                        {"factor_id": str(fid), "build_sha": str(build_sha)}
                    ),
                    factor_implementation_hash=str(build_sha),
                    treatment_preprocess_hash="",
                    market="",
                    frequency=str(frequency),
                    partition_time="",
                    universe_snapshot_id=universe_stable_id(universe),
                    ordered_instrument_hash="",
                    data_snapshot_id=UNKNOWN_SNAPSHOT,
                    calendar_snapshot_id=UNKNOWN_SNAPSHOT,
                    decision_time_policy="",
                    numeric_policy=str(numeric_policy),
                    dtype=str(dtype),
                    build_sha=str(build_sha),
                    block_content_hash=_identity_hash(
                        {"factor_id": str(fid), "version": str(version)}
                    ),
                )
            )
    version_id = f"{backtest_id}::{version}"
    return FeatureSetIdentity.build(version_id, ordered)
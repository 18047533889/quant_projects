"""Conservative eligibility checks for measured-auto request-digest reuse.

This module never computes input identity. It only decides whether an exact
contract object can safely reuse the digest captured at calibration time.
Unknown or excessively nested values fall back to the full request fingerprint.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import FrozenMapping

MAX_IMMUTABILITY_NODES = 1024
MAX_ARRAY_BASE_DEPTH = 16


def _immutable_bytes_array(value: np.ndarray) -> bool:
    """Check read-only ownership without traversing array contents."""
    if type(value) is not np.ndarray or value.flags.writeable:
        return False
    owner = value
    seen = set()
    for _ in range(MAX_ARRAY_BASE_DEPTH):
        if id(owner) in seen:
            return False
        seen.add(id(owner))
        if not isinstance(owner, np.ndarray):
            return type(owner) is bytes
        base = owner.base
        if base is None:
            return False
        owner = base
    return False


def _deeply_immutable(value) -> bool:
    pending = [value]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > MAX_IMMUTABILITY_NODES:
            return False
        current_type = type(current)
        if current is None or current_type in (bool, int, float, str, bytes):
            continue
        remaining = MAX_IMMUTABILITY_NODES - visited - len(pending)
        if current_type is tuple or current_type is frozenset:
            if len(current) > remaining:
                return False
            pending.extend(current)
            continue
        if current_type is FrozenMapping:
            if len(current) > remaining:
                return False
            for key, item in current.items():
                if type(key) is not str:
                    return False
                pending.append(item)
            continue
        if current_type is np.ndarray:
            if current.dtype.hasobject or current.dtype.fields is not None:
                return False
            if not _immutable_bytes_array(current):
                return False
            continue
        if current_type is datetime:
            if current.tzinfo is not None and type(current.tzinfo) is not timezone:
                return False
            continue
        if current_type is date or current_type is timedelta or current_type is Decimal:
            continue
        if isinstance(current, np.generic):
            if current_type is not current.dtype.type:
                return False
            if current.dtype.hasobject or current.dtype.fields is not None:
                return False
            continue
        # In particular, reject Enum: FrozenMapping retains enum instances,
        # and enum.value may itself refer to mutable state.
        return False
    return True


def _safe_axis(axis) -> bool:
    if axis is None:
        return True
    if type(axis) is not AxisRef:
        return False
    # Bypass AxisRef.__getattribute__: for object arrays that accessor copies
    # the whole coordinate buffer just to detach public reads.
    values = object.__getattribute__(axis, "values")
    if values is None:
        return True
    return (not values.dtype.hasobject
            and values.dtype.fields is None
            and _immutable_bytes_array(values))


def supports_request_digest_reuse(batch, label) -> bool:
    """Whether these exact contract objects have cheaply provable deep immutability."""
    if type(batch) is not FactorBatch or type(label) is not LabelBundle:
        return False
    if not (_safe_axis(batch.time_axis) and _safe_axis(batch.asset_axis)
            and _safe_axis(label.asset_axis)):
        return False
    return _deeply_immutable(batch.context_refs)

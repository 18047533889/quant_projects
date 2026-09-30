"""Bounded process-local measured-auto candidates; never stores results/arrays.

A lookup is advisory only. The caller's validator must recompute the full
calibration identity and verify the same key still has a passing cached winner.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import math
import numbers
import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping

MAX_ENTRIES = 32
TTL_SECONDS = 3600.0
_LOCK = threading.RLock()
_RECORDS = OrderedDict()

def _freeze(value):
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if hasattr(value, "__dataclass_fields__"):
        return tuple((name, _freeze(getattr(value, name))) for name in value.__dataclass_fields__)
    return value

@dataclass(frozen=True)
class InputDescriptor:
    factor_ids: tuple[str, ...]
    shape: tuple[int, ...]
    value_hash: str | None
    label_content_hash: str
    dtype: str
    axis_schema: tuple
    metrics: tuple[str, ...]

    @classmethod
    def from_values(cls, *, factor_ids, shape, value_hash, label_content_hash, dtype, axis_schema, metrics):
        return cls(tuple(factor_ids), tuple(shape), value_hash, str(label_content_hash), str(dtype), _freeze(axis_schema), tuple(metrics))

@dataclass(frozen=True)
class MeasuredCandidate:
    cache_ref: object
    cache_key: str
    calibration_policy: tuple
    gpu_policy: tuple
    descriptor: InputDescriptor
    winner: str
    cpu_seconds: float
    cuda_seconds: float
    setup_seconds: float

@dataclass(frozen=True)
class CandidateLookup:
    winner: str | None
    validation_seconds: float | None
    rejection_reason: str | None

def _after_fork_child():
    global _LOCK, _RECORDS
    _LOCK = threading.RLock()
    _RECORDS = OrderedDict()

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)

def _positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False

def _prune(now):
    for key, (created, candidate) in list(_RECORDS.items()):
        if now - created >= TTL_SECONDS or candidate.cache_ref() is None:
            _RECORDS.pop(key, None)

def register_candidate(cache, cache_key: str, *, status: str, winner: str,
                       calibration_record: Mapping, source_check: Mapping,
                       process_source_drifted: bool, calibration_policy,
                       gpu_policy, descriptor: InputDescriptor, setup_seconds: float) -> bool:
    """Register trustworthy measured metadata only when savings pay setup."""
    if status not in ("calibrated", "cache_hit") or process_source_drifted:
        return False
    if source_check.get("mode") != "strict_full_content":
        return False
    if (calibration_policy.relative_tolerance > 1e-10 or calibration_policy.absolute_tolerance > 1e-12
            or calibration_policy.repetitions < 2 or calibration_policy.warmups < 1):
        return False
    if winner not in ("cpu", "cuda_strict") or not isinstance(calibration_record, Mapping):
        return False
    if (calibration_record.get("winner") != winner or calibration_record.get("parity") != "pass"
            or calibration_record.get("selection_basis") != "steady_state_median"
            or calibration_record.get("repetitions", 0) < 2 or calibration_record.get("warmups", 0) < 1):
        return False
    cpu, cuda = calibration_record.get("cpu_median_seconds"), calibration_record.get("cuda_median_seconds")
    if not all(_positive(v) for v in (cpu, cuda, setup_seconds)):
        return False
    selected, other = (cpu, cuda) if winner == "cpu" else (cuda, cpu)
    if float(other) - float(selected) <= float(setup_seconds):
        return False
    try:
        cache_ref = weakref.ref(cache)
    except TypeError:
        return False
    policy, gpu = _freeze(calibration_policy), _freeze(gpu_policy)
    candidate = MeasuredCandidate(cache_ref, str(cache_key), policy, gpu, descriptor,
        winner, float(cpu), float(cuda), float(setup_seconds))
    # One bounded candidate per input and GPU policy.  The candidate retains
    # its calibration policy so the exact cache key can be recomputed later.
    key, now = (descriptor, gpu), time.monotonic()
    with _LOCK:
        _prune(now)
        _RECORDS[key] = (now, candidate)
        _RECORDS.move_to_end(key)
        while len(_RECORDS) > MAX_ENTRIES:
            _RECORDS.popitem(last=False)
    return True

def _drop_if_current(key, candidate):
    with _LOCK:
        current = _RECORDS.get(key)
        if current is not None and current[1] is candidate:
            _RECORDS.pop(key, None)

def lookup_candidate_details(*, descriptor: InputDescriptor, calibration_policy,
                             gpu_policy, static_backend: str,
                             validator: Callable[[MeasuredCandidate], object] | None):
    """Return route and measured validation cost after exact identity checks."""
    key = (descriptor, _freeze(gpu_policy))
    found = None
    with _LOCK:
        _prune(time.monotonic())
        if key in _RECORDS:
            _, found = _RECORDS[key]
            _RECORDS.move_to_end(key)
    if found is None:
        return CandidateLookup(None, 0.0, "no_candidate")
    if (calibration_policy is not None
            and found.calibration_policy != _freeze(calibration_policy)):
        return CandidateLookup(None, 0.0, "calibration_policy_mismatch")
    if found.winner == static_backend:
        return CandidateLookup(None, 0.0, "candidate_matches_static_route")
    if validator is None:
        return CandidateLookup(None, 0.0, "validator_unavailable")
    cache = found.cache_ref()
    try:
        record = None if cache is None else cache.get(found.cache_key)
    except Exception:
        record = None
    if (not isinstance(record, Mapping) or record.get("parity") != "pass"
            or record.get("winner") != found.winner):
        _drop_if_current(key, found)
        return CandidateLookup(None, 0.0, "calibration_cache_missing_or_stale")
    try:
        validation = validator(found)
    except Exception:
        return CandidateLookup(None, 0.0, "identity_validation_failed")
    if isinstance(validation, Mapping):
        validation_seconds = validation.get("validation_seconds")
        reason = validation.get("rejection_reason")
        valid = validation.get("valid", False)
    else:
        validation_seconds, reason, valid = 0.0, None, bool(validation)
    if isinstance(validation, Mapping):
        try:
            if (isinstance(validation_seconds, bool)
                    or not isinstance(validation_seconds, numbers.Real)):
                raise ValueError("validation duration must be numeric")
            validation_seconds = float(validation_seconds)
            if not math.isfinite(validation_seconds) or validation_seconds < 0:
                raise ValueError("validation duration must be finite and nonnegative")
        except (TypeError, ValueError):
            _drop_if_current(key, found)
            return CandidateLookup(None, None, "invalid_validation_cost")
    else:
        validation_seconds = 0.0
    if not valid:
        if reason in {"identity_mismatch", "source_drift", "source_mode_mismatch"}:
            _drop_if_current(key, found)
        return CandidateLookup(None, validation_seconds, reason or "identity_mismatch")
    selected, other = ((found.cpu_seconds, found.cuda_seconds)
                       if found.winner == "cpu"
                       else (found.cuda_seconds, found.cpu_seconds))
    median_advantage = float(other) - float(selected)
    if validation_seconds >= median_advantage:
        _drop_if_current(key, found)
        return CandidateLookup(None, validation_seconds,
                               "validation_cost_exceeds_median_advantage")
    try:
        record = None if cache is None else cache.get(found.cache_key)
    except Exception:
        record = None
    if not isinstance(record, Mapping) or record.get("parity") != "pass" or record.get("winner") != found.winner:
        _drop_if_current(key, found)
        return CandidateLookup(None, validation_seconds,
                               "calibration_cache_missing_or_stale")
    return CandidateLookup(found.winner, validation_seconds, None)

def lookup_candidate(*, descriptor: InputDescriptor, calibration_policy, gpu_policy,
                     static_backend: str, validator: Callable[[MeasuredCandidate], bool] | None):
    """Backward-compatible route-only lookup wrapper."""
    return lookup_candidate_details(
        descriptor=descriptor, calibration_policy=calibration_policy,
        gpu_policy=gpu_policy, static_backend=static_backend,
        validator=validator,
    ).winner

def clear_registry_for_tests():
    with _LOCK:
        _RECORDS.clear()

def registry_size():
    with _LOCK:
        _prune(time.monotonic())
        return len(_RECORDS)
